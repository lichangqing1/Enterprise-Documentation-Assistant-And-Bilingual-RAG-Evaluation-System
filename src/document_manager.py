from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Union

from src.document_loader import SUPPORTED_EXTENSIONS, load_documents, validate_documents
from src.retrieval import build_vector_store
from src.text_splitter import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTERPRISE_DOCUMENT_DIR = PROJECT_ROOT / "data" / "enterprise_documents"


class DocumentValidationError(ValueError):
    """Raised when an enterprise document cannot be safely indexed."""


@dataclass
class EnterpriseDocumentIndex:
    """In-memory index and source data produced by an enterprise reindex."""

    documents: List[Dict[str, str]]
    chunks: List[Dict[str, str]]
    vector_store: Any
    retrieval_mode: str
    semantic_backend: str
    chunk_size: int
    overlap: int
    collection_fingerprint: str

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


def ensure_enterprise_collection(document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR) -> Path:
    """Create the enterprise collection directory if it does not exist."""
    collection_path = Path(document_dir)
    collection_path.mkdir(parents=True, exist_ok=True)
    return collection_path


def _validate_filename(filename: str) -> str:
    """Validate a user-supplied filename without rewriting it."""
    candidate = str(filename).strip()
    if not candidate:
        raise DocumentValidationError("Document filename cannot be empty.")

    path = Path(candidate)
    if path.is_absolute() or path.name != candidate or candidate in {".", ".."}:
        raise DocumentValidationError("Document filename must not contain a directory path.")
    if candidate.startswith("."):
        raise DocumentValidationError("Hidden document filenames are not supported.")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise DocumentValidationError(
            f"Unsupported document type '{extension or 'none'}'. Supported types: {supported}."
        )
    return candidate


def _decode_content(content: Union[str, bytes]) -> str:
    """Decode uploaded content as UTF-8 and reject binary-looking text."""
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentValidationError("Document must contain valid UTF-8 text.") from exc
    elif isinstance(content, str):
        text = content
    else:
        raise DocumentValidationError("Document content must be text or UTF-8 bytes.")

    if not text.strip():
        raise DocumentValidationError("Document cannot be empty or contain only whitespace.")
    if "\x00" in text:
        raise DocumentValidationError("Document contains invalid null characters.")
    return text


def validate_enterprise_document(filename: str, content: Union[str, bytes]) -> str:
    """Validate one enterprise document and return normalized text content."""
    safe_filename = _validate_filename(filename)
    text = _decode_content(content)

    # Reuse the core document validator so managed documents follow the same
    # minimum source/text contract as the benchmark corpus.
    validate_documents([{"source": safe_filename, "text": text}])
    return text


def add_enterprise_document(
    filename: str,
    content: Union[str, bytes],
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
) -> Path:
    """Validate and add a new document without overwriting an existing file."""
    safe_filename = _validate_filename(filename)
    text = validate_enterprise_document(safe_filename, content)
    collection_path = ensure_enterprise_collection(document_dir)
    target_path = collection_path / safe_filename

    try:
        with target_path.open("x", encoding="utf-8") as file:
            file.write(text)
    except FileExistsError as exc:
        raise DocumentValidationError(
            f"A document named '{safe_filename}' already exists. Existing documents are not overwritten."
        ) from exc
    return target_path


def _document_record(file_path: Path) -> Dict[str, object]:
    """Build one listing record, including validation status."""
    record: Dict[str, object] = {
        "filename": file_path.name,
        "type": file_path.suffix.lower() or "none",
        "size_bytes": file_path.stat().st_size,
        "status": "valid",
        "validation_error": "",
    }
    try:
        content = file_path.read_bytes()
        validate_enterprise_document(file_path.name, content)
    except (DocumentValidationError, OSError) as exc:
        record["status"] = "invalid"
        record["validation_error"] = str(exc)
    return record


def list_enterprise_documents(
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
) -> List[Dict[str, object]]:
    """List managed files and report whether each file is indexable."""
    collection_path = ensure_enterprise_collection(document_dir)
    return [
        _document_record(file_path)
        for file_path in sorted(collection_path.iterdir())
        if file_path.is_file() and not file_path.name.startswith(".")
    ]


def enterprise_collection_fingerprint(
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
) -> str:
    """Return a content fingerprint for all visible files in the collection."""
    collection_path = ensure_enterprise_collection(document_dir)
    digest = hashlib.sha256()
    for file_path in sorted(collection_path.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            raise DocumentValidationError(
                f"Could not read enterprise document '{file_path.name}': {exc}"
            ) from exc
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def reindex_enterprise_documents(
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    chunk_size: int = 500,
    overlap: int = 100,
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
) -> EnterpriseDocumentIndex:
    """Validate the enterprise collection and build a separate in-memory index."""
    collection_path = ensure_enterprise_collection(document_dir)
    records = list_enterprise_documents(collection_path)
    invalid_records = [record for record in records if record["status"] != "valid"]
    if invalid_records:
        details = "; ".join(
            f"{record['filename']}: {record['validation_error']}"
            for record in invalid_records
        )
        raise DocumentValidationError(f"Cannot reindex invalid documents. {details}")
    if not records:
        raise DocumentValidationError(
            "The enterprise collection is empty. Add a .md or .txt document before reindexing."
        )

    fingerprint_before = enterprise_collection_fingerprint(collection_path)
    documents = load_documents(str(collection_path), extensions=SUPPORTED_EXTENSIONS)
    validate_documents(documents)
    chunks = create_chunks(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise DocumentValidationError("The enterprise documents did not produce any indexable text chunks.")

    vector_store = build_vector_store(
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
    )
    vector_store.build_index(chunks)
    fingerprint_after = enterprise_collection_fingerprint(collection_path)
    if fingerprint_before != fingerprint_after:
        raise DocumentValidationError(
            "The enterprise collection changed during reindexing. Run reindex again."
        )
    return EnterpriseDocumentIndex(
        documents=documents,
        chunks=chunks,
        vector_store=vector_store,
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
        chunk_size=chunk_size,
        overlap=overlap,
        collection_fingerprint=fingerprint_after,
    )
