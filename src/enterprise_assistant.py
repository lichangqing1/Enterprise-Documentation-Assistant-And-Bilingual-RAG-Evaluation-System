from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from src.document_manager import (
    DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    EnterpriseDocumentIndex,
    enterprise_collection_fingerprint,
    list_enterprise_documents,
)
from src.rag_pipeline import SimpleRAGPipeline


class EnterpriseAssistantUnavailableError(RuntimeError):
    """Raised when enterprise Q&A cannot safely use the current index."""


def enterprise_reindex_reason(
    index: Optional[EnterpriseDocumentIndex],
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    chunk_size: int = 500,
    overlap: int = 100,
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
) -> Optional[str]:
    """Explain why the collection must be indexed again, or return None."""
    records = list_enterprise_documents(document_dir)
    if not records:
        return "The enterprise collection is empty. Add a document, then reindex the collection."

    invalid = [record for record in records if record["status"] != "valid"]
    if invalid:
        filenames = ", ".join(str(record["filename"]) for record in invalid)
        return f"The enterprise collection contains invalid documents ({filenames}). Fix them before reindexing."

    if index is None:
        return "The enterprise collection has not been indexed in this session. Reindex it before asking questions."

    current_fingerprint = enterprise_collection_fingerprint(document_dir)
    if index.collection_fingerprint != current_fingerprint:
        return "The enterprise documents changed after the last index was built. Reindex before asking questions."

    if index.chunk_size != chunk_size or index.overlap != overlap:
        return "The chunk settings changed after the last index was built. Reindex before asking questions."

    if index.retrieval_mode != retrieval_mode or index.semantic_backend != semantic_backend:
        return "The retrieval settings changed after the last index was built. Reindex before asking questions."

    return None


def ask_enterprise_question(
    question: str,
    index: Optional[EnterpriseDocumentIndex],
    document_dir: Union[str, Path] = DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 100,
    retrieval_mode: str = "hybrid",
    semantic_backend: str = "local",
) -> Dict[str, object]:
    """Answer using only a validated, current enterprise-document index."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    reindex_reason = enterprise_reindex_reason(
        index=index,
        document_dir=document_dir,
        chunk_size=chunk_size,
        overlap=overlap,
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
    )
    if reindex_reason is not None:
        raise EnterpriseAssistantUnavailableError(reindex_reason)

    rag = SimpleRAGPipeline(index.vector_store, top_k=top_k)
    result = rag.ask(question)
    answer = str(result.get("answer", ""))
    result["supporting_sources"] = sorted({
        str(source)
        for source in result.get("sources", [])
        if str(source) in answer
    })
    result["answer_mode"] = "extractive"
    result["status"] = "answered" if result["supporting_sources"] else "insufficient_evidence"
    return result
