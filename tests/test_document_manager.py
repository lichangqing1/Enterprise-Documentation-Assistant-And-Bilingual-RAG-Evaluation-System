from pathlib import Path

import pytest

from src.document_manager import (
    DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    DocumentValidationError,
    add_enterprise_document,
    list_enterprise_documents,
    reindex_enterprise_documents,
)


def test_enterprise_collection_is_separate_from_benchmark_documents():
    assert DEFAULT_ENTERPRISE_DOCUMENT_DIR != Path("data/documents").resolve()
    assert DEFAULT_ENTERPRISE_DOCUMENT_DIR.name == "enterprise_documents"


def test_add_and_list_markdown_and_text_documents(tmp_path):
    markdown_path = add_enterprise_document(
        "security_handbook.md",
        "# Security handbook\nUse approved identity controls.",
        tmp_path,
    )
    text_path = add_enterprise_document(
        "leave_policy.txt",
        b"Employees should submit leave requests through the HR portal.",
        tmp_path,
    )

    assert markdown_path.read_text(encoding="utf-8").startswith("# Security handbook")
    assert text_path.exists()
    records = list_enterprise_documents(tmp_path)
    assert [record["filename"] for record in records] == [
        "leave_policy.txt",
        "security_handbook.md",
    ]
    assert [record["type"] for record in records] == [".txt", ".md"]
    assert all(record["size_bytes"] > 0 for record in records)
    assert all(record["status"] == "valid" for record in records)
    assert all(record["validation_error"] == "" for record in records)


@pytest.mark.parametrize(
    ("filename", "content", "expected_message"),
    [
        ("manual.pdf", b"PDF", "Unsupported document type"),
        ("empty.md", "   \n", "cannot be empty"),
        ("binary.txt", b"\xff\xfe", "valid UTF-8"),
        ("invalid.txt", "valid\x00invalid", "null characters"),
        ("../outside.md", "content", "must not contain a directory path"),
    ],
)
def test_add_rejects_invalid_documents(tmp_path, filename, content, expected_message):
    with pytest.raises(DocumentValidationError, match=expected_message):
        add_enterprise_document(filename, content, tmp_path)


def test_add_does_not_overwrite_existing_document(tmp_path):
    add_enterprise_document("policy.md", "Original policy", tmp_path)

    with pytest.raises(DocumentValidationError, match="already exists"):
        add_enterprise_document("policy.md", "Replacement policy", tmp_path)

    assert (tmp_path / "policy.md").read_text(encoding="utf-8") == "Original policy"


def test_listing_reports_manually_added_invalid_files(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "unsupported.csv").write_text("value", encoding="utf-8")

    records = list_enterprise_documents(tmp_path)

    assert [record["status"] for record in records] == ["invalid", "invalid"]
    assert "cannot be empty" in records[0]["validation_error"]
    assert "Unsupported document type" in records[1]["validation_error"]


def test_reindex_rejects_empty_or_invalid_collection(tmp_path):
    with pytest.raises(DocumentValidationError, match="collection is empty"):
        reindex_enterprise_documents(tmp_path, retrieval_mode="lexical")

    (tmp_path / "invalid.md").write_text("", encoding="utf-8")
    with pytest.raises(DocumentValidationError, match="Cannot reindex invalid documents"):
        reindex_enterprise_documents(tmp_path, retrieval_mode="lexical")


def test_reindex_builds_searchable_index_from_enterprise_documents(tmp_path):
    add_enterprise_document(
        "security_handbook.md",
        "Enterprise access reviews happen every quarter.",
        tmp_path,
    )
    add_enterprise_document(
        "leave_policy.txt",
        "Annual leave requests are submitted through the HR portal.",
        tmp_path,
    )

    index = reindex_enterprise_documents(
        tmp_path,
        chunk_size=80,
        overlap=10,
        retrieval_mode="lexical",
    )
    results = index.vector_store.search("quarterly access reviews", top_k=1)

    assert index.document_count == 2
    assert index.chunk_count == 2
    assert results[0]["source"] == "security_handbook.md"
