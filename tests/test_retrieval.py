from src.document_loader import load_markdown_documents
from src.retrieval import build_vector_store
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.lexical_retriever import LexicalRetriever
from src.retrievers.semantic_retriever import LocalSemanticRetriever
from src.text_splitter import create_chunks


def test_retrieval_returns_expected_structure():
    docs = load_markdown_documents("data/documents")
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = HybridRetriever()
    vector_store.build_index(chunks)

    results = vector_store.search("How long does standard shipping take?", top_k=3)

    assert len(results) > 0
    assert "source" in results[0]
    assert "text" in results[0]
    assert "similarity_score" in results[0]


def test_hybrid_retrieval_exposes_bm25_and_embedding_scores():
    docs = load_markdown_documents("data/documents")
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    vector_store = HybridRetriever()
    vector_store.build_index(chunks)

    results = vector_store.search("公司支持支付宝付款吗?", top_k=3)

    assert len(results) == 3
    assert results[0]["retrieval_method"] == "hybrid_lexical_local_score_fusion"
    assert "keyword_score" in results[0]
    assert "semantic_score" in results[0]
    assert "similarity_score" in results[0]


def test_keyword_semantic_and_hybrid_modes_are_explicit():
    docs = load_markdown_documents("data/documents")
    chunks = create_chunks(docs, chunk_size=500, overlap=100)

    cases = [
        (LexicalRetriever(), "lexical", "lexical_bm25_tfidf"),
        (LocalSemanticRetriever(), "semantic", "semantic_local_tfidf_svd"),
        (HybridRetriever(), "hybrid", "hybrid_lexical_local_score_fusion"),
    ]

    for vector_store, expected_mode, expected_method in cases:
        vector_store.build_index(chunks)
        results = vector_store.search("How long does standard shipping take?", top_k=2)
        assert results[0]["retrieval_mode"] == expected_mode
        assert results[0]["retrieval_method"] == expected_method
        assert "similarity_score" in results[0]


def test_retrieval_factory_builds_requested_modes():
    assert isinstance(build_vector_store("keyword"), LexicalRetriever)
    assert isinstance(build_vector_store("lexical"), LexicalRetriever)
    assert isinstance(build_vector_store("semantic"), LocalSemanticRetriever)
    assert isinstance(build_vector_store("hybrid"), HybridRetriever)


def test_single_matching_chunk_keeps_positive_similarity_score():
    chunks = [
        {
            "chunk_id": "policy.md_chunk_0",
            "source": "policy.md",
            "text": "Access reviews are completed every 90 days.",
        }
    ]

    for vector_store in [LexicalRetriever(), LocalSemanticRetriever(), HybridRetriever()]:
        vector_store.build_index(chunks)
        result = vector_store.search("How often are access reviews completed?", top_k=1)[0]
        assert result["similarity_score"] == 1.0
