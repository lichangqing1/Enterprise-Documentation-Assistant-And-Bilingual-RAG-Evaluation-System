from pathlib import Path
import sys

import streamlit as st

PROJECT_PATH = Path(__file__).resolve().parent
if str(PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_PATH))

from src.document_loader import load_markdown_documents, validate_documents
from src.document_manager import (
    DEFAULT_ENTERPRISE_DOCUMENT_DIR,
    DocumentValidationError,
    add_enterprise_document,
    list_enterprise_documents,
    reindex_enterprise_documents,
)
from src.text_splitter import create_chunks
from src.retrieval import build_vector_store
from src.rag_pipeline import SimpleRAGPipeline


st.set_page_config(page_title="Customer Support RAG Assistant", layout="wide")
st.title("AI Customer Support RAG Assistant")
st.caption("RAG chatbot with source-grounded answers and evaluation-focused design.")

DOCUMENT_DIR = PROJECT_PATH / "data" / "documents"


@st.cache_resource
def load_rag_pipeline(chunk_size: int, overlap: int, top_k: int, retrieval_mode: str, semantic_backend: str):
    docs = load_markdown_documents(str(DOCUMENT_DIR))
    validate_documents(docs)

    chunks = create_chunks(docs, chunk_size=chunk_size, overlap=overlap)

    vector_store = build_vector_store(retrieval_mode=retrieval_mode, semantic_backend=semantic_backend)
    vector_store.build_index(chunks)

    rag = SimpleRAGPipeline(vector_store, top_k=top_k)
    return rag, docs, chunks


with st.sidebar:
    st.header("RAG Settings")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1000, value=500, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=300, value=100, step=50)
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=5, value=3, step=1)
    retrieval_mode = st.selectbox("Retrieval mode", ["lexical", "semantic", "hybrid"], index=2)
    semantic_backend = st.selectbox("Semantic backend", ["local", "faiss"], index=0)
    if semantic_backend == "faiss":
        st.warning("FAISS backend requires optional dependencies: pip install -r requirements-vector.txt")

    st.header("Example Questions")
    example_question = st.selectbox(
        "Choose an example",
        [
            "How long does standard shipping usually take?",
            "Can customized products be returned?",
            "What payment methods does the company accept?",
            "Can customer support see my password?",
            "Can I pay with cryptocurrency?",
        ],
    )

rag, docs, chunks = load_rag_pipeline(chunk_size, overlap, top_k, retrieval_mode, semantic_backend)

st.subheader("Knowledge Base Summary")
col1, col2 = st.columns(2)
col1.metric("Documents", len(docs))
col2.metric("Chunks", len(chunks))

question = st.text_input("Ask a customer support question:", value=example_question)

if st.button("Ask") and question.strip():
    response = rag.ask(question)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Retrieved Sources")
    for i, chunk in enumerate(response["retrieved_chunks"], start=1):
        with st.expander(f"Retrieved Chunk {i}: {chunk['source']} | Score: {chunk.get('similarity_score', 0):.4f}"):
            st.write(chunk["text"])

st.divider()
st.subheader("Enterprise document management")
st.caption(
    "Manage a separate enterprise document collection. These files do not change "
    "the legacy benchmark corpus or the current Q&A assistant."
)

st.session_state.setdefault("enterprise_document_index", None)

with st.container(border=True):
    with st.form("enterprise_document_upload", clear_on_submit=True):
        uploaded_document = st.file_uploader(
            "Add a Markdown or text document",
            type=["md", "txt"],
            accept_multiple_files=False,
        )
        add_submitted = st.form_submit_button("Add document")

    if add_submitted:
        if uploaded_document is None:
            st.error("Choose a .md or .txt document before submitting.")
        else:
            try:
                added_path = add_enterprise_document(
                    uploaded_document.name,
                    uploaded_document.getvalue(),
                    DEFAULT_ENTERPRISE_DOCUMENT_DIR,
                )
                st.session_state["enterprise_document_index"] = None
                st.success(f"Added {added_path.name}. Reindex the collection to include it.")
            except DocumentValidationError as exc:
                st.error(str(exc))

    enterprise_documents = list_enterprise_documents(DEFAULT_ENTERPRISE_DOCUMENT_DIR)
    if enterprise_documents:
        st.dataframe(enterprise_documents, hide_index=True)
    else:
        st.info("No enterprise documents have been added yet.")

    if st.button("Validate and reindex collection"):
        try:
            enterprise_index = reindex_enterprise_documents(
                DEFAULT_ENTERPRISE_DOCUMENT_DIR,
                chunk_size=chunk_size,
                overlap=overlap,
                retrieval_mode=retrieval_mode,
                semantic_backend=semantic_backend,
            )
            st.session_state["enterprise_document_index"] = enterprise_index
            st.success(
                f"Indexed {enterprise_index.document_count} documents into "
                f"{enterprise_index.chunk_count} chunks."
            )
        except (DocumentValidationError, ImportError, ValueError) as exc:
            st.error(str(exc))

    current_enterprise_index = st.session_state["enterprise_document_index"]
    if current_enterprise_index is not None:
        st.caption(
            f"Current session index: {current_enterprise_index.document_count} documents, "
            f"{current_enterprise_index.chunk_count} chunks, "
            f"{current_enterprise_index.retrieval_mode}/{current_enterprise_index.semantic_backend}."
        )

st.divider()
st.subheader("Evaluation Design")
st.write(
    "This project includes automated checks for context precision, context recall, "
    "faithfulness, answer relevancy, citation accuracy, safe handling of unanswerable "
    "questions, failed-case extraction, and experiment comparison."
)
