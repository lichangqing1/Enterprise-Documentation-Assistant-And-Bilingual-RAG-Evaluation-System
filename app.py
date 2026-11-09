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
from src.enterprise_assistant import (
    EnterpriseAssistantUnavailableError,
    ask_enterprise_question,
    enterprise_reindex_reason,
)
from src.grounded_llm import (
    GroundedLLMProviderError,
    GroundedLLMResponseError,
    LLMConfigurationError,
    ask_grounded_llm_question,
    get_llm_provider_status,
)
from src.quality_reporting import build_quality_dashboard_report
from src.text_splitter import create_chunks
from src.retrieval import build_vector_store
from src.rag_pipeline import SimpleRAGPipeline


st.set_page_config(page_title="Enterprise Documentation Assistant", layout="wide")
st.title("Enterprise Documentation Assistant")
st.caption("Grounded document Q&A with a preserved customer-support evaluation baseline.")

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


st.session_state.setdefault("enterprise_document_index", None)

with st.sidebar:
    st.header("Assistant mode")
    assistant_mode = st.segmented_control(
        "Q&A collection",
        ["Legacy customer support", "Enterprise documentation"],
        default="Legacy customer support",
        required=True,
    )

    st.header("RAG settings")
    chunk_size = st.slider("Chunk size", min_value=200, max_value=1000, value=500, step=100)
    overlap = st.slider("Overlap", min_value=0, max_value=300, value=100, step=50)
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=5, value=3, step=1)
    retrieval_mode = st.selectbox("Retrieval mode", ["lexical", "semantic", "hybrid"], index=2)
    semantic_backend = st.selectbox("Semantic backend", ["local", "faiss"], index=0)
    if semantic_backend == "faiss":
        st.warning("FAISS backend requires optional dependencies: pip install -r requirements-vector.txt")

if assistant_mode == "Legacy customer support":
    rag, docs, chunks = load_rag_pipeline(chunk_size, overlap, top_k, retrieval_mode, semantic_backend)

    st.subheader("Legacy customer support Q&A")
    st.caption("This mode uses only the historical data/documents benchmark corpus.")
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(docs))
    col2.metric("Chunks", len(chunks))

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
    question = st.text_input(
        "Ask a customer support question",
        value=example_question,
        key="legacy_question",
    )

    if st.button("Ask legacy collection") and question.strip():
        response = rag.ask(question)

        st.subheader("Answer")
        st.write(response["answer"])

        st.subheader("Retrieved sources")
        for i, chunk in enumerate(response["retrieved_chunks"], start=1):
            with st.expander(
                f"Retrieved chunk {i}: {chunk['source']} | "
                f"Score: {chunk.get('similarity_score', 0):.4f}"
            ):
                st.write(chunk["text"])
else:
    enterprise_documents = list_enterprise_documents(DEFAULT_ENTERPRISE_DOCUMENT_DIR)
    enterprise_index = st.session_state["enterprise_document_index"]
    reindex_reason = enterprise_reindex_reason(
        enterprise_index,
        DEFAULT_ENTERPRISE_DOCUMENT_DIR,
        chunk_size=chunk_size,
        overlap=overlap,
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
    )

    st.subheader("Enterprise documentation Q&A")
    st.caption("This mode searches only data/enterprise_documents. It never searches the legacy corpus.")
    col1, col2 = st.columns(2)
    col1.metric("Managed documents", len(enterprise_documents))
    col2.metric("Indexed chunks", enterprise_index.chunk_count if reindex_reason is None else 0)

    if reindex_reason is not None:
        st.warning(reindex_reason)
    else:
        st.success("The enterprise index matches the current documents and retrieval settings.")

    answering_mode = st.segmented_control(
        "Answering mode",
        ["Extractive", "LLM grounded"],
        default="Extractive",
        required=True,
        key="enterprise_answering_mode",
        help="Both modes use the same retrieved enterprise evidence. LLM mode is optional and explicit.",
    )
    provider_status = get_llm_provider_status()
    if answering_mode == "LLM grounded":
        if provider_status.configured:
            st.success(provider_status.message)
            st.caption(
                "Only retrieved enterprise evidence is sent after you click Ask. "
                "Document instructions are treated as untrusted text, and citations are validated."
            )
        else:
            st.info(provider_status.message)

    enterprise_question = st.text_input(
        "Ask about your enterprise documents",
        placeholder="For example: How often is preventive maintenance required?",
        key="enterprise_question",
    )
    ask_disabled = reindex_reason is not None or (
        answering_mode == "LLM grounded" and not provider_status.configured
    )
    if st.button("Ask enterprise collection", disabled=ask_disabled):
        try:
            answer_function = (
                ask_grounded_llm_question
                if answering_mode == "LLM grounded"
                else ask_enterprise_question
            )
            response = answer_function(
                enterprise_question,
                enterprise_index,
                DEFAULT_ENTERPRISE_DOCUMENT_DIR,
                top_k=top_k,
                chunk_size=chunk_size,
                overlap=overlap,
                retrieval_mode=retrieval_mode,
                semantic_backend=semantic_backend,
            )
            st.subheader("Answer")
            if response["status"] == "insufficient_evidence":
                st.info(response["answer"])
            else:
                st.write(response["answer"])
            st.caption(
                "Answering mode: "
                + ("LLM grounded" if response["answer_mode"] == "llm_grounded" else "Extractive")
            )

            supporting_sources = response["supporting_sources"]
            st.subheader("Supporting documents")
            st.write(
                ", ".join(supporting_sources)
                if supporting_sources
                else "No document was cited because the retrieved evidence was insufficient."
            )

            st.subheader("Retrieved evidence")
            for i, chunk in enumerate(response["retrieved_chunks"], start=1):
                with st.expander(
                    f"Evidence {i}: {chunk['source']} | "
                    f"Score: {chunk.get('similarity_score', 0):.4f}"
                ):
                    st.write(chunk["text"])
        except (EnterpriseAssistantUnavailableError, ValueError) as exc:
            st.warning(str(exc))
        except (LLMConfigurationError, GroundedLLMProviderError, GroundedLLMResponseError) as exc:
            st.error(str(exc))

st.divider()
st.subheader("Enterprise document management")
st.caption(
    "Manage the collection used only by Enterprise documentation Q&A. "
    "These files never change the legacy benchmark corpus."
)

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
                st.success(
                    f"Added {added_path.name}. The enterprise collection changed; "
                    "reindex it before asking questions."
                )
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
    current_reindex_reason = enterprise_reindex_reason(
        current_enterprise_index,
        DEFAULT_ENTERPRISE_DOCUMENT_DIR,
        chunk_size=chunk_size,
        overlap=overlap,
        retrieval_mode=retrieval_mode,
        semantic_backend=semantic_backend,
    )
    if current_reindex_reason is None:
        st.caption(
            f"Current session index: {current_enterprise_index.document_count} documents, "
            f"{current_enterprise_index.chunk_count} chunks, "
            f"{current_enterprise_index.retrieval_mode}/{current_enterprise_index.semantic_backend}."
        )
    else:
        st.warning(f"Reindex required: {current_reindex_reason}")

st.divider()
st.subheader("Evaluation Dashboard")
st.write(
    "This project includes automated checks for context precision, context recall, "
    "faithfulness, answer relevancy, citation accuracy, safe handling of unanswerable "
    "questions, failed-case extraction, and experiment comparison."
)

quality_report = build_quality_dashboard_report(PROJECT_PATH / "results")
if quality_report.artifact_warnings:
    st.warning(
        "Some saved evaluation artifacts require attention. "
        + " ".join(quality_report.artifact_warnings)
    )

st.markdown("### Evaluation suite overview")
st.caption(
    "Saved evaluation results from repository artifacts. These figures are not freshly "
    "executed tests and do not represent live-provider requests."
)
suite_rows = []
for suite in quality_report.suites:
    suite_rows.append({
        "Evaluation suite": suite.label,
        "Total cases": suite.total_cases if suite.total_cases is not None else "Not available",
        "Passed cases": suite.passed_cases if suite.passed_cases is not None else "Not available",
        "Failed cases": suite.failed_cases if suite.failed_cases is not None else "Not available",
        "Pass rate": (
            f"{suite.pass_rate:.1%}" if suite.pass_rate is not None else "Not available"
        ),
        "Artifact status": suite.artifact_status,
        "Saved result source": suite.source,
    })
st.dataframe(
    suite_rows,
    hide_index=True,
    width="stretch",
    column_config={
        "Evaluation suite": st.column_config.TextColumn("Evaluation suite", pinned=True),
        "Total cases": st.column_config.TextColumn("Total cases", width="small"),
        "Passed cases": st.column_config.TextColumn("Passed cases", width="small"),
        "Failed cases": st.column_config.TextColumn("Failed cases", width="small"),
        "Pass rate": st.column_config.TextColumn("Pass rate", width="small"),
        "Artifact status": st.column_config.TextColumn("Artifact status", width="small"),
    },
)

enterprise_suite = next(
    suite for suite in quality_report.suites if suite.key == "enterprise_safety"
)
enterprise_case_count = (
    str(enterprise_suite.total_cases)
    if enterprise_suite.total_cases is not None
    else "unavailable"
)
with st.expander(f"Enterprise safety cases — {enterprise_case_count} saved cases"):
    if enterprise_suite.cases.empty:
        st.info(
            "Enterprise safety case details are unavailable because the saved artifact "
            "is missing or malformed."
        )
    else:
        st.dataframe(
            enterprise_suite.cases,
            hide_index=True,
            width="stretch",
            column_config={
                "Test ID": st.column_config.TextColumn("Test ID", pinned=True),
                "Status": st.column_config.TextColumn("Status", width="small"),
            },
        )

st.markdown("### Quality failure categories")
st.caption(
    "This view reads existing result artifacts only. MAIN, CHALLENGE, and SECURITY IDs are "
    "display IDs derived from row order because those legacy CSVs do not persist test IDs. "
    "Enterprise ENT-SAFE IDs come directly from the safety results."
)

measured_categories = sum(
    category.measurement_level == "Measured" for category in quality_report.categories
)
derived_categories = sum(
    category.measurement_level == "Derived diagnostic" for category in quality_report.categories
)
unmeasured_categories = sum(
    category.measurement_level == "Not measured" for category in quality_report.categories
)
with st.container(horizontal=True):
    st.metric("Directly measured categories", measured_categories, border=True)
    st.metric("Derived diagnostics", derived_categories, border=True)
    st.metric("Not measured", unmeasured_categories, border=True)

for category in quality_report.categories:
    with st.expander(f"{category.label} — {category.measurement_level}"):
        st.write(category.definition)
        if category.measurement_level == "Measured":
            st.success(category.measurement_note)
        elif category.measurement_level == "Derived diagnostic":
            st.warning(category.measurement_note)
        else:
            st.info(category.measurement_note)

        if category.measured_cases is not None:
            metric_columns = st.columns(2)
            metric_columns[0].metric("Cases in saved results", category.measured_cases)
            metric_columns[1].metric("Failed cases", category.failed_cases)

        if category.cases.empty:
            st.caption(
                "Test IDs: not available · Expected behavior: not recorded · "
                "Actual behavior: not measured · Status: not measured · "
                "Failure reasons: not available"
            )
        else:
            st.dataframe(
                category.cases,
                hide_index=True,
                width="stretch",
                column_config={
                    "Test ID": st.column_config.TextColumn("Test ID", pinned=True),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                },
            )
