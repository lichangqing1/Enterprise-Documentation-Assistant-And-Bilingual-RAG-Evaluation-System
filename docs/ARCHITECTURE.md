# Architecture

## System Purpose

The repository contains two related systems:

1. an Enterprise Documentation Assistant for managed `.md` and `.txt` documents;
2. a preserved bilingual customer-support RAG benchmark used for regression evaluation.

The design is intentionally framework-light. Retrieval, evidence checks, LLM integration, evaluation, and reporting are explicit Python modules rather than an agent or orchestration framework.

## Entry Points

| Entry point | Responsibility |
|---|---|
| `app.py` | Streamlit document management, legacy and enterprise Q&A, retrieval settings, and Evaluation Dashboard |
| `api.py` | FastAPI compatibility service for the historical customer-support corpus |
| `scripts/run_evaluation.py` | Main 73-case bilingual evaluation |
| `scripts/run_challenge_evaluation.py` | 26-case robustness evaluation |
| `scripts/run_security_evaluation.py` | 12-case legacy security evaluation |
| `scripts/run_enterprise_safety_evaluation.py` | 15-case enterprise safety evaluation with mocked providers |
| `scripts/run_regression_checks.py` | Nine configured quality gates |
| `scripts/run_ai_testing_workflow.py` | Retained legacy compatibility workflow for generated test assets |

## Corpus Isolation

```text
data/documents/                         data/enterprise_documents/
legacy customer-support corpus         managed enterprise corpus
           |                                      |
           v                                      v
load_rag_pipeline()                    reindex_enterprise_documents()
           |                                      |
           v                                      v
legacy vector store                    EnterpriseDocumentIndex
           |                                      |
           +------------- no shared index --------+
```

`data/documents/` is the historical benchmark corpus. Enterprise uploads are written only to `data/enterprise_documents/`. `src/enterprise_assistant.py` receives an `EnterpriseDocumentIndex` directly, which prevents fallback to the legacy vector store.

## Enterprise Document Lifecycle

`src/document_manager.py` owns the managed collection:

```text
upload
  -> filename and extension validation
  -> UTF-8 and content validation
  -> exclusive file creation (no overwrite)
  -> collection listing
  -> SHA-256 collection fingerprint
  -> loader and text splitter
  -> retriever factory
  -> in-memory EnterpriseDocumentIndex
```

The index records the corpus fingerprint, chunk size, overlap, retrieval mode, and semantic backend. `enterprise_reindex_reason()` compares those values with current state before every enterprise question. Document or setting changes produce an explicit reindex requirement instead of silently using stale data.

The index is stored in Streamlit session state. It is not persisted to a database or shared between users.

## Retrieval Layer

`src/retrieval.py` is the retriever factory.

| Mode | Implementation | Dependency profile |
|---|---|---|
| `lexical` / compatibility alias `keyword` | Lightweight term-based retrieval | Default install |
| `semantic` + `local` | Local semantic implementation | Default install |
| `semantic` + `faiss` | Sentence-Transformers embeddings and FAISS index | `requirements-vector.txt` |
| `hybrid` | Lexical and semantic score fusion | Depends on selected semantic backend |

All modes return chunk dictionaries with source text and similarity scores. The Q&A layer does not depend on a framework-specific document type.

## Extractive Answer Path

`src/rag_pipeline.py` implements `SimpleRAGPipeline`:

```text
question
  -> top-k retrieval
  -> prompt-injection and unsupported-constraint checks
  -> question/evidence concept coverage
  -> Markdown-aware sentence splitting
  -> subject, coverage, overlap, and retrieval-score ranking
  -> duplicate removal
  -> narrow-question or broad-policy aggregation
  -> source-cited answer OR citation-free safe fallback
```

The selection logic keeps retrieved chunks available independently of the final answer. It supports multi-sentence answers when separate sentences contribute relevant concepts, while subject and polarity checks reduce unrelated sentence aggregation.

## Optional LLM-Grounded Path

`src/grounded_llm.py` adds generation without replacing extractive Q&A:

```text
current enterprise index
  -> same retrieval path
  -> deterministic extractive evidence gate
  -> quarantine instruction-like document spans
  -> construct trusted-instruction/untrusted-evidence prompt
  -> explicit OpenAI-compatible provider call
  -> strict JSON parsing
  -> citation allow-list validation
  -> quarantined-instruction influence check
  -> numeric and polarity validation
  -> answer + source citations + claim/evidence references
```

Unsupported questions stop before the provider call. Raw retrieved chunks remain in the result for local debugging, while quarantined instructions are not sent as usable evidence.

The provider boundary is configured through `LLM_API_KEY` or `OPENAI_API_KEY`, `LLM_MODEL`, `LLM_API_BASE`, and `LLM_TIMEOUT`. Provider errors are translated into controlled application errors without logging secrets.

## Safety Boundaries

The system uses layered, conservative checks:

- question-side injection and unsafe-request screening;
- document-side instruction quarantine;
- evidence-sufficiency checks before generation;
- retrieved-source citation allow-listing;
- claim-to-chunk evidence association;
- explicit number validation;
- English and Chinese permission/prohibition polarity checks;
- corpus isolation and stale-index rejection;
- bounded provider configuration and error handling.

These checks are deterministic. They reduce known failure modes but do not constitute general semantic verification or complete red-team coverage.

## Evaluation and Reporting

```text
datasets under data/evaluation/
              |
              v
       evaluation runners
              |
              v
case-level CSV + summary CSV + Markdown reports
              |                         |
              v                         v
regression quality gates       src/quality_reporting.py
                                        |
                                        v
                              Streamlit Evaluation Dashboard
```

The main evaluator is `src/evaluation/evaluator.py`. It produces deterministic proxy metrics and compatibility aliases. Quality gates are loaded from `configs/rag_eval_config.yaml`.

The dashboard reads saved artifacts; it does not rerun benchmarks or alter thresholds. Directly measured categories are separated from the derived false-refusal diagnostic and the currently unmeasured over-extraction category.

## FastAPI Compatibility Boundary

`api.py` exposes the legacy customer-support pipeline:

```text
GET  /health
POST /ask
POST /evaluate
POST /feedback
POST /testing/generate
GET  /metrics
GET  /logs/summary
```

It does not currently expose enterprise uploads, enterprise indexing, or LLM-grounded enterprise Q&A. The `/testing/generate` endpoint and `ai_testing` imports are retained for backward compatibility.

## Legacy AI-Assisted Testing Compatibility

The `ai_testing/` package remains operational and covered by tests. It parses requirements, creates schema-validated scenarios and cases, executes generated assets, classifies failures, and optionally uses an OpenAI-compatible client for test generation or RCA.

This package is not part of the main Documentation Assistant request path. It is retained to preserve historical APIs, CI behavior, and generated test assets. See [AI_ASSISTED_TESTING.md](AI_ASSISTED_TESTING.md).

## Deployment Scope

The Dockerfile starts Streamlit on port 8501 with the default dependencies. The current architecture is appropriate for local demonstration and regression evaluation. Production multi-user deployment would require durable storage, authentication and authorization, tenant isolation, document lifecycle controls, and a persistent index.
