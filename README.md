# Enterprise Documentation Assistant & Bilingual RAG Evaluation System

## 1. Project Overview

This repository combines an enterprise documentation assistant with a reproducible bilingual RAG quality-engineering system. Users can manage a separate enterprise document collection, retrieve evidence with lexical, semantic, or hybrid search, and choose deterministic extractive answers or optional LLM-grounded answers.

The original English and Chinese customer-support corpus remains isolated as a regression benchmark. Evaluation scripts, adversarial safety cases, pytest coverage, saved reports, and configurable quality gates make behavior changes reviewable rather than anecdotal.

The project is application-level RAG engineering. Its benchmark results demonstrate success on the included datasets; they do not guarantee universal accuracy, security, or production readiness.

## 2. Key Features

- Separate enterprise and legacy benchmark document collections.
- `.md` and `.txt` upload, validation, listing, fingerprinting, and session-local reindexing.
- Lexical, semantic, and hybrid retrieval through one retriever factory.
- English and Chinese question answering.
- Deterministic extractive Q&A with evidence-sufficiency checks.
- Optional OpenAI-compatible LLM-grounded Q&A, enabled explicitly by environment configuration and UI selection.
- Retrieved evidence, validated source filenames, and safe insufficient-information responses.
- Document prompt-injection quarantine, citation allow-listing, numeric-fact validation, and permission/prohibition polarity checks.
- Main, challenge, legacy security, and enterprise safety evaluation suites.
- Streamlit evaluation dashboard with case-level expected and actual behavior.
- FastAPI compatibility service for the historical customer-support RAG workflow.

## 3. Architecture

```text
                         Streamlit application (app.py)
                                    |
              +---------------------+---------------------+
              |                                           |
              v                                           v
 Legacy customer-support mode                  Enterprise documentation mode
 data/documents/                              data/enterprise_documents/
 preserved benchmark corpus                   managed .md/.txt collection
              |                                           |
              +---------------------+---------------------+
                                    v
                       Loader -> Validator -> Splitter
                                    |
                                    v
                    Retriever factory (lexical/semantic/hybrid)
                                    |
                         +----------+----------+
                         |                     |
                         v                     v
                SimpleRAGPipeline       Grounded LLM service
                extractive answer       deterministic evidence gate
                                       + optional provider call
                         |                     |
                         +----------+----------+
                                    v
                      answer + citations + retrieved evidence

 Evaluation datasets -> runners -> CSV/Markdown artifacts -> quality gates
                                              |
                                              v
                                  Evaluation Dashboard
```

Important boundaries:

- Enterprise retrieval never searches `data/documents/`.
- The legacy benchmark never uses uploaded enterprise documents.
- The enterprise index is kept in Streamlit session state and becomes stale when documents or retrieval settings change.
- LLM mode uses the same retrieved enterprise evidence as extractive mode and is never selected automatically.
- The FastAPI service currently exposes the historical customer-support workflow, not enterprise document management.

See [Architecture](docs/ARCHITECTURE.md) for module-level details.

## 4. Technology Stack

| Area | Repository implementation |
|---|---|
| Language/runtime | Python 3.11 in CI and Docker |
| UI | Streamlit |
| API | FastAPI, Uvicorn, Pydantic |
| Data and evaluation | pandas, NumPy, scikit-learn |
| Default retrieval | Lightweight local lexical and semantic implementations |
| Optional vector retrieval | Sentence-Transformers and FAISS |
| Testing | pytest |
| Configuration | YAML through `configs/rag_eval_config.yaml` |
| Optional LLM | Minimal OpenAI-compatible JSON client |
| Reporting | CSV, Markdown, JSONL logs, Streamlit dashboard |

Default dependencies are listed in `requirements.txt`. FAISS and Sentence-Transformers are isolated in `requirements-vector.txt` so the default test and CI path remains lightweight.

## 5. Quick Start

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the Streamlit application:

```bash
.venv/bin/python -m streamlit run app.py
```

Optional FAISS retrieval:

```bash
.venv/bin/python -m pip install -r requirements-vector.txt
```

Start the compatibility FastAPI service:

```bash
.venv/bin/python -m uvicorn api:app --reload --port 8001
```

Swagger UI is then available at `http://127.0.0.1:8001/docs`.

## 6. Document Upload and Indexing

Select **Enterprise documentation** in the Streamlit sidebar, then use **Enterprise document management** to add files and build an index.

The document manager:

1. accepts `.md` and `.txt` files;
2. validates filenames, UTF-8 content, empty content, null characters, and duplicate names;
3. writes only to `data/enterprise_documents/`;
4. reuses the core document loader and text splitter;
5. builds a separate in-memory retrieval index;
6. fingerprints document contents and retrieval settings;
7. requires reindexing after a document, chunking setting, retrieval mode, or semantic backend changes.

Existing files are not overwritten. `data/documents/` remains the historical benchmark corpus.

![Enterprise document management interface](docs/screenshots/enterprise-document-management.png)

*Enterprise document management keeps uploaded `.md` and `.txt` files separate from the legacy benchmark corpus and makes index freshness visible.*

## 7. Extractive vs. LLM-Grounded Q&A

| Mode | Behavior | External request |
|---|---|---|
| Extractive | Selects supported sentences from retrieved enterprise chunks and appends source citations | No |
| LLM grounded | Uses the same retrieval results, then asks a configured provider for structured grounded output | Yes, only after explicit configuration and selection |

Extractive mode remains available without credentials. It uses deterministic evidence coverage, answer-type checks, subject-aware sentence ranking, duplicate removal, and safe fallback behavior.

![Extractive enterprise question answering](docs/screenshots/enterprise-extractive-qa.png)

*Extractive mode selects the Critical incident response target from structured Markdown-table evidence, returns a concise supported fact, and preserves the source citation and retrieved chunks.*

LLM-grounded mode requires an OpenAI-compatible provider:

```bash
export LLM_API_KEY="your_api_key"
export LLM_MODEL="gpt-4o-mini"
# Optional overrides:
export LLM_API_BASE="https://api.openai.com/v1"
export LLM_TIMEOUT="60"
```

`OPENAI_API_KEY` is also supported. Keys are read from environment variables and must not be committed. The application does not send document evidence to a provider unless LLM-grounded mode is configured, selected, and invoked.

Automated LLM tests use mocked clients and make no network calls. A separate manual live-provider smoke check was previously reported as three successful requests; it is not part of pytest or CI, and no credentials or provider-response artifacts are stored in this repository.

![LLM-grounded enterprise question answering](docs/screenshots/enterprise-llm-grounded_qa.png)

*A manually configured LLM-grounded session answers from retrieved enterprise evidence and displays validated source information. Live-provider behavior is not included in the automated benchmark totals.*

## 8. Evidence Grounding and Citation Validation

Both enterprise answer modes begin with the same isolated retrieval index. The system then applies layered checks:

- question-to-evidence concept coverage;
- answer-type support, including temporal questions;
- source association based on retrieved filenames;
- insufficient-evidence fallback without a citation;
- exact citation allow-listing for LLM output;
- claim-to-evidence chunk references;
- explicit numerical fact checks;
- conservative permission, prohibition, and negation checks.

A valid filename alone is not treated as proof that an answer is supported. Retrieved chunks remain visible separately for debugging even when the final answer is rejected.

These validators are deterministic safeguards, not full semantic entailment. They are strongest for extractive text, explicit numbers, citations, and clear positive/negative policy language.

## 9. AI Safety and Security Controls

The application and evaluation suites cover:

- prompt injection and instruction override attempts;
- malicious instructions embedded in retrieved documents;
- jailbreak and system-prompt disclosure requests;
- credential and sensitive-information requests;
- fabricated or cross-corpus citations;
- incorrect numerical claims;
- permission/prohibition reversals;
- unsupported questions and false acceptance;
- provider timeout, rate-limit, connection, and malformed-response handling;
- strict enterprise/legacy corpus isolation.

Instruction-like document spans are quarantined before provider prompting. Safe document text remains available as evidence, while ordinary policy language such as prohibitions is not automatically treated as malicious instruction text.

See [Security Evaluation](docs/SECURITY_EVALUATION.md) and [Quality Case Studies](docs/QUALITY_CASE_STUDIES.md).

## 10. Evaluation Framework

| Suite | Cases | Purpose | Runner |
|---|---:|---|---|
| Main bilingual benchmark | 73 | English, Chinese, unanswerable, and legacy security regression | `scripts/run_evaluation.py` |
| Challenge benchmark | 26 | Paraphrases, unsupported constraints, and robustness cases | `scripts/run_challenge_evaluation.py` |
| Legacy security benchmark | 12 | Prompt injection, jailbreak, leakage, poisoning, and unsafe requests | `scripts/run_security_evaluation.py` |
| Enterprise safety benchmark | 15 | Enterprise isolation, grounding, citations, injection, contradictions, and provider failures | `scripts/run_enterprise_safety_evaluation.py` |
| Repository pytest suite | 181 | Unit, integration, API, evaluator, dashboard reporting, bilingual, and safety regression tests | `pytest -q` |

The evaluator reports deterministic proxies for context precision, context recall, faithfulness, answer relevancy, citation accuracy, hallucination risk, and safe refusal behavior. These are transparent regression metrics, not external LLM-judge scores.

Quality thresholds are stored in `configs/rag_eval_config.yaml` and checked by `scripts/run_regression_checks.py`.

## 11. Benchmark Results

The latest repository-wide test run and saved evaluation artifacts report:

| Verification | Result |
|---|---:|
| Repository-wide pytest suite | 181 passed |
| Main bilingual benchmark | 73/73 passed |
| Challenge benchmark | 26/26 passed |
| Legacy security benchmark | 12/12 passed |
| Enterprise safety benchmark | 15/15 passed |
| Regression quality gates | 9/9 passed |

The main and challenge summary artifacts report `overall_pass_rate = 1.0000`; the security summaries report pass rates of `1.0000` on their included cases. These results are bounded by the repository datasets and rules. They do not demonstrate complete security coverage, production reliability, or accuracy on arbitrary documents and questions.

## 12. Evaluation Dashboard

The Streamlit dashboard reads saved case-level result artifacts and separates:

- false acceptance;
- false refusal;
- answer over-extraction;
- citation validation failure;
- numerical or polarity contradiction;
- prompt-injection compliance.

Directly measured categories show case IDs, expected behavior, actual behavior, pass/fail status, and available failure reasons. False refusal is explicitly labeled as a derived refusal-marker diagnostic rather than a quality gate. Answer over-extraction is labeled unmeasured because no persisted benchmark artifact provides a dedicated reliable metric; pytest coverage is not converted into a historical count.

The suite overview reports each saved artifact independently; it does not add overlapping dashboard categories to create totals. The 15-case enterprise safety suite is distinct from the prompt-injection compliance category, which currently selects nine relevant cases across the enterprise and legacy security artifacts.

![Evaluation dashboard quality categories](docs/screenshots/evaluation-dashboard.png)

*Category-based reporting distinguishes directly measured results, derived diagnostics, and quality dimensions without a dedicated persisted metric.*

![Saved evaluation suite overview and enterprise safety cases](docs/screenshots/enterprise-safety.png)

*The saved-result overview reports all four suites independently and expands the 15 `ENT-SAFE-*` enterprise safety outcomes. These are saved evaluation results, not freshly executed live-provider tests.*

## 13. Reproducing the Tests

Run the repository-wide test suite:

```bash
.venv/bin/python -m pytest -q
```

Regenerate evaluation artifacts and run gates:

```bash
.venv/bin/python scripts/run_evaluation.py
.venv/bin/python scripts/run_security_evaluation.py
.venv/bin/python scripts/run_enterprise_safety_evaluation.py
.venv/bin/python scripts/run_challenge_evaluation.py
.venv/bin/python scripts/run_regression_checks.py
```

Run the retained legacy generated-asset checks:

```bash
.venv/bin/python scripts/run_ai_testing_workflow.py
.venv/bin/python -m pytest tests/test_generated_assets.py -q
```

The evaluation commands rewrite generated files under `results/`. Review those diffs before committing them. See [Testing and Reproducibility](docs/TESTING.md) for suite boundaries, mocked-provider behavior, and quality-gate details.

## 14. Known Limitations

- Evidence sufficiency and factual validation are deterministic and cannot prove general semantic entailment.
- Citation validation confirms that a filename was retrieved; additional checks cover selected fact classes but not every possible contradiction.
- The enterprise index is in memory and scoped to the current Streamlit session. Restarting the app requires reindexing.
- Only UTF-8 `.md` and `.txt` enterprise documents are supported.
- There is no database, document deletion UI, version history, or durable vector index.
- The application has no production authentication, authorization, tenant isolation, audit policy, or multi-user access controls.
- Provider tests in CI are mocked; live-provider behavior can vary by model, provider, and configuration.
- The prepared benchmarks are finite and domain-specific. A perfect saved pass rate does not imply universal accuracy or security.
- False-refusal dashboard reporting is derived and phrase-based. Answer over-extraction has regression tests but no dedicated persisted benchmark metric.
- The FastAPI compatibility service does not expose enterprise document management or enterprise LLM-grounded Q&A.

## 15. Project Structure

```text
.
├── app.py                         # Streamlit assistant and evaluation dashboard
├── api.py                         # Legacy customer-support FastAPI service
├── ai_testing/                    # Retained compatibility test-generation/RCA package
├── configs/
│   └── rag_eval_config.yaml       # Datasets, retrieval defaults, and quality gates
├── data/
│   ├── documents/                 # Immutable legacy benchmark corpus
│   ├── enterprise_documents/      # Separate managed enterprise corpus
│   └── evaluation/                # Main, challenge, security, and enterprise safety cases
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TESTING.md
│   ├── QUALITY_CASE_STUDIES.md
│   ├── EVALUATION_METRICS.md
│   ├── SECURITY_EVALUATION.md
│   ├── AI_ASSISTED_TESTING.md      # Legacy compatibility documentation
│   └── screenshots/
├── results/                       # Saved case-level and summary evaluation artifacts
├── scripts/                       # Evaluation, safety, compatibility, and gate runners
├── src/
│   ├── document_manager.py
│   ├── enterprise_assistant.py
│   ├── grounded_llm.py
│   ├── quality_reporting.py
│   ├── rag_pipeline.py
│   ├── retrieval.py
│   ├── evaluation/
│   └── retrievers/
├── test_assets/                   # Retained generated-test compatibility assets
├── tests/                         # Repository-wide pytest suite
├── requirements.txt
└── requirements-vector.txt
```

Additional references:

- [Architecture](docs/ARCHITECTURE.md)
- [Testing and Reproducibility](docs/TESTING.md)
- [Quality Case Studies](docs/QUALITY_CASE_STUDIES.md)
- [Evaluation Metrics](docs/EVALUATION_METRICS.md)
- [Security Evaluation](docs/SECURITY_EVALUATION.md)
- [Legacy AI-Assisted Testing Compatibility](docs/AI_ASSISTED_TESTING.md)
