# Testing and Reproducibility

## Scope

The repository uses pytest, deterministic evaluation scripts, saved case-level artifacts, and configuration-driven regression gates. The reported `181 passed` result is repository-wide: it includes document management, retrieval, bilingual RAG, extractive and mocked LLM-grounded Q&A, API behavior, evaluators, safety cases, dashboard reporting, and legacy compatibility tests.

## Environment

Python 3.11 is used in GitHub Actions and the Dockerfile.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

FAISS tests or manual FAISS runs require:

```bash
.venv/bin/python -m pip install -r requirements-vector.txt
```

The default test and evaluation path does not require the optional vector dependencies or an API key.

## Repository-Wide Tests

```bash
.venv/bin/python -m pytest -q
```

Current verified result:

```text
181 passed
```

This number describes the current repository test collection. It is separate from benchmark case counts.

## Evaluation Suites

### Main bilingual benchmark

```bash
.venv/bin/python scripts/run_evaluation.py
```

- Inputs: `rag_eval_en.csv`, `rag_eval_zh.csv`, and `security_questions.csv`.
- Cases: 73.
- Purpose: known English/Chinese retrieval, answer, citation, refusal, and security behavior.
- Outputs: `evaluation_results.csv`, `failed_cases.csv`, `summary_report.csv`, and `evaluation_report.md`.

Saved result: 73/73 passed, `overall_pass_rate = 1.0000`.

### Challenge benchmark

```bash
.venv/bin/python scripts/run_challenge_evaluation.py
```

- Input: `challenge_questions.csv`.
- Cases: 26.
- Purpose: paraphrases, unsupported constraints, ambiguous requests, and bilingual robustness.
- Outputs: `challenge_evaluation_results.csv`, `challenge_failed_cases.csv`, and `challenge_summary.csv`.

Saved result: 26/26 passed, `overall_pass_rate = 1.0000`.

### Legacy security benchmark

```bash
.venv/bin/python scripts/run_security_evaluation.py
```

- Input: `security_questions.csv`.
- Cases: 12.
- Purpose: prompt injection, jailbreak, prompt leakage, sensitive-information disclosure, retrieval poisoning, and unsafe-instruction refusal.
- Outputs: `security_evaluation_results.csv` and `security_summary.csv`.

Saved result: 12/12 passed, `security_pass_rate = 1.0000`.

### Enterprise safety benchmark

```bash
.venv/bin/python scripts/run_enterprise_safety_evaluation.py
```

- Input: `enterprise_safety_cases.json`.
- Cases: 15.
- Purpose: document injection, instruction override, fabricated citations, unsupported questions, number changes, polarity reversals, corpus isolation, credential requests, and provider failures.
- Outputs: `enterprise_safety_evaluation_results.csv` and `enterprise_safety_summary.csv`.

Saved result: 15/15 passed, `enterprise_safety_pass_rate = 1.0000`.

Each enterprise safety case runs against a temporary isolated collection. LLM cases use `CaseLLMClient`, a deterministic provider double that performs no network request.

## Mocked LLM Tests vs. Live Provider Checks

Automated tests inject mocked clients into `ask_grounded_llm_question()`. They verify prompts, provider-call decisions, structured responses, citation validation, injection quarantine, numerical checks, polarity checks, and provider error mapping without credentials, network access, or paid requests.

A manual live OpenAI-compatible provider smoke check was previously reported as three successful requests. That check is not part of pytest, CI, or the saved benchmark artifacts. The repository does not store its credentials or raw responses. Repeating a live check requires the operator to configure a provider explicitly and review data-handling requirements first.

## Regression Quality Gates

Run after generating the main summary:

```bash
.venv/bin/python scripts/run_regression_checks.py
```

The nine thresholds come from `configs/rag_eval_config.yaml`:

| Gate | Required value |
|---|---:|
| `overall_pass_rate` | at least 0.90 |
| `avg_source_match` | at least 0.75 |
| `avg_keyword_recall` | at least 0.60 |
| `avg_unanswerable_safe` | at least 0.80 |
| `avg_context_precision` | at least 0.75 |
| `avg_context_recall` | at least 0.75 |
| `avg_faithfulness` | at least 0.80 |
| `avg_answer_relevancy` | at least 0.60 |
| `max_avg_faithfulness_failure_hallucination_risk` | at most 0.20 |

Current saved summary: 9/9 gates passed. Thresholds are not relaxed to make a run pass.

## Recommended Reproduction Order

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_evaluation.py
.venv/bin/python scripts/run_security_evaluation.py
.venv/bin/python scripts/run_enterprise_safety_evaluation.py
.venv/bin/python scripts/run_challenge_evaluation.py
.venv/bin/python scripts/run_regression_checks.py
```

Evaluation runners write generated artifacts under `results/` and append selected run metadata under `logs/`. Review changed artifacts before committing.

## Legacy Compatibility Tests

The historical AI-assisted testing workflow remains available:

```bash
.venv/bin/python scripts/run_ai_testing_workflow.py
.venv/bin/python -m pytest tests/test_generated_assets.py -q
```

This workflow is preserved for backward compatibility. It is not required for enterprise document upload, indexing, retrieval, or Q&A.

## Dashboard Quality Categories

The Evaluation Dashboard reads existing CSV artifacts and does not create new benchmark results.

| Category | Reporting basis |
|---|---|
| False acceptance | Direct unanswerable-case and enterprise unsupported-question results |
| False refusal | Derived diagnostic using existing safe-refusal phrase detection on supported answers; not a gate |
| Answer over-extraction | No dedicated persisted benchmark metric; shown as unmeasured |
| Citation validation | Benchmark citation flags plus the enterprise fabricated-citation case |
| Numerical/polarity contradiction | Dedicated enterprise safety cases |
| Prompt-injection compliance | Relevant enterprise and legacy security cases |

The dashboard does not invent counts for categories without reliable persisted measurements.

## Interpreting Results

A passing benchmark means the current implementation satisfied the expected behavior encoded in that finite dataset. It does not prove:

- correctness for arbitrary documents or questions;
- complete semantic entailment;
- complete prompt-injection resistance;
- provider-independent behavior;
- production security or multi-user readiness;
- measured latency, throughput, or cost.
