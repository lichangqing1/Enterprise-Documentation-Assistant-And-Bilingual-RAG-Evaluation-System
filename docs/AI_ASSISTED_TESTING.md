# Legacy AI-Assisted Testing Compatibility

> **Status:** Retained for backward compatibility. The `ai_testing` package, generated assets, FastAPI `/testing/generate` endpoint, tests, and CI step remain operational, but this workflow is not the primary Enterprise Documentation Assistant feature set.

This module documents the repository's historical "Using AI for Testing" capability. It turns requirements into structured, reusable testing assets that can feed the preserved RAG evaluation workflow.

## Purpose

The original evaluation suite answers the question:

```text
Can we test the RAG system?
```

The AI-assisted testing layer adds:

```text
Can we use AI-style generation to improve test design and failure analysis?
```

The retained compatibility workflow is:

```text
Requirement -> Test Assets -> Execution -> Evaluation -> Failure Analysis -> Quality Summary
```

## Workflow

```text
Requirement or acceptance criteria
        |
        v
Requirement parser
        |
        v
Structured scenarios
        |
        v
Generated evaluation cases
        |
        v
Template-generated pytest case
        |
        v
RAG evaluation results
        |
        +-> Deterministic failure classification and RCA suggestions
        +-> Optional LLM RCA enrichment
        +-> Metric-grounded quality summary
```

## Design Principle

Generated content is treated as data before it is trusted.

```text
AI-style generation -> Pydantic schema validation -> saved test asset -> deterministic execution
```

The default implementation is deterministic and offline-friendly, which keeps local runs and CI stable. An optional OpenAI-compatible JSON client is available for LLM-backed scenario generation, but it must be requested explicitly.

## Generator Backends

The generator interface separates deterministic CI behavior from optional LLM-backed generation.

| Backend | File | Purpose |
|---|---|---|
| `RuleBasedGenerator` | `ai_testing/generators.py` | Domain-specific deterministic baseline for CI and unit tests |
| `LLMTestGenerator` | `ai_testing/generators.py` | Optional adapter for LLM-backed requirement interpretation |
| `OpenAICompatibleJSONClient` | `ai_testing/llm_client.py` | Minimal dependency-free chat-completions client for JSON-only generation |

The optional LLM backend is deliberately narrow. It should be used for:

- requirement to structured scenarios;
- failed-case explanation or RCA enrichment.

The RAG evaluation stack itself remains framework-light and deterministic by default.

The deterministic generator is intentionally not presented as a generic natural-language test designer. It covers the customer-support RAG domain used by this repository and provides reproducible baseline cases. For arbitrary product requirements such as form validation, pricing rules, or workflow constraints, use the optional LLM backend or add a new domain-specific rule family.

Run deterministic generation:

```bash
python scripts/run_ai_testing_workflow.py --generator rule_based
```

Run optional LLM-backed generation:

```bash
export LLM_API_KEY="your_api_key"
export LLM_MODEL="gpt-4o-mini"
python scripts/run_ai_testing_workflow.py --generator llm
```

If `--generator llm` is used without `LLM_API_KEY` or `OPENAI_API_KEY`, the script raises a clear configuration error instead of silently falling back to deterministic generation.

## Modules

| File | Responsibility |
|---|---|
| `ai_testing/schemas.py` | Pydantic models for requirements, scenarios, generated cases, failure analyses, and quality summaries |
| `ai_testing/requirement_parser.py` | Converts free-form requirement text into a structured requirement spec |
| `ai_testing/scenario_generator.py` | Creates positive, negative, boundary, security, and bilingual scenarios |
| `ai_testing/test_data_generator.py` | Converts scenarios into evaluator-compatible test rows |
| `ai_testing/case_generator.py` | Saves JSON/CSV assets and creates pytest code from a controlled template |
| `ai_testing/executor.py` | Executes generated cases against the real RAG pipeline |
| `ai_testing/generators.py` | Provides deterministic and optional LLM generator backends |
| `ai_testing/llm_client.py` | Minimal optional LLM boundary for JSON-only generation |
| `ai_testing/failure_analyzer.py` | Deterministic failure taxonomy and first-level triage |
| `ai_testing/rca_analyzer.py` | Optional LLM-assisted evidence-grounded root-cause analysis |
| `ai_testing/quality_summary.py` | Converts deterministic metrics into a human-readable quality summary |

Prompt templates live in:

```text
ai_testing/prompts/
├── requirement_analysis.txt
├── test_generation.txt
└── failure_analysis.txt
```

## Generated Assets

Run:

```bash
python scripts/run_ai_testing_workflow.py
```

The workflow writes:

```text
test_assets/generated_cases/ai_testing_scenarios.json
test_assets/generated_cases/ai_generated_cases.json
test_assets/generated_cases/ai_generated_eval_cases.csv
test_assets/generated_cases/generated_pytest_cases.py
test_assets/generated_cases/ai_generated_evaluation_results.csv
test_assets/generated_cases/ai_generated_failed_cases.csv
test_assets/generated_cases/quality_summary.json
test_assets/generated_cases/failure_analysis.json
test_assets/generated_cases/llm_rca_analysis.json
```

The sample input requirement is:

```text
test_assets/requirements/sample_rag_requirement.md
```

## API Endpoint

FastAPI exposes the same generation path:

```text
POST /testing/generate
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/testing/generate \
  -H "Content-Type: application/json" \
  -d '{"requirement_id": "RAG-AI-001", "requirement_text": "RAG answers must cite sources, refuse unsupported questions, and include bilingual examples."}'
```

The endpoint returns:

- parsed requirement metadata;
- structured test scenarios;
- generated evaluation-ready test cases.

It also logs generation events to:

```text
logs/ai_testing.jsonl
```

## Generated-Case Execution

Generated tests are executed by stable pytest code rather than arbitrary generated Python.

```text
test_assets/generated_cases/ai_generated_cases.json
        |
        v
tests/test_generated_assets.py
        |
        v
real RAG pipeline + evaluator
```

This closes the loop from requirement to executable regression check while keeping generated content auditable.

## Failure Taxonomy

Failed cases are mapped into a stable taxonomy:

```text
RETRIEVAL_MISS
RETRIEVAL_NOISE
GROUNDING_FAILURE
CITATION_FAILURE
HALLUCINATION
ABNORMAL_REFUSAL
UNSAFE_RESPONSE
ANSWER_RELEVANCE
SYSTEM_FAILURE
```

This makes RCA summaries easier to aggregate than free-form failure strings.

## Optional LLM RCA

LLM-based RCA is available as an optional enrichment layer after deterministic triage. It receives the failed row, evaluator metrics, retrieved sources, expected behavior, actual answer, and deterministic failure category. The prompt requires the model to use only supplied evidence, and the response must validate against `LLMRootCauseAnalysis`.

Default deterministic workflow:

```bash
python scripts/run_ai_testing_workflow.py
```

Optional LLM RCA workflow:

```bash
export LLM_API_KEY="your_api_key"
python scripts/run_ai_testing_workflow.py --rca llm
```

LLM RCA writes:

```text
test_assets/generated_cases/llm_rca_analysis.json
```

The LLM explains failures; it does not decide CI pass/fail status or replace deterministic quality gates.

## Extension Path

The project includes optional LLM-backed generation and RCA paths:

```text
OpenAI-compatible client -> schema validation -> generated test assets / RCA diagnostics -> current evaluator and CI
```

That keeps the engineering chain stable: the model can propose test assets, but Python validates, saves, executes, and evaluates them.
