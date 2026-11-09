# Evaluation Metrics

This project reports deterministic RAG evaluation proxies with industry-facing names while keeping a few legacy column aliases for backward compatibility. These metrics are transparent regression checks, not external RAGAS or LLM-judge scores.

The metrics are used by two workflows:

- the RAG evaluation scripts, which score the current assistant behavior;
- the retained AI-assisted testing compatibility workflow, which turns generated test assets and failed cases into structured quality summaries.

## Evaluation Layer

The framework uses a compact evaluator design inspired by benchmark systems such as OpenCompass:

- **Dataset**: CSV files under `data/evaluation/` define questions, expected answers, expected sources, expected keywords, and task labels.
- **Inferencer**: `src/rag_pipeline.py` retrieves evidence and generates source-cited extractive answers.
- **Evaluator**: `src/evaluation/evaluator.py` computes retrieval, faithfulness, citation, hallucination-risk, and safety metrics.
- **Reporter**: `src/report_generator.py` writes CSV and Markdown reports under `results/`.
- **Regression gate**: `scripts/run_regression_checks.py` checks minimum quality thresholds.
- **AI-assisted testing**: `ai_testing/quality_summary.py` and `ai_testing/failure_analyzer.py` reuse deterministic metric outputs for release summaries and RCA suggestions.

Future adapters for tools such as RAGAS, DeepEval, or OpenCompass can be added without changing the current deterministic evaluator.

The separate enterprise safety suite stores case-level expected behavior, actual behavior, outcomes, failure reasons, severity, and corpus-isolation status in `results/enterprise_safety_evaluation_results.csv`. It uses mocked providers and is not folded into the main 73-case metric summary.

## Metric Name Mapping

| Earlier idea / legacy column | Industry-facing name |
|---|---|
| `answer_groundedness` | `faithfulness` / `proxy_faithfulness` |
| `context_keyword_recall` | `context_recall` / `proxy_context_recall` |
| retrieved context quality | `context_precision` |
| answer relevance | `answer_relevancy` / `proxy_answer_relevancy` |
| `hallucination_risk` | `faithfulness_failure_hallucination_risk` |
| `source_citation` | `citation_accuracy` |
| `ragas_*` columns | legacy aliases for deterministic `proxy_*` columns |

## `faithfulness`

**Purpose**

Measures whether the generated answer is supported by the retrieved context.

**Formula / logic**

The evaluator compares answer claim sentences with the retrieved context. Exact evidence matches score highly. Partial lexical overlap scores proportionally. Safe refusals for unanswerable questions are skipped.

Legacy alias: `answer_groundedness`.

**Example pass case**

- Retrieved context: "Standard shipping usually takes 3 to 5 business days."
- Answer: "Standard shipping usually takes 3 to 5 business days. Source: shipping_policy.md"
- Result: high faithfulness because the answer claim is directly supported.

**Example fail case**

- Retrieved context: "Standard shipping usually takes 3 to 5 business days."
- Answer: "Standard shipping always arrives overnight."
- Result: low faithfulness because "overnight" is unsupported.

**Limitation**

This is a deterministic proxy, not an LLM judge. It may miss paraphrases or accept superficial token overlap.

## `context_recall`

**Purpose**

Measures whether the retrieved context contains the expected evidence needed to answer the question.

**Formula / logic**

The evaluator checks the proportion of expected keywords that appear in the retrieved context. Unanswerable cases are skipped because the desired behavior is safe refusal, not evidence retrieval.

Legacy alias: `context_keyword_recall`.

**Example pass case**

- Expected keywords: "standard shipping;3 to 5 business days"
- Retrieved context contains both phrases.
- Result: `context_recall = 1.0`

**Example fail case**

- Expected keywords: "refund;30 days"
- Retrieved context only discusses shipping time.
- Result: low context recall.

**Limitation**

Keyword recall can under-score correct paraphrases and over-score context that mentions keywords without fully answering the question.

## `context_precision`

**Purpose**

Measures whether the most relevant retrieved context appears near the top of the retrieved results.

**Formula / logic**

The evaluator checks the rank of the expected source in the retrieved source list. A rank-1 source scores `1.0`; lower ranks score lower using `1 / rank`; missing expected sources score `0.0`.

Legacy alias: `ragas_context_precision`.

**Example pass case**

- Expected source: `shipping_policy.md`
- Retrieved sources: `shipping_policy.md;return_policy.md`
- Result: `context_precision = 1.0`

**Example fail case**

- Expected source: `shipping_policy.md`
- Retrieved sources: `payment_policy.md;return_policy.md`
- Result: `context_precision = 0.0`

**Limitation**

This source-rank proxy does not inspect every sentence in the retrieved chunks. It assumes the labeled expected source is the best evidence.

## `answer_relevancy`

**Purpose**

Measures whether the answer addresses the expected information need.

**Formula / logic**

For answerable cases, the evaluator uses expected keyword coverage in the answer as a deterministic answer-relevancy proxy. Unanswerable cases are handled separately by `unanswerable_safe`.

Legacy alias: `ragas_answer_relevancy`.

## Proxy Metric Naming

The preferred internal names are:

```text
proxy_context_precision
proxy_context_recall
proxy_faithfulness
proxy_answer_relevancy
```

Older `ragas_*` columns are still emitted as compatibility aliases, but the project does not claim to call the external RAGAS package in the default evaluator.

**Example pass case**

- Question: "How long does standard shipping take?"
- Expected keywords: "standard shipping;3 to 5 business days"
- Answer includes both expected pieces of information.
- Result: high answer relevancy.

**Example fail case**

- Question: "How long does standard shipping take?"
- Answer: "We offer several delivery options."
- Result: low answer relevancy because the answer does not provide the expected shipping time.

**Limitation**

Keyword-based relevancy is transparent and reproducible, but it is less flexible than an LLM-based semantic relevance judge.

## `faithfulness_failure_hallucination_risk`

**Purpose**

Flags risk that the answer contains unsupported claims.

**Formula / logic**

The metric is calculated as:

```text
faithfulness_failure_hallucination_risk = 1 - faithfulness
```

Legacy alias: `hallucination_risk`.

**Example pass case**

- Faithfulness: `1.0`
- Hallucination risk: `0.0`
- Interpretation: answer claims are fully supported by retrieved context.

**Example fail case**

- Faithfulness: `0.25`
- Hallucination risk: `0.75`
- Interpretation: much of the answer is unsupported.

**Limitation**

This is a risk proxy, not proof of hallucination. Low lexical overlap can also occur when an answer is a valid paraphrase.

## `citation_accuracy`

**Purpose**

Measures whether the answer cites a source that was actually retrieved.

**Formula / logic**

The evaluator looks for a `Source: filename.md` citation in the answer and checks whether the cited filename appears in the retrieved source list. Answerable cases should cite a retrieved source. Unanswerable safe refusals are skipped.

Legacy alias: `source_citation` only checks whether a citation string exists; `citation_accuracy` checks whether the citation is valid.

**Example pass case**

- Retrieved sources: `shipping_policy.md;return_policy.md`
- Answer citation: `Source: shipping_policy.md`
- Result: `citation_accuracy = 1`

**Example fail case**

- Retrieved sources: `shipping_policy.md;return_policy.md`
- Answer citation: `Source: warranty_policy.md`
- Result: `citation_accuracy = 0`

**Limitation**

This metric validates source filename consistency. It does not prove that every sentence in the answer is supported by the cited source; use it together with `faithfulness`.

## `unanswerable_safe`

**Purpose**

Measures whether unsupported questions are refused safely instead of answered from guesswork.

**Formula / logic**

For rows labeled `unanswerable`, the evaluator checks whether the answer contains safe-refusal language such as "does not mention", "not enough information", "没有提到", or "无法确认".

**Example pass case**

- Question: "Can I pay with cryptocurrency?"
- Documents do not mention cryptocurrency.
- Answer: "The documents do not mention cryptocurrency payments."
- Result: `unanswerable_safe = 1`

**Example fail case**

- Question: "Can I pay with cryptocurrency?"
- Answer: "Yes, cryptocurrency is accepted."
- Result: `unanswerable_safe = 0`

**Limitation**

The current check is phrase-based. It may need more refusal patterns for new languages, domains, or answer styles.

## `overall_pass_rate`

**Purpose**

Summarizes whether the RAG system passes all applicable checks across the evaluation dataset.

**Formula / logic**

Each row gets pass/fail flags for applicable metrics. `overall_pass` is `1` only when all applicable checks pass. `overall_pass_rate` is the mean of row-level `overall_pass`.

**Example pass case**

- Retrieval finds the expected source.
- Context contains expected evidence.
- Answer is faithful, relevant, cited, and safe.
- Result: row-level `overall_pass = 1`.

**Example fail case**

- Retrieval misses the expected source or answer includes unsupported claims.
- Result: row-level `overall_pass = 0`.

**Limitation**

The aggregate score is only as strong as the evaluation dataset. Add more representative questions when expanding the knowledge base.

## How Metrics Feed AI-Assisted Testing

The AI-assisted testing layer does not invent metric values. Python computes the numbers first, then the testing layer summarizes and classifies the results.

```text
evaluation_results.csv -> add_pass_fail_flags -> failed_cases -> failure_analyzer
summary_report.csv     -> quality_summary
```

This keeps the workflow reliable:

- code calculates pass/fail decisions;
- structured schemas validate generated test assets;
- generated summaries explain results without changing the underlying measurements.

## Quality Gates

Regression thresholds are configured in `configs/rag_eval_config.yaml` under `quality_gates`.

`scripts/run_regression_checks.py` reads those values at runtime, which keeps local validation and CI aligned.

## Evaluation Dashboard Categories

`src/quality_reporting.py` maps saved artifacts into additional review categories without changing evaluator outputs or quality gates:

| Category | Measurement status |
|---|---|
| False acceptance | Directly measured from unanswerable and enterprise unsupported-question cases |
| False refusal | Derived phrase-based diagnostic; not a quality gate |
| Answer over-extraction | Not measured by a dedicated persisted benchmark metric |
| Citation validation failure | Direct benchmark flags and enterprise citation-fabrication case |
| Numerical or polarity contradiction | Dedicated enterprise safety cases |
| Prompt-injection compliance | Relevant enterprise and legacy security cases |

The dashboard does not infer historical failure counts from pytest tests or claim comprehensive detection.
