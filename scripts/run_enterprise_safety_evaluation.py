"""Run the isolated Enterprise Documentation Assistant safety evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Dict
from urllib.error import HTTPError, URLError

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_manager import add_enterprise_document, reindex_enterprise_documents
from src.enterprise_assistant import ask_enterprise_question
from src.grounded_llm import (
    GroundedLLMProviderError,
    GroundedLLMResponseError,
    ask_grounded_llm_question,
)


ENTERPRISE_SAFETY_DATASET = PROJECT_ROOT / "data" / "evaluation" / "enterprise_safety_cases.json"
REQUIRED_FIELDS = {
    "test_id",
    "category",
    "control_area",
    "language",
    "answering_mode",
    "document_filename",
    "document_content",
    "question",
    "expected_outcome",
    "expected_provider_called",
    "expected_behavior",
    "pass_criteria",
    "severity",
    "recommended_corrective_action",
}
CONTROL_AREAS = {
    "security_control_failure",
    "unsupported_answer_failure",
    "citation_validation_failure",
    "grounding_failure",
    "provider_integration_failure",
}
SEVERITIES = {"low", "medium", "high", "critical"}
SETTINGS = {
    "top_k": 3,
    "chunk_size": 500,
    "overlap": 100,
    "retrieval_mode": "lexical",
    "semantic_backend": "local",
}


class CaseLLMClient:
    """Deterministic provider double; it never performs network access."""

    def __init__(self, response: object = None, error_name: str = ""):
        self.response = response
        self.error_name = error_name
        self.prompts: list[str] = []

    def complete_json(self, prompt: str) -> object:
        self.prompts.append(prompt)
        if self.error_name == "timeout":
            raise TimeoutError("synthetic provider timeout")
        if self.error_name == "rate_limit":
            raise HTTPError("https://provider.invalid", 429, "synthetic rate limit", {}, None)
        if self.error_name == "connection":
            raise URLError("synthetic connection failure")
        if self.error_name:
            raise RuntimeError(f"Unsupported synthetic provider error: {self.error_name}")
        return self.response


def load_enterprise_safety_cases(
    dataset_path: Path = ENTERPRISE_SAFETY_DATASET,
) -> list[Dict[str, object]]:
    """Load and validate the standalone synthetic enterprise safety dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Enterprise safety dataset not found: {dataset_path}")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Enterprise safety dataset must contain a non-empty JSON list.")

    seen_ids = set()
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Enterprise safety case {position} must be a JSON object.")
        missing = sorted(REQUIRED_FIELDS - set(case))
        if missing:
            raise ValueError(f"Enterprise safety case {position} is missing fields: {missing}")
        test_id = str(case["test_id"])
        if test_id in seen_ids:
            raise ValueError(f"Duplicate enterprise safety test_id: {test_id}")
        seen_ids.add(test_id)
        if case["control_area"] not in CONTROL_AREAS:
            raise ValueError(f"Unsupported control_area in {test_id}: {case['control_area']}")
        if case["severity"] not in SEVERITIES:
            raise ValueError(f"Unsupported severity in {test_id}: {case['severity']}")
        if case["answering_mode"] not in {"extractive", "llm_grounded"}:
            raise ValueError(f"Unsupported answering_mode in {test_id}: {case['answering_mode']}")
        if case["language"] not in {"en", "zh"}:
            raise ValueError(f"Unsupported language in {test_id}: {case['language']}")
    return cases


def _classify_response_error(exc: GroundedLLMResponseError) -> str:
    message = str(exc).lower()
    if (
        "numeric facts" in message
        or "negation" in message
        or "quarantined" in message
        or "permission or prohibition" in message
    ):
        return "grounding_rejected"
    if "citation" in message or "cited" in message:
        return "citation_rejected"
    return "malformed_response_rejected"


def evaluate_enterprise_safety_case(case: Dict[str, object]) -> Dict[str, object]:
    """Execute one case against a fresh isolated enterprise collection."""
    client = CaseLLMClient(
        response=case.get("mock_response"),
        error_name=str(case.get("mock_error", "")),
    )
    response: Dict[str, object] | None = None
    actual_outcome = "unexpected_error"
    actual_behavior = ""

    with TemporaryDirectory(prefix="enterprise-safety-") as temp_dir:
        collection_path = Path(temp_dir)
        add_enterprise_document(
            str(case["document_filename"]),
            str(case["document_content"]),
            collection_path,
        )
        index = reindex_enterprise_documents(
            collection_path,
            chunk_size=SETTINGS["chunk_size"],
            overlap=SETTINGS["overlap"],
            retrieval_mode=SETTINGS["retrieval_mode"],
            semantic_backend=SETTINGS["semantic_backend"],
        )
        try:
            if case["answering_mode"] == "extractive":
                response = ask_enterprise_question(
                    str(case["question"]),
                    index,
                    collection_path,
                    **SETTINGS,
                )
            else:
                response = ask_grounded_llm_question(
                    str(case["question"]),
                    index,
                    collection_path,
                    client=client,
                    **SETTINGS,
                )
            actual_outcome = "refused" if response["status"] == "insufficient_evidence" else "answered"
            actual_behavior = (
                f"{response['answer']} | supporting_sources={response['supporting_sources']}"
            )
        except GroundedLLMResponseError as exc:
            actual_outcome = _classify_response_error(exc)
            actual_behavior = str(exc)
        except GroundedLLMProviderError as exc:
            actual_outcome = "provider_error"
            actual_behavior = str(exc)
        except Exception as exc:  # pragma: no cover - retained for diagnostic reports.
            actual_behavior = f"{type(exc).__name__}: {exc}"

        allowed_sources = {str(case["document_filename"])}
        observed_sources = (
            {str(source) for source in response.get("sources", [])}
            if response is not None
            else {str(chunk.get("source", "")) for chunk in index.chunks}
        )
        corpus_isolated = bool(observed_sources) and observed_sources.issubset(allowed_sources)

    provider_called = bool(client.prompts)
    outcome_matches = actual_outcome == case["expected_outcome"]
    provider_matches = provider_called is bool(case["expected_provider_called"])
    passed = outcome_matches and provider_matches and corpus_isolated

    reasons = []
    if not outcome_matches:
        reasons.append(
            f"expected outcome {case['expected_outcome']}, observed {actual_outcome}"
        )
    if not provider_matches:
        reasons.append(
            f"expected provider_called={case['expected_provider_called']}, observed {provider_called}"
        )
    if not corpus_isolated:
        reasons.append("observed a source outside the isolated enterprise collection")

    return {
        "test_id": case["test_id"],
        "category": case["category"],
        "control_area": case["control_area"],
        "language": case["language"],
        "answering_mode": case["answering_mode"],
        "question": case["question"],
        "document_filename": case["document_filename"],
        "expected_behavior": case["expected_behavior"],
        "expected_outcome": case["expected_outcome"],
        "actual_behavior": actual_behavior,
        "actual_outcome": actual_outcome,
        "pass_criteria": case["pass_criteria"],
        "passed": passed,
        "failure_reason": "" if passed else "; ".join(reasons),
        "failure_type": "" if passed else case["control_area"],
        "severity": case["severity"],
        "recommended_corrective_action": case["recommended_corrective_action"],
        "provider_called": provider_called,
        "corpus_isolated": corpus_isolated,
    }


def summarize_enterprise_safety_results(results: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary while retaining failure severity and category counts."""
    if results.empty:
        raise ValueError("Enterprise safety results cannot be empty.")
    failed = results[~results["passed"]]
    summary: Dict[str, object] = {
        "total_cases": int(len(results)),
        "passed_cases": int(results["passed"].sum()),
        "failed_cases": int(len(failed)),
        "enterprise_safety_pass_rate": float(results["passed"].mean()),
        "critical_failures": int((failed["severity"] == "critical").sum()),
        "high_failures": int((failed["severity"] == "high").sum()),
        "corpus_isolation_pass_rate": float(results["corpus_isolated"].mean()),
    }
    for category, group in results.groupby("category"):
        key = str(category).replace("-", "_")
        summary[f"{key}_case_count"] = int(len(group))
        summary[f"{key}_pass_rate"] = float(group["passed"].mean())
    return pd.DataFrame([summary])


def run_enterprise_safety_evaluation(
    dataset_path: Path = ENTERPRISE_SAFETY_DATASET,
    output_dir: Path = PROJECT_ROOT / "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every synthetic case and persist detailed and summary CSV reports."""
    cases = load_enterprise_safety_cases(dataset_path)
    results = pd.DataFrame(evaluate_enterprise_safety_case(case) for case in cases)
    summary = summarize_enterprise_safety_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "enterprise_safety_evaluation_results.csv", index=False)
    summary.to_csv(output_dir / "enterprise_safety_summary.csv", index=False)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enterprise documentation safety cases.")
    parser.add_argument("--dataset", type=Path, default=ENTERPRISE_SAFETY_DATASET)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    results, summary = run_enterprise_safety_evaluation(args.dataset, args.output_dir)
    print("Enterprise Documentation Assistant Safety Evaluation")
    print(summary.to_string(index=False))
    failed = results[~results["passed"]]
    if failed.empty:
        print("PASSED - all enterprise safety cases passed.")
    else:
        print("FAILED CASES")
        print(failed[["test_id", "category", "failure_type", "severity", "failure_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
