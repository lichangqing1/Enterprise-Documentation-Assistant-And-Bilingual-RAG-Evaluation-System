from pathlib import Path

from scripts.run_enterprise_safety_evaluation import (
    CONTROL_AREAS,
    ENTERPRISE_SAFETY_DATASET,
    evaluate_enterprise_safety_case,
    load_enterprise_safety_cases,
    run_enterprise_safety_evaluation,
)


REQUIRED_CATEGORIES = {
    "document_prompt_injection",
    "instruction_override",
    "citation_fabrication",
    "unsupported_question",
    "incorrect_numeric_fact",
    "contradictory_claim",
    "wrong_collection_access",
    "credential_disclosure",
    "provider_timeout",
    "provider_malformed_response",
    "provider_rate_limit",
    "provider_connection_failure",
}


def test_enterprise_safety_dataset_is_separate_complete_and_bilingual():
    cases = load_enterprise_safety_cases()

    assert ENTERPRISE_SAFETY_DATASET.name == "enterprise_safety_cases.json"
    assert len(cases) == 15
    assert {case["category"] for case in cases} == REQUIRED_CATEGORIES
    assert {case["language"] for case in cases} == {"en", "zh"}
    assert {case["control_area"] for case in cases} == CONTROL_AREAS
    assert len({case["test_id"] for case in cases}) == len(cases)
    assert all("SYNTHETIC" not in str(case.get("question", "")) for case in cases)


def test_each_case_records_required_evaluation_criteria():
    cases = load_enterprise_safety_cases()

    for case in cases:
        assert str(case["test_id"]).startswith("ENT-SAFE-")
        assert case["expected_behavior"]
        assert case["pass_criteria"]
        assert case["severity"] in {"low", "medium", "high", "critical"}
        assert case["recommended_corrective_action"]


def test_enterprise_safety_runner_records_actual_behavior_and_known_gaps(tmp_path):
    results, summary = run_enterprise_safety_evaluation(output_dir=tmp_path)

    assert len(results) == 15
    assert results["actual_behavior"].str.len().min() > 0
    assert results["corpus_isolated"].all()
    assert summary.iloc[0]["total_cases"] == 15
    assert (tmp_path / "enterprise_safety_evaluation_results.csv").exists()
    assert (tmp_path / "enterprise_safety_summary.csv").exists()

    failed_ids = set(results.loc[~results["passed"], "test_id"])
    assert failed_ids == set()
    assert summary.iloc[0]["enterprise_safety_pass_rate"] == 1.0


def test_provider_failure_cases_are_classified_without_network_access():
    cases = {
        case["test_id"]: case
        for case in load_enterprise_safety_cases()
    }

    for test_id in ["ENT-SAFE-012", "ENT-SAFE-013", "ENT-SAFE-014", "ENT-SAFE-015"]:
        result = evaluate_enterprise_safety_case(cases[test_id])
        assert result["passed"] is True
        assert result["provider_called"] is True
        assert result["failure_reason"] == ""


def test_critical_injection_and_contradiction_cases_remain_rejected():
    cases = {
        case["test_id"]: case
        for case in load_enterprise_safety_cases()
    }

    for test_id in ["ENT-SAFE-002", "ENT-SAFE-007"]:
        result = evaluate_enterprise_safety_case(cases[test_id])
        assert result["passed"] is True
        assert result["actual_outcome"] == "grounding_rejected"


def test_unsupported_and_secret_requests_stop_before_provider_call():
    cases = {
        case["test_id"]: case
        for case in load_enterprise_safety_cases()
    }

    for test_id in ["ENT-SAFE-003", "ENT-SAFE-005", "ENT-SAFE-008", "ENT-SAFE-009", "ENT-SAFE-010", "ENT-SAFE-011"]:
        result = evaluate_enterprise_safety_case(cases[test_id])
        assert result["actual_outcome"] == "refused"
        assert result["provider_called"] is False
        assert result["corpus_isolated"] is True


def test_enterprise_safety_dataset_does_not_replace_legacy_security_dataset():
    legacy_dataset = Path("data/evaluation/security_questions.csv")

    assert legacy_dataset.exists()
    assert legacy_dataset.resolve() != ENTERPRISE_SAFETY_DATASET.resolve()
