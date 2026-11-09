import os
from pathlib import Path

import pandas as pd

from src.quality_reporting import CASE_COLUMNS, build_quality_dashboard_report


def _write_results(results_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "question_type": "normal",
                "answer": "Supported answer. Source: policy.md.",
                "expected_answer": "Supported answer.",
                "expected_source": "policy.md",
                "citation_accuracy_pass": 1,
                "unanswerable_pass": None,
                "overall_pass": 1,
            },
            {
                "question_type": "unanswerable",
                "answer": "I could not find enough information.",
                "expected_answer": "Safely refuse.",
                "expected_source": "none",
                "citation_accuracy_pass": None,
                "unanswerable_pass": 1,
                "overall_pass": 1,
            },
        ]
    ).to_csv(results_dir / "evaluation_results.csv", index=False)

    pd.DataFrame(
        [
            {
                "question_type": "normal",
                "answer": "I could not find enough information.",
                "expected_answer": "A supported challenge answer.",
                "expected_source": "challenge.md",
                "citation_accuracy_pass": 0,
                "unanswerable_pass": None,
                "overall_pass": 0,
            },
            {
                "question_type": "unanswerable",
                "answer": "An unsupported claim.",
                "expected_answer": "Safely refuse.",
                "expected_source": "none",
                "citation_accuracy_pass": None,
                "unanswerable_pass": 0,
                "overall_pass": 0,
            },
        ]
    ).to_csv(results_dir / "challenge_evaluation_results.csv", index=False)

    pd.DataFrame(
        [
            {
                "question_type": "prompt_injection",
                "answer": "I could not find enough information.",
                "expected_answer": "Refuse the override.",
                "unanswerable_pass": 1,
                "overall_pass": 1,
            }
        ]
    ).to_csv(results_dir / "security_evaluation_results.csv", index=False)

    enterprise_rows = []
    for test_id, category in [
        ("ENT-SAFE-004", "citation_fabrication"),
        ("ENT-SAFE-005", "unsupported_question"),
        ("ENT-SAFE-006", "incorrect_numeric_fact"),
        ("ENT-SAFE-007", "contradictory_claim"),
        ("ENT-SAFE-002", "document_prompt_injection"),
    ]:
        enterprise_rows.append(
            {
                "test_id": test_id,
                "category": category,
                "expected_behavior": f"Expected behavior for {test_id}",
                "actual_behavior": f"Actual behavior for {test_id}",
                "passed": True,
                "failure_reason": "",
            }
        )
    pd.DataFrame(enterprise_rows).to_csv(
        results_dir / "enterprise_safety_evaluation_results.csv", index=False
    )


def test_quality_report_exposes_all_requested_categories(tmp_path):
    _write_results(tmp_path)

    report = build_quality_dashboard_report(tmp_path)
    categories = {category.key: category for category in report.categories}

    assert set(categories) == {
        "false_acceptance",
        "false_refusal",
        "answer_over_extraction",
        "citation_validation",
        "numerical_polarity",
        "prompt_injection",
    }
    assert all(list(category.cases.columns) == CASE_COLUMNS for category in report.categories)


def test_suite_overview_aggregates_saved_case_outcomes_without_category_sums(tmp_path):
    _write_results(tmp_path)

    report = build_quality_dashboard_report(tmp_path)
    suites = {suite.key: suite for suite in report.suites}

    assert (suites["main"].total_cases, suites["main"].passed_cases) == (2, 2)
    assert (suites["challenge"].total_cases, suites["challenge"].passed_cases) == (2, 0)
    assert (suites["legacy_security"].total_cases, suites["legacy_security"].passed_cases) == (1, 1)
    assert (
        suites["enterprise_safety"].total_cases,
        suites["enterprise_safety"].passed_cases,
        suites["enterprise_safety"].failed_cases,
        suites["enterprise_safety"].pass_rate,
    ) == (5, 5, 0, 1.0)
    assert list(suites["enterprise_safety"].cases["Test ID"])[0].startswith("ENT-SAFE-")


def test_direct_metrics_keep_case_evidence_and_failure_status(tmp_path):
    _write_results(tmp_path)
    report = build_quality_dashboard_report(tmp_path)
    categories = {category.key: category for category in report.categories}

    false_acceptance = categories["false_acceptance"].cases
    failed = false_acceptance[false_acceptance["Status"] == "Fail"].iloc[0]
    assert failed["Test ID"] == "CHALLENGE-002"
    assert failed["Actual behavior"] == "An unsupported claim."
    assert "safe refusal" in failed["Failure reason"]

    numeric_ids = set(categories["numerical_polarity"].cases["Test ID"])
    assert numeric_ids == {"ENT-SAFE-006", "ENT-SAFE-007"}
    assert "ENT-SAFE-004" in set(categories["citation_validation"].cases["Test ID"])


def test_false_refusal_is_labeled_as_derived_not_as_a_quality_gate(tmp_path):
    _write_results(tmp_path)
    report = build_quality_dashboard_report(tmp_path)
    false_refusal = next(category for category in report.categories if category.key == "false_refusal")

    assert false_refusal.measurement_level == "Derived diagnostic"
    failed = false_refusal.cases[false_refusal.cases["Status"] == "Fail"]
    assert list(failed["Test ID"]) == ["CHALLENGE-001"]
    assert "not a quality gate" in false_refusal.measurement_note


def test_over_extraction_does_not_invent_results_or_counts(tmp_path):
    _write_results(tmp_path)
    report = build_quality_dashboard_report(tmp_path)
    over_extraction = next(
        category for category in report.categories if category.key == "answer_over_extraction"
    )

    assert over_extraction.measurement_level == "Not measured"
    assert over_extraction.measured_cases is None
    assert over_extraction.failed_cases is None
    assert over_extraction.cases.empty
    assert "No saved benchmark artifact" in over_extraction.measurement_note


def test_missing_artifacts_are_reported_without_fabricating_cases(tmp_path):
    report = build_quality_dashboard_report(tmp_path)

    assert len(report.artifact_warnings) == 4
    assert all(suite.artifact_status == "Unavailable" for suite in report.suites)
    assert all(suite.pass_rate is None for suite in report.suites)
    assert all(category.cases.empty for category in report.categories)


def test_malformed_suite_artifact_does_not_display_a_pass_rate(tmp_path):
    _write_results(tmp_path)
    pd.DataFrame([{"question": "Missing persisted outcome"}]).to_csv(
        tmp_path / "evaluation_results.csv", index=False
    )

    report = build_quality_dashboard_report(tmp_path)
    main = next(suite for suite in report.suites if suite.key == "main")

    assert main.artifact_status == "Malformed"
    assert main.total_cases is None
    assert main.pass_rate is None
    assert any("missing required column: overall_pass" in warning for warning in report.artifact_warnings)


def test_suite_artifact_older_than_dataset_is_labeled_stale(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_results(results_dir)
    data_dir = tmp_path / "data" / "evaluation"
    data_dir.mkdir(parents=True)
    dataset = data_dir / "rag_eval_en.csv"
    dataset.write_text("question\nnewer case\n", encoding="utf-8")
    result_mtime = (results_dir / "evaluation_results.csv").stat().st_mtime
    os.utime(dataset, (result_mtime + 10, result_mtime + 10))

    report = build_quality_dashboard_report(results_dir, evaluation_data_dir=data_dir)
    main = next(suite for suite in report.suites if suite.key == "main")

    assert main.artifact_status == "Stale"
    assert main.pass_rate == 1.0
    assert any("may be stale" in warning for warning in report.artifact_warnings)


def test_repository_saved_suite_results_match_reported_baseline():
    repository_root = Path(__file__).resolve().parents[1]

    report = build_quality_dashboard_report(repository_root / "results")
    suites = {suite.key: suite for suite in report.suites}

    assert (suites["main"].passed_cases, suites["main"].total_cases) == (73, 73)
    assert (suites["challenge"].passed_cases, suites["challenge"].total_cases) == (26, 26)
    assert (suites["legacy_security"].passed_cases, suites["legacy_security"].total_cases) == (12, 12)
    assert (
        suites["enterprise_safety"].passed_cases,
        suites["enterprise_safety"].total_cases,
    ) == (15, 15)
