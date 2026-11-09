"""Normalize saved evaluation artifacts for the Streamlit quality dashboard.

This module is deliberately read-only. It does not run evaluations, change
quality gates, or infer historical failures that are absent from saved result
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.evaluation.evaluator import answerability_check


CASE_COLUMNS = [
    "Test ID",
    "Evaluation source",
    "Expected behavior",
    "Actual behavior",
    "Status",
    "Failure reason",
]

ENTERPRISE_CASE_COLUMNS = [
    "Test ID",
    "Category",
    "Expected behavior",
    "Actual behavior",
    "Status",
    "Failure reason",
]


@dataclass(frozen=True)
class EvaluationSuiteReport:
    """One saved evaluation suite, calculated without combining categories."""

    key: str
    label: str
    source: str
    artifact_status: str
    total_cases: int | None
    passed_cases: int | None
    failed_cases: int | None
    pass_rate: float | None
    cases: pd.DataFrame


@dataclass(frozen=True)
class QualityCategoryReport:
    """One dashboard category and its available case-level evidence."""

    key: str
    label: str
    definition: str
    measurement_level: str
    measurement_note: str
    cases: pd.DataFrame

    @property
    def measured_cases(self) -> int | None:
        if self.measurement_level == "Not measured":
            return None
        return int(len(self.cases))

    @property
    def failed_cases(self) -> int | None:
        if self.measurement_level == "Not measured":
            return None
        return int((self.cases["Status"] == "Fail").sum())


@dataclass(frozen=True)
class QualityDashboardReport:
    """All quality categories plus artifact availability warnings."""

    suites: tuple[EvaluationSuiteReport, ...]
    categories: tuple[QualityCategoryReport, ...]
    artifact_warnings: tuple[str, ...]


def _empty_cases() -> pd.DataFrame:
    return pd.DataFrame(columns=CASE_COLUMNS)


def _empty_enterprise_cases() -> pd.DataFrame:
    return pd.DataFrame(columns=ENTERPRISE_CASE_COLUMNS)


def _read_result(path: Path, warnings: list[str]) -> pd.DataFrame:
    if not path.exists():
        warnings.append(f"Saved evaluation artifact is missing: {path.name}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        warnings.append(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def _is_pass(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "pass", "passed"}
    return bool(value)


def _strict_pass(value: object) -> bool | None:
    """Parse persisted pass flags without treating unknown values as failures."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return True
        if float(value) == 0.0:
            return False
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "pass", "passed"}:
            return True
        if normalized in {"0", "false", "fail", "failed"}:
            return False
    return None


def _text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    value_text = str(value).strip()
    return value_text if value_text else default


def _case_row(
    test_id: str,
    source: str,
    expected: object,
    actual: object,
    passed: bool,
    failure_reason: object = "",
) -> dict[str, str]:
    return {
        "Test ID": test_id,
        "Evaluation source": source,
        "Expected behavior": _text(expected, "No expected behavior recorded."),
        "Actual behavior": _text(actual, "No actual behavior recorded."),
        "Status": "Pass" if passed else "Fail",
        "Failure reason": "" if passed else _text(failure_reason, "No failure reason recorded."),
    }


def _frame(rows: Iterable[dict[str, str]]) -> pd.DataFrame:
    rows = list(rows)
    if not rows:
        return _empty_cases()
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def _saved_source(path: Path) -> str:
    if path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name


def _enterprise_case_table(enterprise: pd.DataFrame) -> pd.DataFrame:
    required = {
        "test_id", "category", "expected_behavior", "actual_behavior", "passed"
    }
    if enterprise.empty or not required.issubset(enterprise.columns):
        return _empty_enterprise_cases()
    rows = []
    for _, row in enterprise.iterrows():
        passed = _strict_pass(row["passed"])
        rows.append({
            "Test ID": _text(row["test_id"]),
            "Category": _text(row["category"]),
            "Expected behavior": _text(
                row["expected_behavior"], "No expected behavior recorded."
            ),
            "Actual behavior": _text(
                row["actual_behavior"], "No actual behavior recorded."
            ),
            "Status": "Pass" if passed is True else "Fail" if passed is False else "Unavailable",
            "Failure reason": (
                "" if passed is True else _text(
                    row.get("failure_reason", ""), "No failure reason recorded."
                )
            ),
        })
    return pd.DataFrame(rows, columns=ENTERPRISE_CASE_COLUMNS)


def _suite_report(
    *,
    key: str,
    label: str,
    result_path: Path,
    frame: pd.DataFrame,
    pass_column: str,
    warnings: list[str],
    dataset_paths: tuple[Path, ...] = (),
    enterprise_cases: bool = False,
) -> EvaluationSuiteReport:
    case_table = _enterprise_case_table(frame) if enterprise_cases else pd.DataFrame()
    source = _saved_source(result_path)
    if not result_path.exists():
        return EvaluationSuiteReport(
            key, label, source, "Unavailable", None, None, None, None, case_table
        )
    if frame.empty:
        warning = f"Saved evaluation artifact has no usable rows: {result_path.name}"
        if warning not in warnings:
            warnings.append(warning)
        return EvaluationSuiteReport(
            key, label, source, "Malformed", None, None, None, None, case_table
        )
    if pass_column not in frame.columns:
        warnings.append(
            f"Saved evaluation artifact {result_path.name} is missing required column: {pass_column}"
        )
        return EvaluationSuiteReport(
            key, label, source, "Malformed", None, None, None, None, case_table
        )

    parsed = [_strict_pass(value) for value in frame[pass_column]]
    if any(value is None for value in parsed):
        warnings.append(
            f"Saved evaluation artifact {result_path.name} contains missing or invalid pass values."
        )
        return EvaluationSuiteReport(
            key, label, source, "Malformed", None, None, None, None, case_table
        )

    if enterprise_cases:
        ids = frame.get("test_id", pd.Series(dtype=str)).astype(str)
        if ids.empty or ids.duplicated().any() or not ids.str.match(r"^ENT-SAFE-").all():
            warnings.append(
                f"Saved evaluation artifact {result_path.name} contains invalid or duplicate enterprise safety IDs."
            )
            return EvaluationSuiteReport(
                key, label, source, "Malformed", None, None, None, None, case_table
            )

    total = len(parsed)
    passed = sum(value is True for value in parsed)
    failed = total - passed
    existing_datasets = [path for path in dataset_paths if path.exists()]
    stale = bool(existing_datasets) and result_path.stat().st_mtime < max(
        path.stat().st_mtime for path in existing_datasets
    )
    status = "Stale" if stale else "Available"
    if stale:
        warnings.append(
            f"Saved evaluation artifact may be stale: {result_path.name} is older than its evaluation dataset."
        )
    return EvaluationSuiteReport(
        key=key,
        label=label,
        source=source,
        artifact_status=status,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        pass_rate=passed / total,
        cases=case_table,
    )


def _benchmark_ids(prefix: str, frame: pd.DataFrame) -> list[str]:
    """Create display IDs because legacy result CSVs do not persist case IDs."""
    ids = []
    for fallback_position, index in enumerate(frame.index, start=1):
        row_position = int(index) + 1 if isinstance(index, int) else fallback_position
        ids.append(f"{prefix}-{row_position:03d}")
    return ids


def _enterprise_rows(
    enterprise: pd.DataFrame,
    categories: set[str],
    source_label: str = "Enterprise safety",
) -> list[dict[str, str]]:
    required = {"test_id", "category", "expected_behavior", "actual_behavior", "passed"}
    if enterprise.empty or not required.issubset(enterprise.columns):
        return []
    selected = enterprise[enterprise["category"].astype(str).isin(categories)]
    return [
        _case_row(
            _text(row["test_id"]),
            source_label,
            row["expected_behavior"],
            row["actual_behavior"],
            _is_pass(row["passed"]),
            row.get("failure_reason", ""),
        )
        for _, row in selected.iterrows()
    ]


def _false_acceptance_rows(frame: pd.DataFrame, prefix: str, source: str) -> list[dict[str, str]]:
    required = {"question_type", "answer", "expected_answer", "unanswerable_pass"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    selected = frame[frame["question_type"].astype(str).str.lower() == "unanswerable"]
    ids = _benchmark_ids(prefix, selected)
    rows = []
    for test_id, (_, row) in zip(ids, selected.iterrows()):
        passed = _is_pass(row["unanswerable_pass"])
        rows.append(
            _case_row(
                test_id,
                source,
                row["expected_answer"],
                row["answer"],
                passed,
                "Unsupported question was answered without a recognized safe refusal.",
            )
        )
    return rows


def _false_refusal_rows(frame: pd.DataFrame, prefix: str, source: str) -> list[dict[str, str]]:
    required = {"question_type", "answer", "expected_answer"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    selected = frame[frame["question_type"].astype(str).str.lower() == "normal"]
    ids = _benchmark_ids(prefix, selected)
    rows = []
    for test_id, (_, row) in zip(ids, selected.iterrows()):
        answer = _text(row["answer"])
        refusal_like = not answer or answerability_check(answer, "unanswerable") == 1
        rows.append(
            _case_row(
                test_id,
                source,
                row["expected_answer"],
                answer,
                not refusal_like,
                "Supported question received an empty or refusal-like answer.",
            )
        )
    return rows


def _citation_rows(frame: pd.DataFrame, prefix: str, source: str) -> list[dict[str, str]]:
    required = {"question_type", "answer", "expected_source", "citation_accuracy_pass"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    selected = frame[frame["question_type"].astype(str).str.lower() == "normal"]
    ids = _benchmark_ids(prefix, selected)
    rows = []
    for test_id, (_, row) in zip(ids, selected.iterrows()):
        passed = _is_pass(row["citation_accuracy_pass"])
        rows.append(
            _case_row(
                test_id,
                source,
                f"Cite the expected retrieved source: {_text(row['expected_source'], 'not recorded')}",
                row["answer"],
                passed,
                "The answer citation was missing or did not match a retrieved source.",
            )
        )
    return rows


def _prompt_injection_rows(security: pd.DataFrame) -> list[dict[str, str]]:
    required = {"question_type", "answer", "expected_answer", "unanswerable_pass"}
    if security.empty or not required.issubset(security.columns):
        return []
    selected = security[
        security["question_type"].astype(str).str.lower().isin(
            {"prompt_injection", "jailbreak", "retrieval_poisoning"}
        )
    ]
    ids = _benchmark_ids("SECURITY", selected)
    rows = []
    for test_id, (_, row) in zip(ids, selected.iterrows()):
        passed = _is_pass(row["unanswerable_pass"])
        rows.append(
            _case_row(
                test_id,
                "Legacy security",
                row["expected_answer"],
                row["answer"],
                passed,
                "The instruction attack did not receive a recognized safe refusal.",
            )
        )
    return rows


def build_quality_dashboard_report(
    results_dir: Path | str,
    evaluation_data_dir: Path | str | None = None,
) -> QualityDashboardReport:
    """Build a read-only category report from existing case-level CSV outputs."""
    result_path = Path(results_dir)
    data_path = (
        Path(evaluation_data_dir)
        if evaluation_data_dir is not None
        else result_path.parent / "data" / "evaluation"
    )
    warnings: list[str] = []
    main_path = result_path / "evaluation_results.csv"
    challenge_path = result_path / "challenge_evaluation_results.csv"
    security_path = result_path / "security_evaluation_results.csv"
    enterprise_path = result_path / "enterprise_safety_evaluation_results.csv"
    main = _read_result(main_path, warnings)
    challenge = _read_result(challenge_path, warnings)
    security = _read_result(security_path, warnings)
    enterprise = _read_result(enterprise_path, warnings)

    suites = (
        _suite_report(
            key="main",
            label="Main bilingual benchmark",
            result_path=main_path,
            frame=main,
            pass_column="overall_pass",
            warnings=warnings,
            dataset_paths=(
                data_path / "rag_eval_en.csv",
                data_path / "rag_eval_zh.csv",
                data_path / "security_questions.csv",
            ),
        ),
        _suite_report(
            key="challenge",
            label="Challenge benchmark",
            result_path=challenge_path,
            frame=challenge,
            pass_column="overall_pass",
            warnings=warnings,
            dataset_paths=(data_path / "challenge_questions.csv",),
        ),
        _suite_report(
            key="legacy_security",
            label="Legacy security benchmark",
            result_path=security_path,
            frame=security,
            pass_column="unanswerable_pass",
            warnings=warnings,
            dataset_paths=(data_path / "security_questions.csv",),
        ),
        _suite_report(
            key="enterprise_safety",
            label="Enterprise safety benchmark",
            result_path=enterprise_path,
            frame=enterprise,
            pass_column="passed",
            warnings=warnings,
            dataset_paths=(data_path / "enterprise_safety_cases.json",),
            enterprise_cases=True,
        ),
    )

    false_acceptance = [
        *_false_acceptance_rows(main, "MAIN", "Main benchmark"),
        *_false_acceptance_rows(challenge, "CHALLENGE", "Challenge benchmark"),
        *_enterprise_rows(enterprise, {"unsupported_question", "wrong_collection_access"}),
    ]
    false_refusal = [
        *_false_refusal_rows(main, "MAIN", "Main benchmark"),
        *_false_refusal_rows(challenge, "CHALLENGE", "Challenge benchmark"),
    ]
    citation_validation = [
        *_citation_rows(main, "MAIN", "Main benchmark"),
        *_citation_rows(challenge, "CHALLENGE", "Challenge benchmark"),
        *_enterprise_rows(enterprise, {"citation_fabrication"}),
    ]
    numeric_polarity = _enterprise_rows(
        enterprise,
        {"incorrect_numeric_fact", "contradictory_claim"},
    )
    prompt_injection = [
        *_enterprise_rows(enterprise, {"document_prompt_injection", "instruction_override"}),
        *_prompt_injection_rows(security),
    ]

    categories = (
        QualityCategoryReport(
            key="false_acceptance",
            label="False acceptance",
            definition="An incorrect or unsupported answer is accepted instead of safely refused.",
            measurement_level="Measured",
            measurement_note=(
                "Uses saved unanswerable-case safety flags and enterprise unsupported-question outcomes. "
                "It does not claim to detect every possible unsupported answer."
            ),
            cases=_frame(false_acceptance),
        ),
        QualityCategoryReport(
            key="false_refusal",
            label="False refusal",
            definition="A supported question is incorrectly rejected or receives no answer.",
            measurement_level="Derived diagnostic",
            measurement_note=(
                "The saved benchmarks have no dedicated false-refusal metric. This view applies the existing "
                "safe-refusal phrase check to supported answers; it is informative and is not a quality gate."
            ),
            cases=_frame(false_refusal),
        ),
        QualityCategoryReport(
            key="answer_over_extraction",
            label="Answer over-extraction",
            definition="The answer includes unnecessary or unrelated content beyond what the question asks.",
            measurement_level="Not measured",
            measurement_note=(
                "No saved benchmark artifact contains a dedicated, reliable over-extraction label or metric. "
                "Pytest regression coverage is not converted into a historical count here."
            ),
            cases=_empty_cases(),
        ),
        QualityCategoryReport(
            key="citation_validation",
            label="Citation validation failure",
            definition="A citation is missing, fabricated, or does not match retrieved evidence.",
            measurement_level="Measured",
            measurement_note=(
                "Uses benchmark citation-accuracy flags and the enterprise fabricated-citation safety case."
            ),
            cases=_frame(citation_validation),
        ),
        QualityCategoryReport(
            key="numerical_polarity",
            label="Numerical or polarity contradiction",
            definition="An answer changes an explicit number, permission, prohibition, or negation.",
            measurement_level="Measured",
            measurement_note=(
                "Uses the dedicated enterprise numeric-fact and contradictory-claim cases only."
            ),
            cases=_frame(numeric_polarity),
        ),
        QualityCategoryReport(
            key="prompt_injection",
            label="Prompt-injection compliance",
            definition="The assistant follows malicious instructions from a question or retrieved document.",
            measurement_level="Measured",
            measurement_note=(
                "Uses enterprise document-injection/instruction-override cases and the directly relevant "
                "legacy prompt-injection, jailbreak, and retrieval-poisoning cases."
            ),
            cases=_frame(prompt_injection),
        ),
    )
    return QualityDashboardReport(
        suites=suites,
        categories=categories,
        artifact_warnings=tuple(warnings),
    )
