"""JSON/CSV writing and regression-aware checkpoint report comparison."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from puzzel.evaluation.specs import EvaluationDatasetError, load_suite_spec

SUMMARY_CSV_FIELDS = (
    "scope",
    "language",
    "examples",
    "reference_examples",
    "scored_tokens",
    "conditional_nll",
    "perplexity",
    "exact_accuracy",
    "chrf",
    "nonempty_rate",
    "code_sandbox_attempts",
    "code_sandbox_errors",
    "code_execution_pass_rate",
    "unit_test_pass_rate",
    "human_rubric_mean",
    "human_rating_count",
    "train_loss",
    "train_perplexity",
    "validation_loss",
    "validation_perplexity",
)


def _check_output_paths(
    json_path: str | Path,
    csv_path: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    json_destination = Path(json_path)
    csv_destination = Path(csv_path)
    if json_destination.resolve() == csv_destination.resolve():
        raise ValueError("JSON and CSV reports must use different output paths")
    existing = [path for path in (json_destination, csv_destination) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "report output already exists: " + ", ".join(str(path) for path in existing)
        )
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    return json_destination, csv_destination


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict) or not isinstance(metrics.get("overall"), dict):
        raise ValueError("evaluation report has no overall metrics")
    rows: list[dict[str, Any]] = []
    scopes = [("overall", "", metrics["overall"])]
    per_language = metrics.get("per_language", {})
    if not isinstance(per_language, dict):
        raise ValueError("evaluation report per_language field is malformed")
    scopes.extend(
        ("language", language, values)
        for language, values in sorted(per_language.items())
        if isinstance(values, dict)
    )
    for scope, language, values in scopes:
        row = {field: values.get(field) for field in SUMMARY_CSV_FIELDS}
        row["scope"] = scope
        row["language"] = language
        if scope == "overall":
            run_metadata = report.get("run")
            training_metrics = (
                run_metadata.get("training_metrics") or {}
                if isinstance(run_metadata, dict)
                else {}
            )
            for field in (
                "train_loss",
                "train_perplexity",
                "validation_loss",
                "validation_perplexity",
            ):
                row[field] = training_metrics.get(field)
        rows.append(row)
    return rows


def write_evaluation_report(
    report: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a privacy-preserving JSON report and a compact scope CSV."""
    if report.get("format") != "puzzel-evaluation-report-v1":
        raise ValueError("unsupported evaluation report format")
    json_destination, csv_destination = _check_output_paths(
        json_path, csv_path, overwrite=overwrite
    )
    json_destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_summary_rows(report))
    return json_destination, csv_destination


def _load_report(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationDatasetError(f"cannot read evaluation report {source}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != "puzzel-evaluation-report-v1":
        raise EvaluationDatasetError(f"unsupported evaluation report: {source}")
    return value


def _index_reports(paths: Iterable[str | Path], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for path in paths:
        report = _load_report(path)
        suite = report.get("suite")
        if not isinstance(suite, str) or not suite:
            raise EvaluationDatasetError(f"{label} report has no suite name")
        if suite in indexed:
            raise EvaluationDatasetError(f"{label} reports contain duplicate suite {suite!r}")
        indexed[suite] = report
    if not indexed:
        raise EvaluationDatasetError(f"at least one {label} report is required")
    return indexed


def _compare_metric(
    baseline_value: Any,
    candidate_value: Any,
    direction: str,
    max_regression: float,
) -> dict[str, Any] | None:
    if (
        not isinstance(baseline_value, (int, float))
        or isinstance(baseline_value, bool)
        or not math.isfinite(baseline_value)
        or not isinstance(candidate_value, (int, float))
        or isinstance(candidate_value, bool)
        or not math.isfinite(candidate_value)
    ):
        return None
    delta = float(candidate_value) - float(baseline_value)
    regression_amount = -delta if direction == "higher" else delta
    return {
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "direction": direction,
        "regression_amount": regression_amount,
        "regressed": regression_amount > max_regression,
    }


def _compare_scope(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    directions: dict[str, str],
    max_regression: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_name, direction in directions.items():
        comparison = _compare_metric(
            baseline.get(metric_name),
            candidate.get(metric_name),
            direction,
            max_regression,
        )
        if comparison is not None:
            result[metric_name] = comparison
    return result


def compare_evaluation_reports(
    baseline_paths: Iterable[str | Path],
    candidate_paths: Iterable[str | Path],
    *,
    max_regression: float = 0.02,
) -> dict[str, Any]:
    """Compare same-suite, same-data reports and flag metric regressions."""
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max_regression must be finite and non-negative")
    baseline_reports = _index_reports(baseline_paths, "baseline")
    candidate_reports = _index_reports(candidate_paths, "candidate")
    if set(baseline_reports) != set(candidate_reports):
        raise EvaluationDatasetError(
            "baseline and candidate must contain the same evaluation suites"
        )

    suite_results: list[dict[str, Any]] = []
    for suite_name in sorted(baseline_reports):
        baseline = baseline_reports[suite_name]
        candidate = candidate_reports[suite_name]
        baseline_dataset = baseline.get("dataset")
        candidate_dataset = candidate.get("dataset")
        if not isinstance(baseline_dataset, dict) or not isinstance(candidate_dataset, dict):
            raise EvaluationDatasetError(f"{suite_name}: dataset report metadata is malformed")
        baseline_data_hash = baseline_dataset.get("sha256")
        candidate_data_hash = candidate_dataset.get("sha256")
        if not isinstance(baseline_data_hash, str) or not baseline_data_hash:
            raise EvaluationDatasetError(f"{suite_name}: baseline dataset hash is missing")
        if not isinstance(candidate_data_hash, str) or not candidate_data_hash:
            raise EvaluationDatasetError(f"{suite_name}: candidate dataset hash is missing")
        if baseline_data_hash != candidate_data_hash:
            raise EvaluationDatasetError(
                f"{suite_name}: baseline and candidate dataset do not match"
            )
        if baseline.get("suite_spec_sha256") != candidate.get("suite_spec_sha256"):
            raise EvaluationDatasetError(
                f"{suite_name}: baseline and candidate suite_spec_sha256 do not match"
            )
        if baseline_dataset.get("examples_evaluated") != candidate_dataset.get(
            "examples_evaluated"
        ):
            raise EvaluationDatasetError(
                f"{suite_name}: baseline and candidate evaluated different example counts"
            )
        baseline_run = baseline.get("run")
        candidate_run = candidate.get("run")
        if not isinstance(baseline_run, dict) or not isinstance(candidate_run, dict):
            raise EvaluationDatasetError(f"{suite_name}: run metadata is malformed")
        for setting in ("decoding", "max_new_tokens", "max_examples_cap"):
            if baseline_run.get(setting) != candidate_run.get(setting):
                raise EvaluationDatasetError(
                    f"{suite_name}: baseline and candidate {setting} differ"
                )
        baseline_rating_report = baseline.get("human_ratings")
        candidate_rating_report = candidate.get("human_ratings")
        if not isinstance(baseline_rating_report, dict) or not isinstance(
            candidate_rating_report, dict
        ):
            raise EvaluationDatasetError(f"{suite_name}: human rating metadata is malformed")
        baseline_ratings = baseline_rating_report.get("source_sha256")
        candidate_ratings = candidate_rating_report.get("source_sha256")
        if baseline_ratings != candidate_ratings:
            raise EvaluationDatasetError(
                f"{suite_name}: baseline and candidate human ratings do not match"
            )
        spec = load_suite_spec(suite_name)
        if spec.requires_code_sandbox:
            if baseline_run.get("code_sandbox") != candidate_run.get("code_sandbox"):
                raise EvaluationDatasetError(
                    f"{suite_name}: baseline and candidate code sandbox settings differ"
                )
        baseline_metrics = baseline.get("metrics")
        candidate_metrics = candidate.get("metrics")
        if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
            raise EvaluationDatasetError(f"{suite_name}: metrics are malformed")
        baseline_overall = baseline_metrics.get("overall")
        candidate_overall = candidate_metrics.get("overall")
        baseline_languages = baseline_metrics.get("per_language")
        candidate_languages = candidate_metrics.get("per_language")
        if (
            not isinstance(baseline_overall, dict)
            or not isinstance(candidate_overall, dict)
            or not isinstance(baseline_languages, dict)
            or not isinstance(candidate_languages, dict)
        ):
            raise EvaluationDatasetError(f"{suite_name}: metric scopes are malformed")
        if set(baseline_languages) != set(candidate_languages):
            raise EvaluationDatasetError(
                f"{suite_name}: baseline and candidate language coverage differs"
            )
        overall = _compare_scope(
            baseline_overall,
            candidate_overall,
            spec.metric_directions,
            max_regression,
        )
        languages: dict[str, Any] = {}
        for language in sorted(baseline_languages):
            languages[language] = _compare_scope(
                baseline_languages[language],
                candidate_languages[language],
                spec.metric_directions,
                max_regression,
            )
        regressed = any(value["regressed"] for value in overall.values()) or any(
            value["regressed"]
            for language_metrics in languages.values()
            for value in language_metrics.values()
        )
        baseline_checkpoint = baseline.get("checkpoint")
        candidate_checkpoint = candidate.get("checkpoint")
        if not isinstance(baseline_checkpoint, dict) or not isinstance(
            candidate_checkpoint, dict
        ):
            raise EvaluationDatasetError(f"{suite_name}: checkpoint metadata is malformed")
        suite_results.append(
            {
                "suite": suite_name,
                "dataset_sha256": baseline_data_hash,
                "baseline_checkpoint_sha256": baseline_checkpoint.get("sha256"),
                "candidate_checkpoint_sha256": candidate_checkpoint.get("sha256"),
                "overall": overall,
                "per_language": languages,
                "regression_detected": regressed,
                "unscored_metrics": [
                    metric
                    for metric in spec.metric_directions
                    if metric not in overall
                ],
            }
        )
    return {
        "format": "puzzel-evaluation-comparison-v1",
        "regression_threshold": max_regression,
        "regression_detected": any(result["regression_detected"] for result in suite_results),
        "suites": suite_results,
        "interpretation": "Regression flags are thresholded metric changes, not statistical significance or a capability claim.",
    }


def write_comparison_report(
    comparison: dict[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if comparison.get("format") != "puzzel-evaluation-comparison-v1":
        raise ValueError("unsupported comparison report format")
    json_destination, csv_destination = _check_output_paths(
        json_path, csv_path, overwrite=overwrite
    )
    json_destination.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_destination.open("w", newline="", encoding="utf-8") as target:
        fields = (
            "suite",
            "scope",
            "language",
            "metric",
            "baseline",
            "candidate",
            "delta",
            "direction",
            "regression_amount",
            "regressed",
        )
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for suite in comparison.get("suites", []):
            for metric, values in suite["overall"].items():
                writer.writerow(
                    {"suite": suite["suite"], "scope": "overall", "language": "", "metric": metric, **values}
                )
            for language, metrics in suite["per_language"].items():
                for metric, values in metrics.items():
                    writer.writerow(
                        {
                            "suite": suite["suite"],
                            "scope": "language",
                            "language": language,
                            "metric": metric,
                            **values,
                        }
                    )
    return json_destination, csv_destination
