"""CLI for local evaluation and checkpoint-regression reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from puzzel.evaluation.reports import (
    compare_evaluation_reports,
    write_comparison_report,
    write_evaluation_report,
)
from puzzel.evaluation.runner import evaluate_checkpoint
from puzzel.evaluation.sandbox import CodeSandboxCommand
from puzzel.evaluation.specs import EvaluationDatasetError, SUITE_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="puzzel-eval",
        description="Evaluate existing Puzzel checkpoints on local, licensed evaluation JSONL.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="score one checkpoint on one capability suite")
    evaluate.add_argument("--checkpoint", type=Path, required=True, help="existing Puzzel checkpoint")
    evaluate.add_argument("--suite", choices=SUITE_NAMES, required=True, help="suite definition name")
    evaluate.add_argument("--dataset", type=Path, required=True, help="local UTF-8 evaluation JSONL")
    evaluate.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    evaluate.add_argument("--max-examples", type=int, help="optional deterministic prefix cap")
    evaluate.add_argument("--max-new-tokens", type=int, help="override suite generation limit")
    evaluate.add_argument("--training-metrics", type=Path, help="existing Puzzel training metrics CSV")
    evaluate.add_argument("--human-ratings", type=Path, help="separate local human-ratings JSONL")
    evaluate.add_argument(
        "--code-sandbox",
        type=Path,
        help="trusted external isolation launcher; required to execute code, never inferred",
    )
    evaluate.add_argument("--sandbox-timeout", type=float, default=10.0)
    evaluate.add_argument("--json-output", type=Path, required=True)
    evaluate.add_argument("--csv-output", type=Path, required=True)
    evaluate.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser(
        "compare", help="compare baseline/candidate suite reports and flag regressions"
    )
    compare.add_argument("--baseline", nargs="+", type=Path, required=True)
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)
    compare.add_argument("--max-regression", type=float, default=0.02)
    compare.add_argument("--json-output", type=Path, required=True)
    compare.add_argument("--csv-output", type=Path, required=True)
    compare.add_argument("--overwrite", action="store_true")
    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "evaluate":
            sandbox = None
            if args.code_sandbox is not None:
                sandbox = CodeSandboxCommand(
                    args.code_sandbox,
                    timeout_seconds=args.sandbox_timeout,
                )
            report = evaluate_checkpoint(
                args.checkpoint,
                args.dataset,
                args.suite,
                device_name=args.device,
                max_examples=args.max_examples,
                max_new_tokens=args.max_new_tokens,
                training_metrics_path=args.training_metrics,
                human_ratings_path=args.human_ratings,
                code_sandbox=sandbox,
            )
            json_path, csv_path = write_evaluation_report(
                report,
                args.json_output,
                args.csv_output,
                overwrite=args.overwrite,
            )
            _print_json(
                {
                    "status": "ok",
                    "suite": report["suite"],
                    "examples_evaluated": report["dataset"]["examples_evaluated"],
                    "json_report": str(json_path),
                    "csv_report": str(csv_path),
                    "metrics": report["metrics"],
                }
            )
        else:
            comparison = compare_evaluation_reports(
                args.baseline,
                args.candidate,
                max_regression=args.max_regression,
            )
            json_path, csv_path = write_comparison_report(
                comparison,
                args.json_output,
                args.csv_output,
                overwrite=args.overwrite,
            )
            _print_json(
                {
                    "status": "ok",
                    "regression_detected": comparison["regression_detected"],
                    "suites_compared": len(comparison["suites"]),
                    "json_report": str(json_path),
                    "csv_report": str(csv_path),
                }
            )
    except (EvaluationDatasetError, OSError, ValueError, RuntimeError) as exc:
        _print_json({"status": "error", "error": str(exc)})
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
