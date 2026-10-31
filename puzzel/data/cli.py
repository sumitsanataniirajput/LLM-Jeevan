"""Commands for preparing, inspecting, and validating prepared datasets."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from puzzel.config import load_config
from puzzel.data.pipeline import DatasetPreparationError, prepare_dataset
from puzzel.data.prepared import DatasetVerificationError, verify_prepared_dataset
from puzzel.training.hardware import build_hardware_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puzzel.data.cli",
        description="Prepare local TXT/JSONL corpora for the Puzzel model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="clean, split, train BPE, tokenize, and save a dataset"
    )
    prepare.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="one or more local TXT/JSONL files or directories",
    )
    prepare.add_argument("--output", type=Path, required=True, help="new output directory")
    prepare.add_argument("--config", type=Path, required=True, help="model/tokenizer/dataset TOML")

    stats = subparsers.add_parser("stats", help="print a prepared dataset's statistics")
    stats.add_argument("--dataset", type=Path, required=True, help="prepared dataset directory")

    verify = subparsers.add_parser("verify", help="validate prepared data and check leakage")
    verify.add_argument("--dataset", type=Path, required=True, help="prepared dataset directory")

    report = subparsers.add_parser(
        "report", help="write per-language dataset and hardware/memory statistics"
    )
    report.add_argument("--dataset", type=Path, required=True, help="prepared dataset directory")
    report.add_argument(
        "--config",
        type=Path,
        required=True,
        help="the config used to prepare data",
    )
    report.add_argument(
        "--output",
        type=Path,
        default=Path("data/reports/multilingual-v1.json"),
        help="new JSON report path (default: data/reports/multilingual-v1.json)",
    )
    report.add_argument("--overwrite", action="store_true", help="replace an existing report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            run_config = load_config(args.config)
            result = prepare_dataset(args.input, args.output, run_config)
            _print_json(
                {
                    "dataset_dir": result["dataset_dir"],
                    "statistics": result["stats"],
                    "verification": result["verification"],
                }
            )
        elif args.command == "stats":
            stats_path = args.dataset / "stats.json"
            _print_json(_read_json(stats_path))
        elif args.command == "verify":
            _print_json(verify_prepared_dataset(args.dataset))
        else:
            if args.output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"report already exists: {args.output}; choose another path or use --overwrite"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            verification = verify_prepared_dataset(args.dataset)
            run_config = load_config(args.config)
            stats = _read_json(args.dataset / "stats.json")
            manifest = _read_json(args.dataset / "manifest.json")
            if manifest.get("format") == "puzzel-prepared-dataset-v2":
                expected_model = asdict(
                    replace(run_config.model, vocab_size=manifest["vocab_size"])
                )
                if manifest.get("model_config") != expected_model:
                    raise ValueError(
                        "report config model does not match prepared manifest"
                    )
                if manifest.get("tokenizer_config") != asdict(run_config.tokenizer):
                    raise ValueError(
                        "report config tokenizer does not match prepared manifest"
                    )
                if manifest.get("dataset_config") != asdict(run_config.dataset):
                    raise ValueError(
                        "report config dataset settings do not match prepared manifest"
                    )
                if (
                    manifest.get("language_sampling_weights", {})
                    != run_config.training.language_sampling_weights
                ):
                    raise ValueError(
                        "report config sampling weights do not match prepared manifest"
                    )
            actual_model_config = replace(
                run_config.model, vocab_size=verification["vocabulary_size"]
            )
            hardware = build_hardware_report(
                actual_model_config,
                batch_size=run_config.training.batch_size,
                device=run_config.training.device,
                max_steps=run_config.training.max_steps,
                disk_path=args.output.parent,
            )
            report = {
                "format": "puzzel-multilingual-run-report-v1",
                "dataset_dir": str(args.dataset.resolve()),
                "verification": verification,
                "dataset": {
                    "document_count": stats.get("document_count"),
                    "total_characters": stats.get("total_characters"),
                    "total_utf8_bytes": stats.get("total_utf8_bytes"),
                    "total_tokens": stats.get("total_tokens"),
                    "duplicate_count": stats.get("duplicate_count"),
                    "duplicate_count_by_language": stats.get(
                        "duplicate_count_by_language", {}
                    ),
                    "language_sizes": stats.get("language_sizes", {}),
                    "split_sizes": stats.get("split_sizes", {}),
                    "language_sampling_weights": stats.get(
                        "language_sampling_weights", {}
                    ),
                    "vocabulary_statistics": stats.get("vocabulary_statistics", {}),
                    "ingestion": stats.get("ingestion", {}),
                },
                "hardware_and_memory": hardware,
            }
            args.output.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _print_json({"report_path": str(args.output), **report})
    except DatasetPreparationError as exc:
        _print_json({"error": str(exc), "ingestion_report": exc.report})
        raise SystemExit(2) from exc
    except DatasetVerificationError as exc:
        _print_json({"error": str(exc), "checks": exc.errors})
        raise SystemExit(2) from exc
    except (OSError, ValueError) as exc:
        _print_json({"error": str(exc)})
        raise SystemExit(2) from exc


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
