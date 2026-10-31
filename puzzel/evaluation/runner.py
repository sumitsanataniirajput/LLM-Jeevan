"""Deterministic local evaluation against Puzzel-owned checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from puzzel.checkpoint import LoadedCheckpoint, load_checkpoint
from puzzel.evaluation.metrics import chrf_score, exact_match, finite_perplexity
from puzzel.evaluation.sandbox import CodeSandboxCommand
from puzzel.evaluation.specs import (
    EvaluationDatasetError,
    SuiteSpec,
    iter_evaluation_records,
    load_suite_spec,
    validate_ratings_record,
)
from puzzel.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    TextTokenizer,
)
from puzzel.training.loop import resolve_device


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _format_conversation(messages: list[dict[str, str]]) -> str:
    labels = {"system": "System", "user": "User", "assistant": "Assistant"}
    transcript = "\n".join(
        f"{labels[message['role']]}: {message['content']}" for message in messages
    )
    return f"{transcript}\nAssistant:"


def build_prompt(record: dict[str, Any], suite: SuiteSpec) -> str:
    """Build a stable, documented prompt format without modifying the model."""
    if suite.name == "multilingual_conversation":
        return _format_conversation(record["messages"])
    if suite.name == "translation":
        return (
            f"Translate from {record['source_language']} to {record['target_language']}.\n"
            f"{record['source_text']}\nTranslation:"
        )
    return record["prompt"]


@torch.inference_mode()
def greedy_generate(
    model: torch.nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate greedily with bounded context; no sampling or external service."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated: list[int] = []
    model.eval()
    forbidden_ids = (BOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID)
    for _ in range(max_new_tokens):
        active_ids = (prompt_ids + generated)[-model.context_length :]
        input_ids = torch.tensor(active_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(input_ids)[0, -1].clone()
        for token_id in forbidden_ids:
            if token_id < logits.numel():
                logits[token_id] = -torch.inf
        next_token = int(torch.argmax(logits).item())
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)
    return tokenizer.decode(generated)


@torch.inference_mode()
def conditional_token_loss(
    model: torch.nn.Module,
    tokenizer: TextTokenizer,
    prompt: str,
    reference: str,
    *,
    device: torch.device,
) -> tuple[float, int]:
    """Score reference continuation NLL in bounded overlapping context windows."""
    prefix_ids = tokenizer.encode(prompt, add_bos=True)
    continuation_ids = tokenizer.encode(reference, add_eos=True)
    if not continuation_ids:
        return 0.0, 0
    token_ids = prefix_ids + continuation_ids
    context_length = int(model.context_length)
    stride = max(1, context_length // 2)
    history = max(0, context_length - stride)
    target_start = len(prefix_ids)
    next_target = target_start
    nll_sum = 0.0
    token_count = 0
    model.eval()

    while next_target < len(token_ids):
        target_end = min(len(token_ids), next_target + stride)
        context_start = max(0, next_target - history)
        input_tokens = token_ids[context_start : target_end - 1]
        target_tokens = token_ids[next_target:target_end]
        if not input_tokens:
            raise ValueError("evaluation prompt/reference does not form a next-token pair")
        input_tensor = torch.tensor(
            input_tokens, dtype=torch.long, device=device
        ).unsqueeze(0)
        logits = model(input_tensor)[0]
        prediction_positions = torch.arange(
            next_target - context_start - 1,
            target_end - context_start - 1,
            dtype=torch.long,
            device=device,
        )
        targets = torch.tensor(target_tokens, dtype=torch.long, device=device)
        selected_logits = logits.index_select(0, prediction_positions)
        batch_nll = F.cross_entropy(selected_logits, targets, reduction="sum")
        if not torch.isfinite(batch_nll).item():
            raise ValueError("conditional evaluation loss became non-finite")
        nll_sum += float(batch_nll.item())
        token_count += len(target_tokens)
        next_target = target_end
    return nll_sum, token_count


def _references(record: dict[str, Any], suite: SuiteSpec) -> list[str]:
    field = suite.answer_field or suite.reference_field
    if not field:
        return []
    value = record.get(field)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _empty_accumulator() -> dict[str, Any]:
    return {
        "examples": 0,
        "reference_examples": 0,
        "scored_tokens": 0,
        "nll_sum": 0.0,
        "exact_examples": 0,
        "exact_matches": 0,
        "chrf_examples": 0,
        "chrf_sum": 0.0,
        "nonempty_examples": 0,
        "code_sandbox_attempts": 0,
        "code_sandbox_errors": 0,
        "code_execution_examples": 0,
        "code_execution_passes": 0,
        "unit_test_passes": 0,
        "unit_test_total": 0,
        "human_score_sum": 0.0,
        "human_score_count": 0,
        "partition_counts": defaultdict(int),
    }


def _finalize_accumulator(values: dict[str, Any]) -> dict[str, Any]:
    token_count = values["scored_tokens"]
    mean_nll = values["nll_sum"] / token_count if token_count else None
    exact_count = values["exact_examples"]
    chrf_count = values["chrf_examples"]
    nonempty_count = values["examples"]
    code_count = values["code_execution_examples"]
    unit_total = values["unit_test_total"]
    human_count = values["human_score_count"]
    return {
        "examples": values["examples"],
        "reference_examples": values["reference_examples"],
        "scored_tokens": token_count,
        "conditional_nll": mean_nll,
        "perplexity": finite_perplexity(mean_nll),
        "exact_accuracy": (
            values["exact_matches"] / exact_count if exact_count else None
        ),
        "chrf": values["chrf_sum"] / chrf_count if chrf_count else None,
        "nonempty_rate": (
            values["nonempty_examples"] / nonempty_count if nonempty_count else None
        ),
        "code_sandbox_attempts": values["code_sandbox_attempts"],
        "code_sandbox_errors": values["code_sandbox_errors"],
        "code_execution_pass_rate": (
            values["code_execution_passes"] / code_count if code_count else None
        ),
        "unit_test_pass_rate": (
            values["unit_test_passes"] / unit_total if unit_total else None
        ),
        "human_rubric_mean": (
            values["human_score_sum"] / human_count if human_count else None
        ),
        "human_rating_count": human_count,
        "partition_counts": dict(values["partition_counts"]),
    }


def _load_human_ratings(
    path: str | Path | None,
    suite: SuiteSpec,
) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    if not suite.rubric_categories:
        raise EvaluationDatasetError(
            f"suite {suite.name!r} has no human-rating rubric categories"
        )
    source = Path(path)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_raters: set[tuple[str, str]] = set()
    try:
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationDatasetError(
                        f"{source}:{line_number}: invalid ratings JSON: {exc.msg}"
                    ) from exc
                rating = validate_ratings_record(
                    raw,
                    path=source,
                    line_number=line_number,
                    categories=suite.rubric_categories,
                )
                key = (rating["id"], rating["rater_id"])
                if key in seen_raters:
                    raise EvaluationDatasetError(
                        f"{source}:{line_number}: duplicate rater/example pair {key!r}"
                    )
                seen_raters.add(key)
                result[rating["id"]].append(rating)
    except (OSError, UnicodeError) as exc:
        raise EvaluationDatasetError(f"cannot read human ratings {source}: {exc}") from exc
    return dict(result)


def _rating_summary(
    ratings: dict[str, list[dict[str, Any]]],
    valid_ids: set[str],
    suite: SuiteSpec,
) -> dict[str, Any]:
    unknown_ids = set(ratings) - valid_ids
    if unknown_ids:
        examples = sorted(unknown_ids)[:5]
        raise EvaluationDatasetError(f"human ratings refer to unknown example IDs: {examples}")
    category_values: dict[str, list[float]] = {
        category: [] for category in suite.rubric_categories
    }
    per_example: dict[str, list[float]] = defaultdict(list)
    for example_id, entries in ratings.items():
        for entry in entries:
            for category, value in entry["scores"].items():
                numeric = float(value)
                category_values[category].append(numeric)
                per_example[example_id].append(numeric)
    categories = {
        name: {
            "mean": sum(values) / len(values) if values else None,
            "rating_count": len(values),
        }
        for name, values in category_values.items()
    }
    all_values = [value for values in category_values.values() for value in values]
    return {
        "rated_examples": len(per_example),
        "total_examples": len(valid_ids),
        "coverage": len(per_example) / len(valid_ids) if valid_ids else 0.0,
        "mean": sum(all_values) / len(all_values) if all_values else None,
        "categories": categories,
        "per_example_mean": {
            example_id: sum(values) / len(values)
            for example_id, values in per_example.items()
        },
    }


def _read_training_metrics(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    try:
        with source.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise EvaluationDatasetError(f"cannot read training metrics CSV {source}: {exc}") from exc
    if not rows:
        return {"source_sha256": _sha256_file(source), "train_loss": None, "validation_loss": None}

    def latest_value(field: str, *, allow_blank: bool) -> float | None:
        for row in reversed(rows):
            raw = row.get(field, "")
            if allow_blank and not raw:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise EvaluationDatasetError(
                    f"training metrics column {field!r} contains a non-numeric value"
                ) from None
            if not math.isfinite(value) or value < 0:
                raise EvaluationDatasetError(
                    f"training metrics column {field!r} contains an invalid loss"
                )
            return value
        return None

    train_loss = latest_value("train_loss", allow_blank=False)
    validation_loss = latest_value("validation_loss", allow_blank=True)
    return {
        "source_sha256": _sha256_file(source),
        "train_loss": train_loss,
        "train_perplexity": finite_perplexity(train_loss),
        "validation_loss": validation_loss,
        "validation_perplexity": finite_perplexity(validation_loss),
    }


def _add_human_scores(
    accumulator: dict[str, Any],
    scores: list[float],
) -> None:
    accumulator["human_score_sum"] += sum(scores)
    accumulator["human_score_count"] += len(scores)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    suite_name: str,
    *,
    device_name: str = "cpu",
    max_examples: int | None = None,
    max_new_tokens: int | None = None,
    training_metrics_path: str | Path | None = None,
    human_ratings_path: str | Path | None = None,
    code_sandbox: CodeSandboxCommand | None = None,
) -> dict[str, Any]:
    """Evaluate one existing Puzzel checkpoint on one licensed local JSONL suite."""
    if max_examples is not None and (
        not isinstance(max_examples, int)
        or isinstance(max_examples, bool)
        or max_examples <= 0
    ):
        raise ValueError("max_examples must be a positive integer when provided")
    suite = load_suite_spec(suite_name)
    if code_sandbox is not None and not suite.requires_code_sandbox:
        raise ValueError(f"suite {suite.name!r} does not accept a code sandbox")
    resolved_device = resolve_device(device_name)
    checkpoint_source = Path(checkpoint_path)
    dataset_source = Path(dataset_path)
    loaded: LoadedCheckpoint = load_checkpoint(checkpoint_source, map_location="cpu")
    model = loaded.model.to(resolved_device)
    model.eval()
    tokenizer = loaded.tokenizer
    requested_new_tokens = suite.max_new_tokens if max_new_tokens is None else max_new_tokens
    if (
        not isinstance(requested_new_tokens, int)
        or isinstance(requested_new_tokens, bool)
        or requested_new_tokens <= 0
    ):
        raise ValueError("max_new_tokens must be a positive integer")

    if max_examples is not None and human_ratings_path is not None:
        raise ValueError("human ratings require the complete suite; omit --max-examples")
    ratings = _load_human_ratings(human_ratings_path, suite)
    accumulators: dict[str, dict[str, Any]] = {"overall": _empty_accumulator()}
    valid_ids: set[str] = set()
    per_example: list[dict[str, Any]] = []
    total = 0
    records = iter_evaluation_records(dataset_source, suite)
    if max_examples is not None:
        records = islice(records, max_examples)
    for record in records:
        example_id = record["id"]
        valid_ids.add(example_id)
        language = record["language"]
        accumulator = accumulators["overall"]
        language_accumulator = accumulators.setdefault(language, _empty_accumulator())
        prompt = build_prompt(record, suite)
        generated = greedy_generate(
            model,
            tokenizer,
            prompt,
            device=resolved_device,
            max_new_tokens=requested_new_tokens,
        )
        references = _references(record, suite)
        reference = references[0] if references else None
        nll_sum = 0.0
        scored_tokens = 0
        if reference is not None:
            nll_sum, scored_tokens = conditional_token_loss(
                model,
                tokenizer,
                prompt,
                reference,
                device=resolved_device,
            )

        row: dict[str, Any] = {
            "id": example_id,
            "partition": record["partition"],
            "language": language,
            "scored_tokens": scored_tokens,
            "conditional_nll": nll_sum / scored_tokens if scored_tokens else None,
            "perplexity": finite_perplexity(nll_sum / scored_tokens) if scored_tokens else None,
            "exact_match": None,
            "chrf": None,
            "nonempty": bool(generated.strip()),
            "code_execution_ok": None,
            "unit_test_pass_rate": None,
            "sandbox_status": "not_run" if suite.requires_code_sandbox else None,
            "human_rubric_mean": None,
        }
        accumulator["examples"] += 1
        accumulator["partition_counts"][record["partition"]] += 1
        language_accumulator["examples"] += 1
        language_accumulator["partition_counts"][record["partition"]] += 1
        if generated.strip():
            accumulator["nonempty_examples"] += 1
            language_accumulator["nonempty_examples"] += 1
        if reference is not None:
            accumulator["reference_examples"] += 1
            language_accumulator["reference_examples"] += 1
            accumulator["nll_sum"] += nll_sum
            language_accumulator["nll_sum"] += nll_sum
            accumulator["scored_tokens"] += scored_tokens
            language_accumulator["scored_tokens"] += scored_tokens
            exact = exact_match(generated, references) if "exact_accuracy" in suite.metric_directions else None
            score = (
                max(chrf_score(generated, candidate) for candidate in references)
                if "chrf" in suite.metric_directions
                else None
            )
            row["exact_match"] = exact
            row["chrf"] = score
            if exact is not None:
                accumulator["exact_examples"] += 1
                language_accumulator["exact_examples"] += 1
                if exact:
                    accumulator["exact_matches"] += 1
                    language_accumulator["exact_matches"] += 1
            if score is not None:
                accumulator["chrf_examples"] += 1
                language_accumulator["chrf_examples"] += 1
                accumulator["chrf_sum"] += score
                language_accumulator["chrf_sum"] += score
        if suite.requires_code_sandbox and code_sandbox is not None:
            sandbox = code_sandbox.run(generated, record["test_cases"])
            row["sandbox_status"] = sandbox.status
            for target_accumulator in (accumulator, language_accumulator):
                target_accumulator["code_sandbox_attempts"] += 1
                if sandbox.status not in {"ok", "timeout"}:
                    target_accumulator["code_sandbox_errors"] += 1
            if sandbox.status in {"ok", "timeout"}:
                row["code_execution_ok"] = sandbox.execution_ok
                row["unit_test_pass_rate"] = sandbox.unit_test_pass_rate
                for target_accumulator in (accumulator, language_accumulator):
                    target_accumulator["code_execution_examples"] += 1
                    if sandbox.execution_ok:
                        target_accumulator["code_execution_passes"] += 1
                    target_accumulator["unit_test_passes"] += sandbox.passed_tests
                    target_accumulator["unit_test_total"] += sandbox.total_tests
        per_example.append(row)
        total += 1

    if total == 0:
        raise EvaluationDatasetError("evaluation dataset has no eligible examples")
    rating_report = _rating_summary(ratings, valid_ids, suite)
    rating_report["source_sha256"] = (
        None if human_ratings_path is None else _sha256_file(human_ratings_path)
    )
    for example in per_example:
        example_scores = rating_report["per_example_mean"].get(example["id"])
        if example_scores is not None:
            example["human_rubric_mean"] = example_scores
            for accumulator in (
                accumulators["overall"],
                accumulators[example["language"]],
            ):
                _add_human_scores(accumulator, [example_scores])

    summary = {
        "overall": _finalize_accumulator(accumulators["overall"]),
        "per_language": {
            language: _finalize_accumulator(accumulator)
            for language, accumulator in sorted(accumulators.items())
            if language != "overall"
        },
    }
    checkpoint_report = {
        "file_name": checkpoint_source.name,
        "sha256": _sha256_file(checkpoint_source),
        "step": loaded.step,
        "model_config": {
            "vocab_size": loaded.model_config.vocab_size,
            "context_length": loaded.model_config.context_length,
            "d_model": loaded.model_config.d_model,
            "num_layers": loaded.model_config.num_layers,
            "num_heads": loaded.model_config.num_heads,
        },
    }
    sandbox_report: dict[str, Any] = {
        "enabled": code_sandbox is not None,
        "protocol": "puzzel-code-sandbox-json-v1",
    }
    if code_sandbox is not None:
        launcher = Path(code_sandbox.command[0])
        sandbox_report["launcher"] = launcher.name
        sandbox_report["launcher_sha256"] = (
            _sha256_file(launcher) if launcher.is_file() else None
        )
    return {
        "format": "puzzel-evaluation-report-v1",
        "suite": suite.name,
        "task": suite.task,
        "suite_spec_sha256": _sha256_file(suite.source_path),
        "dataset": {
            "file_name": dataset_source.name,
            "sha256": _sha256_file(dataset_source),
            "examples_evaluated": total,
            "partition_counts": dict(accumulators["overall"]["partition_counts"]),
        },
        "checkpoint": checkpoint_report,
        "run": {
            "device": str(resolved_device),
            "decoding": "greedy",
            "max_new_tokens": requested_new_tokens,
            "max_examples_cap": max_examples,
            "code_sandbox": sandbox_report,
            "training_metrics": _read_training_metrics(training_metrics_path),
        },
        "metric_directions": suite.metric_directions,
        "metrics": summary,
        "human_ratings": {
            key: value for key, value in rating_report.items() if key != "per_example_mean"
        },
        "examples": per_example,
        "limitations": [
            "Automatic metrics are diagnostics and are not capability claims.",
            "Exact-match metrics do not infer semantic or symbolic equivalence.",
            "Code execution is not run unless an explicitly configured external sandbox launcher is supplied.",
            "Creative-writing quality requires human rubric ratings; no external or AI judge is used.",
        ],
    }
