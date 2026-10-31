"""Evaluation-suite specifications and strict local JSONL validation."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from puzzel.languages import MULTILINGUAL_LANGUAGE_CODES

SUITE_NAMES = (
    "multilingual_conversation",
    "general_knowledge",
    "programming",
    "mathematics",
    "reasoning",
    "translation",
    "creative_writing",
)
_ALLOWED_PARTITIONS = {"dev", "test"}
_ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant"}
_ALLOWED_METRIC_DIRECTIONS = {"higher", "lower"}


class EvaluationDatasetError(ValueError):
    """Raised for invalid evaluation records or dataset provenance."""


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """A loaded capability-specific evaluation schema and metric policy."""

    name: str
    task: str
    description: str
    required_fields: tuple[str, ...]
    reference_field: str
    answer_field: str
    max_new_tokens: int
    requires_code_sandbox: bool
    rubric_categories: tuple[str, ...]
    metric_directions: dict[str, str]
    source_path: Path


def load_suite_spec(
    name: str,
    *,
    directory: str | Path | None = None,
) -> SuiteSpec:
    """Load a reviewed suite definition from the repository's TOML specs."""
    if name not in SUITE_NAMES:
        raise ValueError(f"unknown evaluation suite {name!r}; choose one of {', '.join(SUITE_NAMES)}")
    spec_dir = (
        Path(directory)
        if directory is not None
        else Path(__file__).resolve().parents[2] / "configs" / "evaluation"
    )
    source = spec_dir / f"{name}.toml"
    try:
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read evaluation suite spec {source}: {exc}") from exc
    if payload.get("format") != "puzzel-evaluation-suite-v1":
        raise ValueError(f"unsupported evaluation suite spec format in {source}")
    if payload.get("name") != name:
        raise ValueError(f"suite spec name does not match filename: {source}")

    required_fields = payload.get("required_fields")
    directions = payload.get("metrics")
    rubric_categories = payload.get("rubric_categories", [])
    max_new_tokens = payload.get("max_new_tokens")
    if (
        not isinstance(required_fields, list)
        or not required_fields
        or any(not isinstance(field, str) or not field for field in required_fields)
    ):
        raise ValueError(f"required_fields is malformed in {source}")
    if (
        not isinstance(directions, dict)
        or not directions
        or any(direction not in _ALLOWED_METRIC_DIRECTIONS for direction in directions.values())
    ):
        raise ValueError(f"metric directions are malformed in {source}")
    if not isinstance(rubric_categories, list) or any(
        not isinstance(category, str) or not category for category in rubric_categories
    ):
        raise ValueError(f"rubric_categories is malformed in {source}")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
    ):
        raise ValueError(f"max_new_tokens must be positive in {source}")
    if not isinstance(payload.get("requires_code_sandbox"), bool):
        raise ValueError(f"requires_code_sandbox must be a boolean in {source}")

    return SuiteSpec(
        name=name,
        task=str(payload.get("task", "")),
        description=str(payload.get("description", "")),
        required_fields=tuple(required_fields),
        reference_field=str(payload.get("reference_field", "")),
        answer_field=str(payload.get("answer_field", "")),
        max_new_tokens=max_new_tokens,
        requires_code_sandbox=payload["requires_code_sandbox"],
        rubric_categories=tuple(rubric_categories),
        metric_directions=dict(directions),
        source_path=source,
    )


def validate_record(record: Any, spec: SuiteSpec, *, line_number: int) -> dict[str, Any]:
    """Validate one record and normalize the spelling alias for ``license``."""
    prefix = f"{spec.name} JSONL line {line_number}"
    if not isinstance(record, dict):
        raise EvaluationDatasetError(f"{prefix}: record must be a JSON object")
    normalized = dict(record)
    if "license" not in normalized and isinstance(normalized.get("licence"), str):
        normalized["license"] = normalized["licence"]
    missing = [field for field in spec.required_fields if field not in normalized]
    if missing:
        raise EvaluationDatasetError(f"{prefix}: missing required fields {missing}")

    for field in ("id", "source", "license"):
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationDatasetError(f"{prefix}: {field} must be a non-empty string")
    if normalized.get("language") not in MULTILINGUAL_LANGUAGE_CODES:
        raise EvaluationDatasetError(
            f"{prefix}: language must be one of {', '.join(MULTILINGUAL_LANGUAGE_CODES)}"
        )
    if normalized.get("partition") not in _ALLOWED_PARTITIONS:
        raise EvaluationDatasetError(
            f"{prefix}: partition must be 'dev' or 'test'; training data is not accepted"
        )

    for field in (
        "prompt",
        "reference",
        "reference_code",
        "reference_translation",
        "source_text",
    ):
        if field in normalized and not isinstance(normalized[field], str):
            raise EvaluationDatasetError(f"{prefix}: {field} must be a string")
    for field in ("accepted_answers", "expected_answer"):
        if field not in normalized:
            continue
        value = normalized[field]
        if isinstance(value, str):
            valid = bool(value.strip())
        elif isinstance(value, list):
            valid = bool(value) and all(isinstance(answer, str) and answer.strip() for answer in value)
        else:
            valid = False
        if not valid:
            raise EvaluationDatasetError(
                f"{prefix}: {field} must be a non-empty string or list of strings"
            )

    if "messages" in normalized:
        messages = normalized["messages"]
        if not isinstance(messages, list) or not messages:
            raise EvaluationDatasetError(f"{prefix}: messages must be a non-empty array")
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or message.get("role") not in _ALLOWED_MESSAGE_ROLES
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise EvaluationDatasetError(
                    f"{prefix}: message {index} needs a supported role and non-empty content"
                )
        if not any(message["role"] == "user" for message in messages):
            raise EvaluationDatasetError(f"{prefix}: conversation needs at least one user message")
        if messages[-1]["role"] != "user":
            raise EvaluationDatasetError(f"{prefix}: final conversation message must be from user")

    if spec.name == "translation":
        source_language = normalized.get("source_language")
        target_language = normalized.get("target_language")
        if source_language not in MULTILINGUAL_LANGUAGE_CODES:
            raise EvaluationDatasetError(f"{prefix}: unsupported source_language")
        if target_language not in MULTILINGUAL_LANGUAGE_CODES:
            raise EvaluationDatasetError(f"{prefix}: unsupported target_language")
        if source_language == target_language:
            raise EvaluationDatasetError(f"{prefix}: source and target languages must differ")
        if normalized["language"] != target_language:
            raise EvaluationDatasetError(f"{prefix}: language must equal target_language")
        if not normalized["source_text"].strip() or not normalized["reference_translation"].strip():
            raise EvaluationDatasetError(f"{prefix}: translation source and reference must be non-empty")

    if spec.name == "programming":
        test_cases = normalized.get("test_cases")
        if not isinstance(test_cases, list) or not test_cases:
            raise EvaluationDatasetError(f"{prefix}: test_cases must be a non-empty array")
        for index, test_case in enumerate(test_cases):
            if not isinstance(test_case, dict) or not {
                "input",
                "expected_output",
            }.issubset(test_case):
                raise EvaluationDatasetError(
                    f"{prefix}: test case {index} needs input and expected_output fields"
                )

    if spec.name == "multilingual_conversation":
        if not normalized["reference"].strip():
            raise EvaluationDatasetError(f"{prefix}: reference must be non-empty")

    if spec.name == "general_knowledge" and "accepted_answers" not in normalized:
        raise EvaluationDatasetError(f"{prefix}: accepted_answers are required")
    if spec.name in {"mathematics", "reasoning"} and "expected_answer" not in normalized:
        raise EvaluationDatasetError(f"{prefix}: expected_answer is required")
    if spec.name == "programming" and not normalized["reference_code"].strip():
        raise EvaluationDatasetError(f"{prefix}: reference_code must be non-empty")
    if spec.name == "creative_writing":
        if not normalized["prompt"].strip():
            raise EvaluationDatasetError(f"{prefix}: prompt must be non-empty")
    elif "prompt" in normalized and not normalized["prompt"].strip():
        raise EvaluationDatasetError(f"{prefix}: prompt must be non-empty")
    return normalized


def iter_evaluation_records(
    path: str | Path,
    spec: SuiteSpec,
) -> Iterator[dict[str, Any]]:
    """Stream-validate UTF-8 JSONL examples and reject duplicate example IDs."""
    source = Path(path)
    seen_ids: set[str] = set()
    try:
        with source.open("r", encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationDatasetError(
                        f"{source}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                record = validate_record(raw_record, spec, line_number=line_number)
                if record["id"] in seen_ids:
                    raise EvaluationDatasetError(
                        f"{source}:{line_number}: duplicate example id {record['id']!r}"
                    )
                seen_ids.add(record["id"])
                yield record
    except (OSError, UnicodeError) as exc:
        raise EvaluationDatasetError(f"cannot read evaluation dataset {source}: {exc}") from exc


def validate_ratings_record(
    record: Any,
    *,
    path: str | Path,
    line_number: int,
    categories: tuple[str, ...],
) -> dict[str, Any]:
    """Validate one independent human-rating record (scores are 1 through 5)."""
    prefix = f"ratings JSONL line {line_number} in {path}"
    if not isinstance(record, dict):
        raise EvaluationDatasetError(f"{prefix}: record must be an object")
    example_id = record.get("id")
    rater_id = record.get("rater_id")
    scores = record.get("scores")
    if not isinstance(example_id, str) or not example_id.strip():
        raise EvaluationDatasetError(f"{prefix}: id must be a non-empty string")
    if not isinstance(rater_id, str) or not rater_id.strip():
        raise EvaluationDatasetError(f"{prefix}: rater_id must be a non-empty string")
    if not isinstance(scores, dict) or not scores:
        raise EvaluationDatasetError(f"{prefix}: scores must be a non-empty object")
    if set(scores) - set(categories):
        raise EvaluationDatasetError(f"{prefix}: scores contain an unknown rubric category")
    for category, score in scores.items():
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not 1 <= score <= 5
        ):
            raise EvaluationDatasetError(
                f"{prefix}: score for {category!r} must be between 1 and 5"
            )
    return {"id": example_id, "rater_id": rater_id, "scores": scores}
