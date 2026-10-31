from __future__ import annotations

import math
import tomllib
from pathlib import Path


def test_instruction_sft_spec_defines_response_only_task_language_sampling() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "instruction_sft.toml"
    spec = tomllib.loads(path.read_text(encoding="utf-8"))
    tasks = spec["record"]["allowed_tasks"]
    languages = spec["record"]["allowed_languages"]
    task_weights = spec["sampling"]["default_task_weights"]
    language_weights = spec["sampling"]["default_language_weights"]

    assert spec["format"] == "puzzel-instruction-response-jsonl-v1"
    assert spec["record"]["required_fields"] == [
        "id",
        "partition",
        "task",
        "language",
        "instruction",
        "response",
        "source",
        "license",
    ]
    assert set(task_weights) == set(tasks)
    assert set(language_weights) == set(languages)
    assert all(math.isfinite(weight) and weight > 0 for weight in task_weights.values())
    assert all(math.isfinite(weight) and weight > 0 for weight in language_weights.values())
    assert sum(language_weights.values()) == 1.0
    assert spec["objective"]["target"] == "response"
    assert spec["objective"]["mask_instruction_and_context_tokens"]
    assert spec["splitting"]["split_before_tokenization"]
    assert spec["sampling"]["policy"] == "hierarchical_weighted_with_replacement"
    assert not spec["sampling"]["require_every_task_in_every_language"]
