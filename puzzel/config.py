"""TOML loading and validated model, tokenizer, and training settings."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from puzzel.languages import MULTILINGUAL_LANGUAGE_CODES
from puzzel.model.config import ModelConfig
from puzzel.tokenizer import VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """Tokenizer selection and learned-vocabulary target."""

    kind: str = "byte"
    vocab_size: int = VOCAB_SIZE
    min_frequency: int = 2

    def __post_init__(self) -> None:
        if self.kind not in {"byte", "bpe"}:
            raise ValueError("tokenizer kind must be 'byte' or 'bpe'")
        if (
            not isinstance(self.vocab_size, int)
            or isinstance(self.vocab_size, bool)
            or self.vocab_size < VOCAB_SIZE
        ):
            raise ValueError(f"tokenizer vocab_size must be an integer >= {VOCAB_SIZE}")
        if self.kind == "byte" and self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"byte tokenizer vocab_size is fixed at {VOCAB_SIZE}")
        if (
            not isinstance(self.min_frequency, int)
            or isinstance(self.min_frequency, bool)
            or self.min_frequency < 1
        ):
            raise ValueError("tokenizer min_frequency must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimizer, schedule, validation, and safety settings for a training run."""

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 1000
    gradient_clip_norm: float = 1.0
    seed: int = 1337
    num_workers: int = 0
    device: str = "auto"
    precision: str = "fp32"
    log_every_steps: int = 10
    save_every_steps: int = 100
    scheduler: str = "cosine"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    validation_every_steps: int = 100
    validation_batches: int = 32
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    unstable_loss_threshold: float = 100.0
    loss_spike_multiplier: float = 10.0
    language_sampling_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        positive_ints = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "log_every_steps": self.log_every_steps,
            "save_every_steps": self.save_every_steps,
            "validation_every_steps": self.validation_every_steps,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        non_negative_ints = {
            "num_workers": self.num_workers,
            "warmup_steps": self.warmup_steps,
            "validation_batches": self.validation_batches,
            "early_stopping_patience": self.early_stopping_patience,
        }
        for name, value in non_negative_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.warmup_steps >= self.max_steps and self.warmup_steps != 0:
            raise ValueError("warmup_steps must be less than max_steps")
        if self.scheduler not in {"cosine", "constant"}:
            raise ValueError("scheduler must be 'cosine' or 'constant'")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if self.precision != "fp32":
            raise ValueError("only fp32 precision is currently supported")
        weights = self.language_sampling_weights
        if not isinstance(weights, dict):
            raise ValueError("language_sampling_weights must be a TOML table")
        if weights:
            if set(weights) != set(MULTILINGUAL_LANGUAGE_CODES):
                raise ValueError(
                    "language_sampling_weights must specify exactly: "
                    + ", ".join(MULTILINGUAL_LANGUAGE_CODES)
                )
            for language, weight in weights.items():
                if (
                    not isinstance(weight, (int, float))
                    or isinstance(weight, bool)
                    or not math.isfinite(weight)
                    or weight <= 0
                ):
                    raise ValueError(
                        f"language sampling weight for {language!r} must be positive and finite"
                    )
            if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("language_sampling_weights must sum to 1.0")

        finite_values = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "min_learning_rate": self.min_learning_rate,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "unstable_loss_threshold": self.unstable_loss_threshold,
            "loss_spike_multiplier": self.loss_spike_multiplier,
        }
        for name, value in finite_values.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must be non-negative")
        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be between zero and learning_rate")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if self.unstable_loss_threshold <= 0:
            raise ValueError("unstable_loss_threshold must be positive")
        if self.loss_spike_multiplier <= 1:
            raise ValueError("loss_spike_multiplier must be greater than one")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Preparation limits and deterministic document-level split ratios."""

    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 1337
    max_document_characters: int = 1_000_000
    jsonl_text_field: str = "text"
    require_document_metadata: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation and test fractions must sum to less than 1")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool):
            raise ValueError("split_seed must be an integer")
        if (
            not isinstance(self.max_document_characters, int)
            or isinstance(self.max_document_characters, bool)
            or self.max_document_characters <= 0
        ):
            raise ValueError("max_document_characters must be a positive integer")
        if not isinstance(self.jsonl_text_field, str) or not self.jsonl_text_field:
            raise ValueError("jsonl_text_field must be a non-empty string")
        if not isinstance(self.require_document_metadata, bool):
            raise ValueError("require_document_metadata must be a boolean")


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    section = document.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return section


def load_config(path: str | Path) -> RunConfig:
    """Read TOML and validate model dimensions and tokenizer vocabulary target."""
    source = Path(path)
    document = tomllib.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("configuration root must be a TOML table")
    unknown_sections = set(document) - {"model", "training", "tokenizer", "dataset"}
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(f"unknown top-level configuration section(s): {names}")

    try:
        model = ModelConfig(**_section(document, "model"))
        training = TrainingConfig(**_section(document, "training"))
        tokenizer = TokenizerConfig(**_section(document, "tokenizer"))
        dataset = DatasetConfig(**_section(document, "dataset"))
    except TypeError as exc:
        raise ValueError(f"invalid configuration field: {exc}") from exc

    if training.language_sampling_weights and not dataset.require_document_metadata:
        raise ValueError(
            "multilingual language sampling requires dataset.require_document_metadata = true"
        )

    if model.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "model.vocab_size must equal tokenizer.vocab_size (the BPE target "
            "size or fixed byte vocabulary size)"
        )
    return RunConfig(
        model=model,
        training=training,
        tokenizer=tokenizer,
        dataset=dataset,
    )
