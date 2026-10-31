from pathlib import Path

import pytest

from puzzel.config import TrainingConfig, load_config
from puzzel.model.config import ModelConfig
from puzzel.training.hardware import estimate_parameter_count


def test_starter_toml_loads_and_matches_tokenizer() -> None:
    root = Path(__file__).resolve().parents[1]

    run_config = load_config(root / "configs" / "tiny.toml")

    assert run_config.model.vocab_size == 384
    assert run_config.tokenizer.kind == "bpe"
    assert run_config.tokenizer.vocab_size == run_config.model.vocab_size
    assert run_config.dataset.validation_fraction == 0.1
    assert run_config.dataset.max_document_characters == 1_000_000
    assert run_config.model.context_length == 128
    assert run_config.training.max_steps == 1000
    assert run_config.training.scheduler == "cosine"
    assert run_config.training.validation_every_steps == 100


def test_tiny_small_medium_experimental_configs_are_monotonic() -> None:
    root = Path(__file__).resolve().parents[1]
    configs = [
        load_config(root / "configs" / f"{name}.toml")
        for name in ("tiny", "small", "medium")
    ]
    parameter_counts = [estimate_parameter_count(config.model) for config in configs]

    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]
    assert all(
        config.model.vocab_size == config.tokenizer.vocab_size for config in configs
    )
    assert all(config.training.device == "auto" for config in configs)


def test_model_config_validates_attention_width() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=10, num_heads=4)


def test_training_config_validates_scheduler_and_warmup() -> None:
    with pytest.raises(ValueError, match="scheduler"):
        TrainingConfig(scheduler="unknown")
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(max_steps=4, warmup_steps=4)
    with pytest.raises(ValueError, match="min_learning_rate"):
        TrainingConfig(learning_rate=1e-5, min_learning_rate=2e-5)
