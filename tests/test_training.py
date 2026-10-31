import csv
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from puzzel.checkpoint import load_checkpoint, save_checkpoint
from puzzel.config import TrainingConfig
from puzzel.data.text_dataset import TextSequenceDataset, encode_text
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import BOS_TOKEN_ID, EOS_TOKEN_ID, ByteTokenizer, BPETokenizer
from puzzel.training.loop import (
    BufferedShuffleSampler,
    LearningRateSchedule,
    TrainingDivergedError,
    evaluate_loss,
    seed_everything,
    train_model,
)


def _small_model() -> tuple[ModelConfig, TransformerLM, ByteTokenizer]:
    tokenizer = ByteTokenizer()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=4,
        d_model=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = TransformerLM(model_config)
    return model_config, model, tokenizer


def _repeated_sequence_dataset(sequence_repeats: int = 40) -> TextSequenceDataset:
    pattern = [BOS_TOKEN_ID, 97, 98, 99, EOS_TOKEN_ID]
    return TextSequenceDataset(pattern * sequence_repeats, sequence_length=4)


def test_ten_step_synthetic_training_and_checkpoint_round_trip(tmp_path) -> None:
    training_text = "abcabc abcabc xyxy xyxy from scratch! " * 12
    tokenizer = BPETokenizer.train(
        training_text,
        target_vocab_size=280,
        min_frequency=2,
    )
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    torch.manual_seed(2026)
    model = TransformerLM(model_config)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    dataset = TextSequenceDataset(
        encode_text(training_text, tokenizer), sequence_length=model_config.context_length
    )
    training_config = TrainingConfig(
        batch_size=2,
        max_steps=10,
        log_every_steps=10,
        save_every_steps=5,
        device="cpu",
        scheduler="constant",
    )
    checkpoint_path = tmp_path / "ten-step-smoke.pt"

    result = train_model(
        model,
        dataset,
        model_config,
        training_config,
        checkpoint_path,
        tokenizer=tokenizer,
    )

    assert result.steps == 10
    assert result.device == "cpu"
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert checkpoint_path.is_file()
    assert checkpoint_path.stat().st_size > 0
    assert result.metrics_path.is_file()
    assert any(
        not torch.equal(initial_parameters[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded.step == 10
    assert loaded.model_config == model_config
    assert loaded.tokenizer.to_dict() == tokenizer.to_dict()
    assert loaded.optimizer_state_dict
    assert loaded.scheduler_state_dict["last_step"] == 10
    assert loaded.torch_rng_state is not None
    assert loaded.data_loader_rng_state is not None
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected)

    restored_optimizer = torch.optim.AdamW(
        loaded.model.parameters(),
        lr=loaded.training_config.learning_rate,
        weight_decay=loaded.training_config.weight_decay,
    )
    restored_optimizer.load_state_dict(loaded.optimizer_state_dict)
    assert restored_optimizer.state


def test_loss_decreases_validation_runs_and_metrics_include_progress(tmp_path) -> None:
    torch.manual_seed(17)
    model_config, model, tokenizer = _small_model()
    dataset = _repeated_sequence_dataset()
    validation_dataset = _repeated_sequence_dataset(12)
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=0.01,
        min_learning_rate=0.001,
        weight_decay=0.0,
        max_steps=40,
        gradient_clip_norm=1.0,
        seed=18,
        device="cpu",
        log_every_steps=100,
        save_every_steps=10,
        scheduler="cosine",
        warmup_steps=2,
        validation_every_steps=5,
        validation_batches=2,
    )
    initial_loss = evaluate_loss(
        model,
        dataset,
        model_config,
        training_config,
        max_batches=4,
        device="cpu",
    )
    checkpoint = tmp_path / "controlled.pt"
    metrics = tmp_path / "controlled.csv"
    result = train_model(
        model,
        dataset,
        model_config,
        training_config,
        checkpoint,
        tokenizer,
        validation_dataset=validation_dataset,
        metrics_path=metrics,
    )
    final_loss = evaluate_loss(
        model,
        dataset,
        model_config,
        training_config,
        max_batches=4,
        device="cpu",
    )

    assert final_loss < initial_loss
    assert result.steps == 40
    assert result.device == "cpu"
    assert math.isfinite(result.final_loss)
    assert result.validation_loss is not None and math.isfinite(result.validation_loss)
    assert result.best_validation_loss is not None
    assert math.isfinite(result.best_validation_loss)
    assert result.best_checkpoint_path is not None and result.best_checkpoint_path.is_file()
    assert result.metrics_path == metrics
    with metrics.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 40
    assert {
        "step",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "gradient_norm",
        "elapsed_seconds",
        "estimated_remaining_seconds",
    } <= set(rows[0])

    expected_validation_steps = {1, *range(5, 41, 5)}
    assert [int(row["step"]) for row in rows] == list(range(1, 41))
    for row in rows:
        step = int(row["step"])
        assert math.isfinite(float(row["train_loss"]))
        assert math.isfinite(float(row["learning_rate"]))
        assert math.isfinite(float(row["gradient_norm"]))
        assert math.isfinite(float(row["elapsed_seconds"]))
        assert math.isfinite(float(row["estimated_remaining_seconds"]))
        if step in expected_validation_steps:
            assert math.isfinite(float(row["validation_loss"]))
        else:
            assert row["validation_loss"] == ""

    assert float(rows[0]["learning_rate"]) == pytest.approx(0.005)
    assert float(rows[1]["learning_rate"]) == pytest.approx(0.01)
    assert float(rows[-1]["learning_rate"]) == pytest.approx(0.001)
    assert result.final_loss == pytest.approx(float(rows[-1]["train_loss"]))
    assert result.validation_loss == pytest.approx(float(rows[-1]["validation_loss"]))
    best_checkpoint = load_checkpoint(result.best_checkpoint_path, map_location="cpu")
    assert best_checkpoint.step <= result.steps
    assert best_checkpoint.best_validation_loss == pytest.approx(result.best_validation_loss)


def test_resume_continues_optimizer_scheduler_rng_and_step(tmp_path) -> None:
    torch.manual_seed(29)
    model_config, model, tokenizer = _small_model()
    dataset = _repeated_sequence_dataset(12)
    initial_config = TrainingConfig(
        batch_size=4,
        learning_rate=0.002,
        weight_decay=0.0,
        max_steps=4,
        log_every_steps=100,
        save_every_steps=2,
        device="cpu",
        scheduler="constant",
        validation_every_steps=2,
    )
    checkpoint = tmp_path / "resume.pt"
    first = train_model(
        model,
        dataset,
        model_config,
        initial_config,
        checkpoint,
        tokenizer,
    )
    loaded = load_checkpoint(checkpoint, map_location="cpu")
    metrics_path = checkpoint.with_suffix(".metrics.csv")
    # Simulate metrics flushed after the latest durable checkpoint but before a
    # crash. Resume must discard those uncommitted rows rather than duplicate steps.
    with metrics_path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write("5,999.00000000,,0.002,0.0,1.0,1.0\n")
    state_before_resume = {
        name: parameter.detach().clone()
        for name, parameter in loaded.model.named_parameters()
    }
    extended_config = replace(initial_config, max_steps=7)
    second = train_model(
        loaded.model,
        dataset,
        model_config,
        extended_config,
        checkpoint,
        tokenizer,
        resume_checkpoint=loaded,
    )
    resumed = load_checkpoint(checkpoint, map_location="cpu")

    assert first.steps == 4
    assert second.steps == 7
    assert resumed.step == 7
    assert resumed.optimizer_state_dict
    assert resumed.scheduler_state_dict["last_step"] == 7
    assert resumed.data_loader_rng_state is not None
    assert resumed.torch_rng_state is not None
    assert any(
        not torch.equal(state_before_resume[name], parameter.detach())
        for name, parameter in loaded.model.named_parameters()
    )
    with metrics_path.open(newline="", encoding="utf-8") as metrics_file:
        resumed_rows = list(csv.DictReader(metrics_file))
    assert [int(row["step"]) for row in resumed_rows] == list(range(1, 8))
    assert all(row["train_loss"] != "999.00000000" for row in resumed_rows)


def test_early_stopping_uses_validation_patience(tmp_path) -> None:
    model_config, model, tokenizer = _small_model()
    dataset = _repeated_sequence_dataset(10)
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=20,
        log_every_steps=100,
        save_every_steps=10,
        device="cpu",
        scheduler="constant",
        validation_every_steps=2,
        validation_batches=1,
        early_stopping_patience=1,
        early_stopping_min_delta=1_000_000.0,
    )
    result = train_model(
        model,
        dataset,
        model_config,
        config,
        tmp_path / "early.pt",
        tokenizer,
        validation_dataset=dataset,
    )
    assert result.stopped_early
    assert result.steps == 2
    assert result.best_checkpoint_path is not None
    assert result.best_checkpoint_path.is_file()


def test_best_checkpoint_tracks_small_validation_improvements(tmp_path, monkeypatch) -> None:
    model_config, model, tokenizer = _small_model()
    dataset = _repeated_sequence_dataset(10)
    validation_losses = iter([3.0, 2.95, 3.1])

    def controlled_validation(*_args, **_kwargs) -> float:
        return next(validation_losses)

    monkeypatch.setattr("puzzel.training.loop.evaluate_loss", controlled_validation)
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=3,
        log_every_steps=100,
        save_every_steps=3,
        device="cpu",
        scheduler="constant",
        validation_every_steps=1,
        early_stopping_patience=5,
        early_stopping_min_delta=0.1,
    )
    result = train_model(
        model,
        dataset,
        model_config,
        config,
        tmp_path / "best-min-delta.pt",
        tokenizer,
        validation_dataset=dataset,
    )

    assert result.best_validation_loss == pytest.approx(2.95)
    assert result.best_checkpoint_path is not None
    saved_best = load_checkpoint(result.best_checkpoint_path, map_location="cpu")
    assert saved_best.step == 2
    assert saved_best.best_validation_loss == pytest.approx(2.95)


def test_corrupt_weights_do_not_overwrite_good_checkpoint(tmp_path, monkeypatch) -> None:
    model_config, model, tokenizer = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    checkpoint = tmp_path / "protected.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=TrainingConfig(max_steps=2, scheduler="constant"),
        tokenizer=tokenizer,
        step=0,
    )
    original_bytes = checkpoint.read_bytes()

    def write_corrupt_archive(_payload, destination) -> None:
        Path(destination).write_bytes(b"not a valid torch checkpoint archive")

    with monkeypatch.context() as patch:
        patch.setattr("puzzel.checkpoint.torch.save", write_corrupt_archive)
        with pytest.raises(zipfile.BadZipFile):
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=TrainingConfig(max_steps=2, scheduler="constant"),
                tokenizer=tokenizer,
                step=1,
            )
    assert checkpoint.read_bytes() == original_bytes

    def write_incomplete_archive(_payload, destination) -> None:
        with zipfile.ZipFile(destination, mode="w"):
            pass

    with monkeypatch.context() as patch:
        patch.setattr("puzzel.checkpoint.torch.save", write_incomplete_archive)
        with pytest.raises(OSError, match="failed safe-load validation"):
            save_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=TrainingConfig(max_steps=2, scheduler="constant"),
                tokenizer=tokenizer,
                step=1,
            )
    assert checkpoint.read_bytes() == original_bytes

    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite model parameter"):
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            training_config=TrainingConfig(max_steps=2, scheduler="constant"),
            tokenizer=tokenizer,
            step=1,
        )

    assert checkpoint.read_bytes() == original_bytes
    assert load_checkpoint(checkpoint).step == 0


def test_nonfinite_loss_stops_training_and_preserves_checkpoint(tmp_path) -> None:
    model_config, good_model, tokenizer = _small_model()
    dataset = _repeated_sequence_dataset(10)
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=4,
        log_every_steps=10,
        save_every_steps=1,
        device="cpu",
        scheduler="constant",
    )
    checkpoint = tmp_path / "finite.pt"
    train_model(good_model, dataset, model_config, config, checkpoint, tokenizer)
    prior = load_checkpoint(checkpoint)
    with torch.no_grad():
        next(prior.model.parameters()).fill_(float("nan"))

    with pytest.raises(TrainingDivergedError, match="non-finite"):
        train_model(
            prior.model,
            dataset,
            model_config,
            config,
            checkpoint,
            tokenizer,
        )
    assert load_checkpoint(checkpoint).step == config.max_steps


def test_training_is_reproducible_for_fixed_seed(tmp_path) -> None:
    model_config, _, tokenizer = _small_model()
    model_config = replace(model_config, dropout=0.1)
    dataset = _repeated_sequence_dataset(10)
    config = TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0,
        max_steps=3,
        seed=123,
        log_every_steps=100,
        save_every_steps=3,
        device="cpu",
        scheduler="constant",
    )
    trained_states = []
    for index in range(2):
        seed_everything(987)
        model = TransformerLM(model_config)
        train_model(
            model,
            dataset,
            model_config,
            config,
            tmp_path / f"seed-{index}.pt",
            tokenizer,
        )
        trained_states.append(
            {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        )
    for name, first in trained_states[0].items():
        torch.testing.assert_close(first, trained_states[1][name], rtol=0, atol=0)


def test_cosine_schedule_resume_extension_does_not_jump_learning_rate() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    original_config = TrainingConfig(
        learning_rate=0.01,
        min_learning_rate=0.001,
        max_steps=10,
        scheduler="cosine",
    )
    original = LearningRateSchedule(optimizer, original_config)
    rate_at_resume = original.step(5)
    state = original.state_dict()

    extended = LearningRateSchedule(
        optimizer,
        replace(original_config, max_steps=20),
        last_step=5,
        resume_state=state,
    )
    next_rate = extended.step(6)

    assert next_rate < rate_at_resume
    assert next_rate > original_config.min_learning_rate


def test_buffered_shuffle_sampler_is_seeded_and_visits_each_example_once() -> None:
    dataset = _repeated_sequence_dataset(10)
    first = list(
        BufferedShuffleSampler(
            dataset, torch.Generator().manual_seed(51), buffer_size=8
        )
    )
    second = list(
        BufferedShuffleSampler(
            dataset, torch.Generator().manual_seed(51), buffer_size=8
        )
    )

    assert first == second
    assert sorted(first) == list(range(len(dataset)))
    assert first != list(range(len(dataset)))
