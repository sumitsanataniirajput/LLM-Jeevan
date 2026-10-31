"""Atomic, validated save/load helpers for Puzzel-owned training checkpoints."""

from __future__ import annotations

import math
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from puzzel.config import TrainingConfig
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import TextTokenizer, tokenizer_from_dict


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Model and resumable state reconstructed from a Puzzel checkpoint."""

    model: TransformerLM
    tokenizer: TextTokenizer
    model_config: ModelConfig
    training_config: TrainingConfig
    step: int
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None = None
    torch_rng_state: Tensor | None = None
    cuda_rng_state: list[Tensor] | None = None
    data_loader_rng_state: Tensor | None = None
    best_validation_loss: float | None = None
    early_stopping_bad_evaluations: int = 0


def _assert_finite_tensor_values(tensors: dict[str, Any], label: str) -> None:
    for name, value in tensors.items():
        if isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex()):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"refusing to save non-finite {label} tensor {name}")
        elif isinstance(value, dict):
            _assert_finite_tensor_values(value, f"{label}.{name}")
        elif isinstance(value, (list, tuple)):
            nested = {str(index): item for index, item in enumerate(value)}
            _assert_finite_tensor_values(nested, label)


def _sync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Directory fsync is not supported by every filesystem.
        pass


def _validate_temporary_checkpoint(
    path: Path,
    *,
    model: TransformerLM,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: TextTokenizer,
    step: int,
) -> None:
    """Safe-load archive metadata on the meta device before replacing a target."""
    try:
        payload = torch.load(path, map_location="meta", weights_only=True)
    except Exception as exc:
        raise OSError("temporary checkpoint failed safe-load validation") from exc
    if not isinstance(payload, dict) or payload.get("format") != "puzzel-checkpoint-v1":
        raise OSError("temporary checkpoint has an invalid format")
    if payload.get("step") != step:
        raise OSError("temporary checkpoint step failed validation")
    if payload.get("model_config") != asdict(model_config):
        raise OSError("temporary checkpoint model configuration failed validation")
    if payload.get("training_config") != asdict(training_config):
        raise OSError("temporary checkpoint training configuration failed validation")
    if payload.get("tokenizer") != tokenizer.to_dict():
        raise OSError("temporary checkpoint tokenizer failed validation")
    state_dict = payload.get("model_state_dict")
    expected_state = model.state_dict()
    if not isinstance(state_dict, dict) or state_dict.keys() != expected_state.keys():
        raise OSError("temporary checkpoint model state failed validation")
    for name, expected in expected_state.items():
        saved = state_dict[name]
        if (
            not isinstance(saved, Tensor)
            or saved.shape != expected.shape
            or saved.dtype != expected.dtype
        ):
            raise OSError(f"temporary checkpoint tensor {name} failed shape/dtype validation")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise OSError("temporary checkpoint optimizer state failed validation")


def save_checkpoint(
    path: str | Path,
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: TextTokenizer,
    step: int,
    scheduler_state_dict: dict[str, Any] | None = None,
    torch_rng_state: Tensor | None = None,
    cuda_rng_state: list[Tensor] | None = None,
    data_loader_rng_state: Tensor | None = None,
    best_validation_loss: float | None = None,
    early_stopping_bad_evaluations: int = 0,
) -> Path:
    """Atomically save training state without replacing a good file on failure.

    Model/optimizer finiteness is checked before writing. The complete temporary
    PyTorch archive is CRC-validated and safe-loaded on the meta device before an
    atomic replacement of the target.
    If serialization, validation, or the replace fails, an existing checkpoint
    is left untouched.
    """
    destination = Path(path)
    if model.vocab_size != model_config.vocab_size:
        raise ValueError("model and model_config vocabulary sizes do not match")
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes do not match")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if (
        not isinstance(early_stopping_bad_evaluations, int)
        or isinstance(early_stopping_bad_evaluations, bool)
        or early_stopping_bad_evaluations < 0
    ):
        raise ValueError("early_stopping_bad_evaluations must be non-negative")
    if best_validation_loss is not None and (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(best_validation_loss)
        or best_validation_loss < 0
    ):
        raise ValueError("best_validation_loss must be finite and non-negative")

    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all().item():
            raise ValueError(f"refusing to save non-finite model parameter {name}")
    optimizer_state = optimizer.state_dict()
    _assert_finite_tensor_values(optimizer_state, "optimizer state")

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        payload = {
            "format": "puzzel-checkpoint-v1",
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state_dict,
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "tokenizer": tokenizer.to_dict(),
            "torch_rng_state": None if torch_rng_state is None else torch_rng_state.cpu(),
            "cuda_rng_state": None
            if cuda_rng_state is None
            else [state.cpu() for state in cuda_rng_state],
            "data_loader_rng_state": None
            if data_loader_rng_state is None
            else data_loader_rng_state.cpu(),
            "best_validation_loss": best_validation_loss,
            "early_stopping_bad_evaluations": early_stopping_bad_evaluations,
        }
        torch.save(payload, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        with zipfile.ZipFile(temporary_path, mode="r") as archive:
            damaged_member = archive.testzip()
            if damaged_member is not None:
                raise OSError(f"checkpoint archive CRC validation failed at {damaged_member}")
        _validate_temporary_checkpoint(
            temporary_path,
            model=model,
            model_config=model_config,
            training_config=training_config,
            tokenizer=tokenizer,
            step=step,
        )
        os.replace(temporary_path, destination)
        _sync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> LoadedCheckpoint:
    """Load a Puzzel checkpoint using PyTorch's restricted weights-only loader.

    The file reconstructs only Puzzel's own architecture and tokenizer; no
    remote or pretrained model path exists. Missing resume-only fields are
    accepted for checkpoints written by earlier Puzzel development phases.
    """
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "puzzel-checkpoint-v1":
        raise ValueError("not a supported Puzzel checkpoint")

    try:
        model_config = ModelConfig(**payload["model_config"])
        training_config = TrainingConfig(**payload["training_config"])
        tokenizer = tokenizer_from_dict(payload["tokenizer"])
        step = payload["step"]
        state_dict = payload["model_state_dict"]
        optimizer_state_dict = payload["optimizer_state_dict"]
        scheduler_state_dict = payload.get("scheduler_state_dict")
        torch_rng_state = payload.get("torch_rng_state")
        cuda_rng_state = payload.get("cuda_rng_state")
        data_loader_rng_state = payload.get("data_loader_rng_state")
        best_validation_loss = payload.get("best_validation_loss")
        early_stopping_bad_evaluations = payload.get("early_stopping_bad_evaluations", 0)
    except (KeyError, TypeError) as exc:
        raise ValueError("checkpoint metadata is incomplete or malformed") from exc

    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("checkpoint tokenizer and model vocabulary sizes differ")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step is invalid")
    if not isinstance(state_dict, dict) or not isinstance(optimizer_state_dict, dict):
        raise ValueError("checkpoint state dictionaries are malformed")
    if scheduler_state_dict is not None and not isinstance(scheduler_state_dict, dict):
        raise ValueError("checkpoint scheduler state is malformed")
    if torch_rng_state is not None and not isinstance(torch_rng_state, Tensor):
        raise ValueError("checkpoint torch RNG state is malformed")
    if data_loader_rng_state is not None and not isinstance(data_loader_rng_state, Tensor):
        raise ValueError("checkpoint data-loader RNG state is malformed")
    if cuda_rng_state is not None and (
        not isinstance(cuda_rng_state, list)
        or any(not isinstance(state, Tensor) for state in cuda_rng_state)
    ):
        raise ValueError("checkpoint CUDA RNG state is malformed")
    if best_validation_loss is not None and (
        not isinstance(best_validation_loss, (int, float))
        or isinstance(best_validation_loss, bool)
        or not math.isfinite(best_validation_loss)
        or best_validation_loss < 0
    ):
        raise ValueError("checkpoint best validation loss is invalid")
    if (
        not isinstance(early_stopping_bad_evaluations, int)
        or isinstance(early_stopping_bad_evaluations, bool)
        or early_stopping_bad_evaluations < 0
    ):
        raise ValueError("checkpoint early-stopping state is invalid")

    model = TransformerLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all().item():
            raise ValueError(f"checkpoint has non-finite model parameter {name}")
    model.to(map_location)
    model.eval()
    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        training_config=training_config,
        step=step,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        torch_rng_state=torch_rng_state,
        cuda_rng_state=cuda_rng_state,
        data_loader_rng_state=data_loader_rng_state,
        best_validation_loss=best_validation_loss,
        early_stopping_bad_evaluations=early_stopping_bad_evaluations,
    )
