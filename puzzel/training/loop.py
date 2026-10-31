"""Reproducible AdamW training, validation, metrics, and safe checkpointing."""

from __future__ import annotations

import csv
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from puzzel.checkpoint import LoadedCheckpoint, save_checkpoint
from puzzel.config import TrainingConfig
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import ByteTokenizer, TextTokenizer


class TrainingDivergedError(RuntimeError):
    """Raised when loss or gradients become non-finite or persistently unstable."""


@dataclass(frozen=True, slots=True)
class TrainResult:
    steps: int
    final_loss: float
    checkpoint_path: Path
    device: str
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    best_checkpoint_path: Path | None = None
    metrics_path: Path | None = None
    elapsed_seconds: float = 0.0
    stopped_early: bool = False


def seed_everything(seed: int) -> None:
    """Seed Python and supported PyTorch devices for repeatable fresh runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto to an available accelerator or CPU, with clear failures."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


class LearningRateSchedule:
    """Small step-indexed warmup/cosine-or-constant learning-rate schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        *,
        last_step: int = 0,
        resume_state: dict[str, Any] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.last_step = last_step
        self._continuation_start_step: int | None = None
        self._continuation_start_lr: float | None = None
        if (
            resume_state is not None
            and self.config.scheduler == "cosine"
            and last_step > self.config.warmup_steps
            and resume_state.get("max_steps") != self.config.max_steps
        ):
            saved_lr = resume_state.get("last_learning_rate")
            if isinstance(saved_lr, (int, float)) and math.isfinite(saved_lr):
                self._continuation_start_step = last_step
                self._continuation_start_lr = float(saved_lr)

    def _learning_rate(self, step: int) -> float:
        if self._continuation_start_step is not None:
            remaining_steps = max(1, self.config.max_steps - self._continuation_start_step)
            progress = min(
                1.0,
                max(0.0, (step - self._continuation_start_step) / remaining_steps),
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.config.min_learning_rate + (
                self._continuation_start_lr - self.config.min_learning_rate
            ) * cosine
        if self.config.warmup_steps and step <= self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        if self.config.scheduler == "constant":
            return self.config.learning_rate
        decay_steps = max(1, self.config.max_steps - self.config.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.config.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (
            self.config.learning_rate - self.config.min_learning_rate
        ) * cosine

    def step(self, step: int) -> float:
        if step <= 0:
            raise ValueError("scheduler step must be positive")
        learning_rate = self._learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.last_step = step
        return learning_rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": self.last_step,
            "scheduler": self.config.scheduler,
            "learning_rate": self.config.learning_rate,
            "min_learning_rate": self.config.min_learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self.config.max_steps,
            "last_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }


class BufferedShuffleSampler(Sampler[int]):
    """Shuffle a map-style dataset with bounded index memory.

    The sampler keeps only a fixed-size index buffer instead of materializing a
    full ``randperm(len(dataset))``, which can be large for mmap corpora.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        generator: torch.Generator,
        *,
        buffer_size: int = 65_536,
    ) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.dataset = dataset
        self.generator = generator
        self.buffer_size = buffer_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        size = len(self.dataset)
        buffer = list(range(min(size, self.buffer_size)))
        next_index = len(buffer)
        while next_index < size:
            draw_count = min(1024, size - next_index)
            positions = torch.randint(
                len(buffer), (draw_count,), generator=self.generator
            ).tolist()
            for position in positions:
                yield buffer[position]
                buffer[position] = next_index
                next_index += 1
        for position in torch.randperm(len(buffer), generator=self.generator).tolist():
            yield buffer[position]


class LanguageBalancedSampler(Sampler[int]):
    """Sample prepared examples with configured per-language proportions.

    Language ranges are contiguous slices in the memory-mapped split. Sampling
    uses replacement so small languages are not drowned out by larger corpora;
    the generator is the same checkpointed generator used by the data loader.
    Only a small chunk of indices is materialized at a time.
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        language_ranges: Mapping[str, tuple[int, int]],
        language_weights: Mapping[str, float],
        generator: torch.Generator,
        *,
        chunk_size: int = 4096,
    ) -> None:
        self.dataset = dataset
        self.generator = generator
        self.chunk_size = chunk_size
        self.length = len(dataset)
        if self.length <= 0:
            raise ValueError("cannot balance an empty training dataset")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if set(language_ranges) != set(language_weights) or not language_weights:
            raise ValueError("language ranges must match configured sampling weights")
        if any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight <= 0
            for weight in language_weights.values()
        ):
            raise ValueError("language sampling weights must be positive finite numbers")
        if not math.isclose(sum(language_weights.values()), 1.0, abs_tol=1e-9):
            raise ValueError("language sampling weights must sum to 1.0")

        if any(
            not isinstance(range_value, tuple) or len(range_value) != 2
            for range_value in language_ranges.values()
        ):
            raise ValueError("language example ranges must be (start, count) pairs")
        expected_start = 0
        for language, (start, count) in sorted(
            language_ranges.items(), key=lambda item: item[1][0]
        ):
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or start != expected_start
                or count <= 0
            ):
                raise ValueError("language example ranges must be contiguous and non-empty")
            expected_start += count
        if expected_start != self.length:
            raise ValueError("language example ranges must cover the entire training split")

        self.languages = tuple(sorted(language_weights))
        self.ranges = dict(language_ranges)
        self.probabilities = torch.tensor(
            [language_weights[language] for language in self.languages],
            dtype=torch.float64,
        )

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        remaining = self.length
        while remaining:
            count = min(self.chunk_size, remaining)
            language_ids = torch.multinomial(
                self.probabilities,
                count,
                replacement=True,
                generator=self.generator,
            ).tolist()
            offsets = torch.rand(count, generator=self.generator).tolist()
            for language_id, offset in zip(language_ids, offsets):
                start, language_count = self.ranges[self.languages[language_id]]
                yield start + min(language_count - 1, int(offset * language_count))
            remaining -= count


def evaluate_loss(
    model: TransformerLM,
    dataset: Dataset[tuple[Tensor, Tensor]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    *,
    max_batches: int | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Compute mean next-token cross entropy without updating model parameters."""
    if len(dataset) <= 0:
        raise ValueError("evaluation dataset is empty")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive or None")
    evaluation_device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    generator = torch.Generator().manual_seed(training_config.seed ^ 0x5A17)
    evaluation_dataset: Dataset[tuple[Tensor, Tensor]] = dataset
    if max_batches is not None:
        sample_count = min(len(dataset), max_batches * training_config.batch_size)
        if sample_count < len(dataset):
            sample_indices = torch.randint(
                len(dataset), (sample_count,), generator=generator
            ).tolist()
            evaluation_dataset = Subset(dataset, sample_indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=evaluation_device.type == "cuda",
        generator=generator,
    )
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    try:
        with torch.no_grad():
            for index, (input_ids, targets) in enumerate(loader):
                if max_batches is not None and index >= max_batches:
                    break
                input_ids = input_ids.to(
                    evaluation_device, non_blocking=evaluation_device.type == "cuda"
                )
                targets = targets.to(
                    evaluation_device, non_blocking=evaluation_device.type == "cuda"
                )
                logits = model(input_ids)
                batch_loss = F.cross_entropy(
                    logits.reshape(-1, model_config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=model_config.pad_token_id,
                    reduction="sum",
                )
                valid_tokens = int((targets != model_config.pad_token_id).sum().item())
                if valid_tokens:
                    if not torch.isfinite(batch_loss).item():
                        raise TrainingDivergedError("validation loss became non-finite")
                    loss_sum += float(batch_loss.item())
                    token_count += valid_tokens
    finally:
        model.train(was_training)
    if token_count == 0:
        raise ValueError("evaluation dataset contains no non-padding targets")
    mean_loss = loss_sum / token_count
    if not math.isfinite(mean_loss):
        raise TrainingDivergedError("validation loss became non-finite")
    return mean_loss


def _compatible_resume_config(
    previous: TrainingConfig, current: TrainingConfig
) -> None:
    for field_name in (
        "learning_rate",
        "weight_decay",
        "scheduler",
        "warmup_steps",
        "min_learning_rate",
        "precision",
        "language_sampling_weights",
    ):
        if getattr(previous, field_name) != getattr(current, field_name):
            raise ValueError(
                f"resumed training must keep {field_name} unchanged "
                f"({getattr(previous, field_name)!r} != {getattr(current, field_name)!r})"
            )


def _save_training_checkpoint(
    path: Path,
    *,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    scheduler: LearningRateSchedule,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    tokenizer: TextTokenizer,
    step: int,
    loader_generator: torch.Generator,
    best_validation_loss: float | None,
    bad_evaluations: int,
) -> None:
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=training_config,
        tokenizer=tokenizer,
        step=step,
        scheduler_state_dict=scheduler.state_dict(),
        torch_rng_state=torch.get_rng_state(),
        cuda_rng_state=cuda_rng_state,
        data_loader_rng_state=loader_generator.get_state(),
        best_validation_loss=best_validation_loss,
        early_stopping_bad_evaluations=bad_evaluations,
    )


def _best_checkpoint_path(path: Path) -> Path:
    suffix = path.suffix or ".pt"
    return path.with_name(f"{path.stem}.best{suffix}")


def _write_metric_header(writer: csv.DictWriter[str], metric_file: Any) -> None:
    writer.writeheader()
    metric_file.flush()


def _truncate_metrics_after_step(
    path: Path, checkpoint_step: int, fields: tuple[str, ...]
) -> None:
    """Drop metrics written after the last durable checkpoint before resuming."""
    temporary_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(temporary_descriptor)
    temporary_path = Path(temporary_name)
    try:
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(fields):
                raise ValueError("existing metrics CSV header does not match this trainer")
            with temporary_path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for row in reader:
                    try:
                        row_step = int(row["step"])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("existing metrics CSV contains an invalid step") from exc
                    if row_step <= checkpoint_step:
                        writer.writerow(row)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def train_model(
    model: TransformerLM,
    dataset: Dataset[tuple[Tensor, Tensor]],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    checkpoint_path: str | Path,
    tokenizer: TextTokenizer | None = None,
    *,
    validation_dataset: Dataset[tuple[Tensor, Tensor]] | None = None,
    metrics_path: str | Path | None = None,
    resume_checkpoint: LoadedCheckpoint | None = None,
    language_sampling_weights: Mapping[str, float] | None = None,
) -> TrainResult:
    """Train next-token predictions with validation, monitoring, and resumable saves.

    ``max_steps`` is the absolute target step, so a resumed run continues from
    ``checkpoint.step + 1``. A validation dataset is optional for API
    compatibility, but early stopping requires one. CLI-prepared corpora always
    provide a distinct document-level validation split.
    """
    if len(dataset) <= 0:
        raise ValueError("training dataset is empty")
    if validation_dataset is not None and len(validation_dataset) <= 0:
        raise ValueError("validation dataset is empty")
    if training_config.early_stopping_patience and validation_dataset is None:
        raise ValueError("early stopping requires a validation dataset")
    if model.vocab_size != model_config.vocab_size:
        raise ValueError("model and model_config vocabulary sizes do not match")
    active_tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
    if active_tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer and model vocabulary sizes do not match")
    if model.context_length != model_config.context_length:
        raise ValueError("model and model_config context lengths do not match")

    start_step = 0
    best_validation_loss: float | None = None
    bad_evaluations = 0
    if resume_checkpoint is not None:
        if model is not resume_checkpoint.model:
            raise ValueError("resume model must be the model loaded from resume_checkpoint")
        if resume_checkpoint.model_config != model_config:
            raise ValueError("resume checkpoint model configuration does not match")
        if resume_checkpoint.tokenizer.to_dict() != active_tokenizer.to_dict():
            raise ValueError("resume checkpoint tokenizer does not match")
        _compatible_resume_config(resume_checkpoint.training_config, training_config)
        if resume_checkpoint.scheduler_state_dict is not None:
            scheduler_step = resume_checkpoint.scheduler_state_dict.get("last_step")
            if scheduler_step != resume_checkpoint.step:
                raise ValueError("resume checkpoint scheduler step does not match checkpoint step")
        if training_config.max_steps <= resume_checkpoint.step:
            raise ValueError("max_steps must be greater than the resumed checkpoint step")
        start_step = resume_checkpoint.step
        best_validation_loss = resume_checkpoint.best_validation_loss
        bad_evaluations = resume_checkpoint.early_stopping_bad_evaluations

    device = resolve_device(training_config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint.optimizer_state_dict)

    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(training_config.seed)
    if resume_checkpoint is not None and resume_checkpoint.data_loader_rng_state is not None:
        loader_generator.set_state(resume_checkpoint.data_loader_rng_state.cpu())
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(training_config.seed ^ 0x5A17)
    if language_sampling_weights:
        if training_config.num_workers != 0:
            raise ValueError(
                "language-aware sampling currently requires num_workers=0 for "
                "checkpoint-exact sampling"
            )
        language_ranges = getattr(dataset, "language_ranges", None)
        if not isinstance(language_ranges, Mapping):
            raise ValueError(
                "language-aware sampling requires a prepared dataset with language ranges"
            )
        sampler: Sampler[int] = LanguageBalancedSampler(
            dataset,
            language_ranges,
            language_sampling_weights,
            loader_generator,
            chunk_size=training_config.batch_size,
        )
    else:
        sampler = BufferedShuffleSampler(dataset, loader_generator)
    loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=sampler,
        num_workers=training_config.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=worker_generator,
    )
    batches = iter(loader)
    scheduler = LearningRateSchedule(
        optimizer,
        training_config,
        last_step=start_step,
        resume_state=None
        if resume_checkpoint is None
        else resume_checkpoint.scheduler_state_dict,
    )

    if resume_checkpoint is not None and resume_checkpoint.torch_rng_state is not None:
        torch.set_rng_state(resume_checkpoint.torch_rng_state.cpu())
        if device.type == "cuda" and resume_checkpoint.cuda_rng_state:
            torch.cuda.set_rng_state_all(resume_checkpoint.cuda_rng_state)
    else:
        seed_everything(training_config.seed)

    destination = Path(checkpoint_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metric_destination = (
        Path(metrics_path)
        if metrics_path is not None
        else destination.with_suffix(".metrics.csv")
    )
    metric_destination.parent.mkdir(parents=True, exist_ok=True)
    metric_fields = (
        "step",
        "train_loss",
        "validation_loss",
        "learning_rate",
        "gradient_norm",
        "elapsed_seconds",
        "estimated_remaining_seconds",
    )
    if resume_checkpoint is not None and metric_destination.exists():
        _truncate_metrics_after_step(metric_destination, resume_checkpoint.step, metric_fields)
    append_metrics = resume_checkpoint is not None and metric_destination.exists()
    metric_file = metric_destination.open(
        "a" if append_metrics else "w", newline="", encoding="utf-8"
    )
    writer: csv.DictWriter[str] = csv.DictWriter(metric_file, fieldnames=metric_fields)
    if not append_metrics or metric_destination.stat().st_size == 0:
        _write_metric_header(writer, metric_file)

    best_path = _best_checkpoint_path(destination) if validation_dataset is not None else None
    final_loss = float("nan")
    latest_validation_loss: float | None = None
    elapsed_start = time.monotonic()
    loss_ema: float | None = None
    consecutive_spikes = 0
    stopped_early = False
    current_step = start_step
    model.train()

    try:
        for step in range(start_step + 1, training_config.max_steps + 1):
            current_step = step
            try:
                input_ids, targets = next(batches)
            except StopIteration:
                batches = iter(loader)
                input_ids, targets = next(batches)

            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            if input_ids.shape[1] > model_config.context_length:
                raise ValueError("training examples exceed configured context length")
            if input_ids.shape != targets.shape:
                raise ValueError("training inputs and targets must have identical shapes")

            learning_rate = scheduler.step(step)
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, model_config.vocab_size),
                targets.reshape(-1),
                ignore_index=model_config.pad_token_id,
            )
            if not torch.isfinite(loss).item():
                raise TrainingDivergedError(
                    f"training loss became non-finite at step {step}; last checkpoint is preserved"
                )
            final_loss = float(loss.detach().item())
            if final_loss > training_config.unstable_loss_threshold:
                raise TrainingDivergedError(
                    f"training loss {final_loss:.4g} exceeded the configured instability "
                    f"threshold at step {step}; last checkpoint is preserved"
                )

            if loss_ema is not None and final_loss > max(
                loss_ema * training_config.loss_spike_multiplier, loss_ema + 5.0
            ):
                consecutive_spikes += 1
                print(
                    f"warning: training loss spike at step {step}: "
                    f"{final_loss:.4f} vs recent mean {loss_ema:.4f} "
                    f"({consecutive_spikes}/3 consecutive spikes)"
                )
                if consecutive_spikes >= 3:
                    raise TrainingDivergedError(
                        f"training loss remained unstable for three consecutive steps "
                        f"ending at step {step}; last checkpoint is preserved"
                    )
            else:
                consecutive_spikes = 0
            loss_ema = final_loss if loss_ema is None else 0.95 * loss_ema + 0.05 * final_loss

            loss.backward()
            max_norm = (
                training_config.gradient_clip_norm
                if training_config.gradient_clip_norm > 0
                else float("inf")
            )
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if not torch.isfinite(gradient_norm).item():
                optimizer.zero_grad(set_to_none=True)
                raise TrainingDivergedError(
                    f"gradient norm became non-finite at step {step}; last checkpoint is preserved"
                )
            optimizer.step()

            do_validate = validation_dataset is not None and (
                step == 1
                or step % training_config.validation_every_steps == 0
                or step == training_config.max_steps
            )
            if do_validate:
                latest_validation_loss = evaluate_loss(
                    model,
                    validation_dataset,
                    model_config,
                    training_config,
                    max_batches=training_config.validation_batches or None,
                    device=device,
                )
                previous_best = best_validation_loss
                new_best = (
                    previous_best is None
                    or latest_validation_loss < previous_best
                )
                improved_for_patience = (
                    previous_best is None
                    or latest_validation_loss
                    < previous_best - training_config.early_stopping_min_delta
                )
                if improved_for_patience:
                    bad_evaluations = 0
                else:
                    bad_evaluations += 1
                if new_best:
                    best_validation_loss = latest_validation_loss
                    assert best_path is not None
                    _save_training_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        model_config=model_config,
                        training_config=training_config,
                        tokenizer=active_tokenizer,
                        step=step,
                        loader_generator=loader_generator,
                        best_validation_loss=best_validation_loss,
                        bad_evaluations=bad_evaluations,
                    )

            elapsed = time.monotonic() - elapsed_start
            remaining = max(0, training_config.max_steps - step)
            eta = (elapsed / max(1, step - start_step)) * remaining
            writer.writerow(
                {
                    "step": step,
                    "train_loss": f"{final_loss:.8f}",
                    "validation_loss": ""
                    if not do_validate or latest_validation_loss is None
                    else f"{latest_validation_loss:.8f}",
                    "learning_rate": f"{learning_rate:.10g}",
                    "gradient_norm": f"{float(gradient_norm.item()):.8f}",
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "estimated_remaining_seconds": f"{eta:.3f}",
                }
            )
            metric_file.flush()

            if step == 1 or step % training_config.log_every_steps == 0 or do_validate:
                validation_display = (
                    "n/a"
                    if latest_validation_loss is None
                    else f"{latest_validation_loss:.4f}"
                )
                print(
                    f"step {step}/{training_config.max_steps} | "
                    f"loss {final_loss:.4f} | val {validation_display} | "
                    f"lr {learning_rate:.3g} | elapsed {elapsed:.1f}s | "
                    f"ETA {eta:.1f}s"
                )

            if step == 1 or step % training_config.save_every_steps == 0:
                _save_training_checkpoint(
                    destination,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_config=model_config,
                    training_config=training_config,
                    tokenizer=active_tokenizer,
                    step=step,
                    loader_generator=loader_generator,
                    best_validation_loss=best_validation_loss,
                    bad_evaluations=bad_evaluations,
                )

            if (
                do_validate
                and training_config.early_stopping_patience > 0
                and bad_evaluations >= training_config.early_stopping_patience
            ):
                stopped_early = True
                print(
                    f"early stopping at step {step}: validation did not improve for "
                    f"{bad_evaluations} evaluations"
                )
                break

        _save_training_checkpoint(
            destination,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            training_config=training_config,
            tokenizer=active_tokenizer,
            step=current_step,
            loader_generator=loader_generator,
            best_validation_loss=best_validation_loss,
            bad_evaluations=bad_evaluations,
        )
    finally:
        metric_file.close()

    elapsed = time.monotonic() - elapsed_start
    return TrainResult(
        steps=current_step,
        final_loss=final_loss,
        checkpoint_path=destination,
        device=str(device),
        validation_loss=latest_validation_loss,
        best_validation_loss=best_validation_loss,
        best_checkpoint_path=best_path,
        metrics_path=metric_destination,
        elapsed_seconds=elapsed,
        stopped_early=stopped_early,
    )
