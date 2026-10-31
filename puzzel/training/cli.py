"""Command-line entry point for explicitly requested local pretraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from puzzel.checkpoint import LoadedCheckpoint, load_checkpoint
from puzzel.config import RunConfig, load_config
from puzzel.data.prepared import PreparedSplitDataset, verify_prepared_dataset
from puzzel.data.quality import normalize_text, quality_rejection_reason
from puzzel.data.text_dataset import TextSequenceDataset, encode_text
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    PAD_TOKEN_ID,
    TextTokenizer,
    tokenizer_from_dict,
)
from puzzel.training.hardware import build_hardware_report
from puzzel.training.loop import resolve_device, seed_everything, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puzzel",
        description=(
            "Train Puzzel's randomly initialized Transformer on a locally prepared "
            "licensed dataset (or a small local text file). No pretrained weights "
            "or external AI services are used."
        ),
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML model/training config")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        type=Path,
        help="verified prepared dataset directory (preferred; uses train/validation splits)",
    )
    source.add_argument(
        "--text",
        type=Path,
        help="one small local UTF-8 text file; split contiguously into train/validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/tiny.pt"),
        help="latest checkpoint (default: checkpoints/tiny.pt)",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="CSV metrics path (default: <output>.metrics.csv)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a Puzzel checkpoint created by this project",
    )
    return parser


def _tokenizer_for_text(run_config: RunConfig, training_text: str) -> TextTokenizer:
    if run_config.tokenizer.kind == "bpe":
        return BPETokenizer.train(
            training_text,
            target_vocab_size=run_config.tokenizer.vocab_size,
            min_frequency=run_config.tokenizer.min_frequency,
        )
    return ByteTokenizer()


def _raw_text_datasets(
    path: Path, run_config: RunConfig, model_config: ModelConfig
) -> tuple[TextTokenizer, TextSequenceDataset, TextSequenceDataset]:
    max_characters = run_config.dataset.max_document_characters
    with path.open("r", encoding="utf-8-sig") as source:
        raw_text = source.read(max_characters + 1)
    if len(raw_text) > max_characters:
        raise ValueError(
            "local text file exceeds dataset.max_document_characters; "
            "use the bounded prepared-dataset pipeline"
        )
    text = normalize_text(raw_text)
    if not text:
        raise ValueError("local text file is empty after normalization; refusing to train")
    reason = quality_rejection_reason(text)
    if reason is not None:
        raise ValueError(f"local text file failed the dataset quality check: {reason}")
    if len(text) < 2:
        raise ValueError("local text file is too short to make train and validation splits")

    validation_fraction = run_config.dataset.validation_fraction
    split_at = int(len(text) * (1.0 - validation_fraction))
    split_at = min(max(split_at, 1), len(text) - 1)
    training_text = normalize_text(text[:split_at])
    validation_text = normalize_text(text[split_at:])
    if not training_text or not validation_text:
        raise ValueError("local text split produced an empty training or validation portion")
    for split_name, split_text in (
        ("training", training_text),
        ("validation", validation_text),
    ):
        split_reason = quality_rejection_reason(split_text)
        if split_reason is not None:
            raise ValueError(f"local {split_name} text failed the quality check: {split_reason}")

    tokenizer = _tokenizer_for_text(run_config, training_text)
    train_ids = encode_text(training_text, tokenizer)
    validation_ids = encode_text(validation_text, tokenizer)
    train_dataset = TextSequenceDataset(train_ids, model_config.context_length)
    validation_dataset = TextSequenceDataset(validation_ids, model_config.context_length)
    return tokenizer, train_dataset, validation_dataset


def _prepared_datasets(
    path: Path,
    run_config: RunConfig,
) -> tuple[TextTokenizer, ModelConfig, PreparedSplitDataset, PreparedSplitDataset]:
    verification = verify_prepared_dataset(path)
    manifest_path = path / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_payload = json.loads(
        (path / manifest.get("tokenizer_file", "tokenizer.json")).read_text(
            encoding="utf-8"
        )
    )
    tokenizer = tokenizer_from_dict(tokenizer_payload)
    prepared_weights = manifest.get("language_sampling_weights", {})
    if prepared_weights != run_config.training.language_sampling_weights:
        raise ValueError("prepared dataset language sampling weights do not match config")
    if tokenizer.vocab_size != verification["vocabulary_size"]:
        raise ValueError("prepared dataset tokenizer and verified vocabulary differ")
    expected_kind = "bpe" if isinstance(tokenizer, BPETokenizer) else "byte"
    if run_config.tokenizer.kind != expected_kind:
        raise ValueError(
            f"config tokenizer kind {run_config.tokenizer.kind!r} does not match "
            f"prepared {expected_kind!r} tokenizer"
        )
    requested_context = run_config.model.context_length
    prepared_context = int(manifest["context_length"])
    if requested_context > prepared_context:
        raise ValueError(
            f"config context_length {requested_context} exceeds prepared data context "
            f"{prepared_context}; prepare data with a larger context or lower the config"
        )
    model_config = replace(run_config.model, vocab_size=tokenizer.vocab_size)
    train_dataset = PreparedSplitDataset(
        path, "train", context_length=requested_context
    )
    validation_dataset = PreparedSplitDataset(
        path, "validation", context_length=requested_context
    )
    print(
        f"Verified prepared dataset: {verification['document_count']} documents, "
        f"train={len(train_dataset):,} examples, "
        f"validation={len(validation_dataset):,} examples"
    )
    return tokenizer, model_config, train_dataset, validation_dataset


def _preflight_training_data(
    tokenizer: TextTokenizer,
    model_config: ModelConfig,
    train_dataset: PreparedSplitDataset | TextSequenceDataset,
    validation_dataset: PreparedSplitDataset | TextSequenceDataset,
) -> None:
    """Fail closed before model construction when either split is empty/invalid."""
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("pretraining safety check: tokenizer/model vocabulary mismatch")
    for split_name, dataset in (
        ("training", train_dataset),
        ("validation", validation_dataset),
    ):
        if len(dataset) <= 0:
            raise ValueError(f"pretraining safety check: {split_name} dataset is empty")
        input_ids, targets = dataset[0]
        if (
            input_ids.ndim != 1
            or targets.ndim != 1
            or input_ids.shape != targets.shape
            or input_ids.numel() != model_config.context_length
        ):
            raise ValueError(
                f"pretraining safety check: {split_name} example has invalid shape"
            )
        if (
            input_ids.min().item() < 0
            or targets.min().item() < 0
            or input_ids.max().item() >= model_config.vocab_size
            or targets.max().item() >= model_config.vocab_size
        ):
            raise ValueError(
                f"pretraining safety check: {split_name} contains out-of-range token IDs"
            )
        if torch.all(input_ids == model_config.pad_token_id).item() or torch.all(
            targets == model_config.pad_token_id
        ).item():
            raise ValueError(
                f"pretraining safety check: {split_name} example has no usable tokens"
            )
    print(
        f"Pretraining safety checks passed: {len(train_dataset):,} training and "
        f"{len(validation_dataset):,} validation examples; vocabulary="
        f"{model_config.vocab_size}, context={model_config.context_length}"
    )


def _print_hardware(config: ModelConfig, run_config: RunConfig, output: Path) -> None:
    report = build_hardware_report(
        config,
        batch_size=run_config.training.batch_size,
        device=run_config.training.device,
        max_steps=run_config.training.max_steps,
        disk_path=output.parent,
    )
    print("Hardware and memory estimate:")
    print(json.dumps(report, indent=2))
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")


def _validate_resume(
    resume: LoadedCheckpoint,
    model_config: ModelConfig,
    tokenizer: TextTokenizer,
) -> TransformerLM:
    if resume.model_config != model_config:
        raise ValueError("resume checkpoint model config does not match this run")
    if resume.tokenizer.to_dict() != tokenizer.to_dict():
        raise ValueError("resume checkpoint tokenizer does not match this dataset")
    return resume.model


def main() -> None:
    args = build_parser().parse_args()
    run_config = load_config(args.config)
    validation_dataset: PreparedSplitDataset | TextSequenceDataset | None

    if args.dataset is not None:
        tokenizer, model_config, train_dataset, validation_dataset = _prepared_datasets(
            args.dataset, run_config
        )
    else:
        if run_config.training.language_sampling_weights:
            raise ValueError(
                "language-aware sampling requires a prepared multilingual dataset; "
                "use --dataset instead of the single-file --text mode"
            )
        model_config = run_config.model
        tokenizer, train_dataset, validation_dataset = _raw_text_datasets(
            args.text, run_config, model_config
        )
        model_config = replace(model_config, vocab_size=tokenizer.vocab_size)

    if model_config.pad_token_id != PAD_TOKEN_ID:
        raise ValueError("model pad_token_id does not match tokenizer")
    if validation_dataset is None:
        raise ValueError("pretraining safety check: validation dataset is required")
    _preflight_training_data(tokenizer, model_config, train_dataset, validation_dataset)
    _print_hardware(model_config, run_config, args.output)
    device = resolve_device(run_config.training.device)

    resume: LoadedCheckpoint | None = None
    if args.resume is not None:
        if args.output.expanduser().resolve() != args.resume.expanduser().resolve():
            raise ValueError(
                "--output must match --resume so the latest and best checkpoint pair "
                "remain together"
            )
        resume = load_checkpoint(args.resume, map_location="cpu")
        model = _validate_resume(resume, model_config, tokenizer)
        print(
            f"Resuming Puzzel checkpoint at step {resume.step}; "
            f"device={device}; no external weights are loaded"
        )
    else:
        seed_everything(run_config.training.seed)
        model = TransformerLM(model_config)
        model.verify_random_initialization()
        print(
            f"Random initialization verified: {model.parameter_count():,} parameters; "
            f"device={device}; tokenizer={run_config.tokenizer.kind}; "
            f"vocabulary={tokenizer.vocab_size}/{run_config.tokenizer.vocab_size}; "
            f"training examples={len(train_dataset):,}"
        )

    result = train_model(
        model=model,
        dataset=train_dataset,
        model_config=model_config,
        training_config=run_config.training,
        checkpoint_path=args.output,
        tokenizer=tokenizer,
        validation_dataset=validation_dataset,
        metrics_path=args.metrics,
        resume_checkpoint=resume,
        language_sampling_weights=run_config.training.language_sampling_weights or None,
    )
    tokenizer_path = args.output.with_suffix(".tokenizer.json")
    tokenizer.save(tokenizer_path)
    best_summary = (
        "none" if result.best_checkpoint_path is None else str(result.best_checkpoint_path)
    )
    validation_summary = (
        "not evaluated" if result.validation_loss is None else f"{result.validation_loss:.4f}"
    )
    print(
        f"Finished at step {result.steps}; train_loss={result.final_loss:.4f}; "
        f"validation_loss={validation_summary}; best_validation={result.best_validation_loss}; "
        f"elapsed={result.elapsed_seconds:.1f}s; checkpoint={result.checkpoint_path}; "
        f"best_checkpoint={best_summary}; metrics={result.metrics_path}; "
        f"tokenizer={tokenizer_path}"
    )


if __name__ == "__main__":
    main()
