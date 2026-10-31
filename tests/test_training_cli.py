from pathlib import Path

import pytest
import torch

from puzzel.checkpoint import load_checkpoint
from puzzel.config import load_config
from puzzel.data.pipeline import prepare_dataset
from puzzel.data.quality import normalize_text, quality_rejection_reason
from puzzel.model.config import ModelConfig
from puzzel.tokenizer import BPETokenizer, ByteTokenizer
from puzzel.training.cli import _preflight_training_data, main


def _write_tiny_config(path: Path, *, max_steps: int = 2) -> None:
    path.write_text(
        f"""[tokenizer]
kind = "bpe"
vocab_size = 280
min_frequency = 2

[model]
vocab_size = 280
context_length = 8
d_model = 16
num_layers = 1
num_heads = 2
dropout = 0.0

[training]
batch_size = 2
learning_rate = 0.001
weight_decay = 0.0
max_steps = {max_steps}
gradient_clip_norm = 1.0
seed = 31
device = "cpu"
log_every_steps = 1
save_every_steps = 1
scheduler = "constant"
validation_every_steps = 1
validation_batches = 1

[dataset]
validation_fraction = 0.2
test_fraction = 0.2
split_seed = 19
max_document_characters = 10000
""",
        encoding="utf-8",
    )


def test_cli_trains_from_prepared_local_data_with_validation(tmp_path, monkeypatch) -> None:
    # Tiny synthetic fixtures validate wiring only; this is not real language
    # data and no production pretraining is started by the test.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for index in range(7):
        (raw_dir / f"doc-{index}.txt").write_text(
            f"Synthetic fixture document {index}: a local sequence for tests! " * 4,
            encoding="utf-8",
        )

    config_path = tmp_path / "tiny-run.toml"
    _write_tiny_config(config_path)
    run_config = load_config(config_path)
    prepared = tmp_path / "prepared"
    prepare_dataset([raw_dir], prepared, run_config)
    checkpoint = tmp_path / "tiny-real-run.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "puzzel",
            "--config",
            str(config_path),
            "--dataset",
            str(prepared),
            "--output",
            str(checkpoint),
        ],
    )

    main()

    loaded = load_checkpoint(checkpoint, map_location="cpu")
    assert loaded.step == 2
    assert loaded.best_validation_loss is not None
    assert checkpoint.with_suffix(".best.pt").is_file()
    assert checkpoint.with_suffix(".metrics.csv").is_file()
    assert checkpoint.with_suffix(".tokenizer.json").is_file()

    # Extend the absolute step target and prove the CLI resumes its own checkpoint.
    _write_tiny_config(config_path, max_steps=3)
    monkeypatch.setattr(
        "sys.argv",
        [
            "puzzel",
            "--config",
            str(config_path),
            "--dataset",
            str(prepared),
            "--output",
            str(checkpoint),
            "--resume",
            str(checkpoint),
        ],
    )
    main()
    assert load_checkpoint(checkpoint, map_location="cpu").step == 3


def test_legacy_text_cli_splits_before_learning_bpe(tmp_path, monkeypatch) -> None:
    text = "".join(
        f"Owned local training text segment {index} with punctuation, Unicode café, "
        f"नमस्ते, and paragraphs.\n\nThis temporary CLI fixture is not a real corpus; "
        f"its record number is {index}. "
        for index in range(30)
    )
    text_path = tmp_path / "local.txt"
    text_path.write_text(text, encoding="utf-8")
    config_path = tmp_path / "text-run.toml"
    _write_tiny_config(config_path, max_steps=1)
    run_config = load_config(config_path)
    normalized_text = normalize_text(text)
    split_at = int(len(normalized_text) * (1.0 - run_config.dataset.validation_fraction))
    expected_training_text = normalized_text[:split_at]
    expected_tokenizer = BPETokenizer.train(
        expected_training_text, target_vocab_size=280, min_frequency=2
    )
    events: list[tuple[str, str]] = []
    original_bpe_train = BPETokenizer.train

    def record_validation(sample: str) -> str | None:
        events.append(("validate", sample))
        return quality_rejection_reason(sample)

    def record_bpe_training(training_text: str, **kwargs):
        events.append(("bpe", training_text))
        return original_bpe_train(training_text, **kwargs)

    monkeypatch.setattr("puzzel.training.cli.quality_rejection_reason", record_validation)
    monkeypatch.setattr(BPETokenizer, "train", record_bpe_training)
    checkpoint = tmp_path / "text-run.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "puzzel",
            "--config",
            str(config_path),
            "--text",
            str(text_path),
            "--output",
            str(checkpoint),
        ],
    )

    main()

    assert [event[0] for event in events] == [
        "validate",
        "validate",
        "validate",
        "bpe",
    ]
    assert events[-1] == ("bpe", expected_training_text)
    loaded = load_checkpoint(checkpoint, map_location="cpu")
    assert loaded.step == 1
    assert loaded.tokenizer.to_dict() == expected_tokenizer.to_dict()
    assert loaded.best_validation_loss is not None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("", "empty after normalization"),
        ("  \n\t  ", "empty after normalization"),
        ("\ufffd damaged text", "failed the dataset quality check"),
        ("x" * 120, "failed the dataset quality check"),
        ("x", "too short to make train and validation splits"),
        ("Original sentence. " * 600, "exceeds dataset.max_document_characters"),
        (b"valid prefix\ninvalid byte: \xff", "codec can't decode"),
    ],
)
def test_cli_refuses_empty_or_invalid_local_text_before_model_init(
    tmp_path, monkeypatch, payload, message
) -> None:
    text_path = tmp_path / "invalid-local.txt"
    if isinstance(payload, bytes):
        text_path.write_bytes(payload)
    else:
        text_path.write_text(payload, encoding="utf-8")
    config_path = tmp_path / "text-run.toml"
    _write_tiny_config(config_path, max_steps=1)
    checkpoint = tmp_path / "must-not-be-created.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "puzzel",
            "--config",
            str(config_path),
            "--text",
            str(text_path),
            "--output",
            str(checkpoint),
        ],
    )

    def unexpected_model_construction(*args, **kwargs):
        raise AssertionError("invalid text must be rejected before model construction")

    monkeypatch.setattr(
        "puzzel.training.cli.TransformerLM", unexpected_model_construction
    )
    with pytest.raises(ValueError, match=message):
        main()
    assert not checkpoint.exists()


def test_training_preflight_rejects_empty_and_out_of_range_examples() -> None:
    tokenizer = ByteTokenizer()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=2,
        d_model=4,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )

    class ExampleDataset:
        def __init__(self, example=None):
            self.example = example

        def __len__(self):
            return 0 if self.example is None else 1

        def __getitem__(self, index):
            if self.example is None or index != 0:
                raise IndexError(index)
            return self.example

    valid = ExampleDataset(
        (torch.tensor([1, 2]), torch.tensor([2, 3]))
    )
    empty = ExampleDataset()
    invalid = ExampleDataset(
        (torch.tensor([1, tokenizer.vocab_size]), torch.tensor([2, 3]))
    )

    with pytest.raises(ValueError, match="training dataset is empty"):
        _preflight_training_data(tokenizer, model_config, empty, valid)
    with pytest.raises(ValueError, match="contains out-of-range token IDs"):
        _preflight_training_data(tokenizer, model_config, valid, invalid)
