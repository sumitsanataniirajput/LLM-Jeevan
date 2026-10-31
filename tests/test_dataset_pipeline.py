import hashlib
import json
import sqlite3
import struct

import pytest

from puzzel.config import (
    DatasetConfig,
    RunConfig,
    TokenizerConfig,
    TrainingConfig,
)
from puzzel.data.pipeline import DatasetPreparationError, prepare_dataset
from puzzel.data.prepared import (
    DatasetVerificationError,
    PreparedSplitDataset,
    verify_prepared_dataset,
)
from puzzel.data.quality import normalize_text
from puzzel.model.config import ModelConfig
from puzzel.tokenizer import BOS_TOKEN_ID, BPETokenizer


def _config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            vocab_size=280,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
        training=TrainingConfig(device="cpu"),
        tokenizer=TokenizerConfig(kind="bpe", vocab_size=280, min_frequency=2),
        dataset=DatasetConfig(
            validation_fraction=0.2,
            test_fraction=0.2,
            split_seed=731,
            max_document_characters=10_000,
        ),
    )


def _create_local_fixtures(raw_dir):
    raw_dir.mkdir()
    unique_documents = []
    for index in range(6):
        text = (
            f"Text file document {index}: café, punctuation!\n\n"
            f"Second paragraph for record {index}."
        )
        (raw_dir / f"doc-{index:02}.txt").write_text(text, encoding="utf-8")
        unique_documents.append(normalize_text(text))

    rows = []
    duplicate = "Text file document 0: café,   punctuation!\n\nSecond paragraph for record 0."
    rows.append({"text": duplicate})
    for index in range(6, 12):
        text = f"JSONL document {index}: नमस्ते — distinct text, sentence {index}!"
        rows.append({"content": text})
        unique_documents.append(normalize_text(text))
    rows.extend(
        [
            {"text": "  \n  "},
            {"text": "damaged \ufffd characters"},
            {"text": "abcabc" * 100},
            {"text": 12345},
        ]
    )
    jsonl_path = raw_dir / "records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
        target.write("{ malformed json\n")

    (raw_dir / "unreadable.txt").write_bytes(b"valid prefix\ninvalid byte: \xff")
    return unique_documents


def test_prepare_txt_jsonl_clean_split_tokenize_and_verify(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    unique_documents = _create_local_fixtures(raw_dir)
    output_dir = tmp_path / "prepared"
    run_config = _config()

    result = prepare_dataset([raw_dir], output_dir, run_config)
    stats = result["stats"]
    ingestion = stats["ingestion"]

    assert result["verification"]["status"] == "ok"
    with pytest.raises(FileExistsError, match="output directory already exists"):
        prepare_dataset([raw_dir], output_dir, run_config)
    with sqlite3.connect(output_dir / "documents.sqlite3") as connection:
        metadata_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)")
        }
    assert "text" not in metadata_columns
    assert stats["document_count"] == 12
    assert stats["total_characters"] == sum(map(len, unique_documents))
    assert stats["total_tokens"] > stats["document_count"]
    assert 260 < stats["vocabulary_size"] <= 280
    assert stats["duplicate_count"] == 1
    assert ingestion["empty_documents"] == 1
    assert ingestion["malformed_jsonl_records"] == 2
    assert ingestion["files_discovered"] == 8
    assert ingestion["files_read"] == 7
    assert len(ingestion["unreadable_files"]) == 1
    assert ingestion["rejected_documents"]["replacement_character"] == 1
    assert ingestion["rejected_documents"]["excessive_repeated_ngrams"] == 1

    split_sizes = stats["split_sizes"]
    assert {split: split_sizes[split]["documents"] for split in split_sizes} == {
        "train": 8,
        "validation": 2,
        "test": 2,
    }
    assert result["manifest"]["tokenizer_training_split"] == "train"

    # Reproduce the deterministic document assignment and prove BPE learned only
    # from the training documents, not validation/test documents.
    ranked_documents = []
    for text in unique_documents:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        split_key = hashlib.sha256(f"731:{digest}".encode("utf-8")).hexdigest()
        ranked_documents.append((split_key, text))
    ranked_documents.sort(key=lambda item: item[0])
    expected_train_texts = [text for _, text in ranked_documents[4:]]
    expected_tokenizer = BPETokenizer.train(
        expected_train_texts,
        target_vocab_size=280,
        min_frequency=2,
    )
    from puzzel.tokenizer import BPETokenizer as BPE

    trained_tokenizer = BPE.load(output_dir / "tokenizer.json")
    assert trained_tokenizer.to_dict() == expected_tokenizer.to_dict()

    assert verify_prepared_dataset(output_dir)["status"] == "ok"
    for split in ("train", "validation", "test"):
        dataset = PreparedSplitDataset(output_dir, split)
        assert len(dataset) == split_sizes[split]["examples"]
        input_ids, target_ids = dataset[0]
        assert input_ids.shape == (run_config.model.context_length,)
        assert target_ids.shape == input_ids.shape
        assert int(input_ids.min()) >= 0
        assert int(input_ids.max()) < stats["vocabulary_size"]
        for row in dataset._storage:
            assert int((row == BOS_TOKEN_ID).sum()) <= 1

    shortened = PreparedSplitDataset(output_dir, "train", context_length=4)
    assert shortened[0][0].shape == (4,)
    with pytest.raises(ValueError, match="no greater than the prepared dataset context"):
        PreparedSplitDataset(output_dir, "train", context_length=9)


def test_prepare_populates_precreated_empty_output_scaffold(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _create_local_fixtures(raw_dir)
    output_dir = tmp_path / "processed" / "multilingual-v1"
    output_dir.mkdir(parents=True)
    marker = output_dir / ".gitkeep"
    marker.touch()

    result = prepare_dataset([raw_dir], output_dir, _config())

    assert result["verification"]["status"] == "ok"
    assert not marker.exists()
    assert (output_dir / "train.bin").is_file()
    assert (output_dir / "validation.bin").is_file()
    assert (output_dir / "test.bin").is_file()


def test_verifier_detects_cross_split_leakage_and_out_of_range(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    _create_local_fixtures(raw_dir)
    output_dir = tmp_path / "prepared"
    result = prepare_dataset([raw_dir], output_dir, _config())

    metadata_path = output_dir / result["manifest"]["metadata_file"]
    with sqlite3.connect(metadata_path) as connection:
        row = connection.execute(
            "SELECT digest, split, language, source, license, characters, "
            "utf8_bytes, tokens FROM documents LIMIT 1"
        ).fetchone()
        digest, original_split, language, source, license_name, characters, utf8_bytes, tokens = row
        leaked_split = "validation" if original_split != "validation" else "test"
        document_id = connection.execute(
            "SELECT MAX(document_id) + 1 FROM documents"
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO documents "
            "(document_id, digest, split, language, source, license, "
            "characters, utf8_bytes, tokens) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                document_id,
                digest,
                leaked_split,
                language,
                source,
                license_name,
                characters,
                utf8_bytes,
                tokens,
            ),
        )
    with pytest.raises(DatasetVerificationError, match="hash appears in more than one split"):
        verify_prepared_dataset(output_dir)

    with sqlite3.connect(metadata_path) as connection:
        connection.execute("DELETE FROM documents WHERE document_id=?", (document_id,))

    train_file = output_dir / "train.bin"
    with train_file.open("r+b") as token_file:
        token_file.write(struct.pack("<i", result["stats"]["vocabulary_size"]))
    with pytest.raises(DatasetVerificationError, match="outside the vocabulary"):
        verify_prepared_dataset(output_dir)


def test_verifier_rejects_manifest_model_vocab_mismatch_and_missing_split_metadata(
    tmp_path,
) -> None:
    raw_dir = tmp_path / "raw"
    _create_local_fixtures(raw_dir)
    output_dir = tmp_path / "prepared"
    prepare_dataset([raw_dir], output_dir, _config())

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_config"]["vocab_size"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(
        DatasetVerificationError, match="model and manifest vocabulary sizes differ"
    ):
        verify_prepared_dataset(output_dir)

    manifest["model_config"]["vocab_size"] = manifest["vocab_size"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with sqlite3.connect(output_dir / "documents.sqlite3") as connection:
        connection.execute("DELETE FROM documents WHERE split='test'")
    with pytest.raises(DatasetVerificationError, match="metadata is missing the test split"):
        verify_prepared_dataset(output_dir)


def test_prepare_rejects_too_few_documents_without_partial_output(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for index in range(2):
        (raw_dir / f"doc-{index}.txt").write_text(f"short document {index}", encoding="utf-8")
    output_dir = tmp_path / "prepared"

    with pytest.raises(DatasetPreparationError, match="at least three unique"):
        prepare_dataset([raw_dir], output_dir, _config())
    assert not output_dir.exists()
