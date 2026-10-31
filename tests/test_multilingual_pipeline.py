from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from collections import Counter
from pathlib import Path

import pytest
import torch

from puzzel.config import (
    DatasetConfig,
    RunConfig,
    TokenizerConfig,
    TrainingConfig,
    load_config,
)
from puzzel.data.pipeline import prepare_dataset
from puzzel.data.prepared import (
    DatasetVerificationError,
    PreparedSplitDataset,
    verify_prepared_dataset,
)
from puzzel.languages import DEFAULT_MULTILINGUAL_WEIGHTS, LANGUAGE_NAMES
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS,
    BPETokenizer,
    ByteTokenizer,
)
from puzzel.training.loop import LanguageBalancedSampler, train_model


SYNTHETIC_LANGUAGE_TEXTS = {
    "en": [
        "English note zero: the quick fox meets a river.\n\nA second English paragraph ends here!",
        (
            "English note one: bright yellow birds cross the quiet valley?"
            "\n\nAnother paragraph follows."
        ),
        "English note two: small tools make careful work easier; item two.",
    ],
    "hi": [
        "हिंदी नोट शून्य: नमस्ते दुनिया!\n\nयह दूसरा अनुच्छेद है।",
        "हिंदी नोट एक: चिड़िया नीले आकाश में उड़ती है।\n\nएक और वाक्य।",
        "हिंदी नोट दो: भाषा, विराम-चिह्न और Unicode सुरक्षित रहें।",
    ],
    "es": [
        "Nota española cero: ¡Hola, mundo!\n\nLa segunda frase termina aquí.",
        "Nota española uno: ¿Qué tal? El pájaro cruza el valle.\n\nOtro párrafo.",
        "Nota española dos: canción, acción y puntuación; elemento dos.",
    ],
    "fr": [
        "Note française zéro : Bonjour le monde !\n\nVoici un deuxième paragraphe.",
        "Note française un : Où est l’oiseau ? Il traverse la vallée.\n\nEncore une phrase.",
        "Note française deux : cœur, déjà-vu et ponctuation ; élément deux.",
    ],
    "ar": [
        "ملاحظة عربية صفر: مرحبًا بالعالم!\n\nهذه فقرة ثانية.",
        "ملاحظة عربية واحد: تطير الطيور فوق الوادي.\n\nجملة أخرى؟",
        "ملاحظة عربية اثنان: اللغة وعلامات الترقيم محفوظة، رقم اثنين.",
    ],
    "zh": [
        "中文记录零：你好，世界！\n\n这是第二段文字。",
        "中文记录一：小鸟飞过安静的山谷。\n\n还有一句话。",
        "中文记录二：Unicode、标点和段落都应保留；编号二。",
    ],
}


def _multilingual_config() -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            vocab_size=300,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
        ),
        training=TrainingConfig(
            device="cpu",
            language_sampling_weights=dict(DEFAULT_MULTILINGUAL_WEIGHTS),
        ),
        tokenizer=TokenizerConfig(kind="bpe", vocab_size=300, min_frequency=1),
        dataset=DatasetConfig(
            validation_fraction=0.1,
            test_fraction=0.1,
            split_seed=2026,
            max_document_characters=10_000,
            require_document_metadata=True,
        ),
    )


def _write_multilingual_corpus(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True)
    for language in LANGUAGE_NAMES:
        language_dir = raw_dir / language
        language_dir.mkdir()
        jsonl_path = language_dir / "documents.jsonl"
        records = []
        for index, text in enumerate(SYNTHETIC_LANGUAGE_TEXTS[language][:2]):
            records.append(
                {
                    "text": text,
                    "language": language,
                    "source": f"synthetic-test:{language}:jsonl:{index}",
                    "license": "test-only",
                }
            )
        if language == "en":
            # A normalized exact duplicate is detected and counted, not kept.
            records.append(dict(records[0], source="synthetic-test:duplicate"))
            records.append(
                {
                    "text": " \n ",
                    "language": language,
                    "source": "synthetic-test:empty",
                    "license": "test-only",
                }
            )
            records.append(
                {
                    "text": "damaged \ufffd text",
                    "language": language,
                    "source": "synthetic-test:corrupt-marker",
                    "license": "test-only",
                }
            )
        with jsonl_path.open("w", encoding="utf-8") as target:
            for record in records:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if language == "hi":
                target.write("{ malformed JSON\n")

        txt_path = language_dir / "document-02.txt"
        txt_path.write_text(SYNTHETIC_LANGUAGE_TEXTS[language][2], encoding="utf-8")
        sidecar = {
            "language": language,
            "source": f"synthetic-test:{language}:txt",
            "license": "test-only",
        }
        txt_path.with_name(txt_path.name + ".meta.json").write_text(
            json.dumps(sidecar, ensure_ascii=False), encoding="utf-8"
        )

    # An invalid UTF-8 source is reported and excluded without corrupting other files.
    (raw_dir / "zh" / "invalid-utf8.txt").write_bytes(b"broken: \xff")


def test_multilingual_jsonl_and_txt_provenance_unicode_and_stratified_splits(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_multilingual_corpus(raw_dir)
    output_dir = tmp_path / "prepared"

    result = prepare_dataset([raw_dir], output_dir, _multilingual_config())
    stats = result["stats"]
    ingestion = stats["ingestion"]

    assert result["verification"]["status"] == "ok"
    assert stats["duplicate_count"] == 1
    assert stats["duplicate_count_by_language"] == {"en": 1}
    assert ingestion["empty_documents"] == 1
    assert ingestion["rejected_documents"]["replacement_character"] == 1
    assert ingestion["malformed_jsonl_records"] == 1
    assert ingestion["metadata_errors"] == 0
    assert len(ingestion["unreadable_files"]) == 1
    assert stats["document_count"] == 6 * 3
    assert stats["language_sampling_weights"] == DEFAULT_MULTILINGUAL_WEIGHTS
    assert set(stats["language_sizes"]) == set(LANGUAGE_NAMES)
    assert set(stats["duplicate_count_by_language"]) == {"en"}

    for split in ("train", "validation", "test"):
        ranges = result["manifest"]["splits"][split]["language_ranges"]
        assert set(ranges) == set(LANGUAGE_NAMES)
        for language in LANGUAGE_NAMES:
            language_split = stats["language_sizes"][language]["splits"][split]
            assert language_split["documents"] == 1
            assert language_split["examples"] == ranges[language]["examples"]
            assert language_split["tokens"] > 0

    with sqlite3.connect(output_dir / "documents.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE TRIM(language)!='' "
            "AND TRIM(source)!='' AND TRIM(license)!=''"
        ).fetchone()[0] == 18
        assert connection.execute(
            "SELECT COUNT(DISTINCT language) FROM documents"
        ).fetchone()[0] == 6
        assert connection.execute(
            "SELECT COUNT(*) FROM documents WHERE source LIKE 'synthetic-test:%'"
        ).fetchone()[0] == 18

    audit = result["verification"]["tokenizer_audit"]
    assert audit["byte_fallback_ids"] == "0-255"
    assert audit["byte_fallback_round_trip"] == "ok"
    assert audit["unicode_probe_round_trip"] == "ok"
    assert audit["special_tokens"] == SPECIAL_TOKENS

    byte_tokenizer = ByteTokenizer()
    bpe_tokenizer = BPETokenizer.load(output_dir / "tokenizer.json")
    for language in LANGUAGE_NAMES:
        for text in SYNTHETIC_LANGUAGE_TEXTS[language]:
            assert byte_tokenizer.decode(byte_tokenizer.encode(text)) == text
            assert bpe_tokenizer.decode(bpe_tokenizer.encode(text)) == text
    marked = bpe_tokenizer.encode("नमस्ते — مرحبًا — 中文", add_bos=True, add_eos=True)
    assert marked[0] == BOS_TOKEN_ID
    assert marked[-1] == EOS_TOKEN_ID
    assert bpe_tokenizer.decode(marked) == "नमस्ते — مرحبًا — 中文"

    for split in ("train", "validation", "test"):
        dataset = PreparedSplitDataset(output_dir, split)
        assert set(dataset.language_ranges) == set(LANGUAGE_NAMES)
        assert len(dataset) == stats["split_sizes"][split]["examples"]


def test_language_balanced_sampler_uses_configured_weights_with_replacement() -> None:
    class DummyDataset:
        def __len__(self) -> int:
            return 12_000

    per_language = 2_000
    language_ranges = {
        language: (index * per_language, per_language)
        for index, language in enumerate(LANGUAGE_NAMES)
    }
    sampler = LanguageBalancedSampler(
        DummyDataset(),
        language_ranges,
        DEFAULT_MULTILINGUAL_WEIGHTS,
        torch.Generator().manual_seed(410),
        chunk_size=1024,
    )
    sampled_indices = list(sampler)
    assert len(sampled_indices) == len(DummyDataset())
    observed = Counter(
        language
        for sample in sampled_indices
        for language, (start, count) in language_ranges.items()
        if start <= sample < start + count
    )
    for language, weight in DEFAULT_MULTILINGUAL_WEIGHTS.items():
        assert abs(observed[language] / len(sampled_indices) - weight) < 0.02


def test_training_loop_uses_prepared_language_ranges(tmp_path: Path) -> None:
    class TinyPreparedLikeDataset:
        language_ranges = {
            language: (index, 1) for index, language in enumerate(LANGUAGE_NAMES)
        }

        def __len__(self) -> int:
            return len(self.language_ranges)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            tokens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long) + index
            return tokens[:-1], tokens[1:]

    model_config = ModelConfig(
        vocab_size=260,
        context_length=4,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    training_config = TrainingConfig(
        batch_size=2,
        max_steps=1,
        device="cpu",
        language_sampling_weights=dict(DEFAULT_MULTILINGUAL_WEIGHTS),
    )
    result = train_model(
        model=TransformerLM(model_config),
        dataset=TinyPreparedLikeDataset(),
        model_config=model_config,
        training_config=training_config,
        checkpoint_path=tmp_path / "multilingual-test.pt",
        tokenizer=ByteTokenizer(),
        language_sampling_weights=DEFAULT_MULTILINGUAL_WEIGHTS,
    )

    assert result.steps == 1
    assert result.checkpoint_path.is_file()


def test_multilingual_verifier_checks_language_ranges_and_statistics(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_multilingual_corpus(raw_dir)
    output_dir = tmp_path / "prepared"
    prepare_dataset([raw_dir], output_dir, _multilingual_config())

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_ranges = json.loads(json.dumps(manifest["splits"]["train"]["language_ranges"]))
    first_language = next(iter(original_ranges))
    manifest["splits"]["train"]["language_ranges"][first_language]["start"] += 1
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DatasetVerificationError, match="language ranges are invalid"):
        verify_prepared_dataset(output_dir)

    manifest["splits"]["train"]["language_ranges"] = original_ranges
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    stats_path = output_dir / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["language_sizes"]["hi"]["splits"]["train"]["tokens"] += 1
    stats_path.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(
        DatasetVerificationError,
        match="language 'hi' train tokens differs from metadata",
    ):
        verify_prepared_dataset(output_dir)


def test_multilingual_config_has_requested_cpu_fp32_dimensions() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "multilingual_tiny.toml"
    config = load_config(config_path)

    assert config.model.d_model == 64
    assert config.model.num_layers == 2
    assert config.model.num_heads == 4
    assert config.model.context_length == 128
    assert config.model.vocab_size == 2048
    assert config.tokenizer.kind == "bpe"
    assert config.tokenizer.vocab_size == 2048
    assert config.training.batch_size == 2
    assert config.training.device == "cpu"
    assert config.training.precision == "fp32"
    assert config.training.language_sampling_weights == DEFAULT_MULTILINGUAL_WEIGHTS
    assert config.dataset.require_document_metadata is True


def test_2048_bpe_target_keeps_model_vocab_at_actual_tokenizer_size(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    _write_multilingual_corpus(raw_dir)
    output_dir = tmp_path / "prepared"
    base = _multilingual_config()
    run_config = replace(
        base,
        model=replace(base.model, vocab_size=2048),
        tokenizer=TokenizerConfig(kind="bpe", vocab_size=2048, min_frequency=100_000),
    )

    result = prepare_dataset([raw_dir], output_dir, run_config)
    manifest = result["manifest"]

    assert manifest["tokenizer_config"]["vocab_size"] == 2048
    assert 260 <= result["stats"]["vocabulary_size"] <= 2048
    assert manifest["model_config"]["vocab_size"] == result["stats"]["vocabulary_size"]
    assert manifest["vocab_size"] == result["stats"]["vocabulary_size"]
    assert verify_prepared_dataset(output_dir)["status"] == "ok"


def test_report_command_includes_per_language_and_memory_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from puzzel.data import cli as data_cli

    raw_dir = tmp_path / "raw"
    _write_multilingual_corpus(raw_dir)
    output_dir = tmp_path / "prepared"
    run_config = _multilingual_config()
    prepare_dataset([raw_dir], output_dir, run_config)
    report_path = tmp_path / "reports" / "multilingual.json"

    monkeypatch.setattr(data_cli, "load_config", lambda _path: run_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "puzzel-data",
            "report",
            "--dataset",
            str(output_dir),
            "--config",
            "unused-test-config.toml",
            "--output",
            str(report_path),
        ],
    )
    data_cli.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dataset"]["document_count"] == 18
    assert set(report["dataset"]["language_sizes"]) == set(LANGUAGE_NAMES)
    assert report["dataset"]["duplicate_count"] == 1
    assert report["verification"]["tokenizer_audit"]["unicode_probe_round_trip"] == "ok"
    assert report["hardware_and_memory"]["model"]["parameter_count"] > 0
    assert report["hardware_and_memory"]["model"]["training_memory_estimate"][
        "estimated_training_bytes"
    ] > 0
