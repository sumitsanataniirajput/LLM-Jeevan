"""Memory-mapped readers and integrity checks for prepared token datasets."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from puzzel.languages import MULTILINGUAL_LANGUAGE_CODES
from puzzel.tokenizer import (
    BOS_TOKEN_ID,
    BYTE_VOCAB_SIZE,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS,
    ByteTokenizer,
    tokenizer_from_dict,
)

SPLITS = ("train", "validation", "test")


class DatasetVerificationError(ValueError):
    """Raised when a prepared dataset fails one or more integrity checks."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("prepared dataset verification failed: " + "; ".join(errors))


class PreparedSplitDataset(Dataset[tuple[Tensor, Tensor]]):
    """Lazy memory-mapped next-token examples from one prepared split."""

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        *,
        context_length: int | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        if split not in SPLITS:
            raise ValueError(f"split must be one of {', '.join(SPLITS)}")
        manifest = _read_json(self.dataset_dir / "manifest.json")
        if manifest.get("format") not in {
            "puzzel-prepared-dataset-v1",
            "puzzel-prepared-dataset-v2",
        }:
            raise ValueError("unsupported prepared dataset format")
        if manifest.get("endianness") != sys.byteorder:
            raise ValueError("prepared token file byte order differs from this machine")
        stored_context_length = int(manifest["context_length"])
        if context_length is None:
            self.context_length = stored_context_length
        elif (
            not isinstance(context_length, int)
            or isinstance(context_length, bool)
            or not 0 < context_length <= stored_context_length
        ):
            raise ValueError(
                "requested context_length must be positive and no greater than "
                "the prepared dataset context"
            )
        else:
            self.context_length = context_length
        self.pad_token_id = int(manifest["pad_token_id"])
        split_info = manifest["splits"][split]
        raw_language_ranges = split_info.get("language_ranges", {})
        if not isinstance(raw_language_ranges, dict):
            raise ValueError("prepared language_ranges metadata is malformed")
        self.language_ranges: dict[str, tuple[int, int]] = {}
        for language, range_info in raw_language_ranges.items():
            if not isinstance(language, str) or not isinstance(range_info, dict):
                raise ValueError("prepared language_ranges metadata is malformed")
            start = range_info.get("start")
            count = range_info.get("examples")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or start < 0
                or count <= 0
            ):
                raise ValueError("prepared language_ranges values are invalid")
            self.language_ranges[language] = (start, count)
        self.examples = int(split_info["examples"])
        self.split = split
        if self.language_ranges:
            next_example = 0
            for start, count in sorted(self.language_ranges.values()):
                if start != next_example:
                    raise ValueError("prepared language_ranges are not contiguous")
                next_example += count
            if next_example != self.examples:
                raise ValueError("prepared language_ranges do not cover the split")
        token_width = stored_context_length + 1
        token_count = self.examples * token_width
        token_path = self.dataset_dir / split_info["file"]
        expected_bytes = token_count * 4
        if self.examples <= 0 or token_path.stat().st_size != expected_bytes:
            raise ValueError(f"invalid token file size for the {split} split")
        self._storage = torch.from_file(
            str(token_path),
            shared=False,
            size=token_count,
            dtype=torch.int32,
        ).reshape(self.examples, token_width)

    def __len__(self) -> int:
        return self.examples

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= self.examples:
            raise IndexError(index)
        tokens = self._storage[index, : self.context_length + 1].to(dtype=torch.long)
        return tokens[:-1], tokens[1:]


def verify_prepared_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate schema, tokenizer, split files, token IDs, and exact leakage."""
    root = Path(dataset_dir)
    errors: list[str] = []
    try:
        manifest = _read_json(root / "manifest.json")
        stats = _read_json(root / "stats.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetVerificationError([f"cannot read manifest/statistics: {exc}"]) from exc

    if manifest.get("format") not in {
        "puzzel-prepared-dataset-v1",
        "puzzel-prepared-dataset-v2",
    }:
        errors.append("unsupported manifest format")
    if manifest.get("tokenizer_training_split") != "train":
        errors.append("tokenizer must be trained on the training split only")
    raw_vocab_size = manifest.get("vocab_size")
    raw_context_length = manifest.get("context_length")
    raw_pad_token_id = manifest.get("pad_token_id")
    valid_vocab_size = (
        isinstance(raw_vocab_size, int)
        and not isinstance(raw_vocab_size, bool)
        and raw_vocab_size > 0
    )
    valid_context_length = (
        isinstance(raw_context_length, int)
        and not isinstance(raw_context_length, bool)
        and raw_context_length > 0
    )
    valid_pad_token_id = (
        isinstance(raw_pad_token_id, int)
        and not isinstance(raw_pad_token_id, bool)
        and valid_vocab_size
        and 0 <= raw_pad_token_id < raw_vocab_size
    )
    if not valid_vocab_size:
        errors.append("manifest vocabulary size is invalid")
    if not valid_context_length:
        errors.append("manifest context length is invalid")
    if not valid_pad_token_id:
        errors.append("manifest pad token ID is invalid")
    vocab_size = raw_vocab_size if valid_vocab_size else 1
    context_length = raw_context_length if valid_context_length else 1
    pad_token_id = raw_pad_token_id if valid_pad_token_id else 0
    if manifest.get("endianness") != sys.byteorder:
        errors.append("token file byte order differs from this machine")
    if manifest.get("token_dtype") != "int32":
        errors.append("unsupported token file dtype")
    model_config = manifest.get("model_config")
    if not isinstance(model_config, dict):
        errors.append("manifest model configuration is missing")
    else:
        if model_config.get("vocab_size") != vocab_size:
            errors.append("model and manifest vocabulary sizes differ")
        if model_config.get("context_length") != context_length:
            errors.append("model and manifest context lengths differ")
        if model_config.get("pad_token_id") != pad_token_id:
            errors.append("model and manifest padding IDs differ")

    manifest_tokenizer_config = manifest.get("tokenizer_config")
    if manifest.get("format") == "puzzel-prepared-dataset-v2":
        if not isinstance(manifest_tokenizer_config, dict):
            errors.append("v2 tokenizer configuration is missing or malformed")
        elif (
            manifest_tokenizer_config.get("kind") not in {"byte", "bpe"}
            or not isinstance(manifest_tokenizer_config.get("vocab_size"), int)
            or isinstance(manifest_tokenizer_config.get("vocab_size"), bool)
        ):
            errors.append("v2 tokenizer configuration is invalid")

    manifest_weights = manifest.get("language_sampling_weights", {})
    if not isinstance(manifest_weights, dict):
        errors.append("manifest language sampling weights are malformed")
        manifest_weights = {}
    if manifest.get("format") == "puzzel-prepared-dataset-v2":
        if manifest_weights:
            if set(manifest_weights) != set(MULTILINGUAL_LANGUAGE_CODES):
                errors.append("manifest sampling weights do not cover the configured languages")
            weight_values = list(manifest_weights.values())
            if (
                any(
                    not isinstance(weight, (int, float))
                    or isinstance(weight, bool)
                    or weight <= 0
                    for weight in weight_values
                )
                or not all(isinstance(weight, (int, float)) for weight in weight_values)
                or not abs(sum(weight_values) - 1.0) <= 1e-9
            ):
                errors.append("manifest language sampling weights are invalid")
            dataset_config = manifest.get("dataset_config")
            if not isinstance(dataset_config, dict) or dataset_config.get(
                "require_document_metadata"
            ) is not True:
                errors.append("multilingual weights require document metadata")
            if stats.get("language_sampling_weights") != manifest_weights:
                errors.append("statistics and manifest sampling weights differ")

    tokenizer = None
    tokenizer_audit: dict[str, Any] = {}
    try:
        tokenizer_path = root / manifest.get("tokenizer_file", "tokenizer.json")
        tokenizer_payload = _read_json(tokenizer_path)
        tokenizer = tokenizer_from_dict(tokenizer_payload)
        if tokenizer.vocab_size != vocab_size:
            errors.append("tokenizer and manifest vocabulary sizes differ")
        if tokenizer_payload.get("special_tokens") != SPECIAL_TOKENS:
            errors.append("tokenizer special-token IDs do not match the reserved IDs")
        probe = "Puzzel audit — नमस्ते ¡Hola! Bonjour مرحبًا 中文。"
        byte_ids = ByteTokenizer().encode(probe)
        if any(token_id < 0 or token_id >= BYTE_VOCAB_SIZE for token_id in byte_ids):
            errors.append("byte fallback emitted an ID outside the 0-255 byte range")
        elif ByteTokenizer().decode(byte_ids) != probe:
            errors.append("UTF-8 byte fallback round-trip failed")
        marked_ids = tokenizer.encode(probe, add_bos=True, add_eos=True)
        if (
            not marked_ids
            or marked_ids[0] != BOS_TOKEN_ID
            or marked_ids[-1] != EOS_TOKEN_ID
            or tokenizer.decode(marked_ids) != probe
        ):
            errors.append("tokenizer Unicode round-trip or BOS/EOS audit failed")
        special_ids = list(SPECIAL_TOKENS.values())
        special_text = "".join(SPECIAL_TOKENS)
        if (
            tokenizer.decode(special_ids) != ""
            or tokenizer.decode(special_ids, skip_special_tokens=False) != special_text
        ):
            errors.append("tokenizer special-token decoding audit failed")
        tokenizer_audit = {
            "byte_fallback_ids": f"0-{BYTE_VOCAB_SIZE - 1}",
            "byte_fallback_round_trip": "ok",
            "unicode_probe_round_trip": "ok",
            "special_tokens": SPECIAL_TOKENS,
            "bos_eos_round_trip": "ok",
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        errors.append(f"tokenizer could not be verified: {exc}")

    metadata_path = root / manifest.get("metadata_file", "documents.sqlite3")
    metadata_counts: dict[str, dict[str, int]] = {}
    metadata_by_language: dict[str, dict[str, dict[str, int]]] = {}
    has_language_metadata = False
    try:
        metadata_uri = f"{metadata_path.resolve().as_uri()}?mode=ro"
        with sqlite3.connect(metadata_uri, uri=True) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(documents)")
            }
            has_language_metadata = "language" in columns
            if manifest.get("format") == "puzzel-prepared-dataset-v2":
                required_columns = {
                    "document_id",
                    "digest",
                    "split",
                    "language",
                    "source",
                    "license",
                    "characters",
                    "utf8_bytes",
                    "tokens",
                }
                if not required_columns.issubset(columns):
                    errors.append("v2 document metadata schema is incomplete")
            byte_sum = "COALESCE(SUM(utf8_bytes), 0)" if "utf8_bytes" in columns else "0"
            rows = connection.execute(
                "SELECT split, COUNT(*), COALESCE(SUM(characters), 0), "
                f"{byte_sum}, COALESCE(SUM(tokens), 0) FROM documents GROUP BY split"
            ).fetchall()
            metadata_counts = {
                split: {
                    "documents": docs,
                    "characters": chars,
                    "utf8_bytes": byte_count,
                    "tokens": tokens,
                }
                for split, docs, chars, byte_count, tokens in rows
            }
            if has_language_metadata and {
                "characters",
                "utf8_bytes",
                "tokens",
            }.issubset(columns):
                language_rows = connection.execute(
                    "SELECT language, split, COUNT(*), COALESCE(SUM(characters), 0), "
                    "COALESCE(SUM(utf8_bytes), 0), COALESCE(SUM(tokens), 0) "
                    "FROM documents GROUP BY language, split"
                ).fetchall()
                for language, split, docs, chars, byte_count, tokens in language_rows:
                    metadata_by_language.setdefault(language, {})[split] = {
                        "documents": docs,
                        "characters": chars,
                        "utf8_bytes": byte_count,
                        "tokens": tokens,
                    }
                if {"source", "license"}.issubset(columns):
                    incomplete_provenance = connection.execute(
                        "SELECT document_id FROM documents WHERE "
                        "TRIM(language)='' OR TRIM(source)='' OR TRIM(license)='' LIMIT 1"
                    ).fetchone()
                    if incomplete_provenance:
                        errors.append("document metadata contains an empty language/source/license")
            if not metadata_counts:
                errors.append("document metadata contains no documents")
            for split in SPLITS:
                if split not in metadata_counts:
                    errors.append(f"document metadata is missing the {split} split")
            invalid_split = connection.execute(
                "SELECT split FROM documents WHERE split NOT IN (?, ?, ?) LIMIT 1",
                SPLITS,
            ).fetchone()
            if invalid_split:
                errors.append(f"metadata contains unknown split {invalid_split[0]!r}")
            leaked_hash = connection.execute(
                "SELECT digest FROM documents GROUP BY digest "
                "HAVING COUNT(DISTINCT split) > 1 LIMIT 1"
            ).fetchone()
            repeated_hash = connection.execute(
                "SELECT digest FROM documents GROUP BY digest HAVING COUNT(*) > 1 LIMIT 1"
            ).fetchone()
            if leaked_hash:
                errors.append("exact document hash appears in more than one split")
            if repeated_hash:
                errors.append("duplicate document hashes remain in prepared metadata")
    except (OSError, sqlite3.Error) as exc:
        errors.append(f"document metadata database could not be verified: {exc}")

    split_stats = stats.get("split_sizes", {})
    if not isinstance(split_stats, dict):
        errors.append("split statistics are malformed")
        split_stats = {}
    split_manifest = manifest.get("splits", {})
    if not isinstance(split_manifest, dict):
        errors.append("manifest split metadata is malformed")
        split_manifest = {}
    total_examples = 0
    verified_language_ranges: dict[str, dict[str, dict[str, int]]] = {}
    for split in SPLITS:
        info = split_manifest.get(split)
        if not isinstance(info, dict):
            errors.append(f"manifest is missing the {split} split")
            continue
        try:
            examples = info.get("examples")
            if (
                not isinstance(examples, int)
                or isinstance(examples, bool)
                or examples <= 0
            ):
                errors.append(f"{split} example count is invalid")
                continue
            raw_ranges = info.get("language_ranges")
            if manifest.get("format") == "puzzel-prepared-dataset-v2":
                if not isinstance(raw_ranges, dict):
                    errors.append(f"{split} language ranges are missing or malformed")
                else:
                    ordered_ranges = sorted(
                        raw_ranges.items(),
                        key=lambda item: item[1].get("start", -1)
                        if isinstance(item[1], dict)
                        else -1,
                    )
                    expected_start = 0
                    validated_ranges: dict[str, dict[str, int]] = {}
                    for language, range_info in ordered_ranges:
                        if not isinstance(language, str) or not isinstance(range_info, dict):
                            errors.append(f"{split} language range entry is malformed")
                            continue
                        start = range_info.get("start")
                        count = range_info.get("examples")
                        if (
                            not isinstance(start, int)
                            or isinstance(start, bool)
                            or not isinstance(count, int)
                            or isinstance(count, bool)
                            or start != expected_start
                            or count <= 0
                        ):
                            errors.append(f"{split} language ranges are invalid or non-contiguous")
                            continue
                        validated_ranges[language] = {"start": start, "examples": count}
                        expected_start += count
                    if expected_start != examples:
                        errors.append(f"{split} language ranges do not cover all examples")
                    target_languages = set(manifest_weights)
                    if target_languages and set(validated_ranges) != target_languages:
                        errors.append(f"{split} does not contain every configured target language")
                    verified_language_ranges[split] = validated_ranges
            token_file = root / info["file"]
            expected_bytes = examples * (context_length + 1) * 4
            if examples <= 0:
                errors.append(f"{split} split has no examples")
                continue
            if token_file.stat().st_size != expected_bytes:
                errors.append(f"{split} token file has an unexpected size")
                continue
            storage = torch.from_file(
                str(token_file),
                shared=False,
                size=examples * (context_length + 1),
                dtype=torch.int32,
            ).reshape(examples, context_length + 1)
            for offset in range(0, examples, 4096):
                rows = storage[offset : offset + 4096]
                if rows.numel() == 0:
                    continue
                if rows.min().item() < 0 or rows.max().item() >= vocab_size:
                    errors.append(f"{split} contains a token ID outside the vocabulary")
                    break
                if torch.any(torch.all(rows[:, :-1] == pad_token_id, dim=1)):
                    errors.append(f"{split} contains an empty input sequence")
                    break
                if torch.any(torch.all(rows[:, 1:] == pad_token_id, dim=1)):
                    errors.append(f"{split} contains an empty target sequence")
                    break
            total_examples += examples
            reported = split_stats.get(split, {})
            if not isinstance(reported, dict):
                errors.append(f"{split} statistics are malformed")
                continue
            for key in ("documents", "characters", "utf8_bytes", "tokens", "examples"):
                if reported.get(key) != info.get(key):
                    errors.append(f"{split} {key} count differs between manifest and statistics")
                if (
                    key != "examples"
                    and key in reported
                    and split in metadata_counts
                    and reported.get(key) != metadata_counts[split].get(key)
                ):
                    errors.append(f"{split} {key} count differs from metadata")
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(f"{split} token data could not be verified: {exc}")

    if sum(item["documents"] for item in metadata_counts.values()) != stats.get(
        "document_count"
    ):
        errors.append("document count differs from metadata")
    if sum(item["characters"] for item in metadata_counts.values()) != stats.get(
        "total_characters"
    ):
        errors.append("total character count differs from metadata")
    if sum(item["tokens"] for item in metadata_counts.values()) != stats.get("total_tokens"):
        errors.append("total token count differs from metadata")
    if "total_utf8_bytes" in stats and sum(
        item["utf8_bytes"] for item in metadata_counts.values()
    ) != stats.get("total_utf8_bytes"):
        errors.append("total UTF-8 byte count differs from metadata")
    if stats.get("vocabulary_size") != vocab_size:
        errors.append("statistics and manifest vocabulary sizes differ")

    vocabulary_statistics = stats.get("vocabulary_statistics")
    if vocabulary_statistics is not None:
        if not isinstance(vocabulary_statistics, dict) or vocabulary_statistics.get(
            "actual_size"
        ) != vocab_size:
            errors.append("vocabulary statistics do not match the manifest")
        else:
            target_size = vocabulary_statistics.get("target_size")
            tokenizer_config = manifest.get("tokenizer_config")
            valid_target = (
                isinstance(target_size, int)
                and not isinstance(target_size, bool)
                and target_size >= vocab_size
            )
            if (
                not valid_target
                or vocabulary_statistics.get("byte_fallback_tokens") != BYTE_VOCAB_SIZE
                or vocabulary_statistics.get("special_token_count") != len(SPECIAL_TOKENS)
            ):
                errors.append("vocabulary target or byte/special-token audit is invalid")
            elif isinstance(tokenizer_config, dict) and target_size != tokenizer_config.get(
                "vocab_size"
            ):
                errors.append("vocabulary target differs from the tokenizer config")
            elif isinstance(target_size, int) and not isinstance(target_size, bool):
                if vocabulary_statistics.get("learned_merge_count") != max(
                    0, vocab_size - BYTE_VOCAB_SIZE - len(SPECIAL_TOKENS)
                ) or vocabulary_statistics.get("maximum_merge_count") != max(
                    0, target_size - BYTE_VOCAB_SIZE - len(SPECIAL_TOKENS)
                ):
                    errors.append("BPE merge statistics are inconsistent with the vocabulary")

    if has_language_metadata:
        reported_languages = stats.get("language_sizes")
        if not isinstance(reported_languages, dict):
            errors.append("per-language statistics are missing or malformed")
        else:
            if set(reported_languages) != set(metadata_by_language):
                errors.append("per-language statistics differ from document metadata")
            configured_languages = set(manifest_weights)
            if configured_languages and set(reported_languages) != configured_languages:
                errors.append("per-language statistics omit a configured language")
            for language, details in reported_languages.items():
                if not isinstance(details, dict) or not isinstance(details.get("splits"), dict):
                    errors.append(f"per-language statistics for {language!r} are malformed")
                    continue
                aggregate = {
                    key: 0
                    for key in (
                        "documents",
                        "characters",
                        "utf8_bytes",
                        "tokens",
                        "examples",
                    )
                }
                for split in SPLITS:
                    reported_split = details["splits"].get(split)
                    actual_split = metadata_by_language.get(language, {}).get(split)
                    if not isinstance(reported_split, dict) or actual_split is None:
                        errors.append(
                            f"language {language!r} is missing the {split} split"
                        )
                        continue
                    for key in ("documents", "characters", "utf8_bytes", "tokens"):
                        if reported_split.get(key) != actual_split.get(key):
                            errors.append(
                                f"language {language!r} {split} {key} differs from metadata"
                            )
                    range_info = verified_language_ranges.get(split, {}).get(language)
                    if (
                        range_info is not None
                        and reported_split.get("examples") != range_info["examples"]
                    ):
                        errors.append(
                            f"language {language!r} {split} examples differ from the manifest"
                        )
                    for key in aggregate:
                        value = reported_split.get(key, 0)
                        if (
                            not isinstance(value, int)
                            or isinstance(value, bool)
                            or value < 0
                        ):
                            errors.append(
                                f"language {language!r} {split} {key} is invalid"
                            )
                        else:
                            aggregate[key] += value
                for key, total in aggregate.items():
                    if details.get(key) != total:
                        errors.append(
                            f"language {language!r} total {key} differs from its splits"
                        )

    duplicate_count = stats.get("duplicate_count")
    duplicate_by_language = stats.get("duplicate_count_by_language")
    if duplicate_count is not None and (
        not isinstance(duplicate_count, int)
        or isinstance(duplicate_count, bool)
        or duplicate_count < 0
    ):
        errors.append("duplicate count is invalid")
    if duplicate_by_language is not None:
        if not isinstance(duplicate_by_language, dict) or any(
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for count in duplicate_by_language.values()
        ):
            errors.append("per-language duplicate counts are malformed")
        elif isinstance(duplicate_count, int) and sum(
            duplicate_by_language.values()
        ) != duplicate_count:
            errors.append("per-language duplicate counts do not match the total")

    if errors:
        raise DatasetVerificationError(errors)
    return {
        "status": "ok",
        "dataset_dir": str(root),
        "document_count": stats["document_count"],
        "vocabulary_size": vocab_size,
        "split_sizes": split_stats,
        "total_examples": total_examples,
        "tokenizer_audit": tokenizer_audit,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value
