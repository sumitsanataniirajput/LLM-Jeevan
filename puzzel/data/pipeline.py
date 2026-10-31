"""Disk-backed local dataset preparation for Puzzel language-model training."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from array import array
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

from puzzel.config import RunConfig
from puzzel.data.prepared import SPLITS, verify_prepared_dataset
from puzzel.data.quality import normalize_text, quality_rejection_reason
from puzzel.languages import LANGUAGE_NAMES, MULTILINGUAL_LANGUAGE_CODES
from puzzel.tokenizer import BPETokenizer, ByteTokenizer, PAD_TOKEN_ID, TextTokenizer

SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".ndjson"}


class DatasetPreparationError(RuntimeError):
    """Raised when source data cannot produce a valid train/validation/test set."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(message)


def prepare_dataset(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    run_config: RunConfig,
) -> dict[str, Any]:
    """Ingest, clean, deduplicate, split, tokenize, and validate local documents.

    Source documents and intermediate BPE token sequences are staged in SQLite.
    Prepared examples are stored as fixed-width, per-document int32 rows so the
    final training data can be memory-mapped without loading the corpus at once.
    The output path may be new or an empty scaffold containing only ``.gitkeep``.
    """
    destination = Path(output_dir).expanduser().resolve()
    replace_empty_destination = False
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise FileExistsError(
                f"output directory already exists: {destination}; choose a new path"
            )
        existing_entries = list(destination.iterdir())
        marker_only = (
            len(existing_entries) == 1
            and existing_entries[0].name == ".gitkeep"
            and existing_entries[0].is_file()
            and not existing_entries[0].is_symlink()
            and existing_entries[0].stat().st_size == 0
        )
        if existing_entries and not marker_only:
            raise FileExistsError(
                f"output directory already exists: {destination}; choose a new path"
            )
        replace_empty_destination = True
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)

    report = _new_ingestion_report()
    source_files = _discover_files(input_paths, report)
    if not source_files:
        raise DatasetPreparationError("no supported input files were found", report)
    report["files_discovered"] = len(source_files)

    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.prepare-", dir=destination.parent
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        output_stage = temporary_root / "prepared"
        output_stage.mkdir()
        database_path = temporary_root / "documents.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _configure_staging_database(connection)
            for source_path in source_files:
                _ingest_file(connection, source_path, run_config, report)
            connection.commit()

            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            if document_count < 3:
                raise DatasetPreparationError(
                    "at least three unique, usable documents are required to create "
                    f"train/validation/test splits (found {document_count})",
                    report,
                )

            language_weights = run_config.training.language_sampling_weights
            if language_weights:
                language_counts = dict(
                    connection.execute(
                        "SELECT language, COUNT(*) FROM documents GROUP BY language"
                    ).fetchall()
                )
                insufficient = {
                    language: language_counts.get(language, 0)
                    for language in language_weights
                    if language_counts.get(language, 0) < 3
                }
                if insufficient:
                    raise DatasetPreparationError(
                        "at least three unique usable documents are required for each "
                        f"language to populate all splits: {insufficient}",
                        report,
                    )

            split_counts = _assign_document_splits(
                connection,
                document_count,
                run_config.dataset,
                language_weights=language_weights,
            )
            report["documents_retained"] = document_count
            report["split_assignment"] = dict(split_counts)

            tokenizer = _train_tokenizer(connection, run_config, temporary_root)
            actual_model_config = replace(
                run_config.model, vocab_size=tokenizer.vocab_size
            )
            if actual_model_config.pad_token_id != PAD_TOKEN_ID:
                raise DatasetPreparationError(
                    "configured pad_token_id does not match the custom tokenizer", report
                )
            tokenizer.save(output_stage / "tokenizer.json")

            split_sizes, language_sizes, language_ranges = _write_token_splits(
                connection,
                output_stage,
                tokenizer,
                actual_model_config.context_length,
                tokenizer.vocab_size,
                PAD_TOKEN_ID,
            )
            _write_metadata_database(connection, output_stage / "documents.sqlite3")

            stats = _build_statistics(
                report,
                split_sizes,
                language_sizes,
                tokenizer.vocab_size,
                run_config.tokenizer.vocab_size,
                run_config.training.language_sampling_weights,
            )
            manifest = _build_manifest(
                run_config,
                actual_model_config,
                tokenizer,
                split_sizes,
                language_ranges,
            )
            _write_json(output_stage / "stats.json", stats)
            _write_json(output_stage / "manifest.json", manifest)
            verification = verify_prepared_dataset(output_stage)
        finally:
            connection.close()

        if replace_empty_destination:
            marker = destination / ".gitkeep"
            if marker.exists():
                marker.unlink()
            destination.rmdir()
        os.replace(output_stage, destination)

    return {
        "dataset_dir": str(destination),
        "stats": stats,
        "manifest": manifest,
        "verification": verification,
    }


def _new_ingestion_report() -> dict[str, Any]:
    return {
        "files_discovered": 0,
        "files_read": 0,
        "unreadable_files": [],
        "missing_inputs": [],
        "unsupported_inputs": [],
        "malformed_jsonl_records": 0,
        "malformed_record_samples": [],
        "metadata_errors": 0,
        "metadata_error_samples": [],
        "documents_seen": 0,
        "documents_retained": 0,
        "empty_documents": 0,
        "duplicate_count": 0,
        "duplicate_count_by_language": {},
        "rejected_documents": {},
    }


def _discover_files(
    input_paths: Sequence[str | Path], report: dict[str, Any]
) -> list[Path]:
    discovered: dict[str, Path] = {}
    for raw_path in input_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            report["missing_inputs"].append(str(path))
            continue
        if path.is_dir():
            try:
                candidates = sorted(item for item in path.rglob("*") if item.is_file())
            except OSError as exc:
                report["unreadable_files"].append(
                    {"path": str(path), "error": str(exc)}
                )
                continue
            for candidate in candidates:
                if candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                    discovered[str(candidate.resolve())] = candidate.resolve()
        elif path.is_file():
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                discovered[str(path.resolve())] = path.resolve()
            else:
                report["unsupported_inputs"].append(str(path))
        else:
            report["missing_inputs"].append(str(path))
    return [discovered[key] for key in sorted(discovered)]


def _configure_staging_database(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE documents ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "digest TEXT NOT NULL UNIQUE, "
        "text TEXT NOT NULL, "
        "language TEXT NOT NULL, "
        "source TEXT NOT NULL, "
        "license TEXT NOT NULL, "
        "characters INTEGER NOT NULL, "
        "utf8_bytes INTEGER NOT NULL, "
        "source_path TEXT NOT NULL, "
        "source_record INTEGER NOT NULL, "
        "split_key TEXT, "
        "split TEXT, "
        "tokens INTEGER NOT NULL DEFAULT 0)"
    )
    connection.execute("CREATE INDEX documents_split_idx ON documents(split, id)")
    connection.commit()


def _ingest_file(
    connection: sqlite3.Connection,
    path: Path,
    run_config: RunConfig,
    report: dict[str, Any],
) -> None:
    local = {
        "files_read": 0,
        "documents_seen": 0,
        "documents_retained": 0,
        "empty_documents": 0,
        "duplicate_count": 0,
        "duplicate_count_by_language": {},
        "malformed_jsonl_records": 0,
        "malformed_record_samples": [],
        "metadata_errors": 0,
        "metadata_error_samples": [],
        "rejected_documents": {},
    }
    connection.execute("SAVEPOINT ingest_file")
    try:
        if path.suffix.lower() == ".txt":
            with path.open("r", encoding="utf-8-sig") as source:
                text = source.read(run_config.dataset.max_document_characters + 1)
            local["documents_seen"] += 1
            if len(text) > run_config.dataset.max_document_characters:
                _increment_rejection(local, "document_too_long")
            elif not text.strip():
                local["empty_documents"] += 1
            else:
                metadata = _txt_document_metadata(path, run_config, local)
                if metadata is not None:
                    _store_document(
                        connection, path, 1, text, local, *metadata
                    )
        else:
            max_line_characters = (
                run_config.dataset.max_document_characters * 6 + 1024
            )
            with path.open("r", encoding="utf-8-sig") as source:
                for line_number, line, too_long in _iter_jsonl_lines(
                    source, max_line_characters
                ):
                    if too_long:
                        local["documents_seen"] += 1
                        _increment_rejection(local, "jsonl_record_too_long")
                        continue
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        local["malformed_jsonl_records"] += 1
                        _append_sample(
                            local["malformed_record_samples"],
                            {"path": str(path), "line": line_number, "error": str(exc)},
                        )
                        continue
                    text = _extract_jsonl_text(
                        record, run_config.dataset.jsonl_text_field
                    )
                    if text is None:
                        local["malformed_jsonl_records"] += 1
                        _append_sample(
                            local["malformed_record_samples"],
                            {
                                "path": str(path),
                                "line": line_number,
                                "error": "record has no configured string text field",
                            },
                        )
                        continue
                    local["documents_seen"] += 1
                    if len(text) > run_config.dataset.max_document_characters:
                        _increment_rejection(local, "document_too_long")
                        continue
                    if not text.strip():
                        local["empty_documents"] += 1
                        continue
                    metadata = _document_metadata(
                        record, path, line_number, run_config, local
                    )
                    if metadata is None:
                        local["malformed_jsonl_records"] += 1
                        continue
                    _store_document(
                        connection, path, line_number, text, local, *metadata
                    )
        local["files_read"] = 1
        connection.execute("RELEASE SAVEPOINT ingest_file")
        connection.commit()
        _merge_ingestion_report(report, local)
    except (OSError, UnicodeError) as exc:
        connection.execute("ROLLBACK TO SAVEPOINT ingest_file")
        connection.execute("RELEASE SAVEPOINT ingest_file")
        connection.commit()
        report["unreadable_files"].append({"path": str(path), "error": str(exc)})


def _iter_jsonl_lines(
    source: Any, max_line_characters: int
) -> Iterator[tuple[int, str, bool]]:
    """Yield bounded JSONL lines, discarding oversized records incrementally."""
    line_number = 0
    while True:
        line = source.readline(max_line_characters + 1)
        if not line:
            return
        line_number += 1
        too_long = len(line) > max_line_characters
        if too_long:
            while line and not line.endswith("\n"):
                line = source.readline(64 * 1024)
        yield line_number, "" if too_long else line, too_long


def _extract_jsonl_text(record: Any, configured_field: str) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for field_name in dict.fromkeys((configured_field, "text", "content", "body")):
        value = record.get(field_name)
        if isinstance(value, str):
            return value
    return None


def _txt_document_metadata(
    path: Path,
    run_config: RunConfig,
    report: dict[str, Any],
) -> tuple[str, str, str] | None:
    sidecar = path.with_name(path.name + ".meta.json")
    if sidecar.exists():
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _metadata_error(report, path, 1, f"invalid TXT metadata sidecar: {exc}")
            return None
    else:
        metadata = {}
    return _document_metadata(metadata, path, 1, run_config, report)


def _document_metadata(
    record: Any,
    path: Path,
    record_number: int,
    run_config: RunConfig,
    report: dict[str, Any],
) -> tuple[str, str, str] | None:
    fields = record if isinstance(record, dict) else {}
    strict = run_config.dataset.require_document_metadata
    inferred_language = _language_from_path(path)
    language_value = fields.get("language")
    source_value = fields.get("source")
    license_value = fields.get("license", fields.get("licence"))

    if language_value is None and not strict:
        language_value = inferred_language
    if not isinstance(language_value, str) or not language_value.strip():
        _metadata_error(report, path, record_number, "language must be a non-empty string")
        return None
    language = language_value.strip().lower()
    if strict and language not in MULTILINGUAL_LANGUAGE_CODES:
        _metadata_error(
            report,
            path,
            record_number,
            f"language must be one of {', '.join(MULTILINGUAL_LANGUAGE_CODES)}",
        )
        return None
    if strict and inferred_language in LANGUAGE_NAMES and language != inferred_language:
        _metadata_error(
            report,
            path,
            record_number,
            f"language {language!r} does not match folder {inferred_language!r}",
        )
        return None

    if source_value is None and not strict:
        source_value = f"{path}:{record_number}"
    if not isinstance(source_value, str) or not source_value.strip():
        _metadata_error(report, path, record_number, "source must be a non-empty string")
        return None

    if license_value is None and not strict:
        license_value = "unspecified"
    if not isinstance(license_value, str) or not license_value.strip():
        _metadata_error(
            report, path, record_number, "license/licence must be a non-empty string"
        )
        return None

    weights = run_config.training.language_sampling_weights
    if weights and language not in weights:
        _metadata_error(
            report,
            path,
            record_number,
            f"language {language!r} is not in the configured sampling target",
        )
        return None
    return language, source_value.strip(), license_value.strip()


def _language_from_path(path: Path) -> str:
    for parent in path.parents:
        if parent.name in LANGUAGE_NAMES:
            return parent.name
    return "und"


def _metadata_error(
    report: dict[str, Any], path: Path, record_number: int, message: str
) -> None:
    report["metadata_errors"] += 1
    _append_sample(
        report["metadata_error_samples"],
        {"path": str(path), "record": record_number, "error": message},
    )


def _store_document(
    connection: sqlite3.Connection,
    source_path: Path,
    source_record: int,
    raw_text: str,
    report: dict[str, Any],
    language: str,
    source: str,
    license_name: str,
) -> None:
    cleaned = normalize_text(raw_text)
    if not cleaned:
        report["empty_documents"] += 1
        return
    reason = quality_rejection_reason(cleaned)
    if reason is not None:
        _increment_rejection(report, reason)
        return
    utf8_bytes = cleaned.encode("utf-8")
    digest = hashlib.sha256(utf8_bytes).hexdigest()
    cursor = connection.execute(
        "INSERT OR IGNORE INTO documents "
        "(digest, text, language, source, license, characters, utf8_bytes, "
        "source_path, source_record) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            digest,
            cleaned,
            language,
            source,
            license_name,
            len(cleaned),
            len(utf8_bytes),
            str(source_path),
            source_record,
        ),
    )
    if cursor.rowcount == 0:
        report["duplicate_count"] += 1
        duplicates = report["duplicate_count_by_language"]
        duplicates[language] = duplicates.get(language, 0) + 1
    else:
        report["documents_retained"] += 1


def _increment_rejection(report: dict[str, Any], reason: str) -> None:
    rejected = report["rejected_documents"]
    rejected[reason] = rejected.get(reason, 0) + 1


def _append_sample(samples: list[dict[str, Any]], item: dict[str, Any]) -> None:
    if len(samples) < 20:
        samples.append(item)


def _merge_ingestion_report(
    report: dict[str, Any], local: dict[str, Any]
) -> None:
    for name in (
        "files_read",
        "documents_seen",
        "documents_retained",
        "empty_documents",
        "duplicate_count",
        "malformed_jsonl_records",
        "metadata_errors",
    ):
        report[name] += local[name]
    for sample_key in ("malformed_record_samples", "metadata_error_samples"):
        report[sample_key].extend(local[sample_key])
        report[sample_key] = report[sample_key][:20]
    for language, count in local["duplicate_count_by_language"].items():
        duplicates = report["duplicate_count_by_language"]
        duplicates[language] = duplicates.get(language, 0) + count
    for reason, count in local["rejected_documents"].items():
        current = report["rejected_documents"].get(reason, 0)
        report["rejected_documents"][reason] = current + count


def _split_document_count(document_count: int, config: Any) -> tuple[int, int]:
    validation_count = max(1, int(document_count * config.validation_fraction))
    test_count = max(1, int(document_count * config.test_fraction))
    while validation_count + test_count >= document_count:
        if test_count > validation_count and test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("split fractions cannot leave a training document")
    return validation_count, test_count


def _assign_document_splits(
    connection: sqlite3.Connection,
    document_count: int,
    config: Any,
    *,
    language_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    if language_weights:
        language_counts = dict(
            connection.execute(
                "SELECT language, COUNT(*) FROM documents GROUP BY language"
            ).fetchall()
        )
        if set(language_counts) != set(language_weights):
            raise ValueError(
                "prepared document languages do not match configured sampling languages"
            )
        for language in sorted(language_weights):
            count = language_counts[language]
            if count < 3:
                raise ValueError(
                    f"language {language!r} needs at least three documents for all splits"
                )
            validation_count, test_count = _split_document_count(count, config)
            last_id = 0
            while True:
                rows = connection.execute(
                    "SELECT id, digest FROM documents WHERE language=? AND id>? "
                    "ORDER BY id LIMIT 512",
                    (language, last_id),
                ).fetchall()
                if not rows:
                    break
                updates = []
                for document_id, digest in rows:
                    key = hashlib.sha256(
                        f"{config.split_seed}:{language}:{digest}".encode("utf-8")
                    ).hexdigest()
                    updates.append((key, document_id))
                    last_id = document_id
                connection.executemany(
                    "UPDATE documents SET split_key=? WHERE id=?", updates
                )
            connection.execute(
                "UPDATE documents SET split='test' WHERE language=? AND id IN "
                "(SELECT id FROM documents WHERE language=? "
                "ORDER BY split_key, id LIMIT ?)",
                (language, language, test_count),
            )
            connection.execute(
                "UPDATE documents SET split='validation' WHERE language=? AND split IS NULL "
                "AND id IN (SELECT id FROM documents WHERE language=? AND split IS NULL "
                "ORDER BY split_key, id LIMIT ?)",
                (language, language, validation_count),
            )
            connection.execute(
                "UPDATE documents SET split='train' WHERE language=? AND split IS NULL",
                (language,),
            )
        connection.commit()
    else:
        validation_count, test_count = _split_document_count(document_count, config)
        last_id = 0
        while True:
            rows = connection.execute(
                "SELECT id, digest FROM documents WHERE id > ? ORDER BY id LIMIT 512",
                (last_id,),
            ).fetchall()
            if not rows:
                break
            updates = []
            for document_id, digest in rows:
                key = hashlib.sha256(
                    f"{config.split_seed}:{digest}".encode("utf-8")
                ).hexdigest()
                updates.append((key, document_id))
                last_id = document_id
            connection.executemany(
                "UPDATE documents SET split_key=? WHERE id=?", updates
            )
        connection.commit()

        connection.execute(
            "UPDATE documents SET split='test' WHERE id IN "
            "(SELECT id FROM documents ORDER BY split_key, id LIMIT ?)",
            (test_count,),
        )
        connection.execute(
            "UPDATE documents SET split='validation' WHERE id IN "
            "(SELECT id FROM documents WHERE split IS NULL "
            "ORDER BY split_key, id LIMIT ?)",
            (validation_count,),
        )
        connection.execute("UPDATE documents SET split='train' WHERE split IS NULL")
        connection.commit()
    return {
        split: connection.execute(
            "SELECT COUNT(*) FROM documents WHERE split=?", (split,)
        ).fetchone()[0]
        for split in SPLITS
    }


def _train_tokenizer(
    connection: sqlite3.Connection,
    run_config: RunConfig,
    temporary_root: Path,
) -> TextTokenizer:
    tokenizer_config = run_config.tokenizer
    if tokenizer_config.kind == "byte":
        return ByteTokenizer()

    def training_documents() -> Iterator[str]:
        cursor = connection.execute(
            "SELECT text FROM documents WHERE split='train' ORDER BY id"
        )
        for (text,) in cursor:
            yield text

    return BPETokenizer.train_streaming(
        training_documents,
        target_vocab_size=tokenizer_config.vocab_size,
        min_frequency=tokenizer_config.min_frequency,
        temp_dir=temporary_root,
    )


def _empty_size_counts() -> dict[str, int]:
    return {
        "documents": 0,
        "characters": 0,
        "utf8_bytes": 0,
        "tokens": 0,
        "examples": 0,
    }


def _write_token_splits(
    connection: sqlite3.Connection,
    output_dir: Path,
    tokenizer: TextTokenizer,
    context_length: int,
    vocab_size: int,
    pad_token_id: int,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, dict[str, int]]],
]:
    split_sizes = {split: _empty_size_counts() for split in SPLITS}
    language_sizes: dict[str, dict[str, Any]] = {}
    language_ranges: dict[str, dict[str, dict[str, int]]] = {
        split: {} for split in SPLITS
    }
    files = {
        split: (output_dir / f"{split}.bin").open("wb") for split in SPLITS
    }
    cursor = connection.execute(
        "SELECT id, split, text, language, characters, utf8_bytes "
        "FROM documents ORDER BY language, id"
    )
    try:
        while True:
            rows = cursor.fetchmany(256)
            if not rows:
                break
            for document_id, split, text, language, characters, utf8_bytes in rows:
                token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(token_ids) < 2:
                    raise ValueError(
                        f"document {document_id} encoded to an invalid sequence"
                    )
                if any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
                    raise ValueError(
                        f"document {document_id} encoded an out-of-range token ID"
                    )
                if tokenizer.decode(token_ids) != text:
                    raise ValueError(
                        f"tokenizer round-trip failed for document {document_id}"
                    )

                transitions = len(token_ids) - 1
                example_count = 0
                for start in range(0, transitions, context_length):
                    row = token_ids[start : start + context_length + 1]
                    if len(row) < context_length + 1:
                        row.extend([pad_token_id] * (context_length + 1 - len(row)))
                    packed = array("i", row)
                    if packed.itemsize != 4:
                        raise RuntimeError("platform does not provide 32-bit integers")
                    if sys.byteorder != "little":
                        packed.byteswap()
                    files[split].write(packed.tobytes())
                    example_count += 1

                counts = split_sizes[split]
                counts["documents"] += 1
                counts["characters"] += characters
                counts["utf8_bytes"] += utf8_bytes
                counts["tokens"] += len(token_ids)
                range_info = language_ranges[split].setdefault(
                    language,
                    {"start": counts["examples"], "examples": 0},
                )
                range_info["examples"] += example_count
                counts["examples"] += example_count

                per_language = language_sizes.setdefault(
                    language,
                    {"splits": {name: _empty_size_counts() for name in SPLITS}},
                )
                language_counts = per_language["splits"][split]
                language_counts["documents"] += 1
                language_counts["characters"] += characters
                language_counts["utf8_bytes"] += utf8_bytes
                language_counts["tokens"] += len(token_ids)
                language_counts["examples"] += example_count
                connection.execute(
                    "UPDATE documents SET tokens=? WHERE id=?",
                    (len(token_ids), document_id),
                )
        connection.commit()
    finally:
        for file_handle in files.values():
            file_handle.close()

    for stats in language_sizes.values():
        totals = _empty_size_counts()
        for split_counts in stats["splits"].values():
            for key in totals:
                totals[key] += split_counts[key]
        stats.update(totals)
    return split_sizes, language_sizes, language_ranges


def _write_metadata_database(source: sqlite3.Connection, destination: Path) -> None:
    with sqlite3.connect(destination) as metadata:
        metadata.execute("PRAGMA journal_mode=OFF")
        metadata.execute("PRAGMA synchronous=OFF")
        metadata.execute(
            "CREATE TABLE documents ("
            "document_id INTEGER NOT NULL, digest TEXT NOT NULL, split TEXT NOT NULL, "
            "language TEXT NOT NULL, source TEXT NOT NULL, license TEXT NOT NULL, "
            "characters INTEGER NOT NULL, utf8_bytes INTEGER NOT NULL, "
            "tokens INTEGER NOT NULL)"
        )
        metadata.execute("CREATE INDEX documents_digest_idx ON documents(digest)")
        metadata.execute("CREATE INDEX documents_split_idx ON documents(split)")
        metadata.execute("CREATE INDEX documents_language_idx ON documents(language)")
        cursor = source.execute(
            "SELECT id, digest, split, language, source, license, characters, "
            "utf8_bytes, tokens FROM documents ORDER BY id"
        )
        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break
            metadata.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )


def _build_statistics(
    ingestion_report: dict[str, Any],
    split_sizes: dict[str, dict[str, int]],
    language_sizes: dict[str, dict[str, Any]],
    vocab_size: int,
    vocabulary_target: int,
    language_sampling_weights: dict[str, float],
) -> dict[str, Any]:
    return {
        "format": "puzzel-dataset-statistics-v2",
        "document_count": sum(item["documents"] for item in split_sizes.values()),
        "total_characters": sum(item["characters"] for item in split_sizes.values()),
        "total_utf8_bytes": sum(item["utf8_bytes"] for item in split_sizes.values()),
        "total_tokens": sum(item["tokens"] for item in split_sizes.values()),
        "vocabulary_size": vocab_size,
        "vocabulary_statistics": {
            "target_size": vocabulary_target,
            "actual_size": vocab_size,
            "byte_fallback_tokens": 256,
            "special_token_count": 4,
            "learned_merge_count": max(0, vocab_size - 260),
            "maximum_merge_count": max(0, vocabulary_target - 260),
        },
        "duplicate_count": ingestion_report["duplicate_count"],
        "duplicate_count_by_language": ingestion_report["duplicate_count_by_language"],
        "language_sampling_weights": language_sampling_weights,
        "language_sizes": language_sizes,
        "split_sizes": split_sizes,
        "ingestion": ingestion_report,
    }


def _build_manifest(
    run_config: RunConfig,
    model_config: Any,
    tokenizer: TextTokenizer,
    split_sizes: dict[str, dict[str, int]],
    language_ranges: dict[str, dict[str, dict[str, int]]],
) -> dict[str, Any]:
    return {
        "format": "puzzel-prepared-dataset-v2",
        "context_length": model_config.context_length,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": model_config.pad_token_id,
        "tokenizer_file": "tokenizer.json",
        "tokenizer_training_split": "train",
        "metadata_file": "documents.sqlite3",
        "token_dtype": "int32",
        "endianness": sys.byteorder,
        "model_config": asdict(model_config),
        "tokenizer_config": asdict(run_config.tokenizer),
        "dataset_config": asdict(run_config.dataset),
        "language_sampling_weights": run_config.training.language_sampling_weights,
        "splits": {
            split: {
                "file": f"{split}.bin",
                "language_ranges": language_ranges[split],
                **split_sizes[split],
            }
            for split in SPLITS
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
