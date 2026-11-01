"""Project-owned UTF-8 byte and learned byte-pair tokenizers.

The byte tokenizer is a deterministic fallback. ``BPETokenizer.train`` learns
merge rules only from caller-supplied local text; no external vocabulary or
pretrained tokenizer is downloaded.

Extended BPE tokenizer supports additional special tokens (language IDs,
mask, separator) for multilingual and instruction-following use cases.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import sys
import tempfile
from array import array
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

BYTE_VOCAB_SIZE = 256
BOS_TOKEN_ID = 256
EOS_TOKEN_ID = 257
PAD_TOKEN_ID = 258
UNK_TOKEN_ID = 259
VOCAB_SIZE = 260

# Extended special token IDs (start after the 4 reserved IDs)
MASK_TOKEN_ID = 260
SEP_TOKEN_ID = 261

# Language-ID token IDs (start after mask/sep)
LANG_EN_TOKEN_ID = 262
LANG_HI_TOKEN_ID = 263
LANG_ES_TOKEN_ID = 264
LANG_FR_TOKEN_ID = 265
LANG_AR_TOKEN_ID = 266
LANG_ZH_TOKEN_ID = 267

LANGUAGE_ID_TOKENS: dict[str, int] = {
    "en": LANG_EN_TOKEN_ID,
    "hi": LANG_HI_TOKEN_ID,
    "es": LANG_ES_TOKEN_ID,
    "fr": LANG_FR_TOKEN_ID,
    "ar": LANG_AR_TOKEN_ID,
    "zh": LANG_ZH_TOKEN_ID,
}

ID_TO_LANGUAGE = {value: key for key, value in LANGUAGE_ID_TOKENS.items()}

# Base special tokens (always present)
BASE_SPECIAL_TOKENS: dict[str, int] = {
    "<|bos|>": BOS_TOKEN_ID,
    "<|eos|>": EOS_TOKEN_ID,
    "<|pad|>": PAD_TOKEN_ID,
    "<|unk|>": UNK_TOKEN_ID,
}

# Backward-compatible alias
SPECIAL_TOKENS = BASE_SPECIAL_TOKENS

# Extended special tokens (optional)
EXTENDED_SPECIAL_TOKENS: dict[str, int] = {
    "<|mask|>": MASK_TOKEN_ID,
    "<|sep|>": SEP_TOKEN_ID,
}

BYTE_FORMAT_NAME = "puzzel-utf8-byte-tokenizer-v1"
BPE_FORMAT_NAME = "puzzel-byte-bpe-tokenizer-v1"
BPE_EXTENDED_FORMAT_NAME = "puzzle-byte-bpe-tokenizer-v2"


class TextTokenizer(Protocol):
    """Shared interface expected by the text data and checkpoint pipelines."""

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]: ...

    def decode(
        self, token_ids: Iterable[int], *, skip_special_tokens: bool = True
    ) -> str: ...

    def to_dict(self) -> dict[str, Any]: ...

    def save(self, path: str | Path) -> None: ...


class ByteTokenizer:
    """Lossless UTF-8 byte tokenizer with four reserved special IDs."""

    @property
    def vocab_size(self) -> int:
        return 260

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        token_ids = [BOS_TOKEN_ID] if add_bos else []
        token_ids.extend(text.encode("utf-8"))
        if add_eos:
            token_ids.append(EOS_TOKEN_ID)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        output: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                output.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for raw_id in token_ids:
            token_id = _as_token_id(raw_id)
            if 0 <= token_id < BYTE_VOCAB_SIZE:
                pending_bytes.append(token_id)
            elif token_id in {BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID}:
                flush_bytes()
                if not skip_special_tokens:
                    id_to_special = {
                        BOS_TOKEN_ID: "<|bos|>",
                        EOS_TOKEN_ID: "<|eos|>",
                        PAD_TOKEN_ID: "<|pad|>",
                        UNK_TOKEN_ID: "<|unk|>",
                    }
                    output.append(id_to_special[token_id])
            else:
                raise ValueError(f"token ID {token_id} is outside [0, 260)")
        flush_bytes()
        return "".join(output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": BYTE_FORMAT_NAME,
            "vocab_size": 260,
            "special_tokens": dict(BASE_SPECIAL_TOKENS),
            "byte_encoding": "UTF-8",
            "version": 1,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ByteTokenizer":
        if payload.get("format") != BYTE_FORMAT_NAME:
            raise ValueError("unsupported byte tokenizer format")
        if payload.get("vocab_size") != 260:
            raise ValueError("saved byte tokenizer vocabulary size does not match")
        return cls()

    def save(self, path: str | Path) -> None:
        _write_tokenizer_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        return cls.from_dict(_read_tokenizer_json(path))


class BPETokenizer:
    """Byte-level BPE tokenizer with optional extended special tokens."""

    def __init__(
        self,
        merges: Sequence[tuple[int, int]] = (),
        *,
        extra_special_tokens: dict[str, int] | None = None,
        _legacy_base_vocab_size: int | None = None,
    ) -> None:
        self.merges = tuple((int(left), int(right)) for left, right in merges)
        self.extra_special_tokens: dict[str, int] = dict(
            extra_special_tokens or {}
        )
        self._legacy_base_vocab_size = _legacy_base_vocab_size

        all_special = {**BASE_SPECIAL_TOKENS, **self.extra_special_tokens}
        if len(all_special) != len(set(all_special.values())):
            raise ValueError("extra special token IDs conflict with base IDs")
        for name, tid in self.extra_special_tokens.items():
            if tid < 260:
                raise ValueError(
                    f"extra special token {name!r} ID {tid} must be >= 260"
                )

        token_bytes: dict[int, bytes] = {tid: bytes((tid,)) for tid in range(256)}
        seen_pairs: set[tuple[int, int]] = set()
        next_merge_id = self._base_vocab_size()
        for merge_index, pair in enumerate(self.merges):
            left, right = pair
            if pair in seen_pairs:
                raise ValueError(f"duplicate BPE merge rule: {pair}")
            if left not in token_bytes or right not in token_bytes:
                raise ValueError(
                    f"merge {merge_index} refers to a token not yet in the vocabulary"
                )
            new_id = next_merge_id + merge_index
            token_bytes[new_id] = token_bytes[left] + token_bytes[right]
            seen_pairs.add(pair)
        self._token_bytes = token_bytes
        self._merge_pairs = {
            pair: next_merge_id + index for index, pair in enumerate(self.merges)
        }

    def _base_vocab_size(self) -> int:
        # Reserve space for bytes (0-255), base specials (256-259), and any
        # caller-provided extra special tokens. BPE merge IDs start after this.
        # Extended tokens (mask/sep/language IDs) only occupy IDs when they are
        # actually included in extra_special_tokens, preserving v1 backward
        # compatibility where merges start at 260.
        # _legacy_base_vocab_size is set when loading a v1 tokenizer so merge
        # IDs align with the saved file.
        if self._legacy_base_vocab_size is not None:
            return self._legacy_base_vocab_size
        return 256 + len(BASE_SPECIAL_TOKENS) + len(self.extra_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self._base_vocab_size() + len(self.merges)

    @property
    def tokenizer_version(self) -> int:
        return 2 if self.extra_special_tokens else 1

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        *,
        target_vocab_size: int,
        min_frequency: int = 2,
        extra_special_tokens: dict[str, int] | None = None,
    ) -> "BPETokenizer":
        if extra_special_tokens is None:
            extra_special_tokens = {}
        base_size = cls._base_vocab_size_with_extra(extra_special_tokens)
        if target_vocab_size < base_size:
            raise ValueError(
                f"target_vocab_size {target_vocab_size} must be >= base vocab size {base_size}"
            )

        documents = [texts] if isinstance(texts, str) else list(texts)
        if not all(isinstance(document, str) for document in documents):
            raise TypeError("BPE training input must contain only strings")
        sequences = [list(document.encode("utf-8")) for document in documents]
        merges: list[tuple[int, int]] = []

        base_size = cls._base_vocab_size_with_extra(extra_special_tokens)
        while base_size + len(merges) < target_vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            highest_count = max(pair_counts.values())
            if highest_count < min_frequency:
                break
            best_pair = min(
                pair for pair, count in pair_counts.items() if count == highest_count
            )
            new_id = base_size + len(merges)
            sequences = [
                _merge_pair(sequence, best_pair, new_id) for sequence in sequences
            ]
            merges.append(best_pair)

        return cls(merges, extra_special_tokens=extra_special_tokens)

    @staticmethod
    def _base_vocab_size_with_extra(extra_special_tokens: dict[str, int]) -> int:
        return 256 + len(BASE_SPECIAL_TOKENS) + len(extra_special_tokens)

    @classmethod
    def train_streaming(
        cls,
        documents: Callable[[], Iterable[str]],
        *,
        target_vocab_size: int,
        min_frequency: int = 2,
        temp_dir: str | Path | None = None,
        extra_special_tokens: dict[str, int] | None = None,
    ) -> "BPETokenizer":
        if extra_special_tokens is None:
            extra_special_tokens = {}
        base_size = cls._base_vocab_size_with_extra(extra_special_tokens)
        if target_vocab_size < base_size:
            raise ValueError(
                f"target_vocab_size {target_vocab_size} must be >= base vocab size {base_size}"
            )
        if not callable(documents):
            raise TypeError("documents must be a callable returning an iterable of strings")

        temporary_parent = None if temp_dir is None else str(Path(temp_dir))
        with tempfile.TemporaryDirectory(
            prefix="puzzel-bpe-", dir=temporary_parent
        ) as temporary_directory:
            database_path = Path(temporary_directory) / "token_sequences.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA cache_size=-8192")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute(
                    "CREATE TABLE sequences (id INTEGER PRIMARY KEY, tokens BLOB NOT NULL)"
                )

                for document in documents():
                    if not isinstance(document, str):
                        raise TypeError("BPE training input must contain only strings")
                    raw_bytes = document.encode("utf-8")
                    if not raw_bytes:
                        continue
                    encoded = array("I")
                    encoded.extend(raw_bytes)
                    if encoded.itemsize != 4:
                        raise RuntimeError(
                            "platform does not provide 32-bit unsigned integers"
                        )
                    if sys.byteorder != "little":
                        encoded.byteswap()
                    connection.execute(
                        "INSERT INTO sequences(tokens) VALUES (?)", (encoded.tobytes(),)
                    )
                connection.commit()

                merges: list[tuple[int, int]] = []
                while base_size + len(merges) < target_vocab_size:
                    pair_counts: Counter[tuple[int, int]] = Counter()
                    for (token_blob,) in connection.execute(
                        "SELECT tokens FROM sequences ORDER BY id"
                    ):
                        previous: int | None = None
                        for (current,) in struct.iter_unpack("<I", token_blob):
                            if previous is not None:
                                pair_counts[(previous, current)] += 1
                            previous = current
                    if not pair_counts:
                        break
                    highest_count = max(pair_counts.values())
                    if highest_count < min_frequency:
                        break
                    best_pair = min(
                        pair
                        for pair, count in pair_counts.items()
                        if count == highest_count
                    )
                    new_id = base_size + len(merges)

                    last_id = 0
                    while True:
                        rows = connection.execute(
                            "SELECT id, tokens FROM sequences WHERE id > ? "
                            "ORDER BY id LIMIT 8",
                            (last_id,),
                        ).fetchall()
                        if not rows:
                            break
                        for sequence_id, token_blob in rows:
                            merged_blob = _merge_pair_blob(
                                token_blob, best_pair, new_id
                            )
                            connection.execute(
                                "UPDATE sequences SET tokens=? WHERE id=?",
                                (merged_blob, sequence_id),
                            )
                            last_id = sequence_id
                        connection.commit()
                    merges.append(best_pair)
                return cls(merges, extra_special_tokens=extra_special_tokens)
            finally:
                connection.close()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        add_language_id: str | None = None,
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        token_ids = list(text.encode("utf-8"))
        base_size = self._base_vocab_size()
        for merge_index, pair in enumerate(self.merges):
            token_ids = _merge_pair(token_ids, pair, base_size + merge_index)
        if add_bos:
            token_ids.insert(0, BOS_TOKEN_ID)
        if add_eos:
            token_ids.append(EOS_TOKEN_ID)
        if add_language_id is not None:
            lang_id = LANGUAGE_ID_TOKENS.get(add_language_id)
            if lang_id is None:
                raise ValueError(
                    f"unknown language code {add_language_id!r}; "
                    f"supported: {', '.join(sorted(LANGUAGE_ID_TOKENS))}"
                )
            if lang_id not in self.extra_special_tokens.values():
                raise ValueError(
                    f"language-ID token for {add_language_id!r} not configured "
                    f"in this tokenizer"
                )
            token_ids.insert(0, lang_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        output: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                output.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        all_special = {**BASE_SPECIAL_TOKENS, **self.extra_special_tokens}
        id_to_special = {v: k for k, v in all_special.items()}
        # Only recognize language IDs that are actually configured in this
        # tokenizer. Otherwise merge IDs in the 262-267 range would be
        # misinterpreted as language tokens.
        id_to_language = {
            v: k
            for k, v in LANGUAGE_ID_TOKENS.items()
            if v in self.extra_special_tokens.values()
        }

        for raw_id in token_ids:
            token_id = _as_token_id(raw_id)
            if token_id in id_to_special:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(id_to_special[token_id])
            elif token_id in id_to_language:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(f"[{id_to_language[token_id]}]")
            elif 0 <= token_id < BYTE_VOCAB_SIZE:
                pending_bytes.append(token_id)
            elif token_id in self._token_bytes:
                pending_bytes.extend(self._token_bytes[token_id])
            else:
                raise ValueError(
                    f"token ID {token_id} is outside this tokenizer vocabulary "
                    f"[0, {self.vocab_size})"
                )
        flush_bytes()
        return "".join(output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": (
                BPE_EXTENDED_FORMAT_NAME
                if self.extra_special_tokens
                else BPE_FORMAT_NAME
            ),
            "vocab_size": self.vocab_size,
            "base_vocab_size": self._base_vocab_size(),
            "special_tokens": dict(BASE_SPECIAL_TOKENS),
            "extra_special_tokens": dict(self.extra_special_tokens),
            "byte_encoding": "UTF-8",
            "version": self.tokenizer_version,
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BPETokenizer":
        tokenizer_format = payload.get("format")
        if tokenizer_format not in {BPE_FORMAT_NAME, BPE_EXTENDED_FORMAT_NAME}:
            raise ValueError(f"unsupported BPE tokenizer format: {tokenizer_format!r}")
        extra_special = payload.get("extra_special_tokens", {})
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("saved BPE merges must be a list")
        try:
            merges = [(int(pair[0]), int(pair[1])) for pair in raw_merges]
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("saved BPE merge rules are invalid") from exc
        # v1 format tokenizers (no extended special tokens) used base vocab
        # size 260 (256 bytes + 4 base specials). Merge IDs start at 260.
        legacy_base = None
        if tokenizer_format == BPE_FORMAT_NAME and not extra_special:
            legacy_base = 256 + len(BASE_SPECIAL_TOKENS)  # 260
        tokenizer = cls(
            merges,
            extra_special_tokens=extra_special,
            _legacy_base_vocab_size=legacy_base,
        )
        actual_vocab = tokenizer.vocab_size
        saved_vocab = payload.get("vocab_size")
        if saved_vocab is not None and saved_vocab != actual_vocab:
            raise ValueError("saved BPE vocabulary size does not match its merge rules")
        return tokenizer

    def save(self, path: str | Path) -> None:
        _write_tokenizer_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(_read_tokenizer_json(path))

    def tokenization_efficiency(self, text: str) -> dict[str, float]:
        token_ids = self.encode(text)
        byte_count = len(text.encode("utf-8"))
        character_count = len(text)
        token_count = len(token_ids)
        if token_count == 0:
            return {
                "chars_per_token": 0.0,
                "tokens_per_char": 0.0,
                "compression_ratio": 0.0,
                "token_count": 0,
                "byte_count": byte_count,
                "character_count": character_count,
            }
        return {
            "chars_per_token": character_count / token_count,
            "tokens_per_char": token_count / character_count if character_count > 0 else 0.0,
            "compression_ratio": byte_count / token_count,
            "token_count": float(token_count),
            "byte_count": float(byte_count),
            "character_count": float(character_count),
        }


def tokenizer_from_dict(payload: dict[str, Any]) -> TextTokenizer:
    tokenizer_format = payload.get("format")
    if tokenizer_format == BYTE_FORMAT_NAME:
        return ByteTokenizer.from_dict(payload)
    if tokenizer_format in {BPE_FORMAT_NAME, BPE_EXTENDED_FORMAT_NAME}:
        return BPETokenizer.from_dict(payload)
    raise ValueError(f"unsupported tokenizer format: {tokenizer_format!r}")


def _validate_bpe_training_options(target_vocab_size: int, min_frequency: int) -> None:
    if (
        not isinstance(target_vocab_size, int)
        or isinstance(target_vocab_size, bool)
        or target_vocab_size < 260
    ):
        raise ValueError("target_vocab_size must be an integer >= 260")
    if (
        not isinstance(min_frequency, int)
        or isinstance(min_frequency, bool)
        or min_frequency < 1
    ):
        raise ValueError("min_frequency must be a positive integer")


def _merge_pair_blob(
    token_blob: bytes, pair: tuple[int, int], new_id: int
) -> bytes:
    tokens = (token[0] for token in struct.iter_unpack("<I", token_blob))
    iterator = iter(tokens)
    current = next(iterator, None)
    merged = array("I")
    while current is not None:
        following = next(iterator, None)
        if following is not None and (current, following) == pair:
            merged.append(new_id)
            current = next(iterator, None)
        else:
            merged.append(current)
            current = following
    if sys.byteorder != "little":
        merged.byteswap()
    return merged.tobytes()


def _merge_pair(
    sequence: Sequence[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    left, right = pair
    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == left
            and sequence[index + 1] == right
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


def _as_token_id(raw_id: int) -> int:
    try:
        token_id = int(raw_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid token ID: {raw_id!r}") from exc
    return token_id


def _read_tokenizer_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer JSON must contain an object")
    return payload


def _write_tokenizer_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BYTE_VOCAB_SIZE",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "PAD_TOKEN_ID",
    "UNK_TOKEN_ID",
    "VOCAB_SIZE",
    "MASK_TOKEN_ID",
    "SEP_TOKEN_ID",
    "LANGUAGE_ID_TOKENS",
    "ID_TO_LANGUAGE",
    "BASE_SPECIAL_TOKENS",
    "SPECIAL_TOKENS",
    "EXTENDED_SPECIAL_TOKENS",
    "BYTE_FORMAT_NAME",
    "BPE_FORMAT_NAME",
    "BPE_EXTENDED_FORMAT_NAME",
    "TextTokenizer",
    "ByteTokenizer",
    "BPETokenizer",
    "tokenizer_from_dict",
]