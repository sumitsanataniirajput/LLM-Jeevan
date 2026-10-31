"""Project-owned UTF-8 byte and learned byte-pair tokenizers.

The byte tokenizer is a deterministic fallback. ``BPETokenizer.train`` learns
merge rules only from caller-supplied local text; no external vocabulary or
pretrained tokenizer is downloaded.
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
BYTE_FORMAT_NAME = "puzzel-utf8-byte-tokenizer-v1"
BPE_FORMAT_NAME = "puzzel-byte-bpe-tokenizer-v1"

SPECIAL_TOKENS: dict[str, int] = {
    "<|bos|>": BOS_TOKEN_ID,
    "<|eos|>": EOS_TOKEN_ID,
    "<|pad|>": PAD_TOKEN_ID,
    "<|unk|>": UNK_TOKEN_ID,
}
ID_TO_SPECIAL = {value: key for key, value in SPECIAL_TOKENS.items()}


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
        return VOCAB_SIZE

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text as bytes, optionally surrounding it with markers."""
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
        """Decode byte IDs as UTF-8, replacing malformed byte sequences."""
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
            elif token_id in ID_TO_SPECIAL:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(ID_TO_SPECIAL[token_id])
            else:
                raise ValueError(f"token ID {token_id} is outside [0, {VOCAB_SIZE})")
        flush_bytes()
        return "".join(output)

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed tokenizer definition in JSON-compatible form."""
        return {
            "format": BYTE_FORMAT_NAME,
            "vocab_size": VOCAB_SIZE,
            "special_tokens": dict(SPECIAL_TOKENS),
            "byte_encoding": "UTF-8",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ByteTokenizer":
        if payload.get("format") != BYTE_FORMAT_NAME:
            raise ValueError("unsupported byte tokenizer format")
        if payload.get("vocab_size") != VOCAB_SIZE:
            raise ValueError("saved byte tokenizer vocabulary size does not match")
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("saved byte tokenizer special IDs do not match")
        if payload.get("byte_encoding") != "UTF-8":
            raise ValueError("saved byte tokenizer encoding does not match UTF-8")
        return cls()

    def save(self, path: str | Path) -> None:
        _write_tokenizer_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        return cls.from_dict(_read_tokenizer_json(path))


class BPETokenizer:
    """Small deterministic byte-level BPE tokenizer trained from local text.

    IDs 0-255 are the UTF-8 byte fallback, 256-259 are reserved special IDs,
    and each learned pair merge receives the next ID starting at 260. Training
    repeatedly merges the most frequent adjacent token pair. Ties are resolved
    lexicographically so the learned vocabulary is reproducible.
    """

    def __init__(self, merges: Sequence[tuple[int, int]] = ()) -> None:
        self.merges = tuple((int(left), int(right)) for left, right in merges)
        token_bytes = {token_id: bytes((token_id,)) for token_id in range(BYTE_VOCAB_SIZE)}
        seen_pairs: set[tuple[int, int]] = set()
        for merge_index, pair in enumerate(self.merges):
            left, right = pair
            if pair in seen_pairs:
                raise ValueError(f"duplicate BPE merge rule: {pair}")
            if left not in token_bytes or right not in token_bytes:
                raise ValueError(
                    f"merge {merge_index} refers to a token not yet in the vocabulary"
                )
            new_id = VOCAB_SIZE + merge_index
            token_bytes[new_id] = token_bytes[left] + token_bytes[right]
            seen_pairs.add(pair)
        self._token_bytes = token_bytes
        self._merge_pairs = {pair: VOCAB_SIZE + index for index, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE + len(self.merges)

    @classmethod
    def train(
        cls,
        texts: str | Iterable[str],
        *,
        target_vocab_size: int,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn up to ``target_vocab_size`` entries from caller-provided text."""
        if (
            not isinstance(target_vocab_size, int)
            or isinstance(target_vocab_size, bool)
            or target_vocab_size < VOCAB_SIZE
        ):
            raise ValueError(f"target_vocab_size must be an integer >= {VOCAB_SIZE}")
        if (
            not isinstance(min_frequency, int)
            or isinstance(min_frequency, bool)
            or min_frequency < 1
        ):
            raise ValueError("min_frequency must be a positive integer")

        documents = [texts] if isinstance(texts, str) else list(texts)
        if not all(isinstance(document, str) for document in documents):
            raise TypeError("BPE training input must contain only strings")
        sequences = [list(document.encode("utf-8")) for document in documents]
        merges: list[tuple[int, int]] = []

        while VOCAB_SIZE + len(merges) < target_vocab_size:
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
            new_id = VOCAB_SIZE + len(merges)
            sequences = [
                _merge_pair(sequence, best_pair, new_id) for sequence in sequences
            ]
            merges.append(best_pair)

        return cls(merges)

    @classmethod
    def train_streaming(
        cls,
        documents: Callable[[], Iterable[str]],
        *,
        target_vocab_size: int,
        min_frequency: int = 2,
        temp_dir: str | Path | None = None,
    ) -> "BPETokenizer":
        """Train BPE with disk-backed per-document token sequences.

        The document factory is consumed once. Token sequences are kept in a
        temporary SQLite file so pair counting and merges operate in bounded
        memory; only one bounded document/batch is resident at a time.
        """
        _validate_bpe_training_options(target_vocab_size, min_frequency)
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
                        raise RuntimeError("platform does not provide 32-bit unsigned integers")
                    if sys.byteorder != "little":
                        encoded.byteswap()
                    connection.execute(
                        "INSERT INTO sequences(tokens) VALUES (?)", (encoded.tobytes(),)
                    )
                connection.commit()

                merges: list[tuple[int, int]] = []
                while VOCAB_SIZE + len(merges) < target_vocab_size:
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
                    new_id = VOCAB_SIZE + len(merges)

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
                return cls(merges)
            finally:
                connection.close()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode text by applying learned merges in training-rank order."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        token_ids = list(text.encode("utf-8"))
        for merge_index, pair in enumerate(self.merges):
            token_ids = _merge_pair(token_ids, pair, VOCAB_SIZE + merge_index)
        if add_bos:
            token_ids.insert(0, BOS_TOKEN_ID)
        if add_eos:
            token_ids.append(EOS_TOKEN_ID)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Expand byte and learned-token IDs back to UTF-8 text."""
        output: list[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                output.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for raw_id in token_ids:
            token_id = _as_token_id(raw_id)
            if token_id in ID_TO_SPECIAL:
                flush_bytes()
                if not skip_special_tokens:
                    output.append(ID_TO_SPECIAL[token_id])
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
        """Return the learned merge vocabulary in JSON-compatible form."""
        return {
            "format": BPE_FORMAT_NAME,
            "vocab_size": self.vocab_size,
            "special_tokens": dict(SPECIAL_TOKENS),
            "byte_encoding": "UTF-8",
            "merges": [list(pair) for pair in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BPETokenizer":
        if payload.get("format") != BPE_FORMAT_NAME:
            raise ValueError("unsupported BPE tokenizer format")
        if payload.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("saved BPE tokenizer special IDs do not match")
        if payload.get("byte_encoding") != "UTF-8":
            raise ValueError("saved BPE tokenizer encoding does not match UTF-8")
        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("saved BPE merges must be a list")
        try:
            merges = [(int(pair[0]), int(pair[1])) for pair in raw_merges]
            tokenizer = cls(merges)
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("saved BPE merge rules are invalid") from exc
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise ValueError("saved BPE vocabulary size does not match its merge rules")
        return tokenizer

    def save(self, path: str | Path) -> None:
        _write_tokenizer_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls.from_dict(_read_tokenizer_json(path))


def tokenizer_from_dict(payload: dict[str, Any]) -> TextTokenizer:
    """Recreate either built-in tokenizer from a serialized definition."""
    tokenizer_format = payload.get("format")
    if tokenizer_format == BYTE_FORMAT_NAME:
        return ByteTokenizer.from_dict(payload)
    if tokenizer_format == BPE_FORMAT_NAME:
        return BPETokenizer.from_dict(payload)
    raise ValueError(f"unsupported tokenizer format: {tokenizer_format!r}")


def _validate_bpe_training_options(target_vocab_size: int, min_frequency: int) -> None:
    if (
        not isinstance(target_vocab_size, int)
        or isinstance(target_vocab_size, bool)
        or target_vocab_size < VOCAB_SIZE
    ):
        raise ValueError(f"target_vocab_size must be an integer >= {VOCAB_SIZE}")
    if (
        not isinstance(min_frequency, int)
        or isinstance(min_frequency, bool)
        or min_frequency < 1
    ):
        raise ValueError("min_frequency must be a positive integer")


def _merge_pair_blob(
    token_blob: bytes, pair: tuple[int, int], new_id: int
) -> bytes:
    """Apply one BPE merge to a little-endian uint32 token stream."""
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
    """Replace non-overlapping left-to-right occurrences of one token pair."""
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
