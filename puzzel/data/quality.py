"""Unicode-preserving text normalization and conservative quality checks."""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from itertools import islice

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Normalize Unicode/spacing while preserving punctuation and paragraphs."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    output = io.StringIO()
    has_content = False
    pending_blank = False
    for raw_line in io.StringIO(text):
        line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line.rstrip("\n")).strip()
        if not line:
            if has_content:
                pending_blank = True
            continue
        if has_content:
            output.write("\n\n" if pending_blank else "\n")
        output.write(line)
        has_content = True
        pending_blank = False
    return output.getvalue()


def quality_rejection_reason(text: str) -> str | None:
    """Return a rejection reason for clearly corrupted/repetitive documents.

    Heuristics are intentionally conservative for multilingual material. They
    sample at most 10,000 characters/windows, so checking a bounded document
    does not allocate memory proportional to its full character count.
    """
    if "\ufffd" in text:
        return "replacement_character"
    if "\x00" in text:
        return "nul_character"

    if text:
        control_count = sum(
            unicodedata.category(character) == "Cc"
            and character not in "\n\r\t\f"
            for character in text
        )
        if control_count >= 2 and control_count / len(text) > 0.01:
            return "excessive_control_characters"

    if len(text) < 100:
        return None

    sample_stride = max(1, len(text) // 10_000)
    sample = text[::sample_stride][:10_000]
    char_counts = Counter(sample)
    if char_counts and max(char_counts.values()) / len(sample) >= 0.92:
        return "excessive_single_character_repetition"

    sampled_lines = [
        line.strip()
        for line in islice(io.StringIO(text), 2048)
        if line.strip()
    ]
    if len(sampled_lines) >= 10:
        line_counts = Counter(sampled_lines)
        if max(line_counts.values()) / len(sampled_lines) >= 0.9:
            return "excessive_repeated_lines"

    if len(sample) >= 100:
        ngram_counts = Counter(
            sample[index : index + 5] for index in range(len(sample) - 4)
        )
        ngram_total = sum(ngram_counts.values())
        if ngram_total and len(ngram_counts) / ngram_total < 0.05:
            return "excessive_repeated_ngrams"
    return None
