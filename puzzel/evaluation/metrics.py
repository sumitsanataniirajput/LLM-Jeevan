"""Transparent, dependency-free metrics for local evaluation reports."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from typing import Iterable


def normalize_answer(text: str) -> str:
    """Normalize answers for conservative exact-match scoring across scripts."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def exact_match(prediction: str, references: Iterable[str]) -> bool:
    """Return whether normalized prediction matches any accepted reference."""
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return False
    return any(
        normalized_prediction == normalize_answer(reference)
        for reference in references
        if normalize_answer(reference)
    )


def chrf_score(
    prediction: str,
    reference: str,
    *,
    character_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Compute a small corpus-independent character n-gram F-score (0-100).

    Whitespace is excluded, Unicode is NFC-normalized, and n-gram overlap uses
    clipped multiset counts. This transparent implementation is a diagnostic
    metric, not a substitute for a published benchmark implementation.
    """
    if character_order <= 0:
        raise ValueError("character_order must be positive")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number")
    hypothesis = "".join(
        character
        for character in unicodedata.normalize("NFC", prediction)
        if not character.isspace()
    )
    reference_text = "".join(
        character
        for character in unicodedata.normalize("NFC", reference)
        if not character.isspace()
    )
    if not hypothesis or not reference_text:
        return 0.0

    overlap_total = 0
    hypothesis_total = 0
    reference_total = 0
    for order in range(1, character_order + 1):
        hypothesis_ngrams = Counter(
            hypothesis[index : index + order]
            for index in range(max(0, len(hypothesis) - order + 1))
        )
        reference_ngrams = Counter(
            reference_text[index : index + order]
            for index in range(max(0, len(reference_text) - order + 1))
        )
        overlap_total += sum((hypothesis_ngrams & reference_ngrams).values())
        hypothesis_total += sum(hypothesis_ngrams.values())
        reference_total += sum(reference_ngrams.values())
    if not hypothesis_total or not reference_total:
        return 0.0
    precision = overlap_total / hypothesis_total
    recall = overlap_total / reference_total
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    if denominator == 0:
        return 0.0
    return 100.0 * (1.0 + beta_squared) * precision * recall / denominator


def finite_perplexity(mean_nll: float | None) -> float | None:
    """Exponentiate mean token NLL, returning None rather than JSON Infinity."""
    if mean_nll is None or not math.isfinite(mean_nll):
        return None
    if mean_nll > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return None
    return math.exp(mean_nll)
