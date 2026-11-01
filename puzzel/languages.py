"""Language identifiers and detection utilities for the Puzzel project.

Supports English, Hindi, Spanish, French, Arabic, and Chinese with optional
language detection based on Unicode script analysis.
"""

from __future__ import annotations

from typing import Optional

# Unicode code point ranges for script detection (Python 3.11 compatible,
# since unicodedata.script() was added in Python 3.13).
_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("Latin", 0x0000, 0x007F),      # Basic Latin
    ("Latin", 0x0080, 0x00FF),      # Latin-1 Supplement
    ("Latin", 0x0100, 0x017F),      # Latin Extended-A
    ("Latin", 0x0180, 0x024F),      # Latin Extended-B
    ("Devanagari", 0x0900, 0x097F), # Devanagari (Hindi)
    ("Arabic", 0x0600, 0x06FF),     # Arabic
    ("Arabic", 0x0750, 0x077F),     # Arabic Supplement
    ("Han", 0x4E00, 0x9FFF),        # CJK Unified Ideographs
    ("Han", 0x3400, 0x4DBF),        # CJK Unified Ideographs Extension A
]


def _get_script(char: str) -> str:
    """Return the Unicode script name for a character using code point ranges.

    Falls back to "Unknown" for characters not in any known range.
    """
    code_point = ord(char)
    for script, start, end in _SCRIPT_RANGES:
        if start <= code_point <= end:
            return script
    return "Unknown"

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "ar": "Arabic",
    "zh": "Chinese",
}

MULTILINGUAL_LANGUAGE_CODES = tuple(LANGUAGE_NAMES)

DEFAULT_MULTILINGUAL_WEIGHTS: dict[str, float] = {
    "en": 0.20,
    "hi": 0.20,
    "es": 0.15,
    "fr": 0.15,
    "ar": 0.15,
    "zh": 0.15,
}

# Unicode script ranges for language detection
_SCRIPT_LANGUAGE_MAP: dict[str, str] = {
    "Devanagari": "hi",      # Hindi
    "Arabic": "ar",          # Arabic
    "Han": "zh",             # Chinese
    "Hiragana": "zh",        # Japanese (mapped to zh as fallback)
    "Katakana": "zh",        # Japanese (mapped to zh as fallback)
    "Hangul": "ko",          # Korean (not in supported list, but detected)
    "Cyrillic": "ru",        # Russian (not in supported list, but detected)
    "Greek": "el",           # Greek (not in supported list, but detected)
}

# Latin-script languages (can't be distinguished by script alone)
_LATIN_SCRIPT_LANGUAGES = {"en", "es", "fr"}


def detect_language(text: str) -> Optional[str]:
    """Detect the language of a text string using Unicode script analysis.

    Uses Unicode character script ranges to identify the likely language.
    For Latin-script text (English, Spanish, French), returns None since
    script analysis cannot distinguish them.

    Args:
        text: Input text string.

    Returns:
        Detected language code, or None if undetectable (e.g., Latin-script
        text or empty string).
    """
    if not text or not text.strip():
        return None

    # Sample up to 1000 non-whitespace characters for script detection
    sample_chars = []
    for char in text:
        if char.isspace():
            continue
        sample_chars.append(char)
        if len(sample_chars) >= 1000:
            break

    if not sample_chars:
        return None

    # Count scripts in the sample
    script_counts: dict[str, int] = {}
    for char in sample_chars:
        script = _get_script(char)
        script_counts[script] = script_counts.get(script, 0) + 1

    # Find the dominant script
    dominant_script = max(script_counts, key=script_counts.get)
    dominant_count = script_counts[dominant_script]
    total = len(sample_chars)

    # Require the dominant script to be at least 60% of the sample
    if dominant_count / total < 0.60:
        return None

    # Map script to language
    detected = _SCRIPT_LANGUAGE_MAP.get(dominant_script)
    if detected is None:
        # Latin script - could be en, es, or fr; can't distinguish by script
        return None

    # Verify the detected language is supported
    if detected not in LANGUAGE_NAMES:
        return None

    return detected


def detect_language_confident(text: str, *, threshold: float = 0.80) -> Optional[str]:
    """Detect language with a higher confidence threshold.

    Like detect_language but requires a higher proportion of the dominant
    script before returning a result.

    Args:
        text: Input text string.
        threshold: Minimum proportion of dominant script (0.0 to 1.0).

    Returns:
        Detected language code, or None if confidence is insufficient.
    """
    if not text or not text.strip():
        return None

    sample_chars = []
    for char in text:
        if char.isspace():
            continue
        sample_chars.append(char)
        if len(sample_chars) >= 1000:
            break

    if not sample_chars:
        return None

    script_counts: dict[str, int] = {}
    for char in sample_chars:
        script = _get_script(char)
        script_counts[script] = script_counts.get(script, 0) + 1

    dominant_script = max(script_counts, key=script_counts.get)
    dominant_count = script_counts[dominant_script]
    total = len(sample_chars)

    if dominant_count / total < threshold:
        return None

    detected = _SCRIPT_LANGUAGE_MAP.get(dominant_script)
    if detected is None:
        return None
    if detected not in LANGUAGE_NAMES:
        return None

    return detected


def is_supported_language(code: str) -> bool:
    """Return True if the language code is one of the six supported languages."""
    return code in LANGUAGE_NAMES


def get_language_name(code: str) -> str:
    """Return the human-readable name for a language code."""
    return LANGUAGE_NAMES.get(code, code)


__all__ = [
    "LANGUAGE_NAMES",
    "MULTILINGUAL_LANGUAGE_CODES",
    "DEFAULT_MULTILINGUAL_WEIGHTS",
    "detect_language",
    "detect_language_confident",
    "is_supported_language",
    "get_language_name",
]