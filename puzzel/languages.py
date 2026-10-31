"""Language identifiers used by the first multilingual Puzzel experiment."""

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
