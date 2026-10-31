from puzzel.data.quality import normalize_text, quality_rejection_reason


def test_normalization_preserves_unicode_punctuation_and_paragraphs() -> None:
    source = "  नमस्ते \t— café!\r\n\r\n\r\n  Second paragraph?  "

    cleaned = normalize_text(source)

    assert cleaned == "नमस्ते — café!\n\nSecond paragraph?"


def test_quality_checks_reject_corruption_and_repetition() -> None:
    assert quality_rejection_reason("damaged \ufffd text") == "replacement_character"
    assert quality_rejection_reason("bad\x00text") == "nul_character"
    assert quality_rejection_reason("abcabc" * 100) == "excessive_repeated_ngrams"
    assert quality_rejection_reason(("same repeated line\n" * 20).strip()) == "excessive_repeated_lines"
    assert quality_rejection_reason("A useful multilingual document — नमस्ते!") is None
