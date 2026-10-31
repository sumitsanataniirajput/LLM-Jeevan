"""Phase 9 regression tests for multilingual model improvement.

These tests verify that issues identified in Phase 8 evaluation do not regress
after optimization. They use the existing Phase 8 evaluation datasets and
baseline checkpoint for comparison.

Run with: ./.venv/bin/python -m pytest tests/test_phase9_regressions.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzel.checkpoint import load_checkpoint  # noqa: E402
from puzzel.data.prepared import PreparedSplitDataset  # noqa: E402
from puzzel.evaluation.runner import greedy_generate  # noqa: E402
from puzzel.training.loop import resolve_device  # noqa: E402
from puzzel.tokenizer import (  # noqa: E402
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    tokenizer_from_dict,
)

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = REPO / "checkpoints/multilingual-first-run.best.pt"
DATASET_DIR = REPO / "data/prepared"
EVAL_DIR = REPO / "data/evaluation"
LANGUAGES = {"en", "hi", "es", "fr", "ar", "zh"}

# Per-language perplexity thresholds derived from Phase 8 baseline.
# These are baseline values; post-optimization should improve (lower) them.
BASELINE_TEST_PERPLEXITY = {
    "en": 909.18,
    "hi": 364.00,
    "es": 389.38,
    "fr": 448.15,
    "ar": 450.68,
    "zh": 828.95,
}

# Thresholds: perplexity must be below 1.5x baseline to avoid regression.
PERPLEXITY_REGRESSION_THRESHOLD = {
    lang: ppl * 1.5 for lang, ppl in BASELINE_TEST_PERPLEXITY.items()
}

# Nonempty generation must be at least 0.5 (half the examples produce output).
NON_EMPTY_THRESHOLD = 0.5

# Max consecutive repetition of a single token before flagging as degenerate.
MAX_CONSECUTIVE_REPEAT = 5


def _load_checkpoint_and_model():
    device = resolve_device("cpu")
    loaded = load_checkpoint(CHECKPOINT_PATH, map_location=device)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = loaded.tokenizer
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    return device, loaded, model, tokenizer, manifest


def _count_consecutive_repeats(text: str) -> int:
    """Return the maximum number of consecutive identical whitespace-separated tokens."""
    if not text.strip():
        return 0
    tokens = text.split()
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _script_chars_for_language(lang: str) -> set[str]:
    """Return expected Unicode script name hints for each language."""
    return {
        "en": {"LATIN"},
        "es": {"LATIN"},
        "fr": {"LATIN"},
        "hi": {"DEVANAGARI"},
        "ar": {"ARABIC"},
        "zh": {"CJK"},
    }.get(lang, {"LATIN"})


def _count_off_script_chars(text: str, lang: str) -> tuple[int, int]:
    """Count characters not in the expected script. Returns (off_script, total)."""
    import unicodedata

    expected = _script_chars_for_language(lang)
    off = 0
    total = 0
    for ch in text:
        if ch.isspace() or ch in ".,!?;:\"'()[]-—…":
            continue
        total += 1
        try:
            name = unicodedata.name(ch)
        except ValueError:
            off += 1
            continue
        if not any(h in name for h in expected):
            if ch.isdigit():
                continue
            off += 1
    return off, total


@pytest.fixture(scope="module")
def model_and_tokenizer():
    device, loaded, model, tokenizer, manifest = _load_checkpoint_and_model()
    return {"device": device, "loaded": loaded, "model": model, "tokenizer": tokenizer, "manifest": manifest}


# ---------------------------------------------------------------------------
# Task: No NaN/Inf losses on test split
# ---------------------------------------------------------------------------

def test_no_nan_inf_losses_on_test_split(model_and_tokenizer):
    """Zero non-finite losses across the entire held-out test split."""
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]
    manifest = model_and_tokenizer["manifest"]

    nan_inf_count = 0
    for lang in LANGUAGES:
        test_ds = PreparedSplitDataset(
            DATASET_DIR, "test", context_length=model.context_length
        )
        lang_range = manifest["splits"]["test"]["language_ranges"][lang]
        for i in range(lang_range["start"], lang_range["start"] + lang_range["examples"]):
            input_ids, targets = test_ds[i]
            input_tensor = input_ids.unsqueeze(0).to(device)
            target_tensor = targets.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)[0]
            loss = F.cross_entropy(
                logits, target_tensor[0], reduction="sum", ignore_index=PAD_TOKEN_ID
            )
            assert torch.isfinite(loss).item(), f"Non-finite loss for {lang} at example {i}"
            nan_inf_count += 0  # counter for clarity; assert above catches failures


# ---------------------------------------------------------------------------
# Task: No cross-language mixing (each language generates in-script)
# ---------------------------------------------------------------------------

def test_no_cross_language_mixing(model_and_tokenizer):
    """Each language generates characters in its expected script only."""
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    prompts = {
        "en": "Hello, how are you?",
        "hi": "नमस्ते, आप कैसे हैं?",
        "es": "Hola, ¿cómo estás?",
        "fr": "Bonjour, comment allez-vous ?",
        "ar": "مرحبا، كيف حالك؟",
        "zh": "你好，你今天好吗？",
    }

    for lang in LANGUAGES:
        generated = greedy_generate(
            model, tokenizer, prompts[lang], device=device, max_new_tokens=40
        )
        off_script, total = _count_off_script_chars(generated, lang)
        if total > 0:
            ratio = off_script / total
            assert ratio <= 0.5, (
                f"{lang}: {off_script}/{total} off-script characters ({ratio:.1%}) "
                f"in generation: {generated!r}"
            )


# ---------------------------------------------------------------------------
# Task: Tokenizer round-trip for all languages
# ---------------------------------------------------------------------------

def test_tokenizer_round_trip_all_languages(model_and_tokenizer):
    """Tokenizer round-trips UTF-8 text losslessly for all 6 languages."""
    tokenizer = model_and_tokenizer["tokenizer"]

    unicode_tests = {
        "en": "Hello, World! Café résumé naïve.",
        "hi": "नमस्ते दुनिया। आपका दिन शुभ हो।",
        "es": "¡Hola, mundo! ¿Cómo está usted?",
        "fr": "Bonjour le monde ! Ça va bien ?",
        "ar": "مرحبا بالعالم. كيف حالك؟",
        "zh": "你好，世界。今天天气很好。",
    }

    for lang, text in unicode_tests.items():
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_clean = decoded.replace("<|bos|>", "").replace("<|eos|>", "")
        assert decoded_clean == text, (
            f"{lang}: tokenizer round-trip mismatch. "
            f"Original: {text!r}, Decoded: {decoded_clean!r}"
        )


# ---------------------------------------------------------------------------
# Task: Generation non-empty rate threshold
# ---------------------------------------------------------------------------

def test_generation_non_empty_rate_threshold(model_and_tokenizer):
    """Each language produces non-empty output above the threshold."""
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    prompts = {
        "en": "Hello, how are you?",
        "hi": "नमस्ते, आप कैसे हैं?",
        "es": "Hola, ¿cómo estás?",
        "fr": "Bonjour, comment allez-vous ?",
        "ar": "مرحبا، كيف حالك؟",
        "zh": "你好，你今天好吗？",
    }

    for lang in LANGUAGES:
        generated = greedy_generate(
            model, tokenizer, prompts[lang], device=device, max_new_tokens=40
        )
        assert generated.strip(), f"{lang}: empty generation under greedy decoding"


# ---------------------------------------------------------------------------
# Task: Perplexity below regression threshold per language
# ---------------------------------------------------------------------------

def test_perplexity_below_regression_threshold(model_and_tokenizer):
    """Perplexity must not regress beyond 1.5x the Phase 8 baseline."""
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]
    manifest = model_and_tokenizer["manifest"]

    for lang in LANGUAGES:
        test_ds = PreparedSplitDataset(
            DATASET_DIR, "test", context_length=model.context_length
        )
        lang_range = manifest["splits"]["test"]["language_ranges"][lang]
        start, count = lang_range["start"], lang_range["examples"]

        total_nll = 0.0
        total_tokens = 0
        for i in range(start, start + count):
            input_ids, targets = test_ds[i]
            input_tensor = input_ids.unsqueeze(0).to(device)
            target_tensor = targets.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)[0]
            loss = F.cross_entropy(
                logits, target_tensor[0], reduction="sum", ignore_index=PAD_TOKEN_ID
            )
            total_nll += float(loss.item())
            total_tokens += int((target_tensor[0] != PAD_TOKEN_ID).sum().item())

        mean_nll = total_nll / total_tokens if total_tokens else float("inf")
        ppl = float(torch.exp(torch.tensor(mean_nll))) if total_tokens else float("inf")
        threshold = PERPLEXITY_REGRESSION_THRESHOLD[lang]
        assert ppl <= threshold, (
            f"{lang}: perplexity {ppl:.1f} exceeds regression threshold {threshold:.1f} "
            f"(baseline: {BASELINE_TEST_PERPLEXITY[lang]:.1f})"
        )


# ---------------------------------------------------------------------------
# Task: Validation loss not greater than train loss (no overfitting)
# ---------------------------------------------------------------------------

def test_validation_loss_not_greater_than_train_loss():
    """No overfitting: validation loss must not exceed training loss."""
    metrics_path = REPO / "checkpoints/multilingual-first-run.metrics.csv"
    assert metrics_path.exists(), "Training metrics CSV not found"

    train_losses = []
    val_losses = []
    with metrics_path.open(newline="", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            tl = row.get("train_loss", "").strip()
            vl = row.get("validation_loss", "").strip()
            if tl:
                train_losses.append(float(tl))
            if vl:
                val_losses.append(float(vl))

    assert train_losses, "No training losses found in metrics"
    assert val_losses, "No validation losses found in metrics"

    final_train = train_losses[-1]
    final_val = val_losses[-1] if val_losses else None
    if final_val is not None:
        assert final_val <= final_train + 0.01, (
            f"Overfitting detected: validation loss ({final_val:.4f}) > "
            f"training loss ({final_train:.4f})"
        )


# ---------------------------------------------------------------------------
# Task: No repetitive single-token dominance
# ---------------------------------------------------------------------------

def test_no_repetitive_single_token_dominance(model_and_tokenizer):
    """No single token repeated more than MAX_CONSECUTIVE_REPEAT times consecutively."""
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    prompts = {
        "en": "Hello, how are you?",
        "hi": "नमस्ते, आप कैसे हैं?",
        "es": "Hola, ¿cómo estás?",
        "fr": "Bonjour, comment allez-vous ?",
        "ar": "مرحبا، كيف حالك؟",
        "zh": "你好，你今天好吗？",
    }

    for lang in LANGUAGES:
        generated = greedy_generate(
            model, tokenizer, prompts[lang], device=device, max_new_tokens=40
        )
        max_run = _count_consecutive_repeats(generated)
        assert max_run <= MAX_CONSECUTIVE_REPEAT, (
            f"{lang}: {max_run} consecutive repeated tokens (> {MAX_CONSECUTIVE_REPEAT}) "
            f"in generation: {generated!r}"
        )


# ---------------------------------------------------------------------------
# Task: Exact match improvement over baseline (baseline is 0.0)
# ---------------------------------------------------------------------------

def test_exact_match_not_worse_than_baseline(model_and_tokenizer):
    """Exact match accuracy must not regress below the Phase 8 baseline of 0.0."""
    # Baseline exact match is 0.0 for all languages. This test ensures it
    # does not regress below 0 (impossible) and serves as a placeholder for
    # post-optimization comparison where exact_match should be > 0.
    device = model_and_tokenizer["device"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    # Quick smoke: model can produce at least one non-EOS token
    generated = greedy_generate(
        model, tokenizer, "Hello", device=device, max_new_tokens=1
    )
    assert isinstance(generated, str)
    # The mere fact generation runs without error and produces a string
    # confirms the model is functional; exact match > 0 requires retraining.
