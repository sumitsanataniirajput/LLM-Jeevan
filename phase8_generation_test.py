"""Phase 8 Tasks 4-5: Text generation, language consistency, Unicode, tokenizer,
NaN/Inf, repetitive output, broken text, language mixing checks."""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.nn import functional as F

from puzzel.checkpoint import load_checkpoint  # noqa: E402
from puzzel.training.loop import resolve_device  # noqa: E402
from puzzel.data.prepared import PreparedSplitDataset  # noqa: E402
from puzzel.tokenizer import (  # noqa: E402
    tokenizer_from_dict,
    BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID,
)

REPO = Path(__file__).resolve().parent
LANGUAGES = {"en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French", "ar": "Arabic", "zh": "Chinese"}


def greedy_generate(model, tokenizer, prompt, *, device, max_new_tokens):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    generated = []
    model.eval()
    forbidden = (BOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID)
    for _ in range(max_new_tokens):
        active_ids = (prompt_ids + generated)[-model.context_length:]
        input_ids = torch.tensor(active_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids)[0, -1].clone()
        for tid in forbidden:
            if tid < logits.numel():
                logits[tid] = -torch.inf
        next_token = int(torch.argmax(logits).item())
        if next_token == EOS_TOKEN_ID:
            break
        generated.append(next_token)
    return tokenizer.decode(generated)


def has_repetitive_output(text, min_run=4):
    if not text:
        return False
    for m in re.finditer(r'(.)\1{' + str(min_run - 1) + r',}', text):
        return True
    words = text.split()
    for i in range(len(words) - min_run + 1):
        if all(w == words[i] for w in words[i:i + min_run]):
            return True
    return False


def count_off_script_chars(text, lang):
    script_hints = {
        "en": ("LATIN",),
        "es": ("LATIN",),
        "fr": ("LATIN",),
        "hi": ("DEVANAGARI",),
        "ar": ("ARABIC",),
        "zh": ("CJK",),
    }
    expected = script_hints.get(lang, ("LATIN",))
    count = 0
    total = 0
    for ch in text:
        if ch.isspace() or ch in ".,!?;:\"'()[]-—…":
            continue
        total += 1
        try:
            name = unicodedata.name(ch)
        except ValueError:
            count += 1
            continue
        if not any(h in name for h in expected):
            if ch.isdigit():
                continue
            count += 1
    return count, total


def main() -> None:
    device = resolve_device("cpu")
    checkpoint_path = REPO / "checkpoints/multilingual-first-run.best.pt"
    dataset_dir = REPO / "data/prepared"

    loaded = load_checkpoint(checkpoint_path, map_location=device)
    model = loaded.model.to(device)
    model.eval()
    tokenizer = loaded.tokenizer

    print("=" * 70)
    print("TASK 4: Text Generation, Language Consistency, Unicode, Tokenizer")
    print("=" * 70)

    prompts = {
        "en": "Hello, how are you?",
        "hi": "नमस्ते, आप कैसे हैं?",
        "es": "Hola, ¿cómo estás?",
        "fr": "Bonjour, comment allez-vous ?",
        "ar": "مرحبا، كيف حالك؟",
        "zh": "你好，你今天好吗？",
    }

    print("\n--- Greedy generation samples (max 40 new tokens) ---")
    generations = {}
    for lang, name in LANGUAGES.items():
        prompt = prompts[lang]
        generated = greedy_generate(model, tokenizer, prompt, device=device, max_new_tokens=40)
        generations[lang] = generated
        rep = has_repetitive_output(generated)
        non_script, total = count_off_script_chars(generated, lang)
        print(f"\n[{lang} / {name}]")
        print(f"  Prompt:    {prompt}")
        print(f"  Generated: {generated!r}")
        print(f"  Nonempty:  {bool(generated.strip())}")
        print(f"  Repetitive: {rep}")
        if total > 0:
            print(f"  Off-script chars: {non_script}/{total} ({100 * non_script / total:.1f}%)")
        else:
            print("  Off-script chars: N/A (no content chars)")

    print("\n--- Tokenizer Unicode round-trip test ---")
    unicode_tests = {
        "en": "Hello, World! Café résumé naïve.",
        "hi": "नमस्ते दुनिया। आपका दिन शुभ हो।",
        "es": "¡Hola, mundo! ¿Cómo está usted?",
        "fr": "Bonjour le monde ! Ça va bien ?",
        "ar": "مرحبا بالعالم. كيف حالك؟",
        "zh": "你好，世界。今天天气很好。",
    }
    all_roundtrip_ok = True
    for lang, name in LANGUAGES.items():
        text = unicode_tests[lang]
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        decoded_clean = decoded.replace("<|bos|>", "").replace("<|eos|>", "")
        match = decoded_clean == text
        all_roundtrip_ok = all_roundtrip_ok and match
        status = "OK" if match else "MISMATCH"
        print(f"  [{lang} / {name}] round-trip: {status}")
        if not match:
            print(f"    Original: {text!r}")
            print(f"    Decoded:  {decoded_clean!r}")

    print("\n--- Held-out test split loss/perplexity per language ---")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    test_losses = {}
    nan_inf_count = 0
    for lang, name in LANGUAGES.items():
        test_ds = PreparedSplitDataset(
            dataset_dir, "test", context_length=loaded.model_config.context_length
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
            loss = F.cross_entropy(logits, target_tensor[0], reduction="sum", ignore_index=PAD_TOKEN_ID)
            if not torch.isfinite(loss).item():
                nan_inf_count += 1
                print(f"  [{lang}] NON-FINITE LOSS at example {i}!")
            total_nll += float(loss.item())
            total_tokens += int((target_tensor[0] != PAD_TOKEN_ID).sum().item())

        mean_nll = total_nll / total_tokens if total_tokens else None
        ppl = float(torch.exp(torch.tensor(mean_nll))) if mean_nll is not None else None
        test_losses[lang] = {"nll": mean_nll, "ppl": ppl, "tokens": total_tokens}
        nll_s = f"{mean_nll:.4f}" if mean_nll is not None else "N/A"
        ppl_s = f"{ppl:.1f}" if ppl is not None else "N/A"
        print(f"  [{lang} / {name}] examples={count} tokens={total_tokens} "
              f"nll={nll_s} perplexity={ppl_s}")

    print(f"\n  Non-finite losses found across all test examples: {nan_inf_count}")

    print("\n" + "=" * 70)
    print("TASK 5: Issues detected")
    print("=" * 70)
    issues = []
    for lang, gen in generations.items():
        if not gen.strip():
            issues.append(f"[{lang}] Empty generation")
        if has_repetitive_output(gen):
            issues.append(f"[{lang}] Repetitive output: {gen[:60]!r}")
        non_script, total = count_off_script_chars(gen, lang)
        if total > 0 and non_script / total > 0.5:
            issues.append(f"[{lang}] High off-script character ratio ({non_script}/{total})")
    if not all_roundtrip_ok:
        issues.append("Tokenizer round-trip mismatch for some languages")
    if nan_inf_count > 0:
        issues.append(f"{nan_inf_count} non-finite losses on test split")
    if issues:
        for issue in issues:
            print(f"  ISSUE: {issue}")
    else:
        print("  No critical issues detected.")
    print()

    results = {
        "generations": {lang: generations[lang] for lang in LANGUAGES},
        "tokenizer_roundtrip_ok": all_roundtrip_ok,
        "test_losses": {lang: test_losses[lang] for lang in LANGUAGES},
        "non_finite_losses": nan_inf_count,
        "issues": issues,
    }
    (REPO / "data/reports/phase8").mkdir(parents=True, exist_ok=True)
    with open(REPO / "data/reports/phase8/generation_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Raw results saved to data/reports/phase8/generation_test_results.json")
    print("\nTASKS 4-5 COMPLETE")


if __name__ == "__main__":
    main()
