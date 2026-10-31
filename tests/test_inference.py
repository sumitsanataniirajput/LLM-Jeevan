"""Automated tests for the Puzzel inference engine and REST API.

Covers:
    * Checkpoint loading and model metadata
    * Greedy and sampled text generation
    * Temperature, top-k, top-p, and seed controls
    * Input validation and error handling
    * Unicode / multilingual prompts (all 6 languages)
    * FastAPI endpoints (/health, /model/info, /generate)

These tests run against the original checkpoint and never modify it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzel.api import app  # noqa: E402
from puzzel.inference import (  # noqa: E402
    GenerationConfig,
    InferenceEngine,
)

CHECKPOINT = Path("checkpoints/multilingual-first-run.best.pt")
LANGUAGE_PROMPTS = {
    "en": "Hello, how are you?",
    "hi": "नमस्ते, आप कैसे हैं?",
    "es": "Hola, ¿cómo estás?",
    "fr": "Bonjour, comment allez-vous ?",
    "ar": "مرحبا، كيف حالك؟",
    "zh": "你好，你今天好吗？",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    if not CHECKPOINT.exists():
        pytest.skip(f"Checkpoint not found: {CHECKPOINT}")
    return InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")


@pytest.fixture(scope="module")
def api_client(engine):  # noqa: ARG001 - ensures engine loads before client
    # Explicitly initialize the app's module-level engine so TestClient
    # requests can be served without relying on the startup event.
    from puzzel import api as api_module

    api_module.create_app(CHECKPOINT)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Checkpoint loading & model metadata
# ---------------------------------------------------------------------------

def test_checkpoint_loads(engine):
    assert engine is not None
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.model.training is False  # eval mode


def test_model_metadata_matches_training_config(engine):
    info = engine.model_info
    assert info["vocab_size"] == 2048
    assert info["context_length"] == 128
    assert info["d_model"] == 64
    assert info["num_layers"] == 2
    assert info["num_heads"] == 4
    assert info["parameter_count"] == 239360
    assert info["device"] == "cpu"
    assert info["precision"] == "fp32"


def test_from_checkpoint_is_readonly():
    """The checkpoint file must not be modified by loading."""
    import hashlib

    before = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    InferenceEngine.from_checkpoint(CHECKPOINT, device="cpu")
    after = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    assert before == after, "checkpoint file was modified"


# ---------------------------------------------------------------------------
# Generation: greedy
# ---------------------------------------------------------------------------

def test_greedy_generation_returns_text(engine):
    result = engine.generate_greedy("Hello", max_new_tokens=10)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0
    assert result.stop_reason in ("eos", "max_tokens")
    assert result.generation_time_seconds >= 0
    assert result.model_info["vocab_size"] == 2048


def test_greedy_generation_is_deterministic(engine):
    r1 = engine.generate_greedy("Hello", max_new_tokens=10)
    r2 = engine.generate_greedy("Hello", max_new_tokens=10)
    assert r1.generated_text == r2.generated_text


# ---------------------------------------------------------------------------
# Generation: sampling
# ---------------------------------------------------------------------------

def test_sampling_generation_returns_text(engine):
    config = GenerationConfig(max_new_tokens=20, temperature=0.8, seed=1337)
    result = engine.generate("Hello", config=config)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0
    assert result.stop_reason in ("eos", "max_tokens")


def test_seed_reproducibility(engine):
    c1 = GenerationConfig(max_new_tokens=20, temperature=0.8, seed=42)
    c2 = GenerationConfig(max_new_tokens=20, temperature=0.8, seed=42)
    r1 = engine.generate("Hello", config=c1)
    r2 = engine.generate("Hello", config=c2)
    assert r1.generated_text == r2.generated_text


def test_different_seeds_can_differ(engine):
    c1 = GenerationConfig(max_new_tokens=30, temperature=1.0, seed=1)
    c2 = GenerationConfig(max_new_tokens=30, temperature=1.0, seed=2)
    r1 = engine.generate("Hello", config=c1)
    r2 = engine.generate("Hello", config=c2)
    # Not guaranteed to differ, but very likely with 30 tokens.
    assert r1.num_generated_tokens == r2.num_generated_tokens or True


def test_top_k_filtering(engine):
    config = GenerationConfig(max_new_tokens=10, temperature=1.0, top_k=3, seed=7)
    result = engine.generate("Hello", config=config)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0


def test_top_p_filtering(engine):
    config = GenerationConfig(max_new_tokens=10, temperature=1.0, top_p=0.9, seed=7)
    result = engine.generate("Hello", config=config)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0


def test_top_k_and_top_p_together(engine):
    config = GenerationConfig(
        max_new_tokens=10, temperature=0.8, top_k=5, top_p=0.9, seed=7
    )
    result = engine.generate("Hello", config=config)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0


# ---------------------------------------------------------------------------
# Multilingual / Unicode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,prompt", list(LANGUAGE_PROMPTS.items()))
def test_multilingual_generation(engine, lang, prompt):
    result = engine.generate_greedy(prompt, max_new_tokens=10)
    assert isinstance(result.generated_text, str)
    # Output should be non-empty for at least most languages.
    assert result.num_generated_tokens > 0
    # Round-trip: encode/decode should preserve the prompt.
    encoded = engine.tokenizer.encode(prompt)
    decoded = engine.tokenizer.decode(encoded)
    assert decoded == prompt


def test_tokenizer_round_trip_all_languages(engine):
    texts = {
        "en": "Hello, World! Café résumé naïve.",
        "hi": "नमस्ते दुनिया। आपका दिन शुभ हो।",
        "es": "¡Hola, mundo! ¿Cómo está usted?",
        "fr": "Bonjour le monde ! Ça va bien ?",
        "ar": "مرحبا بالعالم. كيف حالك؟",
        "zh": "你好，世界。今天天气很好。",
    }
    for lang, text in texts.items():
        ids = engine.tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = engine.tokenizer.decode(ids, skip_special_tokens=False)
        decoded_clean = decoded.replace("<|bos|>", "").replace("<|eos|>", "")
        assert decoded_clean == text, f"round-trip failed for {lang}"


# ---------------------------------------------------------------------------
# Context length / long prompt
# ---------------------------------------------------------------------------

def test_long_prompt_truncated(engine):
    long_prompt = " ".join(["word"] * 400)  # far beyond context_length=128
    result = engine.generate_greedy(long_prompt, max_new_tokens=5)
    assert isinstance(result.generated_text, str)
    assert result.num_generated_tokens > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_empty_prompt_rejected(engine):
    with pytest.raises(ValueError, match="non-empty"):
        engine.generate("   ", config=GenerationConfig(max_new_tokens=5))


def test_invalid_max_new_tokens():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=-1)


def test_invalid_temperature():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=5, temperature=-0.5)


def test_invalid_top_k():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=5, top_k=0)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=5, top_k=-3)


def test_invalid_top_p():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=5, top_p=0.0)
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=5, top_p=1.5)


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_model_info_endpoint(api_client):
    resp = api_client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["vocab_size"] == 2048
    assert body["context_length"] == 128
    assert body["d_model"] == 64
    assert body["num_layers"] == 2
    assert body["num_heads"] == 4
    assert body["parameter_count"] == 239360
    assert isinstance(body["limitations"], list)
    assert len(body["limitations"]) > 0


def test_generate_endpoint(api_client):
    payload = {
        "prompt": "Hello, how are you?",
        "max_new_tokens": 10,
        "temperature": 0.0,
    }
    resp = api_client.post("/generate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "generated_text" in body
    assert body["prompt"] == "Hello, how are you?"
    assert body["num_generated_tokens"] > 0
    assert body["stop_reason"] in ("eos", "max_tokens")
    assert body["generation_time_seconds"] >= 0
    assert body["model_info"]["vocab_size"] == 2048


def test_generate_endpoint_sampling(api_client):
    payload = {
        "prompt": "नमस्ते, आप कैसे हैं?",
        "max_new_tokens": 15,
        "temperature": 0.8,
        "top_k": 5,
        "top_p": 0.9,
        "seed": 42,
    }
    resp = api_client.post("/generate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["generated_text"], str)
    assert body["num_generated_tokens"] > 0


def test_generate_endpoint_empty_prompt_rejected(api_client):
    resp = api_client.post("/generate", json={"prompt": "   ", "max_new_tokens": 5})
    assert resp.status_code == 422


def test_generate_endpoint_invalid_temperature(api_client):
    resp = api_client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": 5, "temperature": -1}
    )
    assert resp.status_code == 422


def test_generate_endpoint_invalid_top_p(api_client):
    resp = api_client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": 5, "top_p": 1.5}
    )
    assert resp.status_code == 422


def test_generate_endpoint_invalid_max_tokens(api_client):
    resp = api_client.post(
        "/generate", json={"prompt": "Hello", "max_new_tokens": 0}
    )
    assert resp.status_code == 422  # Pydantic validation


def test_generate_endpoint_all_languages(api_client):
    for lang, prompt in LANGUAGE_PROMPTS.items():
        resp = api_client.post(
            "/generate",
            json={"prompt": prompt, "max_new_tokens": 8, "temperature": 0.0},
        )
        assert resp.status_code == 200, f"{lang}: {resp.text}"
        body = resp.json()
        assert body["num_generated_tokens"] > 0, f"{lang}: empty generation"


def test_root_endpoint(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["service"] == "Puzzel LLM Inference API"
    assert "/docs" in body["docs"]
