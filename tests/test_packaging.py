"""Tests for Prompt 2: packaging, extras, guarded HF imports, lock file, CI."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_import_puzzel_works_without_hf_extra():
    # This test environment does not install the [hf] extra; importing the
    # package must still succeed (the scratch track has no HF imports).
    import puzzel  # noqa: F401

    assert puzzel is not None


def test_hf_guard_raises_clear_error_when_extra_missing():
    from puzzel import optional_deps

    if optional_deps.hf_available():
        pytest.skip("[hf] extra is installed in this environment")
    assert optional_deps.hf_available() is False
    with pytest.raises(ImportError, match=r"\[hf\]"):
        optional_deps.require_hf("loading a Hugging Face base model")


def test_pyproject_declares_all_extras():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert set(extras) >= {"serve", "hf", "dev"}
    assert any(p.startswith("fastapi") for p in extras["serve"])
    assert any(p.startswith("uvicorn") for p in extras["serve"])
    for pkg in (
        "transformers",
        "peft",
        "accelerate",
        "datasets",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "trl",
    ):
        assert any(p.split(">=")[0].split("<")[0] == pkg for p in extras["hf"]), pkg
    assert any(p.startswith("pytest") for p in extras["dev"])
    assert any(p.startswith("ruff") for p in extras["dev"])
    # Bounds are pinned: every extra dependency has a lower and upper bound.
    for group in ("serve", "hf", "dev"):
        for dep in extras[group]:
            assert ">=" in dep and "<" in dep, dep


def test_requirements_files_are_thin_pointers():
    req = (REPO_ROOT / "requirements.txt").read_text()
    req_dev = (REPO_ROOT / "requirements-dev.txt").read_text()
    assert "-e ." in req and "torch>=" not in req
    assert "-e .[dev,serve]" in req_dev


def test_lock_file_exists_and_covers_hf_extra():
    lock = REPO_ROOT / "uv.lock"
    assert lock.exists()
    text = lock.read_text()
    for pkg in ("transformers", "peft", "accelerate", "datasets", "fastapi", "ruff", "torch"):
        assert f'name = "{pkg}"' in text, pkg


def test_ci_workflow_runs_ruff_and_pytest_on_311_and_312():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert '"3.11"' in ci and '"3.12"' in ci
    assert 'pip install -e ".[dev,serve]"' in ci
    assert "ruff check" in ci
    assert "pytest" in ci
