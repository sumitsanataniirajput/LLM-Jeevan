"""Guards for optional dependency groups.

The ``[hf]`` extra (transformers, peft, accelerate, datasets, ...) is optional:
``import puzzel`` and the whole scratch track must work without it. Any module
that needs a Hugging Face package must import it through :func:`require_hf` (or
check :func:`hf_available` first) so the failure message is clear instead of a
raw ``ModuleNotFoundError`` deep inside a stack trace.
"""

from __future__ import annotations

import importlib.util

HF_PACKAGES = ("transformers", "peft", "accelerate", "datasets", "huggingface_hub")


def hf_available() -> bool:
    """Return True if the core Hugging Face packages of the [hf] extra import."""
    return all(importlib.util.find_spec(name) is not None for name in HF_PACKAGES)


def require_hf(feature: str) -> None:
    """Raise a clear ImportError if the [hf] extra is not installed.

    Args:
        feature: Short description of the feature that needs the extra,
            e.g. "loading a Hugging Face base model".

    Raises:
        ImportError: If any [hf] package is missing, with install instructions.
    """
    missing = [name for name in HF_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"{feature} requires the optional Hugging Face extra, but these "
            f"packages are not installed: {', '.join(missing)}. "
            f"Install with: pip install -e \".[hf]\""
        )
