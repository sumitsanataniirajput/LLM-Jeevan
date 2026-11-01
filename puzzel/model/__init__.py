"""Puzzel model package: Transformer architecture and configuration.

Exports:
    - ModelConfig: validated architecture hyperparameters
    - TransformerLM: modernized decoder-only language model
    - LegacyTransformerLM: original architecture for v1 checkpoint loading
    - MODEL_PROFILES / get_profile / profile_names: predefined size profiles
    - RoPE, RMSNorm, SwiGLU, GQA, KV-cache components
"""

from puzzel.model.config import (
    ModelConfig,
    ModelProfile,
    MODEL_PROFILES,
    get_profile,
    profile_names,
)
from puzzel.model.legacy_adapter import (
    LegacyCausalSelfAttention,
    LegacyFeedForward,
    LegacyTransformerBlock,
    LegacyTransformerLM,
    count_parameters as legacy_count_parameters,
)
from puzzel.model.grouped_query_attention import GroupedQueryAttention
from puzzel.model.kv_cache import KVCache, KVCacheManager
from puzzel.model.rope import RotaryPositionEmbeddings, apply_rotary_pos_emb
from puzzel.model.rmsnorm import RMSNorm
from puzzel.model.swiglu import SwiGLU, swiglu_feed_forward
from puzzel.model.transformer import (
    TransformerLM,
    CausalSelfAttention,
    FeedForward,
    TransformerBlock,
    AttentionNorm,
    count_parameters,
)

__all__ = [
    # Config
    "ModelConfig",
    "ModelProfile",
    "MODEL_PROFILES",
    "get_profile",
    "profile_names",
    # Modern model
    "TransformerLM",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "AttentionNorm",
    # Legacy model (v1 checkpoint compatibility)
    "LegacyTransformerLM",
    "LegacyCausalSelfAttention",
    "LegacyFeedForward",
    "LegacyTransformerBlock",
    # Components
    "GroupedQueryAttention",
    "KVCache",
    "KVCacheManager",
    "RotaryPositionEmbeddings",
    "apply_rotary_pos_emb",
    "RMSNorm",
    "SwiGLU",
    "swiglu_feed_forward",
    # Parameter counting
    "count_parameters",
    "legacy_count_parameters",
]