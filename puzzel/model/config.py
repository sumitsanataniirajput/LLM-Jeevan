"""Validated architecture hyperparameters for the Puzzel Transformer.

Supports the original architecture (MHA, LayerNorm, GELU, learned absolute
positions) as a legacy mode, plus modern options (RoPE, RMSNorm, SwiGLU,
GQA, KV-cache) configurable through ``ModelConfig``. New fields are
optional with defaults matching the original behavior, so existing TOML
configurations continue to load unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from puzzel.tokenizer import PAD_TOKEN_ID, VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings for a causal decoder language model.

    Architecture options:
        - ``use_rope``: Rotary position embeddings instead of learned
          absolute positions.
        - ``use_rmsnorm``: RMSNorm instead of LayerNorm.
        - ``use_swiglu``: SwiGLU activation instead of GELU.
        - ``use_gqa``: Grouped Query Attention instead of standard MHA.
        - ``use_kv_cache``: Enable KV-cache for inference.

    Predefined size profiles (select via ``profile`` or set fields manually):
        - "tiny":  ~1-5M parameters, CPU-friendly.
        - "small": ~33M parameters, main development target.
        - "medium": ~60M parameters, GPU-oriented.
    """

    # --- Core dimensions ---
    vocab_size: int = VOCAB_SIZE
    context_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4

    # --- Attention configuration ---
    num_kv_heads: int | None = None  # None = standard MHA (num_heads)
    head_dim: int | None = None  # None = d_model // num_heads
    dropout: float = 0.1
    attention_dropout: float | None = None  # None = dropout
    bias: bool = False

    # --- Modern architecture flags ---
    use_rope: bool = False
    use_rmsnorm: bool = False
    use_swiglu: bool = False
    use_gqa: bool = False
    use_kv_cache: bool = False

    # --- RoPE settings ---
    rope_base: float = 10000.0
    rope_max_position_embeddings: int = 8192

    # --- Output / misc ---
    pad_token_id: int = PAD_TOKEN_ID
    tie_embeddings: bool = True
    precision: str = "fp32"

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if self.head_dim is not None:
            if (
                not isinstance(self.head_dim, int)
                or isinstance(self.head_dim, bool)
                or self.head_dim <= 0
            ):
                raise ValueError("head_dim must be a positive integer when set")
            if self.num_heads * self.head_dim != self.d_model:
                raise ValueError("num_heads * head_dim must equal d_model")

        # GQA validation
        if self.num_kv_heads is not None:
            if (
                not isinstance(self.num_kv_heads, int)
                or isinstance(self.num_kv_heads, bool)
                or self.num_kv_heads <= 0
            ):
                raise ValueError("num_kv_heads must be a positive integer when set")
            if self.num_kv_heads > self.num_heads:
                raise ValueError("num_kv_heads must not exceed num_heads")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError("num_heads must be divisible by num_kv_heads")
            if self.num_kv_heads < self.num_heads and not self.use_gqa:
                raise ValueError(
                    "num_kv_heads < num_heads requires use_gqa=True"
                )

        if self.use_gqa and self.num_kv_heads is None:
            raise ValueError("use_gqa=True requires num_kv_heads to be set")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.attention_dropout is not None and not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in the range [0, 1)")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a valid vocabulary ID")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")
        if self.rope_max_position_embeddings <= 0:
            raise ValueError("rope_max_position_embeddings must be positive")
        if not isinstance(self.use_rope, bool):
            raise ValueError("use_rope must be a boolean")
        if not isinstance(self.use_rmsnorm, bool):
            raise ValueError("use_rmsnorm must be a boolean")
        if not isinstance(self.use_swiglu, bool):
            raise ValueError("use_swiglu must be a boolean")
        if not isinstance(self.use_gqa, bool):
            raise ValueError("use_gqa must be a boolean")
        if not isinstance(self.use_kv_cache, bool):
            raise ValueError("use_kv_cache must be a boolean")
        if not isinstance(self.bias, bool):
            raise ValueError("bias must be a boolean")

    @property
    def effective_num_kv_heads(self) -> int:
        """Return the number of KV heads (num_heads if GQA is disabled)."""
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def effective_head_dim(self) -> int:
        """Return the per-head dimension."""
        if self.head_dim is not None:
            return self.head_dim
        return self.d_model // self.num_heads

    @property
    def effective_attention_dropout(self) -> float:
        """Return the attention dropout (defaults to dropout)."""
        return self.attention_dropout if self.attention_dropout is not None else self.dropout

    @property
    def parameter_count_estimate(self) -> int:
        """Estimate the number of unique trainable parameters without allocation.

        Matches the actual parameter count of ``TransformerLM`` for the given
        configuration, including bias settings and normalization type.
        """
        width = self.d_model
        expanded = width * self.mlp_ratio
        embeddings = self.vocab_size * width + self.context_length * width

        if self.use_swiglu:
            hidden = (8 * width) // 3
            # QKV projections (q, k, v) + output projection
            attention_params = (
                width * self.num_heads * self.head_dim  # q
                + width * self.effective_num_kv_heads * self.head_dim  # k
                + width * self.effective_num_kv_heads * self.head_dim  # v
                + width * self.num_heads * self.head_dim  # output
            )
            if self.bias:
                attention_params += (
                    self.num_heads * self.head_dim  # q bias
                    + self.effective_num_kv_heads * self.head_dim  # k bias
                    + self.effective_num_kv_heads * self.head_dim  # v bias
                    + width  # output bias
                )
            # SwiGLU: gate + up + down
            ffn_params = width * hidden + width * hidden + hidden * width
            if self.bias:
                ffn_params += hidden + hidden + width
        else:
            # Standard MHA: q, k, v, output projections
            attention_params = 4 * width * width
            if self.bias:
                attention_params += 4 * width
            # GELU MLP: up + down
            ffn_params = width * expanded + expanded * width
            if self.bias:
                ffn_params += expanded + width

        # Norms per block: 2 norms (attention + MLP)
        if self.use_rmsnorm:
            norm_params_per_block = 2 * width  # weight only
        else:
            norm_params_per_block = 4 * width  # weight + bias

        blocks_per_layer = attention_params + ffn_params + norm_params_per_block

        # Final norm
        if self.use_rmsnorm:
            final_norm = width  # weight only
        else:
            final_norm = 2 * width  # weight + bias

        output_head = 0 if self.tie_embeddings else self.vocab_size * width
        if not self.tie_embeddings and self.bias:
            output_head += self.vocab_size

        return embeddings + self.num_layers * blocks_per_layer + final_norm + output_head


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A named model size configuration."""

    name: str
    description: str
    config: ModelConfig

    @property
    def estimated_parameters(self) -> int:
        return self.config.parameter_count_estimate


# Predefined profiles. Tiny is the default for CPU development.
MODEL_PROFILES: dict[str, ModelProfile] = {
    "tiny": ModelProfile(
        name="tiny",
        description="~1-5M parameters, CPU-friendly development and testing",
        config=ModelConfig(
            vocab_size=4096,
            context_length=512,
            d_model=256,
            num_layers=4,
            num_heads=8,
            num_kv_heads=2,
            use_gqa=True,
            use_rope=True,
            use_rmsnorm=True,
            use_swiglu=True,
            use_kv_cache=True,
            dropout=0.1,
            tie_embeddings=True,
            precision="fp32",
        ),
    ),
    "small": ModelProfile(
        name="small",
        description="~33M parameters, main development target",
        config=ModelConfig(
            vocab_size=8000,
            context_length=256,
            d_model=256,
            num_layers=6,
            num_heads=4,
            num_kv_heads=2,
            use_gqa=True,
            use_rope=True,
            use_rmsnorm=True,
            use_swiglu=True,
            use_kv_cache=True,
            dropout=0.1,
            tie_embeddings=True,
            precision="fp32",
        ),
    ),
    "medium": ModelProfile(
        name="medium",
        description="~60M parameters, GPU-oriented profile",
        config=ModelConfig(
            vocab_size=16000,
            context_length=512,
            d_model=512,
            num_layers=8,
            num_heads=8,
            num_kv_heads=4,
            use_gqa=True,
            use_rope=True,
            use_rmsnorm=True,
            use_swiglu=True,
            use_kv_cache=True,
            dropout=0.1,
            tie_embeddings=True,
            precision="fp32",
        ),
    ),
    "legacy": ModelProfile(
        name="legacy",
        description="Original architecture (MHA, LayerNorm, GELU, learned positions)",
        config=ModelConfig(
            vocab_size=2048,
            context_length=128,
            d_model=64,
            num_layers=2,
            num_heads=4,
            use_gqa=False,
            use_rope=False,
            use_rmsnorm=False,
            use_swiglu=False,
            use_kv_cache=False,
            dropout=0.1,
            tie_embeddings=True,
            precision="fp32",
        ),
    ),
}


def get_profile(name: str) -> ModelProfile:
    """Retrieve a predefined model profile by name.

    Args:
        name: Profile name (tiny, small, medium, legacy).

    Returns:
        The matching ModelProfile.

    Raises:
        ValueError: If the profile name is unknown.
    """
    if name not in MODEL_PROFILES:
        available = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(
            f"unknown model profile {name!r}; choose one of: {available}"
        )
    return MODEL_PROFILES[name]


def profile_names() -> list[str]:
    """Return sorted available profile names."""
    return sorted(MODEL_PROFILES)


__all__ = [
    "ModelConfig",
    "ModelProfile",
    "MODEL_PROFILES",
    "get_profile",
    "profile_names",
]