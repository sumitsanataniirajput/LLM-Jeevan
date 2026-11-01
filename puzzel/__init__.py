"""Puzzel: a from-scratch multilingual AI language model platform.

Puzzel is a professional AI intelligence platform built from scratch,
with support for multilingual text generation, RAG, tool-using agents,
authentication, monitoring, and more. All components are local-only;
no external AI APIs or pretrained weights are used.

Modules:
    puzzel.model       - Modern Transformer architecture (RoPE, RMSNorm, SwiGLU, GQA, KV-cache)
    puzzel.tokenizer   - UTF-8 byte and BPE tokenizers with language IDs
    puzzel.languages   - Unicode-based language detection
    puzzel.checkpoint  - Checkpoint save/load with v2 format and migration
    puzzel.inference   - Inference engine with streaming, batching, quantization
    puzzel.data        - Data pipeline, quality, prepared datasets
    puzzel.training    - Training loop, hardware-aware configs
    puzzel.evaluation  - Evaluation runner, metrics, reports
    puzzel.rag         - Retrieval-augmented generation
    puzzel.agents      - Tool registry, sandboxed execution, multi-step agents
    puzzel.auth        - Authentication, authorization, security
    puzzel.database    - SQLite database models, migrations, backup
    puzzel.api         - FastAPI REST API
    puzzel.memory      - User preferences, system prompts
    puzzel.monitoring  - Health, analytics, feedback, safety
    puzzel.admin       - Admin dashboard
    puzzel.offline     - Offline mode
"""

from puzzel.checkpoint import load_checkpoint, save_checkpoint
from puzzel.config import (
    DatasetConfig,
    RunConfig,
    TokenizerConfig,
    TrainingConfig,
)
from puzzel.inference import (
    GenerationConfig,
    InferenceEngine,
    batch_generate,
    stream_generate,
    stream_generate_cumulative,
)
from puzzel.languages import (
    detect_language,
    detect_language_confident,
    get_language_name,
    is_supported_language,
)
from puzzel.model import (
    CausalSelfAttention,
    GroupedQueryAttention,
    KVCache,
    KVCacheManager,
    LegacyTransformerLM,
    ModelConfig,
    RMSNorm,
    RotaryPositionEmbeddings,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    count_parameters,
    get_profile,
    profile_names,
)
from puzzel.tokenizer import (
    BPETokenizer,
    ByteTokenizer,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    LANGUAGE_ID_TOKENS,
    tokenizer_from_dict,
)
from puzzel.training.loop import resolve_device, train_model

__version__ = "0.2.0"

__all__ = [
    # Core config
    "RunConfig",
    "TokenizerConfig",
    "TrainingConfig",
    "DatasetConfig",
    "__version__",
    # Model
    "ModelConfig",
    "TransformerLM",
    "CausalSelfAttention",
    "KVCacheManager",
    "LegacyTransformerLM",
    "GroupedQueryAttention",
    "KVCache",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionEmbeddings",
    "TransformerBlock",
    "count_parameters",
    "get_profile",
    "profile_names",
    # Tokenizer
    "BPETokenizer",
    "ByteTokenizer",
    "SPECIAL_TOKENS",
    "VOCAB_SIZE",
    "LANGUAGE_ID_TOKENS",
    "tokenizer_from_dict",
    # Languages
    "detect_language",
    "detect_language_confident",
    "get_language_name",
    "is_supported_language",
    # Checkpoint
    "load_checkpoint",
    "save_checkpoint",
    # Inference
    "GenerationConfig",
    "InferenceEngine",
    "batch_generate",
    "stream_generate",
    "stream_generate_cumulative",
    # Training
    "resolve_device",
    "train_model",
]