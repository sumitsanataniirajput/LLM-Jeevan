# LLM-Jeevan / Puzzel — Risk and Compatibility Analysis

**Branch:** `arena/26b70af7-llm-jeevan`  
**Analysis date:** 2026-10-31  
**Prepared before implementation** (STEP 4 of PART 13 workflow)

---

## 1. Compatibility Risks

### 1.1 Checkpoint Incompatibility (High Risk)

**Risk:** New architecture (RoPE, RMSNorm, SwiGLU, GQA, different d_model/layers/heads) changes parameter shapes and tensor names. Old `puzzel-checkpoint-v1` checkpoints cannot be loaded directly into new models.

**Mitigation:**
- Add checkpoint format versioning (`puzzel-checkpoint-v2`)
- Create `puzzel/checkpoint_migration.py` with shape adaptation utilities
- Preserve `puzzel-checkpoint-v1` loader for legacy checkpoints
- Add `puzzel/model/legacy_adapter.py` for loading old weights into compatible submodules
- Migration script: `scripts/migrate_checkpoint.py`
- Never silently overwrite old checkpoints; require explicit migration

### 1.2 Tokenizer Vocabulary Change (Medium Risk)

**Risk:** Expanding vocab from 2,048 to 8K-32K changes embedding matrix shape. Old checkpoints with vocab=2048 cannot load into models with vocab=8000+.

**Mitigation:**
- Keep `ByteTokenizer` (vocab=260) as immutable fallback
- New `BPETokenizer` with configurable vocab; old BPE vocab=2048 preserved in `data/prepared/tokenizer.json`
- Checkpoint migration: pad/truncate embeddings for vocab changes (with explicit user opt-in)
- Tokenizer versioning in saved format

### 1.3 Config Precision Restriction (Medium Risk)

**Risk:** `TrainingConfig.__post_init__` rejects non-fp32 precision. Adding fp16/bf16 support requires config changes.

**Mitigation:**
- Modify `TrainingConfig` to accept `fp16`, `bf16`, `fp32`
- Add hardware capability check before enabling mixed precision
- fp32 remains default for CPU-only environments

### 1.4 API Backward Compatibility (Low Risk)

**Risk:** Adding auth, rate limiting, and validation to existing endpoints could break existing clients.

**Mitigation:**
- Auth is opt-in (disabled by default for local development)
- Existing endpoints preserve response schemas
- New fields are additive only
- `create_app()` factory preserved for testing

---

## 2. Implementation Risks

### 2.1 Hardware Limitations (Critical Risk)

**Current environment:** 2 CPU cores, ~3.8 GB RAM, no GPU/MPS, ~12 GB disk free

**Risk:** Training models larger than ~5M parameters on this hardware will be extremely slow or fail due to memory constraints. The Medium config (vocab=8000, d_model=256, layers=6) has ~33M parameters — likely too large for comfortable CPU training.

**Mitigation:**
- Hardware-aware configuration profiles (CPU-safe tiny/small only by default)
- `recommend_safe_batch_size()` already exists; enforce it
- Mixed precision only when CUDA/MPS available
- Gradient accumulation for effective larger batches on constrained hardware
- Explicit hardware check before any training (no automatic large training)
- Document hardware requirements per model profile

### 2.2 Dataset Size (Critical Risk)

**Current corpus:** 174K tokens across 6 languages, 49 documents

**Risk:** 174K tokens is insufficient for any capable language model. Modern multilingual models need billions of tokens. Training on this corpus will produce a toy model regardless of architecture improvements.

**Mitigation:**
- Dataset expansion is Phase A priority (before any training)
- No training on current corpus beyond verification runs
- Public domain sources only (Project Gutenberg, Wikisource)
- Minimum 5-10M tokens per language recommended before medium model training

### 2.3 No Trained Checkpoint Exists (High Risk)

**Risk:** The specification references loading checkpoints for inference, but no `.pt` file exists. The FastAPI server will fail on startup without a checkpoint.

**Mitigation:**
- API startup handles missing checkpoint gracefully (503 on /generate, model_loaded=false on /health)
- Training must be run before inference is functional
- Clear error messages when checkpoint is missing
- Do not fabricate or claim existence of trained weights

### 2.4 Frontend-Backend Integration (Medium Risk)

**Risk:** The frontend must connect to actual backend operations, not be a mock.

**Mitigation:**
- All frontend API calls connect to real FastAPI endpoints
- Backend provides CORS configuration for frontend origin
- Environment variable for API base URL (`PUZZEL_API_BASE_URL`)
- Connection error handling and status display in frontend
- Health check endpoint for connectivity verification

### 2.5 Security Risks (High Risk)

**Risk:** API key auth, user auth, RBAC, file upload, code execution — all introduce security vulnerabilities if implemented incorrectly.

**Mitigation:**
- Password hashing with bcrypt/argon2 (not plain text)
- Short-lived JWT tokens
- Environment variables for secrets (`.env.example` template)
- File upload validation (type, size, content scanning)
- Sandboxed code execution in isolated subprocess (never in main process)
- Authorization at backend, not frontend-only
- Rate limiting and abuse detection
- Input sanitization on all endpoints

### 2.6 RAG Security (Medium Risk)

**Risk:** Document retrieval could leak information across users/KBs if permissions are not enforced.

**Mitigation:**
- User-specific permissions on knowledge bases
- Document-level access control
- No cross-tenant retrieval without explicit authorization
- Complete document deletion from index

### 2.7 Agent/Tool Security (High Risk)

**Risk:** Tool execution, especially sandboxed Python, could escape isolation and harm the host.

**Mitigation:**
- Never execute model-generated Python in the main backend process
- Isolated subprocess execution with timeouts
- Restricted file access, network access, execution time
- Tool argument validation
- Authorization for sensitive actions
- Agent iteration limits to prevent infinite loops
- AI only claims action when tool actually succeeded

### 2.8 No Fabricated Results (Non-negotiable)

**Risk:** Claiming capabilities without measurable evaluation violates project rules.

**Mitigation:**
- All benchmarks produce actual measurements, not claims
- Null/None for unmeasured metrics (never convert to zero)
- Human rubric for creative writing (no AI judge)
- Distinguish automated metrics from human evaluation
- No external AI APIs for evaluation

---

## 3. Migration Path Summary

| Change | Old State | New State | Migration Required |
|---|---|---|---|
| Model architecture | 2 layers, d_model=64, MHA, LayerNorm, GELU | Configurable: RoPE, RMSNorm, SwiGLU, GQA | Checkpoint migration (shape adaptation) |
| Tokenizer vocab | 2,048 (BPE) | 8K-32K (configurable) | Embedding resize with opt-in |
| Precision | fp32 only | fp32/fp16/bf16 | Config change (no data migration) |
| Checkpoint format | puzzel-checkpoint-v1 | puzzel-checkpoint-v2 | Format versioning + migration script |
| API | No auth, no rate limiting | Auth + rate limiting (opt-in) | No breaking changes (opt-in features) |
| Frontend | None | HTML/CSS/JS chat app | New (no migration) |
| Database | None | SQLite/PostgreSQL | New (no migration) |
| RAG | None | Full RAG pipeline | New (no migration) |
| Agents | None | Tool registry + sandbox | New (no migration) |

---

## 4. Rollback Plan

1. **Backup:** Full project backup at `/home/user/LLM-Jeevan-backup-20261031-180933`
2. **Git:** All changes on branch `arena/26b70af7-llm-jeevan`; can revert with `git reset`
3. **Checkpoints:** Old checkpoints never overwritten during migration
4. **Config:** Old TOML configs preserved; new configs additive
5. **API:** Existing endpoints preserved; new auth is opt-in

---

## 5. Approval Gates (Before Implementation)

| Gate | Description | Required Before |
|---|---|---|
| G1 | Dataset expansion plan approved | Step 7 (tokenizer/dataset upgrades) |
| G2 | Hardware path confirmed (GPU or CPU-only) | Step 8 (training implementation) |
| G3 | Architecture targets selected (tiny/small/medium/large) | Step 6 (model modernization) |
| G4 | Security review of auth/RBAC | Step 12 (API and security) |
| G5 | Frontend-backend connection tested | Step 13 (frontend implementation) |

---

## 6. Known Pre-existing Issues (Carried Forward)

1. **3 Phase 9 regression tests fail** against current baseline (documented in `tests/test_phase9_regressions.py`):
   - `test_no_cross_language_mixing` — zh comma flagged as off-script (known false positive)
   - `test_generation_non_empty_rate_threshold` — Spanish empty generation
   - `test_no_repetitive_single_token_dominance` — repetitive output in all languages

2. **No trained checkpoint** — inference API cannot function without training first

3. **Current corpus too small** — 174K tokens insufficient for capable model

4. **CPU-only hardware** — limits model size to ~5M parameters for practical training

These are documented issues, not new risks introduced by the upgrade.
