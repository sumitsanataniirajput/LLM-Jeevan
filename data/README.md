# Data layout and rebuild instructions

Puzzel never downloads a corpus automatically. Every dataset must be placed
there by you, with provenance sidecars, and every training run requires
explicit approval.

## Layout

| Path | Contents | Tracked in Git? |
| --- | --- | --- |
| `data/raw/<lang>/*.txt` + `*.txt.meta.json` | Your licensed source documents and provenance sidecars | **No** (ignored; `data/raw/README.md` is the only tracked file) |
| `data/raw/*.jsonl` | Multilingual JSONL records (`text`, `language`, `source`, `license`) | **No** (ignored) |
| `data/processed/<dataset>/` | Cleaned, split documents before tokenizer fitting | **No** (ignored) |
| `data/prepared/` | BPE tokenizer, `train/validation/test.bin`, manifest, stats, SQLite DB — all **build outputs** | **No** (ignored; see `data/prepared/README.md`) |
| `data/evaluation/*.jsonl` | Held-out evaluation suites (you provide these) | **No** (ignored; `data/evaluation/README.md` tracked) |
| `data/reports/license_inventory.csv` | Generated license inventory of `data/raw` | **Yes** (exception to the ignore rule) |
| `data/reports/sft_dataset_report.json` | Generated SFT dataset report | **Yes** (exception) |

The currently present corpus under `data/raw/` is public-domain /
Project Gutenberg material; see `data/reports/license_inventory.csv` for the
per-file audit and `scripts/fetch_data.sh` for the documented sources.

## Rebuilding `data/prepared` from sources

From the repository root, with your licensed files in place under
`data/raw/<lang>/` (each `.txt` with a sibling `.txt.meta.json`):

```bash
# 1. Audit licenses of everything under data/raw (writes the tracked CSV).
./.venv/bin/python -m puzzel.data.license_audit \
  --raw-dir data/raw --output data/reports/license_inventory.csv

# 2. Prepare: clean, dedupe, split by document, fit BPE on the train split only.
./scripts/data.sh prepare \
  --input data/raw \
  --output data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml

# 3. Verify + stats + report.
./scripts/data.sh verify --dataset data/processed/multilingual-v1
./scripts/data.sh stats  --dataset data/processed/multilingual-v1
./scripts/data.sh report \
  --dataset data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml \
  --output data/reports/multilingual-v1.json
```

`data/prepared/` (tokenizer.json, train/validation/test.bin, manifest.json,
stats.json, documents.sqlite3) is produced by the preparation pipeline from
`data/processed/` and is **never committed**; rebuild it with the commands
above. Do not edit prepared files by hand.

## Synthetic fixtures

Tests use tiny synthetic fixtures under `tests/fixtures/` only. Never point
the training pipeline at `tests/fixtures/`.
