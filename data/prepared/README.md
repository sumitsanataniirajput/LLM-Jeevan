# data/prepared — build outputs, not source

Everything in this directory (tokenizer.json, train.bin, validation.bin,
test.bin, manifest.json, stats.json, documents.sqlite3) is **generated** by
the data pipeline from `data/raw/` + `data/processed/`. It is git-ignored and
must never be committed or hand-edited.

Rebuild from sources — see [../README.md](../README.md) ("Rebuilding
data/prepared from sources"). Preparation, verification and training all
require explicit user approval; nothing here authorizes a training run.
