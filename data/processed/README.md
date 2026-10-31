# Prepared datasets

Prepared datasets are generated locally and excluded from Git. The `multilingual-v1/` scaffold is pre-created with a `.gitkeep`; the data preparation command can populate this empty scaffold only after a successful build and verification. No prepared dataset is present. No preparation, training, or pretraining is run automatically.

A prepared multilingual output contains:

- `train.bin`, `validation.bin`, `test.bin` — fixed-width, per-document int32 next-token rows. Rows do not cross document boundaries; final rows are padded with the tokenizer PAD ID.
- `tokenizer.json` — the custom byte-level BPE vocabulary and merge rules, with UTF-8 byte fallback IDs 0–255 and reserved special IDs 256–259.
- `manifest.json` — model/tokenizer settings, context size, target sampling weights, split files, per-language example ranges, and binary format.
- `stats.json` — total and per-language document/character/UTF-8-byte/token/example counts, split distribution, target/actual vocabulary audit, duplicates, and ingestion/quality results.
- `documents.sqlite3` — document hashes, split, language, source, license, and non-text counts only; original text is not copied into the prepared output.

Use `PreparedSplitDataset(path, "train")` (or `"validation"` / `"test"`) to memory-map examples without loading the complete split into RAM. Run the `stats`, `verify`, and `report` commands documented in `data/raw/README.md` before considering training. The weighted multilingual sampler requires a verified v2 dataset with all configured language ranges present in each split.
