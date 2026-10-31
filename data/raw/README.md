# Local, original or properly licensed text

`data/raw/` is for text that you wrote/own, is public domain, or is covered by a license that permits the intended use. Do not add private/confidential text or data with unclear or incompatible rights. Puzzel does not download a corpus, validate legal rights, or contact an external data service. Keep the source, exact license/terms, required attribution, and restrictions for every document.

The repository contains **no real training corpus**. These folders are only a scaffold and are ignored by Git:

```text
data/raw/{en,hi,es,fr,ar,zh}/
```

Create the folders if needed:

```bash
mkdir -p data/raw/{en,hi,es,fr,ar,zh}
```

Copy only files you have already obtained and licensed for this use. For example, replace each source path below with your own existing file path (these commands do not fetch data):

```bash
cp -- /path/to/your/licensed-english.txt data/raw/en/
cp -- /path/to/your/licensed-hindi.txt data/raw/hi/
cp -- /path/to/your/licensed-spanish.txt data/raw/es/
cp -- /path/to/your/licensed-french.txt data/raw/fr/
cp -- /path/to/your/licensed-arabic.txt data/raw/ar/
cp -- /path/to/your/licensed-chinese.txt data/raw/zh/
```

The paths above are command templates only; do not create placeholder documents or train on example/template text. A source file placed in a language folder must be labeled with the same language code. Supported codes are exactly `en` (English), `hi` (Hindi), `es` (Spanish), `fr` (French), `ar` (Arabic), and `zh` (Chinese).

## TXT documents and provenance sidecars

- A `.txt` or `.TXT` file is one document. It must be valid UTF-8 (an optional UTF-8 BOM is accepted). Unicode characters, punctuation, and paragraph breaks are preserved; normalization is NFC and horizontal whitespace is normalized.
- Every multilingual TXT document needs a sibling UTF-8 JSON sidecar named by appending `.meta.json` to the full filename. For `article.txt`, use `article.txt.meta.json`.
- The sidecar must be one JSON object with non-empty string fields `language`, `source`, and `license`. `licence` is accepted as an alias for `license`; prefer the American spelling `license`.

Exact sidecar schema (replace the angle-bracket values with the real document's information):

```json
{
  "language": "en",
  "source": "<original source title or URL>",
  "license": "<exact license identifier or terms>"
}
```

The same schema is used for the other five language codes. `source` should let you trace the text to its origin; `license` should record the actual permission/terms, not merely say that a document is “safe.” The pipeline records these strings but does not decide whether they are legally sufficient.

## JSONL / NDJSON records

`.jsonl` and `.ndjson` files are UTF-8 (an optional UTF-8 BOM is accepted), with **one complete JSON object on each non-empty line**. The exact multilingual record schema is:

```json
{"text":"<one complete document>","language":"en","source":"<original source title or URL>","license":"<exact license identifier or terms>"}
```

Required fields:

- `text`: a string containing one complete document. Use JSON escapes such as `\n\n` for paragraph breaks inside the string; do not split a record across physical lines.
- `language`: exactly one supported code (`en`, `hi`, `es`, `fr`, `ar`, or `zh`).
- `source`: a non-empty string identifying the original source.
- `license`: a non-empty string recording the actual license/terms. `licence` is accepted as an alias if `license` is absent.

The `[dataset].jsonl_text_field` option selects the preferred text key (default `text`); `content` and `body` are accepted fallbacks. For multilingual preparation, use the exact `text` schema above and include the three provenance fields on every record. Other JSON fields are not copied to prepared metadata. A JSON string by itself is not sufficient for multilingual input because it has no provenance fields.

## Validation and split requirements

The multilingual config requires complete provenance. Files with invalid UTF-8 are reported as unreadable; malformed/oversized JSONL records, missing or invalid metadata, empty text, replacement-character corruption, and exact normalized duplicates are also detected and reported. Rejected or duplicate documents are not silently counted as usable. Exact duplicate content is removed globally to reduce leakage; duplicate totals are reported overall and by the language label on the duplicate record.

Text is normalized to Unicode NFC, line endings are converted to LF, horizontal whitespace is normalized, and paragraphs/punctuation are retained. No non-English script is transliterated or dropped. The custom tokenizer encodes UTF-8 bytes (IDs 0–255) as fallback and uses reserved IDs 256–259 for its special tokens. Preparation and verification audit Unicode round trips and special-token behavior.

To populate all three document-level splits for every configured language, provide at least **three unique, usable documents per language**. This is a technical minimum, not a useful training corpus. More independent and varied documents are preferable. The initial weights are `en=20%`, `hi=20%`, and `es=15%`, `fr=15%`, `ar=15%`, `zh=15%`; the training sampler uses these weights with replacement so a larger language corpus cannot dominate merely because it contains more examples.

## Exact local prepare, verify, and report commands

After placing only actual, authorized documents and their metadata in `data/raw/`, run from the repository root:

```bash
./scripts/data.sh prepare \
  --input data/raw \
  --output data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml

./scripts/data.sh stats --dataset data/processed/multilingual-v1
./scripts/data.sh verify --dataset data/processed/multilingual-v1
./scripts/data.sh report \
  --dataset data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml \
  --output data/reports/multilingual-v1.json
./scripts/hardware.sh --config configs/multilingual_tiny.toml
```

Preparation deterministically splits documents before learning BPE, and learns BPE from training documents only. `stats` reports per-language document/token/example counts, split distribution, duplicates, ingestion issues, and actual/target vocabulary statistics. `verify` checks split/tokenizer integrity, provenance, token ranges, per-language stats/ranges, Unicode round trips, and cross-split hash leakage. `report` writes those corpus results alongside the existing hardware/RAM estimate and the model parameter count. The prepared directory contains token files, tokenizer/config metadata, and a non-text SQLite provenance table; it does not copy source text. Inspect any reported ingestion issues before using the dataset.

These commands prepare, inspect, and estimate resources only. **They do not start training.** Do not invoke the training command until you have reviewed the data, license records, verification, and resource estimate and explicitly approved training.
