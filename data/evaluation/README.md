# Local evaluation data only

No evaluation corpus, benchmark, or human rating set is included. `data/evaluation/` is git-ignored except for this README. Use only evaluation material you own or are licensed to use; record provenance and license on every example. Keep evaluation examples held out from all pretraining/SFT data. Puzzel does not download benchmarks or query external APIs.

Seven independent JSONL suite contracts live under `configs/evaluation/`:

| Suite | Required task payload | Primary automatic/manual metrics |
| --- | --- | --- |
| `multilingual_conversation` | `messages` (system/user/assistant turns, ending in user) and `reference` | conditional NLL, perplexity, chrF diagnostic, human rubric |
| `general_knowledge` | `prompt`, `accepted_answers` | normalized exact accuracy; human review optional |
| `programming` | `prompt`, `reference_code`, `test_cases` | code execution and unit-test pass rates only with an explicit sandbox launcher |
| `mathematics` | `prompt`, `expected_answer` | normalized exact accuracy; human reasoning rubric optional |
| `reasoning` | `prompt`, `expected_answer` | normalized exact final-answer accuracy; explanation rubric optional |
| `translation` | source/target language codes, `source_text`, `reference_translation` | chrF diagnostic and optional human adequacy/fluency rubric |
| `creative_writing` | `prompt` | non-empty generation and human rubric; no automatic creativity score |

Every record must also have a unique non-empty `id`, `partition` (`dev` or `test`), `language` (one of `en`, `hi`, `es`, `fr`, `ar`, `zh`), `source`, and `license`. `licence` is accepted as an alias. The translation suite sets `language` equal to `target_language`. Keep one JSON object per UTF-8 JSONL line. The machine-readable field lists, metric directions, and generation caps are in the individual TOML specs; the validator rejects training partitions and duplicate IDs.

Illustrative schema only (not data; replace every angle-bracket field with a real, licensed example):

```jsonl
{"id":"<unique-id>","partition":"test","language":"hi","prompt":"<licensed evaluation prompt>","expected_answer":"<curated answer>","source":"<source title or URL>","license":"<exact license or terms>"}
```

For `programming`, a `test_cases` entry contains JSON `input` and `expected_output` values. Model-generated code is **never executed by default**. Code execution requires `--code-sandbox` pointing to a trusted launcher that itself provides a container/VM or equivalent OS isolation, no network, no host credentials/filesystem, and CPU/memory/time limits. The Puzzel subprocess adapter uses no shell and adds a wall-clock timeout, but is **not itself a security sandbox**. If no launcher is supplied, code metrics are null/not run.

Human ratings are a separate UTF-8 JSONL file and never use an AI judge:

```jsonl
{"id":"<unique-id>","rater_id":"<reviewer-id>","scores":{"relevance":4,"coherence":5}}
```

Scores are integers or floats from 1 through 5 and category names must match the selected suite's `rubric_categories`. Keep ratings blinded to checkpoint identity where practical. Reports contain IDs and aggregate metrics, not prompts, references, generated text, or evaluator notes.

The evaluation runner uses deterministic greedy decoding, records checkpoint/dataset/spec SHA-256 fingerprints, scores reference continuations with bounded-context token NLL/perplexity, groups metrics by language, and writes JSON plus CSV. Exact match is a normalization-based diagnostic, not semantic equivalence; chrF is a transparent in-repo character n-gram diagnostic, not a benchmark implementation. Reported metrics are not capability claims. Checkpoint comparison requires identical suite specs, evaluation file, generation settings, example counts, and (when used) ratings/sandbox configuration. Regression thresholds are engineering alarms, not statistical significance tests.
