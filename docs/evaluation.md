# Evaluation architecture and local commands

This infrastructure evaluates **existing Puzzel checkpoints only**. It does not train, fetch a checkpoint, download a benchmark, or call an external AI service. There is no trained production checkpoint or real evaluation dataset in the repository; all metrics are currently unmeasured. See [`roadmap.md`](roadmap.md) for program phases and exit gates.

## Layout

- `configs/evaluation/*.toml` — seven independent capability schemas and metric directions.
- `configs/instruction_sft.toml` — SFT instruction-response data/split/sampling design only; no SFT loop is implemented.
- `puzzel/evaluation/specs.py` — strict UTF-8 JSONL, provenance, language, partition, and task-field validation.
- `puzzel/evaluation/runner.py` — Puzzel-checkpoint loading, deterministic greedy generation, response-conditioned token loss/perplexity, per-language aggregation, and human-rating ingestion.
- `puzzel/evaluation/metrics.py` — normalization-based exact match and a transparent in-repository chrF diagnostic.
- `puzzel/evaluation/sandbox.py` — opt-in JSON protocol adapter for an externally isolated code-test launcher.
- `puzzel/evaluation/reports.py` — JSON/CSV output and same-data direction-aware checkpoint comparisons.

The checkpoint SHA-256, evaluation JSONL SHA-256, suite-spec SHA-256, evaluated partitions, example count, generation settings, and optional sandbox/rating hashes are recorded. Reports contain IDs and measurements, not prompts, references, generated answers, or private evaluator notes. Evaluation JSONL and generated reports are ignored by Git.

## Seven independent suites

1. **Multilingual conversation:** prompts are stored as `messages`; the evaluator formats role-prefixed turns and asks the model to continue after a final user turn. NLL/perplexity and chrF are diagnostics; human ratings cover relevance, coherence, instruction-following, and fluency.
2. **General knowledge:** curated `accepted_answers` are compared with normalized exact match. Human review is recommended; exact match is not semantic factuality.
3. **Programming:** generated code is sent only to the configured external sandbox. `code_execution_pass_rate` and `unit_test_pass_rate` remain `null` when no sandbox is supplied.
4. **Mathematics:** normalized final-answer exact match and optional human method/validity ratings. Symbolic equivalence is not inferred.
5. **Reasoning:** normalized final-answer accuracy, plus optional human review of logical validity/completeness. Answer accuracy alone does not validate a chain of reasoning.
6. **Translation:** chrF against a licensed reference and optional human adequacy/fluency/terminology scores, grouped by the target language.
7. **Creative writing:** non-empty rate is a generation diagnostic only. Quality is measured only from separate, human-provided rubric scores; there is no automatic creativity score or model judge.

All suite examples require `id`, `partition`, `language`, `source`, and `license`. Only `dev` or `test` partitions are accepted. Suite-specific required fields are enforced from TOML, and duplicate IDs/missing provenance are errors. `language` is one of `en`, `hi`, `es`, `fr`, `ar`, or `zh`.

## Local evaluation JSONL

Place manually curated, properly licensed evaluation JSONL files under `data/evaluation/`; do not add downloaded or unlicensed benchmark data. The repository contains schemas only. Example **schema template**, not a benchmark example:

```jsonl
{"id":"<unique-id>","partition":"test","language":"en","prompt":"<licensed prompt>","accepted_answers":["<curated answer>"],"source":"<source title or URL>","license":"<exact terms>"}
```

Use the task-specific files in `configs/evaluation/`. Programming records require `reference_code` and a non-empty `test_cases` array of `{ "input": ..., "expected_output": ... }` values. Translation records include `source_language`, `target_language`, `source_text`, and `reference_translation`; `language` must match the target. Conversation messages use `system`, `user`, or `assistant` roles and end in a user turn.

Human rubric ratings are separate JSONL with `id`, `rater_id`, and `scores` (category values from 1 to 5). Rating categories come from the selected suite TOML. Prefer blinded checkpoint identity and independent raters; report coverage and category means. No external/AI judge is used.

## Running an evaluation

After a checkpoint and local evaluation set exist and have been reviewed, use:

```bash
./scripts/evaluate.sh evaluate \
  --checkpoint checkpoints/puzzel-model.pt \
  --suite general_knowledge \
  --dataset data/evaluation/general_knowledge.jsonl \
  --json-output data/reports/general_knowledge.json \
  --csv-output data/reports/general_knowledge.csv
```

This is a command template and has **not** been run on a model checkpoint. The default decoder is deterministic greedy decoding; `--max-examples` caps a deterministic file prefix for smoke tests, and `--training-metrics` imports an existing Puzzel training CSV's latest train/validation loss without recomputing it. The runner also computes reference continuation NLL/perplexity, grouped by language, where a reference answer exists. It does not report missing values as zero.

### Code execution safety

Code execution is disabled unless `--code-sandbox <trusted-launcher>` is supplied. The launcher receives a single JSON object on stdin with `code` and `test_cases`, then returns JSON with `execution_ok`, `passed_tests`, and `total_tests`. The wrapper uses `shell=False`, a temporary working directory, a minimal environment, and a wall-clock timeout. It **is not a security boundary**; the configured launcher must itself provide container/VM or equivalent OS isolation, no network, no host credentials or filesystem access, and strict CPU/memory/output limits. Do not point it at a plain host Python interpreter for untrusted generated code. No generated code has been executed by this project.

### Checkpoint comparison and regression alarms

Evaluate baseline and candidate checkpoints on the exact same suite files, spec versions, generation settings, ratings, and sandbox. Then compare one or more same-suite reports:

```bash
./scripts/evaluate.sh compare \
  --baseline data/reports/baseline-knowledge.json data/reports/baseline-translation.json \
  --candidate data/reports/candidate-knowledge.json data/reports/candidate-translation.json \
  --max-regression 0.02 \
  --json-output data/reports/comparison.json \
  --csv-output data/reports/comparison.csv
```

The comparison refuses suite-spec or evaluation-data hash mismatches and reports changes per metric and language. A metric is flagged when it worsens beyond the configured absolute threshold according to its declared direction (`higher` or `lower` is better). This is an engineering alarm, not a confidence interval or significance test. A missing metric remains unscored; no aggregate score hides a regression in another capability.

## Metric interpretation and limits

- Conditional token NLL is measured on the reference continuation with the prompt as context, bounded by overlapping context windows. Perplexity is `exp(mean NLL)` and may be omitted if it overflows.
- Exact match applies Unicode NFC, case folding, and removal of non-alphanumeric characters. It can accept false matches or reject valid paraphrases and does not infer semantic/symbolic equivalence.
- chrF is a short dependency-free character n-gram F-score (0–100), averaged by example. It is provided for comparison diagnostics; it is not presented as a reproduction of a benchmark package.
- Code rates depend on the exact sandbox/test harness. Human ratings depend on rubric, rater agreement, and blinding. Preserve these versions and hashes for valid comparisons.
- Training/validation loss is imported only when an existing training CSV is explicitly supplied. Evaluation has no training side effect.

Never claim a capability based only on a falling loss, one automatic score, synthetic tests, or a report without reviewed licensed data and appropriate human/domain evaluation.
