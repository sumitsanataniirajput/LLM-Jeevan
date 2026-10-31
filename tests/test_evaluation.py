from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path

import pytest
import torch

from puzzel.checkpoint import save_checkpoint
from puzzel.config import TrainingConfig
from puzzel.evaluation.metrics import chrf_score, exact_match
from puzzel.evaluation.reports import (
    compare_evaluation_reports,
    write_evaluation_report,
)
from puzzel.evaluation.runner import (
    conditional_token_loss,
    evaluate_checkpoint,
    greedy_generate,
)
from puzzel.evaluation.sandbox import CodeSandboxCommand
from puzzel.evaluation.specs import (
    EvaluationDatasetError,
    SUITE_NAMES,
    iter_evaluation_records,
    load_suite_spec,
)
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import ByteTokenizer


def _save_checkpoint(path: Path) -> None:
    model_config = ModelConfig(
        vocab_size=260,
        context_length=16,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    training_config = TrainingConfig(device="cpu", max_steps=1)
    model = TransformerLM(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=training_config,
        tokenizer=ByteTokenizer(),
        step=0,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _knowledge_records() -> list[dict]:
    return [
        {
            "id": "en-01",
            "partition": "test",
            "language": "en",
            "prompt": "Answer with one color: What color is a clear daytime sky?",
            "accepted_answers": ["blue"],
            "source": "synthetic-test-fixture",
            "license": "test-only",
        },
        {
            "id": "hi-01",
            "partition": "test",
            "language": "hi",
            "prompt": "एक रंग में उत्तर दें: साफ़ दिन का आकाश किस रंग का होता है?",
            "accepted_answers": ["नीला"],
            "source": "synthetic-test-fixture",
            "license": "test-only",
        },
    ]


def test_all_capability_specs_load_with_independent_schemas() -> None:
    assert len(SUITE_NAMES) == 7
    for suite_name in SUITE_NAMES:
        suite = load_suite_spec(suite_name)
        assert suite.name == suite_name
        assert suite.required_fields
        assert suite.metric_directions
    assert load_suite_spec("programming").requires_code_sandbox
    assert load_suite_spec("creative_writing").rubric_categories


def test_evaluation_metrics_handle_unicode_and_empty_generations() -> None:
    assert exact_match("  नीला! ", ["नीला"])
    assert exact_match("你好，世界。", ["你好世界"])
    assert exact_match("مرحبا بالعالم", ["مرحبا بالعالم!"])
    assert not exact_match("", ["answer"])
    assert chrf_score("你好，世界", "你好，世界") == pytest.approx(100.0)
    assert chrf_score("", "reference") == 0.0


def test_eval_dataset_validation_requires_test_partition_and_provenance(tmp_path: Path) -> None:
    spec = load_suite_spec("general_knowledge")
    good = _knowledge_records()[0]
    good["partition"] = "train"
    dataset_path = tmp_path / "bad.jsonl"
    _write_jsonl(dataset_path, [good])

    with pytest.raises(EvaluationDatasetError, match="training data is not accepted"):
        list(iter_evaluation_records(dataset_path, spec))

    good["partition"] = "test"
    good.pop("license")
    _write_jsonl(dataset_path, [good])
    with pytest.raises(EvaluationDatasetError, match="missing required fields"):
        list(iter_evaluation_records(dataset_path, spec))


def test_greedy_generation_and_conditional_loss_use_existing_model(tmp_path: Path) -> None:
    config = ModelConfig(
        vocab_size=260,
        context_length=16,
        d_model=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    model = TransformerLM(config).eval()
    tokenizer = ByteTokenizer()
    device = torch.device("cpu")
    generated = greedy_generate(
        model,
        tokenizer,
        "English नमस्ते مرحبا 中文",
        device=device,
        max_new_tokens=3,
    )
    nll, tokens = conditional_token_loss(
        model,
        tokenizer,
        "Question: 2 + 2? Answer:",
        "four",
        device=device,
    )
    assert isinstance(generated, str)
    assert tokens == len(tokenizer.encode("four", add_eos=True))
    assert nll >= 0


def test_evaluation_writes_json_csv_and_language_metrics_without_text_leakage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "random-test-checkpoint.pt"
    _save_checkpoint(checkpoint)
    dataset = tmp_path / "evaluation" / "knowledge.jsonl"
    _write_jsonl(dataset, _knowledge_records())
    training_metrics = tmp_path / "training.csv"
    training_metrics.write_text(
        "step,train_loss,validation_loss\n1,4.5,5.0\n2,4.1,4.7\n",
        encoding="utf-8",
    )

    report = evaluate_checkpoint(
        checkpoint,
        dataset,
        "general_knowledge",
        max_new_tokens=2,
        training_metrics_path=training_metrics,
    )
    json_path = tmp_path / "reports" / "knowledge.json"
    csv_path = tmp_path / "reports" / "knowledge.csv"
    write_evaluation_report(report, json_path, csv_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))

    assert saved["dataset"]["examples_evaluated"] == 2
    assert set(saved["metrics"]["per_language"]) == {"en", "hi"}
    assert saved["metrics"]["overall"]["conditional_nll"] is not None
    assert saved["run"]["training_metrics"]["train_loss"] == pytest.approx(4.1)
    with csv_path.open(newline="", encoding="utf-8") as source:
        csv_rows = list(csv.DictReader(source))
    assert len(csv_rows) == 3
    assert float(csv_rows[0]["train_loss"]) == pytest.approx(4.1)
    assert csv_rows[0]["validation_loss"] == "4.7"
    assert csv_rows[1]["train_loss"] == ""
    assert "prompt" not in saved["examples"][0]
    assert "generated_text" not in saved["examples"][0]


def test_code_execution_is_not_run_without_explicit_sandbox(tmp_path: Path) -> None:
    checkpoint = tmp_path / "random-test-checkpoint.pt"
    _save_checkpoint(checkpoint)
    dataset = tmp_path / "programming.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "code-01",
                "partition": "test",
                "language": "en",
                "prompt": "Write a small program that echoes its input.",
                "reference_code": "print(input())",
                "test_cases": [{"input": "ok", "expected_output": "ok"}],
                "source": "synthetic-test-fixture",
                "license": "test-only",
            }
        ],
    )

    report = evaluate_checkpoint(
        checkpoint, dataset, "programming", max_new_tokens=2
    )
    example = report["examples"][0]
    assert example["sandbox_status"] == "not_run"
    assert example["code_execution_ok"] is None
    assert report["metrics"]["overall"]["code_execution_pass_rate"] is None
    assert report["metrics"]["overall"]["unit_test_pass_rate"] is None


def test_sandbox_adapter_calls_only_the_configured_json_protocol(tmp_path: Path) -> None:
    launcher = tmp_path / "safe-test-launcher.py"
    launcher.write_text(
        "import json, sys\n"
        "payload = json.load(sys.stdin)\n"
        "count = len(payload['test_cases'])\n"
        "print(json.dumps({'execution_ok': True, 'passed_tests': count, 'total_tests': count}))\n",
        encoding="utf-8",
    )
    runner = CodeSandboxCommand([sys.executable, str(launcher)], timeout_seconds=2)
    result = runner.run(
        "not executed by this test launcher",
        [{"input": 1, "expected_output": 1}],
        working_directory=tmp_path,
    )
    assert result.execution_ok
    assert result.passed_tests == result.total_tests == 1
    assert result.unit_test_pass_rate == 1.0


def test_creative_writing_ratings_are_explicit_human_inputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "random-test-checkpoint.pt"
    _save_checkpoint(checkpoint)
    dataset = tmp_path / "creative.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "id": "creative-01",
                "partition": "test",
                "language": "fr",
                "prompt": "Écrivez deux lignes sur une étoile.",
                "source": "synthetic-test-fixture",
                "license": "test-only",
            }
        ],
    )
    ratings = tmp_path / "ratings.jsonl"
    _write_jsonl(
        ratings,
        [
            {
                "id": "creative-01",
                "rater_id": "reviewer-a",
                "scores": {
                    "prompt_adherence": 4,
                    "coherence": 3,
                    "originality": 4,
                    "style": 5,
                },
            }
        ],
    )

    report = evaluate_checkpoint(
        checkpoint,
        dataset,
        "creative_writing",
        max_new_tokens=2,
        human_ratings_path=ratings,
    )
    assert report["metrics"]["overall"]["chrf"] is None
    assert report["metrics"]["overall"]["human_rubric_mean"] == pytest.approx(4.0)
    assert report["human_ratings"]["coverage"] == 1.0


def test_comparison_flags_regression_and_rejects_dataset_hash_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "random-test-checkpoint.pt"
    _save_checkpoint(checkpoint)
    dataset = tmp_path / "evaluation.jsonl"
    _write_jsonl(dataset, _knowledge_records())
    report = evaluate_checkpoint(
        checkpoint, dataset, "general_knowledge", max_new_tokens=2
    )
    baseline = copy.deepcopy(report)
    candidate = copy.deepcopy(report)
    baseline["metrics"]["overall"]["exact_accuracy"] = 0.8
    candidate["metrics"]["overall"]["exact_accuracy"] = 0.7
    baseline["metrics"]["per_language"]["en"]["exact_accuracy"] = 0.8
    candidate["metrics"]["per_language"]["en"]["exact_accuracy"] = 0.7
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    comparison = compare_evaluation_reports(
        [baseline_path], [candidate_path], max_regression=0.02
    )
    assert comparison["regression_detected"]
    assert comparison["suites"][0]["overall"]["exact_accuracy"]["regressed"]

    candidate["dataset"]["sha256"] = "different-evaluation-data"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="dataset do not match"):
        compare_evaluation_reports([baseline_path], [candidate_path])

    candidate["dataset"]["sha256"] = baseline["dataset"]["sha256"]
    candidate["run"]["max_new_tokens"] = baseline["run"]["max_new_tokens"] + 1
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(EvaluationDatasetError, match="max_new_tokens differ"):
        compare_evaluation_reports([baseline_path], [candidate_path])


def test_report_outputs_refuse_overwrite_by_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "random-test-checkpoint.pt"
    _save_checkpoint(checkpoint)
    dataset = tmp_path / "evaluation.jsonl"
    _write_jsonl(dataset, _knowledge_records()[:1])
    report = evaluate_checkpoint(checkpoint, dataset, "general_knowledge", max_new_tokens=1)
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    write_evaluation_report(report, json_path, csv_path)
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_report(report, json_path, csv_path)
