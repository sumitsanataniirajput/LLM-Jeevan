from dataclasses import replace

from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.training.hardware import (
    build_hardware_report,
    detect_hardware,
    estimate_memory_requirements,
    estimate_parameter_count,
    estimate_training_compute,
    recommend_safe_batch_size,
)


def _model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=280,
        context_length=16,
        d_model=16,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
    )


def test_parameter_estimate_matches_existing_model_without_changing_architecture() -> None:
    tied_config = _model_config()
    untied_config = replace(tied_config, tie_embeddings=False)
    for config in (tied_config, untied_config):
        model = TransformerLM(config)
        assert estimate_parameter_count(config) == model.parameter_count()


def test_training_memory_estimate_scales_with_context_and_batch() -> None:
    config = _model_config()
    one = estimate_memory_requirements(config, batch_size=1, context_length=8)
    two = estimate_memory_requirements(config, batch_size=2, context_length=8)
    longer = estimate_memory_requirements(config, batch_size=1, context_length=16)

    assert two["estimated_training_bytes"] > one["estimated_training_bytes"]
    assert longer["activation_estimate_bytes"] > one["activation_estimate_bytes"]
    assert one["estimated_checkpoint_bytes"] == one["parameter_count"] * 12


def test_training_compute_estimate_scales_with_batch_and_steps() -> None:
    config = _model_config()
    one = estimate_training_compute(config, batch_size=1, max_steps=10)
    two = estimate_training_compute(config, batch_size=2, max_steps=10)
    longer = estimate_training_compute(config, batch_size=1, max_steps=20)

    assert one["estimated_flops_per_step"] > 0
    assert two["estimated_flops_per_step"] == 2 * one["estimated_flops_per_step"]
    assert longer["estimated_total_flops"] == 2 * one["estimated_total_flops"]


def test_hardware_detection_and_batch_recommendation_are_conservative(tmp_path) -> None:
    info = detect_hardware(tmp_path)
    assert info.logical_cpu_count >= 1
    assert info.total_ram_bytes > 0
    assert info.available_ram_bytes > 0
    assert info.cuda_device_count >= 0
    assert not info.cuda_available or info.cuda_device_count > 0
    assert info.disk_free_bytes > 0
    assert info.disk_path == str(tmp_path.resolve())

    recommendation = recommend_safe_batch_size(info, _model_config(), device="cpu")
    assert recommendation == 0 or recommendation <= 8

    report = build_hardware_report(
        replace(_model_config(), context_length=16),
        batch_size=1024,
        device="cpu",
        disk_path=tmp_path,
    )
    assert report["selected_device"] == "cpu"
    assert report["warnings"]
