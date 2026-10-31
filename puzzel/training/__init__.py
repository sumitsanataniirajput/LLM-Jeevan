"""Training and optional hardware-inspection APIs."""

from puzzel.training.loop import (
    LearningRateSchedule,
    TrainResult,
    TrainingDivergedError,
    evaluate_loss,
    resolve_device,
    seed_everything,
    train_model,
)

_HARDWARE_EXPORTS = {
    "HardwareInfo",
    "build_hardware_report",
    "detect_hardware",
    "estimate_memory_requirements",
    "estimate_parameter_count",
    "estimate_training_compute",
    "recommend_safe_batch_size",
}

__all__ = [
    "HardwareInfo",
    "LearningRateSchedule",
    "TrainResult",
    "TrainingDivergedError",
    "build_hardware_report",
    "detect_hardware",
    "estimate_memory_requirements",
    "estimate_parameter_count",
    "estimate_training_compute",
    "evaluate_loss",
    "recommend_safe_batch_size",
    "resolve_device",
    "seed_everything",
    "train_model",
]


def __getattr__(name: str):
    if name in _HARDWARE_EXPORTS:
        from puzzel.training import hardware

        return getattr(hardware, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
