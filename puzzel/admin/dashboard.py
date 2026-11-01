"""Admin dashboard for Puzzel.

Provides system overview, model information, checkpoint management,
and other administrative functions.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from puzzel.monitoring.analytics import default_analytics
from puzzel.monitoring.feedback import default_feedback_collector
from puzzel.monitoring.health import health_check


@dataclass
class SystemOverview:
    """System overview information for the admin dashboard.

    Attributes:
        status: Overall system status.
        model_info: Information about the loaded model.
        health: Health check results.
        analytics: Usage analytics summary.
        feedback: Feedback statistics.
        disk_info: Disk usage information.
        config: Active configuration.
    """

    status: str
    model_info: dict[str, Any]
    health: dict[str, Any]
    analytics: dict[str, Any]
    feedback: dict[str, Any]
    disk_info: dict[str, Any]
    config: dict[str, Any]


@dataclass
class CheckpointInfo:
    """Information about a checkpoint file.

    Attributes:
        path: Path to the checkpoint.
        size_bytes: File size in bytes.
        modified_at: ISO modification timestamp.
        format_version: Checkpoint format version (if readable).
    """

    path: str
    size_bytes: int
    modified_at: str
    format_version: str | None = None


class AdminDashboard:
    """Admin dashboard for managing the Puzzel platform.

    Provides read-only views of system state and management
    operations for checkpoints, datasets, and configuration.
    """

    def __init__(self, checkpoint_dir: str | Path = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    def get_system_overview(
        self,
        model_loaded: bool = False,
        model_info: dict[str, Any] | None = None,
        device: str = "cpu",
    ) -> SystemOverview:
        """Get a complete system overview.

        Args:
            model_loaded: Whether a model is loaded.
            model_info: Model information dict.
            device: Device the model runs on.

        Returns:
            SystemOverview with all system information.
        """
        health = health_check(model_loaded=model_loaded, device=device)
        analytics = default_analytics.summary()
        feedback_entries = default_feedback_collector.get_all()
        avg_rating = default_feedback_collector.average_rating()
        rating_dist = default_feedback_collector.rating_distribution()

        # Disk info.
        try:
            disk = shutil.disk_usage(".")
            disk_info = {
                "total_bytes": disk.total,
                "used_bytes": disk.used,
                "free_bytes": disk.free,
            }
        except OSError:
            disk_info = {"total_bytes": 0, "used_bytes": 0, "free_bytes": 0}

        config = {
            "auth_enabled": os.environ.get("PUZZEL_AUTH_ENABLED", "false").lower()
            == "true",
            "device": device,
            "max_context_length": int(
                os.environ.get("PUZZEL_MAX_CONTEXT_LENGTH", "512")
            ),
            "use_kv_cache": os.environ.get("PUZZEL_USE_KV_CACHE", "true").lower()
            == "true",
        }

        return SystemOverview(
            status=health.status,
            model_info=model_info or {},
            health={
                "status": health.status,
                "checks": health.checks,
                "memory_available_bytes": health.memory_available_bytes,
                "disk_free_bytes": health.disk_free_bytes,
                "timestamp": health.timestamp,
            },
            analytics=analytics,
            feedback={
                "total_entries": len(feedback_entries),
                "average_rating": round(avg_rating, 2) if avg_rating else None,
                "rating_distribution": rating_dist,
            },
            disk_info=disk_info,
            config=config,
        )

    def list_checkpoints(self) -> list[CheckpointInfo]:
        """List all checkpoint files in the checkpoint directory."""
        checkpoints: list[CheckpointInfo] = []
        if not self.checkpoint_dir.exists():
            return checkpoints

        for path in sorted(self.checkpoint_dir.iterdir()):
            if path.is_file() and path.suffix in {".pt", ".pth", ".bin"}:
                stat = path.stat()
                checkpoints.append(
                    CheckpointInfo(
                        path=str(path),
                        size_bytes=stat.st_size,
                        modified_at=self._iso_from_timestamp(stat.st_mtime),
                    )
                )
        return checkpoints

    def delete_checkpoint(self, path: str | Path) -> bool:
        """Delete a checkpoint file.

        Args:
            path: Path to the checkpoint file.

        Returns:
            True if deleted, False if not found.
        """
        p = Path(path)
        if not p.exists():
            return False
        p.unlink()
        return True

    @staticmethod
    def _iso_from_timestamp(ts: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# Default dashboard instance.
default_dashboard = AdminDashboard()


def get_system_overview(
    model_loaded: bool = False,
    model_info: dict[str, Any] | None = None,
    device: str = "cpu",
) -> SystemOverview:
    """Convenience function to get system overview from the default dashboard."""
    return default_dashboard.get_system_overview(
        model_loaded=model_loaded,
        model_info=model_info,
        device=device,
    )