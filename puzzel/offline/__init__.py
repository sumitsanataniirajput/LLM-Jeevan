"""Offline inference mode for Puzzel.

Provides local-only inference capabilities when network access is
unavailable or undesired. All model execution is local; no external
services are contacted.
"""

from puzzel.offline.mode import OfflineMode, is_offline, get_offline_status

__all__ = [
    "OfflineMode",
    "is_offline",
    "get_offline_status",
]