"""Offline inference mode for Puzzel.

Detects whether the system is in offline mode and provides a
local-only inference path that never contacts external services.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass


@dataclass
class OfflineStatus:
    """Status of offline mode.

    Attributes:
        offline: Whether the system is operating offline.
        reason: Explanation of why offline mode is active/inactive.
        network_reachable: Whether a network connection was detected.
    """

    offline: bool
    reason: str
    network_reachable: bool


def is_offline() -> bool:
    """Return True if the system is operating in offline mode.

    Checks the PUZZEL_OFFLINE environment variable and attempts a
    quick network connectivity test.
    """
    env_val = os.environ.get("PUZZEL_OFFLINE", "").lower()
    if env_val in ("true", "1", "yes"):
        return True
    if env_val in ("false", "0", "no"):
        return False
    return not _network_reachable()


def get_offline_status() -> OfflineStatus:
    """Return detailed offline mode status."""
    import os

    env_val = os.environ.get("PUZZEL_OFFLINE", "").lower()
    network = _network_reachable()

    if env_val in ("true", "1", "yes"):
        return OfflineStatus(
            offline=True,
            reason="PUZZEL_OFFLINE environment variable set to true",
            network_reachable=network,
        )
    if env_val in ("false", "0", "no"):
        return OfflineStatus(
            offline=False,
            reason="PUZZEL_OFFLINE environment variable set to false",
            network_reachable=network,
        )

    offline = not network
    reason = (
        "No network connectivity detected"
        if offline
        else "Network connectivity detected; online mode"
    )
    return OfflineStatus(
        offline=offline,
        reason=reason,
        network_reachable=network,
    )


def _network_reachable(timeout: float = 2.0) -> bool:
    """Test whether network connectivity is available.

    Attempts to connect to a well-known DNS resolver.
    """
    import os

    try:
        socket.setdefaulttimeout(timeout)
        host = os.environ.get("PUZZEL_OFFLINE_HOST", "1.1.1.1")
        port = int(os.environ.get("PUZZEL_OFFLINE_PORT", "53"))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return True
    except OSError:
        return False


class OfflineMode:
    """Context manager for enforcing offline operation.

    When active, raises an error if any code attempts to make
    network requests. Use this to wrap inference code paths that
    must never contact external services.
    """

    def __init__(self, enforce: bool = True) -> None:
        self.enforce = enforce
        self._previous = is_offline()

    def __enter__(self) -> "OfflineMode":
        if self.enforce:
            os.environ["PUZZEL_OFFLINE"] = "true"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enforce:
            if self._previous:
                os.environ["PUZZEL_OFFLINE"] = "true"
            else:
                os.environ.pop("PUZZEL_OFFLINE", None)

    @property
    def is_offline(self) -> bool:
        """Return True if offline mode is active."""
        return is_offline()