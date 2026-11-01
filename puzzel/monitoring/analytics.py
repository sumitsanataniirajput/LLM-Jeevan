"""Usage analytics for Puzzel.

Tracks API usage, generation metrics, and per-user statistics.
All data is kept in memory; no external analytics services are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class UsageEvent:
    """A single usage event.

    Attributes:
        timestamp: ISO timestamp.
        user_id: User identifier (optional).
        endpoint: API endpoint called.
        latency_ms: Request latency in milliseconds.
        tokens_used: Number of tokens processed.
        success: Whether the request succeeded.
    """

    timestamp: str
    user_id: str | None
    endpoint: str
    latency_ms: float
    tokens_used: int
    success: bool


class UsageAnalytics:
    """Tracks and reports usage analytics."""

    def __init__(self) -> None:
        self._events: list[UsageEvent] = []

    def record(
        self,
        endpoint: str,
        latency_ms: float,
        tokens_used: int,
        success: bool,
        user_id: str | None = None,
    ) -> None:
        """Record a usage event."""
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            endpoint=endpoint,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            success=success,
        )
        self._events.append(event)

    def get_events(self, user_id: str | None = None) -> list[UsageEvent]:
        """Get all events, optionally filtered by user."""
        if user_id is None:
            return list(self._events)
        return [e for e in self._events if e.user_id == user_id]

    def summary(self) -> dict[str, Any]:
        """Return a summary of usage analytics."""
        total = len(self._events)
        if total == 0:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "avg_latency_ms": 0.0,
                "total_tokens": 0,
                "by_endpoint": {},
                "by_user": {},
            }

        successful = sum(1 for e in self._events if e.success)
        failed = total - successful
        avg_latency = sum(e.latency_ms for e in self._events) / total
        total_tokens = sum(e.tokens_used for e in self._events)

        by_endpoint: dict[str, int] = {}
        by_user: dict[str, int] = {}
        for e in self._events:
            by_endpoint[e.endpoint] = by_endpoint.get(e.endpoint, 0) + 1
            if e.user_id:
                by_user[e.user_id] = by_user.get(e.user_id, 0) + 1

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens,
            "by_endpoint": by_endpoint,
            "by_user": by_user,
        }


# Default analytics instance.
default_analytics = UsageAnalytics()