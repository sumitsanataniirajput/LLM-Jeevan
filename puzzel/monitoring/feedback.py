"""User feedback and ratings collection for Puzzel.

Collects user ratings and comments on model responses.
All data is kept in memory; persisted via the database package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FeedbackEntry:
    """A single feedback entry.

    Attributes:
        id: Unique feedback identifier.
        conversation_id: Conversation identifier.
        message_id: Message identifier being rated.
        user_id: User who provided feedback.
        rating: Rating from 1 to 5.
        comment: Optional comment text.
        timestamp: ISO timestamp.
    """

    id: int
    conversation_id: int
    message_id: int
    user_id: str | None
    rating: int
    comment: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackCollector:
    """Collects and manages user feedback."""

    def __init__(self) -> None:
        self._feedback: list[FeedbackEntry] = []
        self._next_id = 1

    def add(
        self,
        conversation_id: int,
        message_id: int,
        rating: int,
        comment: str = "",
        user_id: str | None = None,
    ) -> FeedbackEntry:
        """Add a feedback entry."""
        if not 1 <= rating <= 5:
            raise ValueError("rating must be between 1 and 5")
        entry = FeedbackEntry(
            id=self._next_id,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._next_id += 1
        self._feedback.append(entry)
        return entry

    def get_all(self) -> list[FeedbackEntry]:
        """Return all feedback entries."""
        return list(self._feedback)

    def get_for_conversation(self, conversation_id: int) -> list[FeedbackEntry]:
        """Return feedback for a specific conversation."""
        return [e for e in self._feedback if e.conversation_id == conversation_id]

    def average_rating(self) -> float | None:
        """Return the average rating across all feedback."""
        if not self._feedback:
            return None
        return sum(e.rating for e in self._feedback) / len(self._feedback)

    def rating_distribution(self) -> dict[int, int]:
        """Return count of each rating (1-5)."""
        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for e in self._feedback:
            dist[e.rating] += 1
        return dist


# Default feedback collector instance.
default_feedback_collector = FeedbackCollector()