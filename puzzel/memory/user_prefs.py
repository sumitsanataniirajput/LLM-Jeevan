"""User-specific AI preferences for Puzzel.

Stores and retrieves per-user settings such as preferred language,
temperature, system prompt, and conversation style.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UserPreferences:
    """Preferences for a single user.

    Attributes:
        user_id: Unique user identifier.
        language: Preferred language code (en, hi, es, fr, ar, zh).
        temperature: Default generation temperature.
        max_new_tokens: Default max new tokens.
        system_prompt: Custom system prompt.
        theme: UI theme ("light" or "dark").
        metadata: Additional user-defined metadata.
    """

    user_id: str
    language: str = "en"
    temperature: float = 1.0
    max_new_tokens: int = 50
    system_prompt: str = ""
    theme: str = "light"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserPreferences":
        return cls(**data)


class UserPreferencesManager:
    """Manages user preferences with in-memory caching.

    In production, preferences would be persisted to the database.
    """

    def __init__(self) -> None:
        self._prefs: dict[str, UserPreferences] = {}

    def get(self, user_id: str) -> UserPreferences:
        """Get preferences for a user, returning defaults if not set."""
        if user_id not in self._prefs:
            self._prefs[user_id] = UserPreferences(user_id=user_id)
        return self._prefs[user_id]

    def set(self, prefs: UserPreferences) -> None:
        """Save preferences for a user."""
        self._prefs[prefs.user_id] = prefs

    def update(self, user_id: str, **kwargs: Any) -> UserPreferences:
        """Update specific preference fields for a user."""
        prefs = self.get(user_id)
        for key, value in kwargs.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)
        self._prefs[user_id] = prefs
        return prefs

    def delete(self, user_id: str) -> None:
        """Delete preferences for a user."""
        self._prefs.pop(user_id, None)

    def list_users(self) -> list[str]:
        """Return all user IDs with stored preferences."""
        return sorted(self._prefs.keys())


# Default manager instance.
default_prefs_manager = UserPreferencesManager()