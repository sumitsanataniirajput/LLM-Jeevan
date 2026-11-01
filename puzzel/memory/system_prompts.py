"""Custom system prompts for Puzzel.

Manages per-user and global system prompt templates that guide model
behavior during generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SystemPrompt:
    """A system prompt template.

    Attributes:
        name: Prompt name (e.g., "assistant", "translator", "coder").
        content: The system prompt text.
        language: Language code this prompt is optimized for.
        is_default: Whether this is the default prompt.
        metadata: Additional prompt metadata.
    """

    name: str
    content: str
    language: str = "en"
    is_default: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemPrompt":
        return cls(**data)


class SystemPromptManager:
    """Manages system prompts for users and globally."""

    def __init__(self) -> None:
        self._prompts: dict[str, SystemPrompt] = {}
        self._user_defaults: dict[str, str] = {}  # user_id -> prompt name

    def add(self, prompt: SystemPrompt) -> None:
        """Register a system prompt."""
        self._prompts[prompt.name] = prompt

    def get(self, name: str) -> SystemPrompt | None:
        """Get a prompt by name."""
        return self._prompts.get(name)

    def get_for_user(self, user_id: str) -> SystemPrompt | None:
        """Get the default prompt for a user, or the global default."""
        name = self._user_defaults.get(user_id)
        if name:
            return self._prompts.get(name)
        # Return global default.
        for p in self._prompts.values():
            if p.is_default:
                return p
        return None

    def set_user_default(self, user_id: str, prompt_name: str) -> None:
        """Set a user's default prompt."""
        if prompt_name not in self._prompts:
            raise ValueError(f"prompt '{prompt_name}' not registered")
        self._user_defaults[user_id] = prompt_name

    def list_prompts(self) -> list[str]:
        """Return all registered prompt names."""
        return sorted(self._prompts.keys())

    def remove(self, name: str) -> None:
        """Remove a prompt."""
        self._prompts.pop(name, None)
        # Remove any user defaults pointing to it.
        for uid in list(self._user_defaults):
            if self._user_defaults[uid] == name:
                del self._user_defaults[uid]


# Default manager with built-in prompts.
default_prompt_manager = SystemPromptManager()

default_prompt_manager.add(
    SystemPrompt(
        name="assistant",
        content=(
            "You are Puzzel, a helpful, multilingual AI assistant. "
            "Respond in the same language as the user. Be concise, "
            "accurate, and honest. Do not claim capabilities you do not have."
        ),
        is_default=True,
    )
)

default_prompt_manager.add(
    SystemPrompt(
        name="translator",
        content=(
            "You are Puzzel, a multilingual translation assistant. "
            "Translate text between languages while preserving meaning, "
            "tone, and formatting. Respond only with the translation."
        ),
        language="multi",
    )
)

default_prompt_manager.add(
    SystemPrompt(
        name="coder",
        content=(
            "You are Puzzel, a coding assistant. Write clean, correct, "
            "well-documented code. Explain your reasoning briefly."
        ),
        language="en",
    )
)