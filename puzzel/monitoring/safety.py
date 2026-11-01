"""AI safety monitoring for Puzzel.

Provides content safety checks, toxicity filtering, and policy
enforcement for model inputs and outputs. All checks are local;
no external safety APIs are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SafetyResult:
    """Result of a safety check.

    Attributes:
        safe: Whether the content passed safety checks.
        flagged_categories: Categories that were flagged.
        score: Safety score (0-1, higher = safer).
        details: Additional check details.
    """

    safe: bool
    flagged_categories: list[str]
    score: float
    details: dict[str, Any] = field(default_factory=dict)


class SafetyMonitor:
    """Monitors model inputs and outputs for safety violations.

    Uses simple keyword-based detection for common harmful content
    categories. This is a basic safety layer; production deployments
    should use more sophisticated classifiers.
    """

    # Simple keyword lists for content categories.
    HATE_SPEECH_KEYWORDS = frozenset({
        "hate", "racist", "nazi", "kkk", "supremacist",
    })
    VIOLENCE_KEYWORDS = frozenset({
        "bomb", "kill", "murder", "attack", "weapon",
        "shoot", "terrorist",
    })
    SEXUAL_CONTENT_KEYWORDS = frozenset({
        "porn", "sexual", "nude", "explicit",
    })
    PII_KEYWORDS = frozenset({
        "ssn", "social security", "credit card", "password",
        "api key", "secret key",
    })

    def __init__(self, enabled_categories: frozenset[str] | None = None) -> None:
        self.enabled_categories = enabled_categories or frozenset({
            "hate_speech", "violence", "sexual_content", "pii",
        })

    def check(self, text: str) -> SafetyResult:
        """Check text for safety violations.

        Args:
            text: Text to check.

        Returns:
            SafetyResult with safety status and details.
        """
        text_lower = text.lower()
        flagged: list[str] = []
        details: dict[str, Any] = {}

        if "hate_speech" in self.enabled_categories:
            hits = [kw for kw in self.HATE_SPEECH_KEYWORDS if kw in text_lower]
            if hits:
                flagged.append("hate_speech")
                details["hate_speech"] = hits

        if "violence" in self.enabled_categories:
            hits = [kw for kw in self.VIOLENCE_KEYWORDS if kw in text_lower]
            if hits:
                flagged.append("violence")
                details["violence"] = hits

        if "sexual_content" in self.enabled_categories:
            hits = [kw for kw in self.SEXUAL_CONTENT_KEYWORDS if kw in text_lower]
            if hits:
                flagged.append("sexual_content")
                details["sexual_content"] = hits

        if "pii" in self.enabled_categories:
            hits = [kw for kw in self.PII_KEYWORDS if kw in text_lower]
            if hits:
                flagged.append("pii")
                details["pii"] = hits

        safe = len(flagged) == 0
        score = 1.0 - (len(flagged) / max(len(self.enabled_categories), 1))

        return SafetyResult(
            safe=safe,
            flagged_categories=flagged,
            score=score,
            details=details,
        )

    def check_and_filter(self, text: str) -> tuple[str, SafetyResult]:
        """Check text and return a filtered version if unsafe.

        If unsafe, returns a refusal message instead of the original text.

        Args:
            text: Text to check.

        Returns:
            Tuple of (filtered_text, safety_result).
        """
        result = self.check(text)
        if result.safe:
            return text, result
        return (
            "I cannot provide a response to that request. "
            "It may contain content that violates safety policies.",
            result,
        )


# Default safety monitor instance.
default_safety_monitor = SafetyMonitor()