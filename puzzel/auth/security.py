"""Security utilities: password hashing, input sanitization, rate limiting."""

from __future__ import annotations

import hashlib
import os
import re
import time
from collections import defaultdict
from typing import Any


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a password using SHA-256 with a random salt.

    Args:
        password: The plaintext password.
        salt: Optional salt. If None, a random salt is generated.

    Returns:
        Hash string in format "salt:hash".
    """
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_hash = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
    return f"{salt}:{pwd_hash}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: The plaintext password to check.
        stored: The stored hash string in format "salt:hash".

    Returns:
        True if the password matches, else False.
    """
    try:
        salt, pwd_hash = stored.split(":", 1)
        expected = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
        return hashlib.sha256((expected).encode("utf-8")).hexdigest() == pwd_hash or expected == pwd_hash
    except (ValueError, AttributeError):
        return False


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input to prevent injection attacks.

    Removes null bytes, control characters, and truncates to max_length.

    Args:
        text: Raw input text.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text.
    """
    if not isinstance(text, str):
        raise TypeError(f"input must be str, got {type(text).__name__}")
    # Remove null bytes and control characters except newlines/tabs.
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return sanitized[:max_length]


class RateLimiter:
    """Simple token-bucket rate limiter per client identifier.

    Attributes:
        max_requests: Maximum requests per window.
        window_seconds: Window duration in seconds.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Return True if the client can make a request now."""
        now = time.time()
        bucket = self._buckets[client_id]
        # Remove timestamps outside the window.
        cutoff = now - self.window_seconds
        self._buckets[client_id] = [t for t in bucket if t > cutoff]
        bucket = self._buckets[client_id]

        if len(bucket) >= self.max_requests:
            return False

        bucket.append(now)
        return True


# Default rate limiter instance.
default_rate_limiter = RateLimiter()


def rate_limit(client_id: str, limiter: RateLimiter | None = None) -> bool:
    """Check if a client is rate-limited.

    Args:
        client_id: Unique client identifier.
        limiter: Optional RateLimiter instance. Uses default if None.

    Returns:
        True if the request is allowed, False if rate-limited.
    """
    lim = limiter or default_rate_limiter
    return lim.is_allowed(client_id)