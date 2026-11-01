"""Security utilities: password hashing, input sanitization, rate limiting.

Password hashing uses ``hashlib.scrypt`` (stdlib, memory-hard) with a
per-password random salt and documented cost parameters. Hashes are stored
in a self-describing format::

    scrypt$<n>$<r>$<p>$<salt-hex>$<hash-hex>

Legacy ``salt:hash`` values (single-round salted SHA-256) are **rejected by
default**; they are only readable when the migration flag is explicitly
enabled (parameter ``allow_legacy=True`` or environment variable
``PUZZEL_ALLOW_LEGACY_SHA256_MIGRATION=true``), and any successful legacy
login is rehashed to scrypt by the caller (see ``AuthStore.authenticate``).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time

# --- scrypt cost parameters (documented) -------------------------------------
# n=16384 (2**14), r=8, p=1 => ~16 MiB memory per hash, ~50-100 ms on CPU.
# Memory-hardness makes GPU/ASIC brute-forcing expensive. Do not lower these
# without a security review; raise maxmem accordingly if you raise them.
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024  # 64 MiB ceiling for hashlib.scrypt

SCRYPT_PREFIX = "scrypt"
LEGACY_MIGRATION_ENV_VAR = "PUZZEL_ALLOW_LEGACY_SHA256_MIGRATION"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _legacy_migration_enabled() -> bool:
    """Whether legacy SHA-256 hash verification is allowed (migration flag)."""
    return os.environ.get(LEGACY_MIGRATION_ENV_VAR, "false").strip().lower() in _TRUE_VALUES


def hash_password(password: str, salt: str | None = None) -> str:
    """Hash a password with scrypt and a per-password random salt.

    Args:
        password: The plaintext password.
        salt: Optional hex-encoded salt. If None, a random 16-byte salt is
            generated.

    Returns:
        Self-describing hash string ``scrypt$n$r$p$salt$hash``.
    """
    if not isinstance(password, str):
        raise TypeError(f"password must be str, got {type(password).__name__}")
    if salt is None:
        salt = os.urandom(16).hex()
    try:
        salt_bytes = bytes.fromhex(salt)
    except ValueError as exc:
        raise ValueError("salt must be a hex string") from exc
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt_bytes,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_DKLEN,
        maxmem=SCRYPT_MAXMEM,
    )
    return f"{SCRYPT_PREFIX}${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt}${derived.hex()}"


def _verify_scrypt(password: str, stored: str) -> bool:
    """Verify a password against a ``scrypt$n$r$p$salt$hash`` string."""
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != SCRYPT_PREFIX:
        return False
    _, n_str, r_str, p_str, salt, expected = parts
    try:
        n, r, p = int(n_str), int(r_str), int(p_str)
        salt_bytes = bytes.fromhex(salt)
        expected_bytes = bytes.fromhex(expected)
    except ValueError:
        return False
    if n <= 0 or r <= 0 or p <= 0 or not salt_bytes or not expected_bytes:
        return False
    try:
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt_bytes,
            n=n,
            r=r,
            p=p,
            dklen=len(expected_bytes),
            maxmem=SCRYPT_MAXMEM,
        )
    except (ValueError, OverflowError):
        return False
    return hmac.compare_digest(derived, expected_bytes)


def _verify_legacy_sha256(password: str, stored: str) -> bool:
    """Verify a password against a legacy ``salt:hash`` SHA-256 value.

    Only single-round salted SHA-256 is accepted here. The historical
    double-hash match is deliberately NOT accepted.
    """
    try:
        salt, expected = stored.split(":", 1)
    except ValueError:
        return False
    digest = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, expected)


def verify_password(
    password: str,
    stored: str,
    *,
    allow_legacy: bool | None = None,
) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: The plaintext password to check.
        stored: The stored hash. ``scrypt$...`` values are always accepted
            (constant-time compare). Legacy ``salt:hash`` values are accepted
            only when the migration flag is on.
        allow_legacy: Explicit migration flag override. When None, the
            environment variable ``PUZZEL_ALLOW_LEGACY_SHA256_MIGRATION``
            decides (default: off).

    Returns:
        True if the password matches, else False.
    """
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    if stored.startswith(f"{SCRYPT_PREFIX}$"):
        return _verify_scrypt(password, stored)
    if allow_legacy is None:
        allow_legacy = _legacy_migration_enabled()
    if allow_legacy:
        return _verify_legacy_sha256(password, stored)
    return False


def needs_rehash(stored: str) -> bool:
    """Return True if the stored hash is not in the current scrypt format."""
    return not (isinstance(stored, str) and stored.startswith(f"{SCRYPT_PREFIX}$"))


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
    """Thread-safe token-bucket rate limiter per client identifier.

    Memory is bounded: idle client ids are evicted after
    ``idle_evict_seconds`` (default: one window), and at most
    ``max_clients`` ids are tracked. When the cap is reached, the least
    recently active client is evicted to make room.

    Attributes:
        max_requests: Maximum requests per window per client.
        window_seconds: Window duration in seconds.
        max_clients: Maximum number of tracked client ids.
        idle_evict_seconds: Idle time after which a client bucket is evicted.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        max_clients: int = 10_000,
        idle_evict_seconds: float | None = None,
    ) -> None:
        if max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_clients <= 0:
            raise ValueError("max_clients must be positive")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self.idle_evict_seconds = (
            idle_evict_seconds if idle_evict_seconds is not None else window_seconds
        )
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @property
    def client_count(self) -> int:
        """Number of currently tracked client ids."""
        with self._lock:
            return len(self._buckets)

    def _evict_idle_locked(self, now: float) -> None:
        cutoff = now - self.idle_evict_seconds
        stale = [
            client_id
            for client_id, bucket in self._buckets.items()
            if not bucket or bucket[-1] <= cutoff
        ]
        for client_id in stale:
            del self._buckets[client_id]

    def _evict_oldest_locked(self) -> None:
        if not self._buckets:
            return
        oldest = min(
            self._buckets,
            key=lambda cid: self._buckets[cid][-1] if self._buckets[cid] else 0.0,
        )
        del self._buckets[oldest]

    def is_allowed(self, client_id: str) -> bool:
        """Return True if the client can make a request now."""
        now = time.time()
        with self._lock:
            self._evict_idle_locked(now)
            bucket = self._buckets.get(client_id)
            if bucket is None:
                if len(self._buckets) >= self.max_clients:
                    self._evict_oldest_locked()
                if len(self._buckets) >= self.max_clients:
                    # Cap still reached (single oversized entry); refuse rather
                    # than grow without bound.
                    return False
                bucket = []
                self._buckets[client_id] = bucket
            cutoff = now - self.window_seconds
            while bucket and bucket[0] <= cutoff:
                bucket.pop(0)
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        """Drop all tracked clients (mainly for tests)."""
        with self._lock:
            self._buckets.clear()


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
