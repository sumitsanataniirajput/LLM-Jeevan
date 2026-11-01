"""Tests for Prompt 1 security fixes: scrypt hashing, migration flag,
constant-time comparison, bounded thread-safe rate limiter, loopback defaults.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

from puzzel.auth import security
from puzzel.auth.models import AuthStore, User
from puzzel.auth.security import (
    LEGACY_MIGRATION_ENV_VAR,
    RateLimiter,
    hash_password,
    needs_rehash,
    rate_limit,
    verify_password,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _legacy_hash(password: str, salt: str = "abcd1234") -> str:
    """Build a legacy-format 'salt:hash' value (single-round salted SHA-256)."""
    return f"{salt}:{hashlib.sha256((password + salt).encode()).hexdigest()}"


def _double_legacy_hash(password: str, salt: str = "abcd1234") -> str:
    """Build the historical double-hash value that must NOT be accepted."""
    once = hashlib.sha256((password + salt).encode("utf-8")).hexdigest()
    twice = hashlib.sha256(once.encode("utf-8")).hexdigest()
    return f"{salt}:{twice}"


# --- scrypt hashing -----------------------------------------------------------


def test_hash_password_uses_self_describing_scrypt_format():
    stored = hash_password("s3cret-password")
    parts = stored.split("$")
    assert parts[0] == "scrypt"
    assert len(parts) == 6
    n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
    assert (n, r, p) == (security.SCRYPT_N, security.SCRYPT_R, security.SCRYPT_P)
    assert len(bytes.fromhex(parts[4])) == 16  # 16-byte random salt
    assert len(bytes.fromhex(parts[5])) == security.SCRYPT_DKLEN


def test_hash_password_random_salt_each_call():
    assert hash_password("pw") != hash_password("pw")


def test_verify_password_scrypt_roundtrip():
    stored = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", stored) is True
    assert verify_password("wrong", stored) is False


def test_verify_password_rejects_malformed_scrypt():
    assert verify_password("pw", "scrypt$not$ints$x$y") is False
    assert verify_password("pw", "scrypt$16384$8$1$zz$00") is False
    assert verify_password("pw", "") is False


def test_verify_password_uses_constant_time_compare(monkeypatch):
    calls = []
    real = security.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(security.hmac, "compare_digest", spy)
    stored = hash_password("pw")
    assert verify_password("pw", stored) is True
    assert calls, "hmac.compare_digest must be used for verification"


# --- legacy migration flag -----------------------------------------------------


def test_legacy_hash_rejected_when_migration_flag_off(monkeypatch):
    monkeypatch.delenv(LEGACY_MIGRATION_ENV_VAR, raising=False)
    stored = _legacy_hash("pw")
    assert verify_password("pw", stored) is False
    assert verify_password("pw", stored, allow_legacy=False) is False


def test_legacy_hash_accepted_only_when_flag_on(monkeypatch):
    stored = _legacy_hash("pw")
    assert verify_password("pw", stored, allow_legacy=True) is True
    assert verify_password("wrong", stored, allow_legacy=True) is False
    monkeypatch.setenv(LEGACY_MIGRATION_ENV_VAR, "true")
    assert verify_password("pw", stored) is True


def test_double_hash_branch_removed(monkeypatch):
    """The old double-hash match must never verify, even with the flag on."""
    monkeypatch.setenv(LEGACY_MIGRATION_ENV_VAR, "true")
    stored = _double_legacy_hash("pw")
    assert verify_password("pw", stored) is False
    assert verify_password("pw", stored, allow_legacy=True) is False


def test_needs_rehash():
    assert needs_rehash(_legacy_hash("pw")) is True
    assert needs_rehash(hash_password("pw")) is False


def test_auth_store_rehashes_on_successful_legacy_login(monkeypatch):
    monkeypatch.setenv(LEGACY_MIGRATION_ENV_VAR, "true")
    store = AuthStore()
    store.add_user(User(username="u", password_hash=_legacy_hash("pw")))
    user = store.authenticate("u", "pw")
    assert user is not None
    assert user.password_hash.startswith("scrypt$")
    # After rehash, login works without the migration flag.
    monkeypatch.delenv(LEGACY_MIGRATION_ENV_VAR)
    assert store.authenticate("u", "pw") is not None


def test_auth_store_rejects_legacy_login_when_flag_off(monkeypatch):
    monkeypatch.delenv(LEGACY_MIGRATION_ENV_VAR, raising=False)
    store = AuthStore()
    store.add_user(User(username="u", password_hash=_legacy_hash("pw")))
    assert store.authenticate("u", "pw") is None


# --- rate limiter --------------------------------------------------------------


def test_rate_limiter_enforces_window():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)
    assert [limiter.is_allowed("c") for _ in range(5)] == [True, True, True, False, False]


def test_rate_limiter_evicts_idle_clients():
    limiter = RateLimiter(max_requests=5, window_seconds=0.05, idle_evict_seconds=0.05)
    assert limiter.is_allowed("old") is True
    import time

    time.sleep(0.08)
    assert limiter.is_allowed("new") is True
    assert limiter.client_count <= 2
    # "old" was idle-evicted, so it gets a fresh bucket.
    assert limiter.is_allowed("old") is True


def test_rate_limiter_memory_bounded_under_100k_clients():
    limiter = RateLimiter(max_requests=2, window_seconds=60.0, max_clients=1_000)
    for i in range(100_000):
        limiter.is_allowed(f"client-{i}")
    assert limiter.client_count <= limiter.max_clients


def test_rate_limiter_thread_safe():
    limiter = RateLimiter(max_requests=50, window_seconds=60.0, max_clients=10)
    allowed = []
    lock = threading.Lock()

    def worker():
        count = 0
        for _ in range(200):
            if limiter.is_allowed("shared"):
                count += 1
        with lock:
            allowed.append(count)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(allowed) == 50  # exactly max_requests allowed, no races
    assert limiter.client_count == 1


def test_rate_limit_function_uses_given_limiter():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    assert rate_limit("x", limiter) is True
    assert rate_limit("x", limiter) is False


# --- loopback defaults ----------------------------------------------------------


def test_api_cli_default_host_is_loopback():
    from puzzel.api import DEFAULT_HOST, build_arg_parser, is_loopback_host

    assert DEFAULT_HOST == "127.0.0.1"
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.host == "127.0.0.1"
    assert is_loopback_host(args.host)
    assert not is_loopback_host("0.0.0.0")


def test_api_cli_refuses_non_loopback_without_public_flag(monkeypatch):
    monkeypatch.delenv("PUZZEL_HOST", raising=False)
    from puzzel.api import build_arg_parser

    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        # Simulate what main() does for a non-loopback host without --public.
        args = parser.parse_args(["--host", "0.0.0.0"])
        from puzzel.api import is_loopback_host

        if not is_loopback_host(args.host) and not args.public:
            parser.error("refusing to bind without --public")


def test_env_example_default_host_is_loopback():
    text = (REPO_ROOT / ".env.example").read_text()
    assert "PUZZEL_HOST=127.0.0.1" in text
    assert "PUZZEL_HOST=0.0.0.0" not in text


def test_serve_sh_default_host_is_loopback_and_requires_public():
    text = (REPO_ROOT / "scripts" / "serve.sh").read_text()
    assert 'HOST="127.0.0.1"' in text
    assert 'HOST="0.0.0.0"' not in text
    assert "--public" in text
    assert "Refusing to bind to non-loopback host" in text
    assert "Arena" not in text


def test_no_stale_sandbox_paths_in_serve_sh_and_run_plan():
    for rel in ("scripts/serve.sh", "training_run_plan.md"):
        text = (REPO_ROOT / rel).read_text()
        assert "/home/user" not in text, rel
        assert "arena" not in text.lower(), rel  # no Arena proxy/branch refs
        assert "e2b" not in text.lower(), rel


# --- middleware secret compares (auth hardening) --------------------------------


def test_api_key_auth_uses_constant_time_compare_and_valid_keys_still_work(monkeypatch):
    from puzzel.auth import middleware, security
    from puzzel.auth.middleware import APIKeyAuth, create_default_store

    store = create_default_store()
    auth = APIKeyAuth(store, header_name="X-API-Key")
    calls = []
    real = security.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(middleware.hmac, "compare_digest", spy)
    # valid API-key user: behavior unchanged
    ctx = auth.authenticate({"X-API-Key": "puzzel-admin-key-001"})
    assert ctx.authenticated is True
    assert ctx.auth_method == "api_key"
    assert ctx.user.username == "admin"
    assert calls, "hmac.compare_digest must be used for API key comparison"
    # wrong key: rejected
    assert auth.authenticate({"X-API-Key": "wrong"}).authenticated is False
    # missing key: unauthenticated
    assert auth.authenticate({}).authenticated is False


def test_jwt_auth_uses_constant_time_compare_and_valid_tokens_still_work(monkeypatch):
    from puzzel.auth import middleware, security
    from puzzel.auth.middleware import JWTAuth
    from puzzel.auth.models import AuthStore, User

    store = AuthStore()
    store.add_user(User(username="u", password_hash="", metadata={"jwt_token": "tok-123"}))
    auth = JWTAuth(store)
    calls = []
    real = security.hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real(a, b)

    monkeypatch.setattr(middleware.hmac, "compare_digest", spy)
    ctx = auth.authenticate({"Authorization": "Bearer tok-123"})
    assert ctx.authenticated is True
    assert calls, "hmac.compare_digest must be used for token comparison"
    assert auth.authenticate({"Authorization": "Bearer nope"}).authenticated is False
    assert auth.authenticate({"Authorization": "tok-123"}).authenticated is False


def test_warns_when_auth_disabled_on_non_loopback(caplog):
    import logging

    from puzzel.api import warn_if_auth_disabled_on_non_loopback

    with caplog.at_level(logging.WARNING, logger="puzzel.api"):
        warn_if_auth_disabled_on_non_loopback("0.0.0.0", auth_enabled=False)
    assert any("Authentication is disabled" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="puzzel.api"):
        warn_if_auth_disabled_on_non_loopback("0.0.0.0", auth_enabled=True)
        warn_if_auth_disabled_on_non_loopback("127.0.0.1", auth_enabled=False)
    assert not caplog.records
