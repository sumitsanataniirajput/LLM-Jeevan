"""Authentication and authorization package for Puzzel.

Provides user/role/permission models, API key auth, JWT token support,
and role-based middleware. All auth is local; no external identity
providers are used.
"""

from puzzel.auth.middleware import APIKeyAuth, JWTAuth, RoleMiddleware
from puzzel.auth.models import Permission, Role, User
from puzzel.auth.security import hash_password, rate_limit, sanitize_input, verify_password

__all__ = [
    "User",
    "Role",
    "Permission",
    "APIKeyAuth",
    "JWTAuth",
    "RoleMiddleware",
    "hash_password",
    "verify_password",
    "sanitize_input",
    "rate_limit",
]
