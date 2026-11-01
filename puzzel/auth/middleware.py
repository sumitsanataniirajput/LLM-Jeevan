"""Authentication middleware: API key, JWT, and role-based access control."""

from __future__ import annotations

import hmac
from dataclasses import dataclass

from puzzel.auth.models import AuthStore, Permission, Role, User
from puzzel.auth.security import hash_password


@dataclass
class AuthContext:
    """Security context for an authenticated request.

    Attributes:
        user: The authenticated User, or None if unauthenticated.
        authenticated: Whether authentication succeeded.
        auth_method: "api_key", "jwt", or "none".
    """

    user: User | None = None
    authenticated: bool = False
    auth_method: str = "none"


class APIKeyAuth:
    """API key authentication middleware.

    Checks for an API key in the request header "X-API-Key" and looks
    it up in the auth store.
    """

    def __init__(self, store: AuthStore, header_name: str = "X-API-Key") -> None:
        self.store = store
        self.header_name = header_name

    def authenticate(self, headers: dict[str, str]) -> AuthContext:
        """Authenticate a request based on its headers.

        Args:
            headers: HTTP headers dict.

        Returns:
            AuthContext with the authenticated user if the API key is valid.
        """
        api_key = headers.get(self.header_name)
        if not api_key:
            return AuthContext(authenticated=False, auth_method="none")

        for user in self.store._users.values():
            # constant-time compare: API keys are secrets, so do not leak a
            # prefix match through timing (hmac.compare_digest).
            if (
                user.api_key is not None
                and hmac.compare_digest(user.api_key, api_key)
                and user.is_active
            ):
                return AuthContext(
                    user=user,
                    authenticated=True,
                    auth_method="api_key",
                )
        return AuthContext(authenticated=False, auth_method="api_key")


class JWTAuth:
    """JWT token authentication (simplified, no external crypto deps).

    Tokens are opaque strings stored in the auth store. In production,
    use a proper JWT library with signed tokens.
    """

    def __init__(self, store: AuthStore, header_name: str = "Authorization") -> None:
        self.store = store
        self.header_name = header_name

    def authenticate(self, headers: dict[str, str]) -> AuthContext:
        """Authenticate a request based on its headers.

        Args:
            headers: HTTP headers dict.

        Returns:
            AuthContext with the authenticated user if the token is valid.
        """
        auth_header = headers.get(self.header_name, "")
        if not auth_header.startswith("Bearer "):
            return AuthContext(authenticated=False, auth_method="none")

        token = auth_header[7:]  # Strip "Bearer "
        for user in self.store._users.values():
            stored_token = user.metadata.get("jwt_token")
            if (
                stored_token is not None
                and hmac.compare_digest(str(stored_token), token)
                and user.is_active
            ):
                return AuthContext(
                    user=user,
                    authenticated=True,
                    auth_method="jwt",
                )
        return AuthContext(authenticated=False, auth_method="none")


class RoleMiddleware:
    """Role-based access control middleware.

    Checks whether the authenticated user has the required permission
    for a given endpoint or action.
    """

    def __init__(self, store: AuthStore) -> None:
        self.store = store

    def check_permission(
        self,
        context: AuthContext,
        permission_name: str,
    ) -> bool:
        """Return True if the user has the required permission."""
        if not context.authenticated or context.user is None:
            return False
        return context.user.has_permission(permission_name, self.store._roles)

    def require_permission(
        self,
        context: AuthContext,
        permission_name: str,
    ) -> None:
        """Raise PermissionError if the user lacks the permission."""
        if not self.check_permission(context, permission_name):
            raise PermissionError(
                f"user {context.user.username if context.user else 'anonymous'} "
                f"lacks permission: {permission_name}"
            )


def create_default_store() -> AuthStore:
    """Create an auth store with default roles and a guest user.

    Roles:
        admin: All permissions.
        user: model.generate, model.info.
        guest: model.info only.

    Returns:
        Configured AuthStore instance.
    """
    store = AuthStore()

    # Permissions.
    store.add_permission(Permission("model.generate", "Generate text from model"))
    store.add_permission(Permission("model.info", "View model information"))
    store.add_permission(Permission("admin.users", "Manage users"))
    store.add_permission(Permission("admin.system", "System administration"))

    # Roles.
    store.add_role(Role("admin", {"model.generate", "model.info", "admin.users", "admin.system"}))
    store.add_role(Role("user", {"model.generate", "model.info"}))
    store.add_role(Role("guest", {"model.info"}))

    # Default users.
    store.add_user(
        User(
            username="admin",
            password_hash=hash_password("admin"),
            roles={"admin"},
            api_key="puzzel-admin-key-001",
        )
    )
    store.add_user(
        User(
            username="guest",
            password_hash="",
            roles={"guest"},
            api_key=None,
        )
    )

    return store
