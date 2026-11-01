"""User, role, and permission models for authentication."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Permission:
    """A single permission (e.g., "model.generate", "admin.users")."""

    name: str
    description: str = ""


@dataclass
class Role:
    """A role with a set of permissions.

    Attributes:
        name: Role name (e.g., "admin", "user", "guest").
        permissions: Set of permission names.
    """

    name: str
    permissions: set[str] = field(default_factory=set)

    def has_permission(self, permission_name: str) -> bool:
        """Return True if the role has the given permission."""
        return permission_name in self.permissions


@dataclass
class User:
    """A user account.

    Attributes:
        username: Unique username.
        password_hash: Hashed password (never plaintext).
        roles: Set of role names assigned to the user.
        api_key: Optional API key for programmatic access.
        is_active: Whether the account is active.
        metadata: Additional user metadata.
    """

    username: str
    password_hash: str = ""
    roles: set[str] = field(default_factory=set)
    api_key: str | None = None
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission_name: str, roles: dict[str, Role]) -> bool:
        """Return True if any of the user's roles has the permission."""
        if not self.is_active:
            return False
        for role_name in self.roles:
            role = roles.get(role_name)
            if role and role.has_permission(permission_name):
                return True
        return False


class AuthStore:
    """In-memory store for users, roles, and permissions.

    In production, this would be backed by a database.
    """

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._roles: dict[str, Role] = {}
        self._permissions: dict[str, Permission] = {}

    def add_permission(self, permission: Permission) -> None:
        self._permissions[permission.name] = permission

    def add_role(self, role: Role) -> None:
        self._roles[role.name] = role

    def add_user(self, user: User) -> None:
        self._users[user.username] = user

    def get_user(self, username: str) -> User | None:
        return self._users.get(username)

    def get_role(self, name: str) -> Role | None:
        return self._roles.get(name)

    def get_permission(self, name: str) -> Permission | None:
        return self._permissions.get(name)

    def list_users(self) -> list[str]:
        return sorted(self._users.keys())

    def authenticate(self, username: str, password: str) -> User | None:
        """Authenticate a user by username and password.

        Returns the User if authentication succeeds, else None.
        """
        from puzzel.auth.security import verify_password

        user = self.get_user(username)
        if user is None or not user.is_active:
            return None
        if not user.password_hash:
            return None
        if verify_password(password, user.password_hash):
            return user
        return None