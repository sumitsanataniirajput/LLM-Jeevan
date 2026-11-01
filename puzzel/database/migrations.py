"""Database migrations for Puzzel.

Handles schema versioning, index creation, and data validation
for the SQLite database.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# Migration version.
SCHEMA_VERSION = 1

MIGRATIONS = [
    # Version 1: Initial schema (defined in Database._init_schema).
    {
        "version": 1,
        "description": "Initial schema with all core tables",
        "sql": "",  # Schema is created in Database._init_schema
    },
]


def migrate_database(db_path: str | Path) -> int:
    """Run pending migrations on the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        The schema version after migration.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "version INTEGER PRIMARY KEY)"
        )
        row = conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        current_version = row[0] if row else 0

        for migration in MIGRATIONS:
            if migration["version"] > current_version:
                if migration["sql"]:
                    conn.executescript(migration["sql"])
                conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (migration["version"],),
                )
                current_version = migration["version"]

        conn.commit()
        return current_version
    finally:
        conn.close()


def validate_database(db_path: str | Path) -> list[str]:
    """Validate the database schema and return any issues.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        List of validation issue descriptions (empty if valid).
    """
    issues: list[str] = []
    conn = sqlite3.connect(str(db_path))
    try:
        # Check required tables exist.
        required_tables = [
            "users", "roles", "conversations", "messages",
            "api_keys", "usage_records", "documents",
            "knowledge_bases", "audit_events", "feedback",
        ]
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in required_tables:
            if table not in existing:
                issues.append(f"missing table: {table}")

        # Check schema version.
        try:
            row = conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()
            if row is None:
                issues.append("schema version not set")
            elif row[0] < 1:
                issues.append(f"outdated schema version: {row[0]}")
        except sqlite3.OperationalError:
            issues.append("schema_version table missing")

    finally:
        conn.close()
    return issues