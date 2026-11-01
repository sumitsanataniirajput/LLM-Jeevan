"""Database backup and restore for Puzzel.

Provides SQLite backup and restore functionality. All backups are
local file copies; no external storage services are used.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


def backup_database(
    source_path: str | Path,
    backup_path: str | Path,
) -> Path:
    """Create a backup of the SQLite database.

    Uses SQLite's VACUUM INTO for a consistent backup, falling back
    to a simple file copy if not supported.

    Args:
        source_path: Path to the source database.
        backup_path: Path for the backup file.

    Returns:
        Path to the created backup file.
    """
    source = Path(source_path)
    destination = Path(backup_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"source database not found: {source}")

    try:
        # Use VACUUM INTO for a consistent, compact backup.
        conn = sqlite3.connect(str(source))
        conn.execute(f"VACUUM INTO '{destination}'")
        conn.close()
    except sqlite3.OperationalError:
        # Fall back to simple file copy.
        shutil.copy2(str(source), str(destination))

    return destination


def restore_database(
    backup_path: str | Path,
    target_path: str | Path,
) -> Path:
    """Restore a database from a backup file.

    Args:
        backup_path: Path to the backup file.
        target_path: Path where the database should be restored.

    Returns:
        Path to the restored database file.
    """
    backup = Path(backup_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not backup.exists():
        raise FileNotFoundError(f"backup file not found: {backup}")

    shutil.copy2(str(backup), str(target))
    return target


def verify_backup(backup_path: str | Path) -> bool:
    """Verify that a backup file is a valid SQLite database.

    Args:
        backup_path: Path to the backup file.

    Returns:
        True if the backup is valid, else False.
    """
    try:
        conn = sqlite3.connect(str(backup_path))
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        conn.close()
        return True
    except sqlite3.DatabaseError:
        return False