"""Database package for Puzzel.

Provides data models, migrations, and backup/restore functionality
using SQLite. All data is stored locally; no external database
services are used.
"""

from puzzel.database.backup import backup_database, restore_database
from puzzel.database.migrations import migrate_database
from puzzel.database.models import (
    APIKey,
    AuditEvent,
    Conversation,
    Document,
    Feedback,
    KnowledgeBase,
    Message,
    UsageRecord,
    User,
)

__all__ = [
    "User",
    "Conversation",
    "Message",
    "APIKey",
    "UsageRecord",
    "Document",
    "KnowledgeBase",
    "AuditEvent",
    "Feedback",
    "migrate_database",
    "backup_database",
    "restore_database",
]