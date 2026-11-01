"""Long-term conversational memory for Puzzel.

Provides user-specific preferences, custom system prompts, and
conversation memory. All data is stored locally via the database
package; no external services are used.
"""

from puzzel.memory.system_prompts import SystemPromptManager
from puzzel.memory.user_prefs import UserPreferences

__all__ = [
    "UserPreferences",
    "SystemPromptManager",
]