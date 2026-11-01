"""Admin dashboard package for Puzzel.

Provides system overview, model info, checkpoint management, training
controls, dataset management, and other administrative functions.
"""

from puzzel.admin.dashboard import AdminDashboard, get_system_overview

__all__ = [
    "AdminDashboard",
    "get_system_overview",
]