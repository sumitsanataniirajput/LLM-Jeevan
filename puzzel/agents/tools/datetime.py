"""Date and time tool for agents."""

from __future__ import annotations

from datetime import datetime


def datetime_tool(format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return the current date and time.

    Args:
        format_str: strftime format string for the output.

    Returns:
        Formatted date/time string.
    """
    return datetime.now().strftime(format_str)