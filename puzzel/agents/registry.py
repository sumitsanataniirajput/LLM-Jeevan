"""Tool registration, selection, and routing for agents.

Provides a registry where tools can be registered by name, with metadata
about their parameters and descriptions. Tools can be looked up and
routed based on name or capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDefinition:
    """Metadata for a registered tool.

    Attributes:
        name: Unique tool name.
        description: Human-readable description of what the tool does.
        parameters: JSON-schema-like dict describing the tool's parameters.
        callable: The actual Python callable implementing the tool.
        requires_sandbox: If True, the tool must run in the sandbox.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    callable: Callable[..., Any] | None = None
    requires_sandbox: bool = False


class ToolRegistry:
    """Registry of available tools for agent use.

    Tools are registered by name and can be looked up for execution.
    Duplicate registration raises an error.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        callable: Callable[..., Any] | None = None,
        parameters: dict[str, Any] | None = None,
        requires_sandbox: bool = False,
        replace: bool = False,
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name.
            description: Human-readable description.
            callable: The Python callable implementing the tool.
            parameters: JSON-schema-like parameter dict.
            requires_sandbox: If True, execution must be sandboxed.
            replace: If True, replace an existing tool with the same name.

        Raises:
            ValueError: If the tool is already registered and replace is False.
        """
        if name in self._tools and not replace:
            raise ValueError(f"tool '{name}' is already registered")
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            callable=callable,
            requires_sandbox=requires_sandbox,
        )

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: Tool name to remove.

        Raises:
            KeyError: If the tool is not registered.
        """
        del self._tools[name]

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name.

        Returns None if the tool is not registered.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return a sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        """Return all tool definitions."""
        return list(self._tools.values())

    def has(self, name: str) -> bool:
        """Return True if the tool is registered."""
        return name in self._tools

    def get_tools_for_prompt(self) -> list[dict[str, Any]]:
        """Return tool definitions formatted for model prompting.

        Each entry is a dict with name, description, and parameters.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]