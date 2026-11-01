"""Tool execution engine for agents.

Executes registered tools with sandboxing support, result tracking,
and failure recovery. No external AI services are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from puzzel.agents.registry import ToolRegistry, ToolDefinition
from puzzel.agents.sandbox import Sandbox, SandboxResult


@dataclass
class ToolResult:
    """Result of a tool execution.

    Attributes:
        tool_name: Name of the executed tool.
        success: True if execution succeeded.
        output: The tool's return value or stdout output.
        error: Error message if execution failed.
        execution_time_seconds: Time taken to execute.
        sandboxed: Whether the tool ran in the sandbox.
    """

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_seconds: float = 0.0
    sandboxed: bool = False


class ToolExecutor:
    """Executes registered tools with optional sandboxing.

    Tools that require sandboxing (as indicated in their registry
    definition) are executed in the sandbox. Other tools are called
    directly.

    Attributes:
        registry: The ToolRegistry to look up tools from.
        sandbox: Sandbox instance for isolated execution.
    """

    def __init__(self, registry: ToolRegistry, sandbox: Sandbox | None = None) -> None:
        self.registry = registry
        self.sandbox = sandbox or Sandbox()
        self._execution_history: list[ToolResult] = []

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            ToolResult with the execution outcome.
        """
        arguments = arguments or {}
        start_time = time.perf_counter()

        definition = self.registry.get(tool_name)
        if definition is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"tool '{tool_name}' is not registered",
                execution_time_seconds=time.perf_counter() - start_time,
            )

        if definition.callable is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"tool '{tool_name}' has no callable implementation",
                execution_time_seconds=time.perf_counter() - start_time,
            )

        try:
            if definition.requires_sandbox:
                result = self._execute_sandboxed(definition, arguments)
            else:
                result = self._execute_direct(definition, arguments)

            result.execution_time_seconds = time.perf_counter() - start_time
            self._execution_history.append(result)
            return result
        except Exception as exc:  # noqa: BLE001
            result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
                execution_time_seconds=time.perf_counter() - start_time,
                sandboxed=definition.requires_sandbox,
            )
            self._execution_history.append(result)
            return result

    def _execute_direct(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool directly (not sandboxed)."""
        fn = definition.callable
        assert fn is not None
        output = fn(**arguments)
        return ToolResult(
            tool_name=definition.name,
            success=True,
            output=output,
            sandboxed=False,
        )

    def _execute_sandboxed(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool in the sandbox."""
        fn = definition.callable
        assert fn is not None

        # Serialize the function call as code to run in the sandbox.
        # We reconstruct the call using the function's source location.
        import inspect

        try:
            source_file = inspect.getfile(fn)
            source_lines = inspect.getsourcelines(fn)[0]
            # Build code that imports and calls the function.
            # This is a simplified approach; in production, we'd use
            # a more robust serialization mechanism.
            module_name = fn.__module__
            qualname = fn.__qualname__

            code = (
                f"import importlib\n"
                f"mod = importlib.import_module({module_name!r})\n"
                f"fn = mod\n"
                f"for part in {qualname!r}.split('.'):\n"
                f"    fn = getattr(fn, part)\n"
                f"result = fn({arguments!r})\n"
                f"print(result)\n"
            )
        except (OSError, TypeError):
            # Fallback: if we can't inspect the function, run it directly.
            return self._execute_direct(definition, arguments)

        sandbox_result = self.sandbox.run(code)
        return ToolResult(
            tool_name=definition.name,
            success=sandbox_result.success,
            output=sandbox_result.output if sandbox_result.success else None,
            error=sandbox_result.error,
            sandboxed=True,
        )

    def execution_history(self) -> list[ToolResult]:
        """Return the history of all tool executions."""
        return list(self._execution_history)

    def clear_history(self) -> None:
        """Clear the execution history."""
        self._execution_history.clear()