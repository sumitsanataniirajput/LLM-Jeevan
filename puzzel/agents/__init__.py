"""Tools and Agents package for Puzzel.

Provides tool registration, execution, sandboxing, and multi-step agent
orchestration. All tool execution is sandboxed; no arbitrary code runs
in the main backend process.

No external AI services are used.
"""

from puzzel.agents.executor import ToolExecutor, ToolResult
from puzzel.agents.multi_step import MultiStepAgent
from puzzel.agents.registry import ToolRegistry
from puzzel.agents.sandbox import Sandbox, SandboxResult

__all__ = [
    "ToolRegistry",
    "ToolExecutor",
    "ToolResult",
    "Sandbox",
    "SandboxResult",
    "MultiStepAgent",
]