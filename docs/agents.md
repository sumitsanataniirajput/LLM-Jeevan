# Puzzel Agents Documentation

Puzzel includes a tool-using agent system with sandboxed execution,
multi-step orchestration, and built-in tools.

## Overview

The agent system consists of:

1. **ToolRegistry** — Register and discover tools
2. **Sandbox** — Isolated Python execution (no network, no file system)
3. **ToolExecutor** — Execute tools with sandboxing support
4. **MultiStepAgent** — ReAct-style multi-step agent

No external AI services are used. Tool selection is deterministic
based on keyword matching.

## Built-in Tools

| Tool | Description | Sandboxed |
|------|-------------|-----------|
| `calculator` | Evaluate mathematical expressions | Yes |
| `datetime` | Get current date and time | No |
| `unit_conversion` | Convert units (length, weight) | No |

## Usage

```python
from puzzel.agents import ToolRegistry, ToolExecutor, MultiStepAgent
from puzzel.agents.sandbox import Sandbox
from puzzel.agents.tools import calculator_tool, datetime_tool

# Set up registry
registry = ToolRegistry()
registry.register("calculator", "Calculate math expressions", calculator_tool, requires_sandbox=True)
registry.register("datetime", "Get current time", datetime_tool)

# Set up executor and agent
sandbox = Sandbox(timeout=10)
executor = ToolExecutor(registry, sandbox=sandbox)
agent = MultiStepAgent(registry, executor, max_steps=10)

# Run
result = agent.run("What is 2 + 3 * 4?")
print(result.final_answer)
```

## Sandbox

The sandbox runs untrusted code in a subprocess with:

- Forbidden imports blocked (os, sys, subprocess, socket, etc.)
- Function/class definitions blocked
- Timeout protection (default 10 seconds)
- Output size limits (1 MB max)
- No network access

```python
from puzzel.agents.sandbox import Sandbox

sandbox = Sandbox(timeout=5)
result = sandbox.run("print(2 + 3)")
print(result.output)  # "5\n"
print(result.success)  # True
```

## Multi-Step Agent

The agent follows a ReAct loop:

1. Think about what to do
2. Select a tool based on keyword matching
3. Execute the tool
4. Observe the result
5. Repeat until final answer or step limit

## Custom Tools

```python
def my_tool(text: str) -> str:
    return f"Processed: {text}"

registry.register("my_tool", "Process text", my_tool)
```

## Security

- All sandboxed code runs in a subprocess, not the main process.
- Forbidden imports are statically blocked.
- No network access from sandboxed code.
- Tool execution history is tracked for audit.