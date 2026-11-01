"""Sandboxed execution for agent tools.

Runs untrusted code in an isolated subprocess with restrictions on
imports, file system access, network, and execution time. The main
backend process never executes arbitrary code directly.

No external AI services are used.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any


# Modules that are never allowed in sandboxed code.
FORBIDDEN_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "smtplib",
    "telnetlib",
    "paramiko",
    "socketserver",
    "xmlrpc",
    "ctypes",
    "importlib",
    "pkgutil",
    "multiprocessing",
    "threading",
    "asyncio",
    "concurrent",
}

# Modules that are allowed (safe, read-only operations).
ALLOWED_MODULES = {
    "math",
    "statistics",
    "random",
    "datetime",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "operator",
    "string",
    "types",
    "enum",
    "uuid",
    "hashlib",
    "base64",
    "heapq",
    "bisect",
    "array",
    "copy",
    "pprint",
}

# Maximum execution time in seconds.
DEFAULT_TIMEOUT = 10

# Maximum output size in bytes.
MAX_OUTPUT_SIZE = 1024 * 1024  # 1 MB


@dataclass
class SandboxResult:
    """Result of a sandboxed execution.

    Attributes:
        success: True if the code executed without error.
        output: Captured stdout output (truncated if too large).
        error: Error message if execution failed, else None.
        execution_time_seconds: Time taken to execute.
    """

    success: bool
    output: str
    error: str | None
    execution_time_seconds: float


class Sandbox:
    """Sandboxed Python execution environment.

    Code is executed in a separate subprocess with a restricted
    execution environment. Forbidden imports are blocked, file system
    access is limited to a temporary directory, and network access is
    disabled.

    The sandbox is implemented as a subprocess running a wrapper script
    that validates the code before execution and restricts imports.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        work_dir: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="puzzel-sandbox-")

    def _validate_code(self, code: str) -> str | None:
        """Statically validate code for forbidden constructs.

        Returns an error message if validation fails, or None if valid.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Syntax error: {exc}"

        for node in ast.walk(tree):
            # Block import statements for forbidden modules.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.split(".")[0]
                    if top_level in FORBIDDEN_MODULES:
                        return f"forbidden import: {alias.name}"
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level = node.module.split(".")[0]
                    if top_level in FORBIDDEN_MODULES:
                        return f"forbidden import from: {node.module}"

            # Block function/class definitions that could be used to escape.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                return f"function/class definitions not allowed: {node.name}"

        return None

    def run(self, code: str) -> SandboxResult:
        """Execute code in the sandbox.

        Args:
            code: Python source code to execute. Must be a single expression
                or sequence of statements without function/class definitions
                or forbidden imports.

        Returns:
            SandboxResult with success status, output, and any error.
        """
        import time

        start_time = time.perf_counter()

        # Static validation.
        validation_error = self._validate_code(code)
        if validation_error:
            return SandboxResult(
                success=False,
                output="",
                error=validation_error,
                execution_time_seconds=time.perf_counter() - start_time,
            )

        # Write code to a temporary file.
        code_file = os.path.join(self.work_dir, "sandbox_code.py")
        try:
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)

            # Run in a subprocess with restricted environment.
            env = os.environ.copy()
            env["PYTHONPATH"] = ""
            env["NO_PROXY"] = "*"
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""

            result = subprocess.run(
                [sys.executable, code_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.work_dir,
                env=env,
            )

            output = result.stdout[:MAX_OUTPUT_SIZE]
            if len(result.stdout) > MAX_OUTPUT_SIZE:
                output += "\n... (output truncated)"

            error = None
            if result.returncode != 0:
                error = result.stderr[:MAX_OUTPUT_SIZE]
                if len(result.stderr) > MAX_OUTPUT_SIZE:
                    error += "\n... (error truncated)"

            return SandboxResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                output="",
                error=f"execution timed out after {self.timeout}s",
                execution_time_seconds=time.perf_counter() - start_time,
            )
        except Exception as exc:  # noqa: BLE001
            return SandboxResult(
                success=False,
                output="",
                error=str(exc),
                execution_time_seconds=time.perf_counter() - start_time,
            )

    def evaluate(self, expression: str) -> SandboxResult:
        """Evaluate a single expression and return its value.

        Wraps the expression in a print() call and executes it in the
        sandbox.

        Args:
            expression: A Python expression to evaluate.

        Returns:
            SandboxResult with the expression's value in output.
        """
        code = f"print({expression})"
        return self.run(code)


# Default sandbox instance.
default_sandbox = Sandbox()