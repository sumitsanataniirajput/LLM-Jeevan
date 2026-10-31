"""Opt-in adapter for a user-supplied, externally isolated code sandbox."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

@dataclass(frozen=True, slots=True)
class SandboxResult:
    execution_ok: bool
    passed_tests: int
    total_tests: int
    status: str

    @property
    def unit_test_pass_rate(self) -> float:
        if self.total_tests <= 0:
            return 0.0
        return self.passed_tests / self.total_tests


class CodeSandboxCommand:
    """Call a user-installed sandbox launcher without shell interpolation.

    The launcher receives one JSON object on stdin with ``code`` and
    ``test_cases``. It must print one JSON object containing ``execution_ok``,
    ``passed_tests``, and ``total_tests``. This adapter enforces a wall-clock
    timeout and caps parsed result counts; the launcher, not this subprocess
    wrapper, must provide the actual OS/container isolation boundary.
    """

    def __init__(
        self,
        command: str | Path | Sequence[str],
        *,
        timeout_seconds: float = 10.0,
    ) -> None:
        if isinstance(command, (str, Path)):
            normalized = (str(command),)
        else:
            normalized = tuple(str(part) for part in command)
        if not normalized or any(not part for part in normalized):
            raise ValueError("sandbox command must contain a non-empty executable")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("sandbox timeout must be positive and finite")
        self.command = normalized
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        code: str,
        test_cases: list[dict[str, Any]],
        *,
        working_directory: str | Path | None = None,
    ) -> SandboxResult:
        """Run code through the configured launcher; never evaluates code here."""
        payload = json.dumps(
            {"code": code, "test_cases": test_cases},
            ensure_ascii=False,
            allow_nan=False,
        )
        if working_directory is None:
            with tempfile.TemporaryDirectory(prefix="puzzel-eval-sandbox-") as temp_dir:
                return self._invoke(payload, len(test_cases), Path(temp_dir))
        return self._invoke(payload, len(test_cases), Path(working_directory))

    def _invoke(
        self,
        payload: str,
        test_count: int,
        working_directory: Path,
    ) -> SandboxResult:
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                text=True,
                encoding="utf-8",
                errors="strict",
                capture_output=True,
                timeout=self.timeout_seconds,
                cwd=working_directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, 0, test_count, "timeout")
        except (OSError, UnicodeError) as exc:
            return SandboxResult(False, 0, test_count, f"launcher_error:{type(exc).__name__}")

        if completed.returncode != 0:
            return SandboxResult(False, 0, test_count, "launcher_failed")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return SandboxResult(False, 0, test_count, "invalid_launcher_output")
        if not isinstance(result, dict):
            return SandboxResult(False, 0, test_count, "invalid_launcher_output")
        execution_ok = result.get("execution_ok")
        passed_tests = result.get("passed_tests")
        reported_total = result.get("total_tests")
        if (
            not isinstance(execution_ok, bool)
            or not isinstance(passed_tests, int)
            or isinstance(passed_tests, bool)
            or not isinstance(reported_total, int)
            or isinstance(reported_total, bool)
            or reported_total != test_count
            or not 0 <= passed_tests <= test_count
        ):
            return SandboxResult(False, 0, test_count, "invalid_launcher_result")
        return SandboxResult(execution_ok, passed_tests, test_count, "ok")
