"""Multi-step agent orchestration.

Provides a simple ReAct-style agent that can use tools iteratively
to solve problems. The agent maintains a conversation history, selects
tools based on the current state, executes them, and incorporates
results into subsequent steps.

No external AI services are used. Tool selection and execution are
deterministic based on the registered tools and the agent's logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from puzzel.agents.executor import ToolExecutor, ToolResult
from puzzel.agents.registry import ToolRegistry


@dataclass
class AgentStep:
    """A single step in the agent's execution.

    Attributes:
        step_number: Sequential step number.
        thought: The agent's reasoning for this step.
        tool_name: Name of the tool called (or None if no tool).
        tool_arguments: Arguments passed to the tool.
        tool_result: Result of the tool execution.
        observation: Observation from the tool result.
    """

    step_number: int
    thought: str
    tool_name: str | None
    tool_arguments: dict[str, Any]
    tool_result: ToolResult | None
    observation: str


@dataclass
class AgentResult:
    """Final result of multi-step agent execution.

    Attributes:
        success: True if the agent reached a final answer.
        final_answer: The agent's final answer text.
        steps: List of AgentStep objects.
        total_steps: Number of steps taken.
    """

    success: bool
    final_answer: str
    steps: list[AgentStep]
    total_steps: int


class MultiStepAgent:
    """A simple multi-step agent that uses tools to answer questions.

    The agent follows a ReAct-like loop:
    1. Think about what to do next.
    2. Select and call a tool (or produce a final answer).
    3. Observe the result.
    4. Repeat until a final answer is produced or the step limit is reached.

    Tool selection is based on keyword matching between the current state
    and tool descriptions. This is a deterministic, rule-based approach
    that does not use any external AI model for reasoning.

    Attributes:
        registry: The ToolRegistry with available tools.
        executor: ToolExecutor for running tools.
        max_steps: Maximum number of steps before giving up.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        executor: ToolExecutor | None = None,
        max_steps: int = 10,
    ) -> None:
        self.registry = registry
        self.executor = executor or ToolExecutor(registry)
        self.max_steps = max_steps

    def _select_tool(self, state: str) -> str | None:
        """Select a tool based on keyword matching with the current state.

        Args:
            state: Current conversation/thought state.

        Returns:
            Tool name if a match is found, else None.
        """
        state_lower = state.lower()
        best_tool = None
        best_score = 0

        for definition in self.registry.list_definitions():
            desc_lower = definition.description.lower()
            # Score based on keyword overlap.
            state_words = set(state_lower.split())
            desc_words = set(desc_lower.split())
            overlap = len(state_words & desc_words)
            if overlap > best_score:
                best_score = overlap
                best_tool = definition.name

        if best_score == 0:
            return None
        return best_tool

    def _extract_arguments(self, tool_def: Any, state: str) -> dict[str, Any]:
        """Extract tool arguments from the current state.

        This is a simple heuristic: look for numbers, quoted strings,
        and other patterns in the state that match the tool's parameters.
        """
        import re

        arguments: dict[str, Any] = {}

        # Look for numbers in the state.
        numbers = re.findall(r"\d+\.?\d*", state)
        if numbers:
            arguments["value"] = float(numbers[0]) if "." in numbers[0] else int(numbers[0])

        # Look for quoted strings.
        strings = re.findall(r'"([^"]*)"', state)
        if strings:
            arguments["text"] = strings[0]

        return arguments

    def run(self, question: str) -> AgentResult:
        """Run the agent on a question.

        Args:
            question: The user's question or task.

        Returns:
            AgentResult with the final answer and execution trace.
        """
        state = question
        steps: list[AgentStep] = []

        for step_num in range(1, self.max_steps + 1):
            # Think about what to do.
            thought = f"Step {step_num}: Considering how to address: {state[:200]}"

            # Select a tool.
            tool_name = self._select_tool(state)

            if tool_name is None:
                # No tool applies; produce final answer.
                final_answer = self._produce_final_answer(state, steps)
                return AgentResult(
                    success=True,
                    final_answer=final_answer,
                    steps=steps,
                    total_steps=step_num - 1,
                )

            tool_def = self.registry.get(tool_name)
            assert tool_def is not None

            # Extract arguments.
            arguments = self._extract_arguments(tool_def, state)

            # Execute the tool.
            tool_result = self.executor.execute(tool_name, arguments)

            # Form observation.
            if tool_result.success:
                observation = f"Tool {tool_name} returned: {tool_result.output}"
            else:
                observation = f"Tool {tool_name} failed: {tool_result.error}"

            step = AgentStep(
                step_number=step_num,
                thought=thought,
                tool_name=tool_name,
                tool_arguments=arguments,
                tool_result=tool_result,
                observation=observation,
            )
            steps.append(step)

            # Update state with observation.
            state = f"{state}\nObservation: {observation}"

        # Step limit reached.
        final_answer = self._produce_final_answer(state, steps)
        return AgentResult(
            success=False,
            final_answer=final_answer,
            steps=steps,
            total_steps=self.max_steps,
        )

    def _produce_final_answer(self, state: str, steps: list[AgentStep]) -> str:
        """Produce a final answer from the accumulated state and steps."""
        if not steps:
            return f"No tools were applicable. Based on the question: {state[:200]}"

        # Summarize the observations.
        observations = [
            s.observation for s in steps if s.tool_result and s.tool_result.success
        ]
        if observations:
            return "Based on the available information and tool results: " + " ".join(
                observations
            )
        return "No useful information was obtained from tool execution."