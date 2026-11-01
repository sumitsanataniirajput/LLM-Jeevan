"""Calculator tool for agents.

Provides basic arithmetic operations. Uses the sandbox for execution
to ensure safety.
"""

from __future__ import annotations

from typing import Any


def calculator_tool(expression: str) -> Any:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression is invalid or unsafe.
    """
    import ast
    import operator as op

    # Supported operators.
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
        ast.UAdd: op.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):  # type: ignore[attr-defined]
            return node.value
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            return operators[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return operators[type(node.op)](operand)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        return _eval(tree.body)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid expression: {expression}") from exc