from __future__ import annotations

import ast
import operator
from typing import Any

from app.tools.registry import Tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> int | float:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty math string")

    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    raise ValueError("unsupported expression")


class CalculatorTool(Tool):
    name = "calculate"
    description = "Calculate a math expression, e.g. addition, subtraction, multiplication, division"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to calculate, e.g. 23 * 47",
            }
        },
        "required": ["expression"],
    }

    def execute(self, arguments: dict[str, Any]) -> int | float | str:
        try:
            return safe_calculate(arguments["expression"])
        except ZeroDivisionError:
            return "Error: division by zero"
        except Exception as exc:
            return f"Error: {exc}"
