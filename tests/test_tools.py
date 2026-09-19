import unittest

from app.tools import tool_registry
from app.tools.calculator import CalculatorTool, safe_calculate
from app.tools.datetime import CurrentTimeTool
from app.tools.registry import ToolRegistry


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(CurrentTimeTool())
        self.registry.register(CalculatorTool())

    def test_default_registry_has_existing_tools(self):
        names = {schema["function"]["name"] for schema in tool_registry.schemas()}
        self.assertEqual(names, {"get_current_time", "calculate"})

    def test_execute_by_name_without_if_elif(self):
        result = self.registry.execute("calculate", '{"expression": "23 * 47"}')
        self.assertEqual(result, 1081)

    def test_unknown_tool(self):
        self.assertEqual(self.registry.execute("missing"), "unknown tool")

    def test_invalid_json_arguments(self):
        self.assertEqual(
            self.registry.execute("calculate", "{not-json"),
            "Error: invalid JSON arguments",
        )

    def test_safe_calculate_rejects_eval_style_code(self):
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').system('echo hacked')")

    def test_division_by_zero(self):
        self.assertEqual(
            self.registry.execute("calculate", {"expression": "1 / 0"}),
            "Error: division by zero",
        )

    def test_current_time_format(self):
        result = self.registry.execute("get_current_time", "{}")
        self.assertTrue(str(result).startswith("Now is "))


if __name__ == "__main__":
    unittest.main()
