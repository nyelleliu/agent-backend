from app.tools.calculator import CalculatorTool
from app.tools.datetime import CurrentTimeTool
from app.tools.registry import ToolRegistry


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())
    return registry


tool_registry = create_default_registry()
