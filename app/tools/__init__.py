from app.tools.calculator import CalculatorTool
from app.tools.datetime import CurrentTimeTool
from app.tools.registry import ToolRegistry
from app.tools.travel import TravelBudgetTool


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CurrentTimeTool())
    registry.register(CalculatorTool())
    registry.register(TravelBudgetTool())
    return registry


tool_registry = create_default_registry()
