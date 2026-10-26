from __future__ import annotations

import json
import math
from typing import Any

from app.tools.registry import Tool


# MVP policy coverage is intentionally explicit. These values are simulated
# acceptance data and must not be presented as a real company policy.
POLICY_CITIES = frozenset({"上海", "北京", "深圳", "广州"})
POLICY = {
    "employee": {"lodging": 350, "meal": 100, "transport": 150},
    "senior": {"lodging": 500, "meal": 150, "transport": 250},
    "manager": {"lodging": 700, "meal": 200, "transport": 400},
}
MAX_EXPENSE_AMOUNT = 1_000_000


class TravelBudgetTool(Tool):
    name = "calculate_travel_budget"
    description = (
        "Calculate a domestic business travel budget using simulated MVP data. "
        "City must use one of these canonical Chinese names: 上海, 北京, 深圳, 广州. "
        "Use only after city, positive trip days, and job level are known."
    )
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Canonical Chinese city name: 上海, 北京, 深圳, or 广州.",
            },
            "days": {"type": "integer", "description": "Number of travel days"},
            "job_level": {
                "type": "string",
                "enum": ["employee", "senior", "manager"],
            },
            "lodging_per_day": {"type": "number"},
            "meal_per_day": {"type": "number"},
            "transport_total": {"type": "number"},
        },
        "required": ["city", "days", "job_level"],
    }

    def execute(self, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        if any(not isinstance(key, str) for key in arguments):
            raise ValueError("argument names must be strings")

        allowed = set(self.parameters["properties"])
        unknown = set(arguments) - allowed
        if unknown:
            raise ValueError(f"unexpected arguments: {', '.join(sorted(unknown))}")

        missing = [key for key in self.parameters["required"] if key not in arguments]
        if missing:
            raise ValueError(f"missing required arguments: {', '.join(missing)}")

        city = arguments["city"]
        if not isinstance(city, str):
            raise ValueError("city must be a string")
        city = city.strip()
        if not city:
            raise ValueError("city must be non-empty")
        if city not in POLICY_CITIES:
            raise ValueError(f"unsupported city: {city}")

        days = arguments["days"]
        if type(days) is not int or not 1 <= days <= 90:
            raise ValueError("days must be an integer between 1 and 90")

        level = arguments["job_level"]
        if not isinstance(level, str) or level not in POLICY:
            raise ValueError("job_level must be employee, senior, or manager")

        limits = POLICY[level]
        actual = {
            "lodging_per_day": arguments.get("lodging_per_day", limits["lodging"]),
            "meal_per_day": arguments.get("meal_per_day", limits["meal"]),
            "transport_total": arguments.get("transport_total", limits["transport"]),
        }
        for name, value in actual.items():
            if type(value) not in (int, float):
                raise ValueError(f"{name} must be a finite non-negative number")
            if type(value) is int and value > MAX_EXPENSE_AMOUNT:
                raise ValueError(
                    f"{name} must not exceed {MAX_EXPENSE_AMOUNT} per entry"
                )
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative number")
            if value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
            if value > MAX_EXPENSE_AMOUNT:
                raise ValueError(
                    f"{name} must not exceed {MAX_EXPENSE_AMOUNT} per entry"
                )

        limits_total = days * (limits["lodging"] + limits["meal"]) + limits["transport"]
        actual_total = (
            days * (actual["lodging_per_day"] + actual["meal_per_day"])
            + actual["transport_total"]
        )
        if not math.isfinite(actual_total) or actual_total > MAX_EXPENSE_AMOUNT * 180:
            raise ValueError("actual_total exceeds the supported amount")

        over_limit = []
        if actual["lodging_per_day"] > limits["lodging"]:
            over_limit.append("lodging_per_day")
        if actual["meal_per_day"] > limits["meal"]:
            over_limit.append("meal_per_day")
        if actual["transport_total"] > limits["transport"]:
            over_limit.append("transport_total")

        return json.dumps({
            "ok": True,
            "city": city,
            "days": days,
            "job_level": level,
            "policy_limits": limits,
            "actual_expenses": actual,
            "policy_budget": limits_total,
            "actual_total": actual_total,
            "over_limit_items": over_limit,
            "assumptions": ["国内出差", "住宿和餐补按天计算", "交通为本次行程总额"],
            "citation": "差旅预算 MVP 模拟数据（非真实企业制度）",
            "policy_data_type": "simulated_mvp",
        }, ensure_ascii=False, allow_nan=False)
