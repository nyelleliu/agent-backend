from __future__ import annotations

import datetime
from typing import Any

from app.tools.registry import Tool


class CurrentTimeTool(Tool):
    name = "get_current_time"
    description = "Get the current date, time and day of week"
    parameters = {"type": "object", "properties": {}}

    def execute(self, arguments: dict[str, Any]) -> str:
        now = datetime.datetime.now()
        weekday_map = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        return f"Now is {now.strftime('%Y-%m-%d %H:%M:%S')}, {weekday_map[now.weekday()]}"
