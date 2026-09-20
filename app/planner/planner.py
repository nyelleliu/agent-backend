from typing import Any


class Planner:
    def __init__(self, client):
        self.client = client

    def create_plan(self, user_request: str) -> list[str]:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a task planner. "
                        "Break the user's request into a small number of "
                        "clear, executable steps. "
                        "Return one step per line and nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": user_request,
                },
            ],
        )

        content = response.choices[0].message.content or ""

        return [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]
