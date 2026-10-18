import logging

from app.core.errors import format_tool_error, log_tool_error
from app.skills.base import Skill


logger = logging.getLogger(__name__)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
            }
            for skill in self._skills.values()
        ]

    def schemas(self) -> list[dict]:
        return [
            skill.openai_schema() for skill in self._skills.values()
        ]

    def execute(self, name: str, arguments: dict):
        skill = self.get(name)

        if skill is None:
            return "unknown skill"

        try:
            return skill.run(arguments)
        except Exception as exc:
            # Sanitized for the model; full detail goes to the logs.
            log_tool_error(logger, exc, tool_name=name)

            return format_tool_error(exc, tool_name=name)
