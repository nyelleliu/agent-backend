from app.skills.base import Skill


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
            skill.openai_schema()
            for skill in self._skills.values()
        ]

    def execute(self, name: str, arguments: dict):
        skill = self.get(name)

        if skill is None:
            return "unknown skill"

        try:
            return skill.run(arguments)
        except Exception as exc:
            return f"Error: {exc}"
