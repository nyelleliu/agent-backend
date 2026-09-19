import unittest

from app.skills.base import Skill
from app.skills.registry import SkillRegistry


class FakeSkill(Skill):
    name = "fake_skill"
    description = "A fake skill for testing"

    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
            }
        },
        "required": ["text"],
    }

    def run(self, arguments):
        return f"processed: {arguments['text']}"


class SkillRegistryTests(unittest.TestCase):

    def setUp(self):
        self.registry = SkillRegistry()
        self.registry.register(FakeSkill())

    def test_get_skill(self):
        skill = self.registry.get("fake_skill")
        self.assertIsNotNone(skill)

    def test_execute_skill(self):
        result = self.registry.execute(
            "fake_skill",
            {"text": "hello"},
        )
        self.assertEqual(result, "processed: hello")

    def test_unknown_skill(self):
        result = self.registry.execute(
            "missing_skill",
            {},
        )
        self.assertEqual(result, "unknown skill")

    def test_skill_schema(self):
        schemas = self.registry.schemas()

        self.assertEqual(len(schemas), 1)
        self.assertEqual(
            schemas[0]["function"]["name"],
            "fake_skill",
        )


if __name__ == "__main__":
    unittest.main()
