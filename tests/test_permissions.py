from app.agent.loop import AgentLoop
from app.permissions.checker import PermissionChecker
from app.skills.base import Skill


class FakeSkill(Skill):
    name = "data_analysis"
    description = "Test data analysis skill"
    permission = "data_analysis"

    def run(self, arguments):
        return "analysis result"


class FakeClient:
    pass


def test_employee_is_denied_data_analysis():
    checker = PermissionChecker()

    assert not checker.has_permission(
        "employee",
        "data_analysis",
    )


def test_finance_is_allowed_data_analysis():
    checker = PermissionChecker()

    assert checker.has_permission(
        "finance",
        "data_analysis",
    )


def test_admin_is_allowed_data_analysis():
    checker = PermissionChecker()

    assert checker.has_permission(
        "admin",
        "data_analysis",
    )

def test_unknown_role_has_no_permission():
    checker = PermissionChecker()

    assert not checker.has_permission(
        "unknown",
        "data_analysis",
    )
