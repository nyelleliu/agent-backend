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

def test_employee_can_upload_public_document():
    checker = PermissionChecker()

    assert checker.can_upload_document(
        "employee",
        "public",
    )


def test_employee_can_upload_employee_document():
    checker = PermissionChecker()

    assert checker.can_upload_document(
        "employee",
        "employee",
    )


def test_employee_cannot_upload_finance_document():
    checker = PermissionChecker()

    assert not checker.can_upload_document(
        "employee",
        "finance",
    )


def test_employee_cannot_upload_admin_document():
    checker = PermissionChecker()

    assert not checker.can_upload_document(
        "employee",
        "admin",
    )


def test_finance_can_upload_finance_document():
    checker = PermissionChecker()

    assert checker.can_upload_document(
        "finance",
        "finance",
    )


def test_admin_can_upload_admin_document():
    checker = PermissionChecker()

    assert checker.can_upload_document(
        "admin",
        "admin",
    )
