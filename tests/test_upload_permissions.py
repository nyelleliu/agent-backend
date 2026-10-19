import pytest
from fastapi import HTTPException
from types import SimpleNamespace

import main


def make_user(user_id, role):
    return SimpleNamespace(id=user_id, role=role)


def test_employee_cannot_upload_admin_document():
    data = main.DocumentUpload(
        text="secret document",
        permission="admin",
    )

    current_user = make_user(1, "employee")

    with pytest.raises(HTTPException) as exc_info:
        main.upload_doc(data, current_user)

    assert exc_info.value.status_code == 403


def test_employee_can_upload_employee_document(monkeypatch):
    data = main.DocumentUpload(
        text="employee document",
        permission="employee",
    )

    current_user = make_user(1, "employee")

    called = {}

    def fake_add_document(text, permission, title=None, uploaded_by=None):
        called["text"] = text
        called["permission"] = permission
        called["title"] = title
        called["uploaded_by"] = uploaded_by

    monkeypatch.setattr(main, "add_document", fake_add_document)

    result = main.upload_doc(data, current_user)

    assert result == {"message": "Document stored"}
    assert called["text"] == "employee document"
    assert called["permission"] == "employee"
    assert called["uploaded_by"] == 1


def test_finance_can_upload_finance_document(monkeypatch):
    data = main.DocumentUpload(
        text="finance document",
        permission="finance",
    )

    current_user = make_user(2, "finance")

    called = {}

    def fake_add_document(text, permission, title=None, uploaded_by=None):
        called["text"] = text
        called["permission"] = permission
        called["title"] = title
        called["uploaded_by"] = uploaded_by

    monkeypatch.setattr(main, "add_document", fake_add_document)

    result = main.upload_doc(data, current_user)

    assert result == {"message": "Document stored"}
    assert called["text"] == "finance document"
    assert called["permission"] == "finance"


def test_admin_can_upload_admin_document(monkeypatch):
    data = main.DocumentUpload(
        text="admin document",
        permission="admin",
    )

    current_user = make_user(3, "admin")

    called = {}

    def fake_add_document(text, permission, title=None, uploaded_by=None):
        called["text"] = text
        called["permission"] = permission
        called["title"] = title
        called["uploaded_by"] = uploaded_by

    monkeypatch.setattr(main, "add_document", fake_add_document)

    result = main.upload_doc(data, current_user)

    assert result == {"message": "Document stored"}
    assert called["text"] == "admin document"
    assert called["permission"] == "admin"

def test_invalid_document_permission_is_rejected():
    with pytest.raises(Exception):
        main.DocumentUpload(
            text="invalid document",
            permission="invalid",
        )
