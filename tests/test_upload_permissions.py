import pytest
from fastapi import HTTPException

import main


def test_employee_cannot_upload_admin_document():
    data = main.DocumentUpload(
        text="secret document",
        permission="admin",
    )

    current_user = {
        "user_id": 1,
        "role": "employee",
    }

    with pytest.raises(HTTPException) as exc_info:
        main.upload_doc(data, current_user)

    assert exc_info.value.status_code == 403


def test_employee_can_upload_employee_document(monkeypatch):
    data = main.DocumentUpload(
        text="employee document",
        permission="employee",
    )

    current_user = {
        "user_id": 1,
        "role": "employee",
    }

    called = {}

    def fake_add_document(text, permission):
        called["text"] = text
        called["permission"] = permission

    monkeypatch.setattr(main, "add_document", fake_add_document)

    result = main.upload_doc(data, current_user)

    assert result == {"message": "Document stored"}
    assert called["text"] == "employee document"
    assert called["permission"] == "employee"


def test_finance_can_upload_finance_document(monkeypatch):
    data = main.DocumentUpload(
        text="finance document",
        permission="finance",
    )

    current_user = {
        "user_id": 2,
        "role": "finance",
    }

    called = {}

    def fake_add_document(text, permission):
        called["text"] = text
        called["permission"] = permission

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

    current_user = {
        "user_id": 3,
        "role": "admin",
    }

    called = {}

    def fake_add_document(text, permission):
        called["text"] = text
        called["permission"] = permission

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
