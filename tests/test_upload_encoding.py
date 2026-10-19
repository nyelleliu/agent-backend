"""HTTP-level regression: /upload_doc must preserve Chinese text.

Root cause this locks in: Windows PowerShell 5.1's Invoke-RestMethod
silently replaces CJK characters with literal '?' unless -ContentType
includes charset=utf-8. This test feeds the server the exact raw UTF-8
bytes a correctly-configured client sends and asserts the application
receives the original Chinese unchanged.

Database isolation: /register and /login still run their real
endpoint code, but the ``get_db`` dependency is answered by an
in-memory stub so no test row is ever written to the developer's
agent_db via SessionLocal.
"""

import json
import uuid

from fastapi.testclient import TestClient


CHINESE_TITLE = "RAG API 接口测试"
CHINESE_TEXT = "接口测试：差旅报销标准是一线城市每天500元。"


class _StubQuery:
    """Chainable query over the rows this test stored in memory.

    Filter conditions are SQLAlchemy expressions; the bucket holds
    only this test's own rows, so no narrowing is needed here.
    """

    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, *conditions, **kwargs):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return list(self.rows)


class _StubDB:
    """In-memory stand-in for the SQLAlchemy session.

    Writes stay here instead of the developer's agent_db. ``add``
    mirrors what an INSERT would persist (autoincrement id and the
    ``role`` column default), so tokens built from these rows look
    exactly like ones a real database would return. Unknown models
    raise so a surprise query fails loudly.
    """

    def __init__(self):
        self._rows = {}

    def query(self, model):
        name = getattr(model, "__name__", None)

        if name != "User":
            raise ValueError(f"Unexpected model in this test: {model}")

        return _StubQuery(self._rows.setdefault(name, []))

    def add(self, obj):
        name = type(obj).__name__

        if name != "User":
            raise ValueError(f"Unexpected model in this test: {type(obj)}")

        rows = self._rows.setdefault(name, [])

        if getattr(obj, "id", None) is None:
            obj.id = len(rows) + 1

        if getattr(obj, "role", None) is None:
            obj.role = "employee"

        rows.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


def register_and_login(client):
    username = f"encoding_{uuid.uuid4().hex[:8]}"
    password = "encodingpass123"

    registered = client.post(
        "/register",
        json={"username": username, "password": password},
    )
    assert registered.status_code == 200

    logged_in = client.post(
        "/login",
        json={"username": username, "password": password},
    )
    assert logged_in.status_code == 200

    return logged_in.json()["access_token"]


def test_upload_doc_preserves_chinese_from_utf8_request_bytes(monkeypatch):
    import main as main_module

    store = _StubDB()

    # Precisely scoped override: any pre-existing get_db override is
    # restored (not cleared) when this test finishes.
    overrides = main_module.app.dependency_overrides
    missing = object()
    previous_get_db = overrides.get(main_module.get_db, missing)
    overrides[main_module.get_db] = lambda: store

    try:
        client = TestClient(
            main_module.app,
            client=("198.51.100.20", 40000),
        )

        captured = {}

        def fake_add_document(text, permission, title=None, uploaded_by=None):
            captured["text"] = text
            captured["permission"] = permission
            captured["title"] = title
            captured["uploaded_by"] = uploaded_by

        monkeypatch.setattr(main_module, "add_document", fake_add_document)

        token = register_and_login(client)

        payload = {
            "text": CHINESE_TEXT,
            "permission": "employee",
            "title": CHINESE_TITLE,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        response = client.post(
            "/upload_doc",
            content=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )

        assert response.status_code == 200
        assert captured["text"] == CHINESE_TEXT
        assert captured["title"] == CHINESE_TITLE
        assert "?" not in captured["title"]
        assert "?" not in captured["text"]
    finally:
        if previous_get_db is missing:
            overrides.pop(main_module.get_db, None)
        else:
            overrides[main_module.get_db] = previous_get_db
