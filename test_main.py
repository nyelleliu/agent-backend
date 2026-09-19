import uuid
from fastapi.testclient import TestClient
from main import split_text, is_greeting, app
from app.tools.calculator import safe_calculate

client = TestClient(app)

def test_calculate_basic():
    assert safe_calculate("2 + 3") == 5

def test_calculate_division_by_zero():
    try:
        safe_calculate("2 / 0")
        assert False
    except ZeroDivisionError:
        assert True

def test_calculate_invalid_expression():
    try:
        safe_calculate("abc")
        assert False
    except ValueError:
        assert True

def test_split_text_basic():
    assert split_text("abcdefg", chunk_size=3) == ["abc", "def", "g"]

def test_split_text_empty():
    assert split_text("", chunk_size=3) == []

def test_register_new_user():
    unique_username = f"test_user_{uuid.uuid4().hex[:8]}"
    response = client.post("/register", json={
        "username": unique_username,
        "password": "testpass123"
    })
    assert response.status_code == 200

def test_is_greeting_true():
    assert is_greeting("\u4f60\u597d") == True

def test_is_greeting_false():
    assert is_greeting("\u4e00\u7ebf\u57ce\u5e02\u4f4f\u5bbf\u8d39\u6807\u51c6\u662f\u591a\u5c11") == False
