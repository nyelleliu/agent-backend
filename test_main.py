import uuid
from fastapi.testclient import TestClient
from main import calculate, split_text, is_greeting, app

client = TestClient(app)

def test_calculate_basic():
    result = calculate("2 + 3")
    assert result == 5

def test_calculate_division_by_zero():
    result = calculate("2 / 0")
    assert result == "Error: division by zero"

def test_calculate_invalid_expression():
    result = calculate("abc")
    assert result.startswith("Error:")

def test_split_text_basic():
    result = split_text("abcdefg", chunk_size=3)
    assert result == ["abc", "def", "g"]

def test_split_text_empty():
    result = split_text("", chunk_size=3)
    assert result == []

def test_register_new_user():
    unique_username = f"test_user_{uuid.uuid4().hex[:8]}"
    response = client.post("/register", json={
        "username": unique_username,
        "password": "testpass123"
    })
    assert response.status_code == 200


def test_is_greeting_true():
    assert is_greeting("你好") == True

def test_is_greeting_false():
    assert is_greeting("一线城市住宿费标准是多少") == False
