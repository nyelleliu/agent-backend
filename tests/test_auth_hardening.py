import bcrypt
import pytest

import main
from pydantic import ValidationError


def test_verify_password_accepts_valid_hash():
    hashed = bcrypt.hashpw(b"correct horse", bcrypt.gensalt()).decode()

    assert main.verify_password("correct horse", hashed) is True
    assert main.verify_password("wrong horse", hashed) is False


def test_verify_password_rejects_corrupt_hash_without_crashing():
    assert main.verify_password("anything", "") is False
    assert main.verify_password("anything", "not-a-bcrypt-hash") is False


def test_register_rejects_short_username():
    with pytest.raises(ValidationError):
        main.UserRegister(username="ab", password="longenough1")


def test_register_rejects_short_password():
    with pytest.raises(ValidationError):
        main.UserRegister(username="validuser", password="short")


def test_register_rejects_password_over_bcrypt_limit():
    with pytest.raises(ValidationError):
        main.UserRegister(
            username="validuser",
            password="a" * 80,
        )


def test_register_rejects_whitespace_username():
    with pytest.raises(ValidationError):
        main.UserRegister(username="has space", password="longenough1")


def test_register_accepts_valid_input():
    data = main.UserRegister(
        username="valid_user",
        password="longenough1",
    )

    assert data.username == "valid_user"
    assert main.hash_password(data.password) != data.password
