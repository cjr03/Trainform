import time

import pytest

from app import auth


def test_hash_and_verify_password():
    h = auth.hash_password("correct horse")
    assert h.startswith("scrypt$") and "correct horse" not in h
    assert auth.verify_password("correct horse", h)
    assert not auth.verify_password("wrong horse", h)
    assert auth.hash_password("correct horse") != h  # random salt


@pytest.mark.parametrize("bad", ["", "not-a-hash", "scrypt$zz$zz", "bcrypt$00$00", "scrypt$00"])
def test_verify_malformed_hash_is_false(bad):
    assert not auth.verify_password("anything", bad)


def test_token_roundtrip():
    assert auth.verify_token(auth.create_token(42, 3)) == (42, 3)


@pytest.mark.parametrize("bad", ["", "garbage", "1.2", "x.y.z", "1.99999999999.sig", "1.0.99999999999.sig"])
def test_malformed_token_is_none(bad):
    assert auth.verify_token(bad) is None


def test_tampered_token_is_none():
    uid, version, exp, sig = auth.create_token(1, 0).split(".")
    assert auth.verify_token(f"2.{version}.{exp}.{sig}") is None
    assert auth.verify_token(f"{uid}.1.{exp}.{sig}") is None
    assert auth.verify_token(f"{uid}.{version}.{int(exp) + 1}.{sig}") is None


def test_expired_token_is_none(monkeypatch):
    real = time.time()
    monkeypatch.setattr(time, "time", lambda: real - 8 * 24 * 3600)
    token = auth.create_token(1, 0)
    monkeypatch.setattr(time, "time", lambda: real)
    assert auth.verify_token(token) is None


def test_token_from_other_secret_is_none(monkeypatch):
    token = auth.create_token(1, 0)
    monkeypatch.setenv("TRAINFORM_SECRET", "another-secret-0123456789abcdef0123")
    assert auth.verify_token(token) is None


def test_encrypt_roundtrip():
    c = auth.encrypt_api_key("sk-ant-123")
    assert "sk-ant-123" not in c
    assert auth.decrypt_api_key(c) == "sk-ant-123"


def test_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("TRAINFORM_SECRET")
    for call in (lambda: auth.create_token(1, 0), lambda: auth.encrypt_api_key("k")):
        with pytest.raises(RuntimeError, match="TRAINFORM_SECRET"):
            call()


def test_weak_secret_raises(monkeypatch):
    monkeypatch.setenv("TRAINFORM_SECRET", "x" * 31)
    with pytest.raises(RuntimeError, match="at least 32"):
        auth._secret()
    monkeypatch.setenv("TRAINFORM_SECRET", "x" * 32)
    assert auth._secret() == b"x" * 32
