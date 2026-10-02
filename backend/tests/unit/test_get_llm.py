import pytest
from fastapi import HTTPException

from app import fake, llm
from app.auth import encrypt_api_key
from app.models import User


def test_uses_users_decrypted_key(monkeypatch):
    monkeypatch.setattr(llm, "decrypt_api_key", lambda c: f"plain-{c}")
    bound = llm.get_llm(User(anthropic_key_encrypted="cipher"))
    assert bound.func is llm.call_claude
    assert bound.args == ("plain-cipher",)


def test_server_key_only_for_allow_listed_users(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.delenv("TRAINFORM_SHARED_KEY_USERS", raising=False)
    for allowed in ("", " bob , ada "):
        monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", allowed)
        with pytest.raises(HTTPException) as e:
            llm.get_llm(User(username="eve", anthropic_key_encrypted=None))
        assert e.value.status_code == 400
    assert llm.get_llm(User(username="ada", anthropic_key_encrypted=None)).args == ("server-key",)


def test_no_key_is_400(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(HTTPException) as e:
        llm.get_llm(User(anthropic_key_encrypted=None))
    assert e.value.status_code == 400


def test_unreadable_key_after_secret_rotation_is_400(monkeypatch):
    cipher = encrypt_api_key("sk-ant-old")
    monkeypatch.setenv("TRAINFORM_SECRET", "rotated-secret-0123456789abcdef0123")
    with pytest.raises(HTTPException) as e:
        llm.get_llm(User(anthropic_key_encrypted=cipher))
    assert (e.value.status_code, e.value.detail) == (
        400, "Your linked Claude key can't be read (the server secret changed). Re-link it on the Account page.")


def test_fake_mode_needs_no_key(monkeypatch, caplog):
    monkeypatch.setenv("TRAINFORM_FAKE_LLM", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm._warn_fake_mode.cache_clear()
    assert llm.get_llm(User(anthropic_key_encrypted=None)) is fake.fake_llm
    assert llm.get_llm(User(anthropic_key_encrypted="unreadable")) is fake.fake_llm
    assert [r.levelname for r in caplog.records] == ["WARNING"]  # logged once, not per request
    assert "TRAINFORM_FAKE_LLM" in caplog.text


@pytest.mark.parametrize("value", ["0", "true", ""])
def test_fake_mode_only_when_exactly_1(monkeypatch, value):
    monkeypatch.setenv("TRAINFORM_FAKE_LLM", value)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    assert llm.get_llm(User(username="ada", anthropic_key_encrypted=None)).func is llm.call_claude
