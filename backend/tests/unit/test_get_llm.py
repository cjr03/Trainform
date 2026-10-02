import pytest
from fastapi import HTTPException

from app import llm
from app.models import User


def test_uses_users_decrypted_key(monkeypatch):
    monkeypatch.setattr(llm, "decrypt_api_key", lambda c: f"plain-{c}")
    bound = llm.get_llm(User(anthropic_key_encrypted="cipher"))
    assert bound.func is llm.call_claude
    assert bound.args == ("plain-cipher",)


def test_falls_back_to_server_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    assert llm.get_llm(User(anthropic_key_encrypted=None)).args == ("server-key",)


def test_no_key_is_400(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(HTTPException) as e:
        llm.get_llm(User(anthropic_key_encrypted=None))
    assert e.value.status_code == 400
