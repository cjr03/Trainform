import threading

import pytest
from fastapi import HTTPException

from app import agents, fake, llm
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
        assert llm.get_llm(User(username="eve", anthropic_key_encrypted=None)) is None
    assert llm.get_llm(User(username="ada", anthropic_key_encrypted=None)).args == ("server-key",)


def test_no_key_means_no_claude(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.get_llm(User(anthropic_key_encrypted=None)) is None  # callers fall back to a local model


def test_helper_choice():
    claude = object()
    assert llm.chosen_llm(None, claude) is claude and llm.chosen_llm(None, None) is None  # Claude if linked
    assert llm.chosen_llm("claude", claude) is claude
    assert llm.chosen_llm("local", claude) is None  # None: the caller runs the helpers on a local model
    with pytest.raises(HTTPException) as e:
        llm.chosen_llm("claude", None)
    assert (e.value.status_code, e.value.detail) == (400, "Link your Claude API key on the Account page first.")


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


class StubModel:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, message, digits=False):
        self.calls.append((system, message, digits))
        return self.replies.pop(0)


def test_local_llm_plain_text():
    model = StubModel(" hi there ", "again")
    local = llm.local_llm(model)
    assert local.local is True
    assert local("sys", "msg") == " hi there "
    assert local("sys", "msg", effort="low") == "again"  # Claude's effort setting means nothing to the local model
    assert model.calls == [("sys", "msg", False)] * 2


def test_local_llm_string_field_is_the_whole_reply():
    model = StubModel("You are Acme support.")
    assert llm.local_llm(model)("sys", "msg", agents.PROMPT_SCHEMA) == {"prompt": "You are Acme support."}
    system, message, digits = model.calls[0]
    assert system.startswith("sys") and system != "sys" and (message, digits) == ("msg", False)


def test_local_llm_list_field_is_one_item_per_line():
    model = StubModel("1. Greets the user.\n- Is concise.\n\n* Never invents prices.\n2) Asks first.")
    assert llm.local_llm(model)("sys", "msg", agents.RUBRIC_SCHEMA) == {
        "criteria": ["Greets the user.", "Is concise.", "", "Never invents prices.", "Asks first."]}
    assert "one item per line" in model.calls[0][0]


@pytest.mark.parametrize("reply, score", [("7", 7), ("10", 10), ("1 0", 10), ("7 10", 7), (" 3", 3), ("0", 0)])
def test_local_llm_integer_field_reads_a_digits_only_reply(reply, score):
    model = StubModel(reply)
    assert llm.local_llm(model)("sys", "msg", agents.SCORE_SCHEMA) == {"score": score}
    assert model.calls[0][2] is True


class BrokenModel:
    def complete(self, system, message, digits=False):
        raise RuntimeError("index out of range in self")  # e.g. a prompt longer than the model's context


@pytest.mark.parametrize("model", [StubModel(""), BrokenModel()])
def test_local_model_failures_are_llm_errors(model):
    with pytest.raises(llm.LLMError, match="^The local model failed to reply$"):
        llm.local_llm(model)("sys", "msg", agents.SCORE_SCHEMA)


def test_a_busy_local_model_is_an_llm_error(monkeypatch):
    monkeypatch.setattr(llm, "LOCAL_WAIT", 0.01)
    lock = threading.Lock()
    lock.acquire()
    with pytest.raises(llm.LLMError, match="busy"):
        llm.local_llm(StubModel("x"), lock)("sys", "msg")
    lock.release()
    assert llm.local_llm(StubModel("x"), lock)("sys", "msg") == "x"
    assert lock.acquire(blocking=False)  # released after the call
