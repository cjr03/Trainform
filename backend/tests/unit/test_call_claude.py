from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app import llm
from app.llm import BadReply, LLMError, call_claude

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def block(type, **fields):
    return SimpleNamespace(type=type, **fields)


def reply(*content, stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


def fake_anthropic(monkeypatch, response=None, error=None):
    calls = {}

    def create(**kwargs):
        calls["kwargs"] = kwargs
        if error:
            raise error
        return response

    def client(api_key, max_retries):
        calls["api_key"], calls["max_retries"] = api_key, max_retries
        return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))

    monkeypatch.setattr(anthropic, "Anthropic", client)
    return calls


def test_text_request_shape_and_join(monkeypatch):
    calls = fake_anthropic(monkeypatch, reply(block("thinking", thinking=""), block("text", text=" Hello "),
                                              block("text", text="world. ")))
    assert call_claude("sk-1", "Be terse.", "Hi") == "Hello world."
    assert calls["api_key"] == "sk-1"
    assert calls["max_retries"] == 5  # the SDK backs off and retries 429s, 5xx and overloads; a training runs for hours
    assert calls["kwargs"] == {
        "model": llm.MODEL,
        "max_tokens": 16000,
        "system": "Be terse.",
        "messages": [{"role": "user", "content": "Hi"}],
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "output_config": {"effort": "medium"},
    }
    assert llm.MODEL == "claude-opus-5-5"


def test_schema_request_parses_json(monkeypatch):
    schema = {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"],
              "additionalProperties": False}
    calls = fake_anthropic(monkeypatch, reply(block("text", text='{"prompt": '), block("text", text='"x"}')))
    assert call_claude("k", "s", "m", schema) == {"prompt": "x"}
    assert calls["kwargs"]["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": schema}}
    assert not {"temperature", "top_p", "top_k", "thinking"} & calls["kwargs"].keys()


def test_effort_can_be_lowered(monkeypatch):
    calls = fake_anthropic(monkeypatch, reply(block("text", text="hi")))
    assert call_claude("k", "s", "m", effort="low") == "hi"
    assert calls["kwargs"]["output_config"] == {"effort": "low"}


def test_bad_json_is_a_bad_reply(monkeypatch):
    fake_anthropic(monkeypatch, reply(block("text", text='{"prompt": "x"')))
    with pytest.raises(BadReply, match="malformed JSON"):
        call_claude("k", "s", "m", {"type": "object"})


@pytest.mark.parametrize("schema", [None, {"type": "object"}])
def test_max_tokens_is_llm_error(monkeypatch, schema):
    fake_anthropic(monkeypatch, reply(block("text", text="truncated reply"), stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="cut off"):
        call_claude("k", "s", "m", schema)


def test_refusal_checked_before_content(monkeypatch):
    fake_anthropic(monkeypatch, SimpleNamespace(stop_reason="refusal", content=None))
    with pytest.raises(LLMError, match="declined"):
        call_claude("k", "s", "m")


@pytest.mark.parametrize("error, message", [
    (anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=REQUEST), body=None),
     "Claude rejected the API key"),
    (anthropic.RateLimitError("rate_limit_error", response=httpx2.Response(429, request=REQUEST), body=None),
     "rate limit"),
    (anthropic.InternalServerError("Error code: 529 - {'type': 'error'}", response=httpx2.Response(529, request=REQUEST),
                                   body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
     r"^Claude API error 529: Overloaded$"),
    (anthropic.InternalServerError("Error code: 500 - b'oops'", response=httpx2.Response(500, request=REQUEST),
                                   body="oops"),
     r"^Claude API error 500: unexpected error$"),
    (anthropic.APIConnectionError(request=REQUEST), "reach"),
])
def test_sdk_errors_become_llm_errors(monkeypatch, error, message):
    fake_anthropic(monkeypatch, error=error)
    with pytest.raises(LLMError, match=message) as e:
        call_claude("k", "s", "m")
    assert e.value.__cause__ is error
