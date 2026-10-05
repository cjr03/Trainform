import pytest

from app import llm as llm_module, local_model
from app.llm import LLMError, get_llm
from app.main import app


class FakeLLM:
    def __init__(self, output=None, error=None):
        self.output, self.error, self.calls = output, error, []

    def __call__(self, system, message, schema=None):
        self.calls.append((system, message, schema))
        if self.error:
            raise self.error
        return self.output


def use_llm(fake):
    app.dependency_overrides[get_llm] = lambda: fake
    return fake


def test_prompt_route(authed_client):
    fake = use_llm(FakeLLM({"prompt": "You are Acme support."}))
    r = authed_client.post("/api/agents/prompt", json={"description": "Acme support bot"})
    assert r.status_code == 200
    assert r.json() == {"prompt": "You are Acme support."}
    assert "Acme support bot" in fake.calls[0][1]


def test_rubric_route(authed_client):
    fake = use_llm(FakeLLM({"criteria": ["Greets the user.", " "]}))
    r = authed_client.post("/api/agents/rubric", json={"agent_prompt": "You are Acme support."})
    assert r.status_code == 200
    assert r.json() == {"criteria": ["Greets the user."]}
    assert "You are Acme support." in fake.calls[0][1]


@pytest.mark.parametrize("path, body", [("/api/agents/prompt", {"description": "x"}),
                                        ("/api/agents/rubric", {"agent_prompt": "x"})])
def test_llm_error_is_502(authed_client, path, body):
    use_llm(FakeLLM(error=LLMError("Claude rejected the API key")))
    r = authed_client.post(path, json=body)
    assert r.status_code == 502
    assert r.json() == {"detail": "Claude rejected the API key"}


def test_no_key_is_400(authed_client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = authed_client.post("/api/agents/prompt", json={"description": "x"})
    assert r.status_code == 400
    assert "Claude API key" in r.json()["detail"]


@pytest.mark.parametrize("path, body", [
    ("/api/agents/prompt", {}),
    ("/api/agents/prompt", {"description": ""}),
    ("/api/agents/prompt", {"description": "x" * 4001}),
    ("/api/agents/prompt", {"description": 5}),
    ("/api/agents/rubric", {}),
    ("/api/agents/rubric", {"agent_prompt": ""}),
    ("/api/agents/rubric", {"agent_prompt": "x" * 20001}),
])
def test_invalid_bodies_are_422(authed_client, path, body):
    fake = use_llm(FakeLLM({"prompt": "p", "criteria": ["c"]}))
    assert authed_client.post(path, json=body).status_code == 422
    assert fake.calls == []


def test_limits_are_inclusive(authed_client):
    use_llm(FakeLLM({"prompt": "p", "criteria": ["c"]}))
    assert authed_client.post("/api/agents/prompt", json={"description": "x" * 4000}).status_code == 200
    assert authed_client.post("/api/agents/rubric", json={"agent_prompt": "x" * 20000}).status_code == 200


@pytest.mark.parametrize("path, body", [("/api/agents/prompt", {"description": "x"}),
                                        ("/api/agents/rubric", {"agent_prompt": "x"})])
def test_requires_auth(client, path, body):
    assert client.post(path, json=body).status_code == 401


class StubModel:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, system, message, digits=False):
        return self.reply


@pytest.fixture
def no_claude(monkeypatch, tiny_model_dir):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    loaded = []
    monkeypatch.setattr(llm_module, "_load", lambda name: loaded.append(name) or StubModel(f"{name} reply\nsecond"))
    return loaded


def test_without_claude_the_routes_use_the_chosen_local_model(authed_client, no_claude):
    r = authed_client.post("/api/agents/prompt", json={"description": "Acme support bot", "model": "tiny"})
    assert (r.status_code, r.json()) == (200, {"prompt": "tiny reply\nsecond"})
    r = authed_client.post("/api/agents/rubric", json={"agent_prompt": "You are Acme support.", "model": "tiny"})
    assert (r.status_code, r.json()) == (200, {"criteria": ["tiny reply", "second"]})
    assert no_claude == ["tiny", "tiny"]


@pytest.mark.parametrize("model", [None, "missing", "../tiny"])
def test_without_claude_a_listed_local_model_is_required(authed_client, no_claude, model):
    r = authed_client.post("/api/agents/prompt", json={"description": "x", "model": model})
    assert r.status_code == 400
    assert r.json()["detail"] == "Link your Claude API key on the Account page, or pick a local model first."
    assert no_claude == []


@pytest.mark.parametrize("path, body", [("/api/agents/prompt", {"description": "x"}),
                                        ("/api/agents/rubric", {"agent_prompt": "x"})])
def test_the_chosen_helper_is_used(authed_client, no_claude, path, body):
    fake = use_llm(FakeLLM({"prompt": "p", "criteria": ["c"]}))
    assert authed_client.post(path, json=body | {"helper": "claude", "model": "tiny"}).status_code == 200
    assert len(fake.calls) == 1 and no_claude == []
    # "local" uses the local model even though Claude is linked.
    assert authed_client.post(path, json=body | {"helper": "local", "model": "tiny"}).status_code == 200
    assert len(fake.calls) == 1 and no_claude == ["tiny"]
    r = authed_client.post(path, json=body | {"helper": "local"})
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page, or pick a "
                                                         "local model first."})
    assert authed_client.post(path, json=body | {"helper": "gpt", "model": "tiny"}).status_code == 422


@pytest.mark.parametrize("path, body", [("/api/agents/prompt", {"description": "x"}),
                                        ("/api/agents/rubric", {"agent_prompt": "x"})])
def test_claude_without_a_key_is_400(authed_client, no_claude, path, body):
    r = authed_client.post(path, json=body | {"helper": "claude", "model": "tiny"})
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page first."})
    assert no_claude == []


def test_a_local_model_that_fails_to_load_is_400(authed_client, no_claude, monkeypatch):
    def broken(key):
        raise OSError("bad safetensors header")
    monkeypatch.setattr(llm_module, "_load", broken)
    r = authed_client.post("/api/agents/prompt", json={"description": "x", "model": "tiny"})
    assert (r.status_code, r.json()) == (400, {"detail": "The local model tiny could not be loaded."})


def test_a_busy_local_model_is_503(authed_client, no_claude, monkeypatch):
    monkeypatch.setattr(llm_module, "LOCAL_WAIT", 0.01)
    llm_module._ROUTE_MODEL.acquire()
    try:
        r = authed_client.post("/api/agents/prompt", json={"description": "x", "model": "tiny"})
    finally:
        llm_module._ROUTE_MODEL.release()
    assert (r.status_code, r.json()) == (503, {"detail": "The local model is busy; try again shortly."})


def test_a_replaced_local_model_is_reloaded(tmp_path, monkeypatch):
    # Deleting a model and adding another under the same name must not serve the old weights.
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(llm_module, "LocalModel", lambda name: object())
    (tmp_path / "foo").mkdir()
    first = llm_module._load("foo")
    assert llm_module._load("foo") is first
    (tmp_path / "foo").rmdir()
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "config.json").write_text("{}")
    assert llm_module._load("foo") is not first
