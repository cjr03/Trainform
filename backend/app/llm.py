import json
import logging
import re
import threading
from functools import cache, lru_cache, partial
from typing import Literal

import anthropic
from cryptography.fernet import InvalidToken
from fastapi import Depends, HTTPException

from app import fake, local_model
from app.auth import current_user, decrypt_api_key, shared_key
from app.local_model import LocalModel, list_local_models
from app.models import User

log = logging.getLogger(__name__)
MODEL = "claude-opus-5-5"


class LLMError(Exception):
    """A Claude call failed; the message is safe to show the user."""


class BadReply(LLMError):
    """The model answered, but not in the shape asked for; unlike a refusal or a bad key, asking again can fix it."""


@lru_cache(maxsize=32)
def _client(api_key: str) -> anthropic.Anthropic:
    """One client per key, so its calls reuse open connections instead of each paying a new TLS handshake."""
    # Five retries with the SDK's backoff ride out the 429s, 5xx and overloads an hours-long training will meet.
    return anthropic.Anthropic(api_key=api_key, max_retries=5)


def call_claude(api_key: str, system: str, message: str, schema: dict | None = None, effort: str = "medium"):
    """One Claude call. Returns the text, or the parsed JSON object when `schema` (a JSON Schema) is given."""
    output_config = {"effort": effort}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        response = _client(api_key).beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": message}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config=output_config,
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Claude rejected the API key") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Claude rate limit reached; try again shortly") from e
    except anthropic.APIStatusError as e:
        err = e.body.get("error") if isinstance(e.body, dict) else None
        msg = err.get("message") if isinstance(err, dict) else None
        raise LLMError(f"Claude API error {e.status_code}: {msg or 'unexpected error'}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API") from e
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise LLMError("Claude's reply was cut off at the token limit")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not schema:
        return text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise BadReply("Claude returned malformed JSON") from e


@cache
def _warn_fake_mode():
    log.warning("TRAINFORM_FAKE_LLM=1: Claude calls return canned fake answers and trainings use a fake model")


def get_llm(user: User = Depends(current_user)):
    """FastAPI dependency: an `llm(system, message, schema=None, effort="medium")` callable bound to the user's Claude
    key, or None when the user has no key; callers then run the helper agents on a local model (`local_llm`)."""
    if fake.enabled():
        _warn_fake_mode()
        return fake.fake_llm
    if user.anthropic_key_encrypted:
        try:
            key = decrypt_api_key(user.anthropic_key_encrypted)
        except InvalidToken:
            raise HTTPException(400, "Your linked Claude key can't be read (the server secret changed). "
                                     "Re-link it on the Account page.") from None
    else:
        key = shared_key(user)
    return partial(call_claude, key) if key else None


Helper = Literal["claude", "local"]  # who runs the helper agents; omitted (None) means Claude when linked, else local


def chosen_llm(helper: Helper | None, llm):
    """`llm` from get_llm, or None (run the helpers on a local model) when the user chose "local"."""
    if helper == "claude" and not llm:
        raise HTTPException(400, "Link your Claude API key on the Account page first.")
    return None if helper == "local" else llm


LOCAL_WAIT = 60  # seconds an agent request waits for the shared local model before giving up
_ROUTE_MODEL = threading.Lock()  # the agent routes share one loaded model, which replies to one request at a time
_loaded = {}


def _local_reply(model, system, message, schema):
    if schema is None:
        return model.complete(system, message)
    (field, spec), = schema["properties"].items()
    if spec["type"] == "integer":
        reply = model.complete(system, message, digits=True).replace(" ", "")
        return {field: 10 if reply.startswith("10") else int(reply[:1])}
    if spec["type"] == "array":
        reply = model.complete(f"{system}\n\nWrite one item per line and nothing else.", message)
        return {field: [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line) for line in reply.splitlines()]}
    return {field: model.complete(f"{system}\n\nReply with only the text itself and nothing else.", message)}


def local_llm(model, lock=None):
    """An `llm(system, message, schema=None)` running on a LocalModel, for users who haven't linked Claude.

    Small models can't write JSON, so a schema must have a single field and the reply is mapped onto it: the whole
    reply for a string, one item per line for a list of strings, and a digits-only reply for an integer. `lock`
    serialises callers that share `model`. `effort` is Claude's and is ignored."""
    def llm(system, message, schema=None, effort=None):
        if lock and not lock.acquire(timeout=LOCAL_WAIT):
            raise LLMError("The local model is busy; try again shortly")
        try:
            return _local_reply(model, system, message, schema)
        except Exception as e:  # torch errors (e.g. a prompt past the context window) carry no user-facing detail
            log.exception("Local model failed to reply")
            raise LLMError("The local model failed to reply") from e
        finally:
            if lock:
                lock.release()

    llm.local = True
    return llm


def _load(name: str) -> LocalModel:
    """The routes' model, kept loaded for the next request. Caller holds _ROUTE_MODEL."""
    stat = (local_model.MODELS_DIR / name).stat()
    key = (name, stat.st_ino, stat.st_mtime_ns)  # a deleted and re-added model of the same name gets reloaded
    if key not in _loaded:
        _loaded.clear()  # ponytail: one model at a time, freed before the next loads
        _loaded[key] = LocalModel(name)
    return _loaded[key]


def llm_or_local(llm, user: User, model: str | None):
    """The user's Claude `llm`, else one on `model`, a local model the user can see."""
    if llm:
        return llm
    if model not in {m["name"] for m in list_local_models(user.id)}:
        raise HTTPException(400, "Link your Claude API key on the Account page, or pick a local model first.")
    if not _ROUTE_MODEL.acquire(timeout=LOCAL_WAIT):
        raise HTTPException(503, "The local model is busy; try again shortly.")
    try:
        loaded = _load(model)
    except Exception as e:
        log.exception("Could not load local model %s", model)
        raise HTTPException(400, f"The local model {model} could not be loaded.") from e
    finally:
        _ROUTE_MODEL.release()
    return local_llm(loaded, _ROUTE_MODEL)
