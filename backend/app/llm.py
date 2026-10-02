import json
import logging
import os
from functools import cache, partial

import anthropic
from cryptography.fernet import InvalidToken
from fastapi import Depends, HTTPException

from app import fake
from app.auth import current_user, decrypt_api_key
from app.models import User

log = logging.getLogger(__name__)
MODEL = "claude-opus-5-5"


class LLMError(Exception):
    """A Claude call failed; the message is safe to show the user."""


def call_claude(api_key: str, system: str, message: str, schema: dict | None = None):
    """One Claude call. Returns the text, or the parsed JSON object when `schema` (a JSON Schema) is given."""
    output_config = {"effort": "medium"}
    if schema:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    try:
        response = anthropic.Anthropic(api_key=api_key).beta.messages.create(
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
        raise LLMError("Claude returned malformed JSON") from e


@cache
def _warn_fake_mode():
    log.warning("TRAINFORM_FAKE_LLM=1: Claude calls return canned fake answers and trainings use a fake model")


def get_llm(user: User = Depends(current_user)):
    """FastAPI dependency: an `llm(system, message, schema=None)` callable bound to the user's Claude key."""
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
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(400, "Link your Claude API key on the Account page first.")
    return partial(call_claude, key)
