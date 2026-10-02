import json
import os
from functools import partial

import anthropic
from fastapi import Depends, HTTPException

from app.auth import current_user, decrypt_api_key
from app.models import User

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
        raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API") from e
    if response.stop_reason == "refusal":
        raise LLMError("Claude declined this request")
    text = "".join(block.text for block in response.content if block.type == "text")
    if not schema:
        return text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Claude returned malformed JSON (stop reason: {response.stop_reason})") from e


def get_llm(user: User = Depends(current_user)):
    """FastAPI dependency: an `llm(system, message, schema=None)` callable bound to the user's Claude key."""
    if user.anthropic_key_encrypted:
        key = decrypt_api_key(user.anthropic_key_encrypted)
    else:
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(400, "Link your Claude API key on the Account page first.")
    return partial(call_claude, key)
