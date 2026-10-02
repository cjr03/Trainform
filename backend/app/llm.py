import os
from functools import partial

from fastapi import Depends, HTTPException

from app.auth import current_user, decrypt_api_key
from app.models import User

MODEL = "claude-opus-5-5"


def call_claude(api_key: str, system: str, message: str, schema: dict | None = None):
    """One Claude call. Returns the text, or the parsed JSON object when `schema` (a JSON Schema) is given."""
    raise NotImplementedError


def get_llm(user: User = Depends(current_user)):
    """FastAPI dependency: an `llm(system, message, schema=None)` callable bound to the user's Claude key."""
    if user.anthropic_key_encrypted:
        key = decrypt_api_key(user.anthropic_key_encrypted)
    else:
        key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(400, "Link your Claude API key on the Account page first.")
    return partial(call_claude, key)
