"""Hugging Face Hub access, local model storage, generation, and LoRA SFT."""

import os
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["models"])

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "data" / "models"))


def list_hub_models(search: str | None = None, sort: str = "downloads", limit: int = 20) -> list[dict]:
    raise NotImplementedError


def download_model(repo_id: str) -> str:
    """Download a Hub model into MODELS_DIR; returns its local model name."""
    raise NotImplementedError


def save_upload(fileobj, name: str) -> str:
    """Extract an uploaded zipped model directory into MODELS_DIR; returns its local model name."""
    raise NotImplementedError


def list_local_models() -> list[dict]:
    raise NotImplementedError


class LocalModel:
    """A local causal LM wrapped with a LoRA adapter, so SFT updates stay small enough to store in Postgres."""

    def __init__(self, name: str):
        raise NotImplementedError

    def generate(self, system_prompt: str, conversation: str) -> str:
        raise NotImplementedError

    def sft(self, examples: list[dict]) -> float:
        """Fine-tune on [{"prompt", "input", "output"}]; returns the mean training loss."""
        raise NotImplementedError

    def adapter_bytes(self) -> bytes:
        """The current LoRA adapter as a zip archive."""
        raise NotImplementedError
