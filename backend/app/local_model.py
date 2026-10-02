"""Hugging Face Hub access, local model storage, generation, and LoRA SFT."""

import io
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from peft import LoraConfig, get_peft_model
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.auth import current_user

router = APIRouter(tags=["models"], dependencies=[Depends(current_user)])

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "data" / "models"))
MAX_UPLOAD_BYTES = 20 * 2**30  # uncompressed; zip-bomb guard
NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
# Configs, safetensors weights and tokenizer files only: pickle weights (*.bin, *.pt, ...) can execute code on load.
HUB_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".ckpt")


def _model_dir(name: str) -> Path:
    root = MODELS_DIR.resolve()
    if not NAME_RE.fullmatch(name) or (root / name).resolve().parent != root:
        raise ValueError(f"Invalid model name: {name!r}")
    return root / name


def list_hub_models(search: str | None = None, sort: str = "downloads", limit: int = 20) -> list[dict]:
    if sort not in ("downloads", "lastModified"):
        raise ValueError("sort must be 'downloads' or 'lastModified'")
    models = HfApi().list_models(search=search or None, pipeline_tag="text-generation", sort=sort, limit=limit,
                                 expand=["downloads", "likes", "lastModified"])
    return [{"id": m.id, "downloads": m.downloads, "likes": m.likes,
             "last_modified": m.last_modified.isoformat() if m.last_modified else None} for m in models]


def download_model(repo_id: str) -> str:
    """Download a Hub model into MODELS_DIR; returns its local model name."""
    if not REPO_RE.fullmatch(repo_id):
        raise ValueError(f"Invalid Hugging Face repo id {repo_id!r}; expected 'owner/name'")
    name = repo_id.replace("/", "--")
    path = _model_dir(name)
    existed = path.exists()
    try:
        snapshot_download(repo_id, local_dir=path, allow_patterns=HUB_PATTERNS)
        if not any(path.rglob("*.safetensors")):
            raise ValueError(f"{repo_id} has no safetensors weights")
    except Exception as e:
        if not existed:
            shutil.rmtree(path, ignore_errors=True)
        if isinstance(e, GatedRepoError):
            raise ValueError(f"{repo_id} is gated; accept its license on huggingface.co first") from e
        if isinstance(e, RepositoryNotFoundError):
            raise ValueError(f"{repo_id} not found on the Hugging Face Hub") from e
        raise
    (path / "trainform.json").write_text(json.dumps({"source": "hub", "repo_id": repo_id}))
    return name


def save_upload(fileobj, name: str) -> str:
    """Extract an uploaded zipped model directory into MODELS_DIR; returns its local model name."""
    dest = _model_dir(name)
    if dest.exists():
        raise FileExistsError(name)
    try:
        zf = zipfile.ZipFile(fileobj)
    except zipfile.BadZipFile as e:
        raise ValueError("Upload is not a valid zip file") from e
    with zf:
        entries = []
        for info in zf.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or parts[:1] == ("__MACOSX",):
                continue
            if info.filename.startswith("/") or ".." in parts:
                raise ValueError(f"Unsafe path in zip: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Symlinks are not allowed: {info.filename}")
            if info.filename.lower().endswith(PICKLE_SUFFIXES):
                raise ValueError(f"Pickle weights are not allowed, use safetensors: {info.filename}")
            entries.append((info, parts))
        if sum(info.file_size for info, _ in entries) > MAX_UPLOAD_BYTES:
            raise ValueError("Upload is too large")
        if entries and len({p[0] for _, p in entries}) == 1 and all(len(p) > 1 for _, p in entries):
            entries = [(info, p[1:]) for info, p in entries]
        if ("config.json",) not in [p for _, p in entries]:
            raise ValueError("Zip must contain a model directory with config.json")
        dest.mkdir(parents=True)
        try:
            for info, parts in entries:
                target = dest.joinpath(*parts)
                if not target.resolve().is_relative_to(dest):
                    raise ValueError(f"Unsafe path in zip: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            (dest / "trainform.json").write_text(json.dumps({"source": "upload"}))
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return name


def list_local_models() -> list[dict]:
    if not MODELS_DIR.is_dir():
        return []
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if (d / "config.json").is_file():
            try:
                source = json.loads((d / "trainform.json").read_text()).get("source")
            except (OSError, ValueError):
                source = None
            models.append({"name": d.name, "source": source})
    return models


class LocalModel:
    """A local causal LM wrapped with a LoRA adapter, so SFT updates stay small enough to store in Postgres."""

    def __init__(self, name: str):
        path = _model_dir(name)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Local model {name!r} not found")
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(path, use_safetensors=True)
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora).to(self.device)
        self.model.eval()

    def _prompt_ids(self, system_prompt: str, conversation: str) -> list[int]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": conversation}]
        if self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = f"{system_prompt}\n\nUser: {conversation}\nAssistant:"
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def generate(self, system_prompt: str, conversation: str) -> str:
        ids = torch.tensor([self._prompt_ids(system_prompt, conversation)], device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=256,
                                      do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def sft(self, examples: list[dict]) -> float:
        """Fine-tune on [{"prompt", "input", "output"}]; returns the mean training loss."""
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=2e-4)
        losses = []
        self.model.train()
        try:
            for ex in examples:
                prompt = self._prompt_ids(ex["prompt"], ex["input"])
                output = self.tokenizer(ex["output"] + (self.tokenizer.eos_token or ""),
                                        add_special_tokens=False)["input_ids"]
                ids, labels = (prompt + output)[:1024], ([-100] * len(prompt) + output)[:1024]
                if all(label == -100 for label in labels):
                    continue  # output truncated away: nothing to learn, and the loss would be NaN
                loss = self.model(input_ids=torch.tensor([ids], device=self.device),
                                  labels=torch.tensor([labels], device=self.device)).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
        finally:
            self.model.eval()
        return sum(losses) / len(losses) if losses else 0.0

    def adapter_bytes(self) -> bytes:
        """The current LoRA adapter as a zip archive."""
        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            self.model.save_pretrained(tmp)
            for f in Path(tmp).rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp))
        return buf.getvalue()


class DownloadIn(BaseModel):
    repo_id: str


@router.get("/hub/models")
def hub_models(search: str = "", sort: Literal["downloads", "lastModified"] = "downloads"):
    try:
        return list_hub_models(search, sort)
    except httpx.HTTPError as e:
        raise HTTPException(502, "Hugging Face Hub request failed") from e


# ponytail: downloads block the request until done; move to a background job if multi-GB models time out proxies.
@router.post("/hub/download", status_code=201)
def hub_download(body: DownloadIn):
    try:
        name = download_model(body.repo_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except httpx.HTTPError as e:
        raise HTTPException(502, "Hugging Face Hub request failed") from e
    return {"name": name, "source": "hub"}


@router.post("/hub/upload", status_code=201)
def hub_upload(file: UploadFile = File(), name: str = Form()):
    # UploadFile is already spooled to a temp file on disk, so the zip is never held in memory.
    try:
        return {"name": save_upload(file.file, name), "source": "upload"}
    except FileExistsError as e:
        raise HTTPException(409, f"A model named {name!r} already exists") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/models")
def models():
    return list_local_models()
