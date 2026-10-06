"""Hugging Face Hub access, local model storage, generation, and LoRA SFT."""

import io
import json
import logging
import os
import random
import re
import shutil
import stat
import tempfile
import threading
import zipfile
import zlib
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.utils import filter_repo_objects
from peft import LoraConfig, get_peft_model
from peft.utils import load_peft_weights, set_peft_model_state_dict
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, NoRepeatNGramLogitsProcessor

from app.auth import current_user
from app.models import User

log = logging.getLogger(__name__)
router = APIRouter(tags=["models"], dependencies=[Depends(current_user)])

MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "data" / "models"))
MAX_UPLOAD_BYTES = 20 * 2**30  # per model, uncompressed: zip-bomb and disk-fill guard
# Uploads plus Hub downloads a user added; with MAX_UPLOAD_BYTES this bounds the disk one account can fill.
# ponytail: counted without a lock, so concurrent requests can overshoot by a model or two.
MAX_MODELS_PER_USER = 5
_DOWNLOADS = threading.BoundedSemaphore(2)  # each download holds a worker thread the sync routes share
NAME_RE = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}")  # no leading dot: hidden dirs are staging areas
REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
# Configs, safetensors weights and tokenizer files only: pickle weights (*.bin, *.pt, ...) can execute code on load.
HUB_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".ckpt")
# Room for a ~1000-word system prompt, a message and a reply; a longer example is dropped, not cut short.
MAX_SFT_TOKENS = 4096


def _model_dir(name: str) -> Path:
    root = MODELS_DIR.resolve()
    if not NAME_RE.fullmatch(name) or (root / name).resolve().parent != root:
        raise ValueError(f"Invalid model name: {name!r}")
    return root / name


def _meta(path: Path) -> dict:
    """trainform.json: {"source": "hub" | "upload", "owner": user id, ...}; {} for operator-provided models."""
    try:
        return json.loads((path / "trainform.json").read_text())
    except (OSError, ValueError):
        return {}


def _check_quota(owner: int) -> None:
    if MODELS_DIR.is_dir() and sum(_meta(d).get("owner") == owner for d in MODELS_DIR.iterdir()) >= MAX_MODELS_PER_USER:
        raise ValueError(f"You can keep at most {MAX_MODELS_PER_USER} models; delete one first")


@contextmanager
def _staged(dest: Path, meta: dict):
    """Yield a hidden staging dir that is renamed to dest on success, so half-written models are never listed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=".partial-"))
    try:
        yield tmp
        (tmp / "trainform.json").write_text(json.dumps(meta))
        try:
            os.rename(tmp, dest)
        except OSError as e:
            if dest.exists():
                raise FileExistsError(dest.name) from e
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def list_hub_models(search: str | None = None, sort: str = "downloads", limit: int = 20) -> list[dict]:
    if sort not in ("downloads", "lastModified"):
        raise ValueError("sort must be 'downloads' or 'lastModified'")
    models = HfApi().list_models(search=search or None, pipeline_tag="text-generation", sort=sort, limit=limit,
                                 expand=["downloads", "likes", "lastModified"])
    return [{"id": m.id, "downloads": m.downloads, "likes": m.likes,
             "last_modified": m.last_modified.isoformat() if m.last_modified else None} for m in models]


def download_model(repo_id: str, owner: int) -> str:
    """Download a Hub model into MODELS_DIR for user `owner`; returns its local model name."""
    if not REPO_RE.fullmatch(repo_id):
        raise ValueError(f"Invalid Hugging Face repo id {repo_id!r}; expected 'owner/name'")
    name = repo_id.replace("/", "--")
    dest = _model_dir(name)
    if dest.exists():
        if _meta(dest).get("source") == "hub":
            return name
        raise FileExistsError(name)
    _check_quota(owner)
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        files = list(filter_repo_objects(info.siblings or [], allow_patterns=HUB_PATTERNS, key=lambda s: s.rfilename))
        if not any(s.rfilename.endswith(".safetensors") for s in files):
            raise ValueError(f"{repo_id} has no safetensors weights")
        if sum(s.size or 0 for s in files) > MAX_UPLOAD_BYTES:
            raise ValueError(f"{repo_id} is too large (limit {MAX_UPLOAD_BYTES // 2**30} GiB)")
        with _staged(dest, {"source": "hub", "repo_id": repo_id, "owner": owner}) as tmp:
            # Pinned to the revision whose sizes were checked, so a push in between can't slip past the cap.
            snapshot_download(repo_id, revision=info.sha, local_dir=tmp, allow_patterns=HUB_PATTERNS)
    except GatedRepoError as e:
        raise ValueError(f"{repo_id} is gated; accept its license on huggingface.co first") from e
    except RepositoryNotFoundError as e:
        raise ValueError(f"{repo_id} not found on the Hugging Face Hub") from e
    return name


def save_upload(fileobj, name: str, owner: int) -> str:
    """Extract an uploaded zipped model directory into MODELS_DIR for user `owner`; returns its local model name."""
    dest = _model_dir(name)
    if "--" in name:
        raise ValueError("Model names can't contain '--'; it is reserved for Hugging Face models (owner--name)")
    if dest.exists():
        raise FileExistsError(name)
    _check_quota(owner)
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
        if not any(p[-1].endswith(".safetensors") for _, p in entries):
            raise ValueError("Zip must contain *.safetensors weights")
        with _staged(dest, {"source": "upload", "owner": owner}) as tmp:
            for info, parts in entries:
                target = tmp.joinpath(*parts)
                if not target.resolve().is_relative_to(tmp):
                    raise ValueError(f"Unsafe path in zip: {info.filename}")
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                except (zipfile.BadZipFile, zlib.error, EOFError, FileExistsError, IsADirectoryError,
                        NotADirectoryError) as e:  # bad CRC/deflate data, or paths like "a" and "a/b" colliding
                    raise ValueError(f"Corrupt or invalid zip member {info.filename}") from e  # e has server paths
    return name


def delete_model(name: str, owner: int) -> None:
    path = _model_dir(name)
    if _meta(path).get("owner") != owner:
        raise FileNotFoundError(name)
    shutil.rmtree(path)


def list_local_models(user_id: int) -> list[dict]:
    """Hub and operator-provided models, plus the user's own uploads."""
    if not MODELS_DIR.is_dir():
        return []
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        meta = _meta(d)
        if (not d.name.startswith(".") and (d / "config.json").is_file()
                and (meta.get("source") != "upload" or meta.get("owner") == user_id)):
            models.append({"name": d.name, "source": meta.get("source")})
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
        # bfloat16 halves a GPU's memory traffic, so it generates and trains faster; CPUs are slow at it. PEFT keeps the
        # LoRA adapter in float32, so SFT updates are not lost to rounding.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(path, use_safetensors=True, dtype=dtype)
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lora).to(self.device)
        self.model.eval()

    def _prompt_ids(self, system_prompt: str, conversation: str) -> list[int]:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": conversation}]
        if self.tokenizer.chat_template:
            # Reasoning models (e.g. Qwen3) would otherwise think aloud first: a reply 3x slower, with its reasoning
            # shown to the user, graded and trained on. Templates that don't know the flag ignore it.
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        else:
            text = f"{system_prompt}\n\nUser: {conversation}\nAssistant:"
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def generate(self, system_prompt: str, conversation: str, max_new_tokens: int = 256, **kwargs) -> str:
        ids = torch.tensor([self._prompt_ids(system_prompt, conversation)], device=self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, pad_token_id=self.tokenizer.pad_token_id, **kwargs)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def complete(self, system_prompt: str, message: str, digits: bool = False) -> str:
        """Helper-agent generation (used when Claude isn't linked). Runs on the base weights with the LoRA adapter off,
        so fine-tuning never changes the agents that grade and coach the model. `digits` restricts the reply to a
        number of at most two tokens."""
        start = len(self._prompt_ids(system_prompt, message))
        with self.model.disable_adapter():
            if not digits:
                # Greedy decoding makes small models loop on one line until the token limit. A repetition penalty
                # would also penalise the end-of-turn token (it is in the prompt), so block repeated 4-grams instead,
                # within the reply only: a reply must be free to keep the wording of a prompt it revises.
                block = NoRepeatNGramLogitsProcessor(4)
                return self.generate(system_prompt, message, max_new_tokens=512,
                                     logits_processor=[lambda ids, scores: block(ids[:, start:], scores)])
            ends = self._end_ids()
            return self.generate(system_prompt, message, max_new_tokens=2, prefix_allowed_tokens_fn=lambda _, ids: (
                self._digit_ids if len(ids) == start else self._digit_ids + ends))

    @cached_property
    def _digit_ids(self) -> list[int]:
        # isascii: str.isdigit() also accepts superscripts like "²", which int() rejects.
        return [i for token, i in self.tokenizer.get_vocab().items()
                if (text := self.tokenizer.convert_tokens_to_string([token]).strip()).isascii() and text.isdigit()]

    def _end_ids(self) -> list[int]:
        """Every token that ends the model's turn; chat models often list more than the tokenizer's EOS."""
        eos = self.model.generation_config.eos_token_id
        return sorted(i for i in {self.tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])}
                      if i is not None)

    def sft(self, examples: list[dict], epochs: int = 2, accum: int = 4, seed: int = 0) -> float:
        """Fine-tune on [{"prompt", "input", "output"}], learning only the outputs: `epochs` shuffled passes, each step
        on the clipped mean gradient of `accum` examples. Returns the mean training loss."""
        data = []
        for ex in examples:
            prompt = self._prompt_ids(ex["prompt"], ex["input"])
            output = self.tokenizer(ex["output"] + (self.tokenizer.eos_token or ""),
                                    add_special_tokens=False)["input_ids"]
            if output and len(prompt) + len(output) <= MAX_SFT_TOKENS:
                data.append((prompt + output, [-100] * len(prompt) + output))
        if len(data) < len(examples):
            log.warning("SFT dropped %d of %d examples over %d tokens", len(examples) - len(data), len(examples),
                        MAX_SFT_TOKENS)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=2e-4)
        rng, losses = random.Random(seed), []
        self.model.train()
        try:
            for _ in range(epochs if data else 0):
                rng.shuffle(data)
                for i, (ids, labels) in enumerate(data, 1):
                    loss = self.model(input_ids=torch.tensor([ids], device=self.device),
                                      labels=torch.tensor([labels], device=self.device)).loss
                    (loss / accum).backward()
                    losses.append(loss.item())
                    if i % accum == 0 or i == len(data):
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
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

    def load_adapter(self, data: bytes) -> None:
        """Restore an adapter from adapter_bytes, e.g. to resume a training on the weights it had trained."""
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp)  # our own adapter_bytes, read back from the database
            # Read on the CPU; set_peft_model_state_dict copies into the parameters on whatever device they live.
            set_peft_model_state_dict(self.model, load_peft_weights(tmp, device="cpu"))


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
def hub_download(body: DownloadIn, user: User = Depends(current_user)):
    if not _DOWNLOADS.acquire(blocking=False):
        raise HTTPException(429, "Too many model downloads in progress; try again shortly.")
    try:
        name = download_model(body.repo_id, user.id)
    except FileExistsError as e:
        raise HTTPException(409, f"A model named {e} already exists") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except (httpx.HTTPError, LocalEntryNotFoundError) as e:  # snapshot_download reports an unreachable Hub as the latter
        raise HTTPException(502, "Hugging Face Hub request failed") from e
    finally:
        _DOWNLOADS.release()
    return {"name": name, "source": "hub"}


@router.post("/hub/upload", status_code=201)
def hub_upload(file: UploadFile = File(), name: str = Form(), user: User = Depends(current_user)):
    # UploadFile is already spooled to a temp file on disk (main.py caps its size first), so it never sits in memory.
    try:
        return {"name": save_upload(file.file, name, user.id), "source": "upload"}
    except FileExistsError as e:
        raise HTTPException(409, f"A model named {name!r} already exists") from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@router.get("/models")
def models(user: User = Depends(current_user)):
    return list_local_models(user.id)


@router.delete("/models/{name}", status_code=204)
def delete(name: str, user: User = Depends(current_user)):
    try:
        delete_model(name, user.id)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(404, "Model not found") from e
