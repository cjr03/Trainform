import io
import json
import math
import stat
import zipfile
from datetime import datetime, timezone
from fnmatch import fnmatch
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from app import local_model
from app.local_model import LocalModel

WORDS = ["<unk>", "<pad>", "</s>", "system", "user", "assistant", ":", "be", "helpful", "hi", "hello", "there"]


def make_zip(files, symlink=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, data in files.items():
            zf.writestr(path, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    buf.seek(0)
    return buf


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path / "models")
    return tmp_path / "models"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A 2-layer Llama with a word-level tokenizer, built offline."""
    root = tmp_path_factory.mktemp("models")
    path = root / "tiny"
    tok = Tokenizer(tokenizer_models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}assistant: {% endif %}")
    tokenizer.save_pretrained(path)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, eos_token_id=2, pad_token_id=1, bos_token_id=None)
    LlamaForCausalLM(config).save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny(tiny_model_dir):
    mp = pytest.MonkeyPatch()
    mp.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    yield LocalModel("tiny")
    mp.undo()


# --- names and local listing ---

@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../x", "a b", "x" * 129, "é"])
def test_invalid_names_rejected(models_dir, name):
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip({"config.json": "{}"}), name)


def test_list_local_models(models_dir):
    assert local_model.list_local_models() == []
    for name, source in [("b-model", "upload"), ("a--model", "hub")]:
        (models_dir / name).mkdir(parents=True)
        (models_dir / name / "config.json").write_text("{}")
        (models_dir / name / "trainform.json").write_text(json.dumps({"source": source}))
    (models_dir / "no-config").mkdir()
    assert local_model.list_local_models() == [{"name": "a--model", "source": "hub"},
                                               {"name": "b-model", "source": "upload"}]


# --- uploads ---

def test_upload_extracts_and_strips_top_folder(models_dir):
    zf = make_zip({"my-model/config.json": "{}", "my-model/model.safetensors": "w", "my-model/sub/tokenizer.json": "t",
                   "__MACOSX/my-model/._config.json": "junk"})
    assert local_model.save_upload(zf, "mine") == "mine"
    d = models_dir / "mine"
    assert (d / "config.json").read_text() == "{}"
    assert (d / "sub" / "tokenizer.json").read_text() == "t"
    assert not (d / "__MACOSX").exists()
    assert json.loads((d / "trainform.json").read_text())["source"] == "upload"
    assert local_model.list_local_models() == [{"name": "mine", "source": "upload"}]


def test_upload_flat_zip(models_dir):
    local_model.save_upload(make_zip({"config.json": "{}", "model.safetensors": "w"}), "flat")
    assert (models_dir / "flat" / "model.safetensors").exists()


@pytest.mark.parametrize("files,symlink", [
    ({"config.json": "{}", "../evil.txt": "x"}, None),
    ({"config.json": "{}", "/abs/evil.txt": "x"}, None),
    ({"config.json": "{}", "pytorch_model.bin": "x"}, None),
    ({"config.json": "{}", "weights.PT": "x"}, None),
    ({"config.json": "{}", "x.pth": "x"}, None),
    ({"config.json": "{}", "x.pkl": "x"}, None),
    ({"config.json": "{}", "x.ckpt": "x"}, None),
    ({"model.safetensors": "w"}, None),
    ({"config.json": "{}"}, "link"),
])
def test_upload_rejects_unsafe_or_invalid(models_dir, files, symlink):
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip(files, symlink), "bad")
    assert not (models_dir / "bad").exists()
    assert not (models_dir.parent / "evil.txt").exists()


def test_upload_rejects_non_zip(models_dir):
    with pytest.raises(ValueError):
        local_model.save_upload(io.BytesIO(b"not a zip"), "bad")


def test_upload_zip_bomb_guard(models_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip({"config.json": "{}", "model.safetensors": "x" * 11}), "bomb")
    assert not (models_dir / "bomb").exists()


def test_upload_existing_name(models_dir):
    local_model.save_upload(make_zip({"config.json": "{}"}), "dup")
    with pytest.raises(FileExistsError):
        local_model.save_upload(make_zip({"config.json": "{}"}), "dup")


def test_upload_cleans_up_on_extraction_failure(models_dir, monkeypatch):
    monkeypatch.setattr(local_model.shutil, "copyfileobj", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        local_model.save_upload(make_zip({"config.json": "{}"}), "partial")
    assert not (models_dir / "partial").exists()


# --- hub ---

def test_list_hub_models(monkeypatch):
    calls = {}

    class FakeApi:
        def list_models(self, **kwargs):
            calls.update(kwargs)
            return [SimpleNamespace(id="org/a", downloads=5, likes=2,
                                    last_modified=datetime(2026, 1, 2, tzinfo=timezone.utc)),
                    SimpleNamespace(id="org/b", downloads=1, likes=0, last_modified=None)]

    monkeypatch.setattr(local_model, "HfApi", FakeApi)
    assert local_model.list_hub_models("", "lastModified") == [
        {"id": "org/a", "downloads": 5, "likes": 2, "last_modified": "2026-01-02T00:00:00+00:00"},
        {"id": "org/b", "downloads": 1, "likes": 0, "last_modified": None},
    ]
    assert calls["search"] is None
    assert (calls["pipeline_tag"], calls["sort"], calls["limit"]) == ("text-generation", "lastModified", 20)


def test_list_hub_models_bad_sort():
    with pytest.raises(ValueError):
        local_model.list_hub_models(None, "likes")


def fake_snapshot(files):
    calls = {}

    def snapshot_download(repo_id, local_dir, allow_patterns):
        calls.update(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
        for f in files:
            (local_dir / f).parent.mkdir(parents=True, exist_ok=True)
            (local_dir / f).write_text("x")
    return snapshot_download, calls


def test_download_model(models_dir, monkeypatch):
    snapshot, calls = fake_snapshot(["config.json", "model.safetensors", "tokenizer.json"])
    monkeypatch.setattr(local_model, "snapshot_download", snapshot)
    assert local_model.download_model("org/tiny-1.0") == "org--tiny-1.0"
    assert calls["repo_id"] == "org/tiny-1.0"
    assert calls["local_dir"] == models_dir / "org--tiny-1.0"
    allowed = lambda f: any(fnmatch(f, p) for p in calls["allow_patterns"])  # noqa: E731
    assert all(map(allowed, ["config.json", "model.safetensors", "tokenizer.model", "merges.txt", "chat_template.jinja"]))
    assert not any(map(allowed, ["pytorch_model.bin", "model.pt", "model.pth", "x.pkl", "x.ckpt", "model.onnx"]))
    meta = json.loads((models_dir / "org--tiny-1.0" / "trainform.json").read_text())
    assert meta == {"source": "hub", "repo_id": "org/tiny-1.0"}
    assert local_model.list_local_models() == [{"name": "org--tiny-1.0", "source": "hub"}]


@pytest.mark.parametrize("repo_id", ["tiny", "a/b/c", "../x", "org/", "/x", "org/a b", "org/.."])
def test_download_bad_repo_id(models_dir, repo_id):
    with pytest.raises(ValueError):
        local_model.download_model(repo_id)


@pytest.mark.parametrize("error", [RepositoryNotFoundError, GatedRepoError])
def test_download_not_found_or_gated(models_dir, monkeypatch, error):
    def snapshot_download(**kwargs):
        kwargs["local_dir"].mkdir(parents=True)
        raise error("nope", response=httpx.Response(404, request=httpx.Request("GET", "https://hf.co")))
    monkeypatch.setattr(local_model, "snapshot_download", lambda repo_id, **kw: snapshot_download(**kw))
    with pytest.raises(ValueError, match="org/missing"):
        local_model.download_model("org/missing")
    assert not (models_dir / "org--missing").exists()


def test_download_without_safetensors_rejected(models_dir, monkeypatch):
    monkeypatch.setattr(local_model, "snapshot_download", fake_snapshot(["config.json"])[0])
    with pytest.raises(ValueError, match="safetensors"):
        local_model.download_model("org/old")
    assert not (models_dir / "org--old").exists()


# --- LocalModel ---

def test_generate_returns_string(tiny):
    assert isinstance(tiny.generate("be helpful", "hi there"), str)


def test_generate_without_chat_template(tiny, monkeypatch):
    monkeypatch.setattr(tiny.tokenizer, "chat_template", None)
    assert isinstance(tiny.generate("be helpful", "hi"), str)


def test_sft_updates_lora_weights(tiny):
    before = {n: p.detach().clone() for n, p in tiny.model.named_parameters() if "lora_" in n}
    assert before
    loss = tiny.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 2)
    assert isinstance(loss, float) and math.isfinite(loss) and loss > 0
    after = dict(tiny.model.named_parameters())
    assert any(not before[n].equal(after[n].detach()) for n in before)
    assert not tiny.model.training


def test_sft_no_examples(tiny):
    assert tiny.sft([]) == 0.0


def test_adapter_bytes_is_zip(tiny):
    with zipfile.ZipFile(io.BytesIO(tiny.adapter_bytes())) as zf:
        names = zf.namelist()
        assert "adapter_config.json" in names
        assert any(n.endswith(".safetensors") for n in names)
        assert json.loads(zf.read("adapter_config.json"))["r"] == 8


def test_local_model_missing_or_invalid(models_dir):
    with pytest.raises(ValueError):
        LocalModel("../etc")
    with pytest.raises(FileNotFoundError):
        LocalModel("absent")
