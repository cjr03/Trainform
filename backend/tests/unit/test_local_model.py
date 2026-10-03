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
import torch
from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError, RepositoryNotFoundError

from app import local_model
from app.local_model import LocalModel

MODEL = {"config.json": "{}", "model.safetensors": "w"}


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


def empty(path):
    return not path.exists() or not any(path.iterdir())


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path / "models")
    return tmp_path / "models"


@pytest.fixture(scope="session")
def tiny(tiny_model_dir):
    mp = pytest.MonkeyPatch()
    mp.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    yield LocalModel("tiny")
    mp.undo()


# --- names and local listing ---

@pytest.mark.parametrize("name", ["", ".", "..", ".hidden", "a/b", "../x", "a b", "x" * 129, "é"])
def test_invalid_names_rejected(models_dir, name):
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip(MODEL), name, 1)


def test_list_local_models(models_dir):
    assert local_model.list_local_models(1) == []
    for name, meta in [("b-model", {"source": "upload", "owner": 1}), ("a--model", {"source": "hub", "owner": 2}),
                       (".partial-x", {"source": "upload", "owner": 1}), ("c-bobs", {"source": "upload", "owner": 2}),
                       ("d-legacy", {"source": "upload"}), ("e-operator", None)]:
        (models_dir / name).mkdir(parents=True)
        (models_dir / name / "config.json").write_text("{}")
        if meta:
            (models_dir / name / "trainform.json").write_text(json.dumps(meta))
    (models_dir / "no-config").mkdir()
    # Hub and operator-provided models are shared; uploads are visible to their owner only.
    assert local_model.list_local_models(1) == [{"name": "a--model", "source": "hub"},
                                                {"name": "b-model", "source": "upload"},
                                                {"name": "e-operator", "source": None}]
    assert [m["name"] for m in local_model.list_local_models(2)] == ["a--model", "c-bobs", "e-operator"]


def test_upload_names_cannot_take_hub_names(models_dir):
    with pytest.raises(ValueError, match="--"):
        local_model.save_upload(make_zip(MODEL), "Qwen--Qwen2.5-0.5B-Instruct", 1)
    assert empty(models_dir)


def test_model_quota_per_user(models_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MAX_MODELS_PER_USER", 2)
    fake_hub(monkeypatch, TINY_REPO)
    local_model.download_model("org/two", 1)
    local_model.save_upload(make_zip(MODEL), "one", 1)
    local_model.save_upload(make_zip(MODEL), "bobs", 2)  # other users have their own quota
    with pytest.raises(ValueError, match="at most 2 models"):
        local_model.save_upload(make_zip(MODEL), "three", 1)
    with pytest.raises(ValueError, match="at most 2 models"):
        local_model.download_model("org/three", 1)
    assert local_model.download_model("org/two", 1) == "org--two"  # reusing a downloaded model costs nothing
    local_model.delete_model("one", 1)
    assert local_model.save_upload(make_zip(MODEL), "three", 1) == "three"


def test_delete_model_only_by_its_owner(models_dir):
    local_model.save_upload(make_zip(MODEL), "mine", 1)
    for name, owner in [("mine", 2), ("missing", 1), ("../x", 1)]:
        with pytest.raises((FileNotFoundError, ValueError)):
            local_model.delete_model(name, owner)
    local_model.delete_model("mine", 1)
    assert empty(models_dir)


# --- uploads ---

def test_upload_extracts_and_strips_top_folder(models_dir):
    zf = make_zip({"my-model/config.json": "{}", "my-model/model.safetensors": "w", "my-model/sub/tokenizer.json": "t",
                   "__MACOSX/my-model/._config.json": "junk"})
    assert local_model.save_upload(zf, "mine", 1) == "mine"
    d = models_dir / "mine"
    assert (d / "config.json").read_text() == "{}"
    assert (d / "sub" / "tokenizer.json").read_text() == "t"
    assert not (d / "__MACOSX").exists()
    assert json.loads((d / "trainform.json").read_text()) == {"source": "upload", "owner": 1}
    assert local_model.list_local_models(1) == [{"name": "mine", "source": "upload"}]
    assert [p.name for p in models_dir.iterdir()] == ["mine"]


def test_upload_flat_zip(models_dir):
    local_model.save_upload(make_zip(MODEL), "flat", 1)
    assert (models_dir / "flat" / "model.safetensors").exists()


@pytest.mark.parametrize("files,symlink", [
    ({**MODEL, "../evil.txt": "x"}, None),
    ({**MODEL, "/abs/evil.txt": "x"}, None),
    ({**MODEL, "pytorch_model.bin": "x"}, None),
    ({**MODEL, "weights.PT": "x"}, None),
    ({**MODEL, "x.pth": "x"}, None),
    ({**MODEL, "x.pkl": "x"}, None),
    ({**MODEL, "x.ckpt": "x"}, None),
    ({"model.safetensors": "w"}, None),
    ({"config.json": "{}"}, None),
    ({"config.json": "{}", "tf_model.h5": "x"}, None),
    ({**MODEL, "a": "x", "a/b": "y"}, None),
    ({**MODEL, "a/b": "x", "a": "y"}, None),
    (MODEL, "link"),
])
def test_upload_rejects_unsafe_or_invalid(models_dir, files, symlink):
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip(files, symlink), "bad", 1)
    assert empty(models_dir)
    assert not (models_dir.parent / "evil.txt").exists()


def test_upload_errors_do_not_leak_server_paths(models_dir):
    with pytest.raises(ValueError) as e:
        local_model.save_upload(make_zip({**MODEL, "a": "x", "a/b": "y"}), "bad", 1)
    assert str(models_dir) not in str(e.value) and "Errno" not in str(e.value)


def test_upload_rejects_corrupt_member(models_dir):
    data = bytearray(make_zip({"config.json": "{}", "model.safetensors": "weights-data"}).getvalue())
    data[data.find(b"weights-data")] ^= 1  # stored member: flipping a byte breaks its CRC
    with pytest.raises(ValueError, match="Corrupt"):
        local_model.save_upload(io.BytesIO(bytes(data)), "bad", 1)
    assert empty(models_dir)


def test_upload_rejects_non_zip(models_dir):
    with pytest.raises(ValueError):
        local_model.save_upload(io.BytesIO(b"not a zip"), "bad", 1)


def test_upload_zip_bomb_guard(models_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValueError):
        local_model.save_upload(make_zip({"config.json": "{}", "model.safetensors": "x" * 11}), "bomb", 1)
    assert empty(models_dir)


def test_upload_existing_name(models_dir):
    local_model.save_upload(make_zip(MODEL), "dup", 1)
    with pytest.raises(FileExistsError):
        local_model.save_upload(make_zip(MODEL), "dup", 1)


def test_upload_is_not_listed_until_complete(models_dir, monkeypatch):
    seen = []
    copy = local_model.shutil.copyfileobj
    monkeypatch.setattr(local_model.shutil, "copyfileobj",
                        lambda *a: (seen.append(local_model.list_local_models(1)), copy(*a)))
    local_model.save_upload(make_zip(MODEL), "inprog", 1)
    assert seen and all(s == [] for s in seen)
    assert local_model.list_local_models(1) == [{"name": "inprog", "source": "upload"}]


def test_upload_cleans_up_on_extraction_failure(models_dir, monkeypatch):
    monkeypatch.setattr(local_model.shutil, "copyfileobj", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        local_model.save_upload(make_zip(MODEL), "partial", 1)
    assert empty(models_dir)


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


def fake_hub(monkeypatch, files, snapshot_error=None):
    """Mock HfApi.model_info and snapshot_download for a repo whose files map paths to sizes."""
    calls = {}

    class FakeApi:
        def model_info(self, repo_id, files_metadata=False):
            calls.update(info_repo_id=repo_id, files_metadata=files_metadata)
            siblings = [SimpleNamespace(rfilename=f, size=n) for f, n in files.items()]
            return SimpleNamespace(sha="abc123", siblings=siblings)

    def snapshot_download(repo_id, revision, local_dir, allow_patterns):
        calls.update(repo_id=repo_id, revision=revision, local_dir=local_dir, allow_patterns=allow_patterns)
        if snapshot_error:
            raise snapshot_error
        for f in files:
            if any(fnmatch(f, p) for p in allow_patterns):
                (local_dir / f).parent.mkdir(parents=True, exist_ok=True)
                (local_dir / f).write_text("x")
                assert local_model.list_local_models(1) == []  # not visible while downloading

    monkeypatch.setattr(local_model, "HfApi", FakeApi)
    monkeypatch.setattr(local_model, "snapshot_download", snapshot_download)
    return calls


TINY_REPO = {"config.json": 10, "model.safetensors": 100, "tokenizer.json": 10, "pytorch_model.bin": 10**12}


def test_download_model(models_dir, monkeypatch):
    calls = fake_hub(monkeypatch, TINY_REPO)
    assert local_model.download_model("org/tiny-1.0", 1) == "org--tiny-1.0"
    assert calls["repo_id"] == calls["info_repo_id"] == "org/tiny-1.0" and calls["files_metadata"]
    assert calls["revision"] == "abc123"  # the revision whose sizes were checked
    allowed = lambda f: any(fnmatch(f, p) for p in calls["allow_patterns"])  # noqa: E731
    assert all(map(allowed, ["config.json", "model.safetensors", "tokenizer.model", "merges.txt", "chat_template.jinja"]))
    assert not any(map(allowed, ["pytorch_model.bin", "model.pt", "model.pth", "x.pkl", "x.ckpt", "model.onnx"]))
    d = models_dir / "org--tiny-1.0"
    assert (d / "model.safetensors").exists() and not (d / "pytorch_model.bin").exists()
    assert json.loads((d / "trainform.json").read_text()) == {"source": "hub", "repo_id": "org/tiny-1.0", "owner": 1}
    assert [p.name for p in models_dir.iterdir()] == ["org--tiny-1.0"]
    assert local_model.list_local_models(1) == [{"name": "org--tiny-1.0", "source": "hub"}]
    calls.clear()
    assert local_model.download_model("org/tiny-1.0", 1) == "org--tiny-1.0"  # already downloaded: reused
    assert calls == {}


def test_download_does_not_overwrite_upload(models_dir, monkeypatch):
    (models_dir / "org--x").mkdir(parents=True)  # an upload from before '--' was reserved for Hub names
    (models_dir / "org--x" / "trainform.json").write_text(json.dumps({"source": "upload"}))
    calls = fake_hub(monkeypatch, TINY_REPO)
    with pytest.raises(FileExistsError):
        local_model.download_model("org/x", 1)
    assert calls == {}
    assert json.loads((models_dir / "org--x" / "trainform.json").read_text()) == {"source": "upload"}


def test_download_too_large(models_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MAX_UPLOAD_BYTES", 119)
    calls = fake_hub(monkeypatch, TINY_REPO)  # 120 bytes of allowed files; the 1 TB .bin is not counted
    with pytest.raises(ValueError, match="too large"):
        local_model.download_model("org/big", 1)
    assert "repo_id" not in calls  # rejected before downloading
    monkeypatch.setattr(local_model, "MAX_UPLOAD_BYTES", 120)
    assert local_model.download_model("org/big", 1) == "org--big"


@pytest.mark.parametrize("repo_id", ["tiny", "a/b/c", "../x", "org/", "/x", "org/a b", "org/..", "./x"])
def test_download_bad_repo_id(models_dir, repo_id):
    with pytest.raises(ValueError):
        local_model.download_model(repo_id, 1)


@pytest.mark.parametrize("error", [RepositoryNotFoundError, GatedRepoError])
def test_download_not_found_or_gated(models_dir, monkeypatch, error):
    response = httpx.Response(404, request=httpx.Request("GET", "https://hf.co"))
    fake_hub(monkeypatch, TINY_REPO, error("nope", response=response))
    with pytest.raises(ValueError, match="org/missing"):
        local_model.download_model("org/missing", 1)
    assert empty(models_dir)


def test_download_failure_cleans_up(models_dir, monkeypatch):
    fake_hub(monkeypatch, TINY_REPO, LocalEntryNotFoundError("hub unreachable"))
    with pytest.raises(LocalEntryNotFoundError):
        local_model.download_model("org/tiny", 1)
    assert empty(models_dir)


def test_download_without_safetensors_rejected(models_dir, monkeypatch):
    calls = fake_hub(monkeypatch, {"config.json": 1, "pytorch_model.bin": 1})
    with pytest.raises(ValueError, match="safetensors"):
        local_model.download_model("org/old", 1)
    assert "repo_id" not in calls
    assert empty(models_dir)


# --- LocalModel ---

def test_generate_returns_string(tiny):
    assert isinstance(tiny.generate("be helpful", "hi there"), str)


def test_generate_without_chat_template(tiny, monkeypatch):
    monkeypatch.setattr(tiny.tokenizer, "chat_template", None)
    assert isinstance(tiny.generate("be helpful", "hi"), str)


def test_generate_decodes_only_new_tokens(tiny, monkeypatch):
    hello, eos = tiny.tokenizer.convert_tokens_to_ids("hello"), tiny.tokenizer.eos_token_id
    monkeypatch.setattr(tiny.model, "generate", lambda input_ids, **kw: torch.cat(
        [input_ids, torch.tensor([[hello, eos]], device=input_ids.device)], dim=1))
    assert tiny.generate("be helpful", "hi there") == "hello"


@pytest.fixture
def forward_spy(tiny, monkeypatch):
    calls = []
    forward = tiny.model.forward

    def spy(input_ids, labels):
        calls.append((input_ids[0].tolist(), labels[0].tolist()))
        return forward(input_ids=input_ids, labels=labels)
    monkeypatch.setattr(tiny.model, "forward", spy)
    return calls


def test_sft_updates_lora_weights(tiny):
    before = {n: p.detach().clone() for n, p in tiny.model.named_parameters() if "lora_" in n}
    assert before
    loss = tiny.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 2)
    assert isinstance(loss, float) and math.isfinite(loss) and loss > 0
    after = dict(tiny.model.named_parameters())
    assert any(not before[n].equal(after[n].detach()) for n in before)
    assert not tiny.model.training


def test_sft_masks_prompt_tokens(tiny, forward_spy):
    tiny.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}])
    prompt = tiny._prompt_ids("be helpful", "hi")
    output = tiny.tokenizer("hello there", add_special_tokens=False)["input_ids"] + [tiny.tokenizer.eos_token_id]
    [(ids, labels)] = forward_spy
    assert ids == prompt + output
    assert labels == [-100] * len(prompt) + output


def test_sft_truncates_to_1024_tokens(tiny, forward_spy):
    assert tiny.sft([{"prompt": "be helpful", "input": "hi " * 1100, "output": "hello"}]) == 0.0
    assert forward_spy == []  # output truncated away entirely: skipped
    prompt_len = len(tiny._prompt_ids("be helpful", "hi " * 1000))
    loss = tiny.sft([{"prompt": "be helpful", "input": "hi " * 1000, "output": "hello " * 100}])
    assert math.isfinite(loss) and loss > 0
    [(ids, labels)] = forward_spy
    assert len(ids) == len(labels) == 1024
    assert labels[:prompt_len] == [-100] * prompt_len and -100 not in labels[prompt_len:]


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


def test_complete_with_digits_replies_with_only_a_number(tiny):
    for message in ("hi", "hello there", "be helpful"):
        words = tiny.complete("system", message, digits=True).split()
        assert 1 <= len(words) <= 2 and set(words) <= {"1", "7", "10"}


def test_complete_ignores_the_fine_tuned_adapter(tiny_model_dir, monkeypatch):
    # The helper agents grade and coach the model being trained, so its fine-tuning must never change them.
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    model = LocalModel("tiny")
    before = model.complete("system", "hi there"), model.complete("system", "hi", digits=True)
    tuned_before = model.generate("be helpful", "hi")
    model.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there hello there"}] * 40)
    assert model.generate("be helpful", "hi") != tuned_before  # the adapter did learn something
    assert (model.complete("system", "hi there"), model.complete("system", "hi", digits=True)) == before
