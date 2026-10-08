import io
import json
import math
import os
import shutil
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


def test_reasoning_models_answer_without_thinking_first(tiny, monkeypatch):
    # Qwen3 and similar think aloud by default: a reply 3x slower, its reasoning shown to the user, graded and trained on.
    monkeypatch.setattr(tiny.tokenizer, "chat_template", tiny.tokenizer.chat_template.replace(
        "assistant: ", "assistant: {% if enable_thinking is not defined or enable_thinking %}<think>{% endif %}"))
    assert tiny.tokenizer.unk_token_id not in tiny._prompt_ids("be helpful", "hi")  # "<think>" is not in its vocab


def test_generate_decodes_only_new_tokens(tiny, monkeypatch):
    hello, eos = tiny.tokenizer.convert_tokens_to_ids("hello"), tiny.tokenizer.eos_token_id
    monkeypatch.setattr(tiny.model, "generate", lambda input_ids, **kw: torch.cat(
        [input_ids, torch.tensor([[hello, eos]], device=input_ids.device)], dim=1))
    assert tiny.generate("be helpful", "hi there") == "hello"


@pytest.fixture
def loss_spy(tiny, monkeypatch):
    """The microbatches SFT computes losses on: [[(ids, labels)]]."""
    calls = []
    losses = tiny._losses
    monkeypatch.setattr(tiny, "_losses", lambda batch: calls.append(batch) or losses(batch))
    return calls


@pytest.fixture
def peaked(tiny):
    """`tiny` with a sharpened output layer: a random model's near-tied logits would let padding's float rounding flip
    a greedy pick, which is noise, not a batching bug."""
    w = tiny.model.get_output_embeddings().weight
    original = w.detach().clone()
    with torch.no_grad():
        w.mul_(100)
    yield tiny
    with torch.no_grad():
        w.copy_(original)


def test_generate_many_matches_one_at_a_time_in_input_order(peaked, monkeypatch):
    # Left padding must not change a greedy reply, and replies come back in the order asked whatever the batching.
    tiny = peaked
    monkeypatch.setattr(local_model, "GEN_BATCH", 3)
    pairs = [("be helpful", "hi"), ("be helpful be helpful be helpful", "hello there hi hello"), ("system", "hi there"),
             ("be", "1 7 10 hi hello"), ("be helpful", "hello")] * 2
    batches = []
    generate = tiny.model.generate
    monkeypatch.setattr(tiny.model, "generate", lambda **kw: batches.append(kw["input_ids"].shape[0]) or generate(**kw))
    stops = []
    many = tiny.generate_many(pairs, max_new_tokens=8, stop=lambda: stops.append(1))
    assert batches == [3, 3, 3, 1] and len(stops) > 4  # a stop check before each batch and each token
    assert many == [tiny.generate(*p, max_new_tokens=8) for p in pairs]
    assert tiny.complete_many(pairs, digits=True) == [tiny.complete(*p, digits=True) for p in pairs]
    assert tiny.complete_many(pairs[:5]) == [tiny.complete(*p) for p in pairs[:5]]


def test_generate_many_caps_a_batchs_tokens(tiny, monkeypatch):
    # Eight rows of a ~25k-token prompt asked MPS for 6 GiB at once and failed the training; one row at a time fits.
    seen = []
    monkeypatch.setattr(tiny.model, "generate", lambda input_ids, **kw: seen.append(input_ids.shape) or input_ids)
    long = ("be helpful " * 20, "hi")
    monkeypatch.setattr(local_model, "GEN_TOKENS", 2 * (len(tiny._prompt_ids(*long)) + 8))
    assert tiny.generate_many([long] * 5, max_new_tokens=8) == [""] * 5
    assert [s[0] for s in seen] == [2, 2, 1]


def test_generate_many_batches_prompts_of_similar_length(tiny, monkeypatch):
    # Padding a short prompt to a long one's length costs as much compute as the long one.
    monkeypatch.setattr(local_model, "GEN_BATCH", 2)
    seen = []
    monkeypatch.setattr(tiny.model, "generate", lambda input_ids, **kw: seen.append(input_ids.shape) or input_ids)
    short, long = ("be", "hi"), ("be helpful " * 20, "hi")
    assert tiny.generate_many([short, long, short, long]) == [""] * 4
    assert [s[1] for s in seen] == [len(tiny._prompt_ids(*short)), len(tiny._prompt_ids(*long))]


def test_a_stop_drops_the_generation(tiny):
    def stop():
        raise RuntimeError("paused")

    with pytest.raises(RuntimeError, match="paused"):
        tiny.generate_many([("be", "hi")], stop=stop)


def test_a_stop_drops_a_batch_mid_reply(tiny):
    calls = []

    def stop():  # passes the check before the batch, then pauses during it
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("paused")

    with pytest.raises(RuntimeError, match="paused"):
        tiny.generate_many([("be", "hi")], max_new_tokens=50, stop=stop)
    assert len(calls) == 2


def test_sft_updates_lora_weights(tiny):
    before = {n: p.detach().clone() for n, p in tiny.model.named_parameters() if "lora_" in n}
    assert before
    loss = tiny.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 2)
    assert isinstance(loss, float) and math.isfinite(loss) and loss > 0
    after = dict(tiny.model.named_parameters())
    assert any(not before[n].equal(after[n].detach()) for n in before)
    assert not tiny.model.training


def test_sft_masks_prompt_tokens(tiny, loss_spy):
    tiny.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}])
    prompt = tiny._prompt_ids("be helpful", "hi")
    output = tiny.tokenizer("hello there", add_special_tokens=False)["input_ids"] + [tiny.tokenizer.eos_token_id]
    assert loss_spy == [[(prompt + output, [-100] * len(prompt) + output)]]


def test_a_microbatchs_losses_are_each_examples_own(tiny):
    # Right padding and other examples in the batch must not change an example's loss: the mean over its reply tokens.
    examples = []
    for prompt, output in (("be helpful", "hello there hello"), ("be helpful be helpful be", "hi")):
        ids, out = tiny._prompt_ids(prompt, "hi"), tiny.tokenizer(output)["input_ids"]
        examples.append((ids + out, [-100] * len(ids) + out))
    with torch.no_grad():
        batched = tiny._losses(examples)
        alone = [tiny.model(input_ids=torch.tensor([ids], device=tiny.device),
                            labels=torch.tensor([labels], device=tiny.device)).loss for ids, labels in examples]
    assert torch.allclose(batched, torch.stack(alone), atol=1e-5)


def test_a_stop_drops_the_rest_of_the_sft(tiny, loss_spy):
    def stop():
        if loss_spy:
            raise RuntimeError("paused")

    with pytest.raises(RuntimeError, match="paused"):
        tiny.sft([{"prompt": "be helpful", "input": w, "output": "hello"} for w in ("hi", "hello", "there")], stop=stop)
    assert len(loss_spy) == 1 and not tiny.model.training


def test_sft_learns_from_a_long_prompt(tiny, loss_spy):
    # Generated prompts run to ~1000 words. A 1024-token cap cut every reply away, so SFT trained on nothing.
    loss = tiny.sft([{"prompt": "be helpful " * 600, "input": "hi", "output": "hello there"}])
    assert math.isfinite(loss) and loss > 0
    [(ids, labels)] = loss_spy[0]
    assert len(ids) > 1024 and labels[-3:] != [-100] * 3


def test_sft_drops_an_example_over_the_cap_instead_of_truncating_its_reply(tiny, loss_spy, caplog):
    too_long = {"prompt": "be helpful", "input": "hi " * local_model.MAX_SFT_TOKENS, "output": "hello"}
    assert tiny.sft([too_long]) == 0.0
    assert loss_spy == [] and "1 of 1" in caplog.text


def test_sft_steps_once_per_example_on_clipped_gradients(tiny, monkeypatch):
    steps, clips = [], []
    real_clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: steps.append(1))
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda params, max_norm: clips.append(max_norm) or
                        real_clip(params, max_norm))
    examples = [{"prompt": "be helpful", "input": w, "output": "hello there"} for w in ("hi", "hello", "there")] * 2
    tiny.sft(examples, epochs=2)
    assert len(steps) == len(clips) == 12
    assert set(clips) == {1.0}


def test_sft_loss_falls_as_it_trains(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    model = LocalModel("tiny")
    examples = [{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 4
    losses = [model.sft(examples, seed=k) for k in range(3)]
    assert losses[-1] < losses[0]


def test_sft_no_examples(tiny):
    assert tiny.sft([]) == 0.0


def test_adapter_bytes_is_zip(tiny):
    with zipfile.ZipFile(io.BytesIO(tiny.adapter_bytes())) as zf:
        names = zf.namelist()
        assert "adapter_config.json" in names
        assert any(n.endswith(".safetensors") for n in names)
        assert json.loads(zf.read("adapter_config.json"))["r"] == 8


def test_load_adapter_restores_adapter_bytes(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    trained = LocalModel("tiny")
    trained.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 3)
    # clone: on the CPU, .cpu() returns the same storage, so a snapshot would change along with the model.
    lora = lambda m: {n: p.detach().cpu().clone() for n, p in m.model.named_parameters() if "lora_" in n}  # noqa: E731
    resumed = LocalModel("tiny")
    assert any(not p.equal(lora(trained)[n]) for n, p in lora(resumed).items())
    resumed.load_adapter(trained.adapter_bytes())
    assert lora(resumed).keys() == lora(trained).keys()
    assert all(p.equal(lora(trained)[n]) for n, p in lora(resumed).items())
    before = lora(resumed)
    resumed.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}])  # still trainable after loading
    assert any(not p.equal(before[n]) for n, p in lora(resumed).items())


def test_local_model_missing_or_invalid(models_dir):
    with pytest.raises(ValueError):
        LocalModel("../etc")
    with pytest.raises(FileNotFoundError):
        LocalModel("absent")


def test_complete_with_digits_replies_with_only_a_number(tiny):
    for message in ("hi", "hello there", "be helpful"):
        words = tiny.complete("system", message, digits=True).split()
        assert 1 <= len(words) <= 2 and set(words) <= {"1", "7", "10"}


def test_complete_blocks_repeats_within_its_reply_but_may_quote_its_input(tiny, monkeypatch):
    # Blocking 4-grams from the input too forced misspellings ("approachate") whenever a reply kept the wording of the
    # prompt it was revising.
    calls = []
    monkeypatch.setattr(tiny, "generate_many", lambda pairs, **kwargs: calls.append(kwargs) or ["reply"])
    tiny.complete("system", "keep this")
    start = len(tiny._prompt_ids("system", "keep this"))
    (block,) = calls[0]["options"](start)["logits_processor"]
    quoting = torch.tensor([[5] * (start - 4) + [7, 8, 9, 10] + [7, 8, 9]])
    looping = torch.tensor([[5] * start + [7, 8, 9, 10] + [7, 8, 9]])
    assert block(quoting, torch.zeros(1, 20))[0, 10] == 0
    assert block(looping, torch.zeros(1, 20))[0, 10] == float("-inf")


def test_complete_ignores_the_fine_tuned_adapter(tiny_model_dir, monkeypatch):
    # The helper agents grade and coach the model being trained, so its fine-tuning must never change them.
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    model = LocalModel("tiny")
    before = model.complete("system", "hi there"), model.complete("system", "hi", digits=True)
    weights = model.adapter_bytes()
    model.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there hello there"}] * 40)
    assert model.adapter_bytes() != weights  # the adapter did learn something (its greedy reply may not show it)
    assert (model.complete("system", "hi there"), model.complete("system", "hi", digits=True)) == before


def test_digits_stop_on_any_end_of_turn_token(tiny_model_dir, monkeypatch):
    # Chat models often end a turn on a token other than the tokenizer's EOS (Llama 3, Gemma): listed in generation_config.
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    model = LocalModel("tiny")
    assert model._end_ids() == [2]
    model.model.generation_config.eos_token_id = [4, 2]
    assert model._end_ids() == [2, 4]


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="needs Apple silicon"))])
def test_base_weights_are_bfloat16_on_a_gpu_and_float32_on_cpu(tiny_model_dir, tmp_path, monkeypatch, device):
    # bfloat16 halves the base model's memory and speeds a GPU up; CPUs are slow at it. The LoRA adapter stays float32.
    path = tmp_path / "tiny"
    shutil.copytree(tiny_model_dir, path)
    local_model.AutoModelForCausalLM.from_pretrained(path).to(torch.bfloat16).save_pretrained(path)  # a bf16 checkpoint
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: device == "mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = LocalModel("tiny")
    params = dict(model.model.named_parameters())
    base = torch.float32 if device == "cpu" else torch.bfloat16
    assert {p.dtype for n, p in params.items() if "lora_" not in n} == {base}
    assert {p.dtype for n, p in params.items() if "lora_" in n} == {torch.float32}
    assert math.isfinite(model.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 2))
    assert isinstance(model.generate("be helpful", "hi", max_new_tokens=3), str)


# --- quantizing ---

def test_quantize_model_links_a_copy_as_private_as_its_base(models_dir):
    local_model.save_upload(make_zip(MODEL), "mine", 1)
    assert local_model.quantize_model("mine", 1) == "mine-int8"
    src, dest = models_dir / "mine", models_dir / "mine-int8"
    assert os.path.samefile(src / "config.json", dest / "config.json")  # hard links: no extra disk
    assert local_model._meta(src) == {"source": "upload", "owner": 1}  # its meta is its own, not a link to this
    assert local_model._meta(dest) == {"source": "quantized", "owner": 1, "base": "mine", "private": True}
    assert {"name": "mine-int8", "source": "quantized"} in local_model.list_local_models(1)
    assert all(m["name"] != "mine-int8" for m in local_model.list_local_models(2))
    with pytest.raises(FileExistsError):
        local_model.quantize_model("mine", 2)  # not theirs to reuse
    assert local_model.quantize_model("mine", 1) == "mine-int8"  # another agent on it reuses the copy
    local_model.delete_model("mine", 1)
    assert (dest / "config.json").is_file()  # outlives its base
    with pytest.raises(ValueError, match="already quantized"):
        local_model.quantize_model("mine-int8", 1)


def test_quantized_hub_model_is_shared(models_dir, monkeypatch):
    shared = models_dir / "org--m"
    shared.mkdir(parents=True)
    (shared / "config.json").write_text("{}")
    (shared / "trainform.json").write_text(json.dumps({"source": "hub", "owner": 1}))
    assert local_model.quantize_model("org--m", 1) == "org--m-int8"
    assert {"name": "org--m-int8", "source": "quantized"} in local_model.list_local_models(2)
    assert local_model.quantize_model("org--m", 2) == "org--m-int8"  # another user reuses it
    meta = json.loads((models_dir / "org--m-int8" / "trainform.json").read_text())
    del meta["private"]
    (models_dir / "org--m-int8" / "trainform.json").write_text(json.dumps(meta))
    assert all(m["name"] != "org--m-int8" for m in local_model.list_local_models(2))  # no flag: private
    (models_dir / "other").mkdir()
    (models_dir / "other-int8").mkdir()  # a name someone else holds
    with pytest.raises(FileExistsError):
        local_model.quantize_model("other", 1)


def test_quantized_model_loads_int8_close_to_the_original_and_still_trains(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path)
    shutil.copytree(tiny_model_dir, tmp_path / "tiny")
    local_model.quantize_model("tiny", 1)
    full, quantized = LocalModel("tiny"), LocalModel("tiny-int8")
    linears = [m for m in quantized.model.modules() if isinstance(m, local_model.Int8Linear)]
    assert linears and all(m.weight.dtype == torch.int8 for m in linears)
    assert quantized.model.get_output_embeddings().weight.is_floating_point()
    ids = torch.tensor([[3, 4, 5]], device=full.device)
    logits = lambda m: m.model(input_ids=ids).logits.float()  # noqa: E731
    assert torch.allclose(logits(full), logits(quantized), atol=0.05)
    lora = lambda: {n: p.detach().cpu().clone() for n, p in quantized.model.named_parameters() if "lora_" in n}  # noqa: E731
    before = lora()
    quantized.sft([{"prompt": "be helpful", "input": "hi", "output": "hello there"}] * 2)
    assert any(not p.equal(before[n]) for n, p in lora().items())
