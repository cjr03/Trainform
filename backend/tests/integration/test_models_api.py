import io
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from app import local_model

MODEL = {"config.json": "{}", "model.safetensors": "w"}


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path / "models")
    return tmp_path / "models"


def zipped(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, data in files.items():
            zf.writestr(path, data)
    return buf.getvalue()


def corrupt_zip():
    data = bytearray(zipped({"config.json": "{}", "model.safetensors": "weights-data"}))
    data[data.find(b"weights-data")] ^= 1  # breaks the stored member's CRC
    return bytes(data)


def empty(path):
    return not path.exists() or not any(path.iterdir())


def upload(client, name, data):
    return client.post("/api/hub/upload", data={"name": name}, files={"file": ("m.zip", data, "application/zip")})


@pytest.mark.parametrize("method,path", [("get", "/api/hub/models"), ("post", "/api/hub/download"),
                                         ("post", "/api/hub/upload"), ("get", "/api/models")])
def test_routes_require_auth(client, method, path):
    assert getattr(client, method)(path).status_code == 401


def test_hub_models(authed_client, monkeypatch):
    seen = {}

    def fake(search, sort):
        seen.update(search=search, sort=sort)
        return [{"id": "org/a", "downloads": 1, "likes": 0, "last_modified": None}]
    monkeypatch.setattr(local_model, "list_hub_models", fake)
    r = authed_client.get("/api/hub/models", params={"search": "llama", "sort": "lastModified"})
    assert r.status_code == 200
    assert r.json() == [{"id": "org/a", "downloads": 1, "likes": 0, "last_modified": None}]
    assert seen == {"search": "llama", "sort": "lastModified"}
    authed_client.get("/api/hub/models")
    assert seen == {"search": "", "sort": "downloads"}


def test_hub_models_bad_sort(authed_client):
    assert authed_client.get("/api/hub/models", params={"sort": "likes"}).status_code in (400, 422)


def test_hub_models_network_error_is_502(authed_client, monkeypatch):
    def boom(*a):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(local_model, "list_hub_models", boom)
    assert authed_client.get("/api/hub/models").status_code == 502


def test_hub_download(authed_client, monkeypatch):
    monkeypatch.setattr(local_model, "download_model", lambda repo_id: repo_id.replace("/", "--"))
    r = authed_client.post("/api/hub/download", json={"repo_id": "org/tiny"})
    assert r.status_code == 201
    assert r.json() == {"name": "org--tiny", "source": "hub"}


def test_hub_download_bad_id(authed_client):
    r = authed_client.post("/api/hub/download", json={"repo_id": "../etc"})
    assert r.status_code == 400
    assert r.json()["detail"]


def test_hub_download_not_found(authed_client, monkeypatch):
    def missing(repo_id):
        raise ValueError(f"Model {repo_id} not found")
    monkeypatch.setattr(local_model, "download_model", missing)
    r = authed_client.post("/api/hub/download", json={"repo_id": "org/missing"})
    assert r.status_code in (400, 404)
    assert "org/missing" in r.json()["detail"]


# snapshot_download reports an unreachable Hub as LocalEntryNotFoundError; model_info raises httpx errors.
@pytest.mark.parametrize("error", [LocalEntryNotFoundError("hub unreachable"), httpx.ConnectError("offline")])
def test_hub_download_network_error_is_502(authed_client, models_dir, monkeypatch, error):
    def unreachable(*a, **kw):
        raise error

    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=1)])
    monkeypatch.setattr(local_model, "HfApi", lambda: SimpleNamespace(model_info=lambda *a, **kw: info))
    monkeypatch.setattr(local_model, "snapshot_download", unreachable)
    assert authed_client.post("/api/hub/download", json={"repo_id": "org/tiny"}).status_code == 502
    assert empty(models_dir)
    monkeypatch.setattr(local_model, "HfApi", lambda: SimpleNamespace(model_info=unreachable))
    assert authed_client.post("/api/hub/download", json={"repo_id": "org/tiny"}).status_code == 502


def test_hub_download_over_upload_is_409(authed_client, monkeypatch):
    assert upload(authed_client, "org--x", zipped(MODEL)).status_code == 201
    monkeypatch.setattr(local_model, "snapshot_download", lambda *a, **kw: pytest.fail("must not download"))
    assert authed_client.post("/api/hub/download", json={"repo_id": "org/x"}).status_code == 409


def test_upload_and_list(authed_client, models_dir):
    assert authed_client.get("/api/models").json() == []
    r = upload(authed_client, "mine", zipped({"mine/config.json": "{}", "mine/model.safetensors": "w"}))
    assert r.status_code == 201
    assert r.json() == {"name": "mine", "source": "upload"}
    assert (models_dir / "mine" / "model.safetensors").exists()
    assert authed_client.get("/api/models").json() == [{"name": "mine", "source": "upload"}]


def test_upload_duplicate_is_409(authed_client):
    data = zipped(MODEL)
    assert upload(authed_client, "dup", data).status_code == 201
    assert upload(authed_client, "dup", data).status_code == 409


@pytest.mark.parametrize("name,data", [
    ("bad", b"not a zip"),
    ("bad", zipped({**MODEL, "pytorch_model.bin": "x"})),
    ("bad", zipped({"model.safetensors": "w"})),
    ("bad", zipped({"config.json": "{}"})),
    ("bad", corrupt_zip()),
    ("bad", zipped({**MODEL, "a": "x", "a/b": "y"})),
    ("../bad", zipped(MODEL)),
], ids=["not-zip", "pickle", "no-config", "no-safetensors", "corrupt", "path-collision", "bad-name"])
def test_upload_invalid_is_400(authed_client, models_dir, name, data):
    r = upload(authed_client, name, data)
    assert r.status_code == 400
    assert r.json()["detail"]
    assert empty(models_dir)
