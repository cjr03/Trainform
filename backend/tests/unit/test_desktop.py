import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

import psycopg
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from app import desktop


def test_secret_is_created_private_and_stable(tmp_path):
    first = desktop.ensure_secret(tmp_path)
    assert len(first) >= 32
    assert stat.S_IMODE((tmp_path / "secret").stat().st_mode) == 0o600
    assert desktop.ensure_secret(tmp_path) == first


def test_socket_dir_rejects_paths_macos_cannot_bind():
    assert desktop.socket_dir(Path("/tmp/tf")) == Path("/tmp/tf/run")
    with pytest.raises(RuntimeError, match="too long"):
        desktop.socket_dir(Path("/tmp/" + "x" * 100))


def test_database_url_survives_spaces_in_the_socket_path():
    url = make_url(desktop.database_url(Path("/Users/a/Library/Application Support/Trainform/run")))
    assert url.drivername == "postgresql+psycopg"
    assert (url.username, url.database) == ("trainform", "trainform")
    assert url.query["host"] == "/Users/a/Library/Application Support/Trainform/run"


@pytest.fixture
def frontend(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside")
    app = FastAPI()
    app.get("/api/health")(lambda: {"status": "ok"})
    desktop.mount_frontend(app, dist)
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/train", "/inspect/3"])
def test_spa_routes_serve_index(frontend, path):
    r = frontend.get(path)
    assert r.status_code == 200
    assert r.text == "<div id=root></div>"


def test_static_files_are_served(frontend):
    assert frontend.get("/assets/app.js").text == "console.log(1)"
    assert frontend.get("/favicon.svg").text == "<svg/>"


def test_api_routes_are_not_shadowed(frontend):
    assert frontend.get("/api/health").json() == {"status": "ok"}
    assert frontend.get("/api/missing").status_code == 404
    assert frontend.get("/api").status_code == 404


def test_files_outside_the_frontend_are_not_served(frontend):
    assert "outside" not in frontend.get("/..%2Fsecret.txt").text
    assert "outside" not in frontend.get("/assets/..%2F..%2Fsecret.txt").text


def test_main_wires_env_and_always_stops_postgres(tmp_path, monkeypatch):
    for key in ("DATABASE_URL", "TRAINFORM_SECRET", "MODELS_DIR"):
        monkeypatch.setenv(key, os.environ.get(key, ""))  # restored after the test
    calls = []
    monkeypatch.setattr(desktop, "start_postgres", lambda pg_bin, data_dir: calls.append("start") or Path("/s"))
    monkeypatch.setattr(desktop, "stop_postgres", lambda pg_bin, data_dir: calls.append("stop"))
    monkeypatch.setattr(desktop, "mount_frontend", lambda app, static_dir: calls.append(("mount", static_dir)))

    def crash(app, host, port):
        calls.append(("serve", host, port))
        raise RuntimeError("boom")

    monkeypatch.setattr(desktop.uvicorn, "run", crash)
    data = tmp_path / "data"
    with pytest.raises(RuntimeError, match="boom"):
        desktop.main(["--port", "4321", "--pg-bin", "/pg/bin", "--static-dir", "/fe", "--data-dir", str(data)])
    assert calls == ["start", ("mount", Path("/fe")), ("serve", "127.0.0.1", 4321), "stop"]
    assert os.environ["DATABASE_URL"] == desktop.database_url(Path("/s"))
    assert os.environ["TRAINFORM_SECRET"] == (data / "secret").read_text().strip()
    assert os.environ["MODELS_DIR"] == str(data / "models")
    assert stat.S_IMODE(data.stat().st_mode) == 0o700


def _pg_bin():
    found = os.environ.get("PG_BIN") or (shutil.which("pg_config") and subprocess.run(
        ["pg_config", "--bindir"], capture_output=True, text=True).stdout.strip())
    for candidate in filter(None, [found, "/opt/homebrew/opt/postgresql@17/bin"]):
        if (Path(candidate) / "initdb").exists():
            return Path(candidate)
    pytest.skip("PostgreSQL server binaries (initdb, pg_ctl) not found")


def test_embedded_postgres_initialises_starts_and_restarts():
    pg_bin = _pg_bin()
    data = Path(tempfile.mkdtemp(prefix="tf", dir="/tmp"))  # short: macOS caps socket paths at 103 bytes
    try:
        for _ in range(2):  # first launch runs initdb, the second reuses the cluster and database
            sock = desktop.start_postgres(pg_bin, data)
            try:
                with psycopg.connect(host=str(sock), user="trainform", dbname="trainform") as conn:
                    assert conn.execute("select 1").fetchone() == (1,)
            finally:
                desktop.stop_postgres(pg_bin, data)
        assert stat.S_IMODE(sock.stat().st_mode) == 0o700
        assert not (data / "postgres" / "postmaster.pid").exists()
    finally:
        shutil.rmtree(data)
