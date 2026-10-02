"""Entry point for the macOS app: embedded Postgres + the API + the built frontend, all on 127.0.0.1.

The launcher runs `python -m app.desktop --port N --pg-bin <dir> --static-dir <dir>`. app.main is imported only
after DATABASE_URL and friends are set, because app.db builds its engine at import time.
"""

import argparse
import os
import secrets
import shlex
import subprocess
from pathlib import Path
from urllib.parse import quote

import psycopg
import uvicorn
from fastapi import HTTPException
from fastapi.responses import FileResponse

DEFAULT_DATA_DIR = Path.home() / "Library" / "Application Support" / "Trainform"
SOCKET_PATH_LIMIT = 103  # macOS sockaddr_un.sun_path


def ensure_secret(data_dir: Path) -> str:
    """The app's TRAINFORM_SECRET, generated once; losing it logs everyone out and orphans linked Claude keys."""
    path = data_dir / "secret"
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    return path.read_text().strip()


def socket_dir(data_dir: Path) -> Path:
    path = data_dir / "run"
    if len(str(path / ".s.PGSQL.5432")) > SOCKET_PATH_LIMIT:
        raise RuntimeError(f"Postgres socket path under {path} is too long for macOS; set TRAINFORM_DATA_DIR "
                           "to a shorter directory")
    return path


def database_url(sock: Path) -> str:
    return f"postgresql+psycopg://trainform@/trainform?host={quote(str(sock), safe='/')}"


def _pg(pg_bin: Path, *args: str) -> None:
    subprocess.run([str(pg_bin / args[0]), *args[1:]], check=True, stdout=subprocess.DEVNULL)


def start_postgres(pg_bin: Path, data_dir: Path) -> Path:
    """Start the app's private cluster (socket only, no TCP) and make sure the database exists."""
    pgdata, sock = data_dir / "postgres", socket_dir(data_dir)
    if not (pgdata / "PG_VERSION").exists():
        # Trust auth is safe here: the server listens only on a socket inside a 0700 directory.
        _pg(pg_bin, "initdb", "-D", str(pgdata), "-U", "trainform", "-A", "trust", "-E", "UTF8", "--no-locale")
    sock.mkdir(mode=0o700, exist_ok=True)
    sock.chmod(0o700)
    _pg(pg_bin, "pg_ctl", "-D", str(pgdata), "-l", str(data_dir / "postgres.log"), "-w", "-t", "60",
        "-o", f"-k {shlex.quote(str(sock))} -c listen_addresses=''", "start")
    with psycopg.connect(host=str(sock), user="trainform", dbname="postgres", autocommit=True) as conn:
        if not conn.execute("select 1 from pg_database where datname = 'trainform'").fetchone():
            conn.execute("create database trainform")
    return sock


def stop_postgres(pg_bin: Path, data_dir: Path) -> None:
    subprocess.run([str(pg_bin / "pg_ctl"), "-D", str(data_dir / "postgres"), "-m", "fast", "-w", "stop"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mount_frontend(app, static_dir: Path) -> None:
    """Serve the built SPA; unknown non-API paths get index.html so client-side routes survive a reload."""
    root = static_dir.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path.split("/")[0] == "api":
            raise HTTPException(404)
        file = (root / path).resolve()
        return FileResponse(file if file.is_file() and file.is_relative_to(root) else root / "index.html")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.desktop")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pg-bin", type=Path, required=True)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("TRAINFORM_DATA_DIR", DEFAULT_DATA_DIR)))
    args = parser.parse_args(argv)

    args.data_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.data_dir.chmod(0o700)
    os.environ["TRAINFORM_SECRET"] = ensure_secret(args.data_dir)
    os.environ["MODELS_DIR"] = str(args.data_dir / "models")
    sock = start_postgres(args.pg_bin, args.data_dir)
    try:
        os.environ["DATABASE_URL"] = database_url(sock)
        from app.main import app

        mount_frontend(app, args.static_dir)
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    finally:
        stop_postgres(args.pg_bin, args.data_dir)


if __name__ == "__main__":
    main()
