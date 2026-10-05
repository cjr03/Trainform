from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text, update

from app import agents, auth, local_model, models, training  # noqa: F401  (models registers the tables)
from app.db import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth._secret()  # fail fast: tokens and stored Claude keys both need TRAINFORM_SECRET
    # ponytail: create_all instead of migrations; add Alembic once a deployed schema has to change.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Added after the first release; create_all never alters existing tables.
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version integer NOT NULL DEFAULT 0"))
        conn.execute(text("ALTER TABLE trainings ADD COLUMN IF NOT EXISTS next_prompt text, "
                          "ADD COLUMN IF NOT EXISTS personas json, ADD COLUMN IF NOT EXISTS adapter bytea, "
                          "ADD COLUMN IF NOT EXISTS helper varchar(16)"))
        # Training runs are background threads of this process, so any left active died with the last one. Their
        # progress is saved, so they wait, paused, for the user to resume them.
        conn.execute(update(models.Training).where(models.Training.status.in_(models.ACTIVE))
                     .values(status="paused", error=None))
        # Prompts saved before agents._prompt dropped the tags and colons the prompt generator put around them.
        edge_tags = r"^([\s:]|<[^<>\n]+>)+|([\s:]|<[^<>\n]+>)+$"
        for table, column in (("trainings", "next_prompt"), ("scenarios", "prompt"), ("trained_models", "prompt")):
            conn.execute(text(f"UPDATE {table} SET {column} = regexp_replace({column}, :tags, '', 'g') "
                              f"WHERE {column} ~ :tags"), {"tags": edge_tags})
        # Recreated so an index from before "pausing" existed covers it too.
        conn.execute(text("DROP INDEX IF EXISTS one_active_training_per_user"))
        models.ONE_ACTIVE_TRAINING.create(conn)
    yield


app = FastAPI(title="Trainform", lifespan=lifespan)

for module in (auth, agents, local_model, training):
    app.include_router(module.router, prefix="/api")


@app.middleware("http")
async def cap_upload_size(request: Request, call_next):
    """Reject oversized model uploads before Starlette spools the multipart body to disk."""
    if request.url.path == "/api/hub/upload":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length is required"}, 411)
        if int(length) > local_model.MAX_UPLOAD_BYTES:
            return JSONResponse({"detail": "Upload is too large"}, 413)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok"}
