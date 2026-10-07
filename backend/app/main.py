from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text, update

from app import agent_profiles, agents, auth, local_model, models, social, training  # noqa: F401  (models: the tables)
from app.db import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth._secret()  # fail fast: tokens and stored Claude keys both need TRAINFORM_SECRET
    # ponytail: create_all instead of migrations; add Alembic once a deployed schema has to change.
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        # Added after the first release; create_all never alters existing tables.
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version integer NOT NULL DEFAULT 0"))
        conn.execute(text("ALTER TABLE scenarios ADD COLUMN IF NOT EXISTS target text"))
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar bytea, "
                          "ADD COLUMN IF NOT EXISTS avatar_type varchar(32)"))
        conn.execute(text("ALTER TABLE agents ADD COLUMN IF NOT EXISTS image text, "
                          "ADD COLUMN IF NOT EXISTS overall integer"))
        conn.execute(text("ALTER TABLE trainings ADD COLUMN IF NOT EXISTS next_prompt text, "
                          "ADD COLUMN IF NOT EXISTS personas json, ADD COLUMN IF NOT EXISTS adapter bytea, "
                          "ADD COLUMN IF NOT EXISTS helper varchar(16), "
                          "ADD COLUMN IF NOT EXISTS agent_id integer REFERENCES agents(id) ON DELETE CASCADE, "
                          "ADD COLUMN IF NOT EXISTS battle json"))
        conn.execute(text("ALTER TABLE trained_models ALTER COLUMN mean_score DROP NOT NULL"))
        # Trainings from before agents existed belong to the seed agent, made for their user (on their latest
        # training's model) if they have no agent yet. No one else gets one.
        latest_model = "SELECT base_model FROM trainings t WHERE t.user_id = u.id ORDER BY t.id DESC LIMIT 1"
        conn.execute(text(f"INSERT INTO agents (user_id, name, base_model, prompt) "
                          f"SELECT u.id, :name, ({latest_model}), :prompt FROM users u "
                          f"WHERE NOT EXISTS (SELECT 1 FROM agents a WHERE a.user_id = u.id) "
                          f"AND EXISTS (SELECT 1 FROM trainings t WHERE t.user_id = u.id AND t.agent_id IS NULL)"),
                     models.SEED_AGENT)
        conn.execute(text("UPDATE trainings t SET agent_id = a.id FROM agents a "
                          "WHERE t.agent_id IS NULL AND a.user_id = t.user_id AND a.name = :name"),
                     {"name": models.SEED_AGENT["name"]})
        # Signup used to give everyone the seed agent: drop those still as seeded (same name and prompt, never
        # trained), so no one keeps an untrained Bike Repair Agent they never asked for.
        conn.execute(text("DELETE FROM agents a WHERE a.name = :name AND a.prompt = :prompt AND a.weights IS NULL "
                          "AND NOT EXISTS (SELECT 1 FROM trainings t WHERE t.agent_id = a.id)"), models.SEED_AGENT)
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

for module in (auth, agents, agent_profiles, local_model, social, training):
    app.include_router(module.router, prefix="/api")


@app.middleware("http")
async def cap_upload_size(request: Request, call_next):
    """Reject oversized uploads before Starlette spools the multipart body to disk."""
    cap = {"/api/hub/upload": local_model.MAX_UPLOAD_BYTES,
           "/api/me/avatar": social.MAX_AVATAR_BYTES + 4096}.get(request.url.path)  # + multipart framing
    if cap and request.method in ("POST", "PUT"):
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length is required"}, 411)
        if int(length) > cap:
            return JSONResponse({"detail": "Upload is too large"}, 413)
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok"}
