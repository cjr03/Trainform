from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import update

from app import agents, auth, local_model, models, training  # noqa: F401  (models registers the tables)
from app.db import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth._secret()  # fail fast: tokens and stored Claude keys both need TRAINFORM_SECRET
    # ponytail: create_all instead of migrations; add Alembic once a deployed schema has to change.
    Base.metadata.create_all(engine)
    # Training runs are background threads of this process, so any left pending/running died with the last one.
    with engine.begin() as conn:
        conn.execute(update(models.Training).where(models.Training.status.in_(("pending", "running")))
                     .values(status="failed", error="Interrupted by a server restart"))
    yield


app = FastAPI(title="Trainform", lifespan=lifespan)

for module in (auth, agents, local_model, training):
    app.include_router(module.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
