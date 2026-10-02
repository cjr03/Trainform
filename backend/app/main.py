from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import agents, auth, local_model, models, training  # noqa: F401  (models registers the tables)
from app.db import Base, engine


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # ponytail: create_all instead of migrations; add Alembic once a deployed schema has to change.
    Base.metadata.create_all(engine)
    yield


app = FastAPI(title="Trainform", lifespan=lifespan)

for module in (auth, agents, local_model, training):
    app.include_router(module.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
