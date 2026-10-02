import os

# Tests always run against the test database, never DATABASE_URL from the environment.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://localhost/trainform_test")
os.environ.setdefault("TRAINFORM_SECRET", "test-secret")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app import models  # noqa: E402
from app.auth import current_user  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def _schema():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db(_schema):
    with SessionLocal() as session:
        yield session
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def client(db):
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def user(db):
    u = models.User(email="ada@example.com", username="ada", password_hash="not-a-real-hash")
    db.add(u)
    db.commit()
    return u


@pytest.fixture
def authed_client(client, user):
    """A client whose requests are authenticated as `user` (bypasses token auth)."""
    app.dependency_overrides[current_user] = lambda: user
    return client
