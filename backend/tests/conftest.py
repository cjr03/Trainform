import os

# Tests always run against the test database, never DATABASE_URL from the environment.
os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://localhost/trainform_test")
os.environ.setdefault("TRAINFORM_SECRET", "test-secret")
os.environ.pop("TRAINFORM_FAKE_LLM", None)  # fake mode only where a test sets it

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from app import models  # noqa: E402
from app.auth import current_user  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402

WORDS = ["<unk>", "<pad>", "</s>", "system", "user", "assistant", ":", "be", "helpful", "hi", "hello", "there"]


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


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A 2-layer Llama with a word-level tokenizer, built offline, at <models dir>/tiny."""
    path = tmp_path_factory.mktemp("models") / "tiny"
    tok = Tokenizer(tokenizer_models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")
    tokenizer.chat_template = ("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}assistant: {% endif %}")
    tokenizer.save_pretrained(path)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=2, eos_token_id=2, pad_token_id=1, bos_token_id=None)
    LlamaForCausalLM(config).save_pretrained(path)
    return path
