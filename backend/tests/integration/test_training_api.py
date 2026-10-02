import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import training
from app.db import engine
from app.main import app
from app.models import TrainedModel, Training, User

BODY = {"base_model": "tiny", "prompt": "Be helpful.", "rubric": ["  Clarity ", "Tone"], "scenarios": 2, "rounds": 2}


class FakeModel:
    def __init__(self, name):
        self.trained = False

    def generate(self, system_prompt, conversation):
        return "Hello!"

    def sft(self, examples):
        self.trained = True
        return 0.1

    def adapter_bytes(self):
        return b"PK\x03\x04trained" if self.trained else b"PK\x03\x04"


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    monkeypatch.setattr(training, "list_local_models", lambda: [{"name": "tiny", "source": "upload"}])
    monkeypatch.setattr(training, "LocalModel", FakeModel)
    scores = iter([6, 8, 9, 9])
    llms = []

    def generate_personas(llm, prompt, n, rubric=None):
        llms.append(llm)
        return [{"attributes": [{"criterion": "Clarity", "rating": k}], "prompt": f"persona {k}"} for k in range(1, n + 1)]

    def score_conversation(llm, conv, rubric):
        llms.append(llm)
        return [{"criterion": "Clarity", "score": next(scores)}, {"criterion": "Tone", "score": 8}]

    monkeypatch.setattr(training.agents, "generate_personas", generate_personas)
    monkeypatch.setattr(training.agents, "persona_respond", lambda llm, attributes, context: "Hi?")
    monkeypatch.setattr(training.agents, "primary_respond", lambda generate, prompt, conv: generate(prompt, conv))
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "generate_improved_prompt", lambda llm, prompt, conv, scores: prompt + "!")
    return llms


def other_users_training(db):
    other = User(email="bob@example.com", username="bob", password_hash="x")
    db.add(other)
    db.commit()
    t = Training(user_id=other.id, base_model="tiny", initial_prompt="Secret.", rubric=["Clarity"],
                 scenarios_per_round=1, rounds=1)
    db.add(t)
    db.commit()
    return t


def test_create_runs_training_in_background(authed_client, fakes):
    r = authed_client.post("/api/trainings", json=BODY)
    assert r.status_code == 201
    created = r.json()
    assert created["status"] == "pending"
    assert created["rubric"] == ["Clarity", "Tone"]
    assert (created["latest_prompt"], created["score_windows"], created["best_score"]) == ("Be helpful.", [], None)

    d = authed_client.get(f"/api/trainings/{created['id']}").json()
    assert set(d) == {"id", "status", "error", "base_model", "initial_prompt", "rubric", "rounds",
                      "scenarios_per_round", "current_round", "scenarios_done", "total_scenarios", "latest_prompt",
                      "score_windows", "best_model_id", "best_score", "created_at"}
    progress = (d["status"], d["error"], d["current_round"], d["scenarios_done"], d["total_scenarios"])
    assert progress == ("completed", None, 2, 4, 4)
    assert d["latest_prompt"] == "Be helpful.!!"
    # Scenario means: 7.0, 8.0, 8.5, 8.5 -> rounds 7.5 and 8.5.
    assert d["score_windows"] == [{"label": "n", "runs": 4, "score": 8.0}, {"label": "n/2", "runs": 2, "score": 8.5},
                                  {"label": "n/4", "runs": 1, "score": 8.5}, {"label": "last", "runs": 1, "score": 8.5}]
    assert d["best_score"] == 8.5
    # The background run uses the requesting user's Claude key.
    assert len(fakes) == 6 and all(llm.args == ("server-key",) for llm in fakes)

    rounds = authed_client.get(f"/api/trainings/{d['id']}/rounds").json()
    assert [(x["round"], x["mean_score"], x["scenarios"]) for x in rounds] == [(1, 7.5, 2), (2, 8.5, 2)]
    assert rounds[1]["model_id"] == d["best_model_id"] and rounds[0]["model_id"] != d["best_model_id"]

    scenarios = authed_client.get(f"/api/trainings/{d['id']}/rounds/2/scenarios").json()
    assert [s["scenario_number"] for s in scenarios] == [1, 2]
    assert scenarios[0] == {
        "id": scenarios[0]["id"], "scenario_number": 1,
        "persona": {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"},
        "prompt": "Be helpful.!!", "input": "Hi?", "output": "Hello!",
        "scores": [{"criterion": "Clarity", "score": 9}, {"criterion": "Tone", "score": 8}], "score": 8.5,
    }
    assert authed_client.get(f"/api/trainings/{d['id']}/rounds/3/scenarios").json() == []


def test_list_newest_first(authed_client, fakes, db, user):
    first = authed_client.post("/api/trainings", json=BODY).json()
    db.add(Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                    rounds=1))
    db.commit()
    other_users_training(db)
    listed = authed_client.get("/api/trainings").json()
    assert [t["id"] for t in listed] == [first["id"] + 1, first["id"]]
    assert set(listed[0]) == {"id", "base_model", "status", "rounds", "scenarios_per_round", "current_round",
                              "scenarios_done", "created_at", "best_score"}
    assert (listed[0]["status"], listed[0]["best_score"]) == ("pending", None)
    assert (listed[1]["status"], listed[1]["best_score"]) == ("completed", 8.5)


def test_downloads(authed_client, fakes, db, user):
    t = authed_client.post("/api/trainings", json=BODY).json()
    r = authed_client.get(f"/api/trainings/{t['id']}/download/prompt")
    assert r.status_code == 200
    assert r.text == "Be helpful.!!"
    assert r.headers["content-type"].startswith("text/plain")
    assert r.headers["content-disposition"] == f'attachment; filename="training-{t["id"]}-prompt.txt"'

    r = authed_client.get(f"/api/trainings/{t['id']}/download/weights")
    assert r.status_code == 200
    assert r.content == b"PK\x03\x04trained"
    assert r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="training-{t["id"]}-lora.zip"'

    pending = Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                       rounds=1)
    db.add(pending)
    db.commit()
    for kind in ("prompt", "weights"):
        r = authed_client.get(f"/api/trainings/{pending.id}/download/{kind}")
        assert (r.status_code, r.json()) == (404, {"detail": "Training has no trained model yet"})


@pytest.mark.parametrize("change", [
    {"prompt": ""}, {"prompt": "x" * 20001}, {"rubric": []}, {"rubric": ["   "]}, {"rubric": ["c"] * 31},
    {"rubric": ["x" * 501]}, {"scenarios": 0}, {"scenarios": 101}, {"rounds": 0}, {"rounds": 51}, {"base_model": None},
])
def test_create_validation(authed_client, fakes, db, change):
    assert authed_client.post("/api/trainings", json=BODY | change).status_code == 422
    assert db.query(Training).count() == 0


def test_create_accepts_limits(authed_client, fakes, monkeypatch):
    monkeypatch.setattr(training, "run_training", lambda *args: None)
    body = BODY | {"prompt": "x" * 20000, "rubric": ["c" * 500] * 30, "scenarios": 100, "rounds": 50}
    assert authed_client.post("/api/trainings", json=body).status_code == 201


def test_request_releases_its_db_connection_before_training(authed_client, fakes, monkeypatch):
    checked_out = []
    monkeypatch.setattr(training, "run_training", lambda *args: checked_out.append(engine.pool.checkedout()))
    assert authed_client.post("/api/trainings", json=BODY).status_code == 201
    assert checked_out == [0]


def test_one_training_in_progress_per_user(authed_client, fakes, monkeypatch, db, user):
    monkeypatch.setattr(training, "run_training", lambda *args: None)
    other_users_training(db)  # pending, but someone else's
    first = authed_client.post("/api/trainings", json=BODY)
    assert first.status_code == 201
    r = authed_client.post("/api/trainings", json=BODY)
    assert (r.status_code, r.json()) == (409, {"detail": "You already have a training in progress."})
    # The database enforces it, so two concurrent requests can't both pass a check-then-insert.
    with pytest.raises(IntegrityError):
        db.add(Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                        rounds=1, status="running"))
        db.commit()
    db.rollback()
    db.get(Training, first.json()["id"]).status = "failed"
    db.commit()
    assert authed_client.post("/api/trainings", json=BODY).status_code == 201


def test_global_cap_on_concurrent_trainings(authed_client, fakes, monkeypatch, db, user):
    monkeypatch.setattr(training, "_SLOTS", threading.BoundedSemaphore(1))
    assert training._SLOTS.acquire(blocking=False)  # another user's run holds the only slot
    r = authed_client.post("/api/trainings", json=BODY)
    assert (r.status_code, r.json()) == (429, {"detail": "The server is busy with other trainings; try again later."})
    assert db.query(Training).count() == 0
    training._SLOTS.release()

    assert authed_client.post("/api/trainings", json=BODY).status_code == 201  # runs to completion here
    assert authed_client.post("/api/trainings", json=BODY).status_code == 201  # so its slot was released
    db.add(Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                    rounds=1))
    db.commit()
    assert authed_client.post("/api/trainings", json=BODY).status_code == 409
    assert training._SLOTS.acquire(blocking=False)  # a rejected request gives its slot back


def test_unknown_base_model(authed_client, fakes, db):
    r = authed_client.post("/api/trainings", json=BODY | {"base_model": "missing"})
    assert (r.status_code, r.json()) == (400, {"detail": "Unknown base model: missing"})
    assert db.query(Training).count() == 0


def test_create_without_claude_key(authed_client, fakes, monkeypatch, db):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    r = authed_client.post("/api/trainings", json=BODY)
    assert r.status_code == 400
    assert "Claude" in r.json()["detail"]
    assert db.query(Training).count() == 0


def test_other_users_training_is_404(authed_client, db):
    t = other_users_training(db)
    db.add(TrainedModel(training_id=t.id, round=1, prompt="Secret.", weights=b"PK", mean_score=9.0))
    db.commit()
    t.best_model_id = db.query(TrainedModel).one().id
    db.commit()
    for path in ("", "/rounds", "/rounds/1/scenarios", "/download/prompt", "/download/weights"):
        assert authed_client.get(f"/api/trainings/{t.id}{path}").status_code == 404
    assert authed_client.get("/api/trainings/999999").status_code == 404
    assert authed_client.get("/api/trainings").json() == []


def test_routes_require_a_user(client):
    assert client.post("/api/trainings", json=BODY).status_code == 401
    for path in ("", "/1", "/1/rounds", "/1/rounds/1/scenarios", "/1/download/prompt", "/1/download/weights"):
        assert client.get(f"/api/trainings{path}").status_code == 401


def test_restart_fails_orphaned_trainings(db, user):
    other_users_training(db)  # pending; one active training per user, so the running one is the second user's
    for status in ("running", "completed", "failed"):
        db.add(Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                        rounds=1, status=status))
    db.commit()
    with TestClient(app):
        pass
    db.expire_all()
    rows = db.query(Training).order_by(Training.id).all()
    assert [(t.status, t.error) for t in rows] == [
        ("failed", "Interrupted by a server restart"), ("failed", "Interrupted by a server restart"),
        ("completed", None), ("failed", None),
    ]


def test_startup_adds_the_one_active_training_index_to_an_existing_database(db):
    db.execute(text("DROP INDEX IF EXISTS one_active_training_per_user"))
    db.commit()
    with TestClient(app):
        pass
    assert db.scalar(text("SELECT 1 FROM pg_indexes WHERE indexname = 'one_active_training_per_user'"))
