import threading

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app import training
from app.db import engine
from app.main import app
from app.models import Scenario, TrainedModel, Training, User

BODY = {"base_model": "tiny", "prompt": "Be helpful.", "rubric": ["  Clarity ", "Tone"], "scenarios": 2, "rounds": 2}


class FakeModel:
    def __init__(self, name):
        self.trained = False

    def generate(self, system_prompt, conversation):
        # A revision ("?") and fine-tuning each change the reply, and fakes.score rewards both.
        return "Hello!" + (" Sure?" if system_prompt.endswith("?") else "") + (" Trained." if self.trained else "")

    def sft(self, examples, **_):
        self.trained = True
        return 0.1

    def adapter_bytes(self):
        return b"PK\x03\x04trained" if self.trained else b"PK\x03\x04"

    def load_adapter(self, data):
        self.trained = data == b"PK\x03\x04trained"


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    monkeypatch.setattr(training, "list_local_models", lambda user_id: [{"name": "tiny", "source": "upload"}])
    monkeypatch.setattr(training, "LocalModel", FakeModel)
    llms = []

    def generate_personas(llm, prompt, n, rubric=None, last_scores=None):
        llms.append(llm)
        return [{"attributes": [{"criterion": "Clarity", "rating": k}], "prompt": f"persona {k}"} for k in range(1, n + 1)]

    def score_conversation(llm, conv, rubric):
        llms.append(llm)
        return [{"criterion": "Clarity", "score": 6 + 2 * ("Sure?" in conv) + ("Trained." in conv)},
                {"criterion": "Tone", "score": 8}]

    monkeypatch.setattr(training.agents, "generate_personas", generate_personas)
    monkeypatch.setattr(training.agents, "persona_respond", lambda llm, attributes, context: "Hi?")
    monkeypatch.setattr(training.agents, "primary_respond", lambda generate, prompt, conv: generate(prompt, conv))
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "teacher_reply", lambda llm, prompt, message, rubric: "Ideal.")
    monkeypatch.setattr(training.agents, "review_round", lambda llm, prompt, *a: prompt + "?")
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


def add_training(db, user, status):
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["C"], scenarios_per_round=1,
                 rounds=1, status=status, error="Claude rate limit reached" if status == "failed" else None)
    db.add(t)
    db.commit()
    return t


def set_status(db, t, status):
    db.query(Training).filter_by(id=t.id).update({"status": status})
    db.commit()


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
                      "score_windows", "best_model_id", "best_score", "best_prompt", "baseline_score", "created_at",
                      "helper"}
    progress = (d["status"], d["error"], d["current_round"], d["scenarios_done"], d["total_scenarios"], d["helper"])
    assert progress == ("completed", None, 2, 4, 4, None)
    # Round 2 tested the review's revision ("?") against the prompt on the same users and kept it: it scored higher.
    assert d["latest_prompt"] == "Be helpful.?"
    # Scenario means: 7.0, 7.0 (untrained), then 8.5, 8.5 (revised and fine-tuned).
    assert d["score_windows"] == [{"label": "n", "runs": 4, "score": 7.75}, {"label": "n/2", "runs": 2, "score": 8.5},
                                  {"label": "n/4", "runs": 1, "score": 8.5}, {"label": "last", "runs": 1, "score": 8.5}]
    # The anchor user scored 7.0 before training, 7.5 after round 1's SFT and 8.5 with round 2's revision.
    assert (d["baseline_score"], d["best_score"]) == (7.0, 8.5)
    # The background run uses the requesting user's Claude key.
    assert len(fakes) == 10 and all(llm.args == ("server-key",) for llm in fakes)

    rounds = authed_client.get(f"/api/trainings/{d['id']}/rounds").json()
    assert [(x["round"], x["mean_score"], x["scenarios"], x["eval_score"]) for x in rounds] == [
        (1, 7.0, 2, 7.5), (2, 8.5, 2, 8.5)]
    assert rounds[1]["model_id"] == d["best_model_id"] and rounds[0]["model_id"] != d["best_model_id"]

    scenarios = authed_client.get(f"/api/trainings/{d['id']}/rounds/2/scenarios").json()
    assert [s["scenario_number"] for s in scenarios] == [1, 2]
    assert scenarios[0] == {
        "id": scenarios[0]["id"], "scenario_number": 1,
        "persona": {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"},
        "prompt": "Be helpful.?", "input": "Hi?", "output": "Hello! Sure? Trained.",
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
                              "scenarios_done", "created_at", "best_score", "helper"}
    assert (listed[0]["status"], listed[0]["best_score"]) == ("pending", None)
    assert (listed[1]["status"], listed[1]["best_score"]) == ("completed", 8.5)  # on the anchor user


def test_downloads(authed_client, fakes, db, user):
    t = authed_client.post("/api/trainings", json=BODY).json()
    r = authed_client.get(f"/api/trainings/{t['id']}/download/prompt")
    assert r.status_code == 200
    assert r.text == "Be helpful.?"
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


def test_best_prompt_is_the_downloaded_prompt(authed_client, fakes, monkeypatch):
    # Round 1: practice 6, then its model scores 9 on the anchor. Round 2: the revision ties the prompt (rejected) and
    # its model scores 5 on the anchor, so round 1's model stays the best.
    scores = iter([6, 6, 9, 5, 5, 5, 5, 5])
    lock = threading.Lock()

    def score_conversation(llm, conv, rubric):
        with lock:
            score = next(scores)
        return [{"criterion": c, "score": score} for c in rubric]
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    created = authed_client.post("/api/trainings", json=BODY).json()
    assert created["best_prompt"] is None
    d = authed_client.get(f"/api/trainings/{created['id']}").json()
    assert (d["latest_prompt"], d["best_prompt"], d["best_score"], d["baseline_score"]) == (
        "Be helpful.", "Be helpful.", 9.0, 6.0)
    assert authed_client.get(f"/api/trainings/{d['id']}/download/prompt").text == d["best_prompt"]


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
    set_status(db, db.get(Training, first.json()["id"]), "pausing")  # still in progress until it has paused
    assert authed_client.post("/api/trainings", json=BODY).status_code == 409
    set_status(db, db.get(Training, first.json()["id"]), "paused")
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
    db.query(Training).update({"status": "failed"})
    db.commit()
    monkeypatch.setattr(training, "detail", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        authed_client.post("/api/trainings", json=BODY)
    assert training._SLOTS.acquire(blocking=False)  # a rejected or failed request gives its slot back


def test_unknown_base_model(authed_client, fakes, db):
    r = authed_client.post("/api/trainings", json=BODY | {"base_model": "missing"})
    assert (r.status_code, r.json()) == (400, {"detail": "Unknown base model: missing"})
    assert db.query(Training).count() == 0


def test_create_without_claude_key_uses_the_local_model(authed_client, fakes, monkeypatch, db):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    r = authed_client.post("/api/trainings", json=BODY)
    assert r.status_code == 201
    assert authed_client.get(f"/api/trainings/{r.json()['id']}").json()["status"] == "completed"
    assert fakes and all(getattr(llm, "local", False) for llm in fakes)


def test_create_with_the_chosen_helper(authed_client, fakes, monkeypatch, db):
    r = authed_client.post("/api/trainings", json=BODY | {"helper": "local"})  # even though Claude is linked
    assert (r.status_code, r.json()["helper"]) == (201, "local")
    assert authed_client.get(f"/api/trainings/{r.json()['id']}").json()["status"] == "completed"
    assert fakes and all(getattr(llm, "local", False) for llm in fakes)
    assert authed_client.get("/api/trainings").json()[0]["helper"] == "local"
    fakes.clear()
    r = authed_client.post("/api/trainings", json=BODY | {"helper": "claude"})
    assert (r.status_code, r.json()["helper"]) == (201, "claude")
    assert fakes and all(llm.args == ("server-key",) for llm in fakes)

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    r = authed_client.post("/api/trainings", json=BODY | {"helper": "claude"})
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page first."})
    assert authed_client.post("/api/trainings", json=BODY | {"helper": "gpt"}).status_code == 422
    assert db.query(Training).count() == 2


def test_resume_with_the_stored_helper(authed_client, fakes, monkeypatch, db, user):
    t = add_training(db, user, "paused")
    t.helper = "local"
    db.commit()
    assert authed_client.post(f"/api/trainings/{t.id}/resume").status_code == 200  # Claude is linked
    assert fakes and all(getattr(llm, "local", False) for llm in fakes)

    t = add_training(db, user, "paused")
    t.helper = "claude"
    db.commit()
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # unlinked since it started
    r = authed_client.post(f"/api/trainings/{t.id}/resume")
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page first."})
    db.expire_all()
    assert db.get(Training, t.id).status == "paused"


def test_other_users_training_is_404(authed_client, db):
    t = other_users_training(db)
    db.add(TrainedModel(training_id=t.id, round=1, prompt="Secret.", weights=b"PK", mean_score=9.0))
    db.commit()
    t.best_model_id = db.query(TrainedModel).one().id
    db.commit()
    for path in ("", "/rounds", "/rounds/1/scenarios", "/download/prompt", "/download/weights"):
        assert authed_client.get(f"/api/trainings/{t.id}{path}").status_code == 404
    assert authed_client.post(f"/api/trainings/{t.id}/pause").status_code == 404
    set_status(db, t, "paused")
    assert authed_client.post(f"/api/trainings/{t.id}/resume").status_code == 404
    assert authed_client.delete(f"/api/trainings/{t.id}").status_code == 404
    db.expire_all()
    assert db.get(Training, t.id).status == "paused"
    assert authed_client.get("/api/trainings/999999").status_code == 404
    assert authed_client.get("/api/trainings").json() == []


def test_routes_require_a_user(client):
    assert client.post("/api/trainings", json=BODY).status_code == 401
    for path in ("", "/1", "/1/rounds", "/1/rounds/1/scenarios", "/1/download/prompt", "/1/download/weights"):
        assert client.get(f"/api/trainings{path}").status_code == 401
    assert client.post("/api/trainings/1/pause").status_code == 401
    assert client.post("/api/trainings/1/resume").status_code == 401
    assert client.delete("/api/trainings/1").status_code == 401


def test_restart_pauses_orphaned_trainings(db, user):
    other_users_training(db)  # pending; one active training per user, so each active one is a different user's
    carol = User(email="carol@example.com", username="carol", password_hash="x")
    db.add(carol)
    db.commit()
    add_training(db, carol, "pausing")
    for status in ("running", "completed", "failed"):
        add_training(db, user, status)
    with TestClient(app):
        pass
    db.expire_all()
    rows = db.query(Training).order_by(Training.id).all()
    # Their runs died with the last process; their progress is saved, so the user can resume them.
    assert [(t.status, t.error) for t in rows] == [
        ("paused", None), ("paused", None), ("paused", None), ("completed", None),
        ("failed", "Claude rate limit reached"),
    ]


def test_startup_adds_the_one_active_training_index_to_an_existing_database(db):
    db.execute(text("DROP INDEX IF EXISTS one_active_training_per_user"))
    db.commit()
    with TestClient(app):
        pass
    assert db.scalar(text("SELECT 1 FROM pg_indexes WHERE indexname = 'one_active_training_per_user'"))


def test_startup_recreates_the_one_active_training_index_with_pausing(db):
    db.execute(text("DROP INDEX one_active_training_per_user"))
    db.execute(text("CREATE UNIQUE INDEX one_active_training_per_user ON trainings (user_id) "
                    "WHERE status IN ('pending', 'running')"))  # as the first release made it
    db.commit()
    with TestClient(app):
        pass
    assert "pausing" in db.scalar(text("SELECT indexdef FROM pg_indexes "
                                       "WHERE indexname = 'one_active_training_per_user'"))


def test_startup_adds_the_resume_columns_to_an_existing_database(db):
    db.execute(text("ALTER TABLE trainings DROP COLUMN next_prompt, DROP COLUMN personas, DROP COLUMN adapter, "
                    "DROP COLUMN helper"))
    db.commit()
    with TestClient(app):
        pass
    columns = db.scalars(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'trainings'"))
    assert {"next_prompt", "personas", "adapter", "helper"} <= set(columns)


def test_startup_drops_tags_around_saved_prompts(db, user):
    tagged = ":\n<system_prompt>\n<prompt>:\nBe kind.\n</current_prompt>\n</system_prompt>"
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["a"], scenarios_per_round=1,
                 rounds=1, status="paused", next_prompt=tagged)
    db.add(t)
    db.flush()
    db.add(Scenario(training_id=t.id, round=1, scenario_number=1, persona={}, prompt=tagged, input="i", output="o",
                    scores=[], score=5.0))
    db.add(TrainedModel(training_id=t.id, round=1, prompt=tagged, weights=b"", mean_score=5.0))
    db.commit()
    with TestClient(app):
        pass
    db.expire_all()
    assert db.get(Training, t.id).next_prompt == "Be kind."
    assert db.scalar(select(Scenario.prompt)) == db.scalar(select(TrainedModel.prompt)) == "Be kind."


def test_pause_and_resume_continue_from_saved_progress(authed_client, fakes, monkeypatch, db):
    score = training.agents.score_conversation
    paused = []

    lock = threading.Lock()  # a batch is scored at once

    def press_pause_during_the_first_scenario(llm, conv, rubric):
        with lock:
            if not paused:
                paused.append(authed_client.post(f"/api/trainings/{db.scalar(select(Training.id))}/pause"))
        return score(llm, conv, rubric)

    monkeypatch.setattr(training.agents, "score_conversation", press_pause_during_the_first_scenario)
    created = authed_client.post("/api/trainings", json=BODY).json()
    [r] = paused
    assert r.status_code == 200 and r.json()["status"] == "pausing"
    assert set(r.json()) == set(created)
    d = authed_client.get(f"/api/trainings/{created['id']}").json()
    # Round 1's one batch and its training finish; it stops before round 2.
    assert (d["status"], d["current_round"], d["scenarios_done"]) == ("paused", 1, 2)

    r = authed_client.post(f"/api/trainings/{d['id']}/resume")
    assert r.status_code == 200 and r.json()["status"] == "pending"
    d = authed_client.get(f"/api/trainings/{d['id']}").json()
    # Exactly the uninterrupted run (see test_create_runs_training_in_background): nothing was redone.
    assert (d["status"], d["current_round"], d["scenarios_done"], d["latest_prompt"]) == ("completed", 2, 4,
                                                                                         "Be helpful.?")
    assert [w["score"] for w in d["score_windows"]] == [7.75, 8.5, 8.5, 8.5]
    assert (d["baseline_score"], d["best_score"]) == (7.0, 8.5)
    assert len(fakes) == 10  # 2 persona generations and 8 scorings


def test_pause_only_a_pending_or_running_training(authed_client, db, user):
    t = add_training(db, user, "pending")
    for status in ("pending", "running"):
        set_status(db, t, status)
        r = authed_client.post(f"/api/trainings/{t.id}/pause")
        assert (r.status_code, r.json()["status"]) == (200, "pausing")
    for status in ("pausing", "paused", "completed", "failed"):
        set_status(db, t, status)
        r = authed_client.post(f"/api/trainings/{t.id}/pause")
        assert (r.status_code, r.json()) == (409, {"detail": "Only a pending or running training can be paused."})
        db.expire_all()
        assert db.get(Training, t.id).status == status


def test_retry_a_failed_training_with_the_current_helper(authed_client, fakes, monkeypatch, db, user):
    t = add_training(db, user, "failed")
    monkeypatch.delenv("ANTHROPIC_API_KEY")  # the user's helper is now the local model
    r = authed_client.post(f"/api/trainings/{t.id}/resume")
    assert (r.status_code, r.json()["status"], r.json()["error"]) == (200, "pending", None)
    d = authed_client.get(f"/api/trainings/{t.id}").json()
    assert (d["status"], d["error"], d["scenarios_done"]) == ("completed", None, 1)
    assert fakes and all(getattr(llm, "local", False) for llm in fakes)


def test_resume_rules(authed_client, fakes, monkeypatch, db, user):
    monkeypatch.setattr(training, "run_training", lambda *args: None)
    t = add_training(db, user, "paused")
    for status in ("pending", "running", "pausing", "completed"):
        set_status(db, t, status)
        r = authed_client.post(f"/api/trainings/{t.id}/resume")
        assert (r.status_code, r.json()) == (409, {"detail": "Only a paused or failed training can be resumed."})
    set_status(db, t, "paused")
    other = add_training(db, user, "running")
    r = authed_client.post(f"/api/trainings/{t.id}/resume")
    assert (r.status_code, r.json()) == (409, {"detail": "You already have a training in progress."})
    set_status(db, other, "completed")
    monkeypatch.setattr(training, "_SLOTS", threading.BoundedSemaphore(1))
    assert training._SLOTS.acquire(blocking=False)
    r = authed_client.post(f"/api/trainings/{t.id}/resume")
    assert (r.status_code, r.json()) == (429, {"detail": "The server is busy with other trainings; try again later."})
    db.expire_all()
    assert db.get(Training, t.id).status == "paused"
    training._SLOTS.release()
    assert authed_client.post(f"/api/trainings/{t.id}/resume").status_code == 200
    assert authed_client.post(f"/api/trainings/{t.id}/resume").status_code == 409  # never a second run
    set_status(db, t, "paused")
    acquire = training._SLOTS.acquire  # a second click resumes it while this request waits for its slot
    monkeypatch.setattr(training._SLOTS, "acquire", lambda blocking: set_status(db, t, "pending") or acquire(blocking))
    r = authed_client.post(f"/api/trainings/{t.id}/resume")
    assert (r.status_code, r.json()) == (409, {"detail": "Only a paused or failed training can be resumed."})
    assert acquire(blocking=False)  # every request gave its slot back


def test_delete_a_training(authed_client, fakes, db, user):
    t = authed_client.post("/api/trainings", json=BODY).json()
    assert authed_client.delete(f"/api/trainings/{t['id']}").status_code == 204
    assert (db.query(Training).count(), db.query(Scenario).count(), db.query(TrainedModel).count()) == (0, 0, 0)
    assert authed_client.get(f"/api/trainings/{t['id']}").status_code == 404

    t = add_training(db, user, "pending")
    for status in ("pending", "running", "pausing"):
        set_status(db, t, status)
        r = authed_client.delete(f"/api/trainings/{t.id}")
        assert (r.status_code, r.json()) == (409, {"detail": "Pause the training before deleting it."})
    for status in ("paused", "failed"):
        set_status(db, t, status)
        assert authed_client.delete(f"/api/trainings/{t.id}").status_code == 204
        t = add_training(db, user, "pending")
