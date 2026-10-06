"""End to end through the real app: real auth, real agents, fake mode for Claude. No dependency overrides."""

import io
import json
import zipfile

import pytest

from app import agents, fake, local_model, training
from app.local_model import LocalModel
from app.main import app
from app.models import Scenario, TrainedModel, Training
from app.training import SFT_FLOOR

ACCOUNT = {"email": "ada@example.com", "username": "ada", "password": "password1", "password_confirm": "password1"}
INITIAL_PROMPT = "You help Acme customers track their orders."


@pytest.fixture
def fake_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINFORM_FAKE_LLM", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(local_model, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models" / "demo").mkdir(parents=True)
    (tmp_path / "models" / "demo" / "config.json").write_text("{}")


def signup(client, **extra):
    r = client.post("/api/auth/signup", json=ACCOUNT | extra)
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['token']}"}


def zip_names(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return zf.namelist(), json.loads(zf.read("adapter_config.json"))


def test_fake_mode_training_over_http(client, db, fake_mode):
    assert not app.dependency_overrides
    auth = signup(client)  # no Claude key linked

    r = client.post("/api/agents/prompt", json={"description": "Acme order tracking"}, headers=auth)
    assert r.status_code == 200 and "Acme order tracking" in r.json()["prompt"]
    r = client.post("/api/agents/rubric", json={"agent_prompt": INITIAL_PROMPT}, headers=auth)
    assert r.status_code == 200
    assert 3 <= len(r.json()["criteria"]) <= 4
    rubric = r.json()["criteria"][:2] + ["Offers to book a repair."]  # edited on the Rubric slide

    body = {"base_model": "demo", "prompt": INITIAL_PROMPT, "rubric": rubric, "scenarios": 3, "rounds": 2}
    r = client.post("/api/trainings", json=body, headers=auth)
    assert r.status_code == 201
    tid = r.json()["id"]  # TestClient returns once the background run has finished

    d = client.get(f"/api/trainings/{tid}", headers=auth).json()
    assert (d["status"], d["error"], d["current_round"], d["scenarios_done"]) == ("completed", None, 2, 6)
    assert [(w["label"], w["runs"]) for w in d["score_windows"]] == [("n", 6), ("n/2", 3), ("n/4", 1), ("last", 1)]
    # Fake scores ignore the prompt, so no revision beats it and none is kept.
    assert d["latest_prompt"] == INITIAL_PROMPT

    rounds = client.get(f"/api/trainings/{tid}/rounds", headers=auth).json()
    assert [(x["round"], x["scenarios"]) for x in rounds] == [(1, 3), (2, 3)]
    assert all(x["model_id"] for x in rounds)
    # The best model is the one that scored highest on the anchor users, the newest on a tie.
    top = max(x["eval_score"] for x in rounds)
    assert d["best_score"] == pytest.approx(max(top, d["baseline_score"]))
    assert d["best_model_id"] == [x["model_id"] for x in rounds if x["eval_score"] == pytest.approx(top)][-1]

    scenarios = client.get(f"/api/trainings/{tid}/rounds/1/scenarios", headers=auth).json()
    assert [s["scenario_number"] for s in scenarios] == [1, 2, 3]
    for s in scenarios:
        assert [a["criterion"] for a in s["persona"]["attributes"]] == rubric
        assert all(1 <= a["rating"] <= 10 for a in s["persona"]["attributes"]) and s["persona"]["prompt"]
        assert s["input"] and s["output"]
        assert [x["criterion"] for x in s["scores"]] == rubric
        assert s["score"] == pytest.approx(sum(x["score"] for x in s["scores"]) / len(rubric))
    assert scenarios[0]["prompt"] == INITIAL_PROMPT
    assert max(s["score"] for s in scenarios) >= SFT_FLOOR > min(s["score"] for s in scenarios)

    r = client.get(f"/api/trainings/{tid}/download/prompt", headers=auth)
    assert r.status_code == 200 and r.text == db.get(TrainedModel, d["best_model_id"]).prompt
    assert r.headers["content-disposition"] == f'attachment; filename="training-{tid}-prompt.txt"'
    r = client.get(f"/api/trainings/{tid}/download/weights", headers=auth)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="training-{tid}-lora.zip"'
    names, config = zip_names(r.content)
    assert "adapter_config.json" in names and config["base_model_name_or_path"] == "demo"


def test_unreadable_claude_key_is_400_not_500(client, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")  # must not mask the broken user key
    signup(client, anthropic_api_key="sk-ant-old")
    monkeypatch.setenv("TRAINFORM_SECRET", "rotated-secret-0123456789abcdef0123")
    r = client.post("/api/auth/login", json={"username": "ada", "password": "password1"})
    auth = {"Authorization": f"Bearer {r.json()['token']}"}
    r = client.post("/api/agents/prompt", json={"description": "x"}, headers=auth)
    assert (r.status_code, r.json()["detail"]) == (
        400, "Your linked Claude key can't be read (the server secret changed). Re-link it on the Account page.")


def test_fake_llm_with_real_local_model(db, user, tiny_model_dir, monkeypatch):
    monkeypatch.setattr(local_model, "MODELS_DIR", tiny_model_dir.parent)
    sft_batches = []
    real_sft = LocalModel.sft
    monkeypatch.setattr(LocalModel, "sft", lambda self, examples, **kw: sft_batches.append(examples) or
                        real_sft(self, examples, **kw))
    rubric = ["Greets the user.", "Asks one clarifying question.", "Never invents facts."]
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="be helpful", rubric=rubric,
                 scenarios_per_round=3, rounds=2)
    db.add(t)
    db.commit()

    training.run_training(t.id, fake.fake_llm, LocalModel)

    db.expire_all()
    assert (t.status, t.error) == ("completed", None)
    scenarios = db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number).all()
    assert len(scenarios) == 6
    assert all([a["criterion"] for a in s.persona["attributes"]] == rubric for s in scenarios)
    # Scenario 1 is the held-out anchor; the rest learn from the teacher's replies, and round 2 replays round 1's.
    assert [(s.round, s.scenario_number) for s in scenarios if s.target] == [(1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    teacher = fake.fake_llm("", "", agents.REPLY_SCHEMA)["reply"]
    assert [len(b) for b in sft_batches] == [2, 5] and {ex["output"] for b in sft_batches for ex in b} == {teacher}
    models = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [m.round for m in models] == [0, 1, 2]  # round 0: the baseline on the anchor
    for m in models:
        names, config = zip_names(m.weights)
        assert "adapter_config.json" in names and config["peft_type"] == "LORA"
    assert t.best_model_id in {m.id for m in models}


def test_fake_mode_pause_and_resume_over_http(client, db, fake_mode, monkeypatch):
    auth = signup(client)
    sft = fake.FakeLocalModel.sft

    def pause_during_round_1_sft(self, examples, **kw):
        client.post(f"/api/trainings/{db.query(Training.id).scalar()}/pause", headers=auth)
        return sft(self, examples, **kw)

    monkeypatch.setattr(fake.FakeLocalModel, "sft", pause_during_round_1_sft)
    body = {"base_model": "demo", "prompt": INITIAL_PROMPT, "rubric": ["Greets the user."], "scenarios": 3, "rounds": 2}
    tid = client.post("/api/trainings", json=body, headers=auth).json()["id"]
    d = client.get(f"/api/trainings/{tid}", headers=auth).json()
    assert (d["status"], d["current_round"], d["scenarios_done"]) == ("paused", 1, 3)

    monkeypatch.setattr(fake.FakeLocalModel, "sft", sft)
    assert client.post(f"/api/trainings/{tid}/resume", headers=auth).status_code == 200
    d = client.get(f"/api/trainings/{tid}", headers=auth).json()
    assert (d["status"], d["current_round"], d["scenarios_done"]) == ("completed", 2, 6)
    # Round 2 trained on from the adapter round 1 trained, restored in the resumed run's fresh model.
    baseline, round1, round2 = (zip_names(m.weights)[1]["examples_seen"]
                                for m in db.query(TrainedModel).order_by(TrainedModel.round))
    assert baseline == 0 < round1 < round2
