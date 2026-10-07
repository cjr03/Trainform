import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app import agent_profiles
from app.main import app
from app.models import SEED_AGENT, Agent, TrainedModel, Training, User

BODY = {"name": "  Support  ", "base_model": "tiny", "prompt": "Be helpful."}


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(agent_profiles, "list_local_models", lambda user_id: [{"name": "tiny", "source": "upload"}])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def add_agent(db, user, name="Support", base_model="tiny"):
    a = Agent(user_id=user.id, name=name, base_model=base_model, prompt="Be helpful.")
    db.add(a)
    db.commit()
    return a


def add_training(db, agent, status="completed", scores=()):
    """A training of `agent` with a TrainedModel per round scoring `scores` (round 0, the baseline, first)."""
    t = Training(user_id=agent.user_id, agent_id=agent.id, base_model=agent.base_model, initial_prompt=agent.prompt,
                 rubric=["C"], scenarios_per_round=1, rounds=2, status=status)
    db.add(t)
    db.flush()
    trained = [TrainedModel(training_id=t.id, round=r, prompt=f"P{r}", weights=b"W%d" % r, mean_score=score)
               for r, score in enumerate(scores)]
    db.add_all(trained)
    db.flush()
    if trained:
        t.best_model_id = max(trained, key=lambda m: m.mean_score).id
        t.current_round = len(trained) - 1
    db.commit()
    return t


def test_create_and_list(authed_client, models, db, user):
    r = authed_client.post("/api/agents", json=BODY)
    assert r.status_code == 201
    created = r.json()
    assert created == {"id": created["id"], "name": "Support", "base_model": "tiny", "prompt": "Be helpful.",
                       "image": None, "model_available": True, "overall": None, "trainings": 0,
                       "created_at": created["created_at"]}
    a = add_agent(db, user, name="Another")
    add_training(db, a, scores=[6.0, 7.24, 6.5])
    add_training(db, a, scores=[5.0, 8.26])
    add_training(db, a, scores=[9.9])  # only its baseline: not a training score
    add_training(db, a, status="paused")
    listed = authed_client.get("/api/agents").json()
    # By name; the overall is 10 x the best trained round's score over all its trainings, else Unranked (None).
    assert [(x["name"], x["overall"], x["trainings"]) for x in listed] == [("Another", 83, 4), ("Support", None, 0)]
    assert listed[1] == created


@pytest.mark.parametrize("change", [
    {"name": ""}, {"name": "   "}, {"name": "x" * 101}, {"prompt": ""}, {"prompt": "x" * 20001}, {"base_model": None},
])
def test_create_validation(authed_client, models, db, change):
    assert authed_client.post("/api/agents", json=BODY | change).status_code == 422
    assert db.query(Agent).count() == 0


def test_create_checks_the_model_and_the_name(authed_client, models, monkeypatch, db, user):
    assert authed_client.post("/api/agents", json=BODY | {"name": "x" * 100, "prompt": "x" * 20000}).status_code == 201
    r = authed_client.post("/api/agents", json=BODY | {"base_model": "missing"})
    assert (r.status_code, r.json()) == (400, {"detail": "Unknown base model: missing"})
    r = authed_client.post("/api/agents", json=BODY | {"base_model": "claude"})
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page first."})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    r = authed_client.post("/api/agents", json=BODY | {"base_model": "claude"})
    assert (r.status_code, r.json()["base_model"]) == (201, "claude")
    r = authed_client.post("/api/agents", json=BODY | {"name": "Support"})
    assert (r.status_code, r.json()) == (409, {"detail": "You already have an agent with that name."})
    assert db.query(Agent).count() == 2
    bob = User(email="bob@example.com", username="bob", password_hash="x")
    db.add(bob)
    db.commit()
    add_agent(db, bob)  # names are unique per user only


def test_detail_lists_its_trainings_newest_first(authed_client, models, db, user):
    a = add_agent(db, user)
    first = add_training(db, a, scores=[6.0, 7.0])
    second = add_training(db, a, status="failed")
    add_training(db, add_agent(db, user, name="Other"), scores=[1.0, 9.0])
    second.battle = None  # saved as SQL NULL, so it still counts as a training
    add_training(db, a).battle = {"rounds": 1}  # a battle: listed apart, not as a training
    db.commit()
    assert {x["name"]: x["trainings"] for x in authed_client.get("/api/agents").json()} == {"Support": 2, "Other": 1}
    d = authed_client.get(f"/api/agents/{a.id}").json()
    assert (d["name"], d["overall"]) == ("Support", 70)
    assert d["trainings"] == [
        {"id": second.id, "status": "failed", "created_at": d["trainings"][0]["created_at"], "rounds": 2,
         "scenarios_per_round": 1, "best_score": None, "baseline_score": None},
        {"id": first.id, "status": "completed", "created_at": d["trainings"][1]["created_at"], "rounds": 2,
         "scenarios_per_round": 1, "best_score": 7.0, "baseline_score": 6.0},
    ]


def test_update(authed_client, models, db, user):
    a = add_agent(db, user)
    add_agent(db, user, name="Taken")
    add_training(db, a, scores=[5.0, 6.0])
    r = authed_client.patch(f"/api/agents/{a.id}", json={"prompt": "Be kind."})
    assert (r.status_code, r.json()["prompt"], r.json()["name"], r.json()["trainings"]) == (200, "Be kind.",
                                                                                           "Support", 1)
    r = authed_client.patch(f"/api/agents/{a.id}", json={"name": " Renamed "})
    assert (r.json()["name"], r.json()["prompt"], r.json()["overall"]) == ("Renamed", "Be kind.", 60)
    assert authed_client.patch(f"/api/agents/{a.id}", json={}).json()["name"] == "Renamed"
    r = authed_client.patch(f"/api/agents/{a.id}", json={"name": "Taken"})
    assert (r.status_code, r.json()) == (409, {"detail": "You already have an agent with that name."})
    assert authed_client.patch(f"/api/agents/{a.id}", json={"prompt": ""}).status_code == 422
    t = add_training(db, a, status="running")
    r = authed_client.patch(f"/api/agents/{a.id}", json={"prompt": "Mid-training."})
    assert (r.status_code, r.json()) == (409, {"detail": "Wait for the agent's training to pause or finish first."})
    db.expire_all()
    assert db.get(Agent, a.id).prompt == "Be kind."
    db.query(Training).filter_by(id=t.id).update({"status": "paused"})
    db.commit()
    assert authed_client.patch(f"/api/agents/{a.id}", json={"prompt": "Now."}).json()["prompt"] == "Now."


def test_delete_takes_its_trainings(authed_client, models, db, user):
    a = add_agent(db, user)
    t = add_training(db, a, status="pausing", scores=[5.0])
    r = authed_client.delete(f"/api/agents/{a.id}")
    assert (r.status_code, r.json()) == (409, {"detail": "Pause the agent's training before deleting it."})
    db.query(Training).filter_by(id=t.id).update({"status": "paused"})
    db.commit()
    assert authed_client.delete(f"/api/agents/{a.id}").status_code == 204
    assert (db.query(Agent).count(), db.query(Training).count(), db.query(TrainedModel).count()) == (0, 0, 0)
    assert authed_client.get(f"/api/agents/{a.id}").status_code == 404


def test_other_users_agents_are_404(authed_client, models, db):
    bob = User(email="bob@example.com", username="bob", password_hash="x")
    db.add(bob)
    db.commit()
    a = add_agent(db, bob)
    for method in ("get", "patch", "delete"):
        r = authed_client.request(method, f"/api/agents/{a.id}", json={"prompt": "Mine."})
        assert (r.status_code, r.json()) == (404, {"detail": "Agent not found"})
    assert authed_client.get("/api/agents").json() == []
    assert db.get(Agent, a.id).prompt == "Be helpful."


def test_routes_require_a_user(client):
    assert client.get("/api/agents").status_code == 401
    assert client.post("/api/agents", json=BODY).status_code == 401
    for method in ("get", "patch", "delete"):
        assert client.request(method, "/api/agents/1", json={}).status_code == 401


def test_sync_agent_copies_this_rounds_best_model(db, user):
    a = add_agent(db, user)
    t = add_training(db, a, scores=[7.0, 5.0])  # round 1 scored below the baseline: nothing trained to keep
    agent_profiles.sync_agent(db, t)
    db.commit()
    db.refresh(a)
    assert (a.prompt, a.weights) == ("Be helpful.", None)
    t = add_training(db, a, scores=[5.0, 7.0])
    agent_profiles.sync_agent(db, t)
    db.commit()
    db.refresh(a)
    assert (a.prompt, a.weights) == ("P1", b"W1")
    a.prompt = "Edited."
    db.commit()
    t.current_round = 2  # round 2 lost to round 1: that round synced it already, so the edit stays
    agent_profiles.sync_agent(db, t)
    db.commit()
    db.refresh(a)
    assert (a.prompt, a.weights) == ("Edited.", b"W1")

    claude = add_agent(db, user, name="Claude", base_model="claude")
    t = add_training(db, claude, scores=[5.0, 7.0])
    agent_profiles.sync_agent(db, t)
    db.commit()
    db.refresh(claude)
    assert (claude.prompt, claude.weights) == ("P1", None)  # only its prompt is trained


def test_sync_agent_skips_a_training_on_another_base_model(db, user):
    # Trainings from before agents were all attached to the seeded agent, whatever model they trained.
    a = add_agent(db, user)
    t = add_training(db, a, scores=[5.0, 7.0])
    t.base_model = "other-model"
    agent_profiles.sync_agent(db, t)
    db.commit()
    db.refresh(a)
    assert (a.prompt, a.weights) == ("Be helpful.", None)


def test_signup_gives_no_agent(client):
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "username": "ada", "password": "password1",
                                              "password_confirm": "password1"})
    assert client.get("/api/agents", headers={"Authorization": f"Bearer {r.json()['token']}"}).json() == []


def test_startup_seeds_only_old_trainings_and_drops_untrained_seed_agents(db, user):
    db.execute(text("ALTER TABLE trainings DROP COLUMN agent_id"))  # a database from before agents
    db.commit()
    db.execute(text("INSERT INTO trainings (user_id, base_model, initial_prompt, rubric, scenarios_per_round, rounds,"
                    " status, current_round, scenarios_done) VALUES (:u, 'first', 'P', '[]', 1, 1, 'completed', 0, 0),"
                    " (:u, 'latest', 'P', '[]', 1, 1, 'completed', 0, 0)"), {"u": user.id})
    carol, dave, erin = (User(email=f"{n}@example.com", username=n, password_hash="x") for n in ("carol", "dave", "erin"))
    db.add_all([carol, dave, erin])
    db.commit()
    db.add_all([
        Agent(user_id=dave.id, name="Mine", base_model="tiny", prompt="P"),
        Agent(user_id=carol.id, base_model="tiny", **SEED_AGENT),  # seeded by an older signup, never trained
        Agent(user_id=erin.id, base_model="tiny", name=SEED_AGENT["name"], prompt="Edited."),  # theirs now
    ])
    db.commit()
    for _ in range(2):  # idempotent
        with TestClient(app):
            pass
    db.expire_all()
    agents = {a.user_id: a for a in db.query(Agent)}
    # Only the user with trainings from before agents gets the seed agent, for them; carol's untouched one is gone.
    assert set(agents) == {user.id, dave.id, erin.id} and agents[dave.id].name == "Mine"
    assert agents[erin.id].prompt == "Edited."
    assert (agents[user.id].name, agents[user.id].base_model, agents[user.id].prompt) == (
        "Bike Repair Agent", "latest", SEED_AGENT["prompt"])
    assert {t.agent_id for t in db.query(Training)} == {agents[user.id].id}
