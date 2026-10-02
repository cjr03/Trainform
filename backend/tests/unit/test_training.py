import logging

import pytest
from sqlalchemy.exc import OperationalError

from app import training
from app.db import SessionLocal
from app.models import Scenario, TrainedModel, Training, User


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.generate_calls = []
        self.sft_calls = []

    def generate(self, system_prompt, conversation):
        self.generate_calls.append(system_prompt)
        return f"reply to {conversation}"

    def sft(self, examples):
        self.sft_calls.append(examples)
        return 0.5

    def adapter_bytes(self):
        return b"PK-%d" % len(self.sft_calls)


@pytest.fixture
def fake_agents(monkeypatch):
    """Scores come from `fake_agents.scores` (one value per scenario, applied to every criterion)."""
    class State:
        scores = []
        persona_prompts = []
        progress = []
        rounds = []

    state = State()

    def generate_personas(llm, agent_prompt, n):
        state.persona_prompts.append(agent_prompt)
        return [{"attributes": [{"criterion": "Clarity", "rating": k}], "prompt": f"persona {k}"}
                for k in range(1, n + 1)]

    def score_conversation(llm, conversation, rubric):
        with SessionLocal() as s:  # what another process (the API) can see right now
            t = s.query(Training).one()
            state.progress.append((t.status, t.current_round, t.scenarios_done, s.query(Scenario).count()))
            state.rounds.append(training.list_rounds(t.id, s.get(User, t.user_id), s))
        value = state.scores.pop(0)
        if isinstance(value, Exception):
            raise value
        return [{"criterion": c, "score": value} for c in rubric]

    monkeypatch.setattr(training.agents, "generate_personas", generate_personas)
    monkeypatch.setattr(training.agents, "persona_respond", lambda llm, attributes, context: f"msg from {context}")
    monkeypatch.setattr(training.agents, "primary_respond", lambda generate, prompt, conv: generate(prompt, conv))
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "generate_improved_prompt", lambda llm, prompt, conv, scores: prompt + "+")
    return state


def make_training(db, user, rounds, scenarios):
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=["Clarity", "Tone"],
                 scenarios_per_round=scenarios, rounds=rounds)
    db.add(t)
    db.commit()
    return t


def run(db, t):
    models = []

    def load_model(name):
        models.append(FakeModel(name))
        return models[0]

    training.run_training(t.id, "llm", load_model, session_factory=SessionLocal)
    db.expire_all()
    return db.get(Training, t.id), models


def test_runs_every_round_and_threads_the_prompt(db, user, fake_agents):
    fake_agents.scores = [8, 5, 9, 4, 4, 4]
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=3))

    assert model.name == "tiny"
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 6)
    scenarios = db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number).all()
    assert [(s.round, s.scenario_number) for s in scenarios] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    # Scenario k+1 runs with the prompt improved after scenario k, across rounds too.
    assert [s.prompt for s in scenarios] == ["P", "P+", "P++", "P+++", "P++++", "P+++++"]
    assert model.generate_calls == [s.prompt for s in scenarios]
    assert fake_agents.persona_prompts == ["P", "P+++"]

    s = scenarios[0]
    assert s.persona == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}
    assert (s.input, s.output) == ("msg from persona 1", "reply to User: msg from persona 1")
    assert s.scores == [{"criterion": "Clarity", "score": 8}, {"criterion": "Tone", "score": 8}]
    assert s.score == 8.0

    # Only round 1 had scores >= HIGH_SCORE, and only those exchanges are trained on.
    high = (scenarios[0], scenarios[2])
    assert model.sft_calls == [[{"prompt": x.prompt, "input": x.input, "output": x.output} for x in high]]

    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.prompt, m.weights) for m in trained] == [(1, "P+++", b"PK-1"), (2, "P++++++", b"PK-1")]
    assert trained[0].mean_score == pytest.approx(22 / 3)
    assert trained[1].mean_score == 4.0
    assert t.best_model_id == trained[0].id

    # Progress is committed as it happens: status, round and each finished scenario.
    assert fake_agents.progress == [("running", 1 + n // 3, n, n) for n in range(6)]
    # Mid-round, /rounds shows the running round with a partial mean and no model yet.
    mid = fake_agents.rounds[4]
    assert [(x["round"], x["scenarios"], x["model_id"]) for x in mid] == [(1, 3, trained[0].id), (2, 1, None)]
    assert (mid[0]["mean_score"], mid[1]["mean_score"]) == (pytest.approx(22 / 3), 4.0)


def test_best_model_is_the_highest_round_mean(db, user, fake_agents):
    fake_agents.scores = [4, 9, 6]
    t, [model] = run(db, make_training(db, user, rounds=3, scenarios=1))
    best = db.get(TrainedModel, t.best_model_id)
    assert (best.round, best.mean_score) == (2, 9.0)
    assert len(model.sft_calls) == 1


def test_no_sft_without_high_scores(db, user, fake_agents):
    fake_agents.scores = [6, 6.9]
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=2))
    assert t.status == "completed"
    assert model.sft_calls == []
    assert db.query(TrainedModel).one().weights == b"PK-0"


def test_failure_marks_training_failed_and_logs(db, user, fake_agents, caplog):
    fake_agents.scores = [8, RuntimeError("boom " + "x" * 5000)]
    with caplog.at_level(logging.ERROR):
        t, _ = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert t.status == "failed"
    assert t.error.startswith("boom x") and len(t.error) <= 1000
    assert t.scenarios_done == 1
    assert db.query(Scenario).count() == 1
    assert any(r.exc_info for r in caplog.records)


def test_model_load_failure_marks_training_failed(db, user, fake_agents):
    t = make_training(db, user, rounds=1, scenarios=1)

    def load_model(name):
        raise OSError("no such model")

    training.run_training(t.id, "llm", load_model)
    db.expire_all()
    # Server internals (paths, SQL) stay in the log, not in the API's error field.
    assert (db.get(Training, t.id).status, db.get(Training, t.id).error) == ("failed", training.INTERNAL_ERROR)


def test_missing_training_is_logged_not_raised(db, caplog):
    with caplog.at_level(logging.ERROR):
        training.run_training(424242, "llm", FakeModel)
    assert "424242" in caplog.text


def test_dead_database_is_logged_not_raised(caplog):
    class DeadSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def get(self, *args):
            raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

        rollback = get

    with caplog.at_level(logging.ERROR):
        training.run_training(1, "llm", FakeModel, session_factory=DeadSession)
    assert caplog.records and all(r.exc_info for r in caplog.records)


def test_score_windows():
    assert training.score_windows([]) == []
    assert training.score_windows([5.0]) == [
        {"label": "n", "runs": 1, "score": 5.0}, {"label": "n/2", "runs": 1, "score": 5.0},
        {"label": "n/4", "runs": 1, "score": 5.0}, {"label": "last", "runs": 1, "score": 5.0},
    ]
    assert training.score_windows([3.0, 6.0, 9.0]) == [
        {"label": "n", "runs": 3, "score": 6.0}, {"label": "n/2", "runs": 1, "score": 9.0},
        {"label": "n/4", "runs": 1, "score": 9.0}, {"label": "last", "runs": 1, "score": 9.0},
    ]
    assert training.score_windows([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) == [
        {"label": "n", "runs": 8, "score": 4.5}, {"label": "n/2", "runs": 4, "score": 6.5},
        {"label": "n/4", "runs": 2, "score": 7.5}, {"label": "last", "runs": 1, "score": 8.0},
    ]
