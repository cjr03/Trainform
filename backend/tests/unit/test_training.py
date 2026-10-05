import logging
import threading

import pytest
from sqlalchemy.exc import OperationalError

from app import training
from app.db import SessionLocal
from app.models import Scenario, TrainedModel, Training, User


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.generate_calls = []
        self.conversations = []
        self.sft_calls = []

    def generate(self, system_prompt, conversation):
        self.generate_calls.append(system_prompt)
        self.conversations.append(conversation)
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
        persona_rubrics = []
        progress = []
        rounds = []
        events = []
        improved = []

    state = State()

    def persona_respond(llm, attributes, context):
        state.events.append(f"message {context}")
        return f"msg from {context}"

    def generate_personas(llm, agent_prompt, n, rubric=None):
        state.persona_prompts.append(agent_prompt)
        state.persona_rubrics.append(rubric)
        return [{"attributes": [{"criterion": "Clarity", "rating": k}], "prompt": f"persona {k}"}
                for k in range(1, n + 1)]

    def score_conversation(llm, conversation, rubric):
        state.events.append("score")
        with SessionLocal() as s:  # what another process (the API) can see right now
            t = s.query(Training).one()
            state.progress.append((t.status, t.current_round, t.scenarios_done, s.query(Scenario).count()))
            state.rounds.append(training.list_rounds(t.id, s.get(User, t.user_id), s))
        value = state.scores.pop(0)
        if isinstance(value, Exception):
            raise value
        return [{"criterion": c, "score": value} for c in rubric]

    monkeypatch.setattr(training.agents, "generate_personas", generate_personas)
    monkeypatch.setattr(training.agents, "persona_respond", persona_respond)
    monkeypatch.setattr(training.agents, "primary_respond", lambda generate, prompt, conv: generate(prompt, conv))
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "generate_improved_prompt",
                        lambda llm, prompt, conv, scores: state.improved.append(prompt) or prompt + "+")
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


def test_without_claude_the_helper_agents_run_on_the_base_model(db, user, fake_agents, monkeypatch):
    fake_agents.scores = [8]
    helpers, used = [], []
    monkeypatch.setattr(training, "local_llm", lambda model: helpers.append(model) or "local llm")
    personas = training.agents.generate_personas
    monkeypatch.setattr(training.agents, "generate_personas", lambda llm, *a, **kw: used.append(llm) or personas(llm, *a, **kw))
    models = []
    t = make_training(db, user, rounds=1, scenarios=1)
    training.run_training(t.id, None, lambda name: models.append(FakeModel(name)) or models[-1])
    db.expire_all()
    assert db.get(Training, t.id).status == "completed"
    assert helpers == models and used == ["local llm"]  # the already-loaded model, not a second copy


def test_persona_messages_are_written_at_once_before_the_chain(db, user, fake_agents, monkeypatch):
    fake_agents.scores = [8, 8, 8, 8, 8, 8]
    together = threading.Barrier(3, timeout=10)  # breaks unless the round's three messages are in flight at once
    respond = training.agents.persona_respond
    monkeypatch.setattr(training.agents, "persona_respond", lambda *a: together.wait() * 0 or respond(*a))
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=3))
    assert t.status == "completed"
    round_events = fake_agents.events[:6], fake_agents.events[6:]
    for events in round_events:  # every persona's message is written before the first scenario is scored
        assert sorted(events[:3]) == [f"message persona {k}" for k in (1, 2, 3)] and events[3:] == ["score"] * 3
    # Each scenario still gets its own persona's message, in persona order.
    scenarios = db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number).all()
    assert [s.input for s in scenarios] == [f"msg from persona {k}" for k in (1, 2, 3)] * 2


def test_local_helper_calls_run_in_turn(db, user, fake_agents, monkeypatch):
    # One local model answers one call at a time, so its persona messages are not written concurrently.
    fake_agents.scores = [8, 8]
    def local(system, message, schema=None, effort=None):
        pass

    local.local = True
    monkeypatch.setattr(training, "local_llm", lambda model: local)
    threads = []
    respond = training.agents.persona_respond
    monkeypatch.setattr(training.agents, "persona_respond",
                        lambda *a: threads.append(threading.get_ident()) or respond(*a))
    t = make_training(db, user, rounds=1, scenarios=2)
    training.run_training(t.id, None, FakeModel)
    db.expire_all()
    assert db.get(Training, t.id).status == "completed"
    assert threads == [threading.get_ident()] * 2


def test_runs_every_round_and_threads_the_prompt(db, user, fake_agents):
    fake_agents.scores = [8, 5, 9, 4, 4, 4]
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=3))

    assert model.name == "tiny"
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 6)
    scenarios = db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number).all()
    assert [(s.round, s.scenario_number) for s in scenarios] == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    # Scenario k+1 runs with the prompt improved after scenario k, across rounds too.
    assert [s.prompt for s in scenarios] == ["P", "P+", "P++", "P+++", "P++++", "P+++++"]
    assert fake_agents.improved == [s.prompt for s in scenarios[:-1]]  # nothing would use an improved prompt after the last
    assert model.generate_calls == [s.prompt for s in scenarios]
    assert fake_agents.persona_prompts == ["P", "P+++"]
    assert fake_agents.persona_rubrics == [["Clarity", "Tone"]] * 2  # personas pressure the training's own rubric

    s = scenarios[0]
    assert s.persona == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}
    assert (s.input, s.output) == ("msg from persona 1", "reply to msg from persona 1")
    assert s.scores == [{"criterion": "Clarity", "score": 8}, {"criterion": "Tone", "score": 8}]
    assert s.score == 8.0

    # Only round 1 had scores >= HIGH_SCORE, and only those exchanges are trained on.
    high = (scenarios[0], scenarios[2])
    assert model.sft_calls == [[{"prompt": x.prompt, "input": x.input, "output": x.output} for x in high]]
    # SFT sees the same user turn the model was generating from.
    assert [x["input"] for x in model.sft_calls[0]] == [model.conversations[0], model.conversations[2]]

    # Each round's model is what was evaluated: the round's starting weights and its best-scoring scenario's prompt.
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.prompt, m.weights) for m in trained] == [(1, "P++", b"PK-0"), (2, "P+++", b"PK-1")]
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


def test_best_model_is_what_earned_the_score(db, user, fake_agents):
    fake_agents.scores = [9, 9, 3, 3]
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=2))
    best = db.get(TrainedModel, t.best_model_id)
    round1 = [s.prompt for s in db.query(Scenario).filter_by(round=1)]
    assert (best.round, best.mean_score, best.weights) == (1, 9.0, b"PK-0")  # weights round 1 ran with, before SFT
    assert best.prompt in round1 and best.prompt == "P"


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
