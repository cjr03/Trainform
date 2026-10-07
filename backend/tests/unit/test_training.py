import logging
import re
import threading
import time
from collections import Counter
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from app import training
from app.agents import FOCUS_HEADER
from app.db import SessionLocal
from app.llm import Stopped
from app.models import Agent, Scenario, TrainedModel, Training, User

RUBRIC = ["Clarity", "Tone"]
CONVERSATION = re.compile(r"User: msg from persona (\d+)\nAssistant: reply to msg from persona \d+ under <(.*)> "
                          r"with (\S+)\Z", re.S)


def conversation(k, prompt="P", adapter="PK-0"):
    """Persona k's message and the fake model's reply to it under `prompt` with `adapter`, as the scorer sees it."""
    return f"User: msg from persona {k}\nAssistant: reply to msg from persona {k} under <{prompt}> with {adapter}"


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.generate_calls = []
        self.conversations = []
        self.sft_calls = []
        self.seeds = []
        self.loads = []
        self.adapter = b"PK-0"

    def generate_many(self, pairs, stop):
        stop()
        replies = []
        for system_prompt, conversation in pairs:
            self.generate_calls.append(system_prompt)
            self.conversations.append(conversation)
            # The reply names its prompt and adapter, so a score can depend on them however the scoring threads
            # interleave.
            replies.append(f"reply to {conversation} under <{system_prompt}> with {self.adapter.decode()}")
        return replies

    def sft(self, examples, **kwargs):
        self.sft_calls.append(examples)
        self.seeds.append(kwargs["seed"])
        self.adapter = b"PK-%d" % kwargs["seed"]  # named for its round, so a resumed run names it the same
        return 0.5

    def adapter_bytes(self):
        return self.adapter

    def load_adapter(self, data):
        self.loads.append(data)
        self.adapter = data


@pytest.fixture(autouse=True)
def poll_every_check(monkeypatch):
    monkeypatch.setattr(training, "PAUSE_POLL", 0)


@pytest.fixture
def fake_agents(monkeypatch):
    """A conversation scores `fake_agents.score(persona number, prompt, adapter)` on every criterion, or a list with
    one score per criterion (default 8). Helpers run on worker threads, so calls are recorded under a lock."""
    state = SimpleNamespace(score=lambda k, prompt, adapter: 8, persona_prompts=[], persona_rubrics=[], last_scores=[],
                            progress=[], rounds=[], events=[], reviews=[], teacher=[])
    lock = threading.Lock()

    def record(**values):
        with lock:
            for name, value in values.items():
                getattr(state, name).append(value)

    def persona_respond(llm, attributes, context):
        record(events=f"message {context}")
        return f"msg from {context}"

    def generate_personas(llm, agent_prompt, n, rubric=None, last_scores=None):
        record(events="personas", persona_prompts=agent_prompt, persona_rubrics=rubric, last_scores=last_scores)
        return [{"attributes": [{"criterion": "Clarity", "rating": k}], "prompt": f"persona {k}"}
                for k in range(1, n + 1)]

    def score_conversation(llm, conversation, rubric):
        k, prompt, adapter = CONVERSATION.match(conversation).groups()
        with SessionLocal() as s:  # what another process (the API) can see right now
            t = s.query(Training).order_by(Training.id.desc()).first()
            seen = (t.status, t.current_round, t.scenarios_done, s.query(Scenario).filter_by(training_id=t.id).count())
            rounds = training.list_rounds(t.id, s.get(User, t.user_id), s)
        record(events="score", progress=seen, rounds=rounds)
        value = state.score(int(k), prompt, adapter)
        values = value if isinstance(value, list) else [value] * len(rubric)
        return [{"criterion": c, "score": v} for c, v in zip(rubric, values)]

    def review_round(llm, prompt, criterion_means, weakest):
        record(events="review", reviews=(prompt, criterion_means, weakest))
        return prompt + "R"

    def teacher_reply(llm, prompt, message, rubric):
        record(teacher=(prompt, message, rubric))
        return f"ideal reply to {message}"

    monkeypatch.setattr(training.agents, "generate_personas", generate_personas)
    monkeypatch.setattr(training.agents, "persona_respond", persona_respond)
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "review_round", review_round)
    monkeypatch.setattr(training.agents, "teacher_reply", teacher_reply)
    return state


@pytest.fixture
def local_helper(monkeypatch):
    """No Claude: the helpers run on the loaded model, one call at a time (the batched calls in turn here)."""
    def local(system, message, schema=None, effort=None):
        pass

    local.local = True
    monkeypatch.setattr(training, "local_llm", lambda model, stop: local)
    agents = training.agents
    monkeypatch.setattr(agents, "score_conversations",
                        lambda llm, conversations, rubric: [agents.score_conversation(llm, c, rubric)
                                                            for c in conversations])
    monkeypatch.setattr(agents, "persona_messages", lambda llm, personas: [
        agents.persona_respond(llm, p["attributes"], p["prompt"]) for p in personas])
    return local


def make_training(db, user, rounds, scenarios, rubric=RUBRIC, **values):
    t = Training(**{"user_id": user.id, "base_model": "tiny", "initial_prompt": "P", "rubric": list(rubric),
                    "scenarios_per_round": scenarios, "rounds": rounds} | values)
    db.add(t)
    db.commit()
    return t


def run(db, t, llm="llm", **kwargs):
    models = []

    def load_model(name):
        models.append(FakeModel(name))
        return models[0]

    training.run_training(t.id, llm, load_model, session_factory=SessionLocal, **kwargs)
    db.expire_all()
    return db.get(Training, t.id), models


def scenarios(db, t=None):
    query = db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number)
    return (query.filter_by(training_id=t.id) if t else query).all()


def example(s, output=None):
    return {"prompt": s.prompt, "input": s.input, "output": output or s.output}


def snapshot(db, t):
    """Everything a training saved, to compare a paused and resumed run with an uninterrupted one."""
    db.expire_all()
    t = db.get(Training, t.id)
    models = db.query(TrainedModel).filter_by(training_id=t.id).order_by(TrainedModel.round).all()
    return {
        "training": (t.status, t.error, t.current_round, t.scenarios_done, t.next_prompt, t.personas,
                     db.get(TrainedModel, t.best_model_id).round),
        "scenarios": [(s.round, s.scenario_number, s.persona, s.prompt, s.input, s.output, s.scores, s.score, s.target)
                      for s in scenarios(db, t)],
        "models": [(m.round, m.prompt, m.weights, m.mean_score) for m in models],
    }


def test_without_claude_the_helper_agents_run_on_the_base_model(db, user, fake_agents, monkeypatch):
    # Loading a second copy of the model for the helpers would double the memory a training needs.
    helpers, used = [], []
    monkeypatch.setattr(training, "local_llm", lambda model, stop: helpers.append(model) or "local llm")
    personas = training.agents.generate_personas
    monkeypatch.setattr(training.agents, "generate_personas",
                        lambda llm, *a, **kw: used.append(llm) or personas(llm, *a, **kw))
    t, models = run(db, make_training(db, user, rounds=1, scenarios=1), llm=None)
    assert t.status == "completed"
    assert helpers == models and used == ["local llm"]  # the already-loaded model, not a second copy


def test_persona_messages_are_written_at_once(db, user, fake_agents, monkeypatch):
    # With Claude a round's messages are independent calls; writing them in turn would make each round slower.
    together = threading.Barrier(3, timeout=10)  # breaks unless the round's three messages are in flight at once
    respond = training.agents.persona_respond
    monkeypatch.setattr(training.agents, "persona_respond", lambda *a: together.wait() * 0 or respond(*a))
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=3))
    assert t.status == "completed"
    # Each scenario still gets its own persona's message, in persona order.
    assert [s.input for s in scenarios(db)] == [f"msg from persona {k}" for k in (1, 2, 3)] * 2


def test_runs_every_round_and_threads_the_prompt(db, user, fake_agents):
    # The whole loop end to end with Claude: what each round runs, what the reviews see, what the API sees meanwhile.
    base = {1: 8, 2: 5, 3: 9, 4: 4, 5: 7, 6: 6}
    # Clarity scores 2 above Tone, and "PR" adds 1 to both.
    fake_agents.score = lambda k, prompt, adapter: [base[k] + (prompt == "PR"), base[k] + (prompt == "PR") - 2]
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=6))

    assert model.name == "tiny"
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 12)
    rows = scenarios(db)
    assert [(s.round, s.scenario_number) for s in rows] == [(r, k) for r in (1, 2) for k in range(1, 7)]
    # Round 2 ran every scenario under "P" and its revision "PR" at once; "PR" scored higher, so its rows are saved.
    assert [s.prompt for s in rows] == ["P"] * 6 + ["PR"] * 6
    assert [s.score for s in rows] == [7, 4, 8, 3, 6, 5, 8, 5, 9, 4, 7, 6]
    assert model.generate_calls == ["P"] * 6 + ["P"] * 3 + ["P"] * 6 + ["PR"] * 6 + ["PR"] * 3  # each then anchors
    assert model.conversations[9:15] == model.conversations[15:21]  # a paired test: the same users
    # The revision was written from round 1's mean per criterion and its lowest-scoring conversations, lowest first.
    assert fake_agents.reviews == [
        ("P", [("Clarity", 6.5), ("Tone", 4.5)],
         [(conversation(4), 3.0), (conversation(2), 4.0), (conversation(6), 5.0)])]
    # Round 2's personas are written for the agent's role and aimed at round 1's weakest criterion (Tone).
    assert fake_agents.persona_prompts == ["P", "P"]
    assert fake_agents.last_scores == [None, [6.5, 4.5]]
    assert fake_agents.persona_rubrics == [RUBRIC] * 2  # personas pressure the training's own rubric
    assert model.seeds == [1, 2]

    s = rows[0]
    assert s.persona == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}
    assert (s.input, s.output) == ("msg from persona 1", "reply to msg from persona 1 under <P> with PK-0")
    assert s.scores == [{"criterion": "Clarity", "score": 8}, {"criterion": "Tone", "score": 6}]

    # A round's scenarios are committed together: status, round, scenarios done and saved scenarios, per scoring call.
    progress = [(1, 0)] * 6 + [(1, 6)] * 3 + [(2, 6)] * 12 + [(2, 12)] * 3
    assert fake_agents.progress == [("running", r, n, n) for r, n in progress]
    # While round 2 runs, /rounds shows round 1 with its model.
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    mid = fake_agents.rounds[9]
    assert [(x["round"], x["scenarios"], x["model_id"]) for x in mid] == [(1, 6, trained[1].id)]
    assert (mid[0]["mean_score"], mid[0]["eval_score"]) == (5.5, pytest.approx(19 / 3))


def test_a_revision_replaces_the_prompt_only_when_it_scores_higher_on_the_same_messages(db, user, fake_agents):
    # A paired test on the same users is what makes a higher score the prompt's doing rather than easier users'.
    fake_agents.score = lambda k, prompt, adapter: {"P": 6, "PR": 6.25, "PRR": 6.45}[prompt]
    t, [model] = run(db, make_training(db, user, rounds=3, scenarios=2))
    # Round 2: "PR" gains exactly MIN_GAIN, so it replaces "P". Round 3: "PRR" gains less, so "PR"'s rows are saved.
    assert model.generate_calls == (["P"] * 3 + ["P"] * 2 + ["PR"] * 3 + ["PR"] * 2 + ["PRR"] * 2 + ["PR"])
    rows = scenarios(db)
    assert [s.prompt for s in rows] == ["P"] * 2 + ["PR"] * 4
    assert all(s.output.endswith("under <PR> with PK-2") for s in rows[4:])
    assert t.next_prompt == "PR" and db.query(TrainedModel).filter_by(round=3).one().prompt == "PR"
    # "PRR" was proposed from round 2's rows, under "PR".
    assert [r[0] for r in fake_agents.reviews] == ["P", "PR"]
    assert fake_agents.reviews[1][2] == [(conversation(k, "PR", "PK-1"), 6.25) for k in (1, 2)]


def test_an_unchanged_revision_is_not_tested(db, user, fake_agents, monkeypatch):
    # Running the same prompt twice would only double the round's cost.
    monkeypatch.setattr(training.agents, "review_round", lambda llm, prompt, *a: prompt)
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert t.status == "completed"
    assert model.generate_calls == ["P"] * 6  # two rounds of two and their anchor


def test_local_helpers_revise_the_prompt_in_code_and_rotate_the_focus(db, user, fake_agents, local_helper, monkeypatch):
    # A small local model can't revise a prompt, so code restates the weakest criteria, and a rejected focus moves on.
    rubric = ["Clarity", "Tone", "Accuracy", "Brevity"]
    fake_agents.score = lambda k, prompt, adapter: [8, 5, 7, 6]  # the same for every prompt: each revision loses
    proposed = []
    propose = training.agents.propose_local
    monkeypatch.setattr(training.agents, "propose_local",
                        lambda prompt, means, n: proposed.append((prompt, n)) or propose(prompt, means, n))
    t, [model] = run(db, make_training(db, user, rounds=4, scenarios=2, rubric=rubric), llm=None)
    assert t.status == "completed" and fake_agents.reviews == []
    first, second = (f"P{FOCUS_HEADER}- Tone\n- Brevity", f"P{FOCUS_HEADER}- Accuracy\n- Clarity")
    # Counted from the saved rounds, so a resumed training rotates the same.
    assert proposed == [("P", 0), ("P", 1), ("P", 2)]
    assert model.generate_calls == (["P"] * 3 + ["P"] * 2 + [first] * 2 + ["P"] + ["P"] * 2 + [second] * 2 + ["P"]
                                    + ["P"] * 2 + [first] * 2 + ["P"])
    assert {s.prompt for s in scenarios(db)} == {"P"}


def test_anchors_are_held_out_and_set_the_baseline(db, user, fake_agents):
    # Rounds are only comparable on the same users, and a model trained on those users would grade its own homework.
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 7, "PK-1": 9}[adapter] + (k == 3) / 2
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=4))  # anchors: scenarios 1 and 2
    rows = scenarios(db)
    assert sorted(m for _, m, _ in fake_agents.teacher) == ["msg from persona 3", "msg from persona 4"]
    assert model.sft_calls == [[example(s, f"ideal reply to {s.input}") for s in (rows[3], rows[2])]]  # weakest first
    # The anchors rerun under the trained adapter; round 0 is the baseline: the initial prompt and weights.
    assert model.conversations[-2:] == ["msg from persona 1", "msg from persona 2"]
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.prompt, m.weights, m.mean_score) for m in trained] == [
        (0, "P", b"PK-0", 7.0), (1, "P", b"PK-1", 9.0)]
    # The kept adapter is the best model's; the training's own copy isn't written again.
    assert t.best_model_id == trained[1].id and t.adapter is None and model.loads == []


def test_a_worse_round_reverts_to_the_best_adapter(db, user, fake_agents):
    # Training on from an adapter that got worse compounds the damage; the next round starts from the best one.
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 7, "PK-1": 9, "PK-2": 5, "PK-3": 6}[adapter]
    t, [model] = run(db, make_training(db, user, rounds=3, scenarios=2))
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.weights, m.mean_score) for m in trained] == [
        (0, b"PK-0", 7), (1, b"PK-1", 9), (2, b"PK-2", 5), (3, b"PK-3", 6)]
    assert model.loads == [b"PK-1", b"PK-1"]  # after rounds 2 and 3
    assert [s.output.rsplit(" ", 1)[1] for s in scenarios(db) if s.round == 3] == ["PK-1"] * 2
    assert t.best_model_id == trained[1].id


def test_a_tie_keeps_the_newer_model(db, user, fake_agents):
    # An equally good model that has seen more examples is the better bet, and keeping it saves a reload.
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=2))
    best = db.get(TrainedModel, t.best_model_id)
    assert (best.round, best.weights) == (1, b"PK-1") and model.loads == []


def test_the_agents_starting_weights_are_the_baseline(db, user, fake_agents):
    # An agent trained before starts from its kept adapter, not a fresh one.
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=2, adapter=b"PK-agent"))
    assert model.loads == [b"PK-agent"]
    assert db.query(TrainedModel).filter_by(round=0).one().weights == b"PK-agent"


def test_sft_learns_each_rounds_top_half_and_replays_earlier_rounds(db, user, fake_agents, local_helper):
    # Learning only from a round's better replies (never below the floor) moves the model up; replaying earlier
    # rounds' best, newest first, keeps it from forgetting them. No teacher without Claude.
    table = {"PK-0": {1: 8, 2: 8, 3: 8, 4: 3, 5: 5, 6: 6}, "PK-1": {1: 9, 2: 8.5, 3: 8, 4: 7, 5: 4, 6: 2}}
    fake_agents.score = lambda k, prompt, adapter: table.get(adapter, {}).get(k, 8)
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=6), llm=None)  # anchors: round 1's 1-3
    r1, r2 = ({s.scenario_number: s for s in scenarios(db) if s.round == r} for r in (1, 2))
    # Round 1's median (5) is under the floor, so only 6 qualifies; round 2's median is 7.5.
    assert model.sft_calls == [[example(r1[6])], [example(r2[k]) for k in (1, 2, 3)] + [example(r1[6])]]
    assert model.seeds == [1, 2] and fake_agents.teacher == []


def test_sft_replay_is_capped(db, user, fake_agents, monkeypatch):
    # A long training would otherwise make every round's SFT slower than the last.
    monkeypatch.setattr(training, "REPLAY", 2)
    fake_agents.score = lambda k, prompt, adapter: 8 + k
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=2))
    r2 = [s for s in scenarios(db) if s.round == 2]
    assert [len(x) for x in model.sft_calls] == [1, 2]
    assert model.sft_calls[1] == [example(s, f"ideal reply to {s.input}") for s in r2]  # this round's, weakest first


def test_with_claude_sft_learns_the_teachers_replies(db, user, fake_agents):
    # The reply the model should have sent teaches more than its own best one, most where its own was weakest.
    fake_agents.score = lambda k, prompt, adapter: {1: 8, 2: 2, 3: 9}[k]
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=3))
    rows = scenarios(db)
    assert sorted(fake_agents.teacher) == [("P", f"msg from persona {k}", RUBRIC) for k in (2, 3)]
    assert [s.target for s in rows] == [None, "ideal reply to msg from persona 2", "ideal reply to msg from persona 3"]
    assert model.sft_calls == [[example(rows[1], rows[1].target), example(rows[2], rows[2].target)]]


def test_no_sft_without_good_replies(db, user, fake_agents, local_helper):
    # Training on replies below the floor would teach the model its mistakes.
    fake_agents.score = lambda k, prompt, adapter: 5
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=2), llm=None)
    assert t.status == "completed" and model.sft_calls == []
    assert [m.weights for m in db.query(TrainedModel).order_by(TrainedModel.round)] == [b"PK-0", b"PK-0"]


def test_claude_as_the_agent_trains_only_its_prompt(db, user, fake_agents):
    # Claude's weights can't be fine-tuned: no model is loaded, no teacher or SFT, and the gate keeps the best prompt.
    replies = []

    def claude(system, message):
        replies.append(system)
        return f"reply to {message} under <{system}> with claude"

    # "PR" beats "P" on round 2's users (7 vs 6) but scores 5 on the anchor user, persona 1.
    fake_agents.score = lambda k, prompt, adapter: {"P": 6, "PR": 9, "PRR": 7}[prompt] - (prompt == "PR" and k == 1) * 4
    t, models = run(db, make_training(db, user, rounds=3, scenarios=2, base_model="claude"), agent_llm=claude)
    assert models == [] and (t.status, t.error) == ("completed", None)
    assert fake_agents.teacher == []
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.prompt, m.weights, m.mean_score) for m in trained] == [
        (0, "P", None, 6), (1, "P", None, 6), (2, "PR", None, 5), (3, "PRR", None, 7)]
    # Round 2's model lost to round 1's, so round 3 went on from "P" (and tested "PRR" against it).
    assert Counter(replies) == {"P": 3 + 2 + 2, "PR": 3, "PRR": 3}
    assert [s.prompt for s in scenarios(db)] == ["P", "P", "PR", "PR", "PRR", "PRR"]
    assert t.best_model_id == trained[3].id and t.next_prompt == "PRR"


@pytest.mark.parametrize("llm, agent_llm", [(None, "claude"), ("llm", None)])
def test_a_claude_agent_needs_claude_as_the_helper(db, user, fake_agents, llm, agent_llm):
    t = make_training(db, user, rounds=1, scenarios=1, base_model="claude")
    t, models = run(db, t, llm=llm, agent_llm=agent_llm)
    assert (t.status, t.error, models) == ("failed", "Claude agents need Claude as the helper", [])


def test_failure_marks_training_failed_and_logs(db, user, fake_agents, caplog):
    # The user needs to see why it stopped, without a 5000-character error filling the page.
    def score(k, prompt, adapter):
        if k == 5:
            raise RuntimeError("boom " + "x" * 5000)
        return 8

    fake_agents.score = score
    with caplog.at_level(logging.ERROR):
        t, _ = run(db, make_training(db, user, rounds=2, scenarios=6))
    assert t.status == "failed"
    assert t.error.startswith("boom x") and len(t.error) <= 1000
    assert t.scenarios_done == 0 and db.query(Scenario).count() == 0  # a round's scenarios are saved together
    assert any(r.exc_info for r in caplog.records)


def test_model_load_failure_marks_training_failed(db, user, fake_agents):
    # Server internals (paths, SQL) stay in the log, not in the API's error field.
    t = make_training(db, user, rounds=1, scenarios=1)

    def load_model(name):
        raise OSError("no such model")

    training.run_training(t.id, "llm", load_model)
    db.expire_all()
    assert (db.get(Training, t.id).status, db.get(Training, t.id).error) == ("failed", training.INTERNAL_ERROR)


def test_missing_training_is_logged_not_raised(db, caplog):
    # A background task's exception would go nowhere useful; the log is where it is seen.
    with caplog.at_level(logging.ERROR):
        training.run_training(424242, "llm", FakeModel)
    assert "424242" in caplog.text


def test_dead_database_is_logged_not_raised(caplog):
    # Even marking the training failed can fail; that must not crash the worker thread either.
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


def request_pause(*_):
    """What POST /pause does, from another connection."""
    with SessionLocal() as s:
        s.query(Training).filter(Training.status.in_(("pending", "running"))).update({"status": "pausing"})
        s.commit()


def resume(db, t, **kwargs):
    """What POST /resume does before starting the run again."""
    db.query(Training).filter_by(id=t.id).update({"status": "pending", "error": None})
    db.commit()
    return run(db, t, **kwargs)


def test_failure_mid_round_resumes_from_saved_progress(db, user, fake_agents):
    # A rate limit late in a long training must not cost the personas already paid for.
    failing = [True]

    def score(k, prompt, adapter):
        if k == 5 and failing:
            raise RuntimeError("Claude rate limit reached")
        return 8

    fake_agents.score = score
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=6))
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("failed", "Claude rate limit reached", 1, 0)
    # Saved: round 1's personas with their messages; no round finished yet.
    assert t.next_prompt is None and db.query(TrainedModel).count() == 0
    assert [x["message"] for x in t.personas] == [f"msg from persona {k}" for k in range(1, 7)]
    assert t.personas[0]["persona"] == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}

    failing.clear()
    t, [model] = resume(db, t)
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 12)
    assert [(s.round, s.scenario_number) for s in scenarios(db)] == [(r, k) for r in (1, 2) for k in range(1, 7)]
    # Round 1 is run again with its saved personas.
    assert model.conversations[:6] == [f"msg from persona {k}" for k in range(1, 7)]
    assert fake_agents.persona_prompts == ["P", "P"]
    assert fake_agents.events.count("message persona 1") == 2  # once per round


def uninterrupted(db, user, fake_agents, llm="llm", **values):
    """An uninterrupted run's saved state, before the training under test is made (fake_agents reads the newest)."""
    t, _ = run(db, make_training(db, user, **values), llm=llm)
    expected = snapshot(db, t)
    fake_agents.events.clear()
    return expected


@pytest.mark.parametrize("round_", [1, 2])
def test_a_pause_during_a_rounds_scenarios_drops_them(db, user, fake_agents, monkeypatch, round_):
    # Pause stops at once: the round's scenarios in flight are thrown away, and a resume runs them again.
    fake_agents.score = lambda k, prompt, adapter: {1: 8, 2: 6, 3: 9}[k] + (prompt == "PR") / 2
    expected = uninterrupted(db, user, fake_agents, rounds=2, scenarios=3)
    t = make_training(db, user, rounds=2, scenarios=3)
    generate = FakeModel.generate_many
    calls = []

    def pause_in_round(self, pairs, stop):
        calls.append(1)
        if len(calls) == 2 * round_ - 1:  # each round generates its scenarios, then its anchors
            request_pause()
        return generate(self, pairs, stop)

    monkeypatch.setattr(FakeModel, "generate_many", pause_in_round)
    t, _ = run(db, t)
    assert (t.status, t.current_round, t.scenarios_done) == ("paused", round_, 3 * (round_ - 1))
    assert len(scenarios(db, t)) == 3 * (round_ - 1)
    assert db.query(TrainedModel).filter_by(training_id=t.id).count() == (round_ - 1) * 2  # round 1's, if finished

    t, _ = resume(db, t)
    assert snapshot(db, t) == expected


def test_a_pause_during_sft_drops_it(db, user, fake_agents, monkeypatch):
    # A half-done SFT is thrown away (with the round's model), not saved; a resume trains again from the kept adapter.
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 7, "PK-1": 9, "PK-2": 8}[adapter]
    expected = uninterrupted(db, user, fake_agents, rounds=2, scenarios=2)
    t = make_training(db, user, rounds=2, scenarios=2)
    sft = FakeModel.sft

    def pause_mid_round_2_sft(self, examples, **kw):
        if kw["seed"] == 2:
            request_pause()
            kw["stop"]()  # the next step's check
        return sft(self, examples, **kw)

    monkeypatch.setattr(FakeModel, "sft", pause_mid_round_2_sft)
    t, _ = run(db, t)
    assert (t.status, t.current_round, t.scenarios_done) == ("paused", 2, 4)
    assert [m.round for m in db.query(TrainedModel).filter_by(training_id=t.id)] == [0, 1]
    assert [x["message"] for x in t.personas] == ["msg from persona 1", "msg from persona 2"]  # round 2's, kept

    monkeypatch.setattr(FakeModel, "sft", sft)
    t, [model] = resume(db, t)
    assert model.loads[0] == b"PK-1" and model.seeds == [2]  # from the best model's adapter
    assert snapshot(db, t) == expected


def test_a_pause_during_claude_calls_starts_no_more_of_them(db, user, fake_agents, monkeypatch):
    # The calls in flight finish, but none still queued starts, and what they wrote is dropped.
    sent, pausing = [], []

    def claude(system, message, schema=None, effort=None):
        sent.append(message)
        return "ok"

    def teacher_reply(llm, prompt, message, rubric):
        if pausing and message == "msg from persona 5":
            request_pause()
        llm("teacher", message)
        return f"ideal reply to {message}"

    monkeypatch.setattr(training.agents, "teacher_reply", teacher_reply)
    expected = uninterrupted(db, user, fake_agents, llm=claude, rounds=1, scenarios=40)
    sent.clear()
    pausing.append(True)
    t, _ = run(db, make_training(db, user, rounds=1, scenarios=40), llm=claude)
    assert t.status == "paused" and len(sent) < 36  # 36 teacher replies: all but the 4 anchors
    assert all(s.target is None for s in scenarios(db, t))

    pausing.clear()
    t, _ = resume(db, t, llm=claude)
    assert snapshot(db, t) == expected


def test_pause_before_the_run_starts(db, user, fake_agents):
    # A pause requested while queued must hold; the run must not flip it back to "running".
    t = make_training(db, user, rounds=1, scenarios=1)
    request_pause()
    t, _ = run(db, t)
    assert (t.status, t.current_round) == ("paused", 0)
    assert fake_agents.persona_prompts == []


def test_the_pause_check_reads_the_status_at_most_every_poll(db, user, monkeypatch):
    t = make_training(db, user, rounds=1, scenarios=1)
    reads = []

    class Session:
        def __init__(self):
            reads.append(1)
            self.session = SessionLocal()

        def __enter__(self):
            return self.session.__enter__()

        def __exit__(self, *exc):
            return self.session.__exit__(*exc)

    monkeypatch.setattr(training, "PAUSE_POLL", 3600)
    check = training._pause_check(t.id, Session)
    check()
    request_pause()
    check()  # within the poll interval: not read
    assert reads == [1]
    monkeypatch.setattr(training, "PAUSE_POLL", 0)
    with pytest.raises(Stopped):
        check()
    # Once stopped it stays stopped, even after a resume moves the status on: the stopped run's leftover task (the
    # next round's preparation) must make no more calls.
    db.query(Training).update({"status": "running"})
    db.commit()
    with pytest.raises(Stopped):
        check()


def test_a_pause_stops_waiting_on_a_claude_call_in_flight():
    # A Claude call can take minutes: a pause drops it rather than waiting for its reply.
    started, release, paused = threading.Event(), threading.Event(), []

    def slow_llm(*args):
        started.set()
        release.wait(10)
        return "late reply"

    def check():
        if paused:
            raise Stopped

    llm = training._checked(slow_llm, check)
    threading.Thread(target=lambda: started.wait(10) and paused.append(1)).start()
    begun = time.monotonic()
    with pytest.raises(Stopped):
        llm("system", "message")
    assert time.monotonic() - begun < 5 and not release.is_set()
    release.set()


def test_a_finished_run_stops_its_leftover_tasks(db, user):
    t = make_training(db, user, rounds=1, scenarios=1)
    check = training._pause_check(t.id, SessionLocal)
    check()
    check.stop()  # what run_training does when it ends, however it ends
    with pytest.raises(Stopped):
        check()


def test_the_next_round_is_prepared_while_the_model_trains(db, user, fake_agents, monkeypatch):
    # With Claude, round 2's revision and personas are written during round 1's SFT, both at once, saving their time.
    together = threading.Barrier(3, timeout=10)  # breaks unless all three are in flight at once
    sft, review, personas = FakeModel.sft, training.agents.review_round, training.agents.generate_personas
    monkeypatch.setattr(FakeModel, "sft", lambda self, ex, **kw: (together.wait() if not self.sft_calls else 0) * 0
                        or sft(self, ex, **kw))
    monkeypatch.setattr(training.agents, "review_round", lambda *a: together.wait() * 0 or review(*a))
    monkeypatch.setattr(training.agents, "generate_personas",
                        lambda *a, **kw: (together.wait() if kw["last_scores"] else 0) * 0 or personas(*a, **kw))
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert (t.status, t.error) == ("completed", None)
    assert "PR" in model.generate_calls


def test_a_failed_sft_does_not_wait_for_the_next_rounds_preparation(db, user, fake_agents, monkeypatch):
    # The training is marked failed and its slot freed now; the stray preparation just finishes and is dropped.
    release = threading.Event()
    monkeypatch.setattr(training.agents, "review_round", lambda *a: release.wait(10) and "R")

    def sft(self, ex, **kw):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(FakeModel, "sft", sft)
    start = time.monotonic()
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert (t.status, t.error) == ("failed", "out of memory")
    assert time.monotonic() - start < 5
    release.set()


def test_local_helpers_prepare_the_next_round_after_training(db, user, fake_agents, local_helper, monkeypatch):
    # A local model answers one call at a time and trains with the same weights, so nothing overlaps its SFT.
    sft, propose = FakeModel.sft, training.agents.propose_local
    monkeypatch.setattr(FakeModel, "sft",
                        lambda self, ex, **kw: fake_agents.events.append("sft") or sft(self, ex, **kw))
    monkeypatch.setattr(training.agents, "propose_local",
                        lambda *a: fake_agents.events.append("propose") or propose(*a))
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2), llm=None)
    assert t.status == "completed"
    messages = ["message persona 1", "message persona 2"]
    assert fake_agents.events == ["personas", *messages, "score", "score", "sft", "score",
                                  "propose", "personas", *messages, *["score"] * 4, "sft", "score"]


def test_a_failed_round_review_is_redone_on_resume(db, user, fake_agents, monkeypatch):
    # The next round's preparation isn't saved until it succeeds, so a resume must redo it from the saved round.
    fake_agents.score = lambda k, prompt, adapter: {1: 8, 2: 6}[k]
    review = training.agents.review_round

    def fail(*a):
        raise RuntimeError("Claude API error 500")

    monkeypatch.setattr(training.agents, "review_round", fail)
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=2, adapter=b"PK-agent"))
    # Round 1 is saved with its trained adapter; only round 2's preparation is lost.
    assert (t.status, t.error, t.scenarios_done, t.personas) == ("failed", "Claude API error 500", 2, None)
    assert db.query(TrainedModel).count() == 2 and model.loads == [b"PK-agent"]

    monkeypatch.setattr(training.agents, "review_round", review)
    t, [model] = resume(db, t)
    assert (t.status, t.scenarios_done) == ("completed", 4)
    # Reviewed from round 1's saved scenarios, just as an uninterrupted run would.
    assert fake_agents.reviews == [("P", [("Clarity", 7.0), ("Tone", 7.0)],
                                    [(conversation(2, adapter="PK-agent"), 6.0),
                                     (conversation(1, adapter="PK-agent"), 8.0)])]
    assert fake_agents.last_scores[-1] == [7.0, 7.0]
    # Resumed on the best model's adapter, not the agent's starting one.
    assert model.loads == [b"PK-1"] and model.generate_calls == ["P", "P", "PR", "PR", "P"]


def test_a_retry_after_the_rounds_scenarios_proposes_nothing(db, user, fake_agents, monkeypatch):
    # With no scenario left to test it on, a revision would be a wasted Claude call that could fail the retry again.
    sft = FakeModel.sft

    def fail(self, ex, **kw):
        if self.sft_calls:
            raise RuntimeError("out of memory")
        return sft(self, ex, **kw)

    monkeypatch.setattr(FakeModel, "sft", fail)
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert (t.status, t.scenarios_done) == ("failed", 4) and len(fake_agents.reviews) == 1

    monkeypatch.setattr(FakeModel, "sft", sft)
    t, _ = resume(db, t)
    assert t.status == "completed" and len(fake_agents.reviews) == 1


def test_a_resume_before_round_1s_scenarios_reuses_its_personas(db, user, fake_agents):
    # Paused right after its personas were saved: they are reused, and round 1 has no revision to test.
    t = make_training(db, user, rounds=1, scenarios=2)
    t.personas = [{"persona": {"prompt": f"persona {k}"}, "message": f"msg from persona {k}"} for k in (1, 2)]
    db.commit()
    t, [model] = run(db, t)
    assert t.status == "completed" and fake_agents.persona_prompts == []
    assert model.generate_calls == ["P"] * 3 and fake_agents.reviews == []


def test_a_partial_round_from_an_older_version_is_finished(db, user, fake_agents):
    # Older versions saved a round a few scenarios at a time: those are kept and only the rest are run, against a
    # revision proposed from them.
    t = make_training(db, user, rounds=1, scenarios=3, next_prompt="P")
    t.personas = [{"persona": {"prompt": f"persona {k}"}, "message": f"msg from persona {k}"} for k in (1, 2, 3)]
    db.add(Scenario(training_id=t.id, round=1, scenario_number=1, persona={"prompt": "persona 1"}, prompt="P",
                    input="msg from persona 1", output="old reply", score=8,
                    scores=[{"criterion": c, "score": 8} for c in RUBRIC]))
    t.scenarios_done = 1
    db.commit()
    t, [model] = run(db, t)
    assert (t.status, t.scenarios_done) == ("completed", 3)
    assert [(s.scenario_number, s.input) for s in scenarios(db)] == [(k, f"msg from persona {k}") for k in (1, 2, 3)]
    assert [r[0] for r in fake_agents.reviews] == ["P"]
    assert model.conversations[:4] == ["msg from persona 2", "msg from persona 3"] * 2


def test_a_training_from_before_the_anchor_eval_keeps_each_new_round(db, user, fake_agents):
    # Its earlier rounds scored practice means, not anchor evals: comparing them would throw away the new SFT.
    t = make_training(db, user, rounds=2, scenarios=2)
    db.add(TrainedModel(training_id=t.id, round=1, prompt="P", weights=b"PK-old", mean_score=9.9))
    db.add_all(Scenario(training_id=t.id, round=1, scenario_number=k, persona={"prompt": f"persona {k}"}, prompt="P",
                        input=f"msg from persona {k}", output="old reply", score=8,
                        scores=[{"criterion": c, "score": 8} for c in RUBRIC]) for k in (1, 2))
    t.adapter = b"PK-old"
    db.commit()
    t, [model] = run(db, t)
    new = db.query(TrainedModel).filter_by(round=2).one()
    assert t.status == "completed" and t.best_model_id == new.id
    assert new.weights == b"PK-2" and model.loads == [b"PK-old"]


def test_teacher_replies_are_not_requested_again_on_resume(db, user, fake_agents, monkeypatch):
    # Teacher replies are paid Claude calls; written ones are saved before SFT, so a failed SFT doesn't repeat them.
    sft = FakeModel.sft

    def fail(self, ex, **kw):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(FakeModel, "sft", fail)
    t, _ = run(db, make_training(db, user, rounds=1, scenarios=3))
    assert t.status == "failed" and len(fake_agents.teacher) == 2

    monkeypatch.setattr(FakeModel, "sft", sft)
    t, [model] = resume(db, t)
    assert t.status == "completed" and len(fake_agents.teacher) == 2
    assert sorted(x["output"] for x in model.sft_calls[0]) == ["ideal reply to msg from persona 2",
                                                               "ideal reply to msg from persona 3"]


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


@pytest.fixture
def battle(db, user, fake_agents):
    """A 2-round battle of an agent whose rounds improve on it on the anchor user (round 2, with prompt "PR", is the
    best). `battle.later` is what round 2's model scores on everyone else (default 9: it is better there too)."""
    agent = Agent(user_id=user.id, name="A", base_model="tiny", prompt="P")
    db.add(agent)
    db.commit()
    state = SimpleNamespace(agent=agent, later=9)
    adapters = {"PK-0": 6, "PK-1": 7, "PK-2": 9, "PK-rival": 8}
    fake_agents.score = lambda k, prompt, adapter: (
        state.later if adapter == "PK-2" and k > 1 else adapters[adapter] + (prompt == "PR" and adapter == "PK-1"))
    state.training = make_training(db, user, rounds=2, scenarios=3, agent_id=agent.id, battle={})
    return state


def test_a_battle_keeps_the_agent_until_its_challenger_grades_higher(db, battle, fake_agents):
    t, models = run(db, battle.training)
    db.refresh(battle.agent)
    challenger = db.query(TrainedModel).filter_by(training_id=t.id, round=2).one()
    # Both answered the same new messages: the champion as the agent started, the challenger as round 2 left it.
    assert t.status == "completed" and t.battle["winner"] == "challenger"
    assert t.battle["grades"] == [6, 9]
    assert [(b["message"], b["champion"].rpartition(" under ")[2], b["challenger"].rpartition(" under ")[2],
             b["grades"]) for b in t.battle["bouts"]] == [
        (f"msg from persona {k}", "<P> with PK-0", "<PR> with PK-2", [6, 9]) for k in (1, 2, 3)]
    assert t.best_model_id == challenger.id
    assert (battle.agent.prompt, battle.agent.weights) == ("PR", b"PK-2")


# The challenger still grades 9 on the anchor user, persona 1, so 4.5 on the other two levels its mean with the champion's 6.
@pytest.mark.parametrize("later", [2, 4.5])
def test_a_champion_that_grades_as_high_keeps_the_agent_unchanged(db, battle, later):
    battle.later = later
    t, _ = run(db, battle.training)
    db.refresh(battle.agent)
    assert t.battle["winner"] == "champion"
    assert db.get(TrainedModel, t.best_model_id).round == 0  # its downloads are the agent as it was
    assert (battle.agent.prompt, battle.agent.weights) == ("P", None)  # the rounds' better models never synced


def test_the_side_winning_more_battle_rounds_keeps_the_agent(db, battle, fake_agents):
    # The challenger's mean grade is higher, but it wins only round 1 (users 1-3) of the battle's three.
    fake_agents.score = lambda k, prompt, adapter: (10 if k <= 3 else 5) if adapter == "PK-2" else {"PK-0": 6, "PK-1": 7}[adapter]
    battle.training.battle = {"rounds": 3}
    db.commit()
    t, _ = run(db, battle.training)
    assert [b["message"] for b in t.battle["bouts"]] == [f"msg from persona {k}" for k in range(1, 10)]
    assert t.battle["round_grades"] == [[6, 10], [6, 5], [6, 5]]
    assert t.battle["grades"][1] > t.battle["grades"][0] and t.battle["winner"] == "champion"


def test_a_pause_during_the_battle_redoes_it_on_resume(db, battle, monkeypatch):
    load = FakeModel.load_adapter

    def pause_at_the_battle(self, data):
        if data == b"PK-0":  # only the battle goes back to the starting weights
            request_pause()
        load(self, data)

    monkeypatch.setattr(FakeModel, "load_adapter", pause_at_the_battle)
    t, _ = run(db, battle.training)
    assert (t.status, t.battle) == ("paused", {})
    monkeypatch.setattr(FakeModel, "load_adapter", load)
    t, _ = resume(db, t)
    assert t.status == "completed" and t.battle["winner"] == "challenger"


def test_a_local_battle_meets_new_users_not_round_1s(db, battle, local_helper):
    t, _ = run(db, battle.training, llm=None)
    # Local personas are seeded by position: round 1 met personas 1-3, so the battle meets 4-6.
    assert [b["message"] for b in t.battle["bouts"]] == [f"msg from persona {k}" for k in (4, 5, 6)]


@pytest.mark.parametrize("rival, winner, kept", [(8, "challenger", ("Be brilliant.", b"PK-rival")),
                                                  (5, "champion", ("P", None))])  # a held agent isn't rewritten
def test_a_challenge_grades_the_published_agent_without_training(db, battle, fake_agents, rival, winner, kept):
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 6, "PK-rival": rival}[adapter]
    t = battle.training
    t.rounds, t.battle = 0, {"opponent": {"post_id": 1, "name": "Rival", "author": "bob"}}
    db.add_all([TrainedModel(training_id=t.id, round=0, prompt="P"),  # an untrained agent: the base weights
                TrainedModel(training_id=t.id, round=1, prompt="Be brilliant.", weights=b"PK-rival")])
    db.commit()
    t, models = run(db, t)
    db.refresh(battle.agent)
    assert models[0].sft_calls == [] and scenarios(db, t) == []
    assert [(b["champion"].rpartition(" under ")[2], b["challenger"].rpartition(" under ")[2])
            for b in t.battle["bouts"]] == [("<P> with PK-0", "<Be brilliant.> with PK-rival")] * 3
    assert (t.status, t.battle["winner"], t.battle["grades"]) == ("completed", winner, [6, rival])
    # The untrained agent's side saved the base weights it ran on, so a win downloads real weights.
    assert db.query(TrainedModel).filter_by(training_id=t.id, round=0).one().weights == b"PK-0"
    assert t.battle["opponent"]["name"] == "Rival"
    assert (battle.agent.prompt, battle.agent.weights) == kept


@pytest.mark.parametrize("rival_base, rival_adapter", [("big", "PK-big"), ("claude", "claude")])
def test_your_agent_on_another_base_model_battles_on_its_own(db, battle, fake_agents, rival_base, rival_adapter):
    # Each side runs on its own base model; the winner moves the agent onto its base model, prompt and weights.
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 6, rival_adapter: 8}[adapter]
    t = battle.training
    t.rounds, t.battle = 0, {"opponent": {"agent_id": 9, "name": "Big", "author": "me", "base_model": rival_base}}
    db.add_all([TrainedModel(training_id=t.id, round=0, prompt="P"),
                TrainedModel(training_id=t.id, round=1, prompt="Be big.")])  # an untrained rival: its base weights
    db.commit()
    loaded = {}

    def load_model(name):
        loaded[name] = FakeModel(name)
        loaded[name].adapter = b"PK-0" if name == "tiny" else b"PK-big"
        return loaded[name]

    training.run_training(t.id, "llm", load_model, session_factory=SessionLocal,
                          agent_llm=lambda system, message: f"reply to {message} under <{system}> with claude")
    db.expire_all()
    t = db.get(Training, t.id)
    db.refresh(battle.agent)
    assert sorted(loaded) == sorted({"tiny", rival_base} - {"claude"})
    assert (t.status, t.battle["winner"], t.battle["grades"]) == ("completed", "challenger", [6, 8])
    weights = None if rival_base == "claude" else b"PK-big"
    assert db.get(TrainedModel, t.best_model_id).weights == weights
    assert (battle.agent.base_model, battle.agent.prompt, battle.agent.weights) == (rival_base, "Be big.", weights)
