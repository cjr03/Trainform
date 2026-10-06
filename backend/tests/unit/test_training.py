import logging
import re
import threading
import time
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import OperationalError

from app import training
from app.agents import FOCUS_HEADER
from app.db import SessionLocal
from app.models import Scenario, TrainedModel, Training, User

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

    def generate(self, system_prompt, conversation):
        self.generate_calls.append(system_prompt)
        self.conversations.append(conversation)
        # The reply names its prompt and adapter, so a score can depend on them however the scoring threads interleave.
        return f"reply to {conversation} under <{system_prompt}> with {self.adapter.decode()}"

    def sft(self, examples, **kwargs):
        self.sft_calls.append(examples)
        self.seeds.append(kwargs["seed"])
        self.adapter = b"PK-%d" % len(self.sft_calls)
        return 0.5

    def adapter_bytes(self):
        return self.adapter

    def load_adapter(self, data):
        self.loads.append(data)
        self.adapter = data


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
            t = s.query(Training).one()
            seen = (t.status, t.current_round, t.scenarios_done, s.query(Scenario).count())
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
    monkeypatch.setattr(training.agents, "primary_respond", lambda generate, prompt, conv: generate(prompt, conv))
    monkeypatch.setattr(training.agents, "score_conversation", score_conversation)
    monkeypatch.setattr(training.agents, "review_round", review_round)
    monkeypatch.setattr(training.agents, "teacher_reply", teacher_reply)
    return state


@pytest.fixture
def local_helper(monkeypatch):
    """No Claude: the helpers run on the loaded model, one call at a time."""
    def local(system, message, schema=None, effort=None):
        pass

    local.local = True
    monkeypatch.setattr(training, "local_llm", lambda model: local)
    return local


def make_training(db, user, rounds, scenarios, rubric=RUBRIC):
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="P", rubric=list(rubric),
                 scenarios_per_round=scenarios, rounds=rounds)
    db.add(t)
    db.commit()
    return t


def run(db, t, llm="llm"):
    models = []

    def load_model(name):
        models.append(FakeModel(name))
        return models[0]

    training.run_training(t.id, llm, load_model, session_factory=SessionLocal)
    db.expire_all()
    return db.get(Training, t.id), models


def scenarios(db):
    return db.query(Scenario).order_by(Scenario.round, Scenario.scenario_number).all()


def example(s, output=None):
    return {"prompt": s.prompt, "input": s.input, "output": output or s.output}


def test_without_claude_the_helper_agents_run_on_the_base_model(db, user, fake_agents, monkeypatch):
    # Loading a second copy of the model for the helpers would double the memory a training needs.
    helpers, used = [], []
    monkeypatch.setattr(training, "local_llm", lambda model: helpers.append(model) or "local llm")
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


def test_local_helper_calls_run_in_turn(db, user, fake_agents, local_helper, monkeypatch):
    # One local model answers one call at a time, so its helper calls must not be made concurrently.
    threads = []
    respond, score = training.agents.persona_respond, training.agents.score_conversation
    monkeypatch.setattr(training.agents, "persona_respond",
                        lambda *a: threads.append(threading.get_ident()) or respond(*a))
    monkeypatch.setattr(training.agents, "score_conversation",
                        lambda *a: threads.append(threading.get_ident()) or score(*a))
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2), llm=None)
    assert t.status == "completed"
    assert len(threads) > 4 and set(threads) == {threading.get_ident()}


def test_runs_every_round_and_threads_the_prompt(db, user, fake_agents):
    # The whole loop end to end with Claude: what each batch runs, what the reviews see, what the API sees meanwhile.
    base = {1: 8, 2: 5, 3: 9, 4: 4, 5: 7, 6: 6}
    # Clarity scores 2 above Tone, and "PR" adds 1 to both.
    fake_agents.score = lambda k, prompt, adapter: [base[k] + (prompt == "PR"), base[k] + (prompt == "PR") - 2]
    t, [model] = run(db, make_training(db, user, rounds=2, scenarios=6))

    assert model.name == "tiny"
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 12)
    rows = scenarios(db)
    assert [(s.round, s.scenario_number) for s in rows] == [(r, k) for r in (1, 2) for k in range(1, 7)]
    # Round 1: "PR", proposed after the first batch, beat "P" on batch 2's messages, so batch 2 is saved under it.
    # Round 2: its revision "PRR" lost on both batches, so every scenario ran under "PR".
    assert [s.prompt for s in rows] == ["P"] * 4 + ["PR"] * 8
    assert [s.score for s in rows] == [7, 4, 8, 3, 7, 6, 8, 5, 9, 4, 7, 6]
    assert model.generate_calls == (["P"] * 6 + ["PR"] * 2 + ["PR"] * 3  # round 1, then its anchors
                                    + ["PR"] * 4 + ["PRR"] * 4 + ["PR"] * 2 + ["PRR"] * 2 + ["PR"] * 3)
    # A review sees its prompt's mean per criterion and lowest-scoring conversations, lowest first: mid-round only
    # the rows under the current prompt; at the round's end the whole round, for the next round's revision.
    assert fake_agents.reviews == [
        ("P", [("Clarity", 6.5), ("Tone", 4.5)],
         [(conversation(4), 3.0), (conversation(2), 4.0), (conversation(1), 7.0)]),
        ("PR", [("Clarity", pytest.approx(41 / 6)), ("Tone", pytest.approx(29 / 6))],
         [(conversation(4), 3.0), (conversation(2), 4.0), (conversation(6, "PR"), 6.0)]),
        ("PR", [("Clarity", 7.5), ("Tone", 5.5)], [(conversation(4, "PR", "PK-1"), 4.0),
                                                     (conversation(2, "PR", "PK-1"), 5.0),
                                                     (conversation(1, "PR", "PK-1"), 8.0)]),
    ]
    # Round 2's personas are written for the prompt it starts with and aimed at round 1's weakest criterion (Tone).
    assert fake_agents.persona_prompts == ["P", "PR"]
    assert fake_agents.last_scores == [None, [pytest.approx(41 / 6), pytest.approx(29 / 6)]]
    assert fake_agents.persona_rubrics == [RUBRIC] * 2  # personas pressure the training's own rubric
    assert model.seeds == [1, 2]

    s = rows[0]
    assert s.persona == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}
    assert (s.input, s.output) == ("msg from persona 1", "reply to msg from persona 1 under <P> with PK-0")
    assert s.scores == [{"criterion": "Clarity", "score": 8}, {"criterion": "Tone", "score": 6}]

    # Progress is committed per batch: status, round, scenarios done and saved scenarios, for every scoring call.
    progress = [(1, 0)] * 4 + [(1, 4)] * 4 + [(1, 6)] * 3 + [(2, 6)] * 8 + [(2, 10)] * 4 + [(2, 12)] * 3
    assert fake_agents.progress == [("running", r, n, n) for r, n in progress]
    # Mid-round, /rounds shows the running round with a partial mean and no model yet.
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    mid = fake_agents.rounds[19]  # round 2's second batch
    assert [(x["round"], x["scenarios"], x["model_id"]) for x in mid] == [(1, 6, trained[1].id), (2, 4, None)]
    assert (mid[0]["mean_score"], mid[1]["mean_score"], mid[0]["eval_score"]) == (
        pytest.approx(35 / 6), 6.5, pytest.approx(22 / 3))


def test_a_revision_replaces_the_prompt_only_when_it_scores_higher_on_the_same_messages(db, user, fake_agents):
    # A paired test on the same users is what makes a higher score the prompt's doing rather than easier users'.
    fake_agents.score = lambda k, prompt, adapter: {"P": 6, "PR": 6.25, "PRR": 6.45}[prompt]
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=12))
    # Batch 2: "PR" gains exactly MIN_GAIN, so it replaces "P". Batch 3: "PRR" gains less, so "PR"'s rows are saved.
    assert model.generate_calls == ["P"] * 8 + ["PR"] * 8 + ["PRR"] * 4 + ["PR"] * 4
    sent = model.conversations
    assert sent[4:8] == sent[8:12] and sent[12:16] == sent[16:20]
    rows = scenarios(db)
    assert [s.prompt for s in rows] == ["P"] * 4 + ["PR"] * 8
    assert all(s.output.endswith("under <PR> with PK-0") for s in rows[4:])
    assert t.next_prompt == "PR" and db.query(TrainedModel).filter_by(round=1).one().prompt == "PR"
    # "PRR" was proposed from the evidence under "PR" only.
    assert [r[0] for r in fake_agents.reviews] == ["P", "PR"]
    assert fake_agents.reviews[1][2] == [(conversation(k, "PR"), 6.25) for k in (5, 6, 7)]


def test_an_unchanged_revision_is_not_tested(db, user, fake_agents, monkeypatch):
    # Running the same prompt twice would only double the batch's cost.
    monkeypatch.setattr(training.agents, "review_round", lambda llm, prompt, *a: prompt)
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=8))
    assert t.status == "completed"
    assert model.generate_calls == ["P"] * 12  # two batches and four anchors


def test_local_helpers_revise_the_prompt_in_code_and_rotate_the_focus(db, user, fake_agents, local_helper, monkeypatch):
    # A small local model can't revise a prompt, so code restates the weakest criteria, and a rejected focus moves on.
    rubric = ["Clarity", "Tone", "Accuracy", "Brevity"]
    fake_agents.score = lambda k, prompt, adapter: [8, 5, 7, 6]  # the same for every prompt: each revision loses
    proposed = []
    propose = training.agents.propose_local
    monkeypatch.setattr(training.agents, "propose_local",
                        lambda prompt, means, n: proposed.append((prompt, n)) or propose(prompt, means, n))
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=12, rubric=rubric), llm=None)
    assert t.status == "completed" and fake_agents.reviews == []
    first, second = (f"P{FOCUS_HEADER}- Tone\n- Brevity", f"P{FOCUS_HEADER}- Accuracy\n- Clarity")
    assert proposed == [("P", 0), ("P", 1)]
    assert model.generate_calls == ["P"] * 8 + [first] * 4 + ["P"] * 4 + [second] * 4 + ["P"] * 4
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
    assert t.best_model_id == trained[1].id and t.adapter == b"PK-1" and model.loads == []


def test_a_worse_round_reverts_to_the_best_adapter(db, user, fake_agents):
    # Training on from an adapter that got worse compounds the damage; the next round starts from the best one.
    fake_agents.score = lambda k, prompt, adapter: {"PK-0": 7, "PK-1": 9, "PK-2": 5, "PK-3": 6}[adapter]
    t, [model] = run(db, make_training(db, user, rounds=3, scenarios=2))
    trained = db.query(TrainedModel).order_by(TrainedModel.round).all()
    assert [(m.round, m.weights, m.mean_score) for m in trained] == [
        (0, b"PK-0", 7), (1, b"PK-1", 9), (2, b"PK-2", 5), (3, b"PK-3", 6)]
    assert model.loads == [b"PK-1", b"PK-1"]  # after rounds 2 and 3
    assert [s.output.rsplit(" ", 1)[1] for s in scenarios(db) if s.round == 3] == ["PK-1"] * 2
    assert t.adapter == b"PK-1" and t.best_model_id == trained[1].id


def test_a_tie_keeps_the_newer_model(db, user, fake_agents):
    # An equally good model that has seen more examples is the better bet, and keeping it saves a reload.
    t, [model] = run(db, make_training(db, user, rounds=1, scenarios=2))
    best = db.get(TrainedModel, t.best_model_id)
    assert (best.round, best.weights) == (1, b"PK-1") and model.loads == []


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
    assert t.scenarios_done == 4 and db.query(Scenario).count() == 4  # the first batch was saved
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
        s.query(Training).update({"status": "pausing"})
        s.commit()


def resume(db, t):
    """What POST /resume does before starting the run again."""
    db.query(Training).filter_by(id=t.id).update({"status": "pending", "error": None})
    db.commit()
    return run(db, t)


def test_failure_mid_round_resumes_from_saved_progress(db, user, fake_agents):
    # A rate limit late in a long training must not cost the scenarios and personas already paid for.
    failing = [True]

    def score(k, prompt, adapter):
        if k == 5 and failing:
            raise RuntimeError("Claude rate limit reached")
        return 8

    fake_agents.score = score
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=6))
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("failed", "Claude rate limit reached", 1, 4)
    # Saved: the prompt the next batch runs with and round 1's personas with their messages; no round finished yet.
    assert t.next_prompt == "P" and t.adapter is None
    assert [x["message"] for x in t.personas] == [f"msg from persona {k}" for k in range(1, 7)]
    assert t.personas[0]["persona"] == {"attributes": [{"criterion": "Clarity", "rating": 1}], "prompt": "persona 1"}

    failing.clear()
    t, [model] = resume(db, t)
    assert (t.status, t.error, t.current_round, t.scenarios_done) == ("completed", None, 2, 12)
    assert [(s.round, s.scenario_number) for s in scenarios(db)] == [(r, k) for r in (1, 2) for k in range(1, 7)]
    # The failed batch is redone under the saved prompt, against a revision proposed again from the rows under it;
    # round 1's personas are reused.
    assert model.conversations[:4] == ["msg from persona 5", "msg from persona 6"] * 2
    assert model.generate_calls[:4] == ["P", "P", "PR", "PR"]
    assert [len(weakest) for _, _, weakest in fake_agents.reviews[:1]] == [3]  # from scenarios 1-4
    assert fake_agents.persona_prompts == ["P", "P"]
    assert fake_agents.events.count("message persona 1") == 2  # once per round


def test_pause_at_a_round_boundary_and_resume(db, user, fake_agents, monkeypatch):
    # A pause during SFT finishes the round, so no training work is thrown away, and keeps the next round's personas.
    sft = FakeModel.sft
    monkeypatch.setattr(FakeModel, "sft", lambda self, examples, **kw: request_pause() or sft(self, examples, **kw))
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2))
    assert (t.status, t.current_round, t.scenarios_done, t.next_prompt) == ("paused", 1, 2, "P")
    assert [x["message"] for x in t.personas] == ["msg from persona 1", "msg from persona 2"]  # prepared ahead
    assert t.adapter == b"PK-1"  # the adapter after round 1's SFT
    assert [m.round for m in db.query(TrainedModel).order_by(TrainedModel.round)] == [0, 1]
    assert fake_agents.persona_prompts == ["P", "P"]

    monkeypatch.setattr(FakeModel, "sft", sft)
    t, [model] = resume(db, t)
    assert (t.status, t.current_round, t.scenarios_done) == ("completed", 2, 4)
    # The saved personas are used; the revision wasn't saved, so it is proposed again from round 1's review.
    assert fake_agents.persona_prompts == ["P", "P"] and len(fake_agents.reviews) == 2
    assert fake_agents.reviews[0] == fake_agents.reviews[1]
    assert model.loads == [b"PK-1"] and model.generate_calls == ["P", "P", "PR", "PR", "P"]
    assert model.seeds == [2]
    assert [m.round for m in db.query(TrainedModel).order_by(TrainedModel.round)] == [0, 1, 2]


def test_pause_mid_round_stops_at_a_batch_boundary(db, user, fake_agents):
    # A batch in flight is finished and saved, so a pause loses nothing, and no further batch starts.
    def score(k, prompt, adapter):
        request_pause()
        return 9

    fake_agents.score = score
    t, _ = run(db, make_training(db, user, rounds=1, scenarios=8))
    # The batch's commit doesn't overwrite "pausing".
    assert (t.status, t.scenarios_done, t.next_prompt) == ("paused", 4, "P")
    assert db.query(Scenario).count() == 4 and db.query(TrainedModel).count() == 0

    fake_agents.score = lambda *a: 8
    t, [model] = resume(db, t)
    assert (t.status, t.scenarios_done) == ("completed", 8)
    assert [s.scenario_number for s in scenarios(db)] == list(range(1, 9))
    assert model.conversations[:4] == [f"msg from persona {k}" for k in range(5, 9)]


def test_pause_before_the_run_starts(db, user, fake_agents):
    # A pause requested while queued must hold; the run must not flip it back to "running".
    t = make_training(db, user, rounds=1, scenarios=1)
    request_pause()
    t, _ = run(db, t)
    assert (t.status, t.current_round) == ("paused", 0)
    assert fake_agents.persona_prompts == []


def test_a_failed_proposal_leaves_the_batch_saved(db, user, fake_agents, monkeypatch):
    # A batch is committed before its revision is proposed, so a Claude error doesn't cost the scored scenarios.
    def fail(*a):
        raise RuntimeError("Claude API error 500")

    monkeypatch.setattr(training.agents, "review_round", fail)
    t, _ = run(db, make_training(db, user, rounds=1, scenarios=8))
    assert (t.status, t.error, t.scenarios_done, t.next_prompt) == ("failed", "Claude API error 500", 4, "P")
    assert db.query(Scenario).count() == 4


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
    t, _ = run(db, make_training(db, user, rounds=2, scenarios=2))
    # Round 1 is saved with its trained adapter; only round 2's preparation is lost.
    assert (t.status, t.error, t.scenarios_done, t.personas) == ("failed", "Claude API error 500", 2, None)
    assert t.adapter == b"PK-1" and db.query(TrainedModel).count() == 2

    monkeypatch.setattr(training.agents, "review_round", review)
    t, [model] = resume(db, t)
    assert (t.status, t.scenarios_done) == ("completed", 4)
    # Reviewed from round 1's saved scenarios, just as an uninterrupted run would.
    assert fake_agents.reviews == [("P", [("Clarity", 7.0), ("Tone", 7.0)],
                                    [(conversation(2), 6.0), (conversation(1), 8.0)])]
    assert fake_agents.last_scores[-1] == [7.0, 7.0]
    assert model.generate_calls == ["P", "P", "PR", "PR", "P"]


def test_a_retry_after_the_rounds_scenarios_proposes_nothing(db, user, fake_agents, monkeypatch):
    # With no batch left to test it on, a revision would be a wasted Claude call that could fail the retry again.
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


def test_a_resume_before_round_1s_first_batch_reuses_its_personas(db, user, fake_agents):
    # Paused right after its personas were saved: they are reused, and round 1 has no revision to test.
    t = make_training(db, user, rounds=1, scenarios=2)
    t.personas = [{"persona": {"prompt": f"persona {k}"}, "message": f"msg from persona {k}"} for k in (1, 2)]
    db.commit()
    t, [model] = run(db, t)
    assert t.status == "completed" and fake_agents.persona_prompts == []
    assert model.generate_calls == ["P"] * 3 and fake_agents.reviews == []


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
    assert t.adapter == b"PK-1" and model.loads == [b"PK-old"]


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
