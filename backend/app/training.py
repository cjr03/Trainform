"""Training orchestration plus the train/inspect API."""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from statistics import fmean, median
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app import agent_profiles, agents, fake
from app.auth import current_user
from app.db import SessionLocal, get_db
from app.llm import Helper, LLMError, Stopped, chosen_llm, get_llm, local_llm, unload_route_model
from app.local_model import LocalModel, free_memory, list_local_models
from app.models import ACTIVE, Agent, Post, Scenario, TrainedModel, Training, User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/trainings", tags=["training"])

SFT_FLOOR = 6.0  # a reply scoring below this is never learned from, however it ranks in its round
REPLAY = 24  # most examples a round's SFT trains on: this round's, then earlier rounds' best, so it doesn't forget them
MIN_GAIN = 0.25  # how much higher a revision's mean score on a round's messages must be to replace the prompt
ANCHORS = 4  # round 1's first scenarios, held out of SFT and rerun after each round to compare models like for like
WEAKEST = 3  # lowest-scoring conversations the prompt generator sees
INTERNAL_ERROR = "Training failed; see the server logs."
PAUSE_POLL = 0.25  # seconds
# Claude calls run here while their caller watches for a pause; a dropped call finishes here unwatched.
_CALLS = ThreadPoolExecutor(32)  # ponytail: shared by all runs; per-run pools if many users pause at once between a run's checks for a pause request
# Each run holds a loaded model and one of the 40 worker threads the sync routes share, so cap them process-wide.
# One at a time by default: two local trainings at once don't fit in 16 GiB.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TRAINFORM_MAX_TRAININGS", "1")))


def _set_status(db: Session, t: Training, old: tuple, new: str, **values) -> bool:
    """Move `t` from one of the `old` statuses to `new` in one UPDATE, so a concurrent change can't be lost."""
    changed = db.execute(update(Training).where(Training.id == t.id, Training.status.in_(old))
                         .values(status=new, **values)).rowcount
    db.commit()
    return bool(changed)


def _conversation(message: str, reply: str) -> str:
    return f"User: {message}\nAssistant: {reply}"


def _evaluate(llm, respond, prompts: list[str], rubric: list[str], messages: list[str]) -> list[list[dict]]:
    """The replies to `messages` under each of `prompts`, generated all at once and scored all at once: per prompt,
    [{"input", "output", "scores", "score"}]."""
    pairs = [(prompt, m) for prompt in prompts for m in messages]
    replies = respond(pairs)
    scores = agents.score_conversations(llm, [_conversation(m, r) for (_, m), r in zip(pairs, replies)], rubric)
    rows = [{"input": m, "output": r, "scores": sc, "score": fmean(x["score"] for x in sc)}
            for (_, m), r, sc in zip(pairs, replies, scores)]
    return [rows[i:i + len(messages)] for i in range(0, len(rows), len(messages))]


def _review(rubric: list[str], rows: list[Scenario], prompt: str) -> tuple:
    """`prompt` with its mean score per criterion and its lowest-scoring conversations over `rows`, for proposing a
    revision of it. Plain values, so another thread can use them."""
    means = [fmean(s.scores[i]["score"] for s in rows) for i in range(len(rubric))]
    weakest = [(_conversation(s.input, s.output), s.score) for s in sorted(rows, key=lambda s: s.score)[:WEAKEST]]
    return prompt, list(zip(rubric, means)), weakest


def _propose(llm, review: tuple, rejected: int) -> str:
    """A revision of the reviewed prompt to test against it: Claude writes one; the local helper can't, so code does."""
    if getattr(llm, "local", False):
        return agents.propose_local(review[0], review[1], rejected)
    return agents.review_round(llm, *review)


def _prepare_round(llm, rubric: list[str], n: int, prompt: str, review: tuple | None,
                   rejected: int = 0) -> tuple[str | None, list[dict]]:
    """A round's revision of `prompt` to test (None in round 1) and its personas with their messages
    ([{"persona", "message"}]). After round 1, `review` (see _review) has the last round's results: the revision
    targets its weakest criteria, and so do the personas."""
    last_scores = [mean for _, mean in review[1]] if review else None
    tasks = [lambda: _propose(llm, review, rejected) if review else None,
             lambda: agents.generate_personas(llm, prompt, n, rubric=rubric, last_scores=last_scores)]
    # Personas need only the agent's role, which a revision keeps, so with Claude both are written at once.
    candidate, personas = agents.concurrently(llm, lambda task: task(), tasks)
    messages = agents.persona_messages(llm, personas)
    return candidate, [{"persona": p, "message": m} for p, m in zip(personas, messages)]


def _rejected(db: Session, t: Training, r: int) -> int:
    """How many rounds in a row before round `r` kept their prompt (their revision lost), so the local helper's next
    revision moves on to other criteria. Round 1 tests no revision."""
    prompts = dict(db.execute(select(TrainedModel.round, TrainedModel.prompt)
                              .where(TrainedModel.training_id == t.id)).all())
    n = 0
    while r - n > 2 and prompts.get(r - n - 1) == prompts.get(r - n - 2):
        n += 1
    return n


def _pause_check(training_id: int, session_factory):
    """A check that raises Stopped once the user has asked to pause, reading the status (on its own connection, so
    it never waits on the run's transaction) at most every PAUSE_POLL seconds."""
    last, stopped = [float("-inf")], []

    def check():
        if stopped:  # sticky: a task still running for a stopped run (the next round's preparation) stops too
            raise Stopped
        if time.monotonic() - last[0] < PAUSE_POLL:
            return
        last[0] = time.monotonic()
        with session_factory() as s:
            if s.scalar(select(Training.status).where(Training.id == training_id)) == "pausing":
                stopped.append(True)
                raise Stopped

    check.stop = lambda: stopped.append(True)
    return check


def _checked(llm, check):
    """`llm` (Claude), running `check` before and while each call runs, so a pause starts no more of them and stops
    waiting on those in flight: their replies are dropped (a resume redoes them)."""
    @wraps(llm)
    def call(*args, **kwargs):
        check()
        reply = _CALLS.submit(llm, *args, **kwargs)
        while True:
            try:
                return reply.result(timeout=0.25)
            except TimeoutError:
                check()

    return call


def _anchors(db: Session, t: Training) -> list[Scenario]:
    """Round 1's first scenarios: the same users every round's model is scored on (at most half of round 1, so it
    still has scenarios to learn from)."""
    k = min(ANCHORS, max(1, t.scenarios_per_round // 2))
    return list(db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == 1,
                                                  Scenario.scenario_number <= k).order_by(Scenario.scenario_number)))


def _examples(db: Session, t: Training, held_out: set[int]) -> list[dict]:
    """What a round's SFT learns: each round's teacher replies (for its weakest replies first) and its own replies
    in its top half (and at least SFT_FLOOR, best first), the latest round first, up to REPLAY. Read from the saved
    scenarios, so a resume trains the same."""
    rows = sorted((s for s in db.scalars(select(Scenario).where(Scenario.training_id == t.id)) if s.id not in held_out),
                  key=lambda s: (-s.round, s.score if s.target else -s.score, s.scenario_number))
    bars = {r: max(SFT_FLOOR, median(s.score for s in rows if s.round == r)) for r in {s.round for s in rows}}
    return [{"prompt": s.prompt, "input": s.input, "output": s.target or s.output}
            for s in rows if s.target or s.score >= bars[s.round]][:REPLAY]


def _battle(db: Session, t: Training, llm, model, respond, base: bytes | None) -> None:
    """End a battle: the agent as it started (the champion, round 0) and the challenger (the best model its rounds
    trained, or the published agent it challenges, round 1) answer the same new users and are graded on the rubric.
    Each of the battle's rounds is n new users, won by the side with the higher mean grade on them, and the challenger
    replaces the agent only if it wins more rounds (so a tie keeps the champion). Either way the winner is the training's
    best model, so its downloads are the winner's. A side without weights runs on `base`, the base model's. Run in one
    go: a pause drops it and a resume redoes it."""
    models = select(TrainedModel).where(TrainedModel.training_id == t.id)
    champion = db.scalar(models.where(TrainedModel.round == 0))
    challenger = db.scalar(models.where(TrainedModel.round > 0)
                           .order_by(TrainedModel.mean_score.desc(), TrainedModel.round.desc()).limit(1))
    n, rounds = t.scenarios_per_round, t.battle.get("rounds", 1)
    # A local helper's personas are seeded by position, so its first n are round 1's users: the battle takes the next.
    count = (rounds + getattr(llm, "local", False)) * n
    _, personas = _prepare_round(llm, t.rubric, count, champion.prompt, None)
    messages = [p["message"] for p in personas[-rounds * n:]]
    sides = []
    for m in (champion, challenger):
        if model:
            m.weights = m.weights or base  # saved, so the winner's download and the agent get real weights
            model.load_adapter(m.weights)
        sides.append(_evaluate(llm, respond, [m.prompt], t.rubric, messages)[0])
    grades = [fmean(row["score"] for row in rows) for rows in sides]
    round_grades = [[fmean(row["score"] for row in rows[i:i + n]) for rows in sides] for i in range(0, rounds * n, n)]
    won = sum(b > a for a, b in round_grades) > sum(a > b for a, b in round_grades)
    t.battle = t.battle | {"winner": "challenger" if won else "champion", "grades": grades, "round_grades": round_grades,
                            "bouts": [{"message": a["input"], "champion": a["output"], "challenger": b["output"],
                                       "grades": [a["score"], b["score"]]} for a, b in zip(*sides)]}
    t.best_model_id = (challenger if won else champion).id
    if won:
        agent_profiles.sync_agent(db, t, challenger.id)


def run_training(training_id: int, llm, load_model, session_factory=SessionLocal, agent_llm=None) -> None:
    """Run a training's remaining rounds, continuing from its saved progress (models.Training.next_prompt and
    personas, the round's saved scenarios and its finished rounds' TrainedModels, the best of which holds the adapter
    to go on from). `load_model(name)` returns a LocalModel-like object. `agent_llm` (the user's Claude) is the agent
    when its base model is "claude": then only the prompt is trained. The API writes status too ("pausing"), so this
    only changes it with targeted UPDATEs.

    Each round runs all its scenarios at once under the prompt and a revision of it, keeps the revision only if it
    scored higher on the same messages, then fine-tunes. Each round's model is scored on the same anchor users as the
    ones before it and dropped if it scores lower than the best, so the model never trains on from a worse one. A
    pause drops the task in flight; a resume redoes it from the saved progress."""
    pool = ThreadPoolExecutor(1)
    check = _pause_check(training_id, session_factory)
    with session_factory() as db:
        try:
            t = db.get(Training, training_id)
            _set_status(db, t, ("pending",), "running")  # left alone if the user already asked to pause
            if t.base_model == "claude":  # Claude's weights can't be fine-tuned: no teacher, no SFT, no weights
                if not (llm and agent_llm):
                    raise LLMError("Claude agents need Claude as the helper")
                model = weights = base = None
                agent = _checked(agent_llm, check)
                respond = lambda pairs: agents.concurrently(agent, lambda pair: agent(*pair), pairs)  # noqa: E731
            else:
                model = load_model(t.base_model)
                base = model.adapter_bytes() if t.battle is not None else None  # its untrained adapter
                # What the next round's scenarios run with: the best model's adapter, else the agent's starting one.
                weights = t.best_model_id and db.scalar(select(TrainedModel.weights)
                                                        .where(TrainedModel.id == t.best_model_id))
                weights = weights or t.adapter
                if weights:
                    model.load_adapter(weights)
                else:
                    weights = model.adapter_bytes()
                respond = partial(model.generate_many, stop=check)
            # No Claude: the helpers share the loaded model, with its adapter off.
            llm = _checked(llm, check) if llm else local_llm(model, stop=check)
            local = getattr(llm, "local", False)
            rubric, prompt = t.rubric, t.next_prompt or t.initial_prompt
            finished = db.scalar(select(func.count()).where(TrainedModel.training_id == t.id, TrainedModel.round > 0))
            ahead = None  # with Claude, the next round's revision and personas, prepared while the model trains
            for r in range(finished + 1, t.rounds + 1):
                candidate = None  # a revision of the prompt, tested this round (not saved: a resume re-proposes)
                if ahead:
                    candidate, t.personas = ahead.result()
                    db.commit()
                if _set_status(db, t, ("pausing",), "paused"):  # a safe point: progress is saved
                    return
                t.current_round = r
                db.commit()
                # Rows of a partial round are left from an older version, which saved scenarios a few at a time.
                done = list(db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r)
                                       .order_by(Scenario.scenario_number)))
                if t.personas is None or (candidate is None and len(done) < len(t.personas)):
                    # Resumed mid-round, the revision isn't saved: proposed again from the rows it was proposed from.
                    last = [s for s in done if s.prompt == prompt] or db.scalars(
                        select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r - 1)
                        .order_by(Scenario.scenario_number)).all()
                    review = _review(rubric, last, prompt) if last else None
                    rejected = _rejected(db, t, r) if local else 0
                    if t.personas is None:
                        candidate, t.personas = _prepare_round(llm, rubric, t.scenarios_per_round, prompt, review,
                                                               rejected)
                    elif review:
                        candidate = _propose(llm, review, rejected)
                    db.commit()
                todo = t.personas[len(done):]
                if todo:
                    tested = [prompt] + ([candidate] if candidate and candidate != prompt else [])
                    rows, *tried = _evaluate(llm, respond, tested, rubric, [p["message"] for p in todo])
                    # A paired test: the same users, so a higher score is the prompt's doing, not theirs.
                    if tried and fmean(x["score"] for x in tried[0]) >= fmean(x["score"] for x in rows) + MIN_GAIN:
                        prompt, rows = candidate, tried[0]
                    for p, row in zip(todo, rows):
                        done.append(Scenario(training_id=t.id, round=r, scenario_number=len(done) + 1,
                                             persona=p["persona"], prompt=prompt, **row))
                        db.add(done[-1])
                    # Saved together with the prompt they ran under: a pause or failure before this redoes them all.
                    t.scenarios_done += len(todo)
                    t.next_prompt = prompt
                    db.commit()
                # A local model can't answer while it trains, so it prepares the next round at its start instead.
                ahead = None
                if r < t.rounds and not local:
                    ahead = pool.submit(_prepare_round, llm, rubric, t.scenarios_per_round, prompt,
                                        _review(rubric, done, prompt))
                anchors = _anchors(db, t)
                held_out = {a.id for a in anchors}
                if model and not local:  # Claude writes the replies the model should have sent; a resume skips them
                    todo = [s for s in done if s.target is None and s.id not in held_out]
                    targets = agents.concurrently(
                        llm, lambda s: agents.teacher_reply(llm, s.prompt, s.input, rubric), todo)
                    for s, target in zip(todo, targets):
                        s.target = target
                    db.commit()
                if r == 1:  # the anchors ran under the initial prompt and weights: the baseline to beat, for free
                    db.add(TrainedModel(training_id=t.id, round=0, prompt=t.initial_prompt, weights=weights,
                                        mean_score=fmean(a.score for a in anchors)))
                examples = _examples(db, t, held_out) if model else []
                if examples:
                    model.sft(examples, seed=r, stop=check)
                    weights = model.adapter_bytes()
                [evaluated] = _evaluate(llm, respond, [prompt], rubric, [a.input for a in anchors])
                db.add(trained := TrainedModel(training_id=t.id, round=r, prompt=prompt, weights=weights,
                                               mean_score=fmean(x["score"] for x in evaluated)))
                db.flush()
                # A training started before the anchor eval has no baseline and practice means as its scores: no gate.
                best_id, best_round = db.execute(
                    select(TrainedModel.id, TrainedModel.round).where(TrainedModel.training_id == t.id)
                    .order_by(TrainedModel.mean_score.desc(), TrainedModel.round.desc()).limit(1)).one()
                if not db.scalar(select(func.count()).where(TrainedModel.training_id == t.id, TrainedModel.round == 0)):
                    best_id, best_round = trained.id, r
                t.best_model_id = best_id
                if best_round != r:  # worse than the best so far (a tie keeps the newer): go on from that one
                    if model:
                        weights = db.scalar(select(TrainedModel.weights).where(TrainedModel.id == best_id))
                        model.load_adapter(weights)
                    else:  # only the prompt is trained, so the best one is kept
                        prompt = t.next_prompt = db.scalar(select(TrainedModel.prompt)
                                                           .where(TrainedModel.id == best_id))
                if t.battle is None:  # a battle's agent changes only if its challenger wins (see _battle)
                    agent_profiles.sync_agent(db, t)
                # Committed with the round's model, so a resume starts the next round on the kept adapter.
                t.personas = None
                db.commit()
            if t.battle is not None:
                _battle(db, t, llm, model, respond, base)
            t.status = "completed"
            db.commit()
        except Exception as exc:
            stopped = isinstance(exc, Stopped)  # paused: the task in flight is dropped, its progress is saved
            if not stopped:
                log.exception("Training %s failed", training_id)
            # File paths and SQL are server internals; LLM/agent errors are what the user can act on.
            error = INTERNAL_ERROR if isinstance(exc, (OSError, SQLAlchemyError)) else str(exc)[:1000]
            try:
                db.rollback()
                db.execute(update(Training).where(Training.id == training_id)
                           .values(**({"status": "paused"} if stopped else {"status": "failed", "error": error})))
                db.commit()
            except Exception:
                log.exception("Could not mark training %s %s", training_id, "paused" if stopped else "failed")
        finally:
            # Don't hold the slot and the model for a next round that won't run; a resume prepares it again.
            check.stop()
            pool.shutdown(wait=False, cancel_futures=True)


def score_windows(scores: list[float]) -> list[dict]:
    """Mean score over the last n, n/2, n/4 (at least 1) and last runs."""
    n = len(scores)
    if not n:
        return []
    windows = [("n", n), ("n/2", max(1, n // 2)), ("n/4", max(1, n // 4)), ("last", 1)]
    return [{"label": label, "runs": k, "score": fmean(scores[-k:])} for label, k in windows]


class TrainingIn(BaseModel):
    agent_id: int
    rubric: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(
        min_length=1, max_length=30)
    scenarios: int = Field(ge=1, le=100)
    rounds: int = Field(ge=1, le=50)
    helper: Helper | None = None
    battle: bool = False  # optimise a challenger, then keep whichever of it and the agent wins a head to head
    opponent_post_id: int | None = None  # a battle against this published agent instead: no rounds, no challenger
    battle_rounds: int = Field(1, ge=1, le=7)  # a battle's head-to-head rounds, each on new users (see _battle)


def owned(db: Session, training_id: int, user: User) -> Training:
    t = db.get(Training, training_id)
    if t is None or t.user_id != user.id:
        raise HTTPException(404, "Training not found")
    return t


def best_model(db: Session, t: Training) -> TrainedModel:
    if t.best_model_id is None:
        raise HTTPException(404, "Training has no trained model yet")
    return db.get(TrainedModel, t.best_model_id)


def detail(db: Session, t: Training) -> dict:
    scores = db.scalars(select(Scenario.score).where(Scenario.training_id == t.id)
                        .order_by(Scenario.round, Scenario.scenario_number)).all()
    latest_prompt = db.scalar(select(TrainedModel.prompt).where(TrainedModel.training_id == t.id)
                              .order_by(TrainedModel.round.desc()).limit(1))
    best_score, best_prompt = db.execute(select(TrainedModel.mean_score, TrainedModel.prompt)
                                         .where(TrainedModel.id == t.best_model_id)).first() or (None, None)
    baseline_score = db.scalar(select(TrainedModel.mean_score).where(TrainedModel.training_id == t.id,
                                                                     TrainedModel.round == 0))
    return {
        "id": t.id, "status": t.status, "error": t.error, "base_model": t.base_model,
        "initial_prompt": t.initial_prompt, "rubric": t.rubric, "rounds": t.rounds,
        "scenarios_per_round": t.scenarios_per_round, "current_round": t.current_round, "scenarios_done": t.scenarios_done,
        "total_scenarios": t.rounds * t.scenarios_per_round, "latest_prompt": latest_prompt or t.initial_prompt,
        "score_windows": score_windows(scores), "best_model_id": t.best_model_id, "best_score": best_score,
        "best_prompt": best_prompt, "baseline_score": baseline_score, "created_at": t.created_at, "helper": t.helper,
        "agent_id": t.agent_id, "agent_name": db.scalar(select(Agent.name).where(Agent.id == t.agent_id)),
        "battle": t.battle,
    }


def _start(background: BackgroundTasks, db: Session, llm, write, agent_llm) -> dict:
    """Take a training slot, run `write()`, which commits a pending Training and returns it, and run it in the
    background with helper `llm` and, for a Claude agent, `agent_llm`. Returns the training's detail."""
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(429, "The server is busy with other trainings; try again later.")
    try:
        t = write()
        out = detail(db, t)
    except Exception as e:
        _SLOTS.release()
        if isinstance(e, IntegrityError):  # models.ONE_ACTIVE_TRAINING
            raise HTTPException(409, "You already have a training in progress.") from None
        raise
    db.close()  # get_db only exits after background tasks, so return the connection to the pool now
    # ponytail: each run holds a worker thread and its request open, so shutdown waits for running trainings, and a
    # response that never finishes sending would keep its slot. Move runs to a worker process if that bites.
    background.add_task(_run_in_slot, t.id, llm, fake.FakeLocalModel if fake.enabled() else LocalModel, SessionLocal,
                        agent_llm)
    return out


def _run_in_slot(*args) -> None:
    try:
        unload_route_model()  # the run loads its own copy
        run_training(*args)
    finally:
        _SLOTS.release()
        free_memory()  # else torch keeps the run's peak cached, unusable by anything else


def _helper(base_model: str, helper: Helper | None, llm):
    """The helper llm for a training of a `base_model` agent (see chosen_llm); a Claude agent needs Claude."""
    if base_model == "claude" and (helper == "local" or not llm):
        raise HTTPException(400, "Claude agents need Claude as the helper")
    return chosen_llm(helper, llm)


@router.post("", status_code=201)
def create_training(body: TrainingIn, background: BackgroundTasks, user: User = Depends(current_user),
                    llm=Depends(get_llm), db: Session = Depends(get_db)):
    agent = agent_profiles.owned(db, body.agent_id, user)
    if agent.base_model != "claude" and agent.base_model not in {m["name"] for m in list_local_models(user.id)}:
        raise HTTPException(400, f"Unknown base model: {agent.base_model}")
    helper = _helper(agent.base_model, body.helper, llm)
    post = body.opponent_post_id and db.get(Post, body.opponent_post_id)
    if body.opponent_post_id and not post:
        raise HTTPException(404, "Post not found")
    if post and post.base_model != agent.base_model:  # its weights wouldn't fit your agent
        raise HTTPException(400, f"Only an agent on {agent.base_model} can battle yours; this one is on "
                                 f"{post.base_model}.")

    def insert():
        # The agent's weights are where it starts (see run_training).
        t = Training(user_id=user.id, agent_id=agent.id, base_model=agent.base_model, initial_prompt=agent.prompt,
                     adapter=agent.weights, rubric=body.rubric, scenarios_per_round=body.scenarios,
                     rounds=0 if post else body.rounds, helper=body.helper,
                     battle={"rounds": body.battle_rounds} | (
                         {"opponent": {"post_id": post.id, "name": post.name,
                                       "author": db.scalar(select(User.username).where(User.id == post.user_id))}}
                         if post else {}) if post or body.battle else None)
        db.add(t)
        if post:  # the two sides, copied now so later edits or deletes don't change the battle
            db.flush()
            db.add_all([TrainedModel(training_id=t.id, round=0, prompt=agent.prompt, weights=agent.weights),
                        TrainedModel(training_id=t.id, round=1, prompt=post.prompt, weights=post.weights)])
        db.commit()
        return t

    return _start(background, db, helper, insert, llm)


@router.post("/{training_id}/pause")
def pause_training(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Ask the run to stop at its next safe point, where it saves its progress and becomes "paused"."""
    t = owned(db, training_id, user)
    if not _set_status(db, t, ("pending", "running"), "pausing"):
        raise HTTPException(409, "Only a pending or running training can be paused.")
    return detail(db, t)


@router.post("/{training_id}/resume")
def resume_training(training_id: int, background: BackgroundTasks, user: User = Depends(current_user),
                    llm=Depends(get_llm), db: Session = Depends(get_db)):
    """Continue a paused or failed training from its saved progress (the UI's Retry), with the helper it was started
    with (or, if none was chosen, the user's current one)."""
    t = owned(db, training_id, user)
    if t.status not in ("paused", "failed"):
        raise HTTPException(409, "Only a paused or failed training can be resumed.")
    helper = _helper(t.base_model, t.helper, llm)

    def restart():
        # Only one request can make the change, so a training never gets a second run.
        if not _set_status(db, t, ("paused", "failed"), "pending", error=None):
            raise HTTPException(409, "Only a paused or failed training can be resumed.")
        return t

    return _start(background, db, helper, restart, llm)


@router.delete("/{training_id}", status_code=204)
def delete_training(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = owned(db, training_id, user)
    # Its scenarios and trained models go with it (ON DELETE CASCADE).
    if not db.execute(delete(Training).where(Training.id == t.id, Training.status.not_in(ACTIVE))).rowcount:
        raise HTTPException(409, "Pause the training before deleting it.")
    db.commit()


@router.get("")
def list_trainings(agent_id: int | None = None, user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = (select(Training, TrainedModel.mean_score, Agent.name)
         .outerjoin(TrainedModel, TrainedModel.id == Training.best_model_id)
         .outerjoin(Agent, Agent.id == Training.agent_id)
         .where(Training.user_id == user.id).order_by(Training.id.desc()))
    rows = db.execute(q if agent_id is None else q.where(Training.agent_id == agent_id)).all()
    return [{"id": t.id, "base_model": t.base_model, "status": t.status, "rounds": t.rounds,
             "scenarios_per_round": t.scenarios_per_round, "current_round": t.current_round,
             "scenarios_done": t.scenarios_done, "created_at": t.created_at, "best_score": best_score,
             "helper": t.helper, "agent_id": t.agent_id, "agent_name": agent_name,
             "battle": t.battle and {k: v for k, v in t.battle.items() if k != "bouts"}}  # past battles' results
            for t, best_score, agent_name in rows]


@router.get("/{training_id}")
def get_training(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    return detail(db, owned(db, training_id, user))


@router.get("/{training_id}/rounds")
def list_rounds(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = owned(db, training_id, user)
    models = {r: (i, score) for r, i, score in db.execute(
        select(TrainedModel.round, TrainedModel.id, TrainedModel.mean_score).where(TrainedModel.training_id == t.id))}
    rows = db.execute(select(Scenario.round, func.avg(Scenario.score), func.count()).where(Scenario.training_id == t.id)
                      .group_by(Scenario.round).order_by(Scenario.round)).all()
    # mean_score: the round's practice users, who press harder on its weak spots each round; eval_score: its model on
    # the same anchor users as every other round, so it is the one to compare.
    return [{"round": r, "mean_score": mean, "scenarios": count, "model_id": models.get(r, (None, None))[0],
             "eval_score": models.get(r, (None, None))[1]} for r, mean, count in rows]


@router.get("/{training_id}/rounds/{round}/scenarios")
def list_scenarios(training_id: int, round: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = owned(db, training_id, user)
    rows = db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == round)
                      .order_by(Scenario.scenario_number))
    return [{"id": s.id, "scenario_number": s.scenario_number, "persona": s.persona, "prompt": s.prompt,
             "input": s.input, "output": s.output, "scores": s.scores, "score": s.score} for s in rows]


@router.get("/{training_id}/download/prompt")
def download_prompt(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    m = best_model(db, owned(db, training_id, user))
    return PlainTextResponse(m.prompt, headers={
        "Content-Disposition": f'attachment; filename="training-{training_id}-prompt.txt"'})


@router.get("/{training_id}/download/weights")
def download_weights(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    m = best_model(db, owned(db, training_id, user))
    return Response(m.weights, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="training-{training_id}-lora.zip"'})
