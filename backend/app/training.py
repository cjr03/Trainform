"""Training orchestration plus the train/inspect API."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean, median
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app import agents, fake
from app.auth import current_user
from app.db import SessionLocal, get_db
from app.llm import Helper, chosen_llm, get_llm, local_llm
from app.local_model import LocalModel, list_local_models
from app.models import ACTIVE, Scenario, TrainedModel, Training, User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/trainings", tags=["training"])

SFT_FLOOR = 6.0  # a reply scoring below this is never learned from, however it ranks in its round
REPLAY = 24  # most examples a round's SFT trains on: this round's, then earlier rounds' best, so it doesn't forget them
MINIBATCH = 4  # scenarios a revised prompt is tested on, against the current prompt, before it may replace it
MIN_GAIN = 0.25  # how much higher a revision's mean score on the same messages must be to replace the prompt
ANCHORS = 4  # round 1's first scenarios, held out of SFT and rerun after each round to compare models like for like
WEAKEST = 3  # lowest-scoring conversations the prompt generator sees
INTERNAL_ERROR = "Training failed; see the server logs."
# Each run holds a loaded model and one of the 40 worker threads the sync routes share, so cap them process-wide.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TRAINFORM_MAX_TRAININGS", "2")))


def _set_status(db: Session, t: Training, old: tuple, new: str, **values) -> bool:
    """Move `t` from one of the `old` statuses to `new` in one UPDATE, so a concurrent change can't be lost."""
    changed = db.execute(update(Training).where(Training.id == t.id, Training.status.in_(old))
                         .values(status=new, **values)).rowcount
    db.commit()
    return bool(changed)


def _conversation(message: str, reply: str) -> str:
    return f"User: {message}\nAssistant: {reply}"


def _evaluate(llm, model, prompt: str, rubric: list[str], messages: list[str]) -> list[dict]:
    """The model's replies to `messages` under `prompt`, scored: [{"input", "output", "scores", "score"}]."""
    replies = [agents.primary_respond(model.generate, prompt, m) for m in messages]  # one model: in turn
    scores = agents.concurrently(llm, lambda c: agents.score_conversation(llm, c, rubric),
                                 [_conversation(m, r) for m, r in zip(messages, replies)])
    return [{"input": m, "output": r, "scores": sc, "score": fmean(x["score"] for x in sc)}
            for m, r, sc in zip(messages, replies, scores)]


def _review(rubric: list[str], rows: list[Scenario], prompt: str) -> tuple:
    """`prompt` with its mean score per criterion and its lowest-scoring conversations over `rows`, for proposing a
    revision of it. Plain values, so another thread can use them."""
    means = [fmean(s.scores[i]["score"] for s in rows) for i in range(len(rubric))]
    weakest = [(_conversation(s.input, s.output), s.score) for s in sorted(rows, key=lambda s: s.score)[:WEAKEST]]
    return prompt, list(zip(rubric, means)), weakest


def _propose(llm, review: tuple, rejected: int = 0) -> str:
    """A revision of the reviewed prompt to test against it: Claude writes one; the local helper can't, so code does."""
    if getattr(llm, "local", False):
        return agents.propose_local(review[0], review[1], rejected)
    return agents.review_round(llm, *review)


def _prepare_round(llm, rubric: list[str], n: int, prompt: str, review: tuple | None) -> tuple[str | None, list[dict]]:
    """A round's revision of `prompt` to test (None in round 1) and its personas with their messages
    ([{"persona", "message"}]). After round 1, `review` (see _review) has the last round's results: the revision
    targets its weakest criteria, and so do the personas."""
    last_scores = [mean for _, mean in review[1]] if review else None
    tasks = [lambda: _propose(llm, review) if review else None,
             lambda: agents.generate_personas(llm, prompt, n, rubric=rubric, last_scores=last_scores)]
    # Personas need only the agent's role, which a revision keeps, so with Claude both are written at once.
    candidate, personas = agents.concurrently(llm, lambda task: task(), tasks)
    # A persona's message depends only on the persona, so the round's messages are written at once.
    messages = agents.concurrently(llm, lambda p: agents.persona_respond(llm, p["attributes"], p["prompt"]), personas)
    return candidate, [{"persona": p, "message": m} for p, m in zip(personas, messages)]


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


def run_training(training_id: int, llm, load_model, session_factory=SessionLocal) -> None:
    """Run a training's remaining rounds, continuing from its saved progress (models.Training.next_prompt, personas
    and adapter, the round's saved scenarios and its finished rounds' TrainedModels). `load_model(name)` returns a
    LocalModel-like object. The API writes status too ("pausing"), so this only changes it with targeted UPDATEs.

    The prompt only changes when a revision beats it on the same messages, and each round's adapter is scored on
    the same anchor users as the ones before it and dropped if it scores lower than the best, so the model never
    trains on from a worse one."""
    pool = ThreadPoolExecutor(1)
    with session_factory() as db:
        try:
            t = db.get(Training, training_id)
            _set_status(db, t, ("pending",), "running")  # left alone if the user already asked to pause
            model = load_model(t.base_model)
            weights = t.adapter  # what the next round's scenarios run with, before its SFT
            if weights:
                model.load_adapter(weights)
            else:
                weights = model.adapter_bytes()
            llm = llm or local_llm(model)  # no Claude: the helpers share the loaded model, with its adapter off
            local = getattr(llm, "local", False)
            rubric, prompt = t.rubric, t.next_prompt or t.initial_prompt
            finished = db.scalar(select(func.count()).where(TrainedModel.training_id == t.id, TrainedModel.round > 0))
            ahead = None  # with Claude, the next round's revision and personas, prepared while the model trains
            for r in range(finished + 1, t.rounds + 1):
                candidate = None  # a revision of the prompt, tested on the next batch (not saved: a resume re-proposes)
                if ahead:
                    candidate, t.personas = ahead.result()
                    db.commit()
                if _set_status(db, t, ("pausing",), "paused"):  # a safe point: progress is saved
                    return
                t.current_round = r
                done = list(db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r)
                                       .order_by(Scenario.scenario_number)))
                if t.personas is None or (candidate is None and len(done) < len(t.personas)):
                    # Resumed mid-round, the revision isn't saved: proposed again from the rows under this prompt.
                    last = [s for s in done if s.prompt == prompt] or db.scalars(
                        select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r - 1)
                        .order_by(Scenario.scenario_number)).all()
                    review = _review(rubric, last, prompt) if last else None
                    if t.personas is None:
                        candidate, t.personas = _prepare_round(llm, rubric, t.scenarios_per_round, prompt, review)
                    elif review:
                        candidate = _propose(llm, review)
                db.commit()
                rejected = 0
                while len(done) < len(t.personas):
                    if _set_status(db, t, ("pausing",), "paused"):
                        return
                    batch = t.personas[len(done):len(done) + MINIBATCH]
                    messages = [p["message"] for p in batch]
                    rows = _evaluate(llm, model, prompt, rubric, messages)
                    if candidate and candidate != prompt:
                        # A paired test: the same users, so a higher score is the prompt's doing, not theirs.
                        tried = _evaluate(llm, model, candidate, rubric, messages)
                        if fmean(x["score"] for x in tried) >= fmean(x["score"] for x in rows) + MIN_GAIN:
                            prompt, rows = candidate, tried
                        else:
                            rejected += 1
                    for p, row in zip(batch, rows):
                        done.append(Scenario(training_id=t.id, round=r, scenario_number=len(done) + 1,
                                             persona=p["persona"], prompt=prompt, **row))
                        db.add(done[-1])
                    # Saved together with the prompt they ran under: a failure before this just redoes the batch.
                    t.scenarios_done += len(batch)
                    t.next_prompt = prompt
                    db.commit()
                    if len(done) < len(t.personas):
                        evidence = [s for s in done if s.prompt == prompt]
                        candidate = _propose(llm, _review(rubric, evidence, prompt), rejected)
                # A local model can't answer while it trains, so it prepares the next round at its start instead.
                ahead = None
                if r < t.rounds and not local:
                    ahead = pool.submit(_prepare_round, llm, rubric, t.scenarios_per_round, prompt,
                                        _review(rubric, done, prompt))
                anchors = _anchors(db, t)
                held_out = {a.id for a in anchors}
                if not local:  # Claude writes the replies the model should have sent; a resume skips written ones
                    todo = [s for s in done if s.target is None and s.id not in held_out]
                    targets = agents.concurrently(
                        llm, lambda s: agents.teacher_reply(llm, s.prompt, s.input, rubric), todo)
                    for s, target in zip(todo, targets):
                        s.target = target
                    db.commit()
                if r == 1:  # the anchors ran under the initial prompt and weights: the baseline to beat, for free
                    db.add(TrainedModel(training_id=t.id, round=0, prompt=t.initial_prompt, weights=weights,
                                        mean_score=fmean(a.score for a in anchors)))
                examples = _examples(db, t, held_out)
                if examples:
                    model.sft(examples, seed=r)
                    weights = model.adapter_bytes()
                score = fmean(x["score"] for x in _evaluate(llm, model, prompt, rubric, [a.input for a in anchors]))
                db.add(trained := TrainedModel(training_id=t.id, round=r, prompt=prompt, weights=weights,
                                               mean_score=score))
                db.flush()
                # A training started before the anchor eval has no baseline and practice means as its scores: no gate.
                best = db.scalars(select(TrainedModel).where(TrainedModel.training_id == t.id)
                                  .order_by(TrainedModel.mean_score.desc(), TrainedModel.round.desc()).limit(1)).one()
                if not db.scalar(select(func.count()).where(TrainedModel.training_id == t.id, TrainedModel.round == 0)):
                    best = trained
                t.best_model_id = best.id
                if best.round != r:  # worse than the best so far (a tie keeps the newer): train on from that one
                    weights = best.weights
                    model.load_adapter(weights)
                # Committed with the round's model, so a resume starts the next round on the kept adapter.
                t.adapter, t.personas = weights, None
                db.commit()
            t.status = "completed"
            db.commit()
        except Exception as exc:
            log.exception("Training %s failed", training_id)
            # File paths and SQL are server internals; LLM/agent errors are what the user can act on.
            error = INTERNAL_ERROR if isinstance(exc, (OSError, SQLAlchemyError)) else str(exc)[:1000]
            try:
                db.rollback()
                db.execute(update(Training).where(Training.id == training_id).values(status="failed", error=error))
                db.commit()
            except Exception:
                log.exception("Could not mark training %s failed", training_id)
        finally:
            # Don't hold the slot and the model for a next round that won't run; a resume prepares it again.
            pool.shutdown(wait=False, cancel_futures=True)


def score_windows(scores: list[float]) -> list[dict]:
    """Mean score over the last n, n/2, n/4 (at least 1) and last runs."""
    n = len(scores)
    if not n:
        return []
    windows = [("n", n), ("n/2", max(1, n // 2)), ("n/4", max(1, n // 4)), ("last", 1)]
    return [{"label": label, "runs": k, "score": fmean(scores[-k:])} for label, k in windows]


class TrainingIn(BaseModel):
    base_model: str
    prompt: str = Field(min_length=1, max_length=20000)
    rubric: list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]] = Field(
        min_length=1, max_length=30)
    scenarios: int = Field(ge=1, le=100)
    rounds: int = Field(ge=1, le=50)
    helper: Helper | None = None


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
    }


def _start(background: BackgroundTasks, db: Session, llm, write) -> dict:
    """Take a training slot, run `write()`, which commits a pending Training and returns it, and run it in the
    background. Returns the training's detail."""
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
    background.add_task(_run_in_slot, t.id, llm, fake.FakeLocalModel if fake.enabled() else LocalModel)
    return out


def _run_in_slot(*args) -> None:
    try:
        run_training(*args)
    finally:
        _SLOTS.release()


@router.post("", status_code=201)
def create_training(body: TrainingIn, background: BackgroundTasks, user: User = Depends(current_user),
                    llm=Depends(get_llm), db: Session = Depends(get_db)):
    if body.base_model not in {m["name"] for m in list_local_models(user.id)}:
        raise HTTPException(400, f"Unknown base model: {body.base_model}")
    llm = chosen_llm(body.helper, llm)

    def insert():
        t = Training(user_id=user.id, base_model=body.base_model, initial_prompt=body.prompt, rubric=body.rubric,
                     scenarios_per_round=body.scenarios, rounds=body.rounds, helper=body.helper)
        db.add(t)
        db.commit()
        return t

    return _start(background, db, llm, insert)


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
    llm = chosen_llm(t.helper, llm)

    def restart():
        # Only one request can make the change, so a training never gets a second run.
        if not _set_status(db, t, ("paused", "failed"), "pending", error=None):
            raise HTTPException(409, "Only a paused or failed training can be resumed.")
        return t

    return _start(background, db, llm, restart)


@router.delete("/{training_id}", status_code=204)
def delete_training(training_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    t = owned(db, training_id, user)
    # Its scenarios and trained models go with it (ON DELETE CASCADE).
    if not db.execute(delete(Training).where(Training.id == t.id, Training.status.not_in(ACTIVE))).rowcount:
        raise HTTPException(409, "Pause the training before deleting it.")
    db.commit()


@router.get("")
def list_trainings(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(Training, TrainedModel.mean_score)
                      .outerjoin(TrainedModel, TrainedModel.id == Training.best_model_id)
                      .where(Training.user_id == user.id).order_by(Training.id.desc())).all()
    return [{"id": t.id, "base_model": t.base_model, "status": t.status, "rounds": t.rounds,
             "scenarios_per_round": t.scenarios_per_round, "current_round": t.current_round,
             "scenarios_done": t.scenarios_done, "created_at": t.created_at, "best_score": best_score,
             "helper": t.helper}
            for t, best_score in rows]


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
