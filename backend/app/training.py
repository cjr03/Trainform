"""Training orchestration plus the train/inspect API."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from statistics import fmean
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

HIGH_SCORE = 7.0  # scenarios scoring at least this are used as SFT examples
RECENT_SCORES = 5  # latest scenario scores the prompt generator sees, to tell whether its edits help
WEAKEST = 3  # lowest-scoring conversations a round review sees
INTERNAL_ERROR = "Training failed; see the server logs."
# Each run holds a loaded model and one of the 40 worker threads the sync routes share, so cap them process-wide.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TRAINFORM_MAX_TRAININGS", "2")))


def _set_status(db: Session, t: Training, old: tuple, new: str, **values) -> bool:
    """Move `t` from one of the `old` statuses to `new` in one UPDATE, so a concurrent change can't be lost."""
    changed = db.execute(update(Training).where(Training.id == t.id, Training.status.in_(old))
                         .values(status=new, **values)).rowcount
    db.commit()
    return bool(changed)


def _conversation(s: Scenario) -> str:
    return f"User: {s.input}\nAssistant: {s.output}"


def _review(rubric: list[str], last: list[Scenario]) -> tuple:
    """A finished round's results for agents.review_round: its best-scoring prompt (earliest on a tie), its mean score
    per criterion and its lowest-scoring conversations. Plain values, so another thread can use them."""
    means = [fmean(s.scores[i]["score"] for s in last) for i in range(len(rubric))]
    weakest = [(_conversation(s), s.score) for s in sorted(last, key=lambda s: s.score)[:WEAKEST]]
    return max(last, key=lambda s: s.score).prompt, list(zip(rubric, means)), weakest


def _prepare_round(llm, rubric: list[str], n: int, prompt: str, review: tuple | None) -> tuple[str, list[dict]]:
    """A round's starting prompt and its personas with their messages ([{"persona", "message"}]). After the first
    round, `review` (see _review) has the last round's results: the prompt generator revises its best prompt from them,
    and the personas press hardest on its weakest criteria."""
    last_scores = None
    if review:
        prompt, last_scores = review[0], [mean for _, mean in review[1]]
    tasks = [lambda: agents.review_round(llm, *review) if review else prompt,
             lambda: agents.generate_personas(llm, prompt, n, rubric=rubric, last_scores=last_scores)]
    # Personas need only the agent's role, which the review keeps, so with Claude both are written at once.
    start, personas = agents.concurrently(llm, lambda task: task(), tasks)
    # A persona's message depends only on the persona, so the round's messages are written at once. The rest of each
    # scenario waits for the prompt the one before it produced, so it runs in order.
    messages = agents.concurrently(llm, lambda p: agents.persona_respond(llm, p["attributes"], p["prompt"]), personas)
    return start, [{"persona": p, "message": m} for p, m in zip(personas, messages)]


def run_training(training_id: int, llm, load_model, session_factory=SessionLocal) -> None:
    """Run a training's remaining rounds, continuing from its saved progress (models.Training.next_prompt, personas
    and adapter, the round's saved scenarios and its finished rounds' TrainedModels). `load_model(name)` returns a
    LocalModel-like object. The API writes status too ("pausing"), so this only changes it with targeted UPDATEs."""
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
            prompt = t.next_prompt or t.initial_prompt
            finished = db.scalar(select(func.count()).where(TrainedModel.training_id == t.id))
            ahead = None  # with Claude, the next round's prompt and personas, prepared while the model trains
            for r in range(finished + 1, t.rounds + 1):
                if ahead:
                    prompt, t.personas = ahead.result()
                    t.next_prompt = prompt
                    db.commit()
                if _set_status(db, t, ("pausing",), "paused"):  # a safe point: progress is saved
                    return
                t.current_round = r
                if t.personas is None:
                    last = db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r - 1)
                                      .order_by(Scenario.scenario_number)).all()
                    prompt, t.personas = _prepare_round(llm, t.rubric, t.scenarios_per_round, prompt,
                                                        _review(t.rubric, last) if last else None)
                    t.next_prompt = prompt
                db.commit()
                done = list(db.scalars(select(Scenario).where(Scenario.training_id == t.id, Scenario.round == r)
                                       .order_by(Scenario.scenario_number)))
                for number, p in enumerate(t.personas[len(done):], len(done) + 1):
                    if _set_status(db, t, ("pausing",), "paused"):
                        return
                    message = p["message"]
                    reply = agents.primary_respond(model.generate, prompt, message)  # same user turn SFT trains on
                    conversation = f"User: {message}\nAssistant: {reply}"
                    scores = agents.score_conversation(llm, conversation, t.rubric)
                    s = Scenario(training_id=t.id, round=r, scenario_number=number, persona=p["persona"], prompt=prompt,
                                 input=message, output=reply, scores=scores, score=fmean(x["score"] for x in scores))
                    if number < len(t.personas):  # the next round starts from a review instead
                        # Its track record, from the round's saved scenarios: the best earlier prompt, if it beat
                        # this one, and the latest scores, so it builds on what worked and backs out what didn't.
                        top = max(done, key=lambda x: x.score, default=None)
                        best = (top.prompt, top.score) if top and top.score > s.score else None
                        recent = [x.score for x in done[-(RECENT_SCORES - 1):]] + [s.score]
                        prompt = agents.generate_improved_prompt(llm, prompt, conversation, scores, best, recent)
                    # Saved together with the prompt it produced: a failure before this just redoes the scenario.
                    db.add(s)
                    t.scenarios_done += 1
                    t.next_prompt = prompt
                    db.commit()
                    done.append(s)
                # A local model can't answer while it trains, so it prepares the next round at its start instead.
                ahead = None
                if r < t.rounds and not getattr(llm, "local", False):
                    ahead = pool.submit(_prepare_round, llm, t.rubric, t.scenarios_per_round, prompt,
                                        _review(t.rubric, done))
                # Store what earned the score: the round's weights and its best-scoring scenario's prompt.
                db.add(TrainedModel(training_id=t.id, round=r, prompt=max(done, key=lambda s: s.score).prompt,
                                    weights=weights, mean_score=fmean(s.score for s in done)))
                examples = [{"prompt": s.prompt, "input": s.input, "output": s.output}
                            for s in done if s.score >= HIGH_SCORE]
                if examples:
                    model.sft(examples)
                    weights = model.adapter_bytes()
                t.best_model_id = db.scalar(select(TrainedModel.id).where(TrainedModel.training_id == t.id)
                                            .order_by(TrainedModel.mean_score.desc(), TrainedModel.round).limit(1))
                # Committed with the round's model, so a resume starts the next round on the trained adapter.
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
    return {
        "id": t.id, "status": t.status, "error": t.error, "base_model": t.base_model,
        "initial_prompt": t.initial_prompt, "rubric": t.rubric, "rounds": t.rounds,
        "scenarios_per_round": t.scenarios_per_round, "current_round": t.current_round, "scenarios_done": t.scenarios_done,
        "total_scenarios": t.rounds * t.scenarios_per_round, "latest_prompt": latest_prompt or t.initial_prompt,
        "score_windows": score_windows(scores), "best_model_id": t.best_model_id, "best_score": best_score,
        "best_prompt": best_prompt, "created_at": t.created_at, "helper": t.helper,
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
    model_ids = dict(db.execute(select(TrainedModel.round, TrainedModel.id)
                                .where(TrainedModel.training_id == t.id)).all())
    rows = db.execute(select(Scenario.round, func.avg(Scenario.score), func.count()).where(Scenario.training_id == t.id)
                      .group_by(Scenario.round).order_by(Scenario.round)).all()
    return [{"round": r, "mean_score": mean, "scenarios": count, "model_id": model_ids.get(r)}
            for r, mean, count in rows]


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
