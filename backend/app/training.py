"""Training orchestration plus the train/inspect API."""

import logging
import os
import threading
from statistics import fmean
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app import agents, fake
from app.auth import current_user
from app.db import SessionLocal, get_db
from app.llm import get_llm, local_llm
from app.local_model import LocalModel, list_local_models
from app.models import Scenario, TrainedModel, Training, User

log = logging.getLogger(__name__)
router = APIRouter(prefix="/trainings", tags=["training"])

HIGH_SCORE = 7.0  # scenarios scoring at least this are used as SFT examples
INTERNAL_ERROR = "Training failed; see the server logs."
# Each run holds a loaded model and one of the 40 worker threads the sync routes share, so cap them process-wide.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("TRAINFORM_MAX_TRAININGS", "2")))


def run_training(training_id: int, llm, load_model, session_factory=SessionLocal) -> None:
    """Run every round of a training. `load_model(name)` returns a LocalModel-like object."""
    with session_factory() as db:
        try:
            t = db.get(Training, training_id)
            t.status = "running"
            db.commit()
            model = load_model(t.base_model)
            llm = llm or local_llm(model)  # no Claude: the helpers share the loaded model, with its adapter off
            prompt = t.initial_prompt
            best_score = None
            for r in range(1, t.rounds + 1):
                t.current_round = r
                db.commit()
                weights = model.adapter_bytes()  # what this round's scenarios run with, before its SFT
                done = []
                personas = agents.generate_personas(llm, prompt, t.scenarios_per_round, rubric=t.rubric)
                # A persona's message depends only on the persona, so the round's messages are written at once. The
                # rest of each scenario waits for the prompt the one before it produced, so it runs in order.
                messages = agents.concurrently(llm, lambda p: agents.persona_respond(llm, p["attributes"], p["prompt"]),
                                               personas)
                for number, (persona, message) in enumerate(zip(personas, messages), 1):
                    reply = agents.primary_respond(model.generate, prompt, message)  # same user turn SFT trains on
                    conversation = f"User: {message}\nAssistant: {reply}"
                    scores = agents.score_conversation(llm, conversation, t.rubric)
                    s = Scenario(training_id=t.id, round=r, scenario_number=number, persona=persona, prompt=prompt,
                                 input=message, output=reply, scores=scores, score=fmean(x["score"] for x in scores))
                    db.add(s)
                    t.scenarios_done += 1
                    db.commit()
                    done.append(s)
                    if r < t.rounds or number < len(personas):  # the very last improved prompt would go unused
                        prompt = agents.generate_improved_prompt(llm, prompt, conversation, scores)
                examples = [{"prompt": s.prompt, "input": s.input, "output": s.output}
                            for s in done if s.score >= HIGH_SCORE]
                if examples:
                    model.sft(examples)
                # Store what earned the score: the round's weights and its best-scoring scenario's prompt.
                m = TrainedModel(training_id=t.id, round=r, prompt=max(done, key=lambda s: s.score).prompt,
                                 weights=weights, mean_score=fmean(s.score for s in done))
                db.add(m)
                db.flush()
                if best_score is None or m.mean_score > best_score:
                    best_score, t.best_model_id = m.mean_score, m.id
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
        "best_prompt": best_prompt, "created_at": t.created_at,
    }


@router.post("", status_code=201)
def create_training(body: TrainingIn, background: BackgroundTasks, user: User = Depends(current_user),
                    llm=Depends(get_llm), db: Session = Depends(get_db)):
    if body.base_model not in {m["name"] for m in list_local_models(user.id)}:
        raise HTTPException(400, f"Unknown base model: {body.base_model}")
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(429, "The server is busy with other trainings; try again later.")
    try:
        t = Training(user_id=user.id, base_model=body.base_model, initial_prompt=body.prompt, rubric=body.rubric,
                     scenarios_per_round=body.scenarios, rounds=body.rounds)
        db.add(t)
        db.commit()
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


@router.get("")
def list_trainings(user: User = Depends(current_user), db: Session = Depends(get_db)):
    rows = db.execute(select(Training, TrainedModel.mean_score)
                      .outerjoin(TrainedModel, TrainedModel.id == Training.best_model_id)
                      .where(Training.user_id == user.id).order_by(Training.id.desc())).all()
    return [{"id": t.id, "base_model": t.base_model, "status": t.status, "rounds": t.rounds,
             "scenarios_per_round": t.scenarios_per_round, "current_round": t.current_round,
             "scenarios_done": t.scenarios_done, "created_at": t.created_at, "best_score": best_score}
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
