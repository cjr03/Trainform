"""Agent profiles: a user's agents (a base model, a system prompt and the weights trainings left it), and their API."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, StringConstraints
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from app import svg
from app.agents import _tag, draw_image
from app.auth import current_user
from app.db import get_db
from app.llm import Helper, LLMError, chosen_llm, get_llm
from app.local_model import list_local_models, quantize_model
from app.models import ACTIVE, Agent, TrainedModel, Training, User

router = APIRouter(prefix="/agents", tags=["agent profiles"])

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Prompt = Annotated[str, Field(min_length=1)]


class AgentIn(BaseModel):
    name: Name
    base_model: str
    prompt: Prompt


class AgentPatch(BaseModel):
    name: Name | None = None
    prompt: Prompt | None = None


class ImageIn(BaseModel):
    svg: str | None  # None clears it


class GenerateIn(BaseModel):
    helper: Helper | None = None


def owned(db: Session, agent_id: int, user: User) -> Agent:
    a = db.get(Agent, agent_id)
    if a is None or a.user_id != user.id:
        raise HTTPException(404, "Agent not found")
    return a


def sync_agent(db: Session, t: Training) -> None:
    """Give `t`'s agent its best trained model's prompt and (unless the agent is Claude) its weights, copied inside the
    database, when this round made it."""
    if db.scalar(select(TrainedModel.round).where(TrainedModel.id == t.best_model_id)) != t.current_round:
        return  # untrained, or an earlier round's best, which that round synced
    best = lambda column: select(column).where(TrainedModel.id == t.best_model_id).scalar_subquery()  # noqa: E731
    values = {"prompt": best(TrainedModel.prompt)}
    if t.base_model != "claude":
        values["weights"] = best(TrainedModel.weights)
    # A training from before agents may be on another base model than its agent: its weights wouldn't fit.
    db.execute(update(Agent).where(Agent.id == t.agent_id, Agent.base_model == t.base_model).values(**values))


def available_models(user: User) -> set[str]:
    """The base models the user's agents can run on: "claude" and the user's local models."""
    return {"claude"} | {m["name"] for m in list_local_models(user.id)}


def _out(models: set[str], a: Agent, best: float | None, trainings) -> dict:
    """`best`: the agent's highest score over its trainings' trained rounds (see overall).
    `models`: available_models; an agent copied from a post may be on a model the user has yet to download."""
    return {"id": a.id, "name": a.name, "base_model": a.base_model, "prompt": a.prompt, "image": a.image,
            "model_available": a.base_model in models, "overall": overall(a, best),
            "trainings": trainings, "created_at": a.created_at}


def overall(a: Agent, best: float | None) -> int | None:
    """`best` (see _rows) as a 0-100 score, else the overall of the post it was copied from (None: Unranked)."""
    return a.overall if best is None else round(10 * best)


def _rows(db: Session, user: User, agent_id: int | None = None) -> list:
    """(agent, best score, training count, battles not counted) for the user's agents (or just one), by name."""
    q = (select(Agent, func.max(TrainedModel.mean_score),
                func.count(Training.id.distinct()).filter(Training.battle.is_(None)))
         .outerjoin(Training, Training.agent_id == Agent.id)
         .outerjoin(TrainedModel, (TrainedModel.training_id == Training.id) & (TrainedModel.round > 0))
         .where(Agent.user_id == user.id).group_by(Agent.id).order_by(Agent.name))
    return db.execute(q if agent_id is None else q.where(Agent.id == agent_id)).all()


def _idle(a: Agent):
    """A condition that `a` has no training in progress (which would sync into it)."""
    return ~select(Training.id).where(Training.agent_id == a.id, Training.status.in_(ACTIVE)).exists()


def _name_taken(db: Session) -> HTTPException:
    db.rollback()
    return HTTPException(409, "You already have an agent with that name.")  # (user_id, name) is unique


@router.get("")
def list_agents(user: User = Depends(current_user), db: Session = Depends(get_db)):
    models = available_models(user)
    return [_out(models, *row) for row in _rows(db, user)]


@router.post("", status_code=201)
def create_agent(body: AgentIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if body.base_model == "claude":
        chosen_llm("claude", get_llm(user))  # 400 unless the user can call Claude
    elif body.base_model not in available_models(user):
        raise HTTPException(400, f"Unknown base model: {body.base_model}")
    a = Agent(user_id=user.id, name=body.name, base_model=body.base_model, prompt=body.prompt)
    db.add(a)
    try:
        db.commit()
    except IntegrityError:
        raise _name_taken(db) from None
    return _out({body.base_model}, a, None, 0)


@router.get("/{agent_id}")
def get_agent(agent_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    owned(db, agent_id, user)
    [row] = _rows(db, user, agent_id)
    best, baseline = aliased(TrainedModel), aliased(TrainedModel)
    trainings = db.execute(
        select(Training.id, Training.status, Training.created_at, Training.rounds, Training.scenarios_per_round,
               best.mean_score.label("best_score"), baseline.mean_score.label("baseline_score"))
        .outerjoin(best, best.id == Training.best_model_id)
        .outerjoin(baseline, (baseline.training_id == Training.id) & (baseline.round == 0))
        .where(Training.agent_id == agent_id, Training.battle.is_(None))  # battles are listed apart
        .order_by(Training.id.desc())).mappings().all()
    return _out(available_models(user), *row) | {"trainings": [dict(t) for t in trainings]}


@router.patch("/{agent_id}")
def update_agent(agent_id: int, body: AgentPatch, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = owned(db, agent_id, user)
    changes = body.model_dump(exclude_none=True)
    if changes:
        try:
            changed = db.execute(update(Agent).where(Agent.id == a.id, _idle(a)).values(**changes)).rowcount
        except IntegrityError:
            raise _name_taken(db) from None
        if not changed:
            raise HTTPException(409, "Wait for the agent's training to pause or finish first.")
        db.commit()
    return _out(available_models(user), *_rows(db, user, agent_id)[0])


def _set_image(db: Session, user: User, a: Agent, image: str | None) -> dict:
    a.image = image
    db.commit()
    return _out(available_models(user), *_rows(db, user, a.id)[0])


@router.put("/{agent_id}/image")
def put_image(agent_id: int, body: ImageIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = owned(db, agent_id, user)
    try:
        image = None if body.svg is None else svg.sanitise(body.svg)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    return _set_image(db, user, a, image)


@router.post("/{agent_id}/image/generate")
def generate_image(agent_id: int, body: GenerateIn, user: User = Depends(current_user),
                   db: Session = Depends(get_db), llm=Depends(get_llm)):
    a = owned(db, agent_id, user)
    subject = f"{_tag('agent_name', a.name)}\n\n{_tag('agent_prompt', a.prompt[:4000])}"
    try:
        image = draw_image(chosen_llm(body.helper, llm), subject, a.name)
    except LLMError as e:
        raise HTTPException(502, str(e)) from e
    return _set_image(db, user, a, image)


@router.post("/{agent_id}/quantize", status_code=201)
def quantize_agent(agent_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """A copy of the agent (prompt, trained weights, image) on an int8 copy of its model. Claude's weights aren't
    public, so only local models can be quantized."""
    a = owned(db, agent_id, user)
    if a.base_model == "claude":
        raise HTTPException(400, "Claude's weights aren't public, so a Claude agent can't be quantized.")
    if a.base_model not in available_models(user):
        raise HTTPException(400, f"Download {a.base_model} first.")
    try:
        model = quantize_model(a.base_model, user.id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    except FileExistsError as e:
        raise HTTPException(409, f"A model named {e} already exists") from None
    # The LoRA adapter fits: quantizing keeps every layer's shape.
    q = Agent(user_id=user.id, name=f"{a.name[:93]} (int8)", base_model=model, prompt=a.prompt, weights=a.weights,
              image=a.image)
    db.add(q)
    try:
        db.commit()
    except IntegrityError:
        raise _name_taken(db) from None
    return _out({model}, q, None, 0)


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: int, user: User = Depends(current_user), db: Session = Depends(get_db)):
    a = owned(db, agent_id, user)
    # Its trainings go with it (ON DELETE CASCADE).
    if not db.execute(delete(Agent).where(Agent.id == a.id, _idle(a))).rowcount:
        raise HTTPException(409, "Pause the agent's training before deleting it.")
    db.commit()
