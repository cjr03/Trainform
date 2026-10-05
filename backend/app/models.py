from datetime import datetime

from sqlalchemy import JSON, ForeignKey, Index, LargeBinary, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    token_version: Mapped[int] = mapped_column(server_default="0")  # bumped to revoke all of the user's tokens
    # Fernet ciphertext of the user's Anthropic API key ("Claude linkage"); never stored in plaintext.
    anthropic_key_encrypted: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # The user's list of training_ids.
    trainings: Mapped[list["Training"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Training(Base):
    __tablename__ = "trainings"

    id: Mapped[int] = mapped_column(primary_key=True)  # training_id
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    base_model: Mapped[str] = mapped_column(String(255))  # local model name, see local_model.list_local_models
    initial_prompt: Mapped[str] = mapped_column(Text)
    rubric: Mapped[list[str]] = mapped_column(JSON)
    scenarios_per_round: Mapped[int]
    rounds: Mapped[int]
    # pending | running | pausing | paused | completed | failed
    status: Mapped[str] = mapped_column(String(16), default="pending")
    error: Mapped[str | None] = mapped_column(Text)
    current_round: Mapped[int] = mapped_column(default=0)
    scenarios_done: Mapped[int] = mapped_column(default=0)  # across all rounds
    # The best model for this training (highest round mean score).
    best_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id", use_alter=True, ondelete="SET NULL", name="fk_trainings_best_model")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # Saved progress, so a paused, failed or interrupted training continues where it stopped (see run_training):
    next_prompt: Mapped[str | None] = mapped_column(Text)  # the prompt the next scenario runs with
    personas: Mapped[list[dict] | None] = mapped_column(JSON)  # current round: [{"persona", "message"}]
    # Zipped LoRA adapter after the last finished round's SFT; deferred so listing trainings doesn't load it.
    adapter: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)

    user: Mapped[User] = relationship(back_populates="trainings")


ACTIVE = ("pending", "running", "pausing")  # statuses with a run in progress (or about to be)
# At most one active training per user, enforced by the database so concurrent requests can't race.
ONE_ACTIVE_TRAINING = Index("one_active_training_per_user", Training.user_id, unique=True,
                            postgresql_where=Training.status.in_(ACTIVE))


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("training_id", "round", "scenario_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # The spec's agent_id: each training trains exactly one agent, so training_id identifies it.
    training_id: Mapped[int] = mapped_column(ForeignKey("trainings.id", ondelete="CASCADE"), index=True)
    round: Mapped[int]
    scenario_number: Mapped[int]
    persona: Mapped[dict] = mapped_column(JSON)  # {"attributes": [{"criterion", "rating"}], "prompt": str}
    prompt: Mapped[str] = mapped_column(Text)  # primary agent prompt used for this scenario
    input: Mapped[str] = mapped_column(Text)  # persona agent message
    output: Mapped[str] = mapped_column(Text)  # primary agent response
    scores: Mapped[list[dict]] = mapped_column(JSON)  # [{"criterion": str, "score": int}]
    score: Mapped[float]  # mean of scores
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(primary_key=True)  # model_id
    training_id: Mapped[int] = mapped_column(ForeignKey("trainings.id", ondelete="CASCADE"), index=True)
    round: Mapped[int]
    prompt: Mapped[str] = mapped_column(Text)
    weights: Mapped[bytes | None] = mapped_column(LargeBinary)  # zipped LoRA adapter; None if the round had no SFT data
    mean_score: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
