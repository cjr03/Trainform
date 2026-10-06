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


# Every user gets this agent (see main.lifespan and auth.signup); trainings from before agents existed belong to it.
SEED_AGENT = {"name": "Bike Repair Agent",
              "prompt": "You are the support assistant for Spoke & Chain, a bike repair shop. Help customers with "
                        "repairs, maintenance and booking. Be friendly and clear."}
DEFAULT_BASE_MODEL = "Qwen--Qwen3-0.6B"


class Agent(Base):
    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    base_model: Mapped[str] = mapped_column(String(255))  # a local model name (see Training.base_model) or "claude"
    prompt: Mapped[str] = mapped_column(Text)
    # Zipped LoRA adapter of its best trained model so far (see agent_profiles.sync_agent); None: the base weights.
    weights: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Training(Base):
    __tablename__ = "trainings"

    id: Mapped[int] = mapped_column(primary_key=True)  # training_id
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # The agent it trains; its base_model and initial_prompt are copied from it when the training starts.
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id", ondelete="CASCADE"))
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
    # The best model for this training: the highest score on the anchor messages (see training.run_training).
    best_model_id: Mapped[int | None] = mapped_column(
        ForeignKey("trained_models.id", use_alter=True, ondelete="SET NULL", name="fk_trainings_best_model")
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # Saved progress, so a paused, failed or interrupted training continues where it stopped (see run_training):
    next_prompt: Mapped[str | None] = mapped_column(Text)  # the prompt the next scenario runs with
    personas: Mapped[list[dict] | None] = mapped_column(JSON)  # current round: [{"persona", "message"}]
    # The zipped LoRA adapter to start from (the agent's weights when it started; from older versions, the last
    # finished round's); deferred so listing trainings doesn't load it.
    adapter: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    helper: Mapped[str | None] = mapped_column(String(16))  # llm.Helper the user chose; None: Claude if linked

    user: Mapped[User] = relationship(back_populates="trainings")


ACTIVE = ("pending", "running", "pausing")  # statuses with a run in progress (or about to be)
# At most one active training per user, enforced by the database so concurrent requests can't race.
ONE_ACTIVE_TRAINING = Index("one_active_training_per_user", Training.user_id, unique=True,
                            postgresql_where=Training.status.in_(ACTIVE))


class Scenario(Base):
    __tablename__ = "scenarios"
    __table_args__ = (UniqueConstraint("training_id", "round", "scenario_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    training_id: Mapped[int] = mapped_column(ForeignKey("trainings.id", ondelete="CASCADE"), index=True)
    round: Mapped[int]
    scenario_number: Mapped[int]
    persona: Mapped[dict] = mapped_column(JSON)  # {"attributes": [{"criterion", "rating"}], "prompt": str}
    prompt: Mapped[str] = mapped_column(Text)  # primary agent prompt used for this scenario
    input: Mapped[str] = mapped_column(Text)  # persona agent message
    output: Mapped[str] = mapped_column(Text)  # primary agent response
    scores: Mapped[list[dict]] = mapped_column(JSON)  # [{"criterion": str, "score": int}]
    score: Mapped[float]  # mean of scores
    target: Mapped[str | None] = mapped_column(Text)  # the reply a Claude teacher wrote for it, which SFT learns
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id: Mapped[int] = mapped_column(primary_key=True)  # model_id
    training_id: Mapped[int] = mapped_column(ForeignKey("trainings.id", ondelete="CASCADE"), index=True)
    round: Mapped[int]
    prompt: Mapped[str] = mapped_column(Text)
    # Zipped LoRA adapter; None for a Claude agent. Deferred: only a revert or a download needs it.
    weights: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    mean_score: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
