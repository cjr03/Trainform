"""Training orchestration plus the train/inspect API."""

from fastapi import APIRouter

router = APIRouter(prefix="/trainings", tags=["training"])


def run_training(training_id: int, llm, load_model) -> None:
    """Run every round of a training. `load_model(name)` returns a LocalModel-like object."""
    raise NotImplementedError
