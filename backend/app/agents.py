"""The seven Trainform agents. Each helper takes `llm(system, message, schema=None)` as its first argument."""

from fastapi import APIRouter

router = APIRouter(prefix="/agents", tags=["agents"])


def create_prompt(llm, description: str) -> str:
    """Prompt creator agent: agent description -> system prompt."""
    raise NotImplementedError


def create_rubric(llm, agent_prompt: str) -> list[str]:
    """Rubric creator agent: agent prompt -> scoring criteria."""
    raise NotImplementedError


def generate_improved_prompt(llm, agent_prompt: str, conversation: str, scores: list[dict]) -> str:
    """Prompt generator agent: current prompt + last conversation + [{"criterion", "score"}] -> improved prompt."""
    raise NotImplementedError


def score_conversation(llm, conversation: str, rubric: list[str]) -> list[dict]:
    """Scoring agent: conversation + rubric -> [{"criterion", "score"}] in rubric order, scores 1-10."""
    raise NotImplementedError


def generate_personas(llm, agent_prompt: str, n: int) -> list[dict]:
    """Persona generator agent: n x {"attributes": [{"criterion", "rating"}], "prompt": str}."""
    raise NotImplementedError


def persona_respond(llm, attributes: list[dict], context: str) -> str:
    """Persona agent: attributes + conversational context -> the persona's next message."""
    raise NotImplementedError


def primary_respond(generate, prompt: str, conversation: str) -> str:
    """Primary agent: runs on the local model being trained, via `generate(system_prompt, conversation)`."""
    raise NotImplementedError
