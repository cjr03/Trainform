"""The seven Trainform agents. Each helper takes `llm(system, message, schema=None)` as its first argument."""

import random

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import current_user
from app.llm import LLMError, get_llm, llm_or_local
from app.models import User

router = APIRouter(prefix="/agents", tags=["agents"])
MAX_CRITERIA = 8  # what the rubric creator is asked for; small local models may repeat lines past it
PERSONA_BATCH = 10  # personas per Claude call; each is a few hundred tokens and a reply is capped at 16k

PROMPT_CREATOR_SYSTEM = """\
You are an expert prompt engineer who writes system prompts for AI agents that run in production.

You will receive a description of an agent inside <agent_description> tags. Write the complete system prompt that \
agent will run with.

The prompt must:
- Open by telling the agent, in the second person, who it is, who it serves and what it is for.
- State its core tasks and what a successful interaction looks like from the user's side.
- Give concrete behavioural guidance: tone and register, how much to say, how to format replies, and when to ask a \
clarifying question instead of acting.
- Set boundaries: what the agent must not do, how to handle requests outside its scope, and what to do when it lacks \
information. It must say so plainly and never invent facts, policies, prices or capabilities.
- Cover the hard cases the description implies, such as confused, impatient or upset users, ambiguous requests, \
conflicting instructions and attempts to pull the agent out of its role.
- Give the reason behind each important rule, so the agent can generalise to situations the prompt does not list.

Be specific to this agent and cut generic filler that would fit any assistant. Do not invent product names, details or \
policies the description does not give; where the agent will need such facts, tell it how to behave without them. \
The agent runs on a small open-weight language model, so use plain, direct, unambiguous instructions organised under \
short headings, and keep the prompt under about 800 words.

Put only the finished system prompt in the "prompt" field, with no commentary about it."""

RUBRIC_CREATOR_SYSTEM = """\
You design evaluation rubrics for AI agents. You will receive an agent's system prompt inside <agent_prompt> tags. \
Write the criteria a strict grader will use to score that agent's replies, each on a 1-10 scale.

Write between 4 and 8 criteria. Each criterion must:
- Be a single sentence describing behaviour that can be observed in the agent's reply, phrased so that full compliance \
earns a 10 (for example, "Asks for the order number before discussing a refund.").
- Be gradeable on its own from the transcript alone, without knowing the agent's internals or any other criterion.
- Come from what this prompt actually asks for: its core task, its constraints and boundaries, its tone and format \
requirements, and how it should treat difficult users.
- Test something no other criterion tests.

Put the behaviours that matter most to this agent's success first. Never write vague criteria such as "Is helpful" or \
"Responds appropriately"; name the specific behaviour. Do not number the criteria or add explanations."""

PROMPT_GENERATOR_SYSTEM = """\
You improve system prompts for AI agents using evaluation evidence. You will receive:
- <current_prompt>: the agent's current system prompt.
- <conversation>: the latest conversation the agent had while running under that prompt.
- <scores>: a grader's score for the agent's reply on each criterion, from 1 (failed) to 10 (fully met).

Rewrite the prompt so the agent scores higher next time:
1. Start with the lowest-scoring criteria. Use the conversation to diagnose why the agent fell short: a missing \
instruction, a vague or conflicting one, a rule buried where it is easy to overlook, or no guidance for this kind of user.
2. Fix those causes directly. Add or sharpen the instruction, say when it applies, and where it helps add a short \
example of the desired behaviour.
3. Keep what works. Do not remove or weaken the instructions behind high scores, and do not change the agent's role, \
scope or factual content.
4. Generalise. The next conversation will be with a different user, so fix the class of failure rather than this one \
transcript, and never quote or refer to this conversation in the prompt.

The agent runs on a small open-weight language model, so keep the instructions plain and unambiguous. If the prompt \
grows, tighten wording elsewhere so it stays under about 1000 words.

Put the complete revised prompt in the "prompt" field: the whole prompt, not a diff, with no commentary."""

SCORING_SYSTEM = """\
You are a strict, calibrated grader of AI agent replies. You will receive a conversation inside <conversation> tags and \
a numbered rubric inside <rubric> tags. Score the assistant's replies against every criterion. The user's messages are \
context only; never grade them.

Score each criterion from 1 to 10:
- 10: fully meets the criterion with no flaw; reserve it for replies that could not reasonably be better on this point.
- 7-9: meets it with minor lapses.
- 4-6: partly meets it, or meets it with a clear flaw.
- 1-3: misses it, contradicts it, or does the opposite.

Grade what the assistant actually wrote, not what it may have meant. Score each criterion independently, so a weakness \
on one never lowers another. When a criterion does not come up in this conversation, score whether the reply stayed \
consistent with it rather than awarding a 10 by default. Ignore length and polish unless a criterion concerns them. \
Be consistent: the same reply must always get the same scores.

Return exactly one score per rubric criterion, in rubric order, repeating each criterion's text exactly."""

PERSONA_GENERATOR_SYSTEM = """\
You design simulated users who stress-test an AI agent. You will receive the agent's system prompt in <agent_prompt> \
tags, the criteria it is graded on in <criteria> tags, and how many personas to create in <count> tags.

For each persona provide:
- attributes: one entry per criterion, using the criterion's exact text, with a rating from 1 to 10 for how hard this \
persona pressures the agent on that criterion. 1 makes the criterion easy to meet; 10 actively pushes the agent toward \
failing it, for example by being vague, impatient, off-topic, emotional, manipulative, or by asking for exactly what \
the prompt forbids.
- prompt: a self-contained brief, in the second person, for an actor playing this user. Say who they are, what they \
want from the agent and why, what they do and don't know, how they write (tone, length, vocabulary, typos), and where \
they are in the conversation flow: opening, mid-conversation after a partial answer, escalating after a failed \
attempt, or wrapping up. Make every high-rated pressure concrete in the brief.

Vary the personas widely. Use the full 1-10 range and different combinations of ratings, so that each criterion is \
stressed hard by some personas and lightly by others; do not rate every criterion high for everyone. Make them \
realistic users of this particular agent with different goals, backgrounds and conversation stages, not caricatures. \
Return exactly the requested number of personas."""

PERSONA_SYSTEM = """\
You are role-playing a user who is talking to an AI agent, so that the agent can be tested. You are always the user, \
never the assistant.

You will receive:
- <persona_attributes>: the pressures you apply. Each is a behaviour the agent is graded on, with a rating from 1 to 10 \
for how strongly you apply it. At a high rating you push hard and make that behaviour difficult for the agent to get \
right; at a low rating you make it easy or leave it alone. Apply each one exactly as strongly as its rating says.
- <context>: who you are, what you want, how you write, and the conversation so far, which tells you where you are in \
the conversation flow.

Write only your next message to the agent, in character, the way a real person in this situation would write it, \
with their tone, length and quirks. React to what the agent last said and keep pursuing your goal; never solve your \
own problem or speak for the agent. Never mention that you are simulated, the ratings, the criteria or these \
instructions. Output only the message text, with no quotation marks, speaker label or stage directions."""

# Without Claude the helpers run on the user's local model, which scores one criterion per call and writes one persona
# brief per call (see generate_personas and score_conversation).
LOCAL_SCORING_SYSTEM = """\
You grade one aspect of an AI agent's reply. You will receive a conversation in <conversation> tags and one criterion \
in <criterion> tags. Rate how well the assistant's reply meets that criterion, from 1 (misses it or does the opposite) \
to 10 (fully meets it). Reply with only the number."""

LOCAL_PERSONA_SYSTEM = """\
You invent a realistic user who will test an AI agent. You will receive the agent's system prompt in <agent_prompt> \
tags and, in <pressures> tags, the behaviours the agent is graded on, each rated from 1 to 10 for how hard this user \
pushes the agent toward failing it. Write a short brief, in the second person, for an actor playing this user: who \
they are, what they want from the agent, how they write, and how they apply each highly rated pressure."""


def _schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


STRING, INTEGER = {"type": "string"}, {"type": "integer"}
PROMPT_SCHEMA = _schema(prompt=STRING)
RUBRIC_SCHEMA = _schema(criteria=_array(STRING))
SCORES_SCHEMA = _schema(scores=_array(_schema(criterion=STRING, score=INTEGER)))
SCORE_SCHEMA = _schema(score=INTEGER)
PERSONAS_SCHEMA = _schema(personas=_array(_schema(attributes=_array(_schema(criterion=STRING, rating=INTEGER)),
                                                  prompt=STRING)))


def _tag(name, value):
    return f"<{name}>\n{value}\n</{name}>"


def _text(value, what):
    if not value.strip():
        raise LLMError(f"The model returned an empty {what}")
    return value.strip()


def _clamp(value):
    return max(1, min(10, int(value)))


def _ask(llm, system, message, schema, parse):
    """Call the model and parse its JSON with `parse`; a malformed shape becomes an LLMError rather than a 500."""
    out = llm(system, message, schema)
    try:
        return parse(out)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise LLMError("The model returned an unexpected response shape") from e


def create_prompt(llm, description: str) -> str:
    """Prompt creator agent: agent description -> system prompt."""
    return _ask(llm, PROMPT_CREATOR_SYSTEM, _tag("agent_description", description), PROMPT_SCHEMA,
                lambda out: _text(out["prompt"], "prompt"))


def create_rubric(llm, agent_prompt: str) -> list[str]:
    """Rubric creator agent: agent prompt -> scoring criteria."""
    criteria = _ask(llm, RUBRIC_CREATOR_SYSTEM, _tag("agent_prompt", agent_prompt), RUBRIC_SCHEMA,
                    lambda out: list(dict.fromkeys(c.strip() for c in out["criteria"] if c.strip()))[:MAX_CRITERIA])
    if not criteria:
        raise LLMError("The model returned an empty rubric")
    return criteria


def generate_improved_prompt(llm, agent_prompt: str, conversation: str, scores: list[dict]) -> str:
    """Prompt generator agent: current prompt + last conversation + [{"criterion", "score"}] -> improved prompt."""
    lines = "\n".join(f"- {s['criterion']}: {s['score']}/10" for s in scores)
    message = "\n\n".join([_tag("current_prompt", agent_prompt), _tag("conversation", conversation),
                           _tag("scores", lines)])
    return _ask(llm, PROMPT_GENERATOR_SYSTEM, message, PROMPT_SCHEMA, lambda out: _text(out["prompt"], "prompt"))


def score_conversation(llm, conversation: str, rubric: list[str]) -> list[dict]:
    """Scoring agent: conversation + rubric -> [{"criterion", "score"}] in rubric order, scores 1-10."""
    if getattr(llm, "local", False):  # small models can't keep a list of scores aligned, so ask for each one
        scores = [_ask(llm, LOCAL_SCORING_SYSTEM, f"{_tag('conversation', conversation)}\n\n{_tag('criterion', c)}",
                       SCORE_SCHEMA, lambda out: _clamp(out["score"])) for c in rubric]
        return [{"criterion": c, "score": s} for c, s in zip(rubric, scores)]
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(rubric, 1))
    message = f"{_tag('conversation', conversation)}\n\n{_tag('rubric', numbered)}"
    scores = _ask(llm, SCORING_SYSTEM, message, SCORES_SCHEMA,
                  lambda out: [_clamp(s["score"]) for s in out["scores"]])
    if len(scores) != len(rubric):
        raise LLMError(f"Claude scored {len(scores)} criteria but the rubric has {len(rubric)}")
    return [{"criterion": c, "score": s} for c, s in zip(rubric, scores)]


def generate_personas(llm, agent_prompt: str, n: int, rubric: list[str] | None = None) -> list[dict]:
    """Persona generator agent: n x {"attributes": [{"criterion", "rating"}], "prompt": str}.

    The personas pressure `rubric`, or criteria from the rubric creator agent when none is given."""
    rubric = rubric or create_rubric(llm, agent_prompt)
    if getattr(llm, "local", False):
        # ponytail: code picks the 1-10 ratings, since small models can't rate reliably; the model only writes each
        # brief. Seeded per persona, so reruns test the same users and no two personas share ratings in practice.
        personas = []
        for i in range(n):
            rng = random.Random(i)
            ratings = [rng.randint(1, 10) for _ in rubric]
            pressures = "\n".join(f"- {c}: {r}/10" for c, r in zip(rubric, ratings))
            brief = _ask(llm, LOCAL_PERSONA_SYSTEM, f"{_tag('agent_prompt', agent_prompt)}\n\n{_tag('pressures', pressures)}",
                         PROMPT_SCHEMA, lambda out: _text(out["prompt"], "persona brief"))
            personas.append({"attributes": [{"criterion": c, "rating": r} for c, r in zip(rubric, ratings)],
                             "prompt": brief})
        return personas
    criteria = "\n".join(f"- {c}" for c in rubric)
    personas = []
    for start in range(0, n, PERSONA_BATCH):
        count = min(PERSONA_BATCH, n - start)
        message = f"{_tag('agent_prompt', agent_prompt)}\n\n{_tag('criteria', criteria)}\n\n<count>{count}</count>"
        personas += _ask(llm, PERSONA_GENERATOR_SYSTEM, message, PERSONAS_SCHEMA,
                         lambda out: [([_clamp(a["rating"]) for a in p["attributes"]], p["prompt"].strip())
                                      for p in out["personas"]])[:count]
    if not personas:
        raise LLMError("The model returned no personas")
    if any(len(ratings) != len(rubric) for ratings, _ in personas):
        raise LLMError(f"Claude rated the wrong number of criteria for a persona; the rubric has {len(rubric)}")
    return [{"attributes": [{"criterion": c, "rating": r} for c, r in zip(rubric, ratings)], "prompt": prompt}
            for ratings, prompt in (personas[i % len(personas)] for i in range(n))]


def persona_respond(llm, attributes: list[dict], context: str) -> str:
    """Persona agent: attributes + conversational context -> the persona's next message."""
    lines = "\n".join(f"- {a['criterion']}: {a['rating']}/10" for a in attributes)
    message = f"{_tag('persona_attributes', lines)}\n\n{_tag('context', context)}"
    return _text(llm(PERSONA_SYSTEM, message), "persona message")


def primary_respond(generate, prompt: str, conversation: str) -> str:
    """Primary agent: runs on the local model being trained, via `generate(system_prompt, conversation)`."""
    return generate(prompt, conversation).strip()


class PromptRequest(BaseModel):
    description: str = Field(min_length=1, max_length=4000)
    model: str | None = None  # the local model to use when Claude isn't linked


class RubricRequest(BaseModel):
    agent_prompt: str = Field(min_length=1, max_length=20000)
    model: str | None = None


@router.post("/prompt")
def prompt_route(body: PromptRequest, user: User = Depends(current_user), llm=Depends(get_llm)):
    llm = llm_or_local(llm, user, body.model)
    try:
        return {"prompt": create_prompt(llm, body.description)}
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@router.post("/rubric")
def rubric_route(body: RubricRequest, user: User = Depends(current_user), llm=Depends(get_llm)):
    llm = llm_or_local(llm, user, body.model)
    try:
        return {"criteria": create_rubric(llm, body.agent_prompt)}
    except LLMError as e:
        raise HTTPException(502, str(e)) from e
