"""The eight Trainform agents. Each helper takes `llm(system, message, schema=None, effort="medium")` as its first
argument."""

import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from statistics import fmean

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import svg
from app.auth import current_user
from app.llm import BadReply, Helper, LLMError, chosen_llm, get_llm, llm_or_local
from app.models import User

router = APIRouter(prefix="/agents", tags=["agents"])
MAX_CRITERIA = 8  # what the rubric creator is asked for; small local models may repeat lines past it
PERSONA_BATCH = 10  # personas per Claude call; each is a few hundred tokens and a reply is capped at 16k
CONCURRENCY = 8  # independent Claude calls in flight at once; call_claude's retries absorb any 429s this causes

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
- <round_scores>: the agent's mean score on each criterion over its latest conversations under that prompt, from 1 \
(failed) to 10 (fully met).
- <weakest_conversations>: the lowest-scoring of those conversations, each with its mean score.

Your revision is tested against the current prompt on the same new users and kept only if it scores higher, so make \
the changes most likely to raise the scores:
1. Start with the lowest-scoring criteria. Use the conversations to diagnose why the agent fell short: a missing \
instruction, a vague or conflicting one, a rule buried where it is easy to overlook, or no guidance for this kind of user.
2. Fix those causes directly. Add or sharpen the instruction, say when it applies, and where it helps add a short \
example of the desired behaviour. Treat a failure that recurs across the conversations as the most important.
3. Keep what works. Do not remove or weaken the instructions behind high scores, and do not change the agent's role, \
scope or factual content.
4. Generalise. The next conversations are with different users, so fix the class of failure rather than one \
transcript, and never quote or refer to these conversations in the prompt.

The agent runs on a small open-weight language model, so keep the instructions plain and unambiguous. If the prompt \
grows, tighten wording elsewhere so it stays under about 800 words.

Put the complete revised prompt in the "prompt" field: the whole prompt, not a diff, with no commentary."""

# With Claude, the model learns from the reply it should have sent rather than only from its own best ones.
TEACHER_SYSTEM = """\
You write the reply an AI agent should have sent. You will receive the agent's system prompt in <agent_prompt>, a \
customer's message in <message> and, in <rubric>, the criteria every reply is graded on. Write the one reply that \
follows the prompt and fully meets every criterion, in the agent's own voice: plain, direct and no longer than the \
message needs. A small model will be trained to imitate it, so make it a clear example to copy. Put only the reply \
in the "reply" field."""

SCORING_SYSTEM = """\
You are a strict, calibrated grader of AI agent replies. You will receive a conversation inside <conversation> tags and \
a rubric inside <rubric> tags, one criterion per line, each after its key (c1, c2, ...). Score the assistant's replies \
against every criterion. The user's messages are context only; never grade them.

Score each criterion from 1 to 10:
- 10: fully meets the criterion with no flaw; reserve it for replies that could not reasonably be better on this point.
- 7-9: meets it with minor lapses.
- 4-6: partly meets it, or meets it with a clear flaw.
- 1-3: misses it, contradicts it, or does the opposite.

Grade what the assistant actually wrote, not what it may have meant. Score each criterion independently, so a weakness \
on one never lowers another. When a criterion does not come up in this conversation, score whether the reply stayed \
consistent with it rather than awarding a 10 by default. Ignore length and polish unless a criterion concerns them. \
Be consistent: the same reply must always get the same scores.

Return each criterion's score under that criterion's key."""

PERSONA_GENERATOR_SYSTEM = """\
You design simulated customers who test an AI agent against its grading rubric. You will receive the agent's system \
prompt in <agent_prompt> tags, the criteria its replies are graded on in <criteria> tags, one per line, each after its \
key (c1, c2, ...), and how many personas to create in <count> tags.

The criteria describe the agent's behaviour, never the customer's. Each persona is a test case: a real person who \
would actually come to this agent (a customer, client, patient, student or member of the public, as fits the \
agent) whose situation and opening message create the conditions under which some criteria are tested. For example, \
to test "Asks for the order number before processing a refund", the customer demands a refund without giving an order \
number; to test "Stays calm with angry customers", the customer is angry; to test "Declines requests outside its \
scope", the customer asks for something the agent does not handle; to test "Keeps answers brief", the customer asks a \
broad question that invites a long answer.

For each persona provide:
- attributes: under each criterion's key, a rating from 1 to 10 for how hard this persona tests that criterion. At \
8-10 the customer's situation squarely calls for that behaviour and makes it hard to get right, for example by being \
vague, impatient, emotional, manipulative or off-topic, or by asking for exactly what the prompt forbids; at 4-7 it \
comes up in a milder form; at 1-3 the persona leaves it easy and does nothing to provoke it.
- prompt: a self-contained brief, in the second person, for an actor playing this customer. Say who they are, the \
concrete need they bring and why, with the specific details a real person has (what happened, when, names, numbers, \
what they already tried), what they leave out or get wrong, their mood, and how they write (tone, length, vocabulary, \
typos). Build every high-rated criterion into this situation as something the customer wants, does, feels or leaves \
out. The conversation starts with their message, so the need must be stated or hinted at in it.

The brief describes only the customer. Never quote, paraphrase or name a criterion, never describe what the agent \
should do, and never give the customer the agent's behaviour: a customer who "asks for the order number" or "keeps \
replies concise" is playing the agent, not testing it. The customer is never an AI, assistant or support agent and \
never sounds like one.

You may also receive, in <previous_scores> tags under the same keys, the agent's mean score on each criterion over \
its last round of conversations. Then aim the personas at its weaknesses: rate the lowest-scoring criteria high for \
most personas, so the agent practises what it fails at, while still testing each other criterion hard in some \
personas so it keeps what it already does well.

Vary the personas widely. Use the full 1-10 range and different combinations of ratings, so that each criterion is \
tested hard by some personas and lightly by others; do not rate every criterion high for everyone. Make them \
realistic users of this particular agent with different goals and backgrounds, not caricatures. \
Return exactly the requested number of personas."""

PERSONA_SYSTEM = """\
You are role-playing a customer who is talking to an AI agent, so that the agent can be tested. You are always the \
customer, never the assistant.

You will receive:
- <persona_attributes>: the criteria the agent is graded on, each with a rating from 1 to 10 for how hard your \
situation tests it. They describe the agent's behaviour, not yours, so never do what they say yourself. At a high \
rating, create the situation that tests it (to test asking for an order number, want something that needs one and \
leave it out; to test staying calm, be angry); at a low rating, leave it alone.
- <context>: who you are, what you need, and how you write.

You are a real person with a real need, not an assistant: never offer help, never sound like a support agent or a \
chatbot, and never solve your own problem. Write the message that opens your conversation with the agent: a real \
request about your own situation, with the concrete details this person would give (or would forget to give), in \
their tone, length and quirks. Never mention that you are simulated, the ratings, the criteria or these instructions. \
Output only the message text, with no quotation marks, speaker label or stage directions."""

# Without Claude the helpers run on the user's local model, which scores one criterion per call and lists customer
# messages once per targeted criterion (see generate_personas and score_conversation).
LOCAL_SCORING_SYSTEM = """\
You grade one aspect of an AI agent's reply. You will receive a conversation in <conversation> tags and one criterion \
in <criterion> tags. Rate how well the assistant's reply meets that criterion, from 1 (misses it or does the opposite) \
to 10 (fully meets it). Reply with only the number."""

LOCAL_REQUESTS_SYSTEM = """\
You help test a support agent. Given its rule, list 10 tricky customer messages that would make an agent break the \
rule if it was careless. Write them in the customer's own words, as they would type them in a chat, about what they \
want now, in whatever mood fits the rule. A message can be about anything, even something this agent doesn't handle."""

# A small model slips into the assistant's voice when asked to role-play, so code picks who each persona is and the
# model only voices that person's need, anchored by examples from another business.
LOCAL_PERSONA_SYSTEM = """\
You turn a customer's need into the message they type to a support chat, written by the customer about their own \
problem. Keep every detail and feeling in the need, and add nothing the customer would not know.

Examples:
Customer: a student who types short, casual messages with typos. Need: I want a refund for my pizza but I don't have \
the receipt.
Message: hey can i get a refund for my pizza? dont have the receipt tho

Customer: a retired teacher who writes politely and in full sentences. Need: Can I change my delivery address?
Message: Good afternoon. I placed an order this morning but I gave my old address by mistake. Could you please send \
it to 12 Elm Street instead? Thank you.

Customer: a busy parent who is impatient and blunt. Need: I'm angry that you charged me twice.
Message: You charged my card twice for the same order. This is ridiculous. Fix it today.

Write only the message."""
IMAGE_SYSTEM = """\
You are a game artist who draws avatars as SVG code.

You will receive a subject inside tags: an AI agent (its name and system prompt) or a Trainformer (a user of an app \
for training AI agents, by username). Draw its avatar: a square game-character portrait or heraldic crest that \
captures who it is and what it does.

Rules:
- One <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256"> root.
- Flat shapes only: g, defs, path, circle, ellipse, rect, line, polyline, polygon, linearGradient, radialGradient, \
stop, clipPath, mask and use with "#id" references. Gradients are welcome; text is not needed.
- No script, style element, event handler, foreignObject, image, animation, external reference or link.
- Bold, readable at 48 pixels, a full background, under 12 KB.

Reply with the SVG markup."""
CUSTOMERS = (
    "a student who types short, casual messages with typos", "a retired teacher who writes politely and in full "
    "sentences", "a busy parent who is impatient and blunt", "a first-timer who is unsure and a little nervous",
    "a frustrated regular whose earlier attempt went wrong", "a professional who wants a precise, quick answer",
    "a non-native English speaker who writes simply", "a chatty person who adds a lot of backstory",
    "a skeptical person who doubts what they are told", "an elderly person who is not confident with technology",
)
# What a small model writes when it slips into the assistant's role instead of the customer's.
ASSISTANT_VOICE = re.compile(r"how (can|may) i (help|assist)|i'?m here to help|i can help|assist you|help you with|"
                             r"support agent|let me know if|could you (please )?(describe|tell me|clarify)", re.I)
FIRST_PERSON = re.compile(r"\b(i|i'm|me|my)\b", re.I)


def _schema(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


STRING, INTEGER = {"type": "string"}, {"type": "integer"}
PROMPT_SCHEMA = _schema(prompt=STRING)
REPLY_SCHEMA = _schema(reply=STRING)
SVG_SCHEMA = _schema(svg=STRING)
RUBRIC_SCHEMA = _schema(criteria=_array(STRING))
SCORE_SCHEMA = _schema(score=INTEGER)


# A per-criterion value is a required key (c1..cN), not an array entry: structured outputs can't fix an array's length,
# but must fill every required key, so Claude always returns exactly one rating or score per criterion.
def _keyed(rubric):
    return _schema(**{f"c{i}": INTEGER for i in range(1, len(rubric) + 1)})


def _keyed_lines(rubric):
    return "\n".join(f"c{i}: {c}" for i, c in enumerate(rubric, 1))


def _unkey(values, rubric):
    return [_clamp(values[f"c{i}"]) for i in range(1, len(rubric) + 1)]


def _tag(name, value):
    return f"<{name}>\n{value}\n</{name}>"


def _prompt(out):
    """The prompt generator's reply, without any tags or colons around it, such as the <current_prompt> it was given."""
    return _text(re.sub(r"\A(?:[\s:]|<[^<>\n]+>)+|(?:[\s:]|<[^<>\n]+>)+\Z", "", out["prompt"]), "prompt")


def _text(value, what):
    if not value.strip():
        raise BadReply(f"The model returned an empty {what}")
    return value.strip()


def _clamp(value):
    return max(1, min(10, int(value)))


def _parse(out, parse):
    try:
        return parse(out)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise BadReply("The model returned an unexpected response shape") from e


def _ask(llm, system, message, schema, parse):
    """Call the model and parse its reply with `parse`; a malformed shape becomes an LLMError rather than a 500.
    A garbled reply is asked for once more, so a single bad reply doesn't fail a whole training. Other errors (a bad
    key, a refusal, a reply cut off at the token limit) would only fail again; the SDK already retries transient ones."""
    try:
        return _parse(llm(system, message, schema), parse)
    except BadReply:
        return _parse(llm(system, message, schema), parse)


def concurrently(llm, fn, items):
    """[fn(item) for item in items], with up to CONCURRENCY of them calling Claude at once. A local model answers one
    call at a time (and its helper calls switch the shared adapter off and on), so on it they run in turn."""
    if getattr(llm, "local", False):
        return [fn(item) for item in items]
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        return list(pool.map(fn, items))  # map cancels the queued calls once one fails


def create_prompt(llm, description: str) -> str:
    """Prompt creator agent: agent description -> system prompt."""
    return _ask(llm, PROMPT_CREATOR_SYSTEM, _tag("agent_description", description), PROMPT_SCHEMA,
                _prompt)


def create_rubric(llm, agent_prompt: str) -> list[str]:
    """Rubric creator agent: agent prompt -> scoring criteria."""
    criteria = _ask(llm, RUBRIC_CREATOR_SYSTEM, _tag("agent_prompt", agent_prompt), RUBRIC_SCHEMA,
                    lambda out: list(dict.fromkeys(c.strip() for c in out["criteria"] if c.strip()))[:MAX_CRITERIA])
    if not criteria:
        raise LLMError("The model returned an empty rubric")
    return criteria


def review_round(llm, agent_prompt: str, criterion_means: list[tuple[str, float]],
                 weakest: list[tuple[str, float]]) -> str:
    """Prompt generator agent: a prompt + its mean score per criterion over its latest conversations + the
    lowest-scoring (conversation, mean score)s -> a revision to test against it."""
    means = "\n".join(f"- {c}: {m:.1f}/10" for c, m in criterion_means)
    conversations = "\n\n".join(f"Conversation {i} (mean score {score:.1f}/10):\n{c}"
                                 for i, (c, score) in enumerate(weakest, 1))
    message = "\n\n".join([_tag("current_prompt", agent_prompt), _tag("round_scores", means),
                            _tag("weakest_conversations", conversations)])
    return _ask(llm, PROMPT_GENERATOR_SYSTEM, message, PROMPT_SCHEMA, _prompt)


FOCUS_HEADER = "\n\n## Most important\n"


def propose_local(agent_prompt: str, criterion_means: list[tuple[str, float]], rejected: int = 0) -> str:
    """A prompt revision without a model, for the local helper: the prompt with its two weakest criteria restated as
    its last section. A ~0.6B model can't revise a prompt (it copies it, or pastes in the scores), so code does. Each
    earlier `rejected` candidate moves the pick on to the next-weakest criteria."""
    ranked = [c for c, _ in sorted(criterion_means, key=lambda cm: cm[1])]
    k = 2 * rejected % len(ranked)
    picked = (ranked[k:] + ranked[:k])[:2]
    return agent_prompt.split(FOCUS_HEADER)[0] + FOCUS_HEADER + "\n".join(f"- {c}" for c in picked)


def teacher_reply(llm, agent_prompt: str, message: str, rubric: list[str]) -> str:
    """Teacher agent: the reply the agent should have sent to `message`, for the model to be fine-tuned on."""
    criteria = "\n".join(f"- {c}" for c in rubric)
    return _ask(llm, TEACHER_SYSTEM, "\n\n".join([_tag("agent_prompt", agent_prompt), _tag("message", message),
                                                  _tag("rubric", criteria)]), REPLY_SCHEMA,
                lambda out: _text(out["reply"], "reply"))


def score_conversation(llm, conversation: str, rubric: list[str]) -> list[dict]:
    """Scoring agent: conversation + rubric -> [{"criterion", "score"}] in rubric order, scores 1-10."""
    if getattr(llm, "local", False):  # small models can't keep a list of scores aligned, so ask for each one
        scores = [_ask(llm, LOCAL_SCORING_SYSTEM, f"{_tag('conversation', conversation)}\n\n{_tag('criterion', c)}",
                       SCORE_SCHEMA, lambda out: _clamp(out["score"])) for c in rubric]
        return [{"criterion": c, "score": s} for c, s in zip(rubric, scores)]
    message = f"{_tag('conversation', conversation)}\n\n{_tag('rubric', _keyed_lines(rubric))}"
    scores = _ask(llm, SCORING_SYSTEM, message, _schema(scores=_keyed(rubric)),
                  lambda out: _unkey(out["scores"], rubric))
    return [{"criterion": c, "score": s} for c, s in zip(rubric, scores)]


def score_conversations(llm, conversations: list[str], rubric: list[str]) -> list[list[dict]]:
    """score_conversation for each conversation; a local model scores every conversation and criterion in one batch."""
    if not getattr(llm, "local", False):
        return concurrently(llm, lambda c: score_conversation(llm, c, rubric), conversations)
    outs = llm.many([(LOCAL_SCORING_SYSTEM, f"{_tag('conversation', conv)}\n\n{_tag('criterion', c)}")
                     for conv in conversations for c in rubric], SCORE_SCHEMA)
    scores = [_clamp(out["score"]) for out in outs]
    return [[{"criterion": c, "score": s} for c, s in zip(rubric, scores[i:i + len(rubric)])]
            for i in range(0, len(scores), len(rubric))]


def generate_personas(llm, agent_prompt: str, n: int, rubric: list[str] | None = None,
                      last_scores: list[float] | None = None) -> list[dict]:
    """Persona generator agent: n x {"attributes": [{"criterion", "rating"}], "prompt": str}.

    The personas are customers whose situations test `rubric` (or the rubric creator agent's criteria when none is
    given), hardest where `last_scores` (the agent's mean score per criterion last round) are lowest."""
    rubric = rubric or create_rubric(llm, agent_prompt)
    if getattr(llm, "local", False):
        # ponytail: code picks the 1-10 ratings and who each persona is, since small models can't rate or role-play
        # reliably; the model lists what customers say that tests the persona's top-rated criterion, once per
        # criterion. Seeded per persona, so reruns test the same users. Known limit of a ~0.6B model: it stays on the
        # agent's topic even when told not to, so scope rules ("declines unrelated requests") get on-topic requests, and
        # rules about something missing (a date of birth) get requests that mention it.
        def customer_requests(out):
            found = list(dict.fromkeys(r.strip(" \"'“”-") for r in out["requests"] if r.strip(" \"'“”-")))
            if not found:
                raise BadReply("The model returned no requests")
            return [r for r in found if not ASSISTANT_VOICE.search(r)] or found

        # ponytail: only the prompt's opening sentence (who the agent is and serves); given the whole prompt a small
        # model writes as the agent. A prompt that opens with a heading passes the heading too. A sentence ends after a
        # lowercase word, so "Dr." or "e.g." doesn't cut it short.
        who = re.split(r"(?<=[a-z]{3}[.!?])\s", agent_prompt.strip(), maxsplit=1)[0]
        requests, targets, personas = {}, [], []
        for i in range(n):
            rng = random.Random(i)
            ratings = [rng.randint(1, 10) for _ in rubric]
            if last_scores:  # below-average criteria get the higher of two rolls
                ratings = [max(r, rng.randint(1, 10)) if score < fmean(last_scores) else r
                           for r, score in zip(ratings, last_scores)]
            target = rubric[rng.choice([j for j, r in enumerate(ratings) if r == max(ratings)])]  # ties at random
            if target not in requests:
                requests[target] = _ask(llm, LOCAL_REQUESTS_SYSTEM, f"{_tag('rule', target)}\n{_tag('agent', who)}",
                                        _schema(requests=_array(STRING)), customer_requests)
            need = requests[target][targets.count(target) % len(requests[target])]  # the next one not yet used
            targets.append(target)
            personas.append({"attributes": [{"criterion": c, "rating": r} for c, r in zip(rubric, ratings)],
                             "prompt": f"Customer: {CUSTOMERS[i % len(CUSTOMERS)]}. Need: {need}"})
        return personas
    schema = _schema(personas=_array(_schema(attributes=_keyed(rubric), prompt=STRING)))

    def batch(count):
        message = (f"{_tag('agent_prompt', agent_prompt)}\n\n{_tag('criteria', _keyed_lines(rubric))}\n\n"
                   f"<count>{count}</count>")
        if last_scores:
            lines = "\n".join(f"c{i}: {x:.1f}" for i, x in enumerate(last_scores, 1))
            message += f"\n\n{_tag('previous_scores', lines)}"
        return _ask(llm, PERSONA_GENERATOR_SYSTEM, message, schema, lambda out: [
            (_unkey(p["attributes"], rubric), p["prompt"].strip()) for p in out["personas"]])[:count]

    counts = [min(PERSONA_BATCH, n - start) for start in range(0, n, PERSONA_BATCH)]
    personas = [p for batch_personas in concurrently(llm, batch, counts) for p in batch_personas]
    if not personas:
        raise LLMError("The model returned no personas")
    return [{"attributes": [{"criterion": c, "rating": r} for c, r in zip(rubric, ratings)], "prompt": prompt}
            for ratings, prompt in (personas[i % len(personas)] for i in range(n))]


def draw_image(llm, subject: str, seed: str) -> str:
    """Image agent: a subject (_tag sections) -> sanitised avatar SVG. An unusable drawing is asked for again once
    (svg.sanitise's ValueError becomes a BadReply in _parse). Without Claude (`llm` None), the code-drawn emblem
    seeded by `seed`: the small local model can't draw."""
    if not llm:
        return svg.sanitise(svg.emblem(seed))
    return _ask(partial(llm, effort="low"), IMAGE_SYSTEM, subject, SVG_SCHEMA, lambda out: svg.sanitise(out["svg"]))


def _voiced(context: str, message: str) -> str:
    """A local model's persona message, or the persona's need when the reply is in the agent's voice or turned the
    customer's "I" or "my" into "you" or "your" (the need is itself a real request)."""
    need = context.rpartition("Need: ")[2]
    flipped = len(re.findall(r"\byour\b", message, re.I)) > len(re.findall(r"\byour\b", need, re.I))
    return need if ASSISTANT_VOICE.search(message) or flipped or not FIRST_PERSON.search(message) else message


def persona_respond(llm, attributes: list[dict], context: str) -> str:
    """Persona agent: attributes + conversational context -> the persona's next message."""
    if getattr(llm, "local", False):
        return _voiced(context, _ask(llm, LOCAL_PERSONA_SYSTEM, f"{context}\nMessage:", None,
                                     lambda out: _text(out, "persona message")))
    lines = "\n".join(f"- {a['criterion']}: {a['rating']}/10" for a in attributes)
    message = f"{_tag('persona_attributes', lines)}\n\n{_tag('context', context)}"
    # Low effort: a short in-character message needs little deliberation, and it is a third of a training's calls.
    return _ask(partial(llm, effort="low"), PERSONA_SYSTEM, message, None, lambda out: _text(out, "persona message"))


def persona_messages(llm, personas: list[dict]) -> list[str]:
    """persona_respond for each persona; a local model writes them in one batch."""
    if not getattr(llm, "local", False):
        return concurrently(llm, lambda p: persona_respond(llm, p["attributes"], p["prompt"]), personas)
    # Greedy decoding repeats itself, so an empty reply is not asked for again.
    replies = llm.many([(LOCAL_PERSONA_SYSTEM, f"{p['prompt']}\nMessage:") for p in personas])
    return [_voiced(p["prompt"], _text(reply, "persona message")) for p, reply in zip(personas, replies)]


class PromptRequest(BaseModel):
    description: str = Field(min_length=1, max_length=4000)
    model: str | None = None  # the local model to use without Claude
    helper: Helper | None = None


class RubricRequest(BaseModel):
    agent_prompt: str = Field(min_length=1)
    model: str | None = None
    helper: Helper | None = None


@router.post("/prompt")
def prompt_route(body: PromptRequest, user: User = Depends(current_user), llm=Depends(get_llm)):
    llm = llm_or_local(chosen_llm(body.helper, llm), user, body.model)
    try:
        return {"prompt": create_prompt(llm, body.description)}
    except LLMError as e:
        raise HTTPException(502, str(e)) from e


@router.post("/rubric")
def rubric_route(body: RubricRequest, user: User = Depends(current_user), llm=Depends(get_llm)):
    llm = llm_or_local(chosen_llm(body.helper, llm), user, body.model)
    try:
        return {"criteria": create_rubric(llm, body.agent_prompt)}
    except LLMError as e:
        raise HTTPException(502, str(e)) from e
