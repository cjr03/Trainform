import re
import threading

import pytest

from app import agents
from app.llm import LLMError


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.kwargs = []

    def __call__(self, system, message, schema=None, **kwargs):
        self.calls.append((system, message, schema))
        self.kwargs.append(kwargs)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def assert_strict(schema):
    """Structured outputs need every object closed with additionalProperties false and full required lists."""
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
        for sub in schema["properties"].values():
            assert_strict(sub)
    if schema.get("type") == "array":
        assert_strict(schema["items"])


SYSTEMS = [agents.PROMPT_CREATOR_SYSTEM, agents.RUBRIC_CREATOR_SYSTEM, agents.PROMPT_GENERATOR_SYSTEM,
           agents.SCORING_SYSTEM, agents.PERSONA_GENERATOR_SYSTEM, agents.PERSONA_SYSTEM]


def test_system_prompts_are_distinct_and_substantial():
    assert len(set(SYSTEMS)) == len(SYSTEMS)
    assert all(len(s) > 400 for s in SYSTEMS)


def test_create_prompt():
    llm = FakeLLM({"prompt": "  You are a support agent.  "})
    assert agents.create_prompt(llm, "A support bot for Acme") == "You are a support agent."
    system, message, schema = llm.calls[0]
    assert system == agents.PROMPT_CREATOR_SYSTEM
    assert "<agent_description>\nA support bot for Acme\n</agent_description>" in message
    assert schema["properties"]["prompt"] == {"type": "string"}
    assert_strict(schema)


def test_create_prompt_rejects_empty():
    with pytest.raises(LLMError):
        agents.create_prompt(FakeLLM({"prompt": "  "}, {"prompt": ""}), "x")


def test_create_rubric_strips_blanks():
    llm = FakeLLM({"criteria": [" Greets the user. ", "", "  ", "Asks for the order number."]})
    assert agents.create_rubric(llm, "You are Acme support.") == ["Greets the user.", "Asks for the order number."]
    system, message, schema = llm.calls[0]
    assert system == agents.RUBRIC_CREATOR_SYSTEM
    assert "<agent_prompt>\nYou are Acme support.\n</agent_prompt>" in message
    assert schema["properties"]["criteria"]["items"] == {"type": "string"}
    assert_strict(schema)


def test_create_rubric_drops_repeats_and_keeps_at_most_eight():
    # Small local models often repeat a line until they run out of tokens.
    llm = FakeLLM({"criteria": ["Is polite.", "Is polite. ", *(f"Rule {i}." for i in range(12))]})
    assert agents.create_rubric(llm, "p") == ["Is polite.", *(f"Rule {i}." for i in range(7))]


def test_create_rubric_rejects_empty():
    with pytest.raises(LLMError):
        agents.create_rubric(FakeLLM({"criteria": [" ", ""]}), "p")


def test_generate_improved_prompt():
    llm = FakeLLM({"prompt": " Better prompt. "})
    scores = [{"criterion": "Greets the user.", "score": 3}, {"criterion": "Is concise.", "score": 9}]
    out = agents.generate_improved_prompt(llm, "Old prompt.", "user: hi\nassistant: yo", scores)
    assert out == "Better prompt."
    system, message, schema = llm.calls[0]
    assert system == agents.PROMPT_GENERATOR_SYSTEM
    assert "<current_prompt>\nOld prompt.\n</current_prompt>" in message
    assert "<conversation>\nuser: hi\nassistant: yo\n</conversation>" in message
    assert "Greets the user.: 3/10" in message and "Is concise.: 9/10" in message
    assert_strict(schema)


def test_generate_improved_prompt_rejects_empty():
    with pytest.raises(LLMError):
        agents.generate_improved_prompt(FakeLLM({"prompt": ""}, {"prompt": ""}), "p", "c", [])


KEYED_2 = {"type": "object", "properties": {"c1": {"type": "integer"}, "c2": {"type": "integer"}},
           "required": ["c1", "c2"], "additionalProperties": False}


def test_score_conversation_maps_keys_to_criteria_and_clamps():
    rubric = ["Greets the user.", "Is concise."]
    llm = FakeLLM({"scores": {"c2": 0, "c1": 14}})
    assert agents.score_conversation(llm, "user: hi\nassistant: hello", rubric) == [
        {"criterion": "Greets the user.", "score": 10}, {"criterion": "Is concise.", "score": 1}]
    system, message, schema = llm.calls[0]
    assert system == agents.SCORING_SYSTEM
    assert "<conversation>\nuser: hi\nassistant: hello\n</conversation>" in message
    assert "<rubric>\nc1: Greets the user.\nc2: Is concise.\n</rubric>" in message
    assert schema["properties"]["scores"] == KEYED_2
    assert_strict(schema)


@pytest.mark.parametrize("n", [1, 4, 8, 30])
def test_keyed_schemas_require_exactly_one_value_per_criterion(n):
    # Structured outputs must fill every required key and no others, so Claude can't return the wrong count.
    rubric = [f"Rule {i}." for i in range(n)]
    keyed = {f"c{i}": 5 for i in range(1, n + 1)}
    llm = FakeLLM({"scores": {**keyed, "c99": 1}}, {"personas": [{"attributes": keyed, "prompt": "a"}]})
    assert len(agents.score_conversation(llm, "c", rubric)) == n
    assert len(agents.generate_personas(llm, "p", 1, rubric=rubric)[0]["attributes"]) == n
    for _, _, schema in llm.calls:
        assert_strict(schema)
    scores, personas = llm.calls[0][2], llm.calls[1][2]
    assert scores["properties"]["scores"]["required"] == list(keyed)
    assert personas["properties"]["personas"]["items"]["properties"]["attributes"]["required"] == list(keyed)


def test_a_garbled_reply_is_retried_once():
    llm = FakeLLM({"scores": {"c1": 5}}, {"scores": {"c1": 7, "c2": 8}})
    assert [s["score"] for s in agents.score_conversation(llm, "c", ["a", "b"])] == [7, 8]
    assert len(llm.calls) == 2 and llm.calls[0] == llm.calls[1]


def test_a_failed_call_is_retried_once_then_raised():
    llm = FakeLLM(LLMError("Claude returned malformed JSON"), {"prompt": "x"})
    assert agents.create_prompt(llm, "d") == "x"
    with pytest.raises(LLMError, match="malformed JSON"):
        agents.create_prompt(FakeLLM(*[LLMError("Claude returned malformed JSON")] * 2), "d")


class LocalLLM(FakeLLM):
    local = True


def test_local_scoring_asks_one_criterion_at_a_time_and_clamps():
    llm = LocalLLM({"score": 0}, {"score": 7})
    assert agents.score_conversation(llm, "user: hi\nassistant: hello", ["Greets the user.", "Is concise."]) == [
        {"criterion": "Greets the user.", "score": 1}, {"criterion": "Is concise.", "score": 7}]
    for (system, message, schema), criterion in zip(llm.calls, ["Greets the user.", "Is concise."]):
        assert system == agents.LOCAL_SCORING_SYSTEM and schema == agents.SCORE_SCHEMA
        assert f"<criterion>\n{criterion}\n</criterion>" in message and "assistant: hello" in message
    assert_strict(agents.SCORE_SCHEMA)


RUBRIC = ["Greets the user.", "Is concise."]


def persona(prompt, *ratings, criteria=RUBRIC):
    return {"attributes": [{"criterion": c, "rating": r} for c, r in zip(criteria, ratings)], "prompt": prompt}


def reply(prompt, *ratings):
    """A persona as Claude returns it: one rating per criterion key."""
    return {"attributes": {f"c{i}": r for i, r in enumerate(ratings, 1)}, "prompt": prompt}


def test_generate_personas_calls_rubric_first_and_clamps():
    llm = FakeLLM({"criteria": RUBRIC},
                  {"personas": [reply(" Busy CFO ", 0, 11), reply("Lost student", 5, 6)]})
    out = agents.generate_personas(llm, "You are Acme support.", 2)
    assert out == [persona("Busy CFO", 1, 10), persona("Lost student", 5, 6)]
    assert [c[0] for c in llm.calls] == [agents.RUBRIC_CREATOR_SYSTEM, agents.PERSONA_GENERATOR_SYSTEM]
    _, message, schema = llm.calls[1]
    assert "<agent_prompt>\nYou are Acme support.\n</agent_prompt>" in message
    assert "<criteria>\nc1: Greets the user.\nc2: Is concise.\n</criteria>" in message
    assert "<count>2</count>" in message
    assert schema["properties"]["personas"]["items"]["properties"]["attributes"] == KEYED_2
    assert_strict(schema)


def test_a_persona_missing_a_rating_is_retried_then_an_llm_error():
    bad = {"personas": [reply("a", 5)]}
    llm = FakeLLM(bad, bad)
    with pytest.raises(LLMError, match="unexpected response shape"):
        agents.generate_personas(llm, "p", 1, rubric=RUBRIC)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("call, outputs", [
    (lambda llm: agents.create_prompt(llm, "d"), [{"text": "x"}]),
    (lambda llm: agents.create_rubric(llm, "p"), [{"criteria": [3]}]),
    (lambda llm: agents.generate_improved_prompt(llm, "p", "c", []), ["not json"]),
    (lambda llm: agents.score_conversation(llm, "c", ["a"]), [{"scores": {"c1": "high"}}]),
    (lambda llm: agents.score_conversation(llm, "c", ["a"]), [{"scores": {}}]),
    (lambda llm: agents.generate_personas(llm, "p", 1), [{"criteria": ["x"]}, {"personas": [{"prompt": "a"}]}]),
    (lambda llm: agents.generate_personas(llm, "p", 1), [{"criteria": ["x"]}, {"personas": None}]),
    (lambda llm: agents.persona_respond(llm, [], "c"), [{"not": "text"}]),
])
def test_unexpected_output_shape_is_llm_error(call, outputs):
    with pytest.raises(LLMError, match="unexpected response shape"):
        call(FakeLLM(*outputs, outputs[-1]))  # the bad reply twice: the first is retried


def test_generate_personas_truncates_and_pads_cyclically():
    many = {"personas": [reply(p, 5) for p in "abc"]}
    assert [p["prompt"] for p in agents.generate_personas(FakeLLM({"criteria": ["x"]}, many), "p", 2)] == ["a", "b"]
    few = {"personas": [reply(p, 5) for p in "ab"]}
    out = agents.generate_personas(FakeLLM({"criteria": ["x"]}, few), "p", 5)
    assert [p["prompt"] for p in out] == ["a", "b", "a", "b", "a"]
    assert out[0] is not out[2]


def test_generate_personas_requests_its_batches_at_once_and_keeps_their_order():
    together = threading.Barrier(3, timeout=10)  # breaks unless all three batches are in flight at once
    calls = []

    def llm(system, message, schema=None):
        calls.append((system, message))
        count = int(re.search(r"<count>(\d+)</count>", message).group(1))
        together.wait()
        return {"personas": [reply(f"{count}-{i}", 5, 6) for i in range(count + 2)]}  # extras are dropped

    out = agents.generate_personas(llm, "p", 25, rubric=RUBRIC)
    assert len(out) == 25 and all([a["criterion"] for a in p["attributes"]] == RUBRIC for p in out)
    assert [p["prompt"] for p in out] == [f"10-{i}" for i in range(10)] * 2 + [f"5-{i}" for i in range(5)]
    assert [c[0] for c in calls] == [agents.PERSONA_GENERATOR_SYSTEM] * 3  # no rubric creator call
    assert sorted(re.search(r"<count>(\d+)</count>", c[1]).group(1) for c in calls) == ["10", "10", "5"]
    assert all("c1: Greets the user.\nc2: Is concise." in c[1] for c in calls)


def test_generate_personas_rejects_none():
    with pytest.raises(LLMError):
        agents.generate_personas(FakeLLM({"criteria": ["x"]}, {"personas": []}), "p", 3)


def test_local_personas_get_a_spread_of_ratings_and_a_written_brief():
    llm = LocalLLM(*({"prompt": f" brief {i} "} for i in range(4)))
    personas = agents.generate_personas(llm, "You are Acme support.", 4, rubric=RUBRIC)
    assert [p["prompt"] for p in personas] == ["brief 0", "brief 1", "brief 2", "brief 3"]
    assert all([a["criterion"] for a in p["attributes"]] == RUBRIC for p in personas)
    ratings = [tuple(a["rating"] for a in p["attributes"]) for p in personas]
    assert all(1 <= r <= 10 for rs in ratings for r in rs) and len(set(ratings)) == 4
    for (system, message, schema), rs in zip(llm.calls, ratings):
        assert system == agents.LOCAL_PERSONA_SYSTEM and schema == agents.PROMPT_SCHEMA
        assert "You are Acme support." in message and f"- Is concise.: {rs[1]}/10" in message


def test_local_system_prompts_are_distinct():
    assert len({*SYSTEMS, agents.LOCAL_SCORING_SYSTEM, agents.LOCAL_PERSONA_SYSTEM}) == len(SYSTEMS) + 2


def test_persona_respond():
    llm = FakeLLM("  Where is my refund??  ")
    attributes = [{"criterion": "Stays calm.", "rating": 9}, {"criterion": "Is concise.", "rating": 2}]
    assert agents.persona_respond(llm, attributes, "You are an angry customer. Opening message.") == "Where is my refund??"
    system, message, schema = llm.calls[0]
    assert system == agents.PERSONA_SYSTEM
    assert schema is None and llm.kwargs == [{"effort": "low"}]  # a short in-character message needs little thought
    assert "Stays calm.: 9/10" in message and "Is concise.: 2/10" in message
    assert "<context>\nYou are an angry customer. Opening message.\n</context>" in message


def test_persona_respond_rejects_empty():
    with pytest.raises(LLMError):
        agents.persona_respond(FakeLLM(" ", " "), [], "c")


def test_primary_respond_uses_local_generate():
    calls = []

    def generate(system_prompt, conversation):
        calls.append((system_prompt, conversation))
        return "  Hello there.\n"

    assert agents.primary_respond(generate, "You are Acme support.", "user: hi") == "Hello there."
    assert calls == [("You are Acme support.", "user: hi")]


def test_local_personas_do_not_repeat():
    # Generation is greedy, so equal ratings would give equal briefs and the same simulated user again.
    rubric = [f"Rule {j}." for j in range(10)]
    llm = LocalLLM(*({"prompt": "brief"} for _ in range(20)))
    personas = agents.generate_personas(llm, "p", 20, rubric=rubric)
    assert len({tuple(a["rating"] for a in p["attributes"]) for p in personas}) == 20


def test_concurrently_keeps_order_and_runs_claude_calls_at_once():
    together = threading.Barrier(3, timeout=10)  # breaks unless all three calls are in flight at once
    assert agents.concurrently(FakeLLM(), lambda x: together.wait() * 0 + x * 2, [1, 2, 3]) == [2, 4, 6]


def test_concurrently_runs_local_model_calls_in_turn():
    # The local model answers one call at a time, and its helper calls switch the shared adapter off and on.
    threads = []
    assert agents.concurrently(LocalLLM(), lambda x: threads.append(threading.get_ident()) or x, [1, 2, 3]) == [1, 2, 3]
    assert threads == [threading.get_ident()] * 3
