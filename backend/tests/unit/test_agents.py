import re

import pytest

from app import agents
from app.llm import LLMError


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, system, message, schema=None):
        self.calls.append((system, message, schema))
        return self.outputs.pop(0)


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
        agents.create_prompt(FakeLLM({"prompt": "  "}), "x")


def test_create_rubric_strips_blanks():
    llm = FakeLLM({"criteria": [" Greets the user. ", "", "  ", "Asks for the order number."]})
    assert agents.create_rubric(llm, "You are Acme support.") == ["Greets the user.", "Asks for the order number."]
    system, message, schema = llm.calls[0]
    assert system == agents.RUBRIC_CREATOR_SYSTEM
    assert "<agent_prompt>\nYou are Acme support.\n</agent_prompt>" in message
    assert schema["properties"]["criteria"]["items"] == {"type": "string"}
    assert_strict(schema)


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
        agents.generate_improved_prompt(FakeLLM({"prompt": ""}), "p", "c", [])


def test_score_conversation_aligns_by_order_and_clamps():
    rubric = ["Greets the user.", "Is concise."]
    llm = FakeLLM({"scores": [{"criterion": "greets", "score": 14}, {"criterion": "concise", "score": 0}]})
    assert agents.score_conversation(llm, "user: hi\nassistant: hello", rubric) == [
        {"criterion": "Greets the user.", "score": 10}, {"criterion": "Is concise.", "score": 1}]
    system, message, schema = llm.calls[0]
    assert system == agents.SCORING_SYSTEM
    assert "<conversation>\nuser: hi\nassistant: hello\n</conversation>" in message
    assert "1. Greets the user.\n2. Is concise." in message
    assert schema["properties"]["scores"]["items"]["properties"]["score"] == {"type": "integer"}
    assert_strict(schema)


def test_score_conversation_rejects_count_mismatch():
    llm = FakeLLM({"scores": [{"criterion": "a", "score": 5}]})
    with pytest.raises(LLMError):
        agents.score_conversation(llm, "c", ["a", "b"])


RUBRIC = ["Greets the user.", "Is concise."]


def persona(prompt, *ratings, criteria=RUBRIC):
    return {"attributes": [{"criterion": c, "rating": r} for c, r in zip(criteria, ratings)], "prompt": prompt}


def test_generate_personas_calls_rubric_first_and_clamps():
    llm = FakeLLM({"criteria": RUBRIC},
                  {"personas": [persona(" Busy CFO ", 0, 11), persona("Lost student", 5, 6)]})
    out = agents.generate_personas(llm, "You are Acme support.", 2)
    assert out == [persona("Busy CFO", 1, 10), persona("Lost student", 5, 6)]
    assert [c[0] for c in llm.calls] == [agents.RUBRIC_CREATOR_SYSTEM, agents.PERSONA_GENERATOR_SYSTEM]
    _, message, schema = llm.calls[1]
    assert "<agent_prompt>\nYou are Acme support.\n</agent_prompt>" in message
    assert "- Greets the user.\n- Is concise." in message
    assert "<count>2</count>" in message
    assert_strict(schema)


def test_generate_personas_aligns_attributes_to_rubric():
    reworded = {"personas": [persona("a", 3, 7, criteria=["greets", "concise"])]}
    out = agents.generate_personas(FakeLLM({"criteria": RUBRIC}, reworded), "p", 1)
    assert out == [persona("a", 3, 7)]


@pytest.mark.parametrize("ratings", [(5,), (5, 5, 5)])
def test_generate_personas_rejects_attribute_count_mismatch(ratings):
    bad = {"personas": [persona("a", *ratings, criteria=["x", "y", "z"])]}
    with pytest.raises(LLMError, match="rubric has 2"):
        agents.generate_personas(FakeLLM({"criteria": RUBRIC}, bad), "p", 1)


@pytest.mark.parametrize("call, outputs", [
    (lambda llm: agents.create_prompt(llm, "d"), [{"text": "x"}]),
    (lambda llm: agents.create_rubric(llm, "p"), [{"criteria": [3]}]),
    (lambda llm: agents.generate_improved_prompt(llm, "p", "c", []), ["not json"]),
    (lambda llm: agents.score_conversation(llm, "c", ["a"]), [{"scores": [{"criterion": "a", "score": "high"}]}]),
    (lambda llm: agents.score_conversation(llm, "c", ["a"]), [{"scores": [{"criterion": "a"}]}]),
    (lambda llm: agents.generate_personas(llm, "p", 1), [{"criteria": ["x"]}, {"personas": [{"prompt": "a"}]}]),
    (lambda llm: agents.generate_personas(llm, "p", 1), [{"criteria": ["x"]}, {"personas": None}]),
])
def test_unexpected_output_shape_is_llm_error(call, outputs):
    with pytest.raises(LLMError, match="unexpected response shape"):
        call(FakeLLM(*outputs))


def test_generate_personas_truncates_and_pads_cyclically():
    many = {"personas": [persona(p, 5, criteria=["x"]) for p in "abc"]}
    assert [p["prompt"] for p in agents.generate_personas(FakeLLM({"criteria": ["x"]}, many), "p", 2)] == ["a", "b"]
    few = {"personas": [persona(p, 5, criteria=["x"]) for p in "ab"]}
    out = agents.generate_personas(FakeLLM({"criteria": ["x"]}, few), "p", 5)
    assert [p["prompt"] for p in out] == ["a", "b", "a", "b", "a"]
    assert out[0] is not out[2]


def test_generate_personas_uses_the_given_rubric_in_batches():
    def batch(k):
        return {"personas": [persona(f"p{i}", 5, 6) for i in range(k)]}

    llm = FakeLLM(batch(10), batch(12), batch(5))
    out = agents.generate_personas(llm, "p", 25, rubric=RUBRIC)
    assert len(out) == 25 and all([a["criterion"] for a in p["attributes"]] == RUBRIC for p in out)
    assert [p["prompt"] for p in out[:12]] == [f"p{i}" for i in range(10)] + ["p0", "p1"]
    assert [c[0] for c in llm.calls] == [agents.PERSONA_GENERATOR_SYSTEM] * 3  # no rubric creator call
    assert [re.search(r"<count>(\d+)</count>", c[1]).group(1) for c in llm.calls] == ["10", "10", "5"]
    assert all("- Greets the user.\n- Is concise." in c[1] for c in llm.calls)


def test_generate_personas_rejects_none():
    with pytest.raises(LLMError):
        agents.generate_personas(FakeLLM({"criteria": ["x"]}, {"personas": []}), "p", 3)


def test_persona_respond():
    llm = FakeLLM("  Where is my refund??  ")
    attributes = [{"criterion": "Stays calm.", "rating": 9}, {"criterion": "Is concise.", "rating": 2}]
    assert agents.persona_respond(llm, attributes, "You are an angry customer. Opening message.") == "Where is my refund??"
    system, message, schema = llm.calls[0]
    assert system == agents.PERSONA_SYSTEM
    assert schema is None
    assert "Stays calm.: 9/10" in message and "Is concise.: 2/10" in message
    assert "<context>\nYou are an angry customer. Opening message.\n</context>" in message


def test_persona_respond_rejects_empty():
    with pytest.raises(LLMError):
        agents.persona_respond(FakeLLM(" "), [], "c")


def test_primary_respond_uses_local_generate():
    calls = []

    def generate(system_prompt, conversation):
        calls.append((system_prompt, conversation))
        return "  Hello there.\n"

    assert agents.primary_respond(generate, "You are Acme support.", "user: hi") == "Hello there."
    assert calls == [("You are Acme support.", "user: hi")]
