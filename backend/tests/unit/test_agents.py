import re
import threading
import time

import pytest

from app import agents
from app.llm import BadReply, LLMError


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


@pytest.mark.parametrize("call", [
    lambda llm: agents.create_prompt(llm, "A bike shop's support agent."),
    lambda llm: agents.review_round(llm, "Old.", [("a", 3.0)], [("c", 3.0)]),
])
@pytest.mark.parametrize("reply", [
    "<current_prompt>\nBetter.\n</current_prompt>",  # Claude echoing the tag it was given
    "<system_prompt>\n<prompt>Better.</prompt>\n</system_prompt>",  # any tag, stacked
    "  <prompt>\nBetter.",  # an opening tag alone
    "Better.\n</agent_prompt> ",  # a closing tag alone
    "<prompt>:\n:  \nBetter.\n:",  # a small model's colons after a tag
])
def test_a_prompt_never_starts_or_ends_with_a_tag(call, reply):
    # The tags would show in every prompt field and reach the model being trained.
    assert call(FakeLLM({"prompt": reply})) == "Better."


def test_tags_inside_a_prompt_are_kept():
    prompt = "Wrap code in <code> tags.\nNever say <secret>."
    assert agents.create_prompt(FakeLLM({"prompt": prompt}), "d") == prompt


def test_review_round():
    llm = FakeLLM({"prompt": " Next round's prompt. "})
    out = agents.review_round(llm, "Round's best prompt.", [("Greets the user.", 8.25), ("Is concise.", 4.5)],
                              [("User: a\nAssistant: b", 3.5), ("User: c\nAssistant: d", 4.0)])
    assert out == "Next round's prompt."
    system, message, schema = llm.calls[0]
    assert system == agents.PROMPT_GENERATOR_SYSTEM and schema == agents.PROMPT_SCHEMA
    assert "<current_prompt>\nRound's best prompt.\n</current_prompt>" in message
    assert "<round_scores>\n- Greets the user.: 8.2/10\n- Is concise.: 4.5/10\n</round_scores>" in message
    assert ("<weakest_conversations>\nConversation 1 (mean score 3.5/10):\nUser: a\nAssistant: b\n\n"
            "Conversation 2 (mean score 4.0/10):\nUser: c\nAssistant: d\n</weakest_conversations>") in message
    assert "<conversation>" not in message


def test_teacher_reply_writes_the_reply_the_agent_should_have_sent():
    llm = FakeLLM({"reply": " Sure, which bike is it? "})
    assert agents.teacher_reply(llm, "You fix bikes.", "my chain fell off", ["Asks for the bike first."]) == (
        "Sure, which bike is it?")
    system, message, schema = llm.calls[0]
    assert system == agents.TEACHER_SYSTEM
    assert "<agent_prompt>\nYou fix bikes.\n</agent_prompt>" in message
    assert "<message>\nmy chain fell off\n</message>" in message and "Asks for the bike first." in message
    assert_strict(schema)


def test_propose_local_restates_the_weakest_criteria_as_the_prompts_last_section():
    means = [("Greets the user.", 9.0), ("Asks for the order number.", 3.0), ("Is concise.", 5.0), ("No jargon.", 7.0)]
    first = agents.propose_local("You help.", means)
    assert first == "You help." + agents.FOCUS_HEADER + "- Asks for the order number.\n- Is concise."
    assert agents.propose_local(first, means).count(agents.FOCUS_HEADER) == 1  # replaced, not stacked
    # After rejected candidates, the next ones press on other criteria.
    assert agents.propose_local(first, means, rejected=1).endswith("- No jargon.\n- Greets the user.")


def test_review_round_rejects_empty():
    with pytest.raises(LLMError):
        agents.review_round(FakeLLM({"prompt": ""}, {"prompt": ""}), "p", [], [])


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


def test_malformed_json_is_retried_once_then_raised():
    llm = FakeLLM(BadReply("Claude returned malformed JSON"), {"prompt": "x"})
    assert agents.create_prompt(llm, "d") == "x"
    with pytest.raises(BadReply, match="malformed JSON"):
        agents.create_prompt(FakeLLM(*[BadReply("Claude returned malformed JSON")] * 2), "d")


@pytest.mark.parametrize("error", ["Claude rejected the API key", "Claude declined this request",
                                   "Claude's reply was cut off at the token limit"])
def test_an_error_asking_again_cant_fix_is_not_retried(error):
    llm = FakeLLM(LLMError(error), {"prompt": "x"})
    with pytest.raises(LLMError, match=error):
        agents.create_prompt(llm, "d")
    assert len(llm.calls) == 1


class LocalLLM(FakeLLM):
    local = True

    def many(self, pairs, schema=None):
        self.batches = getattr(self, "batches", []) + [len(pairs)]
        return [self(system, message, schema) for system, message in pairs]


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
    assert "<count>2</count>" in message and "<previous_scores>" not in message
    system = llm.calls[1][0]  # a persona is a customer whose situation tests the criteria, not one who follows them
    assert "test case" in system and "Never quote, paraphrase or name a criterion" in system
    assert schema["properties"]["personas"]["items"]["properties"]["attributes"] == KEYED_2
    assert_strict(schema)


def test_personas_are_aimed_at_the_last_rounds_weakest_criteria():
    llm = FakeLLM({"personas": [reply("a", 9, 2)]})
    agents.generate_personas(llm, "p", 1, rubric=RUBRIC, last_scores=[3.5, 8.25])
    _, message, _ = llm.calls[0]
    assert "<previous_scores>\nc1: 3.5\nc2: 8.2\n</previous_scores>" in message


def test_local_personas_press_harder_on_the_weakest_criteria():
    rubric = ["Weak.", "Strong."]
    plain = agents.generate_personas(local_requests([]), "p", 30, rubric=rubric)
    aimed = agents.generate_personas(local_requests([]), "p", 30, rubric=rubric, last_scores=[3.0, 9.0])
    rating = lambda ps, i: [p["attributes"][i]["rating"] for p in ps]  # noqa: E731
    assert rating(aimed, 1) == rating(plain, 1)  # the strong criterion is pressed as before
    assert all(a >= b for a, b in zip(rating(aimed, 0), rating(plain, 0)))
    assert sum(rating(aimed, 0)) > sum(rating(plain, 0))


def test_a_persona_missing_a_rating_is_retried_then_an_llm_error():
    bad = {"personas": [reply("a", 5)]}
    llm = FakeLLM(bad, bad)
    with pytest.raises(LLMError, match="unexpected response shape"):
        agents.generate_personas(llm, "p", 1, rubric=RUBRIC)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("call, outputs", [
    (lambda llm: agents.create_prompt(llm, "d"), [{"text": "x"}]),
    (lambda llm: agents.create_rubric(llm, "p"), [{"criteria": [3]}]),
    (lambda llm: agents.teacher_reply(llm, "p", "m", ["a"]), ["not json"]),
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


def local_requests(calls, *extra):
    """A local model that lists two customer requests (plus `extra`) for whichever rule it is asked about."""
    def llm(system, message, schema=None):
        calls.append((system, message, schema))
        rule = re.search(r"<rule>\n(.*)\n</rule>", message).group(1)
        return {"requests": [f' "{rule} A" ', "", f"{rule} B", f"{rule} A", *extra]}
    llm.local = True
    return llm


def test_local_personas_bring_requests_that_test_their_top_rated_criterion():
    calls, rubric = [], ["Asks for the order number.", "Stays calm.", "Declines off-topic requests."]
    llm = local_requests(calls, "How can I help you today?")  # the agent's voice, not a customer's: dropped
    personas = agents.generate_personas(llm, "You are Acme bike support. You must fix bikes.", 12, rubric=rubric)
    tops = [p["prompt"].partition(". Need: ")[2][:-2] for p in personas]  # the rule each need was listed for
    assert all(next(a["rating"] for a in p["attributes"] if a["criterion"] == top)
               == max(a["rating"] for a in p["attributes"]) for p, top in zip(personas, tops))
    assert len(set(tops)) > 1
    rules = [re.search(r"<rule>\n(.*)\n</rule>", m).group(1) for _, m, _ in calls]
    assert sorted(rules) == sorted(set(tops))  # one list call per criterion a persona targets, not per persona
    # only who the agent is: given the whole prompt, a small model writes as the agent
    assert all(s == agents.LOCAL_REQUESTS_SYSTEM and m.endswith("<agent>\nYou are Acme bike support.\n</agent>")
               for s, m, _ in calls)
    for i, (p, top) in enumerate(zip(personas, tops)):
        customer, _, need = p["prompt"].partition(". Need: ")
        assert customer == f"Customer: {agents.CUSTOMERS[i % len(agents.CUSTOMERS)]}"
        assert need == f"{top} {'AB'[tops[:i].count(top) % 2]}"  # personas with the same target take turns
        assert [a["criterion"] for a in p["attributes"]] == rubric  # Inspect shows what each persona targets


def test_local_personas_break_rating_ties_at_random_not_toward_the_first_criterion():
    rubric = [f"Rule {j}." for j in range(4)]
    targets = [p["prompt"].partition(". Need: ")[2][:-2]
               for p in agents.generate_personas(local_requests([]), "p", 1000, rubric=rubric)]
    counts = [targets.count(r) for r in rubric]
    assert max(counts) < 1.25 * min(counts)


def test_local_personas_describe_the_agent_by_its_whole_opening_sentence():
    calls = []
    prompt = "You are the support assistant for Dr. Lee's dental clinic, e.g. booking cleanings. Be kind."
    agents.generate_personas(local_requests(calls), prompt, 1, rubric=["Is concise."])
    assert calls[0][1].endswith(
        "<agent>\nYou are the support assistant for Dr. Lee's dental clinic, e.g. booking cleanings.\n</agent>")


def test_local_personas_keep_requests_in_the_agents_voice_when_that_is_all_there_is():
    llm = LocalLLM({"requests": ["How can I help you with your bike?"]})
    [p] = agents.generate_personas(llm, "p", 1, rubric=["Is concise."])
    assert p["prompt"].endswith("Need: How can I help you with your bike?")


def test_local_personas_need_at_least_one_request():
    with pytest.raises(BadReply, match="no requests"):
        agents.generate_personas(LocalLLM({"requests": [" "]}, {"requests": []}), "p", 2, rubric=RUBRIC)


@pytest.mark.parametrize("reply, sent", [
    (" my chain keeps slipping on hills, can u fix it today? ", "my chain keeps slipping on hills, can u fix it today?"),
    ("I'm here to help! Could you please describe the issue?", "My chain slips"),  # slipped into the agent's voice
    ("I can help with your bike issue.", "My chain slips"),
    ("Your chain slips.", "My chain slips"),  # the customer's "my" turned into the agent's "your"
    ("I need to know your chain size for your repair.", "My chain slips"),  # kept "I" but took the agent's side
])
def test_local_persona_voices_the_customers_need(reply, sent):
    llm = LocalLLM(reply)
    context = f"Customer: {agents.CUSTOMERS[0]}. Need: My chain slips"
    assert agents.persona_respond(llm, [{"criterion": "a", "rating": 5}], context) == sent
    assert llm.calls == [(agents.LOCAL_PERSONA_SYSTEM, f"{context}\nMessage:", None)]


def test_local_system_prompts_are_distinct():
    local = {agents.LOCAL_SCORING_SYSTEM, agents.LOCAL_PERSONA_SYSTEM, agents.LOCAL_REQUESTS_SYSTEM}
    assert len({*SYSTEMS, *local}) == len(SYSTEMS) + 3


def test_persona_respond():
    llm = FakeLLM("  Where is my refund??  ")
    attributes = [{"criterion": "Stays calm.", "rating": 9}, {"criterion": "Is concise.", "rating": 2}]
    assert agents.persona_respond(llm, attributes, "You are an angry customer. Opening message.") == "Where is my refund??"
    system, message, schema = llm.calls[0]
    assert system == agents.PERSONA_SYSTEM
    assert schema is None and llm.kwargs == [{"effort": "low"}]  # a short in-character message needs little thought
    assert "Stays calm.: 9/10" in message and "Is concise.: 2/10" in message
    assert "<context>\nYou are an angry customer. Opening message.\n</context>" in message
    assert "not an assistant" in system and "never offer help" in system  # a person with a need, not another bot
    assert "the agent's behaviour, not yours" in system  # the criteria are tested, never acted out


def test_persona_respond_rejects_empty():
    with pytest.raises(LLMError):
        agents.persona_respond(FakeLLM(" ", " "), [], "c")


def test_local_scoring_of_many_conversations_is_one_batch():
    # A training scores a whole round at once; one batch is what lets the local model generate the scores together.
    llm = LocalLLM({"score": 0}, {"score": 7}, {"score": 12}, {"score": 4})
    assert agents.score_conversations(llm, ["conv A", "conv B"], RUBRIC) == [
        [{"criterion": RUBRIC[0], "score": 1}, {"criterion": RUBRIC[1], "score": 7}],
        [{"criterion": RUBRIC[0], "score": 10}, {"criterion": RUBRIC[1], "score": 4}]]
    assert llm.batches == [4]
    assert [(f"<conversation>\n{c}\n</conversation>" in m, f"<criterion>\n{r}\n</criterion>" in m, schema)
            for (_, m, schema), (c, r) in zip(llm.calls, [(c, r) for c in ("conv A", "conv B") for r in RUBRIC])] == [
        (True, True, agents.SCORE_SCHEMA)] * 4


def test_claude_scores_many_conversations_at_once_in_order():
    together = threading.Barrier(3, timeout=10)  # breaks unless all three calls are in flight at once

    def llm(system, message, schema=None, **kwargs):
        together.wait()
        return {"scores": {"c1": int(re.search(r"conv (\d)", message)[1]), "c2": 5}}

    scores = agents.score_conversations(llm, ["conv 1", "conv 2", "conv 3"], RUBRIC)
    assert [[s["score"] for s in x] for x in scores] == [[1, 5], [2, 5], [3, 5]]


def test_local_persona_messages_are_one_batch():
    contexts = [f"Customer: {agents.CUSTOMERS[0]}. Need: My chain slips", "Customer: x. Need: I want a refund"]
    llm = LocalLLM(" my chain slips on hills ", "How can I help you today?")
    assert agents.persona_messages(llm, [persona(c, 5, 5) for c in contexts]) == [
        "my chain slips on hills", "I want a refund"]  # the second slipped into the agent's voice
    assert llm.batches == [2] and llm.calls[1] == (agents.LOCAL_PERSONA_SYSTEM, f"{contexts[1]}\nMessage:", None)
    with pytest.raises(BadReply):
        agents.persona_messages(LocalLLM(" "), [persona(contexts[0], 5, 5)])


def test_claude_writes_persona_messages_at_once_in_order():
    together = threading.Barrier(2, timeout=10)

    def llm(system, message, schema=None, effort=None):
        together.wait()
        return re.search(r"<context>\n(.*)\n</context>", message)[1].upper()

    assert agents.persona_messages(llm, [persona("first", 5, 5), persona("second", 5, 5)]) == ["FIRST", "SECOND"]


def test_local_personas_do_not_repeat():
    # Generation is greedy, so equal ratings would give equal briefs and the same simulated user again.
    rubric = [f"Rule {j}." for j in range(10)]
    personas = agents.generate_personas(local_requests([]), "p", 20, rubric=rubric)
    assert len({tuple(a["rating"] for a in p["attributes"]) for p in personas}) == 20


def test_concurrently_keeps_order_and_runs_claude_calls_at_once():
    together = threading.Barrier(3, timeout=10)  # breaks unless all three calls are in flight at once
    assert agents.concurrently(FakeLLM(), lambda x: together.wait() * 0 + x * 2, [1, 2, 3]) == [2, 4, 6]


def test_concurrently_stops_starting_calls_once_one_fails():
    started = []

    def fn(x):
        started.append(x)
        if x == 0:
            raise LLMError("boom")
        time.sleep(0.05)

    with pytest.raises(LLMError, match="boom"):
        agents.concurrently(FakeLLM(), fn, range(100))
    assert len(started) < 100


def test_concurrently_runs_local_model_calls_in_turn():
    # The local model answers one call at a time, and its helper calls switch the shared adapter off and on.
    threads = []
    assert agents.concurrently(LocalLLM(), lambda x: threads.append(threading.get_ident()) or x, [1, 2, 3]) == [1, 2, 3]
    assert threads == [threading.get_ident()] * 3
