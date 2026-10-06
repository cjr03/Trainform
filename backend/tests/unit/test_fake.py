import io
import json
import math
import zipfile

from app import agents, fake
from app.training import SFT_FLOOR

RUBRIC = ["Greets the user.", "Asks one clarifying question.", "Never invents facts."]


def test_fake_llm_satisfies_every_agent():
    prompt = agents.create_prompt(fake.fake_llm, "A support bot for Acme")
    assert "A support bot for Acme" in prompt
    assert 3 <= len(agents.create_rubric(fake.fake_llm, prompt)) <= 4
    personas = agents.generate_personas(fake.fake_llm, prompt, 5)
    assert len(personas) == 5
    rubric = agents.create_rubric(fake.fake_llm, prompt)
    assert all([a["criterion"] for a in p["attributes"]] == rubric for p in personas)
    means = []
    for p in personas:
        message = agents.persona_respond(fake.fake_llm, p["attributes"], p["prompt"])
        scores = agents.score_conversation(fake.fake_llm, f"User: {message}\nAssistant: Hello!", RUBRIC)
        assert [s["criterion"] for s in scores] == RUBRIC
        means.append(sum(s["score"] for s in scores) / len(scores))
    assert max(means) >= SFT_FLOOR > min(means)
    improved = agents.review_round(fake.fake_llm, prompt, [(c, 5.0) for c in RUBRIC], [("User: hi\nAssistant: hello", 5.0)])
    assert improved != prompt and improved.startswith(prompt)
    assert agents.review_round(fake.fake_llm, improved, [], []) == improved  # does not keep growing
    assert agents.teacher_reply(fake.fake_llm, prompt, "hi", RUBRIC)
    assert agents.draw_image(fake.fake_llm, "<username>\nada\n</username>", "ada").startswith("<svg")


def test_fake_local_model():
    model = fake.FakeLocalModel("demo")
    stops = []
    assert model.generate_many([("Be helpful.", "User: hi")] * 2, stop=lambda: stops.append(1))[1].strip()
    assert len(stops) == 2  # E2E slows each reply down, so a pause is checked before each
    loss = model.sft([{"prompt": "p", "input": "i", "output": "o"}] * 2, stop=lambda: stops.append(1))
    assert isinstance(loss, float) and math.isfinite(loss)
    assert model.examples_seen == 2 and len(stops) == 3
    with zipfile.ZipFile(io.BytesIO(model.adapter_bytes())) as zf:
        assert json.loads(zf.read("adapter_config.json"))["peft_type"] == "LORA"
    resumed = fake.FakeLocalModel("demo")
    resumed.load_adapter(model.adapter_bytes())
    assert resumed.examples_seen == 2 and resumed.adapter_bytes() == model.adapter_bytes()
