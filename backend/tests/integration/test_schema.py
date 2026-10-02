import pytest
from sqlalchemy.exc import IntegrityError

from app.models import Scenario, TrainedModel, Training, User


def make_training(db, user):
    t = Training(user_id=user.id, base_model="tiny", initial_prompt="Be helpful.", rubric=["Clarity"],
                 scenarios_per_round=2, rounds=1)
    db.add(t)
    db.commit()
    return t


def make_scenario(training, number=1, round=1):
    return Scenario(training_id=training.id, round=round, scenario_number=number,
                    persona={"attributes": [{"criterion": "Clarity", "rating": 3}], "prompt": "Confused user"},
                    prompt="Be helpful.", input="Hi?", output="Hello!",
                    scores=[{"criterion": "Clarity", "score": 8}], score=8.0)


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_training_defaults_and_user_training_ids(db, user):
    t = make_training(db, user)
    db.refresh(user)
    assert t.status == "pending"
    assert (t.current_round, t.scenarios_done, t.best_model_id) == (0, 0, None)
    assert [tr.id for tr in user.trainings] == [t.id]


def test_username_and_email_unique(db, user):
    db.add(User(email="other@example.com", username="ada", password_hash="x"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.add(User(email="ada@example.com", username="other", password_hash="x"))
    with pytest.raises(IntegrityError):
        db.commit()


def test_scenario_round_number_unique_per_training(db, user):
    t = make_training(db, user)
    db.add(make_scenario(t))
    db.commit()
    db.add(make_scenario(t))
    with pytest.raises(IntegrityError):
        db.commit()


def test_best_model_link_and_json_roundtrip(db, user):
    t = make_training(db, user)
    db.add(make_scenario(t))
    m = TrainedModel(training_id=t.id, round=1, prompt="Better.", weights=b"PK\x03\x04", mean_score=8.0)
    db.add(m)
    db.commit()
    t.best_model_id = m.id
    db.commit()
    db.expire_all()
    s = db.query(Scenario).one()
    assert s.scores == [{"criterion": "Clarity", "score": 8}]
    assert s.persona["attributes"][0]["rating"] == 3
    assert db.get(Training, t.id).best_model_id == m.id
    assert db.get(TrainedModel, m.id).weights == b"PK\x03\x04"


def test_deleting_user_cascades(db, user):
    t = make_training(db, user)
    db.add(make_scenario(t))
    db.commit()
    db.delete(user)
    db.commit()
    assert db.query(Training).count() == 0
    assert db.query(Scenario).count() == 0
