import io

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app import agent_profiles, auth, social, svg
from app.main import app
from app.models import Agent, Follow, Post, TrainedModel, Training, User, Vote

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4"/></svg>'
PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64
EMAILS = ("ada@example.com", "bob@example.com", "cat@example.com")


@pytest.fixture
def users(client, db, monkeypatch):
    """ada, bob and cat, each a dict of auth headers keyed by username, plus their ids."""
    monkeypatch.setattr(agent_profiles, "list_local_models", lambda user_id: [{"name": "tiny", "source": "upload"}])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = {}
    for email in EMAILS:
        u = User(email=email, username=email.split("@")[0], password_hash="x", token_version=0)
        db.add(u)
        db.commit()
        out[u.username] = {"Authorization": f"Bearer {auth.create_token(u.id, 0)}"}
        out[u.username + "_id"] = u.id
    return out


def agent(db, user_id, name="Support", base_model="tiny", weights=b"W", image=None, score=None):
    a = Agent(user_id=user_id, name=name, base_model=base_model, prompt="Be helpful.", weights=weights, image=image)
    db.add(a)
    db.commit()
    if score is not None:
        t = Training(user_id=user_id, agent_id=a.id, base_model=base_model, initial_prompt="P", rubric=["C"],
                     scenarios_per_round=1, rounds=1, status="completed")
        db.add(t)
        db.flush()
        db.add(TrainedModel(training_id=t.id, round=1, prompt="P", mean_score=score))
        db.commit()
    return a


def publish(client, headers, agent_id, **body):
    r = client.post("/api/social/posts", json={"agent_id": agent_id, **body}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


def assert_no_secrets(body):
    text_ = str(body)
    assert not any(e in text_ for e in EMAILS) and "password" not in text_ and "token" not in text_
    assert "weights" not in text_ and "anthropic" not in text_


def test_everything_requires_auth(client):
    for method, path in [("get", "/api/social/users"), ("get", "/api/social/users/ada"),
                         ("post", "/api/social/users/ada/follow"), ("delete", "/api/social/users/ada/follow"),
                         ("post", "/api/social/posts"), ("get", "/api/social/feed"), ("delete", "/api/social/posts/1"),
                         ("put", "/api/social/posts/1/vote"), ("post", "/api/social/posts/1/copy"),
                         ("get", "/api/social/users/ada/avatar"), ("post", "/api/me/avatar/generate"),
                         ("delete", "/api/me/avatar"), ("put", "/api/agents/1/image"),
                         ("post", "/api/agents/1/image/generate")]:
        assert getattr(client, method)(path).status_code == 401, path


def test_publish_snapshots_the_agent(client, db, users):
    a = agent(db, users["ada_id"], image=SVG, score=7.36)
    post = publish(client, users["ada"], a.id, caption="  My best  ")
    assert post == {"id": post["id"], "author": "ada", "author_avatar_url": None, "agent_id": a.id, "name": "Support",
                    "base_model": "tiny", "prompt": "Be helpful.", "overall": 74, "image": SVG, "caption": "My best",
                    "score": 0, "my_vote": 0, "created_at": post["created_at"]}
    assert db.get(Post, post["id"]).weights == b"W"
    assert publish(client, users["ada"], a.id, caption="   ")["caption"] is None
    # Later edits and deleting the agent don't change the post.
    client.patch(f"/api/agents/{a.id}", json={"name": "Renamed", "prompt": "Other."}, headers=users["ada"])
    assert client.delete(f"/api/agents/{a.id}", headers=users["ada"]).status_code == 204
    [p] = client.get("/api/social/users/ada", headers=users["bob"]).json()["posts"][1:]
    assert (p["name"], p["prompt"], p["agent_id"], p["overall"]) == ("Support", "Be helpful.", None, 74)
    db.expire_all()
    assert db.get(Post, post["id"]).agent_id is None


def test_publish_validation_and_isolation(client, db, users):
    bobs = agent(db, users["bob_id"])
    r = client.post("/api/social/posts", json={"agent_id": bobs.id}, headers=users["ada"])
    assert (r.status_code, r.json()) == (404, {"detail": "Agent not found"})
    r = client.post("/api/social/posts", json={"agent_id": 999}, headers=users["ada"])
    assert r.status_code == 404
    mine = agent(db, users["ada_id"])
    r = client.post("/api/social/posts", json={"agent_id": mine.id, "caption": "x" * 501}, headers=users["ada"])
    assert r.status_code == 422
    assert db.query(Post).count() == 0
    publish(client, users["ada"], mine.id, caption="x" * 500)


def test_search(client, db, users):
    a = agent(db, users["bob_id"])
    publish(client, users["bob"], a.id)
    publish(client, users["bob"], a.id)
    client.post("/api/social/users/bob/follow", headers=users["cat"])
    r = client.get("/api/social/users", params={"q": "B"}, headers=users["ada"])
    assert r.json() == [{"username": "bob", "avatar_url": None, "agents": 2, "followers": 1, "following": False}]
    everyone = client.get("/api/social/users", headers=users["ada"]).json()
    assert [u["username"] for u in everyone] == ["bob", "cat"]  # never yourself
    assert_no_secrets(everyone)
    assert client.get("/api/social/users", params={"q": "%"}, headers=users["ada"]).json() == []  # LIKE escaped
    assert client.get("/api/social/users", params={"q": "_"}, headers=users["ada"]).json() == []


def test_search_caps_at_twenty(client, db, users):
    db.add_all(User(email=f"u{i}@example.com", username=f"user{i:02}", password_hash="x") for i in range(25))
    db.commit()
    assert len(client.get("/api/social/users", params={"q": "user"}, headers=users["ada"]).json()) == 20


def test_follow_unfollow_and_profile(client, db, users):
    url = "/api/social/users/bob/follow"
    for _ in range(2):  # idempotent
        assert client.post(url, headers=users["ada"]).status_code == 204
    assert db.query(Follow).count() == 1
    client.post("/api/social/users/cat/follow", headers=users["bob"])
    prof = client.get("/api/social/users/bob", headers=users["ada"]).json()
    assert prof == {"username": "bob", "avatar_url": None, "followers": 1, "following_count": 1, "following": True,
                    "posts": []}
    assert client.get("/api/social/users/bob", headers=users["cat"]).json()["following"] is False
    for _ in range(2):
        assert client.delete(url, headers=users["ada"]).status_code == 204
    assert client.get("/api/social/users/bob", headers=users["ada"]).json()["followers"] == 0
    r = client.post("/api/social/users/ada/follow", headers=users["ada"])
    assert (r.status_code, r.json()) == (400, {"detail": "You can't follow yourself."})
    for method in ("get", "post", "delete"):
        path = "/api/social/users/nobody" + ("" if method == "get" else "/follow")
        assert getattr(client, method)(path, headers=users["ada"]).status_code == 404
    assert client.get("/api/social/users/ada", headers=users["ada"]).json()["username"] == "ada"  # your own


def test_feed_is_followed_newest_first_and_paginated(client, db, users):
    b, c, a = agent(db, users["bob_id"]), agent(db, users["cat_id"]), agent(db, users["ada_id"])
    ids = [publish(client, users["bob"], b.id)["id"] for _ in range(3)]
    publish(client, users["cat"], c.id)  # not followed
    publish(client, users["ada"], a.id)  # your own aren't in your feed
    assert client.get("/api/social/feed", headers=users["ada"]).json() == []
    client.post("/api/social/users/bob/follow", headers=users["ada"])
    feed = client.get("/api/social/feed", headers=users["ada"]).json()
    assert [p["id"] for p in feed] == ids[::-1]
    assert all(p["agent_id"] is None and p["author"] == "bob" for p in feed)  # someone else's agent id stays hidden
    assert_no_secrets(feed)
    page = client.get("/api/social/feed", params={"limit": 2}, headers=users["ada"]).json()
    assert [p["id"] for p in page] == [ids[2], ids[1]]
    rest = client.get("/api/social/feed", params={"before": ids[1], "limit": 2}, headers=users["ada"]).json()
    assert [p["id"] for p in rest] == [ids[0]]
    assert client.get("/api/social/feed", params={"before": 0}, headers=users["ada"]).json() == []
    for limit in (0, 51):
        assert client.get("/api/social/feed", params={"limit": limit}, headers=users["ada"]).status_code == 422


def test_delete_own_posts_only(client, db, users):
    post = publish(client, users["bob"], agent(db, users["bob_id"]).id)
    r = client.delete(f"/api/social/posts/{post['id']}", headers=users["ada"])
    assert (r.status_code, r.json()) == (404, {"detail": "Post not found"})
    assert db.query(Post).count() == 1
    assert client.delete(f"/api/social/posts/{post['id']}", headers=users["bob"]).status_code == 204
    assert client.delete(f"/api/social/posts/{post['id']}", headers=users["bob"]).status_code == 404
    assert db.query(Post).count() == 0


def test_votes(client, db, users):
    post = publish(client, users["bob"], agent(db, users["bob_id"]).id)
    url = f"/api/social/posts/{post['id']}/vote"

    def put(who, value):
        r = client.put(url, json={"value": value}, headers=users[who])
        assert r.status_code == 200, r.text
        return r.json()

    assert put("ada", 1) == {"score": 1, "my_vote": 1}
    assert put("ada", 1) == {"score": 1, "my_vote": 1}  # one vote each
    assert put("cat", 1) == {"score": 2, "my_vote": 1}
    assert put("ada", -1) == {"score": 0, "my_vote": -1}  # toggled
    assert put("ada", 0) == {"score": 1, "my_vote": 0}  # removed
    assert put("ada", 0) == {"score": 1, "my_vote": 0}
    assert db.query(Vote).count() == 1
    [p] = client.get("/api/social/users/bob", headers=users["cat"]).json()["posts"]
    assert (p["score"], p["my_vote"]) == (1, 1)
    r = client.put(url, json={"value": 1}, headers=users["bob"])
    assert (r.status_code, r.json()) == (400, {"detail": "You can't vote on your own agent."})
    for value in (2, -2, "up", None):
        assert client.put(url, json={"value": value}, headers=users["ada"]).status_code == 422
    assert client.put("/api/social/posts/999/vote", json={"value": 1}, headers=users["ada"]).status_code == 404


def test_copy_post(client, db, users):
    b = agent(db, users["bob_id"], name="Support", weights=b"BOBW", image=SVG, score=9.0)
    post = publish(client, users["bob"], b.id)
    agent(db, users["ada_id"], name="Support")
    agent(db, users["ada_id"], name="Support (2)")
    r = client.post(f"/api/social/posts/{post['id']}/copy", headers=users["ada"])
    assert r.status_code == 201
    copy = r.json()
    assert copy == {"id": copy["id"], "name": "Support (3)", "base_model": "tiny", "prompt": "Be helpful.",
                    "image": SVG, "model_available": True, "overall": None, "trainings": 0,
                    "created_at": copy["created_at"]}
    new = db.get(Agent, copy["id"])
    assert (new.user_id, new.weights) == (users["ada_id"], b"BOBW")
    assert client.post(f"/api/social/posts/{post['id']}/copy", headers=users["ada"]).json()["name"] == "Support (4)"
    # Your own post copies too, and a name at the length limit is cut to fit its suffix.
    long = agent(db, users["cat_id"], name="x" * 100, base_model="missing-model")
    lp = publish(client, users["cat"], long.id)
    first = client.post(f"/api/social/posts/{lp['id']}/copy", headers=users["cat"]).json()
    assert (first["name"], first["model_available"]) == ("x" * 96 + " (2)", False)
    listed = {a["name"]: a for a in client.get("/api/agents", headers=users["cat"]).json()}
    assert listed["x" * 100]["model_available"] is False
    assert client.post("/api/social/posts/999/copy", headers=users["ada"]).status_code == 404


def test_copy_name_race_is_a_conflict(client, db, users, monkeypatch):
    post = publish(client, users["bob"], agent(db, users["bob_id"]).id)
    agent(db, users["ada_id"], name="Support")
    # The names as read before a concurrent copy took "Support".
    monkeypatch.setattr(Session, "scalars", lambda self, *a, **k: iter([]))
    r = client.post(f"/api/social/posts/{post['id']}/copy", headers=users["ada"])
    assert (r.status_code, r.json()) == (409, {"detail": "You already have an agent with that name."})


def test_claude_agents_are_always_available(client, db, users):
    a = agent(db, users["ada_id"], base_model="claude", weights=None)
    assert client.get(f"/api/agents/{a.id}", headers=users["ada"]).json()["model_available"] is True


# --- Agent images


def test_put_agent_image_sanitises_and_clears(client, db, users):
    a = agent(db, users["ada_id"])
    url = f"/api/agents/{a.id}/image"
    evil = '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><script>alert(1)</script><circle r="4"/></svg>'
    r = client.put(url, json={"svg": evil}, headers=users["ada"])
    assert (r.status_code, r.json()["image"]) == (200, SVG.replace('"/>', '" />'))
    db.expire_all()
    assert db.get(Agent, a.id).image == SVG.replace('"/>', '" />')
    for bad, detail in [("<html/>", "The root element must be <svg>."),
                        ('<!DOCTYPE svg [<!ENTITY x "y">]><svg/>', "The SVG can't contain a DOCTYPE or entities."),
                        ("<svg>" + "x" * 20_001 + "</svg>", "The SVG is larger than 20 KB.")]:
        r = client.put(url, json={"svg": bad}, headers=users["ada"])
        assert (r.status_code, r.json()) == (400, {"detail": detail})
    assert client.put(url, json={"svg": None}, headers=users["ada"]).json()["image"] is None
    assert client.put(url, json={}, headers=users["ada"]).status_code == 422
    bobs = agent(db, users["bob_id"])
    assert client.put(f"/api/agents/{bobs.id}/image", json={"svg": SVG}, headers=users["ada"]).status_code == 404
    assert client.post(f"/api/agents/{bobs.id}/image/generate", json={}, headers=users["ada"]).status_code == 404
    db.expire_all()
    assert db.get(Agent, bobs.id).image is None


def test_generate_agent_image(client, db, users, monkeypatch):
    a = agent(db, users["ada_id"], name="Knight")
    url = f"/api/agents/{a.id}/image/generate"
    # Without Claude: the code-drawn emblem, no model call.
    r = client.post(url, json={}, headers=users["ada"])
    assert (r.status_code, r.json()["image"]) == (200, svg.sanitise(svg.emblem("Knight")))
    r = client.post(url, json={"helper": "claude"}, headers=users["ada"])
    assert (r.status_code, r.json()) == (400, {"detail": "Link your Claude API key on the Account page first."})
    seen = []

    def claude(system, message, schema=None, effort="medium"):
        seen.append(message)
        return {"svg": '<svg xmlns="http://www.w3.org/2000/svg"><rect onclick="x()"/></svg>'}
    monkeypatch.setattr("app.llm.call_claude", lambda key, *a, **k: claude(*a, **k))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    r = client.post(url, json={"helper": "claude"}, headers=users["ada"])
    assert r.json()["image"] == '<svg xmlns="http://www.w3.org/2000/svg"><rect /></svg>'
    assert "<agent_name>\nKnight\n</agent_name>" in seen[0] and "Be helpful." in seen[0]
    r = client.post(url, json={"helper": "local"}, headers=users["ada"])
    assert r.json()["image"] == svg.sanitise(svg.emblem("Knight"))
    monkeypatch.setattr("app.llm.call_claude", lambda key, *a, **k: {"svg": "<html/>"})
    r = client.post(url, json={}, headers=users["ada"])
    assert (r.status_code, r.json()) == (502, {"detail": "The model returned an unexpected response shape"})


# --- Profile pictures


def test_avatar_generate_get_and_delete(client, db, users):
    r = client.post("/api/me/avatar/generate", json={}, headers=users["ada"])
    assert r.status_code == 200
    assert r.json()["avatar_url"].startswith("/social/users/ada/avatar?v=")
    r = client.get("/api/social/users/ada/avatar", headers=users["bob"])
    assert r.status_code == 200
    assert r.content.decode() == svg.sanitise(svg.emblem("ada"))
    assert r.headers["content-type"] == "image/svg+xml"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["content-security-policy"] == "default-src 'none'; style-src 'unsafe-inline'"
    assert r.headers["cache-control"] == "private, no-cache"  # a changed picture shows at once
    url = "/social/users/ada/avatar"
    assert client.get("/api/me", headers=users["ada"]).json()["avatar_url"] == url
    assert client.get("/api/social/users", params={"q": "ada"}, headers=users["bob"]).json()[0]["avatar_url"] == url
    assert client.get("/api/social/users/ada", headers=users["bob"]).json()["avatar_url"] == url
    post = publish(client, users["ada"], agent(db, users["ada_id"]).id)
    assert post["author_avatar_url"] == url
    assert client.delete("/api/me/avatar", headers=users["ada"]).status_code == 204
    assert client.get("/api/social/users/ada/avatar", headers=users["bob"]).status_code == 404
    assert client.get("/api/me", headers=users["ada"]).json()["avatar_url"] is None
    assert client.get("/api/social/users/nobody/avatar", headers=users["bob"]).status_code == 404


def test_avatar_generate_with_claude_uses_the_same_sanitiser(client, db, users, monkeypatch):
    monkeypatch.setattr("app.llm.call_claude", lambda key, system, message, schema=None, effort="medium": {
        "svg": '<svg xmlns="http://www.w3.org/2000/svg"><circle r="4" onmouseover="x()"/><style>*{}</style></svg>'})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "server-key")
    monkeypatch.setenv("TRAINFORM_SHARED_KEY_USERS", "ada")
    assert client.post("/api/me/avatar/generate", json={"helper": "claude"}, headers=users["ada"]).status_code == 200
    assert client.get("/api/social/users/ada/avatar", headers=users["ada"]).text == SVG.replace('"/>', '" />')
    monkeypatch.setattr("app.llm.call_claude", lambda *a, **k: {"svg": "<script/>"})
    assert client.post("/api/me/avatar/generate", json={}, headers=users["ada"]).status_code == 502


def upload(client, headers, data, name="a.png", content_type="image/png"):
    return client.put("/api/me/avatar", files={"file": (name, data, content_type)}, headers=headers)


@pytest.mark.parametrize("data,kind", [
    (PNG, "image/png"), (b"\xff\xd8\xff\xe0" + b"\0" * 64, "image/jpeg"),
    (b"RIFF\x10\0\0\0WEBPVP8 " + b"\0" * 64, "image/webp"),
])
def test_avatar_upload_raster_by_magic_bytes(client, db, users, data, kind):
    r = upload(client, users["ada"], data, "pic.bin", "application/octet-stream")  # the client's type is ignored
    assert r.status_code == 200
    got = client.get("/api/social/users/ada/avatar", headers=users["ada"])
    assert (got.content, got.headers["content-type"]) == (data, kind)


def test_avatar_upload_svg_is_sanitised(client, db, users):
    evil = b'\xef\xbb\xbf<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"><circle r="4"/></svg>'
    assert upload(client, users["ada"], evil, "a.svg", "image/svg+xml").status_code == 200
    got = client.get("/api/social/users/ada/avatar", headers=users["ada"])
    assert (got.text, got.headers["content-type"]) == (SVG.replace('"/>', '" />'), "image/svg+xml")


@pytest.mark.parametrize("data,detail", [
    (b"<html><body><script>alert(1)</script></body></html>",
     "Upload a PNG, JPEG, WebP or SVG picture. The root element must be <svg>."),
    (b"GIF89a\xff" + b"\0" * 32, "Upload a PNG, JPEG, WebP or SVG picture."),  # not UTF-8, no known magic
    (b"\x89PNX fake png", "Upload a PNG, JPEG, WebP or SVG picture."),
    (b"just text pretending to be a png", "Upload a PNG, JPEG, WebP or SVG picture. That isn't valid SVG markup."),
    (b"RIFF\0\0\0\0WAVE", "Upload a PNG, JPEG, WebP or SVG picture. That isn't valid SVG markup."),
    (b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg>&x;</svg>',
     "Upload a PNG, JPEG, WebP or SVG picture. The SVG can't contain a DOCTYPE or entities."),
])
def test_avatar_upload_rejects_spoofed_and_unknown_types(client, db, users, data, detail):
    r = upload(client, users["ada"], data)  # all claim image/png
    assert (r.status_code, r.json()) == (400, {"detail": detail})
    db.expire_all()
    assert db.get(User, users["ada_id"]).avatar_type is None


def test_avatar_upload_size_cap(client, db, users):
    r = upload(client, users["ada"], PNG + b"\0" * (512 * 1024 - len(PNG) + 1))
    assert r.status_code == 413  # the middleware, from Content-Length
    assert upload(client, users["ada"], PNG + b"\0" * (512 * 1024 - len(PNG))).status_code == 200


def test_avatar_size_cap_without_the_middleware(client, db, users):
    """A body whose Content-Length undersells it still stops at 512 KB in the route."""
    big = UploadFile(io.BytesIO(PNG + b"\0" * (512 * 1024)))
    me = db.get(User, users["ada_id"])
    with pytest.raises(HTTPException) as e:
        social.put_avatar(big, me, db)
    assert (e.value.status_code, e.value.detail) == (413, "Pictures can be up to 512 KB.")


def test_avatar_upload_needs_a_content_length(client, users):
    def chunks():
        yield b"x"
    r = client.put("/api/me/avatar", content=chunks(), headers=users["ada"] | {"Content-Type": "multipart/form-data"})
    assert (r.status_code, r.json()) == (411, {"detail": "Content-Length is required"})
    assert client.delete("/api/me/avatar", headers=users["ada"]).status_code == 204  # no body, no cap


def test_avatar_changes_only_your_own(client, db, users):
    upload(client, users["bob"], PNG)
    upload(client, users["ada"], b"\xff\xd8\xff" + b"\0" * 8)
    client.delete("/api/me/avatar", headers=users["ada"])
    client.post("/api/me/avatar/generate", json={}, headers=users["ada"])
    got = client.get("/api/social/users/bob/avatar", headers=users["ada"])
    assert (got.content, got.headers["content-type"]) == (PNG, "image/png")


def test_no_email_or_secrets_in_any_social_response(client, db, users):
    upload(client, users["bob"], PNG)
    post = publish(client, users["bob"], agent(db, users["bob_id"], image=SVG).id)
    client.post("/api/social/users/bob/follow", headers=users["ada"])
    for body in (client.get("/api/social/users", headers=users["ada"]).json(),
                 client.get("/api/social/users/bob", headers=users["ada"]).json(),
                 client.get("/api/social/feed", headers=users["ada"]).json(),
                 client.put(f"/api/social/posts/{post['id']}/vote", json={"value": 1}, headers=users["ada"]).json(),
                 post):
        assert_no_secrets(body)


def test_social_tables_cascade(db, users, client):
    a = agent(db, users["bob_id"])
    post = publish(client, users["bob"], a.id)
    client.post("/api/social/users/bob/follow", headers=users["ada"])
    client.put(f"/api/social/posts/{post['id']}/vote", json={"value": 1}, headers=users["ada"])
    db.execute(text("DELETE FROM users WHERE username = 'bob'"))
    db.commit()
    assert (db.query(Post).count(), db.query(Follow).count(), db.query(Vote).count()) == (0, 0, 0)


def test_startup_adds_the_new_columns(db, user):
    db.execute(text("ALTER TABLE users DROP COLUMN avatar, DROP COLUMN avatar_type"))
    db.execute(text("ALTER TABLE agents DROP COLUMN image"))
    db.commit()
    with TestClient(app):
        pass
    db.expire_all()
    assert db.execute(text("SELECT avatar, avatar_type FROM users")).one() == (None, None)
    assert db.execute(text("SELECT count(image) FROM agents")).scalar() == 0
