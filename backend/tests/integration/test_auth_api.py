import pytest

from app import auth
from app.models import User

SIGNUP = {"email": "Ada@Example.com", "username": "ada", "password": "password1", "password_confirm": "password1"}


def signup(client, **overrides):
    return client.post("/api/auth/signup", json={**SIGNUP, **overrides})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("overrides,detail", [
    ({"email": "other@example.com"}, "Username already taken"),
    ({"username": "other"}, "Email already registered"),
])
def test_signup_race_integrity_error_is_409(client, monkeypatch, overrides, detail):
    signup(client)
    real, results = auth._conflict, [None]  # the pre-check misses the concurrent insert
    monkeypatch.setattr(auth, "_conflict", lambda *a: results.pop() if results else real(*a))
    r = signup(client, **overrides)
    assert (r.status_code, r.json()["detail"]) == (409, detail)


def test_signup_without_secret_leaves_no_account(client, db, monkeypatch):
    monkeypatch.delenv("TRAINFORM_SECRET")
    with pytest.raises(RuntimeError, match="TRAINFORM_SECRET"):
        signup(client)
    assert db.query(User).count() == 0


def test_full_account_flow(client):
    r = signup(client)
    assert r.status_code == 201
    body = r.json()
    assert body["user"] == {"id": body["user"]["id"], "email": "ada@example.com", "username": "ada",
                            "claude_linked": False}
    assert client.get("/api/me", headers=bearer(body["token"])).json() == body["user"]

    r = client.post("/api/auth/login", json={"username": "ada", "password": "password1"})
    assert r.status_code == 200
    token = r.json()["token"]
    assert r.json()["user"] == body["user"]

    r = client.post("/api/me/password", headers=bearer(token), json={
        "current_password": "password1", "new_password": "password2", "new_password_confirm": "password2"})
    assert r.status_code == 204
    assert client.post("/api/auth/login", json={"username": "ada", "password": "password1"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ada", "password": "password2"}).status_code == 200


def test_password_stored_hashed(client, db):
    signup(client)
    stored = db.query(User).one().password_hash
    assert stored.startswith("scrypt$") and "password1" not in stored


def test_signup_with_claude_key_stored_encrypted(client, db):
    r = signup(client, anthropic_api_key="  sk-ant-secret  ")
    assert r.json()["user"]["claude_linked"] is True
    assert "sk-ant-secret" not in r.text
    stored = db.query(User).one().anthropic_key_encrypted
    assert "sk-ant-secret" not in stored
    assert auth.decrypt_api_key(stored) == "sk-ant-secret"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_signup_blank_claude_key_not_linked(client, db, key):
    assert signup(client, anthropic_api_key=key).json()["user"]["claude_linked"] is False
    assert db.query(User).one().anthropic_key_encrypted is None


@pytest.mark.parametrize("overrides", [
    {"email": "not-an-email"},
    {"username": "ab"},
    {"username": "a" * 33},
    {"username": "bad name!"},
    {"password": "short", "password_confirm": "short"},
    {"password": "p" * 129, "password_confirm": "p" * 129},
    {"anthropic_api_key": "k" * 513},
])
def test_signup_validation_422(client, overrides):
    assert signup(client, **overrides).status_code == 422


def test_signup_password_mismatch(client):
    r = signup(client, password_confirm="password2")
    assert (r.status_code, r.json()["detail"]) == (400, "Passwords do not match")


def test_signup_duplicates(client):
    signup(client)
    r = signup(client, email="other@example.com")
    assert (r.status_code, r.json()["detail"]) == (409, "Username already taken")
    r = signup(client, username="other", email="ADA@example.com")
    assert (r.status_code, r.json()["detail"]) == (409, "Email already registered")


@pytest.mark.parametrize("creds", [{"username": "ada", "password": "wrongpass"},
                                   {"username": "nobody", "password": "password1"}])
def test_login_failures_identical(client, creds):
    signup(client)
    r = client.post("/api/auth/login", json=creds)
    assert (r.status_code, r.json()["detail"]) == (401, "Invalid username or password")


def test_login_unknown_user_runs_dummy_verify(client, monkeypatch):
    calls = []
    real = auth.verify_password
    monkeypatch.setattr(auth, "verify_password", lambda pw, stored: calls.append(stored) or real(pw, stored))
    assert client.post("/api/auth/login", json={"username": "nobody", "password": "password1"}).status_code == 401
    assert calls == [auth._DUMMY_HASH]


def token(client):
    return signup(client).json()["token"]


def change(client, t, current, new, confirm):
    return client.post("/api/me/password", headers=bearer(t), json={
        "current_password": current, "new_password": new, "new_password_confirm": confirm})


def test_change_password_errors(client):
    t = token(client)
    r = change(client, t, "wrongpass", "password2", "password2")
    assert (r.status_code, r.json()["detail"]) == (400, "Current password is incorrect")
    r = change(client, t, "password1", "password2", "password3")
    assert (r.status_code, r.json()["detail"]) == (400, "Passwords do not match")
    assert change(client, t, "password1", "short", "short").status_code == 422
    assert client.post("/api/me/password", json={
        "current_password": "password1", "new_password": "password2", "new_password_confirm": "password2"}
    ).status_code == 401


def test_claude_key_link_and_unlink(client, db):
    t = token(client)
    r = client.put("/api/me/claude-key", headers=bearer(t), json={"api_key": " sk-ant-linked "})
    assert r.status_code == 200 and r.json()["claude_linked"] is True
    assert "sk-ant-linked" not in r.text
    stored = db.query(User).one().anthropic_key_encrypted
    assert "sk-ant-linked" not in stored and auth.decrypt_api_key(stored) == "sk-ant-linked"
    assert client.get("/api/me", headers=bearer(t)).json()["claude_linked"] is True

    assert client.put("/api/me/claude-key", headers=bearer(t), json={"api_key": "  "}).status_code == 422
    assert client.put("/api/me/claude-key", headers=bearer(t), json={"api_key": "k" * 513}).status_code == 422

    r = client.delete("/api/me/claude-key", headers=bearer(t))
    assert r.status_code == 200 and r.json()["claude_linked"] is False
    db.expire_all()
    assert db.query(User).one().anthropic_key_encrypted is None


@pytest.mark.parametrize("headers", [
    {},
    {"Authorization": ""},
    {"Authorization": "Bearer"},
    {"Authorization": "Bearer garbage"},
    {"Authorization": "Basic YWRhOnBhc3N3b3JkMQ=="},
    {"Authorization": "Bearer " + "1.9999999999.forged"},
])
def test_bad_auth_header_401(client, headers):
    r = client.get("/api/me", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_token_for_deleted_user_401(client, db):
    t = token(client)
    db.query(User).delete()
    db.commit()
    assert client.get("/api/me", headers=bearer(t)).status_code == 401


@pytest.mark.parametrize("method,path", [("put", "/api/me/claude-key"), ("delete", "/api/me/claude-key")])
def test_claude_key_requires_auth(client, method, path):
    assert client.request(method, path, json={"api_key": "k"}).status_code == 401
