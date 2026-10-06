import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Annotated

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field, StringConstraints
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import fake
from app.db import get_db
from app.models import DEFAULT_BASE_MODEL, SEED_AGENT, Agent, User

router = APIRouter(tags=["auth"])

# Tokens carry the user's token_version; password changes and sign-outs bump it to revoke every session.
TOKEN_TTL = 7 * 24 * 3600
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
LOGIN_WINDOW, LOGIN_MAX_FAILURES = 15 * 60, 5
# ponytail: per-process failure log keyed on (username, client IP); behind a proxy every client shares its IP, so
# this becomes a per-username lockout. Move it to the proxy or Redis when running several workers.
_login_failures: dict[tuple[str, str], list[float]] = {}


def _secret() -> bytes:
    secret = os.environ.get("TRAINFORM_SECRET")
    if not secret:
        raise RuntimeError("TRAINFORM_SECRET is not set; set it to a long random string, e.g. the output of "
                           "`python -c 'import secrets; print(secrets.token_urlsafe(32))'`")
    if len(secret.encode()) < 32:  # it keys token HMACs and the Claude-key cipher; any token is an offline oracle
        raise RuntimeError("TRAINFORM_SECRET must be at least 32 bytes; use e.g. `openssl rand -hex 32`")
    return secret.encode()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    return f"scrypt${salt.hex()}${hashlib.scrypt(password.encode(), salt=salt, **SCRYPT).hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(digest)
        return hmac.compare_digest(hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **SCRYPT), expected)
    except ValueError:
        return False


_DUMMY_HASH = hash_password("timing-dummy")


def _sign(payload: str) -> str:
    mac = hmac.new(_secret(), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()


def create_token(user_id: int, version: int) -> str:
    payload = f"{user_id}.{version}.{int(time.time()) + TOKEN_TTL}"
    return f"{payload}.{_sign(payload)}"


def verify_token(token: str) -> tuple[int, int] | None:
    """(user_id, token_version) for a valid, unexpired token, else None."""
    payload, _, sig = token.rpartition(".")
    if not payload or not hmac.compare_digest(sig.encode(), _sign(payload).encode()):
        return None
    try:
        user_id, version, expires = map(int, payload.split("."))
    except ValueError:
        return None
    return (user_id, version) if expires > time.time() else None


def _fernet() -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"trainform-claude-key:" + _secret()).digest()))


def encrypt_api_key(api_key: str) -> str:
    return _fernet().encrypt(api_key.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: the User for a valid `Authorization: Bearer <token>` header, else 401."""
    scheme, _, token = (authorization or "").partition(" ")
    claims = verify_token(token) if scheme.lower() == "bearer" else None
    user = db.get(User, claims[0]) if claims else None
    if user is None or user.token_version != claims[1]:
        raise HTTPException(401, "Not authenticated", headers={"WWW-Authenticate": "Bearer"})
    return user


Password = Annotated[str, Field(min_length=8, max_length=128)]


class Signup(BaseModel):
    email: EmailStr
    username: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{3,32}$")]
    password: Password
    password_confirm: str
    anthropic_api_key: Annotated[str, StringConstraints(strip_whitespace=True, max_length=512)] | None = None


class Login(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: Password
    new_password_confirm: str


class ClaudeKey(BaseModel):
    api_key: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


def shared_key(user: User) -> str | None:
    """The operator's Claude key, for allow-listed usernames only, so open sign-up can't spend it."""
    # ponytail: no per-user budget; add a daily call counter if allow-listed users need limits.
    shared = {u.strip() for u in os.environ.get("TRAINFORM_SHARED_KEY_USERS", "").split(",")}
    return os.environ.get("ANTHROPIC_API_KEY") if user.username in shared else None


def avatar_url(username: str, avatar_type: str | None) -> str | None:
    """The API path (under /api) of a Trainformer's picture (see social.get_avatar), or None when they have none."""
    return avatar_type and f"/social/users/{username}/avatar"


def user_out(user: User) -> dict:
    linked = user.anthropic_key_encrypted is not None
    return {"id": user.id, "email": user.email, "username": user.username, "claude_linked": linked,
            "claude_available": linked or bool(shared_key(user)) or fake.enabled(),  # what Train can offer
            "avatar_url": avatar_url(user.username, user.avatar_type)}


def _session(user: User) -> dict:
    return {"token": create_token(user.id, user.token_version), "user": user_out(user)}


def _conflict(db: Session, username: str, email: str) -> str | None:
    if db.query(User.id).filter_by(username=username).first():
        return "Username already taken"
    if db.query(User.id).filter_by(email=email).first():
        return "Email already registered"
    return None


@router.post("/auth/signup", status_code=201)
def signup(body: Signup, db: Session = Depends(get_db)):
    if body.password != body.password_confirm:
        raise HTTPException(400, "Passwords do not match")
    email = body.email.lower()
    if conflict := _conflict(db, body.username, email):
        raise HTTPException(409, conflict)
    user = User(email=email, username=body.username, password_hash=hash_password(body.password), token_version=0,
                anthropic_key_encrypted=encrypt_api_key(body.anthropic_api_key) if body.anthropic_api_key else None)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:  # lost a race with a concurrent signup
        db.rollback()
        raise HTTPException(409, _conflict(db, body.username, email) or "Username already taken")
    db.add(Agent(user_id=user.id, base_model=DEFAULT_BASE_MODEL, **SEED_AGENT))
    session = _session(user)  # before commit: a missing TRAINFORM_SECRET must not leave an account behind
    db.commit()
    return session


@router.post("/auth/login")
def login(body: Login, request: Request, db: Session = Depends(get_db)):
    key, now = (body.username, request.client.host if request.client else ""), time.time()
    recent = [t for t in _login_failures.get(key, []) if t > now - LOGIN_WINDOW]
    if len(recent) >= LOGIN_MAX_FAILURES:
        raise HTTPException(429, "Too many failed sign-in attempts; try again later.")
    user = db.query(User).filter_by(username=body.username).first()
    if not verify_password(body.password, user.password_hash if user else _DUMMY_HASH) or user is None:
        if len(_login_failures) > 10_000:  # bounds memory under a spray of usernames
            _login_failures.clear()
        _login_failures[key] = recent + [now]
        raise HTTPException(401, "Invalid username or password")
    _login_failures.pop(key, None)
    return _session(user)


@router.post("/auth/logout", status_code=204)
def logout(user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.token_version += 1  # revokes every session of this user
    db.commit()


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@router.post("/me/password")
def change_password(body: PasswordChange, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    if body.new_password != body.new_password_confirm:
        raise HTTPException(400, "Passwords do not match")
    user.password_hash = hash_password(body.new_password)
    user.token_version += 1  # revokes every other session; this one continues with the returned token
    db.commit()
    return {"token": create_token(user.id, user.token_version)}


@router.put("/me/claude-key")
def link_claude(body: ClaudeKey, user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.anthropic_key_encrypted = encrypt_api_key(body.api_key)
    db.commit()
    return user_out(user)


@router.delete("/me/claude-key")
def unlink_claude(user: User = Depends(current_user), db: Session = Depends(get_db)):
    user.anthropic_key_encrypted = None
    db.commit()
    return user_out(user)
