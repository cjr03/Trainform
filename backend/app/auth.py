import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Annotated

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, EmailStr, Field, StringConstraints
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User

router = APIRouter(tags=["auth"])

# ponytail: stateless tokens survive a password change until expiry; rotating TRAINFORM_SECRET logs
# everyone out. Sign a per-user value (e.g. a hash of password_hash) into the token if revocation is needed.
TOKEN_TTL = 7 * 24 * 3600
SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def _secret() -> bytes:
    secret = os.environ.get("TRAINFORM_SECRET")
    if not secret:
        raise RuntimeError("TRAINFORM_SECRET is not set")
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


def create_token(user_id: int) -> str:
    payload = f"{user_id}.{int(time.time()) + TOKEN_TTL}"
    return f"{payload}.{_sign(payload)}"


def verify_token(token: str) -> int | None:
    payload, _, sig = token.rpartition(".")
    if not payload or not hmac.compare_digest(sig.encode(), _sign(payload).encode()):
        return None
    try:
        user_id, expires = map(int, payload.split("."))
    except ValueError:
        return None
    return user_id if expires > time.time() else None


def _fernet() -> Fernet:
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(b"trainform-claude-key:" + _secret()).digest()))


def encrypt_api_key(api_key: str) -> str:
    return _fernet().encrypt(api_key.encode()).decode()


def decrypt_api_key(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: the User for a valid `Authorization: Bearer <token>` header, else 401."""
    scheme, _, token = (authorization or "").partition(" ")
    user_id = verify_token(token) if scheme.lower() == "bearer" else None
    user = db.get(User, user_id) if user_id is not None else None
    if user is None:
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


def user_out(user: User) -> dict:
    return {"id": user.id, "email": user.email, "username": user.username,
            "claude_linked": user.anthropic_key_encrypted is not None}


def _session(user: User) -> dict:
    return {"token": create_token(user.id), "user": user_out(user)}


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
    user = User(email=email, username=body.username, password_hash=hash_password(body.password),
                anthropic_key_encrypted=encrypt_api_key(body.anthropic_api_key) if body.anthropic_api_key else None)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:  # lost a race with a concurrent signup
        db.rollback()
        raise HTTPException(409, _conflict(db, body.username, email) or "Username already taken")
    session = _session(user)  # before commit: a missing TRAINFORM_SECRET must not leave an account behind
    db.commit()
    return session


@router.post("/auth/login")
def login(body: Login, db: Session = Depends(get_db)):
    user = db.query(User).filter_by(username=body.username).first()
    if not verify_password(body.password, user.password_hash if user else _DUMMY_HASH) or user is None:
        raise HTTPException(401, "Invalid username or password")
    return _session(user)


@router.get("/me")
def me(user: User = Depends(current_user)):
    return user_out(user)


@router.post("/me/password", status_code=204)
def change_password(body: PasswordChange, user: User = Depends(current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    if body.new_password != body.new_password_confirm:
        raise HTTPException(400, "Passwords do not match")
    user.password_hash = hash_password(body.new_password)
    db.commit()


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
