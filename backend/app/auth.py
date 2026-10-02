from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User

router = APIRouter(tags=["auth"])


def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: the User for a valid `Authorization: Bearer <token>` header, else 401."""
    raise NotImplementedError


def encrypt_api_key(api_key: str) -> str:
    raise NotImplementedError


def decrypt_api_key(ciphertext: str) -> str:
    raise NotImplementedError
