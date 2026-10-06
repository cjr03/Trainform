"""Trainformers' social layer: follows, published agents (posts), votes, copying a post, and profile pictures.

Trainformers are known by username only: nothing here returns an email, password hash, key or token."""

import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, StringConstraints
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import svg
from app.agent_profiles import GenerateIn, _name_taken, _out, _rows, available_models, owned
from app.agents import _tag, draw_image
from app.auth import avatar_url, current_user
from app.db import get_db
from app.llm import LLMError, chosen_llm, get_llm
from app.models import Agent, Follow, Post, User, Vote

router = APIRouter(tags=["social"])
MAX_AVATAR_BYTES = 512 * 1024
RASTERS = {b"\x89PNG\r\n\x1a\n": "image/png", b"\xff\xd8\xff": "image/jpeg"}  # magic bytes; WebP: RIFF....WEBP


class PostIn(BaseModel):
    agent_id: int
    caption: Annotated[str, StringConstraints(strip_whitespace=True, max_length=500)] | None = None


class VoteIn(BaseModel):
    value: Literal[-1, 0, 1]


def _user(db: Session, username: str) -> User:
    u = db.scalar(select(User).where(User.username == username))
    if u is None:
        raise HTTPException(404, "Trainformer not found")
    return u


def _post(db: Session, post_id: int) -> Post:
    p = db.get(Post, post_id)
    if p is None:
        raise HTTPException(404, "Post not found")
    return p


def _posts(db: Session, me: User, *where, limit: int | None = None) -> list[dict]:
    """Posts matching `where`, newest first, as JSON (never their weights)."""
    score = select(func.coalesce(func.sum(Vote.value), 0)).where(Vote.post_id == Post.id).scalar_subquery()
    mine = select(Vote.value).where(Vote.post_id == Post.id, Vote.user_id == me.id).scalar_subquery()
    rows = db.execute(select(Post, User.username, User.avatar_type, score, mine).join(User, User.id == Post.user_id)
                      .where(*where).order_by(Post.id.desc()).limit(limit)).all()
    return [{"id": p.id, "author": username, "author_avatar_url": avatar_url(username, avatar_type),
             "agent_id": p.agent_id if p.user_id == me.id else None, "name": p.name, "base_model": p.base_model,
             "prompt": p.prompt, "overall": p.overall, "image": p.image, "caption": p.caption, "score": total,
             "my_vote": my_vote or 0, "created_at": p.created_at}
            for p, username, avatar_type, total, my_vote in rows]


def _followers(user_id):
    return select(func.count()).where(Follow.followee_id == user_id).scalar_subquery()


def _following(me: User, user_id):
    return select(Follow.follower_id).where(Follow.follower_id == me.id, Follow.followee_id == user_id).exists()


@router.get("/social/users")
def search_users(q: str = "", me: User = Depends(current_user), db: Session = Depends(get_db)):
    pattern = "%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    posts = select(func.count()).where(Post.user_id == User.id).scalar_subquery()
    rows = db.execute(select(User.username, User.avatar_type, posts, _followers(User.id), _following(me, User.id))
                      .where(User.username.ilike(pattern, escape="\\"), User.id != me.id)
                      .order_by(User.username).limit(20)).all()
    return [{"username": u, "avatar_url": avatar_url(u, t), "agents": n, "followers": f, "following": following}
            for u, t, n, f, following in rows]


@router.get("/social/users/{username}")
def profile(username: str, me: User = Depends(current_user), db: Session = Depends(get_db)):
    u = _user(db, username)
    followers, following = db.execute(select(_followers(u.id), _following(me, u.id))).one()
    return {"username": u.username, "avatar_url": avatar_url(u.username, u.avatar_type), "followers": followers,
            "following_count": db.scalar(select(func.count()).where(Follow.follower_id == u.id)),
            "following": following, "posts": _posts(db, me, Post.user_id == u.id)}


@router.post("/social/users/{username}/follow", status_code=204)
def follow(username: str, me: User = Depends(current_user), db: Session = Depends(get_db)):
    u = _user(db, username)
    if u.id == me.id:
        raise HTTPException(400, "You can't follow yourself.")
    db.execute(insert(Follow).values(follower_id=me.id, followee_id=u.id).on_conflict_do_nothing())
    db.commit()


@router.delete("/social/users/{username}/follow", status_code=204)
def unfollow(username: str, me: User = Depends(current_user), db: Session = Depends(get_db)):
    u = _user(db, username)
    db.execute(delete(Follow).where(Follow.follower_id == me.id, Follow.followee_id == u.id))
    db.commit()


@router.post("/social/posts", status_code=201)
def publish(body: PostIn, me: User = Depends(current_user), db: Session = Depends(get_db)):
    a = owned(db, body.agent_id, me)
    _, best, _ = _rows(db, me, a.id)[0]
    p = Post(user_id=me.id, agent_id=a.id, caption=body.caption or None, name=a.name, base_model=a.base_model,
             prompt=a.prompt, weights=a.weights, image=a.image, overall=None if best is None else round(10 * best))
    db.add(p)
    db.commit()
    return _posts(db, me, Post.id == p.id)[0]


@router.get("/social/feed")
def feed(before: int | None = None, limit: int = Query(20, ge=1, le=50), me: User = Depends(current_user),
         db: Session = Depends(get_db)):
    followed = select(Follow.followee_id).where(Follow.follower_id == me.id)
    return _posts(db, me, Post.user_id.in_(followed), *([Post.id < before] if before is not None else []), limit=limit)


@router.delete("/social/posts/{post_id}", status_code=204)
def delete_post(post_id: int, me: User = Depends(current_user), db: Session = Depends(get_db)):
    if not db.execute(delete(Post).where(Post.id == post_id, Post.user_id == me.id)).rowcount:
        raise HTTPException(404, "Post not found")
    db.commit()


@router.put("/social/posts/{post_id}/vote")
def vote(post_id: int, body: VoteIn, me: User = Depends(current_user), db: Session = Depends(get_db)):
    if _post(db, post_id).user_id == me.id:
        raise HTTPException(400, "You can't vote on your own agent.")
    if body.value:
        db.execute(insert(Vote).values(user_id=me.id, post_id=post_id, value=body.value)
                   .on_conflict_do_update(index_elements=["user_id", "post_id"], set_={"value": body.value}))
    else:
        db.execute(delete(Vote).where(Vote.user_id == me.id, Vote.post_id == post_id))
    db.commit()
    [p] = _posts(db, me, Post.id == post_id)
    return {"score": p["score"], "my_vote": p["my_vote"]}


@router.post("/social/posts/{post_id}/copy", status_code=201)
def copy_post(post_id: int, me: User = Depends(current_user), db: Session = Depends(get_db)):
    """"Find an agent": the post's agent, as a new agent of yours named apart from your others ("Name (2)", ...)."""
    p = _post(db, post_id)
    taken = set(db.scalars(select(Agent.name).where(Agent.user_id == me.id)))
    name, n = p.name, 1
    while name in taken:
        n += 1
        name = f"{p.name[:100 - len(f' ({n})')]} ({n})"
    a = Agent(user_id=me.id, name=name, base_model=p.base_model, prompt=p.prompt, weights=p.weights, image=p.image)
    db.add(a)
    try:
        db.commit()
    except IntegrityError:  # a concurrent copy took the name
        raise _name_taken(db) from None
    return _out(available_models(me), a, None, 0)


def _set_avatar(db: Session, me: User, data: bytes | None, kind: str | None) -> dict:
    me.avatar, me.avatar_type = data, kind
    db.commit()
    # A new ?v= makes the pages showing it fetch the new picture now (get_avatar is revalidated on every use too).
    return {"avatar_url": data and f"{avatar_url(me.username, kind)}?v={time.time_ns()}"}


@router.post("/me/avatar/generate")
def generate_avatar(body: GenerateIn, me: User = Depends(current_user), db: Session = Depends(get_db),
                    llm=Depends(get_llm)):
    try:
        image = draw_image(chosen_llm(body.helper, llm), _tag("username", me.username), me.username)
    except LLMError as e:
        raise HTTPException(502, str(e)) from e
    return _set_avatar(db, me, image.encode(), "image/svg+xml")


@router.put("/me/avatar")
def put_avatar(file: UploadFile = File(), me: User = Depends(current_user), db: Session = Depends(get_db)):
    """A PNG, JPEG or WebP (by its magic bytes, not the client's say) up to 512 KB, or an SVG that passes the sanitiser."""
    data = file.file.read(MAX_AVATAR_BYTES + 1)
    if len(data) > MAX_AVATAR_BYTES:
        raise HTTPException(413, "Pictures can be up to 512 KB.")
    kind = next((t for magic, t in RASTERS.items() if data.startswith(magic)), None)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "image/webp"
    if kind is None:
        try:
            data, kind = svg.sanitise(data.decode("utf-8-sig")).encode(), "image/svg+xml"
        except UnicodeDecodeError:
            raise HTTPException(400, "Upload a PNG, JPEG, WebP or SVG picture.") from None
        except ValueError as e:
            raise HTTPException(400, f"Upload a PNG, JPEG, WebP or SVG picture. {e}") from None
    return _set_avatar(db, me, data, kind)


@router.delete("/me/avatar", status_code=204)
def delete_avatar(me: User = Depends(current_user), db: Session = Depends(get_db)):
    _set_avatar(db, me, None, None)


@router.get("/social/users/{username}/avatar")
def get_avatar(username: str, _me: User = Depends(current_user), db: Session = Depends(get_db)):
    u = _user(db, username)
    if not u.avatar_type:
        raise HTTPException(404, "No picture")
    return Response(u.avatar, media_type=u.avatar_type, headers={
        "X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-cache",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})
