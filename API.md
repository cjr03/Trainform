# Trainform API

Everything the Trainform app does goes through this HTTP API: the React frontend is only a client of it
(`frontend/src/api.js`). Anything you can do in the app you can do with `curl` or a script.

`backend/tests/unit/test_api_doc.py` fails if a route is added, removed or renamed without updating this file.

## Basics

**Base URL.** The macOS app serves the API on `http://127.0.0.1:47821/api` while it is open. In development it is
`http://localhost:8000/api` (the Vite dev server proxies `/api` there). The desktop server only answers requests whose
`Host` is `127.0.0.1` or `localhost`.

**Interactive docs.** FastAPI serves Swagger UI at `/docs`, ReDoc at `/redoc` and the OpenAPI schema at
`/openapi.json` (for example `http://127.0.0.1:47821/docs`). The schema gives exact field types and limits; this file
explains what each endpoint does.

**Auth.** Sign up or log in to get a token, then send it on every other request:

```
Authorization: Bearer <token>
```

Tokens last 7 days; logging out or changing the password revokes all of a user's tokens sooner. A missing, invalid,
expired or revoked token gets `401 {"detail": "Not authenticated"}` with a `WWW-Authenticate: Bearer` header.
Everything is scoped to the token's user: another user's agent or training answers `404` as if it didn't exist.

**Bodies and errors.** Request and response bodies are JSON unless noted. Errors are
`{"detail": "<message>"}`; a body that fails validation gets `422` with `detail` as a list of
`{"loc", "msg", "type"}`. Common statuses:

| Status | Meaning |
| --- | --- |
| 400 | The request can't be done as asked (message says why) |
| 401 | Not signed in, or wrong username/password at login |
| 404 | Not found, or not yours |
| 409 | Conflicts with the current state (name taken, training in progress, ...) |
| 413 / 411 | Upload too large / upload without `Content-Length` |
| 422 | Body or query failed validation |
| 429 | Rate limited, or the server is busy with other trainings or downloads |
| 502 | Claude or the Hugging Face Hub failed |

**Helpers.** Several endpoints use LLM *helper agents* (prompt writer, rubric writer, image drawer, personas, grader).
They take an optional `helper`: `"claude"` runs them on Claude (the user's linked API key; `400` if none is linked),
`"local"` runs them on a local model, and omitting it means Claude when linked, else local.

### Quick start

```sh
API=http://127.0.0.1:47821/api
TOKEN=$(curl -s $API/auth/login -H 'Content-Type: application/json' \
  -d '{"username": "ada", "password": "correct horse"}' | jq -r .token)
curl -s $API/agents -H "Authorization: Bearer $TOKEN"
```

---

## Health

### GET /api/health

Returns `{"status": "ok"}` when the server is up. No auth.

---

## Account and auth

A **user** object is `{"id", "email", "username", "claude_linked", "claude_available", "avatar_url"}`.
`claude_linked` means the user linked their own Claude key; `claude_available` means Claude can be the helper (linked,
an operator-shared key, or fake mode). `avatar_url` is an API path (see
[GET /api/social/users/{username}/avatar](#get-apisocialusersusernameavatar)) or `null`.

### POST /api/auth/signup

Create an account and sign in. No auth.

Body: `{"email", "username", "password", "password_confirm", "anthropic_api_key"?}`. `username` is 3–32 of
`A-Z a-z 0-9 _ . -`; `password` is 8–128 characters. `anthropic_api_key` optionally links Claude now.

`201` → `{"token", "user"}`. `400` passwords don't match; `409` username taken or email registered.

### POST /api/auth/login

Body: `{"username", "password"}`. No auth.

`200` → `{"token", "user"}`. `401` wrong username or password; `429` after 5 failures for that username from that
address in 15 minutes.

### POST /api/auth/logout

Revokes every token the user has (all devices). `204`.

### GET /api/me

The signed-in **user**.

### POST /api/me/password

Body: `{"current_password", "new_password", "new_password_confirm"}`.

`200` → `{"token"}`, a new token for this session; every other token is revoked. `400` current password wrong or new
passwords don't match.

### PUT /api/me/claude-key

Link a Claude (Anthropic) API key, stored encrypted. Body: `{"api_key"}`. `200` → the **user**.

### DELETE /api/me/claude-key

Unlink the Claude key. `200` → the **user**.

### POST /api/me/avatar/generate

Have a helper draw an SVG profile picture from the username. Body: `{"helper"?}`.

`200` → `{"avatar_url"}` (with a `?v=` cache buster). `502` the helper failed.

### PUT /api/me/avatar

Upload a profile picture as multipart form field `file`: PNG, JPEG or WebP (detected by content) up to 512 KB, or an
SVG, which is sanitised. Needs a `Content-Length` header.

`200` → `{"avatar_url"}`. `400` not a supported or safe image; `413` too large.

### DELETE /api/me/avatar

Remove the profile picture. `204`.

---

## Local models

Agents run on a **base model**: `"claude"`, or a local Hugging Face model on the server's disk. Each user can add up to
5 models (downloads plus uploads), each up to 20 GiB.

### GET /api/hub/models

Search the Hugging Face Hub's text-generation models. No auth. Query: `search` (default all), `sort` (`downloads`
default, or `lastModified`).

`200` → up to 20 `{"id", "downloads", "likes", "last_modified"}`. `502` the Hub failed.

### POST /api/hub/download

Download a Hub model (safetensors weights only). Body: `{"repo_id": "owner/name"}`. Blocks until done.

`201` → `{"name", "source": "hub"}`; `name` is the repo id with `/` as `--`. `400` invalid id, no safetensors, too
large, gated or not found, or over your model quota; `409` already downloaded; `429` too many downloads at once; `502`
the Hub failed.

### POST /api/hub/upload

Upload a model as a zip of a model directory (`config.json` plus `*.safetensors`; no pickle weights, symlinks or
unsafe paths). Multipart form fields `file` (the zip) and `name`; needs a `Content-Length` header. Only you can see
your uploads.

`201` → `{"name", "source": "upload"}`. `400` invalid name or zip, or over quota; `409` name taken; `413` too large.

### GET /api/models

The models your agents can use: `[{"name", "source"}]`, where `source` is `"hub"`, `"upload"` or `null` (provided by
the server's operator).

### DELETE /api/models/{name}

Delete a model you downloaded or uploaded. `204`. `404` not found or not yours. (The app has no button for this.)

---

## Agents

An **agent** is a named system prompt on a base model, plus LoRA weights once trained on a local model. The agent
object is `{"id", "name", "base_model", "prompt", "image", "model_available", "overall", "trainings", "created_at"}`:
`image` is an SVG string or `null`; `model_available` is false when its base model isn't on this server (e.g. copied
from someone else's post); `overall` is its 0–100 best score (`null`: unranked); `trainings` is a count, except in
[GET /api/agents/{agent_id}](#get-apiagentsagent_id).

### POST /api/agents/prompt

Have a helper write a system prompt from a description. Body: `{"description", "model"?, "helper"?}`; `model` is the
local model to use when the helper is local.

`200` → `{"prompt"}`. `502` the helper failed.

### POST /api/agents/rubric

Have a helper write a grading rubric for a prompt. Body: `{"agent_prompt", "model"?, "helper"?}`.

`200` → `{"criteria": ["...", ...]}` (4–8 criteria, each scored 1–10 during training). `502` the helper failed.

### GET /api/agents

Your agents, by name.

### POST /api/agents

Body: `{"name", "base_model", "prompt"}`. `base_model` is `"claude"` (needs Claude linked) or a name from
[GET /api/models](#get-apimodels).

`201` → the agent. `400` unknown base model or Claude not linked; `409` you already have an agent with that name.

### GET /api/agents/{agent_id}

The agent, with `trainings` as its trainings (not battles), newest first:
`[{"id", "status", "created_at", "rounds", "scenarios_per_round", "best_score", "baseline_score"}]`.

### PATCH /api/agents/{agent_id}

Rename it or change its prompt. Body: `{"name"?, "prompt"?}`.

`200` → the agent. `409` name taken, or a training of it is in progress.

### DELETE /api/agents/{agent_id}

Delete the agent and its trainings. `204`. `409` a training of it is in progress (pause it first).

### PUT /api/agents/{agent_id}/image

Set its picture. Body: `{"svg": "<svg ...>"}` (sanitised), or `{"svg": null}` to clear it.

`200` → the agent. `400` the SVG isn't safe or valid.

### POST /api/agents/{agent_id}/image/generate

Have a helper draw its picture from its name and prompt. Body: `{"helper"?}`.

`200` → the agent. `502` the helper failed.

---

## Trainings and battles

A **training** runs `rounds` rounds of `scenarios` simulated users against an agent: personas message it, a grader
scores each reply on the rubric, the prompt is revised, and (for local models) good replies fine-tune LoRA weights.
The agent takes on the best round's prompt and weights. A **battle** pits the agent (the *champion*) against a
*challenger* on fresh users and keeps the winner.

Statuses: `pending` → `running` → `completed`, or `pausing` → `paused`, or `failed`. One training per user can be
`pending`, `running` or `pausing` at a time, and the server runs one at a time overall (configurable with
`TRAINFORM_MAX_TRAININGS`).

The **training detail** object is `{"id", "status", "error", "base_model", "initial_prompt", "rubric", "rounds",
"scenarios_per_round", "current_round", "scenarios_done", "total_scenarios", "latest_prompt", "score_windows",
"best_model_id", "best_score", "best_prompt", "baseline_score", "created_at", "helper", "agent_id", "agent_name",
"battle"}`. Scores are 1–10. `baseline_score` is the untrained agent's (round 0); `score_windows` is the mean score
over the last n, n/2, n/4 and 1 scenarios: `[{"label", "runs", "score"}]`.

`battle` is `null` for a training. For a battle it is `{"rounds", "opponent"?: {"post_id", "name", "author"}}` and,
once decided, also `"winner": "champion" | "challenger"`, `"grades": [champion, challenger]` (mean grades),
`"round_grades": [[champion, challenger], ...]` and `"bouts": [{"message", "champion", "challenger", "grades"}]` (each
user's message, both replies and their grades). Lists of trainings leave out `bouts`.

### POST /api/trainings

Start a training or battle in the background. Body:

| Field | |
| --- | --- |
| `agent_id` | The agent to train |
| `rubric` | 1–30 criteria (each up to 500 characters) |
| `scenarios` | Simulated users per round, 1–100 |
| `rounds` | Rounds, 1–50 |
| `helper` | Optional; a Claude agent needs `"claude"` or Claude linked |
| `battle` | `true`: train a challenger for `rounds` rounds, then battle it against the agent |
| `opponent_post_id` | Battle the agent of this [post](#social) instead (no training; same base model only) |
| `battle_rounds` | Head-to-head rounds in a battle, 1–7 (default 1), each on `scenarios` new users |

`201` → the training detail. Poll [GET /api/trainings/{training_id}](#get-apitrainingstraining_id) for progress.
`400` unknown base model, a Claude agent without Claude, or the opponent is on another base model; `404` agent or
post not found; `409` you already have a training in progress; `429` the server is busy with other trainings.

### GET /api/trainings

Your trainings and battles, newest first. Query: `agent_id` (optional) to filter by agent.

`200` → `[{"id", "base_model", "status", "rounds", "scenarios_per_round", "current_round", "scenarios_done",
"created_at", "best_score", "helper", "agent_id", "agent_name", "battle"}]`.

### GET /api/trainings/{training_id}

The training detail.

### POST /api/trainings/{training_id}/pause

Ask a `pending` or `running` training to stop at its next safe point; it becomes `pausing`, then `paused` with its
progress saved. `200` → the training detail. `409` it isn't pending or running.

### POST /api/trainings/{training_id}/resume

Continue a `paused` or `failed` training from its saved progress (the app's Retry). `200` → the training detail.
`409` it isn't paused or failed, or another training is in progress; `429` the server is busy.

### DELETE /api/trainings/{training_id}

Delete it with its scenarios and models. `204`. `409` it is in progress (pause it first).

### GET /api/trainings/{training_id}/rounds

Per-round results: `[{"round", "mean_score", "scenarios", "model_id", "eval_score"}]`. `mean_score` is over the
round's own users, who press harder on weak spots each round; `eval_score` is the round's model on the same held-out
users as every other round, so it is the one to compare. Round 0 is the untrained baseline.

### GET /api/trainings/{training_id}/rounds/{round}/scenarios

The round's conversations: `[{"id", "scenario_number", "persona", "prompt", "input", "output", "scores", "score"}]`;
`persona` describes the simulated user, `prompt` is the system prompt the agent ran with, `input` the user's message,
`output` the agent's reply, `scores` the per-criterion grades and `score` their mean.

### GET /api/trainings/{training_id}/download/prompt

The best model's system prompt as a `text/plain` attachment. `404` no trained model yet.

### GET /api/trainings/{training_id}/download/weights

The best model's LoRA adapter as an `application/zip` attachment (load it with PEFT on the base model). `404` no
trained model yet.

---

## Social

Users ("Trainformers") publish agents as **posts**, follow each other and vote. A **post** is
`{"id", "author", "author_avatar_url", "agent_id", "name", "base_model", "prompt", "overall", "image", "caption",
"score", "my_vote", "created_at"}`: a snapshot of the agent when published, so later edits don't change it.
`agent_id` is only shown to the author; `score` is the sum of votes and `my_vote` is yours (-1, 0 or 1).

### GET /api/social/users

Search other users by username. Query: `q` (substring, case-insensitive).

`200` → up to 20 `[{"username", "avatar_url", "agents", "followers", "following"}]`; `agents` is their post count and
`following` whether you follow them.

### GET /api/social/users/{username}

Their profile: `{"username", "avatar_url", "followers", "following_count", "following", "posts"}`. `404` no such user.

### GET /api/social/users/{username}/avatar

Their profile picture, as the image itself (PNG, JPEG, WebP or SVG). `404` no such user or no picture.

### POST /api/social/users/{username}/follow

Follow them. `204` (also when already following). `400` that's you; `404` no such user.

### DELETE /api/social/users/{username}/follow

Unfollow them. `204`. `404` no such user.

### POST /api/social/posts

Publish one of your agents. Body: `{"agent_id", "caption"?}` (caption up to 500 characters).

`201` → the post. `404` agent not found.

### GET /api/social/feed

Posts by people you follow, newest first. Query: `limit` (1–50, default 20) and `before` (a post id: only older
posts, for the next page). `200` → `[post]`.

### DELETE /api/social/posts/{post_id}

Delete your post. `204`. `404` not found or not yours.

### PUT /api/social/posts/{post_id}/vote

Body: `{"value": 1 | -1 | 0}` (`0` removes your vote). `200` → `{"score", "my_vote"}`. `400` it's your own post;
`404` post not found.

### POST /api/social/posts/{post_id}/copy

Copy the post's agent (prompt, weights and picture) into a new agent of yours, renamed "Name (2)" and so on if you
already have the name. `201` → the agent. `404` post not found.
