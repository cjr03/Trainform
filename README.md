# Trainform

Trainform trains an agent's system prompt *and* its weights against simulated users. You pick a local Hugging Face
model, write (or have Claude draft) a system prompt and a scoring rubric, and Trainform plays out persona-driven
scenarios. Each reply is scored, the prompt is rewritten after every scenario, and high-scoring exchanges LoRA
fine-tune the model at the end of every round. When it finishes you download the best round's prompt and LoRA adapter.

## The seven agents

| Agent | Runs on | Job |
| --- | --- | --- |
| Prompt creator | Claude | Agent description → system prompt (optional, Prompt step) |
| Rubric creator | Claude | System prompt → 4–8 criteria scored 1–10 (optional, Rubric step) |
| Persona generator | Claude | Prompt + criteria → N personas, each with a 1–10 pressure rating per criterion and an actor brief |
| Persona agent | Claude | Plays one persona and writes its message to the agent |
| Primary agent | Local model | The model being trained; replies under the current prompt |
| Scoring agent | Claude | Grades the reply against the training's rubric, 1–10 per criterion |
| Prompt generator | Claude | Current prompt + conversation + scores → improved prompt |

## The training loop

For each round (`backend/app/training.py`, `run_training`):

1. The persona generator creates `scenarios_per_round` personas for the current prompt. It derives their criteria
   by running the rubric creator on that prompt.
2. For each persona, the persona agent writes one message, the primary agent replies, and the scoring agent scores
   the exchange against your rubric. The result is stored as a scenario, and the prompt generator rewrites the prompt
   before the next scenario. Conversations are a single turn: one persona message and one reply.
3. Scenarios with a mean score of 7 or more become SFT examples. The model is fine-tuned on them through a LoRA
   adapter (r=8, all linear layers). Rounds without such examples skip the fine-tune.
4. The round is saved as a trained model: the current prompt, the zipped LoRA adapter and the round's mean score.
   The round with the highest mean becomes the training's best model.

The Train page shows progress and the mean score over the last n, n/2, n/4 and 1 scenarios. The Inspect page drills
down from training to round to scenario and shows the transcript, the scores and the prompt that was used.

## Architecture

- **Backend** (`backend/app`): FastAPI + SQLAlchemy on PostgreSQL. `auth.py` handles accounts, tokens and Claude
  keys. `agents.py` holds the six Claude agents. `llm.py` makes Claude calls (`claude-opus-5-5`).
  `local_model.py` handles Hugging Face Hub search, download and upload, plus generation and LoRA SFT with
  transformers + PEFT (on MPS, CUDA or CPU). `training.py` runs the loop and the train/inspect API. All routes are
  under `/api`.
- **Frontend** (`frontend/src`): React + Vite with four pages: Auth (sign up / sign in), Train (six-step workflow:
  Model, Prompt, Rubric, Scenarios, Training, Complete), Inspect and Account. In development, Vite proxies `/api` to
  `localhost:8000`.
- **Runs** are FastAPI background tasks in the API process. One worker thread per running training.

## Data model

| Table | Holds |
| --- | --- |
| `users` | email, username, scrypt password hash, encrypted Anthropic key; owns a list of trainings |
| `trainings` | `id` (training_id), base model, initial prompt, rubric, rounds × scenarios per round, status (`pending`/`running`/`completed`/`failed`), progress, error, `best_model_id` |
| `scenarios` | `training_id`, `round`, `scenario_number` (unique together), persona, prompt used, `input` (persona message), `output` (model reply), `scores` per criterion, mean `score` |
| `trained_models` | `id` (model_id), `training_id`, `round`, `prompt`, `weights` (zipped LoRA adapter), `mean_score` |

Tables are created on startup with `create_all`. There are no migrations, so a schema change on an existing database
must be applied by hand. Base models live on disk under `MODELS_DIR` (default `backend/data/models`), not in the
database.

## Run locally

```bash
# Postgres
brew install postgresql@17 && brew services start postgresql@17
createdb trainform && createdb trainform_test

# Backend (http://localhost:8000)
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TRAINFORM_SECRET="$(openssl rand -hex 32)"   # required; keep it stable (see below)
export ANTHROPIC_API_KEY=sk-ant-...                 # optional fallback for users without a linked key
.venv/bin/uvicorn app.main:app --reload

# Frontend (second terminal, http://localhost:5173)
cd frontend && npm install && npm run dev
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAINFORM_SECRET` | none (required) | Signs session tokens and derives the key that encrypts linked Claude keys |
| `DATABASE_URL` | `postgresql+psycopg://localhost/trainform` | Application database |
| `ANTHROPIC_API_KEY` | none | Claude key for users who have not linked their own |
| `MODELS_DIR` | `backend/data/models` | Where downloaded and uploaded base models are stored |
| `TRAINFORM_FAKE_LLM` | unset | `1` turns on demo mode (below; added by PR #9). Never set it in production |
| `TEST_DATABASE_URL` | `postgresql+psycopg://localhost/trainform_test` | Database the test suite uses (it never touches `DATABASE_URL`) |

### Claude linkage

Each user can link an Anthropic API key at sign-up or on the Account page. The key is stored Fernet-encrypted and
never returned by the API, which reports only whether a key is linked. Claude calls use the user's own key and fall
back to `ANTHROPIC_API_KEY`. With neither, Claude-backed actions return 400 and ask the user to link a key.

### Base models

On the Model step you can search the Hub for text-generation models and download one, or upload a zipped model
directory containing `config.json` and `*.safetensors`. Downloads fetch only configs, tokenizer files and
safetensors weights. Gated repos need their license accepted on huggingface.co. Uploads reject pickle weights
(`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`), symlinks and paths that escape the archive. Models are capped at 20 GiB.

### Demo / fake mode

`TRAINFORM_FAKE_LLM=1` runs Trainform without Claude or a real model, for demos and end-to-end tests. It is added by
PR #9 and does nothing until that PR is merged. No API key is needed, but a model must still be listed under
`MODELS_DIR`. For a demo, a directory containing only a `config.json` of `{}` is enough.

**Never set `TRAINFORM_FAKE_LLM` in production.** Any signed-in user can then run trainings without a Claude key, and
all scores, prompts and adapters are canned.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q          # uses trainform_test; override with TEST_DATABASE_URL
cd frontend && npx vitest --run && npm run lint && npm run build
createdb trainform_e2e 2>/dev/null; cd frontend && npx playwright install chromium && npm run e2e
```

- **Unit** (`backend/tests/unit`, `frontend/src/**/*.test.{js,jsx}`): auth, agents, Claude call handling, the local
  model (a tiny model built in the test), the training loop, the API client and each page. They never call the
  Claude API or the Hugging Face Hub.
- **Integration** (`backend/tests/integration`): the HTTP API against a real PostgreSQL test database. It is
  dropped and recreated each session.
- **E2E** (`frontend/e2e`): Playwright drives Chromium through sign-up, account, a full training, Inspect and a
  390px-wide phone layout. `playwright.config.js` starts the backend in fake mode on port 8000 (database
  `trainform_e2e`, or `E2E_DATABASE_URL`; models in a fresh temp dir; Python from `E2E_PYTHON` or
  `backend/.venv`) and serves the production build with `vite preview`. Ports 8000 and 4173 must be free.

## CI

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. The backend job runs pytest on Python
3.14 with CPU torch against a Postgres 17 service. The frontend job runs `npm ci`, lint, Vitest and the production
build. The e2e job runs Playwright against its own Postgres 17 service and uploads the HTML report when it fails.

## Operations and security notes

- **Passwords** are hashed with scrypt (n=2¹⁴, r=8, p=1, random salt). Login does a dummy hash for unknown usernames
  so response timing does not reveal which accounts exist.
- **Viewing the password:** passwords are stored hashed, so the Account page shows a masked value. It can't show the
  real one. Users can change it there.
- **Sessions** are stateless HMAC-signed bearer tokens that last 7 days. The frontend keeps them in `localStorage`.
  Changing a password does **not** revoke existing tokens. They stay valid until they expire.
- **Rotating `TRAINFORM_SECRET`** invalidates every token, which logs everyone out. It also makes every linked Claude
  key unreadable, because the encryption key comes from the secret, so users must link their keys again. Until they do, their Claude actions fail (HTTP 500 today; a 400 with a
  re-link message once PR #9 merges), and `ANTHROPIC_API_KEY` is not used as a fallback. Re-linking on the Account
  page fixes it. Keep the secret stable and back it up.
- **Weights:** only safetensors weights are accepted from the Hub and from uploads, and models load with
  `use_safetensors=True`, because pickle weights can execute code on load.
- **One active training per user:** starting a second training while one is pending or running returns 409.
- **Restarts:** trainings run inside the API process. On startup, any training still `pending` or `running` is marked
  `failed` with "Interrupted by a server restart". Shutdown waits for running trainings to finish.
- **Fake mode:** never set `TRAINFORM_FAKE_LLM` in production. Any signed-in user can then run trainings without a
  Claude key, and all scores, prompts and adapters are canned.
- **Isolation:** every training, scenario and download endpoint checks ownership and returns 404 for other users'
  data. Hub and model endpoints require sign-in.
