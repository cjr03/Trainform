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

1. The persona generator creates `scenarios_per_round` personas for the current prompt, rating how hard each one
   pressures every criterion of your rubric (the one the rubric creator drafted and you edited). It asks Claude for
   at most 10 personas per call, sending those calls at once.
2. The persona agent writes each persona's message, up to 8 at a time, since a message depends only on its persona.
   Then, for each persona in turn, the primary agent replies, and the scoring agent scores the exchange against your
   rubric. The result is stored as a scenario, and the prompt generator rewrites the prompt before the next scenario.
   Conversations are a single turn: one persona message and one reply.
3. Scenarios with a mean score of 7 or more become SFT examples. The model is fine-tuned on them through a LoRA
   adapter (r=8, all linear layers). Rounds without such examples skip the fine-tune.
4. The round is saved as a trained model holding what earned its score: the LoRA adapter the round's scenarios ran
   with (before step 3), the prompt of its highest-scoring scenario and the round's mean score. The round with the
   highest mean becomes the training's best model, which is what the downloads return.

The Train page shows progress and the mean score over all scenarios so far (Total runs), the last 50%, the last 25% and
the last one. It keeps the setup and the training on screen for the tab, so switching pages or reloading loses neither;
signing out clears it. The Inspect page drills down from training to round to scenario and shows the transcript, the scores and the prompt that was used.

## Architecture

- **Backend** (`backend/app`): FastAPI + SQLAlchemy on PostgreSQL. `auth.py` handles accounts, tokens and Claude
  keys. `agents.py` holds the six Claude agents. `llm.py` makes Claude calls (`claude-opus-5-5`, medium effort;
  low for the persona agent's messages). Claude's structured outputs key every per-criterion rating and score by
  criterion (`c1`…`cN`, all required), so a reply always has exactly one per criterion. The SDK retries rate limits
  and overloads up to 5 times, and an agent retries a failed call or garbled reply once before the training fails.
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
| `users` | email, username, scrypt password hash, `token_version` (bumped to revoke sessions), encrypted Anthropic key; owns a list of trainings |
| `trainings` | `id` (training_id), base model, initial prompt, rubric, rounds × scenarios per round, status (`pending`/`running`/`completed`/`failed`), progress, error, `best_model_id` |
| `scenarios` | `training_id` (the agent_id: one training trains one agent), `round`, `scenario_number` (unique together), persona, prompt used, `input` (persona message), `output` (model reply), `scores` per criterion, mean `score` |
| `trained_models` | `id` (model_id), `training_id`, `round`, `prompt`, `weights` (zipped LoRA adapter), `mean_score` |

Tables are created on startup with `create_all`. There are no migrations: startup adds the two later additions
(`users.token_version` and the one-active-training index) to an existing database, and any other schema change must
be applied by hand. Base models live on disk under `MODELS_DIR` (default `backend/data/models`), not in the
database.

## Run locally

```bash
# Postgres
brew install postgresql@17 && brew services start postgresql@17
createdb trainform && createdb trainform_test

# Backend (http://localhost:8000)
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TRAINFORM_SECRET="$(openssl rand -hex 32)"   # required, at least 32 bytes; keep it stable (see below)
export ANTHROPIC_API_KEY=sk-ant-...                 # optional, for the users in TRAINFORM_SHARED_KEY_USERS
.venv/bin/uvicorn app.main:app --reload

# Frontend (second terminal, http://localhost:5173)
cd frontend && npm install && npm run dev
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRAINFORM_SECRET` | none (required) | Signs session tokens and derives the key that encrypts linked Claude keys. At least 32 bytes, or startup fails |
| `DATABASE_URL` | `postgresql+psycopg://localhost/trainform` | Application database |
| `ANTHROPIC_API_KEY` | none | Claude key for allow-listed users who have not linked their own |
| `TRAINFORM_SHARED_KEY_USERS` | none | Comma-separated usernames allowed to use `ANTHROPIC_API_KEY` |
| `TRAINFORM_MAX_TRAININGS` | `2` | Trainings that may run at once across all users; more return 429 |
| `MODELS_DIR` | `backend/data/models` | Where downloaded and uploaded base models are stored |
| `TRAINFORM_FAKE_LLM` | unset | `1` turns on demo mode (below). Never set it in production |
| `TEST_DATABASE_URL` | `postgresql+psycopg://localhost/trainform_test` | Database the test suite uses (it never touches `DATABASE_URL`) |

### Claude linkage

Each user can link an Anthropic API key at sign-up or on the Account page. The key is stored Fernet-encrypted and
never returned by the API, which reports only whether a key is linked. Claude calls use the user's own key. Only
users listed in `TRAINFORM_SHARED_KEY_USERS` fall back to `ANTHROPIC_API_KEY`, so open sign-up can't spend the
operator's credits. A linked key that can't be decrypted returns 400, asking the user to re-link it (no fallback).

With no key, the helper agents run on the local model chosen on the Model step (the frontend sends it as `model` to
`/api/agents/prompt` and `/api/agents/rubric`; a training uses its base model). They run on the base weights with the
LoRA adapter switched off, so fine-tuning never changes the agents that grade and coach the model. During a training
they share the already-loaded model, one call at a time; the agent routes keep the last-used model loaded. Small
models can't write JSON, so on this path:

- prompts are plain text, and rubrics are one criterion per line (repeats dropped, at most 8);
- the scorer rates one criterion per call, with generation limited to digit tokens;
- persona ratings follow a fixed 1-10 spread across personas and criteria, and the model writes only each brief.

Expect much weaker results than with Claude. Qwen2.5-0.5B-Instruct produces usable prompts and rubrics. A 135M model
writes fluent text but ignores the instructions. Without a key or a chosen model, the agent routes return 400.

### Base models

On the Model step you can search the Hub for text-generation models and download one, or upload a zipped model
directory containing `config.json` and `*.safetensors`. Downloads fetch only configs, tokenizer files and
safetensors weights. Gated repos need their license accepted on huggingface.co. Uploads reject pickle weights
(`.bin`, `.pt`, `.pth`, `.pkl`, `.ckpt`), symlinks and paths that escape the archive. Models are capped at 20 GiB,
and an upload request must declare a `Content-Length` within that cap (else 411/413).

Downloaded Hub models are shared by all users. Uploads are private: only their owner sees and can train them, and
upload names can't contain `--`, which Hub models use (`owner--name`). Each user can add at most 5 models (uploads
plus downloads) and remove their own with `DELETE /api/models/{name}`. At most two Hub downloads run at once (429).

### Demo / fake mode

`TRAINFORM_FAKE_LLM=1` (exactly `1`) runs Trainform without Claude or real model weights, for demos and the E2E
suite. `backend/app/fake.py` supplies the stand-ins: every Claude agent returns canned, deterministic answers (four
fixed criteria, three personas used in turn), even for users with a linked key, and trainings use a fake model that
loads no weights and saves a placeholder LoRA adapter zip. No API key is needed, but the base model must still be listed
under `MODELS_DIR`: any directory with a `config.json` counts, so `{}` is enough.

```bash
cd backend
mkdir -p data/models/demo && echo '{}' > data/models/demo/config.json
TRAINFORM_FAKE_LLM=1 TRAINFORM_SECRET="$(openssl rand -hex 32)" .venv/bin/uvicorn app.main:app --reload
```

Hub search and download still call the real Hugging Face Hub in fake mode.

**Never set `TRAINFORM_FAKE_LLM` in production.** Any signed-in user can then run trainings without a Claude key, and
all scores, prompts and adapters are canned.

## macOS app

```bash
macos/build.sh        # -> dist/Trainform-0.1.0-arm64.dmg (Apple silicon, macOS 13+), about 2 minutes, 290 MB
macos/smoke-test.sh   # starts the built bundle's server headlessly and checks the API, UI and a clean shutdown
```

Open the DMG and drag Trainform into Applications. The app needs nothing installed: it bundles Python with the
backend's dependencies, PostgreSQL 17 and the built frontend. The build needs Xcode's command line tools (`swiftc`)
and Node; Python and PostgreSQL are pinned, checksummed downloads cached in `macos/.cache`.

- `macos/Trainform.swift` is a small native window (WKWebView). It starts `python -m app.desktop`
  (`backend/app/desktop.py`), waits for `/api/health`, and shows the UI from `http://127.0.0.1:47821`. Quitting stops
  the server and Postgres. If the app is force-quit, the server notices its stdin closing and shuts down on its own.
- Data lives in `~/Library/Application Support/Trainform`: the Postgres cluster (socket only, no TCP port), downloaded
  models and the generated `TRAINFORM_SECRET`. Delete that folder to reset the app. To keep it elsewhere, launch with
  `open --env TRAINFORM_DATA_DIR=<dir> -a Trainform`; macOS caps the Postgres socket path at 103 bytes, so keep it
  short. The server log is `~/Library/Logs/Trainform.log`.
- The app is signed ad hoc, which is fine on the Mac that built it. Other Macs block it as from an unidentified
  developer: right-click > Open, or System Settings > Privacy & Security > Open Anyway. Distributing it properly needs
  a Developer ID signature and notarization.

## Tests

```bash
cd backend && .venv/bin/python -m pytest -q --cov=app --cov-branch --cov-fail-under=95   # uses trainform_test
cd frontend && npm run lint && npx vitest --run --coverage && npm run build                # 95% lines, vite.config.js
createdb trainform_e2e 2>/dev/null; cd frontend && npx playwright install chromium && npm run e2e
```

The backend suite uses `TEST_DATABASE_URL` (default `postgresql+psycopg://localhost/trainform_test`). The E2E run
reads these variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `E2E_DATABASE_URL` | `postgresql+psycopg://localhost/trainform_e2e` | Database for the E2E backend (not dropped between runs) |
| `E2E_PYTHON` | `backend/.venv/bin/python` | Python that runs uvicorn and builds the model-zip fixture |
| `E2E_MODELS_DIR` | a fresh temp dir | `MODELS_DIR` for the E2E backend |

- **Unit** (`backend/tests/unit`, `frontend/src/**/*.test.{js,jsx}`): auth, agents, Claude call handling, the local
  model (a tiny model built in the test), the training loop, the API client and each page. They never call the
  Claude API or the Hugging Face Hub.
- **Integration** (`backend/tests/integration`): the HTTP API against a real PostgreSQL test database. It is
  dropped and recreated each session.
- **E2E** (`frontend/e2e`): Playwright drives Chromium through sign-up, account, a full training, Inspect and a
  390px-wide phone layout. `playwright.config.js` starts the backend on port 8000 with `TRAINFORM_FAKE_LLM=1` and
  a fixed 32+ byte `TRAINFORM_SECRET`, then builds the frontend and serves it with `vite preview` on port 4173. Both ports
  must be free. Failures keep a trace, and the HTML report is written to `frontend/playwright-report`.

## CI

`.github/workflows/ci.yml` runs on every pull request and every push to `main`. The backend job runs pytest on Python
3.14 with CPU torch against a Postgres 17 service, and fails below 95% coverage of `app` (statements and branches).
The frontend job runs `npm ci`, lint, Vitest with coverage (fails below 95% of lines) and the production build. The
e2e job builds the backend venv, installs Chromium, runs `npx playwright test` against its own Postgres 17 service
and uploads the HTML report when it fails.

## Operations and security notes

- **Passwords** are hashed with scrypt (n=2¹⁴, r=8, p=1, random salt). Login does a dummy hash for unknown usernames
  so response timing does not reveal which accounts exist. After 5 failed sign-ins for a username from one IP within
  15 minutes, login returns 429 until the window passes (per process; behind a proxy, rate-limit there too).
- **Viewing the password:** passwords are stored hashed, so the Account page shows a masked value. It can't show the
  real one. Users can change it there.
- **Sessions** are HMAC-signed bearer tokens that last 7 days and carry the user's `token_version`. The frontend
  keeps them in `localStorage`. Signing out (`POST /api/auth/logout`) and changing the password bump the version,
  which revokes every existing token; a password change returns a fresh token for the current session.
- **Rotating `TRAINFORM_SECRET`** invalidates every token, which logs everyone out. It also makes every linked Claude
  key unreadable, because the encryption key comes from the secret, so users must link their keys again. Until they
  do, their Claude actions return 400 with a message to re-link the key on the Account page, and
  `ANTHROPIC_API_KEY` is not used as a fallback. Keep the secret stable and back it up.
- **Weights:** only safetensors weights are accepted from the Hub and from uploads, and models load with
  `use_safetensors=True`, because pickle weights can execute code on load.
- **One active training per user:** starting a second training while one is pending or running returns 409. A
  partial unique index enforces it, so concurrent requests can't both start one. Across all users, at most
  `TRAINFORM_MAX_TRAININGS` run at once, because each holds a loaded model and a worker thread the API shares.
- **Restarts:** trainings run inside the API process. On startup, any training still `pending` or `running` is marked
  `failed` with "Interrupted by a server restart". Shutdown waits for running trainings to finish.
- **Fake mode:** never set `TRAINFORM_FAKE_LLM` in production. Any signed-in user can then run trainings without a
  Claude key, and all scores, prompts and adapters are canned.
- **Isolation:** every training, scenario and download endpoint checks ownership and returns 404 for other users'
  data. Hub and model endpoints require sign-in.
