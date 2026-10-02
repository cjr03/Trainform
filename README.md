# Trainform

Train an agent's prompt *and* weights against simulated users. Pick a Hugging Face model, write (or generate) a
system prompt and a scoring rubric, and Trainform plays out persona-driven scenarios: a persona agent talks to your
model, a scoring agent grades the reply against the rubric, a prompt generator rewrites the prompt, and the
high-scoring exchanges are used to LoRA-fine-tune the model at the end of every round.

## Stack

- **Backend:** FastAPI, SQLAlchemy, PostgreSQL, transformers + PEFT (LoRA), Anthropic SDK (helper agents)
- **Frontend:** React (Vite), react-router
- **CI:** GitHub Actions (pytest with a Postgres service, Vitest, Playwright E2E)

## Run locally

```bash
# Postgres
brew install postgresql@17 && brew services start postgresql@17
createdb trainform && createdb trainform_test

# Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export TRAINFORM_SECRET="$(openssl rand -hex 32)"   # signs sessions + encrypts linked Claude keys; keep it stable
.venv/bin/uvicorn app.main:app --reload

# Frontend (second terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173, proxies /api to :8000
```

Each user links their own Anthropic API key during signup or on the Account page (stored encrypted).
`ANTHROPIC_API_KEY` in the backend environment is used as a fallback for users without one.

## Tests

```bash
cd backend && .venv/bin/pytest          # unit + integration (uses the trainform_test database)
cd frontend && npx vitest --run         # component/unit tests
```
