import { mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { defineConfig } from '@playwright/test'

const python = process.env.E2E_PYTHON || resolve(import.meta.dirname, '../backend/.venv/bin/python')
// Workers re-evaluate this file but inherit the env, so the runner and its workers share one models dir.
process.env.E2E_MODELS_DIR ||= mkdtempSync(join(tmpdir(), 'trainform-e2e-models-'))

export default defineConfig({
  testDir: 'e2e',
  forbidOnly: Boolean(process.env.CI),
  reporter: [['list'], ['html', { open: 'never' }]],
  use: { baseURL: 'http://localhost:4173', trace: 'retain-on-failure' },
  webServer: [
    {
      command: `"${python}" -m uvicorn app.main:app --port 8000`,
      cwd: '../backend',
      url: 'http://127.0.0.1:8000/api/health',
      env: {
        DATABASE_URL: process.env.E2E_DATABASE_URL || 'postgresql+psycopg://localhost/trainform_e2e',
        TRAINFORM_SECRET: 'e2e-secret',
        TRAINFORM_FAKE_LLM: '1',
        MODELS_DIR: process.env.E2E_MODELS_DIR,
      },
    },
    {
      command: 'npm run build && npx vite preview --port 4173 --strictPort',
      url: 'http://localhost:4173',
    },
  ],
})
