import { execFileSync } from 'node:child_process'
import { expect } from '@playwright/test'

export const uid = (prefix = 'e2e') => `${prefix}_${crypto.randomUUID().slice(0, 13)}`

export async function createUser(request) {
  const username = uid()
  const user = { username, email: `${username}@example.com`, password: 'password-e2e' }
  const res = await request.post('/api/auth/signup', { data: { ...user, password_confirm: user.password } })
  expect(res.status()).toBe(201)
  return { ...user, auth: { Authorization: `Bearer ${(await res.json()).token}` } }
}

export async function signIn(page, { username, password }) {
  await page.goto('/')
  await page.getByLabel('Username').fill(username)
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByRole('heading', { name: 'Train an agent' })).toBeVisible()
}

// save_upload only needs config.json and a *.safetensors file (here an empty one: u64 header length + "{}");
// fake mode loads neither. The backend's Python (set in playwright.config.js) writes the zip.
const zipScript = `
import io, sys, zipfile
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("config.json", "{}")
    z.writestr("model.safetensors", b"\\x02" + bytes(7) + b"{}")
sys.stdout.buffer.write(buf.getvalue())
`
export const modelZip = () => execFileSync(process.env.E2E_PYTHON, ['-c', zipScript])

export async function uploadModel(request, auth) {
  const name = uid('model')
  const file = { name: `${name}.zip`, mimeType: 'application/zip', buffer: modelZip() }
  const res = await request.post('/api/hub/upload', { headers: auth, multipart: { file, name } })
  expect(res.status()).toBe(201)
  return name
}

// A finished fake-mode training (2 scenarios x 1 round) on a freshly uploaded model.
export async function createTraining(request, auth, rubric) {
  const base_model = await uploadModel(request, auth)
  const res = await request.post('/api/trainings', {
    headers: auth,
    data: { base_model, prompt: 'You help bike shop customers.', rubric, scenarios: 2, rounds: 1 },
  })
  expect(res.status()).toBe(201)
  const { id } = await res.json()
  await expect
    .poll(async () => (await (await request.get(`/api/trainings/${id}`, { headers: auth })).json()).status)
    .toBe('completed')
  return id
}

// Opens Inspect on round 1, scenario 1 of training `id` and returns the transcript region.
export async function openScenario(page, id) {
  await page.getByRole('link', { name: 'Inspect' }).click()
  await page.getByRole('region', { name: 'Trainings' }).getByRole('button', { name: new RegExp(`^#${id} `) }).click()
  await page.getByRole('region', { name: 'Rounds' }).getByRole('button', { name: /^Round 1/ }).click()
  await page.getByRole('region', { name: 'Scenarios' }).getByRole('button', { name: /^Scenario 1/ }).click()
  return page.getByRole('region', { name: 'Scenario 1', exact: true })
}
