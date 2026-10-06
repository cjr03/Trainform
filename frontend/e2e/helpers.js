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

// An agent on a freshly uploaded model; returns {id, name}.
export async function createAgent(request, auth) {
  const base_model = await uploadModel(request, auth)
  const name = uid('agent')
  const res = await request.post('/api/agents', {
    headers: auth,
    data: { name, base_model, prompt: 'You help bike shop customers.' },
  })
  expect(res.status()).toBe(201)
  return { id: (await res.json()).id, name }
}

// A finished fake-mode training (2 scenarios x 1 round) of a new agent; returns {id, agent}.
export async function createTraining(request, auth, rubric) {
  const agent = await createAgent(request, auth)
  const res = await request.post('/api/trainings', {
    headers: auth,
    data: { agent_id: agent.id, rubric, scenarios: 2, rounds: 1 },
  })
  expect(res.status()).toBe(201)
  const { id } = await res.json()
  await expect
    .poll(async () => (await (await request.get(`/api/trainings/${id}`, { headers: auth })).json()).status)
    .toBe('completed')
  return { id, agent }
}

// Opens Inspect on `agent`, then its training `id`.
export async function openTraining(page, { id, agent }) {
  await page.getByRole('link', { name: 'Inspect', exact: true }).click()
  await page.getByRole('region', { name: 'Agents' }).getByRole('button', { name: new RegExp(`^${agent.name}`) }).click()
  await page.getByRole('region', { name: 'Trainings' }).getByRole('button', { name: new RegExp(`^#${id} `) }).click()
}

// Opens Inspect on round 1, scenario 1 of `training` and returns the transcript region.
export async function openScenario(page, training) {
  await openTraining(page, training)
  await page.getByRole('region', { name: 'Rounds' }).getByRole('button', { name: /^Round 1/ }).click()
  await page.getByRole('region', { name: 'Scenarios' }).getByRole('button', { name: /^Scenario 1/ }).click()
  return page.getByRole('region', { name: 'Scenario 1', exact: true })
}

// Picks `item` ("Account", "Sign out" or your username) from the header's account menu.
export async function accountMenu(page, item) {
  await page.getByRole('button', { name: 'Account menu' }).click()
  await page.getByRole('menuitem', { name: item, exact: true }).click()
}
