import { readFile } from 'node:fs/promises'
import { expect, test } from '@playwright/test'
import { createAgent, createUser, modelZip, signIn, uid } from './helpers'

test('make an agent from a model upload, train it and see its overall', async ({ page, request }) => {
  const user = await createUser(request)
  // Fake mode never calls Claude, so any key links it; a linked key makes Claude the default helper.
  expect((await request.put('/api/me/claude-key', { headers: user.auth, data: { api_key: 'fake-key' } })).status()).toBe(200)
  await signIn(page, user)
  page.on('dialog', (dialog) => dialog.accept()) // "Replace your prompt?"

  // Agents: a new user starts with none.
  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await expect(page.getByText('No agents yet. Make your first one.')).toBeVisible()
  const agents = page.getByRole('list', { name: 'Your agents' })

  // New agent: upload a model zip through the UI and select it.
  const name = uid('agent')
  await page.getByRole('button', { name: 'New agent' }).click()
  await page.getByLabel('Name', { exact: true }).fill(name)
  const model = uid('model')
  await page.getByRole('radiogroup', { name: 'Model source' }).getByRole('radio', { name: 'Upload' }).check()
  await page.getByLabel('Model archive (.zip)').setInputFiles({ name: 'model.zip', mimeType: 'application/zip', buffer: modelZip() })
  await page.getByLabel('Model name').fill(model)
  await page.getByRole('button', { name: 'Upload model' }).click()
  const option = page.getByRole('radiogroup', { name: 'Local models' }).getByRole('radio', { name: model })
  await option.check()
  await expect(option).toBeChecked()

  // Prompt: let (fake) Claude draft one, then replace it with an uploaded file.
  await page.getByRole('radio', { name: 'Describe my agent' }).check()
  await page.getByLabel('Agent description').fill('A support agent for a bike shop')
  await page.getByRole('button', { name: 'Generate prompt' }).click()
  await expect(page.getByLabel('Agent prompt')).toHaveValue(/^You are an AI agent for: A support agent for a bike shop/)
  const uploaded = 'You are the support agent for a bike shop.\n\nBe brief and polite.'
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.getByRole('button', { name: 'Upload prompt' }).click()])
  await chooser.setFiles({ name: 'prompt.md', mimeType: 'text/markdown', buffer: Buffer.from(uploaded) })
  await expect(page.getByLabel('Agent prompt')).toHaveValue(uploaded)
  await page.getByRole('button', { name: 'Create agent' }).click()
  const detail = page.getByRole('region', { name })
  await expect(detail.getByRole('img', { name: 'Unranked' })).toBeVisible()
  await expect(detail.getByRole('region', { name: 'Prompt' })).toContainText('Be brief and polite.')

  // Train: pick the agent; Claude runs the helpers.
  await page.getByRole('link', { name: 'Train', exact: true }).click()
  const next = page.getByRole('button', { name: 'Next' })
  const stepButton = (name) => page.getByRole('list', { name: 'Training workflow' }).getByRole('button', { name })
  await expect(stepButton(/^Rubric/)).toHaveCSS('opacity', '0.4') // locked
  await expect(next).toBeDisabled()
  await page.getByRole('radio', { name: new RegExp(`^${name}`) }).check()
  await expect(page.getByRole('radiogroup', { name: 'Train with' }).getByRole('radio', { name: 'Claude' })).toBeChecked()
  await next.click()

  // Rubric: generate, then add, edit and delete criteria.
  await page.getByRole('button', { name: 'Generate from prompt' }).click()
  const criteria = page.getByRole('textbox', { name: /^Criterion \d+$/ })
  const criterion = (n) => page.getByRole('textbox', { name: `Criterion ${n}`, exact: true })
  await expect(criteria).toHaveCount(4)
  await page.getByLabel('New criterion').fill('Offers to book a repair.')
  await page.getByRole('button', { name: 'Add criterion' }).click()
  await expect(criterion(5)).toHaveValue('Offers to book a repair.')
  await criterion(1).fill('Greets the customer by name.')
  await page.getByRole('button', { name: 'Delete criterion 2' }).click()
  await expect(criteria).toHaveCount(4)
  await expect(criterion(1)).toHaveValue('Greets the customer by name.')
  await expect(criterion(4)).toHaveValue('Offers to book a repair.')
  await next.click()

  // Scenarios: 2 x 2.
  await page.getByLabel('Simulated scenarios per round').fill('2')
  await page.getByLabel('Training rounds').fill('2')
  await expect(page.getByTestId('total-runs')).toHaveText('4')

  // Hold the progress polls so the progress screen can be checked before the (instant) fake run completes.
  let release
  const held = new Promise((resolve) => (release = resolve))
  await page.route('**/api/trainings/*', async (route) => {
    await held
    await route.continue()
  })
  const started = page.waitForResponse((r) => r.url().endsWith('/api/trainings') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Start training' }).click()
  const { id } = await (await started).json()

  await expect(page.getByRole('heading', { name: 'Training', exact: true })).toBeVisible()
  await expect(page.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuemax', '4')
  await expect(page.getByRole('progressbar', { name: 'Rounds completed' })).toHaveAttribute('aria-valuemax', '2')
  for (const name of ['Total runs', 'Last 50%', 'Last 25%', 'Last run'])
    await expect(page.getByRole('group', { name, exact: true })).toBeVisible()
  release()

  await expect(page.getByRole('heading', { name: 'Complete' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('best-score')).toHaveText(/^\d+\.\d\/ 10$/)
  // Finished steps are done, not dimmed like locked ones.
  await expect(stepButton(/^Agent/)).toHaveCSS('opacity', '1')

  const download = async (name) => {
    const [file] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name }).click()])
    return [file.suggestedFilename(), await readFile(await file.path())]
  }
  const [promptName, prompt] = await download('Download prompt')
  expect(promptName).toBe(`training-${id}-prompt.txt`)
  expect(prompt.toString()).toContain(uploaded)
  expect(prompt.toString()).toBe(await page.getByTestId('best-prompt').textContent())
  const [weightsName, weights] = await download('Download weights')
  expect(weightsName).toBe(`training-${id}-lora.zip`)
  expect(weights.subarray(0, 2).toString()).toBe('PK')

  // The finished run's progress screen is full.
  await page.getByRole('button', { name: /^Training/ }).click()
  await expect(page.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '4')
  await expect(page.getByRole('progressbar', { name: 'Rounds completed' })).toHaveAttribute('aria-valuenow', '2')
  await expect(page.getByRole('group', { name: 'Total runs' })).toContainText('4 runs')

  // The agent's overall is its best score x 10, and its past trainings list this one.
  const best = Number((await page.getByTestId('best-score').textContent()).split('/')[0])
  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await expect(agents.getByRole('button', { name: new RegExp(`^${name} .* Overall ${Math.round(best * 10)}$`) })).toBeVisible()
  await agents.getByRole('button', { name: new RegExp(name) }).click()
  await expect(detail.getByRole('region', { name: 'Past trainings' })).toContainText(`#${id}`)
})

test('pause a training, find it in Inspect and resume it to completion', async ({ page, request }) => {
  const user = await createUser(request)
  const agent = await createAgent(request, user.auth)
  // 20 scenarios at 0.25 s a reply (TRAINFORM_FAKE_DELAY) leave time to pause; the Train page opens the running one.
  const res = await request.post('/api/trainings', {
    headers: user.auth,
    data: { agent_id: agent.id, rubric: ['Is polite.'], scenarios: 5, rounds: 4 },
  })
  expect(res.status()).toBe(201)
  const { id } = await res.json()
  const get = async () => (await request.get(`/api/trainings/${id}`, { headers: user.auth })).json()
  await signIn(page, user)

  await page.getByRole('button', { name: 'Pause' }).click()
  // Pausing drops the step in progress instead of finishing it: only whole rounds are kept.
  await expect(page.getByRole('status')).toHaveText('paused', { timeout: 5_000 })
  await expect(page.getByText(/step in progress was dropped/)).toBeVisible()
  const saved = await get()
  expect(saved.status).toBe('paused')
  expect(saved.scenarios_done).toBeLessThan(20)
  expect(saved.scenarios_done % 5).toBe(0)
  await expect(page.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', String(saved.scenarios_done))

  // Paused trainings stay in Inspect, which resumes them and opens the Train page on the running one.
  await page.getByRole('link', { name: 'Inspect', exact: true }).click()
  await page.getByRole('region', { name: 'Agents' }).getByRole('button', { name: new RegExp(`^${agent.name}`) }).click()
  const option = page.getByRole('region', { name: 'Trainings' }).getByRole('button', { name: new RegExp(`^#${id} `) })
  await expect(option).toContainText('paused')
  await option.click()
  await page.getByRole('button', { name: 'Resume training' }).click()
  await expect(page).toHaveURL(/\/train$/)
  await expect(page.getByRole('heading', { name: 'Complete' })).toBeVisible({ timeout: 30_000 })

  // It continued from the saved progress: every scenario ran exactly once.
  expect((await get()).scenarios_done).toBe(20)
  const rounds = await (await request.get(`/api/trainings/${id}/rounds`, { headers: user.auth })).json()
  expect(rounds.map((r) => r.scenarios)).toEqual([5, 5, 5, 5])
})
