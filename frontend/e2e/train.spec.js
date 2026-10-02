import { readFile } from 'node:fs/promises'
import { expect, test } from '@playwright/test'
import { createUser, modelZip, signIn, uid } from './helpers'

test('train an agent from model upload to downloads', async ({ page, request }) => {
  await signIn(page, await createUser(request))
  const next = page.getByRole('button', { name: 'Next' })

  // Model: upload a zip through the UI and select it.
  const model = uid('model')
  await page.getByRole('group', { name: 'Model source' }).getByRole('radio', { name: 'Upload' }).check()
  await page.getByLabel('Model archive (.zip)').setInputFiles({ name: 'model.zip', mimeType: 'application/zip', buffer: modelZip() })
  await page.getByLabel('Model name').fill(model)
  await page.getByRole('button', { name: 'Upload model' }).click()
  const option = page.getByRole('group', { name: 'Local models' }).getByRole('radio', { name: model })
  await option.check()
  await expect(option).toBeChecked()
  await next.click()

  // Prompt: describe the agent and let (fake) Claude draft it.
  await page.getByRole('radio', { name: 'Describe my agent' }).check()
  await page.getByLabel('Agent description').fill('A support agent for a bike shop')
  await page.getByRole('button', { name: 'Generate prompt' }).click()
  await expect(page.getByLabel('Agent prompt')).toHaveValue(/^You are an AI agent for: A support agent for a bike shop/)
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
  await expect(page.getByRole('progressbar', { name: 'Training rounds' })).toHaveAttribute('aria-valuemax', '2')
  for (const name of ['Last n runs', 'Last n/2', 'Last n/4', 'Last run'])
    await expect(page.getByRole('group', { name, exact: true })).toBeVisible()
  release()

  await expect(page.getByRole('heading', { name: 'Complete' })).toBeVisible({ timeout: 30_000 })
  await expect(page.getByTestId('best-score')).toHaveText(/^\d+\.\d\/ 10$/)

  const download = async (name) => {
    const [file] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name }).click()])
    return [file.suggestedFilename(), await readFile(await file.path())]
  }
  const [promptName, prompt] = await download('Download prompt')
  expect(promptName).toBe(`training-${id}-prompt.txt`)
  expect(prompt.toString()).toContain('You are an AI agent for: A support agent for a bike shop')
  const [weightsName, weights] = await download('Download weights')
  expect(weightsName).toBe(`training-${id}-lora.zip`)
  expect(weights.subarray(0, 2).toString()).toBe('PK')

  // The finished run's progress screen is full.
  await page.getByRole('button', { name: /^Training/ }).click()
  await expect(page.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '4')
  await expect(page.getByRole('progressbar', { name: 'Training rounds' })).toHaveAttribute('aria-valuenow', '2')
  await expect(page.getByRole('group', { name: 'Last n runs' })).toContainText('4 runs')
})
