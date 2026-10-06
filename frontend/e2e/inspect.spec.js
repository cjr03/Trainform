import { readFile } from 'node:fs/promises'
import { expect, test } from '@playwright/test'
import { createTraining, createUser, openScenario, openTraining, signIn } from './helpers'

test('inspect a finished training down to one scenario', async ({ page, request }) => {
  const user = await createUser(request)
  const training = await createTraining(request, user.auth, ['Is polite.', 'Is accurate.'])
  const { id } = training

  await signIn(page, user)
  // Inspect starts with agent selection; an agent's trainings show once it is picked.
  await page.getByRole('link', { name: 'Inspect', exact: true }).click()
  const agents = page.getByRole('region', { name: 'Agents' })
  await expect(agents.getByRole('button')).toHaveCount(1) // only the trained agent: no seeded one
  await expect(agents.getByRole('button', { name: new RegExp(`^${training.agent.name}`) })).toContainText('OVR')
  await expect(page.getByRole('region', { name: 'Trainings' })).toHaveCount(0)
  const transcript = await openScenario(page, training)
  await expect(transcript.getByRole('article', { name: 'Persona' })).toContainText("Hi! I'm new here.")
  await expect(transcript.getByRole('article', { name: 'Agent' })).toContainText('Happy to help')
  const scores = transcript.getByRole('list', { name: 'Criterion scores' }).getByRole('listitem')
  await expect(scores).toHaveCount(2)
  await expect(scores.first()).toContainText(/Is polite\.\s*\d+\/10/)
  await expect(transcript.getByRole('img', { name: /^Overall score [\d.]+ out of 10$/ })).toBeVisible()

  // A finished training's downloads stay reachable from Inspect.
  const [file] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download prompt' }).click()])
  expect(file.suggestedFilename()).toBe(`training-${id}-prompt.txt`)
  const expected = await request.get(`/api/trainings/${id}/download/prompt`, { headers: user.auth })
  expect((await readFile(await file.path())).toString()).toBe(await expected.text())
})

test('delete a finished training from Inspect', async ({ page, request }) => {
  const user = await createUser(request)
  const training = await createTraining(request, user.auth, ['Is polite.'])
  const { id } = training

  await signIn(page, user)
  await openTraining(page, training)
  await page.getByRole('button', { name: 'Delete training' }).click()
  await page.getByRole('alertdialog', { name: `Delete training #${id}?` }).getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.getByText('No trainings yet')).toBeVisible()
  await expect(page.getByRole('region', { name: 'Rounds' })).toHaveCount(0)
  expect((await request.get(`/api/trainings/${id}`, { headers: user.auth })).status()).toBe(404)
})
