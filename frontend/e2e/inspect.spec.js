import { expect, test } from '@playwright/test'
import { createUser, signIn, uploadModel } from './helpers'

test('inspect a finished training down to one scenario', async ({ page, request }) => {
  const user = await createUser(request)
  const base_model = await uploadModel(request, user.auth)
  const res = await request.post('/api/trainings', {
    headers: user.auth,
    data: { base_model, prompt: 'You help bike shop customers.', rubric: ['Is polite.', 'Is accurate.'], scenarios: 2, rounds: 1 },
  })
  expect(res.status()).toBe(201)
  const { id } = await res.json()
  await expect
    .poll(async () => (await (await request.get(`/api/trainings/${id}`, { headers: user.auth })).json()).status)
    .toBe('completed')

  await signIn(page, user)
  await page.getByRole('link', { name: 'Inspect' }).click()
  await page.getByRole('region', { name: 'Trainings' }).getByRole('button', { name: new RegExp(`^#${id} `) }).click()
  await page.getByRole('region', { name: 'Rounds' }).getByRole('button', { name: /^Round 1/ }).click()
  await page.getByRole('region', { name: 'Scenarios' }).getByRole('button', { name: /^Scenario 1/ }).click()

  const transcript = page.getByRole('region', { name: 'Scenario 1', exact: true })
  await expect(transcript.getByRole('article', { name: 'Persona' })).toContainText("Hi! I'm new here.")
  await expect(transcript.getByRole('article', { name: 'Agent' })).toContainText('Happy to help')
  const scores = transcript.getByRole('list', { name: 'Criterion scores' }).getByRole('listitem')
  await expect(scores).toHaveCount(2)
  await expect(scores.first()).toContainText(/Is polite\.\s*\d+\/10/)
  await expect(transcript.getByRole('img', { name: /^Overall score [\d.]+ out of 10$/ })).toBeVisible()
})
