import { expect, test } from '@playwright/test'
import { createTraining, createUser, openScenario, signIn } from './helpers'

test('inspect a finished training down to one scenario', async ({ page, request }) => {
  const user = await createUser(request)
  const id = await createTraining(request, user.auth, ['Is polite.', 'Is accurate.'])

  await signIn(page, user)
  const transcript = await openScenario(page, id)
  await expect(transcript.getByRole('article', { name: 'Persona' })).toContainText("Hi! I'm new here.")
  await expect(transcript.getByRole('article', { name: 'Agent' })).toContainText('Happy to help')
  const scores = transcript.getByRole('list', { name: 'Criterion scores' }).getByRole('listitem')
  await expect(scores).toHaveCount(2)
  await expect(scores.first()).toContainText(/Is polite\.\s*\d+\/10/)
  await expect(transcript.getByRole('img', { name: /^Overall score [\d.]+ out of 10$/ })).toBeVisible()
})
