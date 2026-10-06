import { expect, test } from '@playwright/test'
import { createAgent, createUser, signIn } from './helpers'

test('battle an agent against its optimised challenger; a tie keeps the agent', async ({ page, request }) => {
  const user = await createUser(request)
  // Fake mode never calls Claude; a linked key makes the pairwise judge decide the bouts.
  expect((await request.put('/api/me/claude-key', { headers: user.auth, data: { api_key: 'fake-key' } })).status()).toBe(200)
  const agent = await createAgent(request, user.auth)
  await signIn(page, user)

  await page.getByRole('link', { name: 'Battle', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Battle your agent' })).toBeVisible()
  await page.getByRole('radio', { name: new RegExp(`^${agent.name}`) }).check()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByLabel('New criterion').fill('Is polite.')
  await page.getByRole('button', { name: 'Add criterion' }).click()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByLabel('Simulated scenarios per round').fill('2')
  await page.getByLabel('Training rounds').fill('1')
  await page.getByRole('button', { name: 'Start battle' }).click()

  // The fake model's replies are the same whatever its weights, so every bout ties and the champion holds.
  await expect(page.getByTestId('battle-score')).toHaveText('0 – 0', { timeout: 30_000 })
  await expect(page.getByText('Your agent holds. It keeps its prompt and weights.')).toBeVisible()
  await page.getByText('Bouts', { exact: true }).click()
  await expect(page.getByText(/^Bout \d: Tie$/)).toHaveCount(2)
  const saved = await (await request.get(`/api/agents/${agent.id}`, { headers: user.auth })).json()
  expect(saved.prompt).toBe('You help bike shop customers.')

  // The Train page doesn't pick up the battle.
  await page.getByRole('link', { name: 'Train', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Train an agent' })).toBeVisible()
  await expect(page.getByRole('list', { name: 'Training workflow' }).getByRole('button', { name: /^Agent/ })).toHaveAttribute('aria-current', 'step')
})
