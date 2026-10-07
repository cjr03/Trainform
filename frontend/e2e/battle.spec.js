import { expect, test } from '@playwright/test'
import { createAgent, createUser, signIn } from './helpers'

test('battle an agent against its optimised challenger; level grades keep the agent', async ({ page, request }) => {
  const user = await createUser(request)
  // Fake mode never calls Claude; a linked key makes it the helper that grades both sides.
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
  await page.getByLabel('Battle rounds').selectOption({ label: 'Best of 3' })
  await page.getByRole('button', { name: 'Start battle' }).click()

  // The fake model's replies are the same whatever its weights, so both sides grade the same and the champion holds.
  await expect(page.getByTestId('battle-score')).toHaveText(/^(\d+\.\d) – \1$/, { timeout: 30_000 })
  await expect(page.getByText('Your agent holds. It keeps its prompt and weights.')).toBeVisible()
  await page.locator('summary', { hasText: 'Scenarios' }).click()
  await expect(page.getByTestId('rounds-won')).toHaveText('Rounds won 0 – 0') // level every round: a tie keeps yours
  await expect(page.getByText(/^Scenario \d: (\d+\.\d) – \1$/)).toHaveCount(6) // 2 new users a round
  const saved = await (await request.get(`/api/agents/${agent.id}`, { headers: user.auth })).json()
  expect(saved.prompt).toBe('You help bike shop customers.')
  await expect(page.getByRole('region', { name: 'Past battles' }).getByRole('button')).toContainText(
    '3 battle rounds · challenger trained for 1 round · Your agent held')

  // The Train page doesn't pick up the battle.
  await page.getByRole('link', { name: 'Train', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Train an agent' })).toBeVisible()
  await expect(page.getByRole('list', { name: 'Training workflow' }).getByRole('button', { name: /^Agent/ })).toHaveAttribute('aria-current', 'step')
})

test('challenge a published agent on the same model', async ({ page, request }) => {
  const [me, rival] = [await createUser(request), await createUser(request)]
  // Two Claude agents: the same base model without a Hub download (uploaded models are their uploader's only).
  const add = async (user, name) => {
    await request.put('/api/me/claude-key', { headers: user.auth, data: { api_key: 'fake-key' } })
    const res = await request.post('/api/agents', { headers: user.auth, data: { name, base_model: 'claude', prompt: `You are ${name}.` } })
    expect(res.status()).toBe(201)
    return (await res.json()).id
  }
  const mine = await add(me, 'Mine')
  const theirs = await add(rival, 'Rival')
  expect((await request.post('/api/social/posts', { headers: rival.auth, data: { agent_id: theirs } })).status()).toBe(201)
  expect((await request.post(`/api/social/users/${rival.username}/follow`, { headers: me.auth })).status()).toBe(204)
  await signIn(page, me)

  await page.getByRole('link', { name: 'Battle', exact: true }).click()
  await page.getByRole('radio', { name: /^Mine/ }).check()
  await page.getByRole('radio', { name: 'Challenge an agent' }).check()
  await page.getByRole('button', { name: 'Challenge' }).click()
  await expect(page.getByText(`Rival by ${rival.username}`)).toBeVisible()
  await page.getByRole('button', { name: 'Next' }).click()
  await page.getByLabel('New criterion').fill('Is polite.')
  await page.getByRole('button', { name: 'Add criterion' }).click()
  await page.getByRole('button', { name: 'Next' }).click()
  await expect(page.getByLabel('Training rounds')).toBeHidden()
  await page.getByLabel('Simulated scenarios per round').fill('2')
  await page.getByRole('button', { name: 'Start battle' }).click()

  await expect(page.getByTestId('battle-score')).toBeVisible({ timeout: 30_000 })
  await expect(page.getByText(`Your agent vs Rival by ${rival.username}`)).toBeVisible()
  const agent = await (await request.get(`/api/agents/${mine}`, { headers: me.auth })).json()
  expect(agent.prompt).toBe('You are Mine.') // level grades: yours holds
})
