import { expect, test } from '@playwright/test'
import { createAgent, createTraining, createUser, openScenario, signIn } from './helpers'

test.use({ viewport: { width: 390, height: 844 } })

const fits = (page) => page.evaluate(() => document.scrollingElement.scrollWidth <= window.innerWidth)

test('the auth card fits a phone', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('tab', { name: 'Create account' }).click()
  await expect(page.getByLabel('Email')).toBeVisible()
  expect(await fits(page)).toBe(true)
})

test('the signed-in pages fit a phone', async ({ page, request }) => {
  const user = await createUser(request)
  const agent = await createAgent(request, user.auth)
  await signIn(page, user)
  await expect(page.getByRole('radio', { name: new RegExp(`^${agent.name}`) })).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('button', { name: new RegExp(agent.name) }).click()
  await expect(page.getByRole('region', { name: agent.name })).toContainText('No trainings yet')
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Inspect', exact: true }).click()
  await page.getByRole('region', { name: 'Agents' }).getByRole('button', { name: new RegExp(`^${agent.name}`) }).click()
  await expect(page.getByText('No trainings yet')).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Account', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Claude linkage' })).toBeVisible()
  expect(await fits(page)).toBe(true)
})

test('a filled Inspect page fits a phone, even with an unbroken criterion', async ({ page, request }) => {
  const user = await createUser(request)
  const training = await createTraining(request, user.auth, [`Links https://example.com/${'a'.repeat(200)}`, 'Is polite.'])

  await signIn(page, user)
  const transcript = await openScenario(page, training)
  const scores = transcript.getByRole('list', { name: 'Criterion scores' }).getByRole('listitem')
  await expect(scores).toHaveCount(2)
  expect(await fits(page)).toBe(true)
  await expect(transcript).toBeFocused()
  await expect(transcript).toBeInViewport()

  // Each criterion gets the full width for its name, with its bar on a row below.
  const name = await scores.nth(1).getByText('Is polite.').boundingBox()
  const bar = await scores.nth(1).locator('[aria-hidden="true"]').boundingBox()
  const row = await scores.nth(1).boundingBox()
  expect(bar.y).toBeGreaterThanOrEqual(name.y + name.height)
  expect(bar.width).toBeGreaterThan(row.width * 0.9)
})
