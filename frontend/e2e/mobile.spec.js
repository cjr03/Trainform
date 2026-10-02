import { expect, test } from '@playwright/test'
import { createTraining, createUser, openScenario, signIn, uploadModel } from './helpers'

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
  const model = await uploadModel(request, user.auth)
  await signIn(page, user)
  await expect(page.getByRole('radio', { name: model })).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Inspect' }).click()
  await expect(page.getByText('No trainings yet')).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Account' }).click()
  await expect(page.getByRole('heading', { name: 'Claude linkage' })).toBeVisible()
  expect(await fits(page)).toBe(true)
})

test('a filled Inspect page fits a phone, even with an unbroken criterion', async ({ page, request }) => {
  const user = await createUser(request)
  const id = await createTraining(request, user.auth, [`Links https://example.com/${'a'.repeat(200)}`, 'Is polite.'])

  await signIn(page, user)
  const transcript = await openScenario(page, id)
  const scores = transcript.getByRole('list', { name: 'Criterion scores' }).getByRole('listitem')
  await expect(scores).toHaveCount(2)
  expect(await fits(page)).toBe(true)
  await expect(transcript).toBeFocused()
  await expect(transcript).toBeInViewport()

  // Each criterion gets the full width for its name, with its bar on a row below.
  const name = await scores.nth(1).getByText('Is polite.').boundingBox()
  const bar = await scores.nth(1).locator('.inspect-bar').boundingBox()
  const row = await scores.nth(1).boundingBox()
  expect(bar.y).toBeGreaterThanOrEqual(name.y + name.height)
  expect(bar.width).toBeGreaterThan(row.width * 0.9)
})
