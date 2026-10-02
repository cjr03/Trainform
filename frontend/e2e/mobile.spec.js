import { expect, test } from '@playwright/test'
import { createUser, signIn } from './helpers'

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
  await signIn(page, await createUser(request))
  await expect(page.getByRole('group', { name: 'Local models' }).locator('.spinner')).toHaveCount(0)
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Inspect' }).click()
  await expect(page.getByText('No trainings yet')).toBeVisible()
  expect(await fits(page)).toBe(true)

  await page.getByRole('link', { name: 'Account' }).click()
  await expect(page.getByRole('heading', { name: 'Claude linkage' })).toBeVisible()
  expect(await fits(page)).toBe(true)
})
