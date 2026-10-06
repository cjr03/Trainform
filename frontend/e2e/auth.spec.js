import { expect, test } from '@playwright/test'
import { accountMenu, createUser, signIn, uid } from './helpers'

test('the auth card offers sign in and create account', async ({ page }) => {
  await page.goto('/')
  const tabs = page.getByRole('tablist', { name: 'Account access' })
  await expect(tabs.getByRole('tab', { name: 'Sign in' })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByRole('button', { name: 'Sign in' })).toBeVisible()

  await tabs.getByRole('tab', { name: 'Create account' }).click()
  await expect(tabs.getByRole('tab', { name: 'Create account' })).toHaveAttribute('aria-selected', 'true')
  await expect(page.getByLabel('Email')).toBeVisible()
})

test('sign up through every phase, skip Claude, land on Train and sign out', async ({ page }) => {
  const username = uid()
  await page.goto('/')
  await page.getByRole('tab', { name: 'Create account' }).click()
  const current = page.getByRole('list', { name: 'Sign-up progress' }).locator('[aria-current="step"]')

  await expect(current).toHaveText(/Identity/)
  await page.getByLabel('Email').fill(`${username}@example.com`)
  await page.getByLabel('Username').fill(username)
  await page.getByRole('button', { name: 'Continue' }).click()

  await expect(current).toHaveText(/Security/)
  await page.getByLabel('Password', { exact: true }).fill('password-e2e')
  await page.getByLabel('Confirm password').fill('password-e2e')
  await page.getByRole('button', { name: 'Continue' }).click()

  await expect(current).toHaveText(/Link Claude/)
  await expect(page.getByLabel('Anthropic API key')).toBeVisible()
  await page.getByRole('button', { name: 'Skip for now' }).click()

  await expect(page).toHaveURL(/\/train$/)
  await expect(page.getByRole('heading', { name: 'Train an agent' })).toBeVisible()

  await accountMenu(page, 'Sign out')
  await expect(page.getByRole('tab', { name: 'Sign in' })).toBeVisible()
})

test('sign in with an existing account', async ({ page, request }) => {
  await signIn(page, await createUser(request))
  await expect(page).toHaveURL(/\/train$/)
})

test('a wrong password shows an error', async ({ page, request }) => {
  const user = await createUser(request)
  await page.goto('/')
  await page.getByLabel('Username').fill(user.username)
  await page.getByLabel('Password').fill('not-the-password')
  await page.getByRole('button', { name: 'Sign in' }).click()
  await expect(page.getByRole('alert')).toHaveText('Invalid username or password')
  await expect(page.getByRole('tab', { name: 'Sign in' })).toBeVisible()
})
