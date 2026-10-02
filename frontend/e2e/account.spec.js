import { expect, test } from '@playwright/test'
import { createUser, signIn } from './helpers'

let user
test.beforeEach(async ({ page, request }) => {
  user = await createUser(request)
  await signIn(page, user)
  await page.getByRole('link', { name: 'Account' }).click()
  await expect(page.getByRole('heading', { name: 'Profile' })).toBeVisible()
})

test('shows the profile with a masked password', async ({ page }) => {
  await expect(page.getByText(user.username, { exact: true })).toBeVisible()
  await expect(page.getByText(user.email)).toBeVisible()
  await expect(page.getByText('••••••••')).toBeVisible()
})

test('change the password, then sign in with the new one', async ({ page }) => {
  await page.getByLabel('Current password').fill(user.password)
  await page.getByLabel('New password', { exact: true }).fill('new-password-e2e')
  await page.getByLabel('Confirm new password').fill('new-password-e2e')
  await page.getByRole('button', { name: 'Change password' }).click()
  await expect(page.getByRole('status')).toHaveText('Password changed.')

  await page.getByRole('button', { name: 'Sign out' }).click()
  await signIn(page, { ...user, password: 'new-password-e2e' })
})

test('link and unlink a Claude key', async ({ page }) => {
  const card = page.getByRole('region', { name: 'Claude linkage' })
  await expect(card.getByText('Not linked')).toBeVisible()

  await card.getByLabel('Anthropic API key').fill('sk-ant-e2e-not-a-real-key')
  await card.getByRole('button', { name: 'Link key' }).click()
  await expect(card.getByRole('status')).toHaveText('Key saved.')
  await expect(card.getByText('Linked', { exact: true })).toBeVisible()

  await card.getByRole('button', { name: 'Unlink' }).click()
  await expect(card.getByRole('status')).toHaveText('Key removed.')
  await expect(card.getByText('Not linked')).toBeVisible()
})
