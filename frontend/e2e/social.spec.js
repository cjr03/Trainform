import { expect, test } from '@playwright/test'
import { accountMenu, createAgent, createUser, signIn } from './helpers'

test('the header has icon links, and the account menu opens Account and signs out', async ({ page, request }) => {
  const user = await createUser(request)
  await signIn(page, user)
  for (const name of ['Train', 'Agents', 'Inspect']) await expect(page.getByRole('link', { name, exact: true })).toBeVisible()
  await page.getByRole('link', { name: 'Feed' }).click()
  await expect(page.getByRole('heading', { name: 'Feed' })).toBeVisible()
  await page.getByRole('link', { name: 'Search Trainformers' }).click()
  await expect(page.getByRole('heading', { name: 'Find Trainformers' })).toBeVisible()
  await accountMenu(page, 'Account')
  await expect(page.getByRole('heading', { name: 'Claude linkage' })).toBeVisible()
  await accountMenu(page, 'Sign out')
  await expect(page.getByRole('tab', { name: 'Sign in' })).toBeVisible()
})

test('B publishes an agent with an image; A follows B, votes on it and copies it', async ({ page, request }) => {
  const [a, b] = [await createUser(request), await createUser(request)]
  const agent = await createAgent(request, b.auth)

  // B gives the agent a generated image and publishes it.
  await signIn(page, b)
  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('list', { name: 'Your agents' }).getByRole('button', { name: new RegExp(agent.name) }).click()
  const detail = page.getByRole('region', { name: agent.name })
  await detail.getByRole('button', { name: 'Generate image' }).click()
  await expect(detail.getByRole('img', { name: `${agent.name} image` })).toBeVisible()
  await page.getByRole('button', { name: 'Publish an agent' }).click()
  const dialog = page.getByRole('dialog', { name: 'Publish an agent' })
  await dialog.getByRole('radio', { name: new RegExp(agent.name) }).check()
  await dialog.getByLabel('Caption (optional)').fill('Knows bikes.')
  await dialog.getByRole('button', { name: 'Publish' }).click()
  await expect(dialog.getByRole('status')).toContainText('Published!')
  await page.keyboard.press('Escape')
  await accountMenu(page, 'Sign out')

  // A finds B, follows them and sees the post in the feed.
  await signIn(page, a)
  await page.getByRole('link', { name: 'Search Trainformers' }).click()
  await page.getByRole('searchbox', { name: 'Search Trainformers' }).fill(b.username)
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await page.getByRole('button', { name: `Follow ${b.username}` }).click()
  await expect(page.getByRole('button', { name: `Unfollow ${b.username}` })).toBeVisible()

  await page.getByRole('link', { name: 'Feed' }).click()
  const post = page.getByRole('article', { name: agent.name })
  await expect(post).toContainText('Knows bikes.')
  await post.getByRole('button', { name: 'Upvote' }).click()
  await expect(post.getByRole('button', { name: 'Upvote' })).toHaveAttribute('aria-pressed', 'true')
  await expect(post.getByRole('group', { name: 'Votes' })).toHaveText('1')

  // "Find an agent" copies it into A's agents.
  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('button', { name: 'New agent' }).click()
  await page.getByRole('radio', { name: 'Find an agent' }).check()
  await page.getByRole('article', { name: agent.name }).getByRole('button', { name: 'Use this agent' }).click()
  await expect(page.getByRole('list', { name: 'Your agents' }).getByRole('button', { name: new RegExp(agent.name) })).toBeVisible()
  await expect(page.getByRole('region', { name: agent.name }).getByRole('img', { name: `${agent.name} image` })).toBeVisible()
})
