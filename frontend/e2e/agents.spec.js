import { expect, test } from '@playwright/test'
import { createTraining, createUser, signIn, uid } from './helpers'

test("edit an agent's prompt by upload, open it in Inspect and delete it", async ({ page, request }) => {
  const user = await createUser(request)
  const { id, agent } = await createTraining(request, user.auth, ['Is polite.'])
  await signIn(page, user)
  page.on('dialog', (dialog) => dialog.accept()) // "Replace your prompt?"

  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('list', { name: 'Your agents' }).getByRole('button', { name: new RegExp(agent.name) }).click()
  const detail = page.getByRole('region', { name: agent.name })
  await expect(detail.getByRole('img', { name: /^Overall \d+$/ })).toBeVisible()
  await expect(detail.getByRole('region', { name: 'Past trainings' })).toContainText(`#${id}`)

  await detail.getByRole('button', { name: 'Edit prompt' }).click()
  const [chooser] = await Promise.all([page.waitForEvent('filechooser'), detail.getByRole('button', { name: 'Upload prompt' }).click()])
  await chooser.setFiles({ name: 'prompt.txt', mimeType: 'text/plain', buffer: Buffer.from('Uploaded prompt.\n  Indented line.') })
  await detail.getByRole('button', { name: 'Save prompt' }).click()
  await expect(detail.getByRole('region', { name: 'Prompt' })).toContainText('Uploaded prompt.\n  Indented line.', { useInnerText: true })
  const saved = await (await request.get(`/api/agents/${agent.id}`, { headers: user.auth })).json()
  expect(saved.prompt).toBe('Uploaded prompt.\n  Indented line.')

  await detail.getByRole('link', { name: 'Open in Inspect' }).click()
  await expect(page).toHaveURL(new RegExp(`/inspect\\?agent=${agent.id}$`))
  await expect(page.getByRole('region', { name: 'Trainings' }).getByRole('button', { name: new RegExp(`^#${id} `) })).toBeVisible()

  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('list', { name: 'Your agents' }).getByRole('button', { name: new RegExp(agent.name) }).click()
  await detail.getByRole('button', { name: 'Delete agent' }).click()
  await page.getByRole('alertdialog', { name: `Delete ${agent.name}?` }).getByRole('button', { name: 'Delete', exact: true }).click()
  await expect(page.getByRole('list', { name: 'Your agents' }).getByRole('button', { name: new RegExp(agent.name) })).toHaveCount(0)
  expect((await request.get(`/api/trainings/${id}`, { headers: user.auth })).status()).toBe(404)
})

test('a Claude agent trains its prompt only, with Claude as the helper', async ({ page, request }) => {
  const user = await createUser(request)
  expect((await request.put('/api/me/claude-key', { headers: user.auth, data: { api_key: 'fake-key' } })).status()).toBe(200)
  await signIn(page, user)

  const name = uid('claude')
  const note = "Claude's weights can't be fine-tuned, so training will only improve its prompt."
  await page.getByRole('link', { name: 'Agents', exact: true }).click()
  await page.getByRole('button', { name: 'New agent' }).click()
  await page.getByLabel('Name', { exact: true }).fill(name)
  await page.getByRole('radio', { name: 'Claude' }).check()
  await expect(page.getByText(note)).toBeVisible()
  await page.getByLabel('Agent prompt').fill('You help bike shop customers.')
  await page.getByRole('button', { name: 'Create agent' }).click()
  await expect(page.getByRole('region', { name })).toContainText('Claude · 0 trainings')

  await page.getByRole('link', { name: 'Train', exact: true }).click()
  await page.getByRole('radio', { name: new RegExp(`^${name}`) }).check()
  await expect(page.getByText(`${note} Claude agents need Claude as the helper.`)).toBeVisible()
  const helper = page.getByRole('radiogroup', { name: 'Train with' })
  await helper.getByRole('radio', { name: 'Local model' }).check()
  await expect(page.getByRole('button', { name: 'Next' })).toBeDisabled()
  await helper.getByRole('radio', { name: 'Claude' }).check()
  await expect(page.getByRole('button', { name: 'Next' })).toBeEnabled()
})
