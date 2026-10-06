import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { clearToken, getToken, setToken } from './api'
import App from './App'

const renderAt = (path) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )

beforeEach(() => clearToken())
afterEach(() => vi.unstubAllGlobals())

it('shows the auth card when signed out, whatever the URL', () => {
  renderAt('/inspect')
  expect(screen.getByTestId('auth-page')).toBeInTheDocument()
  expect(screen.queryByRole('navigation')).not.toBeInTheDocument()
})

it('routes signed-in users and defaults to Train', () => {
  setToken('tok')
  renderAt('/')
  expect(screen.getByRole('navigation')).toBeInTheDocument()
  expect(screen.getByTestId('train-page')).toBeInTheDocument()
  for (const name of ['Train', 'Agents', 'Inspect', 'Feed', 'Search Trainformers']) {
    expect(screen.getByRole('link', { name })).toBeInTheDocument()
  }
  for (const name of ['Publish an agent', 'Account menu']) {
    expect(screen.getByRole('button', { name })).toBeInTheDocument()
  }
  expect(screen.queryByRole('button', { name: 'Sign out' })).not.toBeInTheDocument()
})

it('shows the tabs as icons in order, each named on hover and on keyboard focus', async () => {
  setToken('tok')
  renderAt('/train')
  const tabs = screen.getAllByRole('link').filter((l) => l.closest('nav'))
  expect(tabs.map((l) => l.getAttribute('aria-label'))).toEqual(['Agents', 'Inspect', 'Train', 'Battle'])
  expect(tabs.map((l) => l.textContent)).toEqual(['', '', '', '']) // icons only
  expect(screen.getByRole('link', { name: 'Train' })).toHaveAttribute('aria-current', 'page')
  const user = userEvent.setup()
  await user.hover(screen.getByRole('link', { name: 'Battle' }))
  expect(await screen.findByRole('tooltip')).toHaveTextContent('Battle')
  await user.unhover(screen.getByRole('link', { name: 'Battle' }))
  await user.tab() // the first control in the header: the Agents tab
  expect(await screen.findByRole('tooltip')).toHaveTextContent('Agents')
})

it('opens the account menu: your profile, Account and Sign out', async () => {
  const json = (body) => new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })
  const bodies = {
    '/api/me': { username: 'ada', avatar_url: null },
    '/api/social/users/ada': { username: 'ada', followers: 0, following_count: 0, following: false, posts: [] },
  }
  vi.stubGlobal('fetch', vi.fn(async (url) => json(bodies[url] ?? [])))
  setToken('tok')
  renderAt('/train')
  const user = userEvent.setup()
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/me', expect.anything()))
  await user.click(screen.getByRole('button', { name: 'Account menu' }))
  await user.click(await screen.findByRole('menuitem', { name: 'Account' }))
  expect(await screen.findByTestId('account-page')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Account menu' }))
  await user.click(await screen.findByRole('menuitem', { name: 'ada' }))
  expect(await screen.findByTestId('profile-page')).toBeInTheDocument()
})

it.each([
  ['/agents', 'agents-page'],
  ['/inspect', 'inspect-page'],
  ['/account', 'account-page'],
  ['/feed', 'feed-page'],
  ['/search', 'search-page'],
])('renders %s', (path, testId) => {
  setToken('tok')
  renderAt(path)
  expect(screen.getByTestId(testId)).toBeInTheDocument()
})

it('returns to the auth card on a logout event', () => {
  setToken('tok')
  renderAt('/train')
  act(() => window.dispatchEvent(new Event('trainform:logout')))
  expect(screen.getByTestId('auth-page')).toBeInTheDocument()
})

it('stores the token and goes to Train after signing in', async () => {
  const json = (body) => new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })
  vi.stubGlobal('fetch', vi.fn(async (url) => json(url === '/api/auth/login' ? { token: 'new-tok' } : [])))
  renderAt('/account')
  await userEvent.type(screen.getByLabelText('Username'), 'ada')
  await userEvent.type(screen.getByLabelText('Password'), 'secret-pw')
  await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
  expect(await screen.findByTestId('train-page')).toBeInTheDocument()
  expect(getToken()).toBe('new-tok')
})

it("starts from a fresh Train page after signing in, not the previous user's draft", async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 1, prompt: 'previous user prompt' }))
  const json = (body) => new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })
  vi.stubGlobal('fetch', vi.fn(async (url) => json(url === '/api/auth/login' ? { token: 'new-tok' } : [])))
  renderAt('/train')
  await userEvent.type(screen.getByLabelText('Username'), 'ada')
  await userEvent.type(screen.getByLabelText('Password'), 'secret-pw')
  await userEvent.click(screen.getByRole('button', { name: 'Sign in' }))
  expect(await screen.findByTestId('train-page')).toBeInTheDocument()
  expect(sessionStorage.getItem('trainform_train_draft')).not.toContain('previous user prompt')
})

it('signs out even when storage is blocked', () => {
  setToken('tok')
  vi.spyOn(Storage.prototype, 'clear').mockImplementationOnce(() => {
    throw new DOMException('blocked', 'SecurityError')
  })
  expect(() => clearToken()).not.toThrow()
  expect(getToken()).toBeNull()
})

it('clears the token and the Train draft on Sign out', async () => {
  setToken('tok')
  renderAt('/train')
  expect(sessionStorage.getItem('trainform_train_draft')).not.toBeNull()
  await userEvent.click(screen.getByRole('button', { name: 'Account menu' }))
  await userEvent.click(await screen.findByRole('menuitem', { name: 'Sign out' }))
  expect(screen.getByTestId('auth-page')).toBeInTheDocument()
  expect(getToken()).toBeNull()
  expect(sessionStorage.getItem('trainform_train_draft')).toBeNull()
})
