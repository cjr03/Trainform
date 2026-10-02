import { act, render, screen } from '@testing-library/react'
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
  for (const name of ['Train', 'Inspect', 'Account']) {
    expect(screen.getByRole('link', { name })).toBeInTheDocument()
  }
})

it.each([
  ['/inspect', 'inspect-page'],
  ['/account', 'account-page'],
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

it('clears the token on Sign out', async () => {
  setToken('tok')
  renderAt('/train')
  await userEvent.click(screen.getByRole('button', { name: 'Sign out' }))
  expect(screen.getByTestId('auth-page')).toBeInTheDocument()
  expect(getToken()).toBeNull()
})
