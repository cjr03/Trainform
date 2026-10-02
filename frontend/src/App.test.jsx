import { act, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it } from 'vitest'
import { clearToken, setToken } from './api'
import App from './App'

const renderAt = (path) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )

beforeEach(() => clearToken())

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
