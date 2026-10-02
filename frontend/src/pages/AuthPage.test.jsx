import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, expect, it, vi } from 'vitest'
import { api, ApiError } from '../api'
import AuthPage from './AuthPage'

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal()),
  api: { login: vi.fn(), signup: vi.fn() },
}))

const result = { token: 'tok', user: { id: 1, email: 'ada@x.io', username: 'ada', claude_linked: false } }

function setup() {
  const onAuth = vi.fn()
  render(<AuthPage onAuth={onAuth} />)
  return { onAuth, user: userEvent.setup() }
}

async function toSecurity(user) {
  await user.click(screen.getByRole('tab', { name: 'Create account' }))
  await user.type(screen.getByLabelText('Email'), 'ada@x.io')
  await user.type(screen.getByLabelText('Username'), 'ada')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
}

async function toLinkClaude(user) {
  await toSecurity(user)
  await user.type(screen.getByLabelText('Password'), 'hunter22')
  await user.type(screen.getByLabelText('Confirm password'), 'hunter22')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
}

const currentStep = () => screen.getByRole('list', { name: 'Sign-up progress' }).querySelector('[aria-current="step"]')

beforeEach(() => vi.clearAllMocks())

it('shows both options on one card', async () => {
  const { user } = setup()
  const page = screen.getByTestId('auth-page')
  const signIn = within(page).getByRole('tab', { name: 'Sign in' })
  const create = within(page).getByRole('tab', { name: 'Create account' })
  expect(signIn).toHaveAttribute('aria-selected', 'true')
  expect(signIn.closest('.card')).toBe(create.closest('.card'))
  await user.click(create)
  expect(create).toHaveAttribute('aria-selected', 'true')
  expect(screen.getByLabelText('Email')).toBeInTheDocument()
})

it('signs in and hands the result to onAuth', async () => {
  const { onAuth, user } = setup()
  api.login.mockResolvedValue(result)
  await user.type(screen.getByLabelText('Username'), 'ada')
  await user.type(screen.getByLabelText('Password'), 'hunter22')
  await user.click(screen.getByRole('button', { name: 'Sign in' }))
  expect(api.login).toHaveBeenCalledWith('ada', 'hunter22')
  expect(onAuth).toHaveBeenCalledWith(result)
})

it('shows the server error when sign-in fails and disables the button while pending', async () => {
  const { onAuth, user } = setup()
  let reject
  api.login.mockReturnValue(new Promise((_, r) => (reject = r)))
  await user.type(screen.getByLabelText('Username'), 'ada')
  await user.type(screen.getByLabelText('Password'), 'wrong-pass')
  await user.click(screen.getByRole('button', { name: 'Sign in' }))
  expect(screen.getByRole('button', { name: /Sign in/ })).toBeDisabled()
  reject(new ApiError(401, 'Invalid username or password'))
  expect(await screen.findByRole('alert')).toHaveTextContent('Invalid username or password')
  expect(screen.getByRole('button', { name: 'Sign in' })).toBeEnabled()
  expect(onAuth).not.toHaveBeenCalled()
})

it('creates an account through all phases with a Claude key', async () => {
  const { onAuth, user } = setup()
  api.signup.mockResolvedValue(result)
  await toLinkClaude(user)
  expect(currentStep()).toHaveTextContent('Link Claude')
  expect(screen.getByText(/your own Claude key/i)).toBeInTheDocument()
  const key = screen.getByLabelText('Anthropic API key')
  expect(key).toHaveAttribute('type', 'password')
  await user.type(key, 'sk-ant-123')
  await user.click(screen.getByRole('button', { name: 'Link & create account' }))
  expect(api.signup).toHaveBeenCalledWith({
    email: 'ada@x.io',
    username: 'ada',
    password: 'hunter22',
    password_confirm: 'hunter22',
    anthropic_api_key: 'sk-ant-123',
  })
  expect(onAuth).toHaveBeenCalledWith(result)
})

it('creates an account without a key when skipping', async () => {
  const { onAuth, user } = setup()
  api.signup.mockResolvedValue(result)
  await toLinkClaude(user)
  expect(screen.getByRole('button', { name: 'Link & create account' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: 'Skip for now' }))
  expect(api.signup).toHaveBeenCalledWith(expect.objectContaining({ anthropic_api_key: null }))
  expect(onAuth).toHaveBeenCalledWith(result)
})

it('blocks progress until identity is valid', async () => {
  const { user } = setup()
  await user.click(screen.getByRole('tab', { name: 'Create account' }))
  await user.type(screen.getByLabelText('Email'), 'not-an-email')
  await user.type(screen.getByLabelText('Username'), 'ada')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByRole('alert')).toHaveTextContent(/valid email/i)
  expect(currentStep()).toHaveTextContent('Identity')
})

it('blocks progress on short or mismatched passwords', async () => {
  const { user } = setup()
  await toSecurity(user)
  await user.type(screen.getByLabelText('Password'), 'short')
  await user.type(screen.getByLabelText('Confirm password'), 'short')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByRole('alert')).toHaveTextContent(/at least 8/i)
  await user.clear(screen.getByLabelText('Password'))
  await user.type(screen.getByLabelText('Password'), 'hunter22')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByRole('alert')).toHaveTextContent(/do not match/i)
  expect(currentStep()).toHaveTextContent('Security')
})

it('keeps entered values when going back', async () => {
  const { user } = setup()
  await toLinkClaude(user)
  await user.type(screen.getByLabelText('Anthropic API key'), 'sk-ant-1')
  await user.click(screen.getByRole('button', { name: 'Back' }))
  expect(screen.getByLabelText('Password')).toHaveValue('hunter22')
  expect(screen.getByLabelText('Confirm password')).toHaveValue('hunter22')
  await user.click(screen.getByRole('button', { name: 'Back' }))
  expect(screen.getByLabelText('Email')).toHaveValue('ada@x.io')
  expect(screen.getByLabelText('Username')).toHaveValue('ada')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByLabelText('Anthropic API key')).toHaveValue('sk-ant-1')
})

it('returns to identity when the username or email is taken', async () => {
  const { onAuth, user } = setup()
  api.signup.mockRejectedValue(new ApiError(409, 'Username already taken'))
  await toLinkClaude(user)
  await user.click(screen.getByRole('button', { name: 'Skip for now' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Username already taken')
  expect(currentStep()).toHaveTextContent('Identity')
  expect(screen.getByLabelText('Username')).toHaveValue('ada')
  expect(onAuth).not.toHaveBeenCalled()
})

it('stays on the last phase for other server errors', async () => {
  const { user } = setup()
  api.signup.mockRejectedValue(new ApiError(400, 'Anthropic API key was rejected'))
  await toLinkClaude(user)
  await user.type(screen.getByLabelText('Anthropic API key'), 'bad')
  await user.click(screen.getByRole('button', { name: 'Link & create account' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Anthropic API key was rejected')
  expect(currentStep()).toHaveTextContent('Link Claude')
})

it('switches tabs with the arrow keys and keeps focus on the tab', async () => {
  const { user } = setup()
  const signIn = screen.getByRole('tab', { name: 'Sign in' })
  const create = screen.getByRole('tab', { name: 'Create account' })
  signIn.focus()
  await user.keyboard('{ArrowRight}')
  expect(create).toHaveAttribute('aria-selected', 'true')
  expect(create).toHaveFocus()
  await user.keyboard('{ArrowLeft}')
  expect(signIn).toHaveAttribute('aria-selected', 'true')
  expect(signIn).toHaveFocus()
})

it('blocks usernames the server would reject', async () => {
  const { user } = setup()
  await user.click(screen.getByRole('tab', { name: 'Create account' }))
  await user.type(screen.getByLabelText('Email'), 'ada@x.io')
  await user.type(screen.getByLabelText('Username'), 'a b')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByRole('alert')).toHaveTextContent(/username/i)
  expect(currentStep()).toHaveTextContent('Identity')
  expect(screen.getByLabelText('Username')).toHaveAttribute('maxLength', '32')
})

it('caps signup passwords at the server limit', async () => {
  const { user } = setup()
  await toSecurity(user)
  expect(screen.getByLabelText('Password')).toHaveAttribute('maxLength', '128')
  expect(screen.getByLabelText('Confirm password')).toHaveAttribute('maxLength', '128')
})

it('returns to identity on an email conflict or an invalid email', async () => {
  const { user } = setup()
  api.signup.mockRejectedValueOnce(new ApiError(409, 'Email already registered'))
  await toLinkClaude(user)
  await user.click(screen.getByRole('button', { name: 'Skip for now' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Email already registered')
  expect(currentStep()).toHaveTextContent('Identity')

  api.signup.mockRejectedValueOnce(new ApiError(422, 'value is not a valid email address: It should have a period.'))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.click(screen.getByRole('button', { name: 'Skip for now' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('not a valid email')
  expect(currentStep()).toHaveTextContent('Identity')
})

it('disables both final buttons while signup is pending', async () => {
  const { user } = setup()
  api.signup.mockReturnValue(new Promise(() => {}))
  await toLinkClaude(user)
  await user.type(screen.getByLabelText('Anthropic API key'), 'sk-ant-1')
  await user.click(screen.getByRole('button', { name: 'Link & create account' }))
  expect(screen.getByRole('button', { name: 'Skip for now' })).toBeDisabled()
  expect(screen.getByRole('button', { name: /Link & create account/ })).toBeDisabled()
})

it('does not carry a sign-in password into signup', async () => {
  const { user } = setup()
  await user.type(screen.getByLabelText('Password'), 'other-account-pw')
  await toSecurity(user)
  expect(screen.getByLabelText('Password')).toHaveValue('')
})

it('treats a whitespace-only key as no key', async () => {
  const { user } = setup()
  api.signup.mockResolvedValue(result)
  await toLinkClaude(user)
  await user.type(screen.getByLabelText('Anthropic API key'), '   ')
  expect(screen.getByRole('button', { name: 'Link & create account' })).toBeDisabled()
  await user.type(screen.getByLabelText('Anthropic API key'), ' sk-ant-1 ')
  await user.click(screen.getByRole('button', { name: 'Link & create account' }))
  expect(api.signup).toHaveBeenCalledWith(expect.objectContaining({ anthropic_api_key: 'sk-ant-1' }))
})
