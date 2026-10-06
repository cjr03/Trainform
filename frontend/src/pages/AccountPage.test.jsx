import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError } from '../api'
import AccountPage from './AccountPage'

vi.mock('../api', async (importOriginal) => ({
  ...(await importOriginal()),
  api: {
    me: vi.fn(),
    changePassword: vi.fn(),
    setClaudeKey: vi.fn(),
    removeClaudeKey: vi.fn(),
    avatar: vi.fn(),
    generateAvatar: vi.fn(),
    uploadAvatar: vi.fn(),
    removeAvatar: vi.fn(),
  },
}))

const ada = { id: 1, email: 'ada@x.io', username: 'ada', claude_linked: false }

async function setup(me = ada) {
  api.me.mockResolvedValue(me)
  render(<AccountPage />)
  const user = userEvent.setup()
  await screen.findByText('ada@x.io')
  return user
}

const claude = () => screen.getByRole('region', { name: 'Claude linkage' })

async function fillPassword(user, current, next, confirm) {
  await user.type(screen.getByLabelText('Current password'), current)
  await user.type(screen.getByLabelText('New password'), next)
  await user.type(screen.getByLabelText('Confirm new password'), confirm)
  await user.click(screen.getByRole('button', { name: 'Change password' }))
}

beforeEach(() => vi.clearAllMocks())

it('shows a loading state, then the account with a masked password', async () => {
  api.me.mockResolvedValue(ada)
  render(<AccountPage />)
  expect(screen.getByTestId('account-page')).toBeInTheDocument()
  expect(screen.getByRole('status')).toHaveTextContent(/loading/i)
  expect(await screen.findByText('ada@x.io')).toBeInTheDocument()
  expect(screen.getByText('ada')).toBeInTheDocument()
  expect(screen.getByText('••••••••')).toBeInTheDocument()
  expect(screen.getByText(/stored hashed/i)).toBeInTheDocument()
})

it('shows an error when the account fails to load', async () => {
  api.me.mockRejectedValue(new ApiError(500, 'Database unavailable'))
  render(<AccountPage />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Database unavailable')
})

it('changes the password and clears the form', async () => {
  const user = await setup()
  api.changePassword.mockResolvedValue(null)
  await fillPassword(user, 'old-pass1', 'new-pass1', 'new-pass1')
  expect(api.changePassword).toHaveBeenCalledWith('old-pass1', 'new-pass1', 'new-pass1')
  expect(await screen.findByText('Password changed.')).toBeInTheDocument()
  for (const label of ['Current password', 'New password', 'Confirm new password']) {
    expect(screen.getByLabelText(label)).toHaveValue('')
  }
})

it('blocks mismatched new passwords', async () => {
  const user = await setup()
  await fillPassword(user, 'old-pass1', 'new-pass1', 'new-pass2')
  expect(screen.getByRole('alert')).toHaveTextContent(/do not match/i)
  expect(api.changePassword).not.toHaveBeenCalled()
})

it('shows the server error when the password change fails', async () => {
  const user = await setup()
  api.changePassword.mockRejectedValue(new ApiError(400, 'Current password is incorrect'))
  await fillPassword(user, 'bad-pass1', 'new-pass1', 'new-pass1')
  expect(await screen.findByRole('alert')).toHaveTextContent('Current password is incorrect')
  expect(screen.getByLabelText('Current password')).toHaveValue('bad-pass1')
})

it('links a Claude key and updates the badge', async () => {
  const user = await setup()
  api.setClaudeKey.mockResolvedValue({ ...ada, claude_linked: true })
  expect(within(claude()).getByText('Not linked')).toBeInTheDocument()
  const key = within(claude()).getByLabelText('Anthropic API key')
  expect(key).toHaveAttribute('type', 'password')
  await user.type(key, 'sk-ant-123')
  await user.click(within(claude()).getByRole('button', { name: 'Link key' }))
  expect(api.setClaudeKey).toHaveBeenCalledWith('sk-ant-123')
  expect(await within(claude()).findByText('Linked')).toBeInTheDocument()
  expect(within(claude()).getByRole('status')).toHaveTextContent('Key saved.')
  expect(key).toHaveValue('')
})

it('ignores a whitespace-only key', async () => {
  const user = await setup()
  await user.type(within(claude()).getByLabelText('Anthropic API key'), '   ')
  expect(within(claude()).getByRole('button', { name: 'Link key' })).toBeDisabled()
})

it('unlinks the Claude key and updates the badge', async () => {
  const user = await setup({ ...ada, claude_linked: true })
  api.removeClaudeKey.mockResolvedValue(ada)
  expect(within(claude()).getByText('Linked')).toBeInTheDocument()
  expect(within(claude()).getByRole('button', { name: 'Replace key' })).toBeInTheDocument()
  await user.click(within(claude()).getByRole('button', { name: 'Unlink' }))
  expect(api.removeClaudeKey).toHaveBeenCalled()
  expect(await within(claude()).findByText('Not linked')).toBeInTheDocument()
  expect(within(claude()).getByRole('status')).toHaveTextContent('Key removed.')
  expect(within(claude()).queryByRole('button', { name: 'Unlink' })).not.toBeInTheDocument()
})

it('shows the server error when linking fails', async () => {
  const user = await setup()
  api.setClaudeKey.mockRejectedValue(new ApiError(400, 'Anthropic API key was rejected'))
  await user.type(within(claude()).getByLabelText('Anthropic API key'), 'bad')
  await user.click(within(claude()).getByRole('button', { name: 'Link key' }))
  expect(await within(claude()).findByRole('alert')).toHaveTextContent('Anthropic API key was rejected')
  expect(within(claude()).getByText('Not linked')).toBeInTheDocument()
})

it('explains what linking Claude adds and that training works without it', async () => {
  await setup()
  expect(within(claude()).getByText(/stronger prompts, rubrics, personas and grading/i)).toBeInTheDocument()
  expect(within(claude()).getByText(/training still works, with the helpers running locally/i)).toBeInTheDocument()
  expect(within(claude()).getByRole('link', { name: /anthropic console/i })).toHaveAttribute(
    'href',
    'https://console.anthropic.com/settings/keys',
  )
})

describe('profile picture', () => {
  const picture = () => screen.getByRole('region', { name: 'Profile picture' })

  beforeEach(() => {
    URL.createObjectURL = vi.fn(() => 'blob:pic')
    URL.revokeObjectURL = vi.fn()
    api.avatar.mockResolvedValue(new Blob(['png']))
  })
  afterEach(() => {
    vi.restoreAllMocks()
    delete HTMLImageElement.prototype.decode
  })

  it('generates one, shows it here and hands it to the header', async () => {
    api.me.mockResolvedValue(ada)
    api.generateAvatar.mockResolvedValue({ avatar_url: '/social/users/ada/avatar?v=2' })
    const onUser = vi.fn()
    render(<AccountPage onUser={onUser} />)
    const user = userEvent.setup()
    await screen.findByText('ada@x.io')
    expect(within(picture()).queryByRole('button', { name: 'Remove' })).toBeNull()
    await user.click(within(picture()).getByRole('button', { name: 'Generate' }))
    expect(api.generateAvatar).toHaveBeenCalled()
    expect(await within(picture()).findByAltText('Your profile picture')).toHaveAttribute('src', 'blob:pic')
    expect(api.avatar).toHaveBeenCalledWith('/social/users/ada/avatar?v=2')
    expect(onUser).toHaveBeenCalledWith(expect.objectContaining({ username: 'ada', avatar_url: '/social/users/ada/avatar?v=2' }))
  })

  it('uploads a picked image downscaled to a 256x256 PNG', async () => {
    const drawImage = vi.fn()
    HTMLImageElement.prototype.decode = vi.fn(async () => {}) // jsdom doesn't decode images
    vi.spyOn(HTMLImageElement.prototype, 'naturalWidth', 'get').mockReturnValue(800)
    vi.spyOn(HTMLImageElement.prototype, 'naturalHeight', 'get').mockReturnValue(600)
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage })
    const png = new Blob(['png'], { type: 'image/png' })
    vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(function (cb, type) {
      expect([this.width, this.height, type]).toEqual([256, 256, 'image/png'])
      cb(png)
    })
    api.uploadAvatar.mockResolvedValue({ avatar_url: '/social/users/ada/avatar' })
    const user = await setup()
    await user.upload(within(picture()).getByLabelText('Picture file'), new File(['x'], 'me.jpg', { type: 'image/jpeg' }))
    expect(drawImage).toHaveBeenCalledWith(expect.anything(), 100, 0, 600, 600, 0, 0, 256, 256)
    expect(api.uploadAvatar).toHaveBeenCalledWith(png)
    expect(await within(picture()).findByAltText('Your profile picture')).toBeInTheDocument()
  })

  it("rejects a file that isn't an image", async () => {
    HTMLImageElement.prototype.decode = vi.fn(async () => {
      throw new Error('decode')
    })
    const user = await setup()
    await user.click(within(picture()).getByRole('button', { name: 'Upload' }))
    await user.upload(within(picture()).getByLabelText('Picture file'), new File(['<html>'], 'x.png', { type: 'image/png' }))
    expect(await within(picture()).findByRole('alert')).toHaveTextContent("That file isn't an image we can read.")
    expect(api.uploadAvatar).not.toHaveBeenCalled()
  })

  it('removes it, back to the user icon', async () => {
    api.removeAvatar.mockResolvedValue(null)
    const user = await setup({ ...ada, avatar_url: '/social/users/ada/avatar' })
    expect(await within(picture()).findByAltText('Your profile picture')).toBeInTheDocument()
    await user.click(within(picture()).getByRole('button', { name: 'Remove' }))
    expect(api.removeAvatar).toHaveBeenCalled()
    await waitFor(() => expect(within(picture()).queryByRole('img')).toBeNull())
    expect(within(picture()).queryByRole('button', { name: 'Remove' })).toBeNull()
  })
})
