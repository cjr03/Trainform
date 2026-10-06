import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import SearchPage from './SearchPage'

vi.mock('../api', () => ({ api: { searchUsers: vi.fn(), follow: vi.fn(), unfollow: vi.fn(), avatar: vi.fn() } }))

beforeEach(() => vi.resetAllMocks())

it('searches Trainformers and follows one from the results', async () => {
  const user = userEvent.setup()
  api.searchUsers.mockResolvedValue([
    { username: 'bea', agents: 1, followers: 0, following: false, avatar_url: null },
    { username: 'cy', agents: 2, followers: 1, following: true, avatar_url: null },
  ])
  render(<SearchPage />, { wrapper: MemoryRouter })
  expect(screen.getByRole('button', { name: 'Search' })).toBeDisabled()
  await user.type(screen.getByRole('searchbox', { name: 'Search Trainformers' }), ' b ')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(api.searchUsers).toHaveBeenCalledWith('b')
  const list = await screen.findByRole('list', { name: 'Trainformers' })
  const [bea, cy] = within(list).getAllByRole('listitem')
  expect(within(bea).getByRole('link', { name: 'bea' })).toHaveAttribute('href', '/u/bea')
  expect(bea).toHaveTextContent('1 agent · 0 followers')
  expect(cy).toHaveTextContent('2 agents · 1 follower')
  await user.click(within(bea).getByRole('button', { name: 'Follow bea' }))
  expect(api.follow).toHaveBeenCalledWith('bea')
  expect(bea).toHaveTextContent('1 follower')
  expect(within(bea).getByRole('button', { name: 'Unfollow bea' })).toBeInTheDocument()
})

it('says when nobody matches, and shows errors', async () => {
  const user = userEvent.setup()
  api.searchUsers.mockResolvedValueOnce([]).mockRejectedValueOnce(new Error('Server down'))
  render(<SearchPage />, { wrapper: MemoryRouter })
  await user.type(screen.getByRole('searchbox'), 'zz{Enter}')
  expect(await screen.findByText('No Trainformers match that name.')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
})
