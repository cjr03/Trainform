import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import ProfilePage from './ProfilePage'

vi.mock('../api', () => ({
  api: { getProfile: vi.fn(), follow: vi.fn(), unfollow: vi.fn(), vote: vi.fn(), deletePost: vi.fn(), avatar: vi.fn() },
}))

const POST = { id: 3, author: 'bea', name: 'Fixer', base_model: 'claude', prompt: 'p', overall: 80, score: 2, my_vote: 0, created_at: '2026-10-05T10:00:00Z' }
const profile = (over = {}) => ({ username: 'bea', followers: 1, following_count: 4, following: false, avatar_url: null, posts: [POST], ...over })
const renderAt = (username, me = 'ada') =>
  render(
    <MemoryRouter initialEntries={[`/u/${username}`]}>
      <Routes>
        <Route path="/u/:username" element={<ProfilePage me={me} />} />
      </Routes>
    </MemoryRouter>,
  )

beforeEach(() => {
  vi.resetAllMocks()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

it("shows a Trainformer's counts and posts, and follows them", async () => {
  const user = userEvent.setup()
  api.getProfile.mockResolvedValue(profile())
  api.vote.mockResolvedValue({ score: 3, my_vote: 1 })
  renderAt('bea')
  expect(await screen.findByRole('heading', { level: 1, name: 'bea' })).toBeInTheDocument()
  expect(screen.getByText('1 follower · 4 following')).toBeInTheDocument()
  expect(screen.getByRole('article', { name: 'Fixer' })).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Follow bea' }))
  expect(api.follow).toHaveBeenCalledWith('bea')
  expect(screen.getByText('2 followers · 4 following')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Upvote' }))
  expect(screen.getByRole('group', { name: 'Votes' })).toHaveTextContent('3')
})

it('on my own profile: no follow button, and I can delete my posts', async () => {
  api.getProfile.mockResolvedValue(profile({ username: 'ada', posts: [{ ...POST, author: 'ada', agent_id: 1 }] }))
  renderAt('ada')
  await userEvent.click(await screen.findByRole('button', { name: 'Delete post' }))
  expect(api.deletePost).toHaveBeenCalledWith(3)
  expect(await screen.findByText('No published agents yet.')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /Follow/ })).not.toBeInTheDocument()
})

it('shows an unknown Trainformer as an error', async () => {
  api.getProfile.mockRejectedValue(new Error('Trainformer not found'))
  renderAt('nobody')
  expect(await screen.findByRole('alert')).toHaveTextContent('Trainformer not found')
})
