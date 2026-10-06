import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import FeedPage from './FeedPage'

vi.mock('../api', () => ({ api: { feed: vi.fn(), vote: vi.fn() } }))

const post = (id) => ({ id, author: 'bea', name: `Agent ${id}`, base_model: 'claude', prompt: 'p', overall: null, score: 0, my_vote: 0, created_at: '2026-10-05T10:00:00Z' })

beforeEach(() => vi.resetAllMocks())

it('points to search when the feed is empty', async () => {
  api.feed.mockResolvedValue([])
  render(<FeedPage />, { wrapper: MemoryRouter })
  expect(await screen.findByRole('link', { name: 'Search for Trainformers' })).toHaveAttribute('href', '/search')
  expect(api.feed).toHaveBeenCalledWith(undefined, 20)
})

it('shows followed posts newest first, loads older pages and applies votes', async () => {
  const user = userEvent.setup()
  const first = Array.from({ length: 20 }, (_, i) => post(40 - i))
  api.feed.mockResolvedValueOnce(first).mockResolvedValueOnce([post(5)])
  api.vote.mockResolvedValue({ score: 1, my_vote: 1 })
  render(<FeedPage />, { wrapper: MemoryRouter })
  const articles = await screen.findAllByRole('article')
  expect(articles[0]).toHaveAccessibleName('Agent 40')
  await user.click(screen.getByRole('button', { name: 'Load more' }))
  expect(api.feed).toHaveBeenLastCalledWith(21, 20)
  expect(await screen.findByRole('article', { name: 'Agent 5' })).toBeInTheDocument()
  expect(screen.getAllByRole('article')).toHaveLength(21)
  expect(screen.queryByRole('button', { name: 'Load more' })).not.toBeInTheDocument()
  await user.click(screen.getAllByRole('button', { name: 'Upvote' })[0])
  expect(screen.getAllByRole('button', { name: 'Upvote' })[0]).toHaveAttribute('aria-pressed', 'true')
})

it('shows load errors', async () => {
  api.feed.mockRejectedValue(new Error('Server down'))
  render(<FeedPage />, { wrapper: MemoryRouter })
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
})
