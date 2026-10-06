import { render as rtlRender, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import { AgentImage, Avatar } from './shared'
import { FindAgent, FollowButton, PostCard, PublishDialog } from './social'

vi.mock('../api', () => ({
  api: {
    vote: vi.fn(),
    copyPost: vi.fn(),
    deletePost: vi.fn(),
    feed: vi.fn(),
    getProfile: vi.fn(),
    searchUsers: vi.fn(),
    listAgents: vi.fn(),
    publishPost: vi.fn(),
    follow: vi.fn(),
    unfollow: vi.fn(),
    avatar: vi.fn(),
  },
}))

// Radix sizes a radio inside a form with ResizeObserver, which jsdom lacks.
globalThis.ResizeObserver ??= class {
  observe() {}
  unobserve() {}
  disconnect() {}
}

const render = (ui) => rtlRender(ui, { wrapper: MemoryRouter })
const SVG = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><circle r="4"/></svg>'
const POST = {
  id: 7,
  author: 'bea',
  name: 'Fixer',
  base_model: 'qwen-0.5b',
  prompt: 'You fix bikes.',
  overall: 91,
  caption: 'My best one',
  score: 3,
  my_vote: 0,
  image: SVG,
  created_at: '2026-10-05T10:00:00Z',
}

beforeEach(() => {
  vi.resetAllMocks()
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})

it('renders agent SVG only as an <img> data URL, with the knight helmet as the fallback', () => {
  const { container } = render(
    <>
      <AgentImage svg={SVG} alt="Fixer image" />
      <AgentImage svg={null} />
    </>,
  )
  const img = screen.getByAltText('Fixer image')
  expect(img.getAttribute('src')).toBe(`data:image/svg+xml;utf8,${encodeURIComponent(SVG)}`)
  expect(container.querySelector('script')).toBeNull()
  expect(container.querySelectorAll('svg[aria-hidden="true"]')).toHaveLength(1)
})

it('shows a profile picture from an object URL, the user icon until then or without one', async () => {
  URL.createObjectURL = vi.fn(() => 'blob:pic')
  URL.revokeObjectURL = vi.fn()
  api.avatar.mockResolvedValue(new Blob(['png']))
  const { container, unmount } = render(
    <>
      <Avatar url="/social/users/bea/avatar" alt="Bea" />
      <Avatar url={null} />
    </>,
  )
  expect(container.querySelectorAll('svg')).toHaveLength(2)
  expect(await screen.findByAltText('Bea')).toHaveAttribute('src', 'blob:pic')
  expect(api.avatar).toHaveBeenCalledTimes(1)
  unmount()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:pic')
})

it('keeps the user icon when the picture fails to load', async () => {
  api.avatar.mockRejectedValue(new Error('404'))
  const { container } = render(<Avatar url="/x" />)
  await vi.waitFor(() => expect(api.avatar).toHaveBeenCalled())
  expect(container.querySelector('img')).toBeNull()
})

it('shows a post: author link, image, overall, caption, collapsible prompt and score', () => {
  render(<PostCard post={POST} onChange={() => {}} />)
  const card = screen.getByRole('article', { name: 'Fixer' })
  expect(within(card).getByRole('link', { name: 'bea' })).toHaveAttribute('href', '/u/bea')
  expect(within(card).getByRole('img', { name: 'Overall 91' })).toBeInTheDocument()
  expect(card).toHaveTextContent('qwen-0.5b')
  expect(card).toHaveTextContent('My best one')
  expect(within(card).getByText('You fix bikes.').closest('details')).not.toHaveAttribute('open')
  expect(within(card).getByRole('group', { name: 'Votes' })).toHaveTextContent('3')
  expect(card.querySelector('img[src^="data:image/svg+xml"]')).not.toBeNull()
})

it('votes up and down, toggling off my own vote, with aria-pressed on mine', async () => {
  const user = userEvent.setup()
  const onChange = vi.fn()
  api.vote.mockResolvedValue({ score: 4, my_vote: 1 })
  const { rerender } = render(<PostCard post={POST} onChange={onChange} />)
  const up = screen.getByRole('button', { name: 'Upvote' })
  expect(up).toHaveAttribute('aria-pressed', 'false')
  await user.click(up)
  expect(api.vote).toHaveBeenLastCalledWith(7, 1)
  expect(onChange).toHaveBeenLastCalledWith({ ...POST, score: 4, my_vote: 1 })

  rerender(<PostCard post={{ ...POST, score: 4, my_vote: 1 }} onChange={onChange} />)
  expect(screen.getByRole('button', { name: 'Upvote' })).toHaveAttribute('aria-pressed', 'true')
  api.vote.mockResolvedValue({ score: 3, my_vote: 0 })
  await user.click(screen.getByRole('button', { name: 'Upvote' }))
  expect(api.vote).toHaveBeenLastCalledWith(7, 0)
  await user.click(screen.getByRole('button', { name: 'Downvote' }))
  expect(api.vote).toHaveBeenLastCalledWith(7, -1)
})

it('shows a vote error', async () => {
  api.vote.mockRejectedValue(new Error("You can't vote on your own post"))
  render(<PostCard post={POST} onChange={() => {}} />)
  await userEvent.click(screen.getByRole('button', { name: 'Downvote' }))
  expect(await screen.findByRole('alert')).toHaveTextContent("You can't vote on your own post")
})

it('copies a post to my agents, noting a model that needs downloading', async () => {
  const onCopied = vi.fn()
  const copied = { id: 30, name: 'Fixer (2)', base_model: 'qwen-0.5b', model_available: false }
  api.copyPost.mockResolvedValue(copied)
  render(<PostCard post={POST} onChange={() => {}} onCopied={onCopied} />)
  await userEvent.click(screen.getByRole('button', { name: 'Copy to my agents' }))
  expect(api.copyPost).toHaveBeenCalledWith(7)
  expect(onCopied).toHaveBeenCalledWith(copied)
  const status = await screen.findByRole('status')
  expect(status).toHaveTextContent('Copied to your agents as Fixer (2). Download qwen-0.5b before training it.')
  expect(within(status).getByRole('link', { name: 'agents' })).toHaveAttribute('href', '/agents')
})

it('lets me delete my own post, after confirming, and not vote on it', async () => {
  const onDelete = vi.fn()
  api.deletePost.mockResolvedValue(null)
  render(<PostCard post={POST} mine onChange={() => {}} onDelete={onDelete} />)
  expect(screen.getByRole('button', { name: 'Upvote' })).toBeDisabled()
  expect(screen.queryByRole('button', { name: 'Copy to my agents' })).not.toBeInTheDocument()
  window.confirm.mockReturnValueOnce(false)
  await userEvent.click(screen.getByRole('button', { name: 'Delete post' }))
  expect(api.deletePost).not.toHaveBeenCalled()
  await userEvent.click(screen.getByRole('button', { name: 'Delete post' }))
  expect(api.deletePost).toHaveBeenCalledWith(7)
  expect(onDelete).toHaveBeenCalledWith(POST)
})

it('follows and unfollows, updating the follower count', async () => {
  const onChange = vi.fn()
  const bea = { username: 'bea', followers: 2, following: false }
  const { rerender } = render(<FollowButton user={bea} onChange={onChange} />)
  await userEvent.click(screen.getByRole('button', { name: 'Follow bea' }))
  expect(api.follow).toHaveBeenCalledWith('bea')
  expect(onChange).toHaveBeenLastCalledWith({ ...bea, following: true, followers: 3 })
  rerender(<FollowButton user={{ ...bea, following: true, followers: 3 }} onChange={onChange} />)
  await userEvent.click(screen.getByRole('button', { name: 'Unfollow bea' }))
  expect(api.unfollow).toHaveBeenCalledWith('bea')
  expect(onChange).toHaveBeenLastCalledWith({ ...bea, following: false, followers: 2 })
  api.follow.mockRejectedValue(new Error('Nope'))
  rerender(<FollowButton user={bea} onChange={onChange} />)
  await userEvent.click(screen.getByRole('button', { name: 'Follow bea' }))
  expect(await screen.findByText('Nope')).toBeInTheDocument()
})

it('finds an agent in my feed or from a searched Trainformer, and copies it', async () => {
  const user = userEvent.setup()
  const onCopied = vi.fn()
  api.feed.mockResolvedValue([POST])
  api.searchUsers.mockResolvedValue([{ username: 'cy', agents: 1, followers: 0, following: false, avatar_url: null }])
  api.getProfile.mockResolvedValue({ username: 'cy', posts: [{ ...POST, id: 8, author: 'cy', name: 'Writer' }] })
  api.copyPost.mockResolvedValue({ id: 31, name: 'Writer' })
  render(<FindAgent onCopied={onCopied} />)
  expect(await screen.findByRole('article', { name: 'Fixer' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'From Trainformers you follow' })).toBeInTheDocument()

  await user.type(screen.getByRole('searchbox', { name: 'Search Trainformers' }), ' cy ')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(api.searchUsers).toHaveBeenCalledWith('cy')
  await user.click(await screen.findByRole('button', { name: /cy · 1/ }))
  expect(api.getProfile).toHaveBeenCalledWith('cy')
  expect(await screen.findByRole('heading', { name: "cy's agents" })).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Upvote' }))
  await user.click(await screen.findByRole('button', { name: 'Use this agent' }))
  expect(api.copyPost).toHaveBeenCalledWith(8)
  expect(onCopied).toHaveBeenCalledWith({ id: 31, name: 'Writer' })

  api.feed.mockResolvedValue([])
  await user.click(screen.getByRole('button', { name: 'Back to your feed' }))
  expect(await screen.findByText(/Your feed is empty/)).toBeInTheDocument()
})

it('says when a search or a Trainformer has nothing, and shows load errors', async () => {
  const user = userEvent.setup()
  api.feed.mockRejectedValue(new Error('Server down'))
  api.searchUsers.mockResolvedValue([{ username: 'cy', agents: 0, followers: 0, following: false }])
  api.getProfile.mockResolvedValue({ username: 'cy', posts: [] })
  render(<FindAgent onCopied={() => {}} />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
  await user.type(screen.getByRole('searchbox'), 'cy')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  await user.click(await screen.findByRole('button', { name: /cy/ }))
  expect(await screen.findByText('No published agents yet.')).toBeInTheDocument()
  api.searchUsers.mockResolvedValue([])
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(await screen.findByText('No Trainformers match that name.')).toBeInTheDocument()
})

it('publishes one of my agents with a caption from the header dialog', async () => {
  const user = userEvent.setup()
  api.listAgents.mockResolvedValue([{ id: 4, name: 'Bike Repair Agent', overall: null, image: null }])
  api.publishPost.mockResolvedValue({ id: 9 })
  render(<PublishDialog username="ada" />)
  await user.click(screen.getByRole('button', { name: 'Publish an agent' }))
  const dialog = await screen.findByRole('dialog', { name: 'Publish an agent' })
  const publish = within(dialog).getByRole('button', { name: 'Publish' })
  expect(publish).toBeDisabled()
  await user.click(await within(dialog).findByRole('radio', { name: /Bike Repair Agent/ }))
  await user.type(within(dialog).getByLabelText('Caption (optional)'), '  Try it  ')
  await user.click(publish)
  expect(api.publishPost).toHaveBeenCalledWith(4, 'Try it')
  const link = await within(dialog).findByRole('link', { name: 'your profile' })
  expect(link).toHaveAttribute('href', '/u/ada')
  await user.click(link)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
})

it('publishes without a caption, and shows an empty list and errors', async () => {
  const user = userEvent.setup()
  api.listAgents.mockResolvedValue([])
  render(<PublishDialog username="ada" />)
  await user.click(screen.getByRole('button', { name: 'Publish an agent' }))
  expect(await screen.findByText(/No agents yet/)).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Cancel' }))

  api.listAgents.mockResolvedValue([{ id: 4, name: 'A', overall: 70, image: SVG }])
  api.publishPost.mockRejectedValue(new Error('Agent not found'))
  await user.click(screen.getByRole('button', { name: 'Publish an agent' }))
  await user.click(await screen.findByRole('radio', { name: /A/ }))
  await user.click(screen.getByRole('button', { name: 'Publish' }))
  expect(api.publishPost).toHaveBeenCalledWith(4, null)
  expect(await screen.findByRole('alert')).toHaveTextContent('Agent not found')
})
