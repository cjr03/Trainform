import { act, render as rtlRender, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import AgentsPage from './AgentsPage'

vi.mock('../api', () => ({
  api: {
    me: vi.fn(),
    listAgents: vi.fn(),
    getAgent: vi.fn(),
    createAgent: vi.fn(),
    updateAgent: vi.fn(),
    deleteAgent: vi.fn(),
    quantizeAgent: vi.fn(),
    listLocalModels: vi.fn(),
    searchHubModels: vi.fn(),
    downloadHubModel: vi.fn(),
    uploadModel: vi.fn(),
    generatePrompt: vi.fn(),
    generateAgentImage: vi.fn(),
    setAgentImage: vi.fn(),
    feed: vi.fn(),
    copyPost: vi.fn(),
    listTrainings: vi.fn(),
    getTraining: vi.fn(),
  },
}))

const render = (ui) => rtlRender(ui, { wrapper: MemoryRouter })

const PROMPT = 'You are the support assistant.\n\n  - Be friendly.'
const AGENTS = [
  { id: 1, name: 'Bike Repair Agent', base_model: 'qwen-0.5b', prompt: PROMPT, overall: 84, trainings: 2 },
  { id: 2, name: 'Claude Writer', base_model: 'claude', prompt: 'Write well.', overall: null, trainings: 0 },
]
const TRAININGS = [
  { id: 9, status: 'completed', created_at: '2026-10-02T10:00:00Z', rounds: 3, scenarios_per_round: 5, best_score: 8.4, baseline_score: 6 },
  { id: 8, status: 'failed', created_at: '2026-10-01T10:00:00Z', rounds: 2, scenarios_per_round: 4, best_score: null, baseline_score: null },
]
const agent = (over = {}) => ({ ...AGENTS[0], trainings: TRAININGS, ...over })
const HUB_MODEL = { id: 'Qwen/Qwen2.5-0.5B', downloads: 512, likes: 34, last_modified: '2026-09-01T00:00:00Z' }

beforeEach(() => {
  vi.resetAllMocks()
  api.me.mockResolvedValue({ claude_available: true })
  api.listAgents.mockResolvedValue(AGENTS)
  api.getAgent.mockResolvedValue(agent())
  api.listLocalModels.mockResolvedValue([{ name: 'qwen-0.5b', source: 'hub' }])
  api.listTrainings.mockResolvedValue([])
})

const card = (name) => screen.findByRole('button', { name: new RegExp(name) })
const openNew = async (user) => {
  render(<AgentsPage />)
  await user.click(screen.getByRole('button', { name: 'New agent' }))
}

it('lists the agents with a Madden-style overall, Unranked before any training', async () => {
  render(<AgentsPage />)
  expect(screen.getByRole('heading', { level: 1, name: 'Agents' })).toHaveClass('text-center')
  const bike = await card('Bike Repair Agent')
  expect(within(bike).getByRole('img', { name: 'Overall 84' })).toHaveTextContent('84OVR')
  expect(bike).toHaveTextContent('qwen-0.5b · 2 trainings')
  const writer = await card('Claude Writer')
  expect(within(writer).getByRole('img', { name: 'Unranked' })).toHaveTextContent('--')
  expect(writer).toHaveTextContent('Claude · 0 trainings')
})

it('shows the empty state and load errors', async () => {
  api.listAgents.mockResolvedValue([])
  const view = render(<AgentsPage />)
  expect(await screen.findByText('No agents yet. Make your first one.')).toBeInTheDocument()
  view.unmount()
  api.listAgents.mockRejectedValue(new Error('Server down'))
  render(<AgentsPage />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
})

it('shows the selected agent: its full prompt, past trainings and a link to Inspect', async () => {
  const user = userEvent.setup()
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  expect(api.getAgent).toHaveBeenCalledWith(1)
  expect(await card('Bike Repair Agent')).toHaveAttribute('aria-pressed', 'true')
  const detail = await screen.findByRole('region', { name: 'Bike Repair Agent' })
  const overall = within(detail).getByRole('img', { name: 'Overall 84' })
  const name = within(detail).getByRole('heading', { name: 'Bike Repair Agent' })
  expect(name.compareDocumentPosition(overall) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy() // right of the name
  const pre = within(detail).getByText((_, el) => el.tagName === 'PRE')
  expect(pre.textContent).toBe(PROMPT) // whitespace kept
  expect(pre).toHaveClass('whitespace-pre-wrap')
  const past = within(within(detail).getByRole('region', { name: 'Past trainings' })).getAllByRole('listitem')
  expect(past[0]).toHaveTextContent('#9')
  expect(past[0]).toHaveTextContent('completed')
  expect(past[0]).toHaveTextContent('8.4')
  expect(past[0]).toHaveTextContent('3 rounds × 5 scenarios')
  expect(past[1]).toHaveTextContent('failed')
  expect(past[1]).toHaveTextContent('—')
  expect(within(detail).getByRole('link', { name: 'Open in Inspect' })).toHaveAttribute('href', '/inspect?agent=1')
})

it("shows the agent's past battles and opens one", async () => {
  const user = userEvent.setup()
  const battle = { rounds: 3, winner: 'challenger', grades: [6, 8], round_grades: [[6, 9], [7, 8], [5, 7]] }
  api.listTrainings.mockResolvedValue([
    { id: 12, status: 'completed', agent_name: 'Bike Repair Agent', created_at: '2026-10-03T10:00:00Z', rounds: 2, battle },
    { id: 9, status: 'completed', battle: null },
  ])
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  const past = within(await screen.findByRole('region', { name: 'Past battles' }))
  expect(api.listTrainings).toHaveBeenCalledWith(1)
  const row = past.getByRole('button')
  expect(row).toHaveTextContent('3 battle rounds · challenger trained for 2 rounds · challenger won')

  const bouts = [9, 8, 7].map((g) => ({ message: 'Hi', champion: 'Hello!', challenger: 'Hey.', grades: [6, g] }))
  api.getTraining.mockResolvedValue({ id: 12, rounds: 2, battle: { ...battle, bouts } })
  await user.click(row)
  expect(api.getTraining).toHaveBeenCalledWith(12)
  expect(await screen.findByTestId('rounds-won')).toHaveTextContent('Rounds won 0 – 3')
  expect(screen.getByText('Round 2: 7.0 – 8.0')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Close battle' }))
  expect(screen.queryByTestId('battle-score')).not.toBeInTheDocument()
})

it('says when an agent has no battles yet', async () => {
  const user = userEvent.setup()
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  const past = within(await screen.findByRole('region', { name: 'Past battles' }))
  expect(await past.findByText(/No battles yet/)).toBeInTheDocument()
  expect(past.getByRole('link', { name: 'Battle page' })).toHaveAttribute('href', '/battle')
})

it('notes that a Claude agent only trains its prompt, and links an untrained agent to Train', async () => {
  api.getAgent.mockResolvedValue({ ...AGENTS[1], trainings: [] })
  const user = userEvent.setup()
  render(<AgentsPage />)
  await user.click(await card('Claude Writer'))
  const detail = await screen.findByRole('region', { name: 'Claude Writer' })
  expect(detail).toHaveTextContent("Claude's weights can't be fine-tuned, so training will only improve its prompt.")
  expect(within(detail).getByRole('img', { name: 'Unranked' })).toBeInTheDocument()
  expect(within(detail).getByRole('link', { name: 'Train page' })).toHaveAttribute('href', '/train')
})

it('shows why an agent could not be loaded', async () => {
  api.getAgent.mockRejectedValue(new Error('Agent not found'))
  const user = userEvent.setup()
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  expect(await screen.findByRole('alert')).toHaveTextContent('Agent not found')
})

it('edits the prompt, saves it and refreshes the list', async () => {
  const user = userEvent.setup()
  api.updateAgent.mockResolvedValue({ ...AGENTS[0], prompt: 'New prompt.' })
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  await user.click(await screen.findByRole('button', { name: 'Edit prompt' }))
  const box = screen.getByRole('textbox', { name: 'Agent prompt' })
  expect(box).toHaveValue(PROMPT)
  await user.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('textbox', { name: 'Agent prompt' })).not.toBeInTheDocument()

  await user.click(screen.getByRole('button', { name: 'Edit prompt' }))
  await user.clear(screen.getByRole('textbox', { name: 'Agent prompt' }))
  expect(screen.getByRole('button', { name: 'Save prompt' })).toBeDisabled()
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), ' New prompt. ')
  await user.click(screen.getByRole('button', { name: 'Save prompt' }))
  expect(api.updateAgent).toHaveBeenCalledWith(1, { prompt: 'New prompt.' })
  expect(await screen.findByText('New prompt.', { selector: 'pre' })).toBeInTheDocument()
  expect(api.listAgents).toHaveBeenCalledTimes(2)
})

it('a prompt saved while an image generates keeps both', async () => {
  const user = userEvent.setup()
  const svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'
  let finishImage
  api.generateAgentImage.mockReturnValue(new Promise((resolve) => (finishImage = resolve)))
  api.updateAgent.mockResolvedValue({ ...AGENTS[0], prompt: 'New prompt.' })
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  const detail = await screen.findByRole('region', { name: 'Bike Repair Agent' })
  await user.click(within(detail).getByRole('button', { name: 'Generate image' }))
  await user.click(within(detail).getByRole('button', { name: 'Edit prompt' }))
  await user.clear(within(detail).getByRole('textbox', { name: 'Agent prompt' }))
  await user.type(within(detail).getByRole('textbox', { name: 'Agent prompt' }), 'New prompt.')
  await user.click(within(detail).getByRole('button', { name: 'Save prompt' }))
  expect(await within(detail).findByText('New prompt.', { selector: 'pre' })).toBeInTheDocument()
  finishImage(agent({ image: svg }))
  expect(await within(detail).findByAltText('Bike Repair Agent image')).toBeInTheDocument()
  expect(within(detail).getByText('New prompt.', { selector: 'pre' })).toBeInTheDocument()
})

it('shows why a save was refused', async () => {
  const user = userEvent.setup()
  api.updateAgent.mockRejectedValue(new Error('This agent is training; wait for it to finish.'))
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  await user.click(await screen.findByRole('button', { name: 'Edit prompt' }))
  await user.click(screen.getByRole('button', { name: 'Save prompt' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('This agent is training')
  expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toBeInTheDocument()
})

it('uploads a prompt from a text file, asking before it replaces one', async () => {
  const user = userEvent.setup()
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  await user.click(await screen.findByRole('button', { name: 'Edit prompt' }))
  const input = screen.getByLabelText('Prompt file')
  const click = vi.spyOn(input, 'click')
  await user.click(screen.getByRole('button', { name: 'Upload prompt' }))
  expect(click).toHaveBeenCalledOnce()

  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
  const file = new File(['Uploaded\n  prompt.'], 'prompt.txt', { type: 'text/plain' })
  await user.upload(input, file)
  expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toHaveValue(PROMPT)
  await user.upload(input, file)
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toHaveValue('Uploaded\n  prompt.'))
  expect(confirm).toHaveBeenCalledTimes(2)
  confirm.mockRestore()
})

it('deletes an agent after confirming, then clears the selection', async () => {
  const user = userEvent.setup()
  api.deleteAgent.mockResolvedValue(null)
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  await user.click(await screen.findByRole('button', { name: 'Delete agent' }))
  const dialog = screen.getByRole('alertdialog', { name: 'Delete Bike Repair Agent?' })
  await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  expect(api.deleteAgent).not.toHaveBeenCalled()

  api.listAgents.mockResolvedValue([AGENTS[1]])
  await user.click(screen.getByRole('button', { name: 'Delete agent' }))
  await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Delete' }))
  expect(api.deleteAgent).toHaveBeenCalledWith(1)
  await waitFor(() => expect(screen.queryByRole('button', { name: /Bike Repair Agent/ })).not.toBeInTheDocument())
  expect(screen.queryByRole('region', { name: 'Bike Repair Agent' })).not.toBeInTheDocument()
})

it('locks editing and deleting while one of its trainings is in progress', async () => {
  const user = userEvent.setup()
  api.getAgent.mockResolvedValue(agent({ trainings: [{ ...TRAININGS[0], status: 'running', best_score: null }] }))
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  expect(await screen.findByRole('button', { name: 'Edit prompt' })).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Delete agent' })).toBeDisabled()
})

it('creates a local agent from a picked model and a written prompt, then opens it', async () => {
  const user = userEvent.setup()
  api.createAgent.mockResolvedValue({ ...AGENTS[0], id: 5, name: 'Helper' })
  api.getAgent.mockResolvedValue(agent({ id: 5, name: 'Helper', trainings: [] }))
  await openNew(user)
  expect(screen.queryByRole('button', { name: 'New agent' })).not.toBeInTheDocument()
  const create = screen.getByRole('button', { name: 'Create agent' })
  await user.type(screen.getByRole('textbox', { name: 'Name' }), '  Helper ')
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Be helpful.')
  expect(create).toBeDisabled() // no model yet
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(create)
  expect(api.createAgent).toHaveBeenCalledWith({ name: 'Helper', base_model: 'qwen-0.5b', prompt: 'Be helpful.' })
  expect(await screen.findByRole('region', { name: 'Helper' })).toBeInTheDocument()
  expect(api.getAgent).toHaveBeenCalledWith(5)
  expect(screen.queryByRole('button', { name: 'Create agent' })).not.toBeInTheDocument()
})

it('creates a Claude agent, noting that only its prompt can be trained', async () => {
  const user = userEvent.setup()
  api.createAgent.mockRejectedValue(new Error('An agent with that name already exists.'))
  await openNew(user)
  const model = within(screen.getByRole('radiogroup', { name: 'Model' }))
  expect(model.getByRole('radio', { name: 'Local model' })).toBeChecked()
  await waitFor(() => expect(model.getByRole('radio', { name: 'Claude' })).toBeEnabled())
  await user.click(model.getByRole('radio', { name: 'Claude' }))
  expect(screen.getByText("Claude's weights can't be fine-tuned, so training will only improve its prompt.")).toBeInTheDocument()
  expect(screen.queryByRole('radiogroup', { name: 'Model source' })).not.toBeInTheDocument()
  await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Writer')
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Write.')
  await user.click(screen.getByRole('button', { name: 'Create agent' }))
  expect(api.createAgent).toHaveBeenCalledWith({ name: 'Writer', base_model: 'claude', prompt: 'Write.' })
  expect(await screen.findByRole('alert')).toHaveTextContent('An agent with that name already exists.')

  await user.click(screen.getByRole('button', { name: 'Cancel' }))
  expect(screen.getByRole('button', { name: 'New agent' })).toBeInTheDocument()
})

it('offers Claude only once a key is linked on the Account page', async () => {
  api.me.mockResolvedValue({ claude_available: false })
  const user = userEvent.setup()
  await openNew(user)
  const claude = within(screen.getByRole('radiogroup', { name: 'Model' })).getByRole('radio', { name: 'Claude' })
  await waitFor(() => expect(claude).toHaveAccessibleDescription(/Link your Claude API key on the Account page/))
  expect(claude).toBeDisabled()
  expect(screen.getByRole('link', { name: 'Account' })).toHaveAttribute('href', '/account')
})

it('drafts the prompt from a description, asking before it replaces one, and retries a failure', async () => {
  const user = userEvent.setup()
  api.generatePrompt.mockRejectedValueOnce(new Error('Claude is overloaded')).mockResolvedValue({ prompt: 'You are a support bot.' })
  await openNew(user)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  const prompt = screen.getByRole('textbox', { name: 'Agent prompt' })
  await user.type(prompt, 'My own prompt.')
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A support bot')

  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValue(true)
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(api.generatePrompt).not.toHaveBeenCalled()
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(api.generatePrompt).toHaveBeenCalledWith('A support bot', 'qwen-0.5b', null)
  expect(await screen.findByRole('alert')).toHaveTextContent('Claude is overloaded')
  expect(prompt).toHaveValue('My own prompt.')

  await user.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(prompt).toHaveValue('You are a support bot.'))
  expect(confirm).toHaveBeenCalledTimes(2)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  confirm.mockRestore()
})

it('drafts a Claude agent prompt without a local model', async () => {
  const user = userEvent.setup()
  api.generatePrompt.mockResolvedValue({ prompt: 'Drafted.' })
  await openNew(user)
  await waitFor(() => expect(screen.getByRole('radio', { name: 'Claude' })).toBeEnabled())
  await user.click(screen.getByRole('radio', { name: 'Claude' }))
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A writer')
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(api.generatePrompt).toHaveBeenCalledWith('A writer', null, null)
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toHaveValue('Drafted.'))
})

// The local-model picker, moved here from the Train page.
it('searches the hub with the chosen sort', async () => {
  const user = userEvent.setup()
  api.searchHubModels.mockResolvedValue([HUB_MODEL])
  await openNew(user)
  await user.type(screen.getByRole('searchbox', { name: 'Search models' }), 'qwen')
  await user.selectOptions(screen.getByRole('combobox', { name: 'Sort by' }), 'Latest')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(api.searchHubModels).toHaveBeenCalledWith('qwen', 'lastModified')
  const result = (await screen.findByRole('heading', { name: HUB_MODEL.id })).closest('li')
  expect(result).toHaveTextContent('512')
  expect(result).toHaveTextContent('34')
})

it('shows hub search states: none found and errors', async () => {
  const user = userEvent.setup()
  api.searchHubModels.mockResolvedValueOnce([]).mockRejectedValue(new Error('Hub unreachable'))
  await openNew(user)
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(await screen.findByText(/No models match that search/)).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Hub unreachable')
})

it('downloads a hub model, refreshes local models and selects it', async () => {
  const user = userEvent.setup()
  let finish
  api.searchHubModels.mockResolvedValue([HUB_MODEL])
  api.downloadHubModel.mockReturnValue(new Promise((resolve) => (finish = resolve)))
  api.listLocalModels.mockResolvedValueOnce([]).mockResolvedValueOnce([{ name: 'Qwen2.5-0.5B', source: null }])
  await openNew(user)
  expect(await screen.findByText(/No local models yet/)).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Search' }))
  const download = await screen.findByRole('button', { name: `Download ${HUB_MODEL.id}` })
  await user.click(download)
  expect(api.downloadHubModel).toHaveBeenCalledWith(HUB_MODEL.id)
  expect(download).toBeDisabled()
  expect(download).toHaveTextContent('Downloading')
  await act(async () => finish({ name: 'Qwen2.5-0.5B', source: null }))
  const radio = await screen.findByRole('radio', { name: 'Qwen2.5-0.5B' })
  expect(radio).toBeChecked()
  expect(radio.closest('label').querySelector('[data-slot="badge"]')).toBeNull()
})

it('uploads a zipped model and selects it, and shows model list errors', async () => {
  const user = userEvent.setup()
  api.uploadModel.mockResolvedValue({ name: 'mine', source: 'upload' })
  api.listLocalModels.mockRejectedValueOnce(new Error('Server down')).mockResolvedValue([{ name: 'mine', source: 'upload' }])
  await openNew(user)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
  await user.click(screen.getByRole('radio', { name: 'Upload' }))
  const file = new File(['zip'], 'model.zip', { type: 'application/zip' })
  await user.upload(screen.getByLabelText('Model archive (.zip)'), file)
  await user.type(screen.getByRole('textbox', { name: 'Model name' }), 'mine')
  await user.click(screen.getByRole('button', { name: 'Upload model' }))
  expect(api.uploadModel).toHaveBeenCalledWith(file, 'mine')
  expect(await screen.findByRole('radio', { name: /mine/ })).toBeChecked()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('finds an agent from the feed as a new agent and opens it', async () => {
  const user = userEvent.setup()
  api.feed.mockResolvedValue([
    { id: 5, author: 'bea', name: 'Fixer', base_model: 'claude', prompt: 'Fix.', overall: 90, score: 0, my_vote: 0, created_at: '2026-10-05T10:00:00Z' },
  ])
  api.copyPost.mockResolvedValue({ id: 3, name: 'Fixer' })
  await openNew(user)
  await user.click(screen.getByRole('radio', { name: 'Find an agent' }))
  expect(screen.queryByRole('button', { name: 'Create agent' })).not.toBeInTheDocument()
  await user.click(await screen.findByRole('button', { name: 'Use this agent' }))
  expect(api.copyPost).toHaveBeenCalledWith(5)
  expect(api.getAgent).toHaveBeenCalledWith(3)
  expect(screen.queryByRole('heading', { name: 'New agent' })).not.toBeInTheDocument()
})

it('shows agent images on cards (helmet without one) and generates, edits and clears one', async () => {
  const user = userEvent.setup()
  const svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'
  api.listAgents.mockResolvedValue([{ ...AGENTS[0], image: svg }, AGENTS[1]])
  api.getAgent.mockResolvedValue(agent({ image: null }))
  render(<AgentsPage />)
  const bike = await card('Bike Repair Agent')
  expect(bike.querySelector('img').getAttribute('src')).toBe(`data:image/svg+xml;utf8,${encodeURIComponent(svg)}`)
  expect((await card('Claude Writer')).querySelector('img')).toBeNull()

  await user.click(bike)
  const detail = await screen.findByRole('region', { name: 'Bike Repair Agent' })
  api.generateAgentImage.mockResolvedValue(agent({ image: svg }))
  await user.click(within(detail).getByRole('button', { name: 'Generate image' }))
  expect(api.generateAgentImage).toHaveBeenCalledWith(1)
  expect(await within(detail).findByAltText('Bike Repair Agent image')).toBeInTheDocument()

  await user.click(within(detail).getByRole('button', { name: 'Edit SVG' }))
  const code = within(detail).getByRole('textbox', { name: 'SVG code' })
  expect(code).toHaveValue(svg)
  await user.clear(code)
  await user.type(code, '<svg><circle r="2"/></svg>')
  expect(within(detail).getByAltText('Preview').getAttribute('src')).toContain(encodeURIComponent('<circle r="2"/>'))
  api.setAgentImage.mockRejectedValue(new Error('SVG element not allowed: script'))
  await user.click(within(detail).getByRole('button', { name: 'Save image' }))
  expect(api.setAgentImage).toHaveBeenCalledWith(1, '<svg><circle r="2"/></svg>')
  expect(await within(detail).findByRole('alert')).toHaveTextContent('SVG element not allowed: script')

  api.setAgentImage.mockResolvedValue(agent({ image: null }))
  await user.click(within(detail).getByRole('button', { name: 'Clear image' }))
  expect(api.setAgentImage).toHaveBeenLastCalledWith(1, null)
  expect(await within(detail).findByRole('button', { name: 'Edit SVG' })).toBeInTheDocument()
  expect(within(detail).queryByAltText('Bike Repair Agent image')).toBeNull()

  await user.click(within(detail).getByRole('button', { name: 'Edit SVG' }))
  await user.click(within(detail).getByRole('button', { name: 'Cancel' }))
  expect(within(detail).queryByRole('textbox', { name: 'SVG code' })).toBeNull()
})

it('quantizes an open-weights agent into a new agent and opens it, but never a Claude one', async () => {
  const user = userEvent.setup()
  const quantized = { ...AGENTS[0], id: 3, name: 'Bike Repair Agent (int8)', base_model: 'qwen-0.5b-int8' }
  api.getAgent.mockImplementation(async (id) =>
    id === 3 ? { ...quantized, model_available: true, trainings: [] } : agent({ model_available: true }),
  )
  api.quantizeAgent.mockResolvedValue(quantized)
  render(<AgentsPage />)
  await user.click(await card('Bike Repair Agent'))
  api.listAgents.mockResolvedValue([...AGENTS, quantized])
  await user.click(await screen.findByRole('button', { name: 'Quantize' }))
  expect(api.quantizeAgent).toHaveBeenCalledWith(1)
  expect(await screen.findByRole('region', { name: 'Bike Repair Agent (int8)' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Quantize' })).not.toBeInTheDocument() // already quantized

  api.getAgent.mockResolvedValue({ ...AGENTS[1], model_available: true, trainings: [] })
  await user.click(await card('Claude Writer'))
  await screen.findByRole('region', { name: 'Claude Writer' })
  expect(screen.queryByRole('button', { name: 'Quantize' })).not.toBeInTheDocument()
})
