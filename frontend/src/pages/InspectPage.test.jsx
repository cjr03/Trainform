import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { api } from '../api'
import InspectPage from './InspectPage'

vi.mock('../api', () => ({
  api: { listTrainings: vi.fn(), listRounds: vi.fn(), listScenarios: vi.fn() },
}))

const trainings = [
  { id: 2, base_model: 'qwen-0.5b', status: 'completed', best_score: 8.25, created_at: '2026-09-30T10:00:00Z' },
  { id: 1, base_model: 'llama-1b', status: 'running', best_score: null, created_at: '2026-09-29T10:00:00Z' },
]
const rounds = {
  2: [
    { round: 1, mean_score: 6.5, scenarios: 2, model_id: 10 },
    { round: 2, mean_score: 8.25, scenarios: 2, model_id: 11 },
  ],
  1: [{ round: 1, mean_score: 4, scenarios: 1, model_id: null }],
}
const scenario = {
  id: 31,
  scenario_number: 1,
  persona: {
    attributes: [{ criterion: 'Patience', rating: 2 }],
    prompt: 'You are an impatient customer.',
  },
  prompt: 'You are a support agent.',
  input: 'My order never arrived!',
  output: 'Sorry to hear that, let me check your order.',
  scores: [
    { criterion: 'Empathy', score: 9 },
    { criterion: 'Accuracy', score: 6 },
  ],
  score: 7.5,
}

const renderPage = () =>
  render(
    <MemoryRouter>
      <InspectPage />
    </MemoryRouter>,
  )

beforeEach(() => {
  vi.resetAllMocks()
  api.listTrainings.mockResolvedValue(trainings)
  api.listRounds.mockImplementation(async (id) => rounds[id])
  api.listScenarios.mockResolvedValue([scenario, { ...scenario, id: 32, scenario_number: 2, score: 5 }])
})

async function openScenario(user) {
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  await user.click(await screen.findByRole('button', { name: /Scenario 1/ }))
}

it('lists trainings with id, status, best score and date', async () => {
  renderPage()
  expect(screen.getByTestId('inspect-page')).toBeInTheDocument()
  expect(screen.getByText(/Loading trainings/)).toBeInTheDocument()
  const buttons = await screen.findAllByRole('button', { pressed: false })
  expect(buttons.map((b) => b.textContent)).toEqual([expect.stringContaining('qwen-0.5b'), expect.stringContaining('llama-1b')])
  expect(buttons[0]).toHaveTextContent('#2')
  expect(buttons[1]).toHaveTextContent('#1')
  expect(within(buttons[0]).getByText('completed')).toBeInTheDocument()
  expect(within(buttons[0]).getByText('8.3')).toBeInTheDocument()
  expect(within(buttons[1]).getByText('running')).toBeInTheDocument()
  expect(within(buttons[1]).getByText('—')).toBeInTheDocument()
})

it('selecting a training loads its rounds', async () => {
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(api.listRounds).toHaveBeenCalledWith(2)
  expect(screen.getByRole('button', { name: /qwen-0\.5b/, pressed: true })).toBeInTheDocument()
  const round = await screen.findByRole('button', { name: /Round 1/ })
  expect(round).toHaveTextContent('6.5')
  expect(round).toHaveTextContent('2 scenarios')
  expect(screen.getByRole('button', { name: /Round 2/ })).toBeInTheDocument()
})

it('selecting a round loads its scenarios', async () => {
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  expect(api.listScenarios).toHaveBeenCalledWith(2, 2)
  expect(await screen.findByRole('button', { name: /Scenario 1/ })).toHaveTextContent('7.5')
  expect(screen.getByRole('button', { name: /Scenario 2/ })).toHaveTextContent('5')
})

it('shows the selected scenario as a chat with its scores and details', async () => {
  const user = userEvent.setup()
  renderPage()
  await openScenario(user)
  expect(within(screen.getByRole('article', { name: 'Persona' })).getByText(scenario.input)).toBeInTheDocument()
  expect(within(screen.getByRole('article', { name: 'Agent' })).getByText(scenario.output)).toBeInTheDocument()
  expect(screen.getByRole('img', { name: 'Overall score 7.5 out of 10' })).toBeInTheDocument()
  const criteria = within(screen.getByRole('list', { name: 'Criterion scores' }))
  expect(criteria.getByText('Empathy').closest('li')).toHaveTextContent('9/10')
  expect(criteria.getByText('Accuracy').closest('li')).toHaveTextContent('6/10')
  expect(screen.getByText('Patience').closest('li')).toHaveTextContent('2/10')
  expect(screen.getByText(scenario.persona.prompt)).toBeInTheDocument()
  expect(screen.getByText(scenario.prompt)).toBeInTheDocument()
})

it('changing the training clears the round and scenario', async () => {
  const user = userEvent.setup()
  renderPage()
  await openScenario(user)
  await user.click(screen.getByRole('button', { name: /llama-1b/ }))
  expect(api.listRounds).toHaveBeenLastCalledWith(1)
  expect(screen.queryByRole('article', { name: 'Persona' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /Scenario 1/ })).not.toBeInTheDocument()
  expect(await screen.findByRole('button', { name: /Round 1/, pressed: false })).toHaveTextContent('4')
  expect(screen.queryByRole('button', { name: /Round 2/ })).not.toBeInTheDocument()
})

it('changing the round clears the scenario', async () => {
  const user = userEvent.setup()
  renderPage()
  await openScenario(user)
  await user.click(screen.getByRole('button', { name: /Round 1/ }))
  expect(screen.queryByRole('article', { name: 'Persona' })).not.toBeInTheDocument()
  expect(await screen.findByRole('button', { name: /Scenario 1/, pressed: false })).toBeInTheDocument()
})

it('shows the empty state with a link to Train', async () => {
  api.listTrainings.mockResolvedValue([])
  renderPage()
  expect(await screen.findByText(/No trainings yet — start one on the/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Train page' })).toHaveAttribute('href', '/train')
})

it('shows load errors', async () => {
  api.listTrainings.mockRejectedValue(new Error('Server down'))
  renderPage()
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
})

it('shows round load errors', async () => {
  api.listRounds.mockRejectedValue(new Error('Training not found'))
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Training not found')
})

it('ignores a late reply for a training that is no longer selected', async () => {
  let resolveQwen
  api.listRounds.mockImplementation((id) => (id === 2 ? new Promise((r) => (resolveQwen = r)) : Promise.resolve(rounds[1])))
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(screen.getByRole('button', { name: /llama-1b/ }))
  expect(await screen.findByRole('button', { name: /Round 1/ })).toHaveTextContent('4')
  resolveQwen([{ round: 7, mean_score: 9, scenarios: 3, model_id: 12 }])
  await new Promise((r) => setTimeout(r))
  expect(screen.queryByRole('button', { name: /Round 7/ })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Round 1/ })).toBeInTheDocument()
  expect(screen.queryByRole('status')).not.toBeInTheDocument()
})

it('shows the empty rounds state', async () => {
  api.listRounds.mockResolvedValue([])
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(await screen.findByText('No rounds finished yet.')).toBeInTheDocument()
})

it('shows scenario load errors', async () => {
  api.listScenarios.mockRejectedValue(new Error('Round not found'))
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  const col = screen.getByRole('region', { name: 'Scenarios' })
  expect(await within(col).findByRole('alert')).toHaveTextContent('Round not found')
})
