import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Link, MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api, saveBlob } from '../api'
import InspectPage from './InspectPage'

vi.mock('../api', () => ({
  api: {
    listAgents: vi.fn(),
    listTrainings: vi.fn(),
    getTraining: vi.fn(),
    listRounds: vi.fn(),
    listScenarios: vi.fn(),
    downloadPrompt: vi.fn(),
    downloadWeights: vi.fn(),
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
    deleteTraining: vi.fn(),
  },
  saveBlob: vi.fn(),
}))
Element.prototype.scrollIntoView = vi.fn() // jsdom has no layout

const agents = [
  { id: 1, name: 'Bike Repair Agent', base_model: 'tiny-lm', prompt: 'Fix bikes.', overall: 83, trainings: 2 },
  { id: 3, name: 'Writer', base_model: 'claude', prompt: 'Write.', overall: null, trainings: 0 },
]
const trainings = [
  { id: 2, base_model: 'qwen-0.5b', status: 'completed', best_score: 8.25, created_at: '2026-09-30T10:00:00Z', helper: 'claude' },
  { id: 1, base_model: 'llama-1b', status: 'running', best_score: null, created_at: '2026-09-29T10:00:00Z', helper: 'local' },
]
const rounds = {
  2: [
    { round: 1, mean_score: 6.5, scenarios: 2, model_id: 10, eval_score: 7.25 },
    { round: 2, mean_score: 8.25, scenarios: 2, model_id: 11, eval_score: 8.25 },
  ],
  1: [{ round: 1, mean_score: 4, scenarios: 1, model_id: null, eval_score: null }],
}
const details = {
  2: { initial_prompt: 'Answer support questions.', best_prompt: 'Answer support questions calmly and cite the order.', best_score: 8.25 },
  1: { initial_prompt: 'Help with billing.', best_prompt: null, best_score: null },
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

function Location() {
  const { search } = useLocation()
  return (
    <>
      <p data-testid="location">{search}</p>
      <Link to="/inspect">Inspect without agent</Link>
    </>
  )
}
const renderPage = (path = '/inspect?agent=1') =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route
          path="/inspect"
          element={
            <>
              <InspectPage />
              <Location />
            </>
          }
        />
        <Route path="/train" element={<p>Train page stub</p>} />
      </Routes>
    </MemoryRouter>,
  )

beforeEach(() => {
  vi.resetAllMocks()
  api.listAgents.mockResolvedValue(agents)
  api.listTrainings.mockResolvedValue(trainings)
  api.listRounds.mockImplementation(async (id) => rounds[id])
  api.getTraining.mockImplementation(async (id) => details[id])
  api.listScenarios.mockResolvedValue([scenario, { ...scenario, id: 32, scenario_number: 2, score: 5 }])
})
afterEach(() => vi.useRealTimers())

async function openScenario(user) {
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  await user.click(await screen.findByRole('button', { name: /Scenario 1/ }))
}

it('lists trainings with id, status, best score, date and helper', async () => {
  renderPage()
  expect(screen.getByTestId('inspect-page')).toBeInTheDocument()
  expect(screen.getByText(/Loading trainings/)).toBeInTheDocument()
  const buttons = await within(screen.getByRole('region', { name: 'Trainings' })).findAllByRole('button')
  expect(buttons.map((b) => b.textContent)).toEqual([expect.stringContaining('qwen-0.5b'), expect.stringContaining('llama-1b')])
  expect(buttons[0]).toHaveTextContent('#2')
  expect(buttons[1]).toHaveTextContent('#1')
  expect(within(buttons[0]).getByText('completed')).toBeInTheDocument()
  expect(within(buttons[0]).getByText('8.3')).toBeInTheDocument()
  expect(within(buttons[1]).getByText('running')).toBeInTheDocument()
  expect(within(buttons[1]).getByText('—')).toBeInTheDocument()
  expect(buttons[0]).toHaveTextContent('· Claude')
  expect(buttons[1]).toHaveTextContent('· Local model')
})

it('selecting a training loads its rounds', async () => {
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(api.listRounds).toHaveBeenCalledWith(2)
  expect(screen.getByRole('button', { name: /qwen-0\.5b/, pressed: true })).toBeInTheDocument()
  const round = await screen.findByRole('button', { name: /Round 1/ })
  // Its model's score on the same eval users as every round; the practice users get harder each round.
  expect(round).toHaveTextContent('7.3')
  expect(round).toHaveTextContent('2 scenarios · practice 6.5')
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

it('dropping the agent from the URL clears the selection under it', async () => {
  const user = userEvent.setup()
  renderPage()
  await openScenario(user)
  await user.click(screen.getByRole('link', { name: 'Inspect without agent' }))
  expect(screen.getByTestId('location')).toHaveTextContent('')
  expect(screen.queryByRole('article', { name: 'Persona' })).not.toBeInTheDocument()
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

it('downloads the prompt and weights of a completed training', async () => {
  const user = userEvent.setup()
  const promptBlob = new Blob(['Be kind.'])
  const weightsBlob = new Blob(['zip'])
  api.downloadPrompt.mockResolvedValue(promptBlob)
  api.downloadWeights.mockResolvedValue(weightsBlob)
  renderPage()
  await user.click(await screen.findByRole('button', { name: /llama-1b/ }))
  expect(screen.queryByRole('button', { name: 'Download prompt' })).not.toBeInTheDocument()

  await user.click(screen.getByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(screen.getByRole('button', { name: 'Download prompt' }))
  expect(api.downloadPrompt).toHaveBeenCalledWith(2)
  await waitFor(() => expect(saveBlob).toHaveBeenCalledWith(promptBlob, 'training-2-prompt.txt'))
  await user.click(screen.getByRole('button', { name: 'Download weights' }))
  expect(api.downloadWeights).toHaveBeenCalledWith(2)
  await waitFor(() => expect(saveBlob).toHaveBeenCalledWith(weightsBlob, 'training-2-lora.zip'))
})

it('shows download errors', async () => {
  const user = userEvent.setup()
  api.downloadPrompt.mockRejectedValue(new Error('Training has no trained model yet'))
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(screen.getByRole('button', { name: 'Download prompt' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Training has no trained model yet')
})

it('refreshes every 2 s while a training is in progress, keeping the selection', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
  const tick = () => act(() => vi.advanceTimersByTimeAsync(2000))
  renderPage()
  await user.click(await screen.findByRole('button', { name: /llama-1b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 1/ }))
  await screen.findByRole('button', { name: /Scenario 1/ })

  api.listTrainings.mockResolvedValue([trainings[0], { ...trainings[1], status: 'completed', best_score: 6 }])
  api.listRounds.mockResolvedValue([...rounds[1], { round: 2, mean_score: 6, scenarios: 1, model_id: 13 }])
  await tick()
  const llama = screen.getByRole('button', { name: /llama-1b/, pressed: true })
  expect(within(llama).getByText('completed')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Round 2/ })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /Round 1/, pressed: true })).toBeInTheDocument()
  expect(api.listScenarios).toHaveBeenLastCalledWith(1, 1)
  expect(screen.getByRole('button', { name: /Scenario 1/ })).toBeInTheDocument()

  const calls = api.listTrainings.mock.calls.length
  await tick()
  await tick()
  expect(api.listTrainings).toHaveBeenCalledTimes(calls)
})

it('brings each new panel into view and focuses the transcript', async () => {
  const user = userEvent.setup()
  const revealed = () => Element.prototype.scrollIntoView.mock.contexts.map((el) => el.getAttribute('aria-label'))
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  await user.click(await screen.findByRole('button', { name: /Scenario 1/ }))
  expect(revealed()).toEqual(['Trainings', 'Rounds', 'Scenarios', 'Scenario 1'])
  expect(screen.getByRole('region', { name: 'Scenario 1' })).toHaveFocus()
})

it('shows the original and highest-scoring prompts of the selected training', async () => {
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(api.getTraining).toHaveBeenCalledWith(2)
  const prompts = within(screen.getByRole('region', { name: 'Prompts' }))
  expect(prompts.getByRole('heading', { name: 'Original prompt' })).toBeInTheDocument()
  expect(await prompts.findByRole('region', { name: 'Original prompt' })).toHaveTextContent(details[2].initial_prompt)
  const best = prompts.getByRole('region', { name: 'Highest-scoring prompt' })
  expect(best).toHaveTextContent(details[2].best_prompt)
  expect(best).toHaveTextContent('8.3/10')
  expect(prompts.getByRole('heading', { name: 'Highest-scoring prompt Score 8.3/10' })).toBeInTheDocument()
})

it('shows an empty best prompt until a round finishes, and follows the selected training', async () => {
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /llama-1b/ }))
  const prompts = within(screen.getByRole('region', { name: 'Prompts' }))
  expect(await prompts.findByText(details[1].initial_prompt)).toBeInTheDocument()
  expect(prompts.getByText('No finished round yet')).toBeInTheDocument()
  expect(prompts.queryByText(/\/10/)).not.toBeInTheDocument()

  await user.click(screen.getByRole('button', { name: /qwen-0\.5b/ }))
  expect(await prompts.findByText(details[2].best_prompt)).toBeInTheDocument()
  expect(prompts.getByText(details[2].initial_prompt)).toBeInTheDocument()
  expect(prompts.queryByText(details[1].initial_prompt)).not.toBeInTheDocument()
  expect(prompts.queryByText('No finished round yet')).not.toBeInTheDocument()
})

it('shows the prompts loading state and load errors', async () => {
  let reject
  api.getTraining.mockImplementation(() => new Promise((_, r) => (reject = r)))
  const user = userEvent.setup()
  renderPage()
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  const prompts = within(screen.getByRole('region', { name: 'Prompts' }))
  expect(prompts.getByRole('status')).toHaveTextContent('Loading prompts…')
  reject(new Error('Training not found'))
  expect(await prompts.findByRole('alert')).toHaveTextContent('Training not found')
  expect(prompts.queryByRole('status')).not.toBeInTheDocument()
})

const more = [
  { id: 5, base_model: 'phi-2', status: 'paused', best_score: 6, created_at: '2026-10-01T10:00:00Z' },
  { id: 4, base_model: 'gemma-2b', status: 'failed', best_score: null, created_at: '2026-10-01T09:00:00Z' },
  { id: 3, base_model: 'mistral-7b', status: 'pausing', best_score: null, created_at: '2026-10-01T08:00:00Z' },
]
const option = (name) => screen.findByRole('button', { name })

it('shows paused and pausing statuses', async () => {
  api.listTrainings.mockResolvedValue(more)
  renderPage()
  expect(within(await option(/phi-2/)).getByText('paused')).toHaveClass('badge-paused')
  expect(within(await option(/mistral-7b/)).getByText('pausing')).toHaveClass('badge-pausing')
})

it('offers only the actions that fit the selected training', async () => {
  api.listTrainings.mockResolvedValue([...trainings, ...more])
  const user = userEvent.setup()
  renderPage()
  const actions = () => ['Pause training', 'Resume training', 'Retry training', 'Delete training']
    .filter((name) => screen.queryByRole('button', { name }))
  expect(actions()).toEqual([])
  for (const [name, expected] of [
    [/llama-1b/, ['Pause training']],
    [/mistral-7b/, []],
    [/phi-2/, ['Resume training', 'Delete training']],
    [/gemma-2b/, ['Retry training', 'Delete training']],
    [/qwen-0\.5b/, ['Delete training']],
  ]) {
    await user.click(await option(name))
    expect(actions(), String(name)).toEqual(expected)
  }
})

it('pauses a running training and refreshes the list', async () => {
  const user = userEvent.setup()
  api.pauseTraining.mockResolvedValue({})
  renderPage()
  await user.click(await option(/llama-1b/))
  api.listTrainings.mockResolvedValue([trainings[0], { ...trainings[1], status: 'pausing' }])
  await user.click(screen.getByRole('button', { name: 'Pause training' }))
  expect(api.pauseTraining).toHaveBeenCalledWith(1)
  expect(await within(screen.getByRole('button', { name: /llama-1b/, pressed: true })).findByText('pausing')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Pause training' })).not.toBeInTheDocument()
})

it('keeps refreshing while a training is pausing', async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  api.listTrainings.mockResolvedValue([more[2]])
  renderPage()
  await option(/mistral-7b/)
  api.listTrainings.mockResolvedValue([{ ...more[2], status: 'paused' }])
  await act(() => vi.advanceTimersByTimeAsync(2000))
  expect(within(screen.getByRole('button', { name: /mistral-7b/ })).getByText('paused')).toBeInTheDocument()
  const calls = api.listTrainings.mock.calls.length
  await act(() => vi.advanceTimersByTimeAsync(6000))
  expect(api.listTrainings).toHaveBeenCalledTimes(calls)
})

it('resumes a paused training and opens it on the Train page', async () => {
  api.listTrainings.mockResolvedValue(more)
  api.resumeTraining.mockResolvedValue({})
  const user = userEvent.setup()
  renderPage()
  await user.click(await option(/phi-2/))
  await user.click(screen.getByRole('button', { name: 'Resume training' }))
  expect(api.resumeTraining).toHaveBeenCalledWith(5)
  expect(await screen.findByText('Train page stub')).toBeInTheDocument()
})

it('shows a failed training error and retries it from the keyboard', async () => {
  api.listTrainings.mockResolvedValue(more)
  api.getTraining.mockResolvedValue({ ...details[1], error: 'Claude API error: overloaded' })
  api.resumeTraining.mockResolvedValue({})
  const user = userEvent.setup()
  renderPage()
  await user.click(await option(/gemma-2b/))
  expect(await screen.findByRole('alert')).toHaveTextContent('Claude API error: overloaded')
  screen.getByRole('button', { name: 'Retry training' }).focus()
  await user.keyboard('{Enter}')
  expect(api.resumeTraining).toHaveBeenCalledWith(4)
  expect(await screen.findByText('Train page stub')).toBeInTheDocument()
})

it('shows why an action was refused', async () => {
  api.listTrainings.mockResolvedValue(more)
  api.resumeTraining.mockRejectedValue(new Error('You already have a training in progress.'))
  const user = userEvent.setup()
  renderPage()
  await user.click(await option(/phi-2/))
  await user.click(screen.getByRole('button', { name: 'Resume training' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('You already have a training in progress.')
  expect(screen.getByRole('button', { name: 'Resume training' })).toBeEnabled()
  expect(screen.queryByText('Train page stub')).not.toBeInTheDocument()
})

it('deletes a training after confirming in a dialog, then clears the selection', async () => {
  api.deleteTraining.mockResolvedValue(null)
  const user = userEvent.setup()
  renderPage()
  await user.click(await option(/qwen-0\.5b/))
  await screen.findByRole('button', { name: /Round 2/ })
  await user.click(screen.getByRole('button', { name: 'Delete training' }))
  const dialog = screen.getByRole('alertdialog', { name: 'Delete training #2?' })
  expect(dialog).toHaveTextContent('all its rounds and scenarios')
  await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
  expect(api.deleteTraining).not.toHaveBeenCalled()

  api.listTrainings.mockResolvedValue([trainings[1]])
  await user.click(screen.getByRole('button', { name: 'Delete training' }))
  await user.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: 'Delete' }))
  expect(api.deleteTraining).toHaveBeenCalledWith(2)
  await waitFor(() => expect(screen.queryByRole('button', { name: /qwen-0\.5b/ })).not.toBeInTheDocument())
  expect(screen.queryByRole('region', { name: 'Rounds' })).not.toBeInTheDocument()
  expect(screen.queryByRole('region', { name: 'Prompts' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Delete training' })).not.toBeInTheDocument()
})

it('shows the criteria each persona pushes hardest, strongest first', async () => {
  const attributes = [
    { criterion: 'Accuracy', rating: 3 },
    { criterion: 'Empathy', rating: 7 },
    { criterion: 'Asks for the order number', rating: 9 },
  ]
  api.listScenarios.mockResolvedValue([
    { ...scenario, persona: { ...scenario.persona, attributes } },
    { ...scenario, id: 32, scenario_number: 2 },
  ])
  const user = userEvent.setup()
  renderPage()
  await openScenario(user)
  const tested = screen.getByRole('list', { name: 'Criteria this persona tests' })
  expect(within(tested).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['Asks for the order number', 'Empathy'])

  await user.click(screen.getByRole('button', { name: /Scenario 2/ }))
  expect(screen.queryByRole('list', { name: 'Criteria this persona tests' })).not.toBeInTheDocument()
  expect(screen.getByText('This persona keeps every criterion easy.')).toBeInTheDocument()
})

it('keeps working where scrollIntoView returns a promise (current Chromium)', async () => {
  Element.prototype.scrollIntoView.mockResolvedValue(undefined)
  const user = userEvent.setup()
  renderPage()
  await user.click(await option(/qwen-0\.5b/))
  await user.click(await screen.findByRole('button', { name: /Round 2/ }))
  await user.click(await option(/llama-1b/))
  expect(await screen.findByRole('button', { name: /Round 1/ })).toHaveTextContent('4')
  expect(screen.getByRole('button', { name: /llama-1b/, pressed: true })).toBeInTheDocument()
})

it('starts with the agents, their overall, and no trainings until one is picked', async () => {
  const user = userEvent.setup()
  renderPage('/inspect')
  const bike = await screen.findByRole('button', { name: /Bike Repair Agent/ })
  expect(within(bike).getByRole('img', { name: 'Overall 83' })).toHaveTextContent('83')
  expect(bike).toHaveTextContent('tiny-lm · 2 trainings')
  const writer = screen.getByRole('button', { name: /Writer/ })
  expect(within(writer).getByRole('img', { name: 'Unranked' })).toBeInTheDocument()
  expect(writer).toHaveTextContent('Claude · 0 trainings')
  expect(screen.queryByRole('region', { name: 'Trainings' })).not.toBeInTheDocument()
  expect(api.listTrainings).not.toHaveBeenCalled()

  await user.click(bike)
  expect(api.listTrainings).toHaveBeenCalledWith(1)
  expect(screen.getByTestId('location')).toHaveTextContent('?agent=1')
  expect(screen.getByRole('button', { name: /Bike Repair Agent/, pressed: true })).toBeInTheDocument()
  expect(await screen.findByRole('button', { name: /qwen-0\.5b/ })).toBeInTheDocument()
})

it('opens the agent named in the URL and clears the training when the agent changes', async () => {
  const user = userEvent.setup()
  renderPage()
  expect(await screen.findByRole('button', { name: /Bike Repair Agent/, pressed: true })).toBeInTheDocument()
  expect(api.listTrainings).toHaveBeenCalledWith(1)
  await user.click(await screen.findByRole('button', { name: /qwen-0\.5b/ }))
  expect(await screen.findByRole('region', { name: 'Rounds' })).toBeInTheDocument()

  api.listTrainings.mockResolvedValue([])
  await user.click(screen.getByRole('button', { name: /Writer/ }))
  expect(api.listTrainings).toHaveBeenLastCalledWith(3)
  expect(await screen.findByText(/No trainings yet/)).toBeInTheDocument()
  expect(screen.queryByRole('region', { name: 'Rounds' })).not.toBeInTheDocument()
  expect(screen.getByTestId('location')).toHaveTextContent('?agent=3')
})

it('links to the Agents page when there are no agents', async () => {
  api.listAgents.mockResolvedValue([])
  renderPage('/inspect')
  expect(await screen.findByText(/No agents yet — make one on the/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Agents page' })).toHaveAttribute('href', '/agents')
})

it('offers no weights for a completed Claude training', async () => {
  api.listTrainings.mockResolvedValue([{ ...trainings[0], base_model: 'claude' }])
  const user = userEvent.setup()
  renderPage('/inspect?agent=3')
  await user.click(await screen.findByRole('button', { name: /#2/ }))
  expect(screen.getByRole('button', { name: 'Download prompt' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Download weights' })).not.toBeInTheDocument()
})

it("shows each agent's image in the picker, the knight helmet without one", async () => {
  api.listAgents.mockResolvedValue([{ ...agents[0], image: '<svg/>' }, agents[1]])
  renderPage()
  const bike = await screen.findByRole('button', { name: /Bike Repair Agent/ })
  expect(bike.querySelector('img')).toHaveAttribute('src', `data:image/svg+xml;utf8,${encodeURIComponent('<svg/>')}`)
  expect(screen.getByRole('button', { name: /Writer/ }).querySelector('img')).toBeNull()
})
