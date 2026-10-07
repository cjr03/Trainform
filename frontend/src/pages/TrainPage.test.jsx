import { act, render as rtlRender, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import userEvent from '@testing-library/user-event'
import { readdirSync, readFileSync } from 'node:fs'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api, saveBlob } from '../api'
import TrainPage from './TrainPage'

const render = (ui) => rtlRender(ui, { wrapper: MemoryRouter })

vi.mock('../api', () => ({
  api: {
    me: vi.fn(),
    listAgents: vi.fn(),
    listTrainings: vi.fn(),
    generateRubric: vi.fn(),
    startTraining: vi.fn(),
    getTraining: vi.fn(),
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
    downloadPrompt: vi.fn(),
    downloadWeights: vi.fn(),
    feed: vi.fn(),
    getProfile: vi.fn(),
    copyPost: vi.fn(),
  },
  saveBlob: vi.fn(),
}))

const AGENTS = [
  { id: 1, name: 'Bike Repair Agent', base_model: 'qwen-0.5b', prompt: 'Be helpful.', overall: 84, trainings: 1 },
  { id: 2, name: 'Claude Writer', base_model: 'claude', prompt: 'Write well.', overall: null, trainings: 0 },
]
const detail = (over = {}) => ({
  id: 7,
  status: 'running',
  error: null,
  base_model: 'qwen-0.5b',
  initial_prompt: 'Be helpful.',
  rubric: ['Clarity'],
  rounds: 3,
  scenarios_per_round: 5,
  current_round: 0,
  scenarios_done: 0,
  total_scenarios: 15,
  latest_prompt: 'Be helpful.',
  score_windows: [],
  best_model_id: null,
  best_score: null,
  best_prompt: null,
  created_at: '2026-10-02T10:00:00Z',
  ...over,
})

beforeEach(() => {
  vi.resetAllMocks()
  sessionStorage.clear()
  api.listAgents.mockResolvedValue(AGENTS)
  api.listTrainings.mockResolvedValue([])
  api.me.mockResolvedValue({ claude_available: false })
})
afterEach(() => vi.useRealTimers())

const stepper = () => within(screen.getByRole('list', { name: 'Training workflow' }))
const step = (name) => stepper().getByRole('button', { name: new RegExp(name) })
const nextButton = () => screen.getByRole('button', { name: 'Next' })

async function reachScenarios(user) {
  const view = render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /Bike Repair Agent/ }))
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  return view
}
const bike = () => screen.findByRole('radio', { name: /Bike Repair Agent/ })

function polling() {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  return userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
}
const tick = (ms = 2000) => act(() => vi.advanceTimersByTimeAsync(ms))

async function startTraining(user, polled, ms = 2000) {
  const view = await reachScenarios(user)
  api.startTraining.mockResolvedValue(detail({ status: 'pending' }))
  api.getTraining.mockResolvedValue(polled)
  await user.click(screen.getByRole('button', { name: 'Start training' }))
  await tick(ms)
  return view
}

const completed = detail({
  status: 'completed',
  current_round: 3,
  scenarios_done: 15,
  best_score: 8.4,
  baseline_score: 6.1,
  latest_prompt: 'Be brief.',
  best_prompt: 'Be kind.',
})

it('unlocks each step only once the previous ones are valid', async () => {
  const user = userEvent.setup()
  render(<TrainPage />)
  expect(screen.getByTestId('train-page')).toBeInTheDocument()
  expect(step('Agent')).toHaveAttribute('aria-current', 'step')
  expect(nextButton()).toBeDisabled()
  expect(step('Rubric')).toBeDisabled()
  expect(stepper().queryByRole('button', { name: /Model|Prompt/ })).not.toBeInTheDocument()

  await user.click(await bike())
  expect(step('Rubric')).toBeEnabled()
  expect(step('Scenarios')).toBeDisabled()
  await user.click(nextButton())
  expect(step('Rubric')).toHaveAttribute('aria-current', 'step')
  // Next is now disabled, so focus must move into the new slide rather than drop to <body>.
  expect(screen.getByRole('region', { name: 'Rubric' })).toHaveFocus()

  expect(nextButton()).toBeDisabled()
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  expect(nextButton()).toBeEnabled()
  expect(step('Scenarios')).toBeEnabled()
  expect(step('Training')).toBeDisabled()

  await user.click(screen.getByRole('button', { name: 'Back' }))
  expect(step('Agent')).toHaveAttribute('aria-current', 'step')
  expect(step('Rubric')).toHaveAccessibleName(/done/)
  expect(step('Training').closest('li')).toHaveAttribute('data-state', 'locked')
})

it('lists the agents with their overall and links to the Agents page', async () => {
  render(<TrainPage />)
  const option = (await bike()).closest('label')
  expect(option).toHaveTextContent('qwen-0.5b')
  expect(within(option).getByRole('img', { name: 'Overall 84' })).toBeInTheDocument()
  const claude = screen.getByRole('radio', { name: /Claude Writer/ }).closest('label')
  expect(within(claude).getByRole('img', { name: 'Unranked' })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Agents page' })).toHaveAttribute('href', '/agents')
})

it('points to the Agents page when there are no agents, and shows load errors', async () => {
  api.listAgents.mockResolvedValue([])
  const view = render(<TrainPage />)
  expect(await screen.findByText(/No agents yet/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Agents page' })).toHaveAttribute('href', '/agents')
  view.unmount()
  api.listAgents.mockRejectedValue(new Error('Server down'))
  render(<TrainPage />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')
})

it('trains a Claude agent only with Claude as the helper, and only its prompt', async () => {
  const user = userEvent.setup()
  api.me.mockResolvedValue({ claude_available: true })
  api.generateRubric.mockResolvedValue({ criteria: ['Clarity'] })
  render(<TrainPage />)
  await waitFor(() => expect(helper().getByRole('radio', { name: 'Claude' })).toBeChecked())
  await user.click(screen.getByRole('radio', { name: /Claude Writer/ }))
  expect(screen.getByText(/training will only improve its prompt. Claude agents need Claude as the helper./)).toBeInTheDocument()
  await user.click(helper().getByRole('radio', { name: 'Local model' }))
  expect(nextButton()).toBeDisabled()
  await user.click(helper().getByRole('radio', { name: 'Claude' }))
  await user.click(nextButton())
  await user.click(screen.getByRole('button', { name: 'Generate from prompt' }))
  expect(api.generateRubric).toHaveBeenCalledWith('Write well.', null, 'claude')
})

const helper = () => within(screen.getByRole('radiogroup', { name: 'Train with' }))

it('trains with Claude by default when it is linked, or with the chosen local model', async () => {
  const user = userEvent.setup()
  api.me.mockResolvedValue({ claude_available: true })
  api.generateRubric.mockResolvedValue({ criteria: ['Clarity'] })
  const view = render(<TrainPage />)
  await waitFor(() => expect(helper().getByRole('radio', { name: 'Claude' })).toBeChecked())
  expect(screen.queryByRole('link', { name: 'Account' })).not.toBeInTheDocument()
  await user.click(await bike())
  await user.click(nextButton())
  await user.click(screen.getByRole('button', { name: 'Generate from prompt' }))
  expect(api.generateRubric).toHaveBeenLastCalledWith('Be helpful.', 'qwen-0.5b', 'claude')

  await user.click(step('Agent'))
  await user.click(helper().getByRole('radio', { name: 'Local model' }))
  view.unmount() // the choice is kept with the rest of the draft
  render(<TrainPage />)
  await waitFor(() => expect(api.me).toHaveBeenCalledTimes(2))
  expect(helper().getByRole('radio', { name: 'Local model' })).toBeChecked()
})

it('offers only the local model until Claude is linked on the Account page', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ helper: 'claude' })) // unlinked since
  render(<TrainPage />)
  await waitFor(() => expect(helper().getByRole('radio', { name: 'Local model' })).toBeChecked())
  const claude = helper().getByRole('radio', { name: 'Claude' })
  expect(claude).toBeDisabled()
  expect(claude).toHaveAccessibleDescription(/Link your Claude API key on the Account page/)
  expect(screen.getByRole('link', { name: 'Account' })).toHaveAttribute('href', '/account')
})

it('leaves the helper to the server when the account cannot be read', async () => {
  api.me.mockRejectedValue(new Error('Server down'))
  render(<TrainPage />)
  await waitFor(() => expect(api.me).toHaveBeenCalled())
  expect(helper().getByRole('radio', { name: 'Local model' })).not.toBeChecked()
  expect(helper().getByRole('radio', { name: 'Claude' })).not.toBeChecked()
})

it('generates the rubric and supports add, edit, delete and confirmed regenerate', async () => {
  const user = userEvent.setup()
  api.generateRubric
    .mockResolvedValueOnce({ criteria: ['Clarity', 'Empathy'] })
    .mockResolvedValueOnce({ criteria: ['Speed'] })
  render(<TrainPage />)
  await user.click(await bike())
  await user.click(nextButton())
  await user.click(screen.getByRole('button', { name: 'Generate from prompt' }))
  expect(api.generateRubric).toHaveBeenCalledWith('Be helpful.', 'qwen-0.5b', 'local')

  const first = await screen.findByRole('textbox', { name: 'Criterion 1' })
  expect(first).toHaveValue('Clarity')
  expect(screen.getByRole('textbox', { name: 'Criterion 2' })).toHaveValue('Empathy')
  await user.clear(first)
  await user.type(first, 'Clear answers')
  await user.click(screen.getByRole('button', { name: 'Delete criterion 2' }))
  expect(screen.queryByRole('textbox', { name: 'Criterion 2' })).not.toBeInTheDocument()
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Accuracy')
  await user.click(screen.getByRole('button', { name: 'Add criterion' }))
  expect(screen.getByRole('textbox', { name: 'Criterion 2' })).toHaveValue('Accuracy')
  expect(screen.getByRole('textbox', { name: 'New criterion' })).toHaveValue('')

  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
  await user.click(screen.getByRole('button', { name: 'Regenerate' }))
  expect(confirm).toHaveBeenCalledOnce()
  expect(api.generateRubric).toHaveBeenCalledOnce()
  expect(screen.getByRole('textbox', { name: 'Criterion 1' })).toHaveValue('Clear answers')

  await user.click(screen.getByRole('button', { name: 'Regenerate' }))
  await waitFor(() => expect(screen.getByRole('textbox', { name: 'Criterion 1' })).toHaveValue('Speed'))
  expect(screen.queryByRole('textbox', { name: 'Criterion 2' })).not.toBeInTheDocument()
  confirm.mockRestore()
})

it('starts training with the exact config and locks the setup steps', async () => {
  const user = userEvent.setup()
  await reachScenarios(user)
  const scenarios = screen.getByRole('spinbutton', { name: 'Simulated scenarios per round' })
  const rounds = screen.getByRole('spinbutton', { name: 'Training rounds' })
  expect(scenarios).toHaveValue(5)
  expect(rounds).toHaveValue(3)
  const start = screen.getByRole('button', { name: 'Start training' })

  await user.clear(scenarios)
  await user.type(scenarios, '0')
  expect(start).toBeDisabled()
  await user.clear(scenarios)
  await user.type(scenarios, '8')
  await user.clear(rounds)
  await user.type(rounds, '51')
  expect(start).toBeDisabled()
  await user.clear(rounds)
  await user.type(rounds, '4')
  expect(screen.getByTestId('total-runs')).toHaveTextContent('32')

  api.startTraining.mockResolvedValue(detail({ status: 'pending' }))
  api.getTraining.mockResolvedValue(detail())
  await user.click(start)
  expect(screen.getByText('Bike Repair Agent', { selector: 'dd' })).toBeInTheDocument()
  expect(api.startTraining).toHaveBeenCalledWith({
    agent_id: 1,
    rubric: ['Clarity'],
    scenarios: 8,
    rounds: 4,
    helper: 'local',
    battle: false,
    opponent_post_id: null,
    opponent_agent_id: null,
  })
  await waitFor(() => expect(step('Training')).toHaveAttribute('aria-current', 'step'))
  for (const name of ['Agent', 'Rubric', 'Scenarios']) expect(step(name)).toBeDisabled()
})

it('shows a failed start inline on the Scenarios slide', async () => {
  const user = userEvent.setup()
  await reachScenarios(user)
  api.startTraining.mockRejectedValue(new Error('You already have a training in progress.'))
  await user.click(screen.getByRole('button', { name: 'Start training' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('You already have a training in progress.')
  expect(step('Scenarios')).toHaveAttribute('aria-current', 'step')
})

it('opens the running training when Start reports one is already in progress', async () => {
  const user = userEvent.setup()
  await reachScenarios(user)
  api.startTraining.mockRejectedValue(Object.assign(new Error('You already have a training in progress.'), { status: 409 }))
  api.listTrainings.mockResolvedValue([{ id: 9, status: 'running' }])
  api.getTraining.mockResolvedValue(detail({ id: 9, scenarios_done: 2 }))
  await user.click(screen.getByRole('button', { name: 'Start training' }))
  await waitFor(() => expect(step('Training')).toHaveAttribute('aria-current', 'step'))
  expect(api.getTraining).toHaveBeenCalledWith(9)
  expect(screen.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '2')
})

it('reopens a training that is still running', async () => {
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'running' }, { id: 6, status: 'completed' }])
  api.getTraining.mockResolvedValue(detail({ scenarios_done: 4 }))
  render(<TrainPage />)
  await waitFor(() => expect(step('Training')).toHaveAttribute('aria-current', 'step'))
  expect(api.getTraining).toHaveBeenCalledWith(7)
  expect(screen.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '4')
  expect(step('Agent')).toBeDisabled()
})

it('polls exactly every 2 s and never overlaps requests', async () => {
  const user = polling()
  await startTraining(user, detail(), 0)
  const at = []
  let release
  api.getTraining.mockImplementation(() => {
    at.push(Date.now())
    return new Promise((resolve) => (release = () => resolve(detail())))
  })
  await tick(1900)
  expect(api.getTraining).not.toHaveBeenCalled()
  await tick(100)
  expect(api.getTraining).toHaveBeenCalledOnce()
  await tick(10000)
  expect(api.getTraining).toHaveBeenCalledOnce()

  await act(async () => release())
  await tick(2000)
  expect(api.getTraining).toHaveBeenCalledTimes(2)
  expect(at[1] - at[0]).toBeGreaterThanOrEqual(12000)
  await act(async () => release())
  await tick(2000)
  expect(at[2] - at[1]).toBe(2000)
})

it('shows a retry notice when a poll fails', async () => {
  const user = polling()
  await startTraining(user, detail(), 0)
  api.getTraining.mockRejectedValueOnce(new Error('Network down')).mockResolvedValue(detail())
  await tick()
  expect(screen.getByRole('alert')).toHaveTextContent('Network down')
  expect(screen.getByRole('alert')).toHaveTextContent('Retrying')
  await tick()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('polls every 2 s, rendering both progress bars and the four score windows, until unmounted', async () => {
  const user = polling()
  const view = await startTraining(
    user,
    detail({
      current_round: 2,
      scenarios_done: 6,
      score_windows: [
        { label: 'n', runs: 6, score: 6.5 },
        { label: 'n/2', runs: 3, score: 7 },
        { label: 'n/4', runs: 1, score: 7.5 },
        { label: 'last', runs: 1, score: 8 },
      ],
    }),
  )
  expect(api.getTraining).toHaveBeenCalledWith(7)

  const scenarios = screen.getByRole('progressbar', { name: 'Scenarios tested' })
  expect(scenarios).toHaveAttribute('aria-valuenow', '6')
  expect(scenarios).toHaveAttribute('aria-valuemax', '15')
  // Round 2 is still running, so only one round is complete.
  const rounds = screen.getByRole('progressbar', { name: 'Rounds completed' })
  expect(rounds).toHaveAttribute('aria-valuenow', '1')
  expect(rounds).toHaveAttribute('aria-valuemax', '3')
  expect(screen.getByRole('status')).toHaveTextContent(/running/i)

  for (const [name, runs, score] of [
    ['Total runs', '6 runs', '6.5'],
    ['Last 50%', '3 runs', '7.0'],
    ['Last 25%', '1 run', '7.5'],
    ['Last run', '1 run', '8.0'],
  ]) {
    const card = screen.getByRole('group', { name })
    expect(card).toHaveTextContent(runs)
    expect(card).toHaveTextContent(score)
    expect(card).toHaveTextContent('/ 10')
  }

  view.unmount()
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)
})

it('shows the error and stops polling when training fails', async () => {
  const user = polling()
  await startTraining(user, detail({ status: 'failed', error: 'CUDA out of memory' }))
  expect(screen.getByRole('alert')).toHaveTextContent('CUDA out of memory')
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)
})

it('advances to the completion screen and downloads the prompt and weights', async () => {
  const user = polling()
  await startTraining(user, completed)
  expect(step('Complete')).toHaveAttribute('aria-current', 'step')
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)
  expect(screen.getByTestId('best-score')).toHaveTextContent('8.4')
  expect(screen.getByText('6.1 before training, on the same eval users')).toBeInTheDocument()
  // The prompt shown is the best round's, which is what Download prompt saves.
  expect(screen.getByRole('heading', { name: 'Best prompt' })).toBeInTheDocument()
  expect(screen.getByTestId('best-prompt')).toHaveTextContent('Be kind.')
  expect(screen.queryByText('Be brief.')).not.toBeInTheDocument()

  const promptBlob = new Blob(['Be kind.'])
  const weightsBlob = new Blob(['zip'])
  api.downloadPrompt.mockResolvedValue(promptBlob)
  api.downloadWeights.mockResolvedValue(weightsBlob)
  await user.click(screen.getByRole('button', { name: 'Download prompt' }))
  expect(api.downloadPrompt).toHaveBeenCalledWith(7)
  await waitFor(() => expect(saveBlob).toHaveBeenCalledWith(promptBlob, 'training-7-prompt.txt'))
  await user.click(screen.getByRole('button', { name: 'Download weights' }))
  expect(api.downloadWeights).toHaveBeenCalledWith(7)
  await waitFor(() => expect(saveBlob).toHaveBeenCalledWith(weightsBlob, 'training-7-lora.zip'))

  await user.click(step('Training'))
  expect(screen.getByRole('progressbar', { name: 'Rounds completed' })).toHaveAttribute('aria-valuenow', '3')
})

it('resets the dashboard for a new training', async () => {
  const user = polling()
  await startTraining(user, completed)
  await user.click(screen.getByRole('button', { name: 'Start a new training' }))

  expect(step('Agent')).toHaveAttribute('aria-current', 'step')
  expect(step('Rubric')).toBeDisabled()
  const agent = await bike()
  expect(agent).not.toBeChecked()
  await user.click(agent)
  await user.click(nextButton())
  expect(screen.queryByRole('textbox', { name: 'Criterion 1' })).not.toBeInTheDocument()
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)
})

it('keeps the setup when the page is left and reopened', async () => {
  const user = userEvent.setup()
  const view = await reachScenarios(user)
  await user.clear(screen.getByRole('spinbutton', { name: 'Training rounds' }))
  await user.type(screen.getByRole('spinbutton', { name: 'Training rounds' }), '4')
  view.unmount() // what navigating to Inspect or Account does

  render(<TrainPage />)
  expect(step('Scenarios')).toHaveAttribute('aria-current', 'step')
  expect(screen.getByRole('spinbutton', { name: 'Training rounds' })).toHaveValue(4)
  expect(screen.getByTestId('total-runs')).toHaveTextContent('20')
  expect(await screen.findByText('Bike Repair Agent', { selector: 'dd' })).toBeInTheDocument()
  await user.click(step('Rubric'))
  expect(screen.getByRole('textbox', { name: 'Criterion 1' })).toHaveValue('Clarity')
  await user.click(step('Agent'))
  expect(await bike()).toBeChecked()
})

it('reopens the finished training it was showing', async () => {
  const user = polling()
  const view = await startTraining(user, completed)
  view.unmount()
  render(<TrainPage />)
  expect(step('Complete')).toHaveAttribute('aria-current', 'step')
  expect(screen.getByTestId('best-prompt')).toHaveTextContent('Be kind.')
})

it('starts the next setup from scratch after a reset', async () => {
  const user = polling()
  const view = await startTraining(user, completed)
  await user.click(screen.getByRole('button', { name: 'Start a new training' }))
  view.unmount()
  render(<TrainPage />)
  expect(step('Agent')).toHaveAttribute('aria-current', 'step')
  expect(await bike()).not.toBeChecked()
})

it('still works when session storage is unavailable', async () => {
  const user = userEvent.setup()
  const fail = () => {
    throw new DOMException('denied', 'SecurityError')
  }
  const get = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(fail)
  const set = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(fail)
  const del = vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(fail)
  await reachScenarios(user)
  expect(step('Scenarios')).toHaveAttribute('aria-current', 'step')
  for (const spy of [get, set, del]) spy.mockRestore()
})

it('retries a failed rubric generation', async () => {
  const user = userEvent.setup()
  api.generateRubric.mockRejectedValueOnce(new Error('Claude is overloaded')).mockResolvedValue({ criteria: ['Clarity'] })
  render(<TrainPage />)
  await user.click(await bike())
  await user.click(nextButton())
  await user.click(screen.getByRole('button', { name: 'Generate from prompt' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Claude is overloaded')
  await user.click(screen.getByRole('button', { name: 'Retry' }))
  expect(await screen.findByRole('textbox', { name: 'Criterion 1' })).toHaveValue('Clarity')
  expect(api.generateRubric).toHaveBeenCalledTimes(2)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('pauses a running training, polls until it is paused, then resumes it to completion', async () => {
  const user = polling()
  await startTraining(user, detail({ current_round: 1, scenarios_done: 2 }))
  api.pauseTraining.mockResolvedValue(detail({ status: 'pausing', current_round: 1, scenarios_done: 2 }))
  await user.click(screen.getByRole('button', { name: 'Pause' }))
  expect(api.pauseTraining).toHaveBeenCalledWith(7)
  expect(await screen.findByRole('button', { name: 'Pausing…' })).toBeDisabled()
  expect(screen.getByRole('status')).toHaveTextContent('pausing')

  api.getTraining.mockResolvedValue(detail({ status: 'paused', current_round: 1, scenarios_done: 3 }))
  await tick()
  expect(screen.getByRole('status')).toHaveTextContent('paused')
  expect(screen.getByText(/step in progress was dropped; resuming redoes it/i)).toBeInTheDocument()
  expect(screen.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '3')
  expect(screen.queryByRole('button', { name: /Paus/ })).not.toBeInTheDocument()
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)

  api.resumeTraining.mockResolvedValue(detail({ status: 'pending', current_round: 1, scenarios_done: 3 }))
  api.getTraining.mockResolvedValue(completed)
  await user.click(screen.getByRole('button', { name: 'Resume' }))
  expect(api.resumeTraining).toHaveBeenCalledWith(7)
  expect(await screen.findByRole('button', { name: 'Pause' })).toBeEnabled() // polling starts once it shows
  await tick()
  expect(step('Complete')).toHaveAttribute('aria-current', 'step')
})

it('pauses a pending training too, and shows a failed pause', async () => {
  const user = polling()
  await startTraining(user, detail({ status: 'pending' }))
  api.pauseTraining.mockRejectedValue(new Error('Only a pending or running training can be paused.'))
  await user.click(screen.getByRole('button', { name: 'Pause' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Only a pending or running training can be paused.')
})

it('retries a failed training from its saved progress', async () => {
  const user = polling()
  await startTraining(user, detail({ status: 'failed', error: 'Claude API error: overloaded', scenarios_done: 4 }))
  expect(screen.getByRole('alert')).toHaveTextContent('Claude API error: overloaded')
  api.resumeTraining.mockResolvedValue(detail({ status: 'pending', scenarios_done: 4 }))
  api.getTraining.mockResolvedValue(detail({ scenarios_done: 5 }))
  await user.click(screen.getByRole('button', { name: 'Retry' }))
  expect(api.resumeTraining).toHaveBeenCalledWith(7)
  expect(await screen.findByRole('button', { name: 'Pause' })).toBeEnabled()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  await tick()
  expect(screen.getByRole('status')).toHaveTextContent('running')
  expect(screen.getByRole('progressbar', { name: 'Scenarios tested' })).toHaveAttribute('aria-valuenow', '5')
})

it('shows why a retry was refused', async () => {
  const user = polling()
  await startTraining(user, detail({ status: 'failed', error: 'Claude API error: overloaded' }))
  api.resumeTraining.mockRejectedValue(new Error('The server is busy with other trainings; try again later.'))
  await user.click(screen.getByRole('button', { name: 'Retry' }))
  expect(await screen.findByText('The server is busy with other trainings; try again later.')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled()
})

it('reopens a training that is pausing', async () => {
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'pausing' }])
  api.getTraining.mockResolvedValue(detail({ status: 'pausing' }))
  render(<TrainPage />)
  expect(await screen.findByRole('button', { name: 'Pausing…' })).toBeDisabled()
  expect(step('Training')).toHaveAttribute('aria-current', 'step')
})

it('reopens the active training even when a newer one has finished', async () => {
  api.listTrainings.mockResolvedValue([{ id: 8, status: 'completed' }, { id: 7, status: 'running' }])
  api.getTraining.mockResolvedValue(detail())
  render(<TrainPage />)
  await waitFor(() => expect(step('Training')).toHaveAttribute('aria-current', 'step'))
  expect(api.getTraining).toHaveBeenCalledWith(7)
  expect(api.getTraining).not.toHaveBeenCalledWith(8)
})

it('refreshes the saved training from the server when the page opens', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 3, training: detail() }))
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'paused' }])
  api.getTraining.mockResolvedValue(detail({ status: 'paused', scenarios_done: 3 }))
  render(<TrainPage />)
  expect(await screen.findByRole('button', { name: 'Resume' })).toBeInTheDocument()
  expect(api.getTraining).toHaveBeenCalledWith(7)
  expect(screen.getByRole('status')).toHaveTextContent('paused')
  expect(step('Training')).toHaveAttribute('aria-current', 'step')
})

it('shows the Complete step when the saved training finished elsewhere', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 3, training: detail({ status: 'paused' }) }))
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'completed' }])
  api.getTraining.mockResolvedValue(completed)
  render(<TrainPage />)
  await waitFor(() => expect(step('Complete')).toHaveAttribute('aria-current', 'step'))
})

it('starts a fresh setup when the saved training was deleted', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 4, agentId: 1, training: completed }))
  api.getTraining.mockRejectedValue(Object.assign(new Error('Training not found'), { status: 404 }))
  render(<TrainPage />)
  await waitFor(() => expect(step('Agent')).toHaveAttribute('aria-current', 'step'))
  expect(await bike()).not.toBeChecked()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('keeps the saved training when refreshing it fails for another reason', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 4, training: completed }))
  api.getTraining.mockRejectedValue(Object.assign(new Error('Server down'), { status: 500 }))
  render(<TrainPage />)
  await waitFor(() => expect(api.getTraining).toHaveBeenCalledWith(7))
  expect(step('Complete')).toHaveAttribute('aria-current', 'step')
  expect(screen.getByTestId('best-prompt')).toHaveTextContent('Be kind.')
})

it('offers no weights to download for a Claude agent', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 4, training: { ...completed, base_model: 'claude' } }))
  api.getTraining.mockResolvedValue({ ...completed, base_model: 'claude' })
  render(<TrainPage />)
  await waitFor(() => expect(api.getTraining).toHaveBeenCalledWith(7))
  expect(screen.getByRole('button', { name: 'Download prompt' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Download weights' })).not.toBeInTheDocument()
})

it('defines no CSS classes that another page stylesheet also defines', () => {
  const dir = import.meta.dirname
  const classes = (file) =>
    new Set(readFileSync(`${dir}/${file}`, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').match(/\.[a-z][\w-]*/gi))
  const shared = classes('../index.css')
  const own = [...classes('TrainPage.css')].filter((c) => !shared.has(c))
  for (const file of readdirSync(dir).filter((f) => f.endsWith('.css') && f !== 'TrainPage.css')) {
    const theirs = classes(file)
    expect(own.filter((c) => theirs.has(c)), file).toEqual([])
  }
})

it('leaves how it works to the signed-out landing page', () => {
  render(<TrainPage />)
  expect(screen.getByRole('heading', { level: 1, name: 'Train an agent' })).toHaveClass('text-center')
  expect(screen.queryByRole('region', { name: 'How it works' })).not.toBeInTheDocument()
})

it('finds an agent: copies a feed post into my agents and selects it', async () => {
  const user = userEvent.setup()
  const copy = { id: 3, name: 'Fixer', base_model: 'qwen-0.5b', prompt: 'Fix.', overall: 90, trainings: 0, model_available: true, image: '<svg/>' }
  api.feed.mockResolvedValue([
    { id: 5, author: 'bea', name: 'Fixer', base_model: 'qwen-0.5b', prompt: 'Fix.', overall: 90, score: 0, my_vote: 0, created_at: '2026-10-05T10:00:00Z' },
  ])
  api.copyPost.mockResolvedValue(copy)
  render(<TrainPage />)
  await bike()
  await user.click(screen.getByRole('radio', { name: 'Find an agent' }))
  api.listAgents.mockResolvedValue([...AGENTS, copy])
  await user.click(await screen.findByRole('button', { name: 'Use this agent' }))
  expect(api.copyPost).toHaveBeenCalledWith(5)
  expect(await screen.findByRole('radio', { name: /Fixer/ })).toBeChecked()
  expect(screen.getByRole('radio', { name: 'Use my own' })).toBeChecked()
  expect(nextButton()).toBeEnabled()
})

it("won't train a copied agent until its base model is downloaded", async () => {
  api.listAgents.mockResolvedValue([{ ...AGENTS[0], model_available: false }])
  const user = userEvent.setup()
  render(<TrainPage />)
  await user.click(await bike())
  expect(screen.getByRole('status')).toHaveTextContent("qwen-0.5b isn't downloaded yet")
  expect(nextButton()).toBeDisabled()
})

it('battles the agent and shows both sides\' grades and which side kept it', async () => {
  const user = userEvent.setup()
  render(<TrainPage battle />)
  expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('Battle your agent')
  await user.click(await bike())
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  const bouts = [
    { message: 'Brakes squeal', champion: 'Bring it in.', challenger: 'Try cleaning the pads first.', grades: [5, 9] },
    { message: 'Price?', champion: 'From $20.', challenger: 'It varies.', grades: [8, 6.5] },
  ]
  const won = detail({ status: 'completed', best_prompt: 'Be helpful.', battle: { winner: 'challenger', grades: [6.5, 7.75], bouts } })
  api.startTraining.mockResolvedValue(won)
  await user.click(screen.getByRole('button', { name: 'Start battle' }))
  expect(api.startTraining).toHaveBeenCalledWith(expect.objectContaining({ agent_id: 1, battle: true, opponent_post_id: null }))
  const steps = within(screen.getByRole('list', { name: 'Battle workflow' }))
  await user.click(steps.getByRole('button', { name: /Result/ }))
  expect(screen.getByTestId('battle-score')).toHaveTextContent('6.5 – 7.8')
  expect(screen.getByText('Mean grade out of 10 over 2 new users')).toBeInTheDocument()
  expect(screen.getByRole('status')).toHaveTextContent('The challenger wins. Your agent now has its prompt and weights.')
  expect(screen.getByText('Winning prompt')).toBeInTheDocument()
  expect(screen.getByText('Scenario 2: 8.0 – 6.5')).toBeInTheDocument()
  expect(screen.getByText('Challenger: Try cleaning the pads first.')).toBeInTheDocument()
})

it('says the agent held when the champion wins, and starts a new battle', async () => {
  const user = userEvent.setup()
  const battle = { winner: 'champion', grades: [7, 7], bouts: [{ message: 'Hi', champion: 'Hello!', challenger: 'Hi.', grades: [7, 7] }] }
  sessionStorage.setItem('trainform_battle_draft', JSON.stringify({ step: 4, training: detail({ status: 'completed', battle }) }))
  api.getTraining.mockResolvedValue(detail({ status: 'completed', battle }))
  render(<TrainPage battle />)
  expect(await screen.findByTestId('battle-score')).toHaveTextContent('7.0 – 7.0')
  expect(screen.getByRole('status')).toHaveTextContent('Your agent holds. It keeps its prompt and weights.')
  await user.click(screen.getByRole('button', { name: 'Start a new battle' }))
  expect(sessionStorage.getItem('trainform_battle_draft')).not.toContain('champion')
})

it('lists past battles with their rounds, and opens one', async () => {
  const user = userEvent.setup()
  const battle = { winner: 'challenger', grades: [6, 8] }
  api.listTrainings.mockResolvedValue([
    { id: 8, status: 'completed', agent_name: 'Bike Repair Agent', created_at: '2026-10-02T10:00:00Z', rounds: 2, battle },
    { id: 6, status: 'completed', battle: null }, // a training
    { id: 5, status: 'completed', agent_name: 'Bike Repair Agent', created_at: '2026-10-01T10:00:00Z', rounds: 0,
      battle: { opponent: { name: 'Rival Repair', author: 'bob' }, winner: 'champion', grades: [9, 7] } },
  ])
  render(<TrainPage battle />)
  const past = within(await screen.findByRole('region', { name: 'Past battles' }))
  const [trained, challenge] = past.getAllByRole('button')
  expect(trained).toHaveTextContent('Bike Repair Agent vs challenger6.0 – 8.0')
  expect(trained).toHaveTextContent('1 battle round · challenger trained for 2 rounds · challenger won')
  expect(challenge).toHaveTextContent('1 battle round · a challenge, no training · Your agent held')

  const bouts = [{ message: 'Hi', champion: 'Hello!', challenger: 'Hi.', grades: [6, 8] }]
  api.getTraining.mockResolvedValue(detail({ id: 8, status: 'completed', rounds: 2, battle: { ...battle, bouts } }))
  await user.click(trained)
  expect(api.getTraining).toHaveBeenCalledWith(8)
  expect(await screen.findByTestId('battle-score')).toHaveTextContent('6.0 – 8.0')
  expect(screen.getByText('1 battle round · challenger trained for 2 rounds')).toBeInTheDocument()
  expect(screen.queryByTestId('rounds-won')).not.toBeInTheDocument() // one round: the grades decide
})

it('shows past battles only on the first and result slides, and sends the battle rounds picked', async () => {
  const user = userEvent.setup()
  api.listTrainings.mockResolvedValue([
    { id: 8, status: 'completed', agent_name: 'Bike Repair Agent', created_at: '2026-10-02T10:00:00Z', rounds: 2,
      battle: { winner: 'champion', grades: [8, 6] } },
  ])
  render(<TrainPage battle />)
  expect(await screen.findByRole('region', { name: 'Past battles' })).toBeInTheDocument()
  await user.click(await bike())
  await user.click(nextButton())
  expect(screen.queryByRole('region', { name: 'Past battles' })).not.toBeInTheDocument()
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  expect(screen.queryByRole('region', { name: 'Past battles' })).not.toBeInTheDocument()
  const rounds = screen.getByRole('combobox', { name: 'Battle rounds' })
  expect([...rounds.options].map((o) => o.text)).toEqual(['1 round', 'Best of 3', 'Best of 5', 'Best of 7'])
  await user.selectOptions(rounds, 'Best of 3')

  const bouts = [9, 5, 5].map((g) => ({ message: 'Hi', champion: 'Hello!', challenger: 'Hey.', grades: [6, g] }))
  const battle = { rounds: 3, winner: 'champion', grades: [6, 6.3], round_grades: [[6, 9], [6, 5], [6, 5]], bouts }
  api.startTraining.mockResolvedValue(detail({ status: 'pending', battle: { rounds: 3 } }))
  api.getTraining.mockResolvedValue(detail({ status: 'completed', battle }))
  await user.click(screen.getByRole('button', { name: 'Start battle' }))
  expect(api.startTraining).toHaveBeenCalledWith(expect.objectContaining({ battle: true, battle_rounds: 3 }))
  expect(screen.queryByRole('region', { name: 'Past battles' })).not.toBeInTheDocument() // while it runs
  expect(await screen.findByTestId('rounds-won', {}, { timeout: 4000 })).toHaveTextContent('Rounds won 2 – 1')
  expect(await screen.findByRole('region', { name: 'Past battles' })).toBeInTheDocument() // refetched once it finished
})

it('keeps trainings and battles apart', async () => {
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'running', battle: true }])
  const { unmount } = render(<TrainPage />)
  await waitFor(() => expect(api.listTrainings).toHaveBeenCalled())
  expect(api.getTraining).not.toHaveBeenCalled()
  unmount()
  api.getTraining.mockResolvedValue(detail({ battle: {} }))
  render(<TrainPage battle />)
  await waitFor(() => expect(api.getTraining).toHaveBeenCalledWith(7))
})

it('says a training is in progress, rather than opening it, when a battle is started', async () => {
  const user = userEvent.setup()
  render(<TrainPage battle />)
  await user.click(await bike())
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  api.startTraining.mockRejectedValue(Object.assign(new Error('You already have a training in progress.'), { status: 409 }))
  api.listTrainings.mockResolvedValue([{ id: 9, status: 'running', battle: false }])
  await user.click(screen.getByRole('button', { name: 'Start battle' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('You already have a training in progress.')
  expect(api.getTraining).not.toHaveBeenCalled()
})

const RIVAL = {
  id: 31, author: 'bob', name: 'Rival Repair', base_model: 'qwen-0.5b', prompt: 'Be brilliant.', overall: 91,
  image: null, caption: null, score: 0, my_vote: 0, created_at: '2026-10-02T10:00:00Z',
}

it('challenges a published agent: no rounds, and its name on the result', async () => {
  api.feed.mockResolvedValue([RIVAL, { ...RIVAL, id: 32, name: 'Big Model', base_model: 'llama-70b' }])
  const user = userEvent.setup()
  render(<TrainPage battle />)
  await user.click(await bike())
  await user.click(screen.getByRole('radio', { name: 'Challenge an agent' }))
  const challenge = await screen.findAllByRole('button', { name: 'Challenge' })
  await user.click(challenge[1]) // another base model: its weights wouldn't fit
  expect(screen.getByRole('status')).toHaveTextContent('Only an agent on qwen-0.5b can battle yours')
  expect(nextButton()).toBeDisabled()
  await user.click(screen.getByRole('button', { name: 'Change' }))
  await user.click((await screen.findAllByRole('button', { name: 'Challenge' }))[0])
  expect(screen.getByText('Rival Repair by bob')).toBeInTheDocument()
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  const rounds = screen.getByRole('spinbutton', { name: 'Training rounds', hidden: true })
  expect(rounds).not.toBeVisible()
  await user.clear(rounds) // left invalid while hidden: the challenge sends no rounds of its own
  expect(screen.getByText('Rival Repair', { selector: 'dd' })).toBeInTheDocument()

  const bouts = [{ message: 'Hi', champion: 'Hello!', challenger: 'Hey there, how can I help?', grades: [6, 9] }]
  const opponent = { post_id: 31, name: 'Rival Repair', author: 'bob' }
  api.startTraining.mockResolvedValue(
    detail({ status: 'completed', battle: { opponent, winner: 'challenger', grades: [6, 9], bouts } }),
  )
  await user.click(screen.getByRole('button', { name: 'Start battle' }))
  expect(api.startTraining).toHaveBeenCalledWith(expect.objectContaining({ battle: true, opponent_post_id: 31, rounds: 1 }))
  await user.click(within(screen.getByRole('list', { name: 'Battle workflow' })).getByRole('button', { name: /Result/ }))
  expect(screen.getByText('Your agent vs Rival Repair by bob')).toBeInTheDocument()
  expect(screen.getByText('Rival Repair by bob wins. Your agent now has its prompt and weights.')).toBeInTheDocument()
  expect(screen.getByTestId('battle-score')).toHaveTextContent('6.0 – 9.0')
  expect(screen.getByText('Rival Repair: Hey there, how can I help?')).toBeInTheDocument()
})

it('shows a battle from before grading as a training rather than crashing', async () => {
  const battle = { winner: 'champion', bouts: [{ message: 'Hi', champion: 'Hello!', challenger: 'Hi.', verdict: 0 }] }
  sessionStorage.setItem('trainform_battle_draft', JSON.stringify({ step: 4, training: detail({ status: 'completed', battle }) }))
  api.getTraining.mockResolvedValue(detail({ status: 'completed', best_score: 7.5, battle }))
  render(<TrainPage battle />)
  expect(await screen.findByTestId('best-score')).toHaveTextContent('7.5')
})

it('battles two of your own agents', async () => {
  const twin = { ...AGENTS[0], id: 3, name: 'Bike Twin' }
  api.listAgents.mockResolvedValue([...AGENTS, twin])
  const user = userEvent.setup()
  render(<TrainPage battle />)
  await user.click(await bike())
  await user.click(screen.getByRole('radio', { name: 'Battle my own agent' }))
  const pick = screen.getByRole('combobox', { name: 'Your opponent' })
  expect(within(pick).queryByRole('option', { name: /Bike Repair Agent/ })).toBeNull() // not itself
  await user.selectOptions(pick, '2') // another base model: its weights wouldn't fit
  expect(screen.getByRole('status')).toHaveTextContent('Only an agent on qwen-0.5b can battle yours')
  expect(nextButton()).toBeDisabled()
  await user.selectOptions(pick, '3')
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  expect(screen.getByText('Bike Twin', { selector: 'dd' })).toBeInTheDocument()
  api.startTraining.mockResolvedValue(detail({ status: 'pending', battle: { rounds: 1 } }))
  api.getTraining.mockResolvedValue(detail({ battle: { rounds: 1 } }))
  await user.click(screen.getByRole('button', { name: 'Start battle' }))
  expect(api.startTraining).toHaveBeenCalledWith(
    expect.objectContaining({ battle: true, opponent_agent_id: 3, opponent_post_id: null, rounds: 1 }),
  )
})
