import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { readdirSync, readFileSync } from 'node:fs'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { api, saveBlob } from '../api'
import TrainPage from './TrainPage'

vi.mock('../api', () => ({
  api: {
    me: vi.fn(),
    searchHubModels: vi.fn(),
    downloadHubModel: vi.fn(),
    uploadModel: vi.fn(),
    listLocalModels: vi.fn(),
    listTrainings: vi.fn(),
    generatePrompt: vi.fn(),
    generateRubric: vi.fn(),
    startTraining: vi.fn(),
    getTraining: vi.fn(),
    pauseTraining: vi.fn(),
    resumeTraining: vi.fn(),
    downloadPrompt: vi.fn(),
    downloadWeights: vi.fn(),
  },
  saveBlob: vi.fn(),
}))

const HUB_MODEL = { id: 'Qwen/Qwen2.5-0.5B', downloads: 512, likes: 34, last_modified: '2026-09-01T00:00:00Z' }
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
  api.listLocalModels.mockResolvedValue([{ name: 'qwen-0.5b', source: 'hub' }])
  api.listTrainings.mockResolvedValue([])
  api.me.mockResolvedValue({ claude_available: false })
})
afterEach(() => vi.useRealTimers())

const stepper = () => within(screen.getByRole('list', { name: 'Training workflow' }))
const step = (name) => stepper().getByRole('button', { name: new RegExp(name) })
const nextButton = () => screen.getByRole('button', { name: 'Next' })

async function reachScenarios(user) {
  const view = render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Be helpful.')
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  return view
}

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
  latest_prompt: 'Be brief.',
  best_prompt: 'Be kind.',
})

it('unlocks each step only once the previous ones are valid', async () => {
  const user = userEvent.setup()
  render(<TrainPage />)
  expect(screen.getByTestId('train-page')).toBeInTheDocument()
  expect(step('Model')).toHaveAttribute('aria-current', 'step')
  expect(nextButton()).toBeDisabled()
  expect(step('Prompt')).toBeDisabled()

  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  expect(step('Prompt')).toBeEnabled()
  expect(step('Rubric')).toBeDisabled()
  await user.click(nextButton())
  expect(step('Prompt')).toHaveAttribute('aria-current', 'step')
  // Next is now disabled, so focus must move into the new slide rather than drop to <body>.
  expect(screen.getByRole('region', { name: 'Prompt' })).toHaveFocus()

  const prompt = screen.getByRole('textbox', { name: 'Agent prompt' })
  await user.type(prompt, '   ')
  expect(nextButton()).toBeDisabled()
  await user.type(prompt, 'Be helpful.')
  await user.click(nextButton())

  expect(nextButton()).toBeDisabled()
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  expect(nextButton()).toBeEnabled()
  expect(step('Scenarios')).toBeEnabled()
  expect(step('Training')).toBeDisabled()

  await user.click(screen.getByRole('button', { name: 'Back' }))
  expect(step('Prompt')).toHaveAttribute('aria-current', 'step')
  await user.click(step('Model'))
  expect(step('Model')).toHaveAttribute('aria-current', 'step')
  expect(step('Prompt')).toHaveAccessibleName(/done/)
  expect(step('Rubric')).toHaveAccessibleName(/done/)
  expect(step('Training').closest('li')).toHaveAttribute('data-state', 'locked')
})

it('searches the hub with the chosen sort', async () => {
  const user = userEvent.setup()
  api.searchHubModels.mockResolvedValue([HUB_MODEL])
  render(<TrainPage />)
  await user.type(screen.getByRole('searchbox', { name: 'Search models' }), 'qwen')
  await user.selectOptions(screen.getByRole('combobox', { name: 'Sort by' }), 'Latest')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(api.searchHubModels).toHaveBeenCalledWith('qwen', 'lastModified')

  const card = (await screen.findByRole('heading', { name: HUB_MODEL.id })).closest('li')
  expect(card).toHaveTextContent('512')
  expect(card).toHaveTextContent('34')
  expect(within(card).getByRole('button', { name: `Download ${HUB_MODEL.id}` })).toBeInTheDocument()

  await user.selectOptions(screen.getByRole('combobox', { name: 'Sort by' }), 'Most downloaded')
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(api.searchHubModels).toHaveBeenLastCalledWith('qwen', 'downloads')
})

it('shows hub errors inline', async () => {
  const user = userEvent.setup()
  api.searchHubModels.mockRejectedValue(new Error('Hub unreachable'))
  render(<TrainPage />)
  await user.click(screen.getByRole('button', { name: 'Search' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Hub unreachable')
})

it('clears stale model errors once a later action succeeds', async () => {
  const user = userEvent.setup()
  const model = { name: 'Qwen2.5-0.5B', source: 'hub' }
  api.listLocalModels.mockRejectedValueOnce(new Error('Server down')).mockResolvedValue([model])
  api.searchHubModels.mockResolvedValue([HUB_MODEL])
  api.downloadHubModel.mockRejectedValueOnce(new Error('Disk full')).mockResolvedValue(model)
  render(<TrainPage />)
  expect(await screen.findByRole('alert')).toHaveTextContent('Server down')

  await user.click(screen.getByRole('button', { name: 'Search' }))
  await user.click(await screen.findByRole('button', { name: `Download ${HUB_MODEL.id}` }))
  await waitFor(() => expect(screen.getByText('Disk full')).toBeInTheDocument())
  await user.click(screen.getByRole('button', { name: 'Search' }))
  await waitFor(() => expect(screen.queryByText('Disk full')).not.toBeInTheDocument())

  await user.click(screen.getByRole('button', { name: `Download ${HUB_MODEL.id}` }))
  expect(await screen.findByRole('radio', { name: /Qwen2.5-0.5B/ })).toBeChecked()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('downloads a hub model, refreshes local models and selects it', async () => {
  const user = userEvent.setup()
  let finish
  api.searchHubModels.mockResolvedValue([HUB_MODEL])
  api.downloadHubModel.mockReturnValue(new Promise((resolve) => (finish = resolve)))
  api.listLocalModels.mockResolvedValueOnce([]).mockResolvedValueOnce([{ name: 'Qwen2.5-0.5B', source: 'hub' }])
  render(<TrainPage />)
  await user.click(screen.getByRole('button', { name: 'Search' }))
  const download = await screen.findByRole('button', { name: `Download ${HUB_MODEL.id}` })
  await user.click(download)

  expect(api.downloadHubModel).toHaveBeenCalledWith(HUB_MODEL.id)
  expect(download).toBeDisabled()
  expect(download).toHaveTextContent('Downloading')

  await act(async () => finish({ name: 'Qwen2.5-0.5B', source: 'hub' }))
  expect(await screen.findByRole('radio', { name: /Qwen2.5-0.5B/ })).toBeChecked()
  expect(api.listLocalModels).toHaveBeenCalledTimes(2)
  expect(step('Prompt')).toBeEnabled()
})

it('uploads a zipped model and selects it', async () => {
  const user = userEvent.setup()
  api.uploadModel.mockResolvedValue({ name: 'mine', source: 'upload' })
  api.listLocalModels.mockResolvedValueOnce([]).mockResolvedValueOnce([{ name: 'mine', source: 'upload' }])
  render(<TrainPage />)
  expect(screen.getByRole('radio', { name: 'Hugging Face Hub' })).toBeChecked()
  await user.click(screen.getByRole('radio', { name: 'Upload' }))
  const file = new File(['zip'], 'model.zip', { type: 'application/zip' })
  await user.upload(screen.getByLabelText('Model archive (.zip)'), file)
  await user.type(screen.getByRole('textbox', { name: 'Model name' }), 'mine')
  await user.click(screen.getByRole('button', { name: 'Upload model' }))

  expect(api.uploadModel).toHaveBeenCalledWith(file, 'mine')
  expect(await screen.findByRole('radio', { name: /mine/ })).toBeChecked()
})

const helper = () => within(screen.getByRole('radiogroup', { name: 'Train with' }))

it('trains with Claude by default when it is linked, or with the chosen local model', async () => {
  const user = userEvent.setup()
  api.me.mockResolvedValue({ claude_available: true })
  api.generatePrompt.mockResolvedValue({ prompt: 'You are a support bot.' })
  const view = render(<TrainPage />)
  await waitFor(() => expect(helper().getByRole('radio', { name: 'Claude' })).toBeChecked())
  expect(screen.queryByRole('link', { name: 'Account' })).not.toBeInTheDocument()
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A support bot')
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(api.generatePrompt).toHaveBeenLastCalledWith('A support bot', 'qwen-0.5b', 'claude')

  await user.click(step('Model'))
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

it('generates a prompt from a description into the editable prompt', async () => {
  const user = userEvent.setup()
  api.generatePrompt.mockResolvedValue({ prompt: 'You are a support bot.' })
  render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A support bot')
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))

  expect(api.generatePrompt).toHaveBeenCalledWith('A support bot', 'qwen-0.5b', 'local') // no Claude: the local model
  const prompt = screen.getByRole('textbox', { name: 'Agent prompt' })
  await waitFor(() => expect(prompt).toHaveValue('You are a support bot.'))
  await user.type(prompt, ' Be kind.')
  expect(prompt).toHaveValue('You are a support bot. Be kind.')
  expect(nextButton()).toBeEnabled()
})

it('asks before a generated prompt replaces a written one', async () => {
  const user = userEvent.setup()
  api.generatePrompt.mockResolvedValue({ prompt: 'You are a support bot.' })
  render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  const prompt = screen.getByRole('textbox', { name: 'Agent prompt' })
  await user.type(prompt, 'My own prompt.')
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A support bot')

  const confirm = vi.spyOn(window, 'confirm').mockReturnValueOnce(false).mockReturnValueOnce(true)
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(confirm).toHaveBeenCalledOnce()
  expect(api.generatePrompt).not.toHaveBeenCalled()
  expect(prompt).toHaveValue('My own prompt.')

  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  await waitFor(() => expect(prompt).toHaveValue('You are a support bot.'))
  confirm.mockRestore()
})

it('shows no source badge for a model without one', async () => {
  api.listLocalModels.mockResolvedValue([{ name: 'demo', source: null }])
  render(<TrainPage />)
  const option = (await screen.findByRole('radio', { name: 'demo' })).closest('label')
  expect(option.querySelector('[data-slot="badge"]')).toBeNull()
})

it('generates the rubric and supports add, edit, delete and confirmed regenerate', async () => {
  const user = userEvent.setup()
  api.generateRubric
    .mockResolvedValueOnce({ criteria: ['Clarity', 'Empathy'] })
    .mockResolvedValueOnce({ criteria: ['Speed'] })
  render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Be helpful.')
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
  expect(api.startTraining).toHaveBeenCalledWith({
    base_model: 'qwen-0.5b',
    prompt: 'Be helpful.',
    rubric: ['Clarity'],
    scenarios: 8,
    rounds: 4,
    helper: 'local',
  })
  await waitFor(() => expect(step('Training')).toHaveAttribute('aria-current', 'step'))
  for (const name of ['Model', 'Prompt', 'Rubric', 'Scenarios']) expect(step(name)).toBeDisabled()
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
  expect(step('Model')).toBeDisabled()
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

  expect(step('Model')).toHaveAttribute('aria-current', 'step')
  expect(step('Prompt')).toBeDisabled()
  const model = await screen.findByRole('radio', { name: /qwen-0.5b/ })
  expect(model).not.toBeChecked()
  await user.click(model)
  await user.click(nextButton())
  expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toHaveValue('')
  const calls = api.getTraining.mock.calls.length
  await tick(6000)
  expect(api.getTraining).toHaveBeenCalledTimes(calls)
})

it('keeps the setup when the page is left and reopened', async () => {
  const user = userEvent.setup()
  const view = render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Be helpful.')
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'New criterion' }), 'Clarity{Enter}')
  await user.click(nextButton())
  await user.clear(screen.getByRole('spinbutton', { name: 'Training rounds' }))
  await user.type(screen.getByRole('spinbutton', { name: 'Training rounds' }), '4')
  view.unmount() // what navigating to Inspect or Account does

  render(<TrainPage />)
  expect(step('Scenarios')).toHaveAttribute('aria-current', 'step')
  expect(screen.getByRole('spinbutton', { name: 'Training rounds' })).toHaveValue(4)
  expect(screen.getByTestId('total-runs')).toHaveTextContent('20')
  await user.click(step('Rubric'))
  expect(screen.getByRole('textbox', { name: 'Criterion 1' })).toHaveValue('Clarity')
  await user.click(step('Prompt'))
  expect(screen.getByRole('textbox', { name: 'Agent prompt' })).toHaveValue('Be helpful.')
  await user.click(step('Model'))
  expect(await screen.findByRole('radio', { name: /qwen-0.5b/ })).toBeChecked()
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
  expect(step('Model')).toHaveAttribute('aria-current', 'step')
  expect(await screen.findByRole('radio', { name: /qwen-0.5b/ })).not.toBeChecked()
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

it('retries a failed prompt generation without asking again', async () => {
  const user = userEvent.setup()
  api.generatePrompt.mockRejectedValueOnce(new Error('Claude is overloaded')).mockResolvedValue({ prompt: 'You are a support bot.' })
  render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  const prompt = screen.getByRole('textbox', { name: 'Agent prompt' })
  await user.type(prompt, 'My own prompt.')
  await user.click(screen.getByRole('radio', { name: 'Describe my agent' }))
  await user.type(screen.getByRole('textbox', { name: 'Agent description' }), 'A support bot')
  const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true)
  await user.click(screen.getByRole('button', { name: 'Generate prompt' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Claude is overloaded')
  expect(prompt).toHaveValue('My own prompt.')

  await user.click(screen.getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(prompt).toHaveValue('You are a support bot.'))
  expect(api.generatePrompt).toHaveBeenCalledTimes(2)
  expect(api.generatePrompt).toHaveBeenLastCalledWith('A support bot', 'qwen-0.5b', 'local')
  expect(confirm).toHaveBeenCalledOnce()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
  confirm.mockRestore()
})

it('retries a failed rubric generation', async () => {
  const user = userEvent.setup()
  api.generateRubric.mockRejectedValueOnce(new Error('Claude is overloaded')).mockResolvedValue({ criteria: ['Clarity'] })
  render(<TrainPage />)
  await user.click(await screen.findByRole('radio', { name: /qwen-0.5b/ }))
  await user.click(nextButton())
  await user.type(screen.getByRole('textbox', { name: 'Agent prompt' }), 'Be helpful.')
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
  expect(screen.getByText(/progress is saved/i)).toBeInTheDocument()
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
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 4, training: detail() }))
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'paused' }])
  api.getTraining.mockResolvedValue(detail({ status: 'paused', scenarios_done: 3 }))
  render(<TrainPage />)
  expect(await screen.findByRole('button', { name: 'Resume' })).toBeInTheDocument()
  expect(api.getTraining).toHaveBeenCalledWith(7)
  expect(screen.getByRole('status')).toHaveTextContent('paused')
  expect(step('Training')).toHaveAttribute('aria-current', 'step')
})

it('shows the Complete step when the saved training finished elsewhere', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 4, training: detail({ status: 'paused' }) }))
  api.listTrainings.mockResolvedValue([{ id: 7, status: 'completed' }])
  api.getTraining.mockResolvedValue(completed)
  render(<TrainPage />)
  await waitFor(() => expect(step('Complete')).toHaveAttribute('aria-current', 'step'))
})

it('starts a fresh setup when the saved training was deleted', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 5, baseModel: 'qwen-0.5b', training: completed }))
  api.getTraining.mockRejectedValue(Object.assign(new Error('Training not found'), { status: 404 }))
  render(<TrainPage />)
  await waitFor(() => expect(step('Model')).toHaveAttribute('aria-current', 'step'))
  expect(await screen.findByRole('radio', { name: /qwen-0.5b/ })).not.toBeChecked()
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})

it('keeps the saved training when refreshing it fails for another reason', async () => {
  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 5, training: completed }))
  api.getTraining.mockRejectedValue(Object.assign(new Error('Server down'), { status: 500 }))
  render(<TrainPage />)
  await waitFor(() => expect(api.getTraining).toHaveBeenCalledWith(7))
  expect(step('Complete')).toHaveAttribute('aria-current', 'step')
  expect(screen.getByTestId('best-prompt')).toHaveTextContent('Be kind.')
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

it('explains the pipeline and its output during setup, not once training has started', async () => {
  const view = render(<TrainPage />)
  const how = screen.getByRole('region', { name: 'How it works' })
  expect(within(how).getByText(/LoRA weights/)).toBeInTheDocument()
  view.unmount()

  sessionStorage.setItem('trainform_train_draft', JSON.stringify({ step: 5, training: completed }))
  api.getTraining.mockResolvedValue(completed)
  render(<TrainPage />)
  expect(screen.queryByRole('region', { name: 'How it works' })).not.toBeInTheDocument()
})
