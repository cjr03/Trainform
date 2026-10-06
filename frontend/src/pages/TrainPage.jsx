import { CircleAlert, Download, Loader2, Pause, Play, Plus, RotateCcw, Sparkles, Trash2, Upload } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'
import { api, saveBlob } from '../api'
import './TrainPage.css'

const STEPS = [
  ['Model', 'Choose whether Claude or your local model runs the helper agents, then pick the base model to fine-tune. Download one from Hugging Face or upload your own.'],
  ['Prompt', 'Write the system prompt your agent starts from, or describe the agent and let the helper draft it.'],
  ['Rubric', 'List the criteria every simulated conversation is scored against, 1 to 10.'],
  ['Scenarios', 'Choose how many simulated personas each round tests, and how many rounds to train.'],
  ['Training', 'Each round tests the agent against new personas, scores it and trains a better version.'],
  ['Complete', 'Training finished. Take the best prompt and LoRA weights with you.'],
]
const WINDOWS = [
  ['n', 'Total runs'],
  ['n/2', 'Last 50%'],
  ['n/4', 'Last 25%'],
  ['last', 'Last run'],
]
const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
const inRange = (n, max) => Number.isInteger(n) && n >= 1 && n <= max
const score = (s) => (s == null ? '—' : s.toFixed(1))
const ACTIVE = ['pending', 'running', 'pausing'] // a user has at most one training in these

// The setup and the training on screen, kept for this tab so leaving the page or reloading loses neither.
// Signing in or out clears it (clearToken). Storage can be unavailable (e.g. blocked); the page then just forgets.
const DRAFT_KEY = 'trainform_train_draft'
const loadDraft = () => {
  try {
    return JSON.parse(sessionStorage.getItem(DRAFT_KEY)) ?? {}
  } catch {
    return {}
  }
}
const storeDraft = (draft) => {
  try {
    if (draft) sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    else sessionStorage.removeItem(DRAFT_KEY)
  } catch {
    // Unavailable or full: nothing to keep.
  }
}

function useAction() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (fn) => {
    setBusy(true)
    setError('')
    try {
      await fn()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }
  return [run, busy, error]
}

const Spinner = () => <Loader2 className="animate-spin" />
const ErrorAlert = ({ children }) =>
  children ? (
    <Alert variant="destructive">
      <CircleAlert />
      <AlertDescription className="text-destructive">{children}</AlertDescription>
    </Alert>
  ) : null

// Pill-shaped choices: one radio group with a visible radio in each pill.
function Choice({ options, ...props }) {
  return (
    <RadioGroup className="flex flex-wrap gap-2" {...props}>
      {options.map(([key, text, props]) => (
        <label
          key={key}
          className="flex cursor-pointer items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors has-disabled:cursor-not-allowed has-disabled:opacity-50 has-data-[state=checked]:border-primary has-data-[state=checked]:bg-primary/10 has-data-[state=checked]:text-primary"
        >
          <RadioGroupItem value={key} {...props} />
          {text}
        </label>
      ))}
    </RadioGroup>
  )
}

function ModelStep({ value, onChange }) {
  const [tab, setTab] = useState('hub')
  const [models, setModels] = useState(null)
  const [listError, setListError] = useState('')
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('downloads')
  const [results, setResults] = useState(null)
  const [downloading, setDownloading] = useState('')
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [runHub, hubBusy, hubError] = useAction()
  const [runUpload, uploading, uploadError] = useAction()

  const refresh = async () => {
    setModels(await api.listLocalModels())
    setListError('')
  }
  useEffect(() => {
    api.listLocalModels().then(setModels, (e) => setListError(e.message))
  }, [])

  const onSearch = (e) => {
    e.preventDefault()
    runHub(async () => setResults(await api.searchHubModels(search.trim(), sort)))
  }
  const download = async (id) => {
    setDownloading(id)
    await runHub(async () => {
      const model = await api.downloadHubModel(id)
      await refresh()
      onChange(model.name)
    })
    setDownloading('')
  }
  const upload = (e) => {
    e.preventDefault()
    runUpload(async () => {
      const model = await api.uploadModel(file, name.trim())
      await refresh()
      onChange(model.name)
    })
  }

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div className="grid gap-4">
        <Choice
          aria-label="Model source"
          value={tab}
          onValueChange={setTab}
          options={[
            ['hub', 'Hugging Face Hub'],
            ['upload', 'Upload'],
          ]}
        />

        <div hidden={tab !== 'hub'} className="grid gap-4">
          <form role="search" className="grid gap-2 sm:grid-cols-[1fr_auto_auto]" onSubmit={onSearch}>
            <Input
              type="search"
              aria-label="Search models"
              placeholder="Search models, e.g. Qwen"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select
              aria-label="Sort by"
              className="h-9 rounded-md border border-input bg-input/30 px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [&_option]:bg-background"
              value={sort}
              onChange={(e) => setSort(e.target.value)}
            >
              <option value="downloads">Most downloaded</option>
              <option value="lastModified">Latest</option>
            </select>
            <Button disabled={hubBusy}>
              {hubBusy && !downloading && <Spinner />}Search
            </Button>
          </form>
          <ErrorAlert>{hubError}</ErrorAlert>
          {results === null ? (
            <p className="m-0 text-sm text-muted-foreground">
              Search the Hub to find a base model. Small instruct models train fastest.
            </p>
          ) : results.length === 0 ? (
            <p className="m-0 text-sm text-muted-foreground">No models match that search. Try a broader name.</p>
          ) : (
            <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(min(230px,100%),1fr))] gap-3 p-0">
              {results.map((m) => (
                <li key={m.id} className="flex flex-col gap-3 rounded-lg border bg-background/40 p-4">
                  <h3 className="m-0 font-sans text-sm font-semibold tracking-normal [overflow-wrap:anywhere]">
                    {m.id}
                  </h3>
                  <dl className="m-0 grid grid-cols-3 gap-2 text-sm tabular-nums">
                    {[
                      ['Downloads', compact.format(m.downloads ?? 0)],
                      ['Likes', compact.format(m.likes ?? 0)],
                      ['Updated', m.last_modified ? new Date(m.last_modified).toLocaleDateString() : '—'],
                    ].map(([term, detail]) => (
                      <div key={term}>
                        <dt className="text-xs text-muted-foreground">{term}</dt>
                        <dd className="m-0">{detail}</dd>
                      </div>
                    ))}
                  </dl>
                  <Button
                    type="button"
                    variant="outline"
                    className="mt-auto"
                    aria-label={`Download ${m.id}`}
                    disabled={Boolean(downloading)}
                    onClick={() => download(m.id)}
                  >
                    {downloading === m.id ? (
                      <>
                        <Spinner />
                        Downloading…
                      </>
                    ) : (
                      <>
                        <Download />
                        Download
                      </>
                    )}
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <form hidden={tab !== 'upload'} className="grid gap-4" onSubmit={upload}>
          <p className="m-0 text-sm text-muted-foreground">
            Zip a Hugging Face model folder (config, tokenizer and weights) and upload it.
          </p>
          <div className="grid gap-2">
            <Label htmlFor="train-upload-file">Model archive (.zip)</Label>
            <Input
              id="train-upload-file"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="train-upload-name">Model name</Label>
            <Input id="train-upload-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <ErrorAlert>{uploadError}</ErrorAlert>
          <div>
            <Button disabled={!file || !name.trim() || uploading}>
              {uploading ? <Spinner /> : <Upload />}Upload model
            </Button>
          </div>
        </form>
      </div>

      <div className="grid gap-3 rounded-lg border bg-background/40 p-4">
        <span id="train-local-models" className="text-sm font-medium">
          Local models
        </span>
        {!models && !listError && <Spinner />}
        <ErrorAlert>{listError}</ErrorAlert>
        {models?.length === 0 && (
          <p className="m-0 text-sm text-muted-foreground">
            No local models yet. Download one from the Hub or upload a zip.
          </p>
        )}
        <RadioGroup aria-labelledby="train-local-models" value={value} onValueChange={onChange} className="gap-1">
          {models?.map((m) => (
            <label
              key={m.name}
              className="flex cursor-pointer items-center gap-3 rounded-md border border-transparent px-3 py-2.5 text-sm transition-colors hover:bg-accent/50 has-data-[state=checked]:border-border-strong has-data-[state=checked]:bg-primary/10"
            >
              <RadioGroupItem value={m.name} />
              <span className="mr-auto [overflow-wrap:anywhere]">{m.name}</span>
              {m.source && <Badge variant="outline">{m.source}</Badge>}
            </label>
          ))}
        </RadioGroup>
      </div>
    </div>
  )
}

function PromptStep({ model, helper, prompt, onChange }) {
  const [mode, setMode] = useState('write')
  const [description, setDescription] = useState('')
  const [run, busy, error] = useAction()
  const request = () => run(async () => onChange((await api.generatePrompt(description.trim(), model, helper)).prompt))
  const generate = () => {
    if (prompt.trim() && !window.confirm('Replace your prompt with a generated one?')) return
    request()
  }

  return (
    <div className="grid gap-4">
      <Choice
        aria-label="How to create the prompt"
        value={mode}
        onValueChange={setMode}
        options={[
          ['write', 'Write my own'],
          ['describe', 'Describe my agent'],
        ]}
      />
      {mode === 'describe' && (
        <div className="grid gap-2">
          <Label htmlFor="train-description">Agent description</Label>
          <Textarea
            id="train-description"
            placeholder="A support agent for a bike shop that handles returns and repair bookings."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <ErrorAlert>{error}</ErrorAlert>
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" disabled={!description.trim() || busy} onClick={generate}>
              {busy ? <Spinner /> : <Sparkles />}Generate prompt
            </Button>
            {error && (
              <Button type="button" variant="ghost" disabled={!description.trim() || busy} onClick={request}>
                Retry
              </Button>
            )}
          </div>
        </div>
      )}
      <div className="grid gap-2">
        <Label htmlFor="train-prompt">Agent prompt</Label>
        <Textarea
          id="train-prompt"
          className="min-h-60 font-mono"
          placeholder="You are a friendly support agent for…"
          value={prompt}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  )
}

function RubricStep({ model, helper, prompt, criteria, onChange }) {
  const [draft, setDraft] = useState('')
  const [run, busy, error] = useAction()
  const request = () => run(async () => onChange((await api.generateRubric(prompt.trim(), model, helper)).criteria))
  const generate = () => {
    if (criteria.length && !window.confirm('Replace your criteria with a newly generated set?')) return
    request()
  }
  const add = (e) => {
    e.preventDefault()
    if (!draft.trim()) return
    onChange([...criteria, draft.trim()])
    setDraft('')
  }

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" disabled={!prompt.trim() || busy} onClick={generate}>
          {busy ? <Spinner /> : <Sparkles />}
          {criteria.length ? 'Regenerate' : 'Generate from prompt'}
        </Button>
        {error && (
          <Button type="button" variant="ghost" disabled={!prompt.trim() || busy} onClick={request}>
            Retry
          </Button>
        )}
      </div>
      <ErrorAlert>{error}</ErrorAlert>
      {criteria.length > 0 && (
        <ol className="m-0 grid list-none gap-2 p-0">
          {criteria.map((c, i) => (
            <li key={i} className="grid grid-cols-[2rem_1fr_auto] items-center gap-2">
              <span className="text-center font-display text-sm text-brand" aria-hidden="true">
                {i + 1}
              </span>
              <Input
                aria-label={`Criterion ${i + 1}`}
                value={c}
                onChange={(e) => onChange(criteria.map((x, j) => (j === i ? e.target.value : x)))}
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="text-destructive hover:text-destructive"
                aria-label={`Delete criterion ${i + 1}`}
                onClick={() => onChange(criteria.filter((_, j) => j !== i))}
              >
                <Trash2 />
              </Button>
            </li>
          ))}
        </ol>
      )}
      <form className="grid gap-2 sm:grid-cols-[1fr_auto] sm:pl-10" onSubmit={add}>
        <Input
          aria-label="New criterion"
          placeholder="Add a criterion, e.g. Resolves the issue in one reply"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <Button variant="outline" disabled={!draft.trim()}>
          <Plus />
          Add criterion
        </Button>
      </form>
    </div>
  )
}

function Meter({ label, value, max }) {
  return (
    <div className="grid gap-2">
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span className="font-display text-brand tabular-nums">
          {value} / {max}
        </span>
      </div>
      {/* The ui Progress keeps `value` for the bar, so the counts go on the progressbar directly. */}
      <Progress
        aria-label={label}
        aria-valuemax={max}
        aria-valuenow={value}
        value={max ? (value / max) * 100 : 0}
        className="h-2.5"
      />
    </div>
  )
}

const STATUS_DOT = {
  running: 'bg-brand animate-pulse',
  pausing: 'bg-brand-2 animate-pulse',
  paused: 'bg-brand-2',
  completed: 'bg-success',
  failed: 'bg-destructive',
}

// `busy` and `error` belong to the pause/resume request; resuming (Retry, after a failure) continues from saved progress.
function TrainingStep({ training, pollError, onReset, onPause, onResume, busy, error }) {
  if (!training) return null
  const { status } = training
  return (
    <div className="grid gap-5">
      <p role="status" className="m-0 inline-flex w-fit items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm font-semibold capitalize">
        <span className={cn('size-2 rounded-full bg-muted-foreground', STATUS_DOT[status])} />
        {status}
      </p>
      {status === 'failed' && <ErrorAlert>{training.error || 'Training stopped without an error message.'}</ErrorAlert>}
      {status === 'paused' && (
        <p className="m-0 text-sm text-muted-foreground">Paused. Its progress is saved, so it resumes where it stopped.</p>
      )}
      <ErrorAlert>{error}</ErrorAlert>
      {status !== 'completed' && (
        <div className="flex flex-wrap gap-2">
          {(status === 'pending' || status === 'running') && (
            <Button type="button" variant="outline" disabled={busy} onClick={onPause}>
              <Pause />
              Pause
            </Button>
          )}
          {status === 'pausing' && (
            <Button type="button" variant="outline" disabled>
              <Spinner />
              Pausing…
            </Button>
          )}
          {(status === 'paused' || status === 'failed') && (
            <>
              <Button type="button" disabled={busy} onClick={onResume}>
                {busy ? <Spinner /> : status === 'paused' ? <Play /> : <RotateCcw />}
                {status === 'paused' ? 'Resume' : 'Retry'}
              </Button>
              <Button type="button" variant="ghost" onClick={onReset}>
                Start a new training
              </Button>
            </>
          )}
        </div>
      )}
      <ErrorAlert>{pollError && `Lost contact with the server (${pollError}). Retrying…`}</ErrorAlert>
      <div className="grid gap-5 sm:grid-cols-2">
        <Meter label="Scenarios tested" value={training.scenarios_done} max={training.total_scenarios} />
        <Meter
          label="Rounds completed"
          value={training.status === 'completed' ? training.rounds : Math.max(training.current_round - 1, 0)}
          max={training.rounds}
        />
      </div>
      <h3 className="m-0 text-base">Practice scores</h3>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {WINDOWS.map(([key, title]) => {
          const w = training.score_windows.find((x) => x.label === key)
          return (
            <div
              key={key}
              role="group"
              aria-labelledby={`train-window-${key}`}
              className="flex flex-col gap-1 rounded-lg border bg-background/40 p-4"
            >
              <span id={`train-window-${key}`} className="text-sm text-muted-foreground">
                {title}
              </span>
              <span className="font-display text-3xl leading-tight tabular-nums">
                {score(w?.score)}
                <small className="ml-1.5 text-sm text-muted-foreground">/ 10</small>
              </span>
              <span className="text-sm text-muted-foreground">
                {w ? `${w.runs} ${w.runs === 1 ? 'run' : 'runs'}` : 'No runs yet'}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function CompleteStep({ training, onReset }) {
  const [run, busy, error] = useAction()
  if (!training) return null
  const { id } = training
  const save = (fetchBlob, filename) => run(async () => saveBlob(await fetchBlob(id), filename))

  return (
    <div className="grid gap-4">
      <div className="flex flex-col items-center gap-1 pt-3 pb-2">
        <span className="text-sm text-muted-foreground">Best score</span>
        <span
          className="font-display text-6xl leading-tight font-bold tabular-nums text-gradient sm:text-7xl"
          data-testid="best-score"
        >
          {score(training.best_score)}
          <small className="ml-2 text-lg text-muted-foreground">/ 10</small>
        </span>
        {training.baseline_score != null && (
          <span className="text-sm text-muted-foreground">
            {score(training.baseline_score)} before training, on the same eval users
          </span>
        )}
      </div>
      <h3 className="m-0 text-base">Best prompt</h3>
      <pre
        className="m-0 max-h-80 overflow-auto rounded-lg border bg-background/60 p-4 text-sm whitespace-pre-wrap [overflow-wrap:anywhere]"
        data-testid="best-prompt"
      >
        {training.best_prompt}
      </pre>
      <ErrorAlert>{error}</ErrorAlert>
      <div className="flex flex-wrap gap-2">
        <Button type="button" disabled={busy} onClick={() => save(api.downloadPrompt, `training-${id}-prompt.txt`)}>
          <Download />
          Download prompt
        </Button>
        <Button type="button" disabled={busy} onClick={() => save(api.downloadWeights, `training-${id}-lora.zip`)}>
          <Download />
          Download weights
        </Button>
        <Button type="button" variant="ghost" onClick={onReset}>
          Start a new training
        </Button>
      </div>
    </div>
  )
}

function Dashboard({ onReset }) {
  const [draft] = useState(loadDraft)
  const [step, setStep] = useState(draft.step ?? 0)
  const [baseModel, setBaseModel] = useState(draft.baseModel ?? '')
  const [helper, setHelper] = useState(draft.helper ?? null) // 'claude' | 'local'; null: the server picks
  const [claudeAvailable, setClaudeAvailable] = useState(null)
  const [prompt, setPrompt] = useState(draft.prompt ?? '')
  const [rubric, setRubric] = useState(draft.rubric ?? [])
  const [scenarios, setScenarios] = useState(draft.scenarios ?? '5')
  const [rounds, setRounds] = useState(draft.rounds ?? '3')
  // A running one is refreshed by the effects below; a finished one no longer changes.
  const [training, setTraining] = useState(draft.training ?? null)
  const [pollError, setPollError] = useState('')
  const [runStart, starting, startError] = useAction()
  const [runControl, controlling, controlError] = useAction()
  const slides = useRef([])
  const shown = useRef(step)

  const criteria = rubric.map((c) => c.trim()).filter(Boolean)
  const valid = [
    Boolean(baseModel),
    Boolean(prompt.trim()),
    criteria.length > 0,
    inRange(Number(scenarios), 100) && inRange(Number(rounds), 50),
  ]
  const completed = training?.status === 'completed'
  const reachable = (i) => (training ? i === 4 || (i === 5 && completed) : i < 4 && valid.slice(0, i).every(Boolean))
  // Scenarios has valid defaults, so it only counts as done once training starts.
  const done = (i) => (training ? i < 4 || (i === 4 && completed) : i < 3 && i !== step && valid[i] && reachable(i))
  const state = (i) => (i === step ? 'current' : done(i) ? 'done' : reachable(i) ? 'todo' : 'locked')

  useEffect(
    () => storeDraft({ step, baseModel, helper, prompt, rubric, scenarios, rounds, training }),
    [step, baseModel, helper, prompt, rubric, scenarios, rounds, training],
  )

  // Claude by default when linked; only the local model when not.
  useEffect(() => {
    api.me().then(
      (me) => {
        setClaudeAvailable(me.claude_available)
        setHelper((h) => (me.claude_available ? (h ?? 'claude') : 'local'))
      },
      () => {}, // unknown: the choice stays unset, so the server's default (Claude if linked) applies
    )
  }, [])

  // Show the training as the server has it: the one in progress (e.g. after navigating away, reloading or resuming it
  // from Inspect), else the draft's, which may have been paused, resumed or deleted elsewhere since.
  useEffect(() => {
    let live = true
    api
      .listTrainings()
      .then(async (list) => {
        const id = list.find((t) => ACTIVE.includes(t.status))?.id ?? draft.training?.id
        if (id == null) return
        const t = await api.getTraining(id)
        if (!live) return
        setTraining(t)
        // A finished training opens on Complete, unless it was already finished and shown on Training.
        setStep((s) => (t.status !== 'completed' ? 4 : draft.training?.status === 'completed' ? s : 5))
      })
      .catch((e) => {
        if (live && e.status === 404) onReset() // the draft's training was deleted
      })
    return () => {
      live = false
    }    // Runs once, when the page opens.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const id = training?.id
  // Restarting on every status change drops a poll already in flight, so it can't undo a pause it predates.
  const status = training?.status
  useEffect(() => {
    if (!ACTIVE.includes(status)) return
    let live = true
    let timer
    const poll = async () => {
      try {
        const t = await api.getTraining(id)
        if (!live) return
        setTraining(t)
        setPollError('')
        if (t.status === 'completed') setStep(5)
      } catch (e) {
        if (live) setPollError(e.message)
      }
      if (live) timer = setTimeout(poll, 2000)
    }
    timer = setTimeout(poll, 2000)
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [id, status])

  // Move focus into the new slide when the control that changed it has gone (or is now inert or disabled).
  useEffect(() => {
    if (shown.current === step) return
    shown.current = step
    const active = document.activeElement
    if (!active || active === document.body || active.disabled || active.closest('[inert]'))
      slides.current[step]?.focus({ preventScroll: true })
  }, [step])

  const start = () =>
    runStart(async () => {
      const config = {
        base_model: baseModel,
        prompt: prompt.trim(),
        rubric: criteria,
        scenarios: Number(scenarios),
        rounds: Number(rounds),
        helper,
      }
      // 409: one is already in progress (e.g. started in another tab), so open it as the resume effect does.
      const t = await api.startTraining(config).catch(async (e) => {
        if (e.status !== 409) throw e
        const active = (await api.listTrainings()).find((t) => ACTIVE.includes(t.status))
        if (!active) throw e // it finished in the meantime
        return api.getTraining(active.id)
      })
      setTraining(t)
      setStep(4)
    })

  const content = [
    <div key="model" className="grid gap-6">
      <div className="grid gap-2">
        <span className="text-sm font-medium" id="train-helper">
          Train with
        </span>
        <Choice
          aria-labelledby="train-helper"
          value={helper ?? ''}
          onValueChange={setHelper}
          options={[
            [
              'claude',
              'Claude',
              {
                disabled: !claudeAvailable,
                'aria-describedby': claudeAvailable === false ? 'train-helper-hint' : undefined,
              },
            ],
            ['local', 'Local model'],
          ]}
        />
        {claudeAvailable === false && (
          <p id="train-helper-hint" className="m-0 text-sm text-muted-foreground">
            Link your Claude API key on the{' '}
            <Link to="/account" className="text-primary underline-offset-4 hover:underline">
              Account
            </Link>{' '}
            page to train with Claude.
          </p>
        )}
      </div>
      <ModelStep value={baseModel} onChange={setBaseModel} />
    </div>,
    <PromptStep key="prompt" model={baseModel} helper={helper} prompt={prompt} onChange={setPrompt} />,
    <RubricStep key="rubric" model={baseModel} helper={helper} prompt={prompt} criteria={rubric} onChange={setRubric} />,
    <div key="scenarios" className="grid gap-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="grid gap-2">
          <Label htmlFor="train-scenarios">Simulated scenarios per round</Label>
          <Input
            id="train-scenarios"
            type="number"
            min="1"
            max="100"
            step="1"
            value={scenarios}
            onChange={(e) => setScenarios(e.target.value)}
          />
        </div>
        <div className="grid gap-2">
          <Label htmlFor="train-rounds">Training rounds</Label>
          <Input
            id="train-rounds"
            type="number"
            min="1"
            max="50"
            step="1"
            value={rounds}
            onChange={(e) => setRounds(e.target.value)}
          />
        </div>
      </div>
      {!valid[3] && <p className="m-0 text-sm text-muted-foreground">Scenarios must be 1 to 100 and rounds 1 to 50.</p>}
      <dl className="m-0 grid grid-cols-3 gap-2 rounded-lg border p-4">
        {[
          ['Base model', baseModel || '—'],
          ['Criteria', criteria.length],
          ['Total runs', valid[3] ? Number(scenarios) * Number(rounds) : '—', 'total-runs'],
        ].map(([term, detail, testId]) => (
          <div key={term}>
            <dt className="text-xs text-muted-foreground">{term}</dt>
            <dd className="m-0 text-lg tabular-nums [overflow-wrap:anywhere]" data-testid={testId}>
              {detail}
            </dd>
          </div>
        ))}
      </dl>
      <ErrorAlert>{startError}</ErrorAlert>
    </div>,
    <TrainingStep
      key="training"
      training={training}
      pollError={pollError}
      onReset={onReset}
      onPause={() => runControl(async () => setTraining(await api.pauseTraining(id)))}
      onResume={() => runControl(async () => setTraining(await api.resumeTraining(id)))}
      busy={controlling}
      error={controlError}
    />,
    <CompleteStep key="complete" training={training} onReset={onReset} />,
  ]

  return (
    <div className="train-page" data-testid="train-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Train an agent</h1>

      <ol className="stepper" aria-label="Training workflow" style={{ '--progress': step / (STEPS.length - 1) }}>
        {STEPS.map(([name], i) => (
          <li key={name} className="stepper-item" data-state={state(i)}>
            <button
              type="button"
              className="stepper-node"
              aria-current={i === step ? 'step' : undefined}
              disabled={i !== step && !reachable(i)}
              onClick={() => setStep(i)}
            >
              <span className="stepper-dot" aria-hidden="true">
                {done(i) ? '✓' : i + 1}
              </span>
              <span className="stepper-name">{name}</span>
              {done(i) && <span className="sr-only"> (done)</span>}
            </button>
          </li>
        ))}
      </ol>

      <div className="slides">
        <div className="slides-track" style={{ transform: `translateX(${-100 * step}%)` }}>
          {STEPS.map(([name, blurb], i) => (
            <section
              key={name}
              ref={(el) => {
                slides.current[i] = el
              }}
              className="slide"
              tabIndex={-1}
              aria-hidden={i !== step}
              inert={i !== step}
              aria-labelledby={`train-slide-${i}`}
            >
              <Card className="gap-5 max-sm:py-5">
                <CardHeader className="max-sm:px-5">
                  <h2 id={`train-slide-${i}`} className="m-0 text-xl">
                    {name}
                  </h2>
                  <CardDescription className="max-w-prose">{blurb}</CardDescription>
                </CardHeader>
                <CardContent className="max-sm:px-5">{content[i]}</CardContent>
              </Card>
            </section>
          ))}
        </div>
      </div>

      {step < 4 && (
        <div className="flex justify-between">
          <Button type="button" variant="ghost" disabled={step === 0} onClick={() => setStep(step - 1)}>
            Back
          </Button>
          {step < 3 ? (
            <Button type="button" disabled={!valid[step]} onClick={() => setStep(step + 1)}>
              Next
            </Button>
          ) : (
            <Button type="button" disabled={!valid.every(Boolean) || starting} onClick={start}>
              {starting ? <Spinner /> : <Play />}Start training
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

export default function TrainPage() {
  const [run, setRun] = useState(0)
  const reset = () => {
    storeDraft(null)
    setRun((r) => r + 1)
  }
  return <Dashboard key={run} onReset={reset} />
}
