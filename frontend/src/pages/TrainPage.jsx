import { useEffect, useRef, useState } from 'react'
import { api, saveBlob } from '../api'
import './TrainPage.css'

const STEPS = [
  ['Model', 'Pick the base model to fine-tune. Download one from Hugging Face or upload your own.'],
  ['Prompt', 'Write the system prompt your agent starts from, or describe the agent and let Claude draft it.'],
  ['Rubric', 'List the criteria every simulated conversation is scored against, 1 to 10.'],
  ['Scenarios', 'Choose how many simulated personas each round tests, and how many rounds to train.'],
  ['Training', 'Each round tests the agent against new personas, scores it and trains a better version.'],
  ['Complete', 'Training finished. Take the best prompt and LoRA weights with you.'],
]
const WINDOWS = [
  ['n', 'Last n runs'],
  ['n/2', 'Last n/2'],
  ['n/4', 'Last n/4'],
  ['last', 'Last run'],
]
const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
const inRange = (n, max) => Number.isInteger(n) && n >= 1 && n <= max
const score = (s) => (s == null ? '—' : s.toFixed(1))

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

const Spinner = () => <span className="spinner" aria-hidden="true" />
const Alert = ({ children }) =>
  children ? (
    <p className="error" role="alert">
      {children}
    </p>
  ) : null

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
    <div className="model-step">
      <div className="stack">
        <fieldset className="train-toggle">
          <legend className="sr-only">Model source</legend>
          {[
            ['hub', 'Hugging Face Hub'],
            ['upload', 'Upload'],
          ].map(([key, label]) => (
            <label key={key}>
              <input type="radio" name="train-model-source" checked={tab === key} onChange={() => setTab(key)} />
              <span>{label}</span>
            </label>
          ))}
        </fieldset>

        <div hidden={tab !== 'hub'} className="stack">
          <form role="search" className="hub-search" onSubmit={onSearch}>
            <input
              type="search"
              className="input"
              aria-label="Search models"
              placeholder="Search models, e.g. Qwen"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select className="input" aria-label="Sort by" value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="downloads">Most downloaded</option>
              <option value="lastModified">Latest</option>
            </select>
            <button className="btn btn-primary" disabled={hubBusy}>
              {hubBusy && !downloading && <Spinner />}Search
            </button>
          </form>
          <Alert>{hubError}</Alert>
          {results === null ? (
            <p className="muted">Search the Hub to find a base model. Small instruct models train fastest.</p>
          ) : results.length === 0 ? (
            <p className="muted">No models match that search. Try a broader name.</p>
          ) : (
            <ul className="hub-results">
              {results.map((m) => (
                <li key={m.id} className="hub-card">
                  <h3 className="hub-id">{m.id}</h3>
                  <dl className="hub-stats">
                    <div>
                      <dt>Downloads</dt>
                      <dd>{compact.format(m.downloads ?? 0)}</dd>
                    </div>
                    <div>
                      <dt>Likes</dt>
                      <dd>{compact.format(m.likes ?? 0)}</dd>
                    </div>
                    <div>
                      <dt>Updated</dt>
                      <dd>{m.last_modified ? new Date(m.last_modified).toLocaleDateString() : '—'}</dd>
                    </div>
                  </dl>
                  <button
                    type="button"
                    className="btn"
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
                      'Download'
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <form hidden={tab !== 'upload'} className="stack" onSubmit={upload}>
          <p className="muted">Zip a Hugging Face model folder (config, tokenizer and weights) and upload it.</p>
          <div className="field">
            <label className="label" htmlFor="train-upload-file">
              Model archive (.zip)
            </label>
            <input
              id="train-upload-file"
              type="file"
              accept=".zip,application/zip"
              className="input"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="train-upload-name">
              Model name
            </label>
            <input id="train-upload-name" className="input" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <Alert>{uploadError}</Alert>
          <div className="row">
            <button className="btn btn-primary" disabled={!file || !name.trim() || uploading}>
              {uploading && <Spinner />}Upload model
            </button>
          </div>
        </form>
      </div>

      <fieldset className="model-list">
        <legend className="label">Local models</legend>
        {!models && !listError && <Spinner />}
        <Alert>{listError}</Alert>
        {models?.length === 0 && (
          <p className="muted">No local models yet. Download one from the Hub or upload a zip.</p>
        )}
        {models?.map((m) => (
          <label key={m.name} className="model-option">
            <input type="radio" name="train-base-model" checked={value === m.name} onChange={() => onChange(m.name)} />
            <span className="model-name">{m.name}</span>
            {m.source && <span className="badge">{m.source}</span>}
          </label>
        ))}
      </fieldset>
    </div>
  )
}

function PromptStep({ prompt, onChange }) {
  const [mode, setMode] = useState('write')
  const [description, setDescription] = useState('')
  const [run, busy, error] = useAction()
  const generate = () => {
    if (prompt.trim() && !window.confirm('Replace your prompt with a generated one?')) return
    run(async () => onChange((await api.generatePrompt(description.trim())).prompt))
  }

  return (
    <div className="stack">
      <fieldset className="train-toggle">
        <legend className="sr-only">How to create the prompt</legend>
        {[
          ['write', 'Write my own'],
          ['describe', 'Describe my agent'],
        ].map(([key, label]) => (
          <label key={key}>
            <input type="radio" name="train-prompt-mode" checked={mode === key} onChange={() => setMode(key)} />
            <span>{label}</span>
          </label>
        ))}
      </fieldset>
      {mode === 'describe' && (
        <div className="field">
          <label className="label" htmlFor="train-description">
            Agent description
          </label>
          <textarea
            id="train-description"
            className="input"
            placeholder="A support agent for a bike shop that handles returns and repair bookings."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <Alert>{error}</Alert>
          <div className="row">
            <button type="button" className="btn" disabled={!description.trim() || busy} onClick={generate}>
              {busy && <Spinner />}Generate prompt
            </button>
          </div>
        </div>
      )}
      <div className="field">
        <label className="label" htmlFor="train-prompt">
          Agent prompt
        </label>
        <textarea
          id="train-prompt"
          className="input prompt-input"
          placeholder="You are a friendly support agent for…"
          value={prompt}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    </div>
  )
}

function RubricStep({ prompt, criteria, onChange }) {
  const [draft, setDraft] = useState('')
  const [run, busy, error] = useAction()
  const generate = () => {
    if (criteria.length && !window.confirm('Replace your criteria with a newly generated set?')) return
    run(async () => onChange((await api.generateRubric(prompt.trim())).criteria))
  }
  const add = (e) => {
    e.preventDefault()
    if (!draft.trim()) return
    onChange([...criteria, draft.trim()])
    setDraft('')
  }

  return (
    <div className="stack">
      <div className="row">
        <button type="button" className="btn" disabled={!prompt.trim() || busy} onClick={generate}>
          {busy && <Spinner />}
          {criteria.length ? 'Regenerate' : 'Generate from prompt'}
        </button>
      </div>
      <Alert>{error}</Alert>
      {criteria.length > 0 && (
        <ol className="rubric-list">
          {criteria.map((c, i) => (
            <li key={i} className="rubric-item">
              <input
                className="input"
                aria-label={`Criterion ${i + 1}`}
                value={c}
                onChange={(e) => onChange(criteria.map((x, j) => (j === i ? e.target.value : x)))}
              />
              <button
                type="button"
                className="btn btn-ghost btn-danger"
                aria-label={`Delete criterion ${i + 1}`}
                onClick={() => onChange(criteria.filter((_, j) => j !== i))}
              >
                Delete
              </button>
            </li>
          ))}
        </ol>
      )}
      <form className="rubric-add" onSubmit={add}>
        <input
          className="input"
          aria-label="New criterion"
          placeholder="Add a criterion, e.g. Resolves the issue in one reply"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button className="btn" disabled={!draft.trim()}>
          Add criterion
        </button>
      </form>
    </div>
  )
}

function Meter({ label, value, max }) {
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="meter-count">
          {value} / {max}
        </span>
      </div>
      <div
        className="progress"
        role="progressbar"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-valuenow={value}
      >
        <div className="progress-bar" style={{ width: `${max ? (value / max) * 100 : 0}%` }} />
      </div>
    </div>
  )
}

function TrainingStep({ training, pollError, onReset }) {
  if (!training) return null
  return (
    <div className="stack">
      <p role="status" className="train-status" data-status={training.status}>
        {training.status}
      </p>
      {training.status === 'failed' && (
        <>
          <Alert>{training.error || 'Training stopped without an error message.'}</Alert>
          <div className="row">
            <button type="button" className="btn btn-primary" onClick={onReset}>
              Start a new training
            </button>
          </div>
        </>
      )}
      <Alert>{pollError && `Lost contact with the server (${pollError}). Retrying…`}</Alert>
      <div className="meters">
        <Meter label="Scenarios tested" value={training.scenarios_done} max={training.total_scenarios} />
        <Meter
          label="Rounds completed"
          value={training.status === 'completed' ? training.rounds : Math.max(training.current_round - 1, 0)}
          max={training.rounds}
        />
      </div>
      <h3 className="section-title">Latest model scores</h3>
      <div className="score-grid">
        {WINDOWS.map(([key, title]) => {
          const w = training.score_windows.find((x) => x.label === key)
          return (
            <div key={key} role="group" aria-labelledby={`train-window-${key}`} className="score-card">
              <span id={`train-window-${key}`} className="score-title">
                {title}
              </span>
              <span className="score-value">
                {score(w?.score)}
                <small>/ 10</small>
              </span>
              <span className="muted">{w ? `${w.runs} ${w.runs === 1 ? 'run' : 'runs'}` : 'No runs yet'}</span>
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
    <div className="stack">
      <div className="best">
        <span className="label">Best score</span>
        <span className="best-score" data-testid="best-score">
          {score(training.best_score)}
          <small>/ 10</small>
        </span>
      </div>
      <h3 className="section-title">Best prompt</h3>
      <pre className="prompt-preview" data-testid="best-prompt">
        {training.best_prompt}
      </pre>
      <Alert>{error}</Alert>
      <div className="row">
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => save(api.downloadPrompt, `training-${id}-prompt.txt`)}
        >
          Download prompt
        </button>
        <button
          type="button"
          className="btn btn-primary"
          disabled={busy}
          onClick={() => save(api.downloadWeights, `training-${id}-lora.zip`)}
        >
          Download weights
        </button>
        <button type="button" className="btn btn-ghost" onClick={onReset}>
          Start a new training
        </button>
      </div>
    </div>
  )
}

function Dashboard({ onReset }) {
  const [step, setStep] = useState(0)
  const [baseModel, setBaseModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [rubric, setRubric] = useState([])
  const [scenarios, setScenarios] = useState('5')
  const [rounds, setRounds] = useState('3')
  const [training, setTraining] = useState(null)
  const [pollError, setPollError] = useState('')
  const [runStart, starting, startError] = useAction()
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

  // Reopen a training still in progress, e.g. after navigating away or reloading.
  useEffect(() => {
    let live = true
    api
      .listTrainings()
      .then(async ([latest]) => {
        if (latest?.status !== 'pending' && latest?.status !== 'running') return
        const t = await api.getTraining(latest.id)
        if (!live) return
        setTraining(t)
        setStep(4)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  const id = training?.id
  const polling = training?.status === 'pending' || training?.status === 'running'
  useEffect(() => {
    if (!polling) return
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
  }, [id, polling])

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
      }
      // 409: one is already in progress (e.g. started in another tab), so open it as the resume effect does.
      const t = await api.startTraining(config).catch(async (e) => {
        if (e.status !== 409) throw e
        const [latest] = await api.listTrainings()
        return api.getTraining(latest.id)
      })
      setTraining(t)
      setStep(4)
    })

  const content = [
    <ModelStep key="model" value={baseModel} onChange={setBaseModel} />,
    <PromptStep key="prompt" prompt={prompt} onChange={setPrompt} />,
    <RubricStep key="rubric" prompt={prompt} criteria={rubric} onChange={setRubric} />,
    <div key="scenarios" className="stack">
      <div className="grid-2">
        <div className="field">
          <label className="label" htmlFor="train-scenarios">
            Simulated scenarios per round
          </label>
          <input
            id="train-scenarios"
            type="number"
            min="1"
            max="100"
            step="1"
            className="input"
            value={scenarios}
            onChange={(e) => setScenarios(e.target.value)}
          />
        </div>
        <div className="field">
          <label className="label" htmlFor="train-rounds">
            Training rounds
          </label>
          <input
            id="train-rounds"
            type="number"
            min="1"
            max="50"
            step="1"
            className="input"
            value={rounds}
            onChange={(e) => setRounds(e.target.value)}
          />
        </div>
      </div>
      {!valid[3] && <p className="muted">Scenarios must be 1 to 100 and rounds 1 to 50.</p>}
      <dl className="summary">
        <div>
          <dt>Base model</dt>
          <dd>{baseModel || '—'}</dd>
        </div>
        <div>
          <dt>Criteria</dt>
          <dd>{criteria.length}</dd>
        </div>
        <div>
          <dt>Total runs</dt>
          <dd data-testid="total-runs">{valid[3] ? Number(scenarios) * Number(rounds) : '—'}</dd>
        </div>
      </dl>
      <Alert>{startError}</Alert>
    </div>,
    <TrainingStep key="training" training={training} pollError={pollError} onReset={onReset} />,
    <CompleteStep key="complete" training={training} onReset={onReset} />,
  ]

  return (
    <div className="train-page" data-testid="train-page">
      <h1 className="page-title">Train an agent</h1>

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
              <div className="slide-card glass card">
                <h2 id={`train-slide-${i}`} className="slide-title">
                  {name}
                </h2>
                <p className="muted slide-blurb">{blurb}</p>
                {content[i]}
              </div>
            </section>
          ))}
        </div>
      </div>

      {step < 4 && (
        <div className="train-footer">
          <button type="button" className="btn btn-ghost" disabled={step === 0} onClick={() => setStep(step - 1)}>
            Back
          </button>
          {step < 3 ? (
            <button type="button" className="btn btn-primary" disabled={!valid[step]} onClick={() => setStep(step + 1)}>
              Next
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              disabled={!valid.every(Boolean) || starting}
              onClick={start}
            >
              {starting && <Spinner />}Start training
            </button>
          )}
        </div>
      )}
    </div>
  )
}

export default function TrainPage() {
  const [run, setRun] = useState(0)
  return <Dashboard key={run} onReset={() => setRun((r) => r + 1)} />
}
