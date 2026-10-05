import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, saveBlob } from '../api'
import './InspectPage.css'

const fmt = (score) => (score == null ? '—' : String(Math.round(score * 10) / 10))
// Below 860px the panels stack, so a newly opened one can start below the fold.
const reveal = (el) => el?.scrollIntoView({ block: 'nearest' })

// Loads data for `key`, reloading when `version` changes; results for a previous key are ignored, so fast
// clicking never shows stale lists, and a reload keeps showing the current data until the new data arrives.
function useLoad(load, key, version) {
  const [state, setState] = useState({ key: null })
  useEffect(() => {
    if (key == null) return
    let live = true
    load().then(
      (data) => live && setState({ key, data }),
      (error) => live && setState({ key, error }),
    )
    return () => {
      live = false
    }
    // `load` is a fresh closure each render; `key` identifies the request.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [key, version])
  return state.key === key ? state : { loading: key != null }
}

function LoadStatus({ state, loadingText }) {
  return (
    <>
      {state.loading && (
        <p className="row muted" role="status">
          <span className="spinner" /> {loadingText}
        </p>
      )}
      {state.error && (
        <p className="error" role="alert">
          {state.error.message}
        </p>
      )}
    </>
  )
}

function Column({ ref, title, state, loadingText, emptyText, children }) {
  return (
    <section ref={ref} className="inspect-col glass" aria-label={title}>
      <h2 className="inspect-col-title">{title}</h2>
      <LoadStatus state={state} loadingText={loadingText} />
      {state.data && (state.data.length ? <ul className="inspect-list">{children}</ul> : <p className="muted">{emptyText}</p>)}
    </section>
  )
}

// The training's starting prompt beside the prompt of its best-scoring round (null until a round finishes).
function Prompts({ state }) {
  const t = state.data
  return (
    <section className="inspect-prompts glass" aria-label="Prompts">
      <h2 className="inspect-col-title">Prompts</h2>
      <LoadStatus state={state} loadingText="Loading prompts…" />
      {t && (
        <div className="prompt-pair">
          <section className="prompt-block" aria-label="Original prompt">
            <h3>Original prompt</h3>
            <pre tabIndex={0}>{t.initial_prompt}</pre>
          </section>
          <section className="prompt-block prompt-best" aria-label="Highest-scoring prompt">
            <h3>
              Highest-scoring prompt{' '}
              {t.best_prompt != null && <span className="prompt-score">Score {fmt(t.best_score)}/10</span>}
            </h3>
            {t.best_prompt != null ? <pre tabIndex={0}>{t.best_prompt}</pre> : <p className="muted">No finished round yet</p>}
          </section>
        </div>
      )}
    </section>
  )
}

function Option({ selected, onSelect, children }) {
  return (
    <li>
      <button type="button" className="inspect-option" aria-pressed={selected} onClick={onSelect}>
        {children}
      </button>
    </li>
  )
}

function Bar({ value }) {
  return (
    <span className="inspect-bar" aria-hidden="true">
      <span style={{ width: `${value * 10}%` }} />
    </span>
  )
}

function Ratings({ label, items }) {
  return (
    <ul className="criteria" aria-label={label}>
      {items.map(([name, value]) => (
        <li key={name}>
          <span>{name}</span>
          <Bar value={value} />
          <span className="criteria-value">{value}/10</span>
        </li>
      ))}
    </ul>
  )
}

function Transcript({ scenario }) {
  const ref = useRef(null)
  useEffect(() => {
    reveal(ref.current)
    ref.current.focus({ preventScroll: true })
  }, [scenario.id])
  return (
    <section
      ref={ref}
      tabIndex={-1}
      className="inspect-transcript glass card"
      aria-label={`Scenario ${scenario.scenario_number}`}
    >
      <h2 className="inspect-col-title">Scenario {scenario.scenario_number}</h2>
      <div className="chat">
        <article className="bubble bubble-persona" aria-label="Persona">
          <span className="bubble-who">Persona</span>
          <p>{scenario.input}</p>
        </article>
        <article className="bubble bubble-agent" aria-label="Agent">
          <span className="bubble-who">Agent</span>
          <p>{scenario.output}</p>
        </article>
      </div>

      <div className="inspect-score">
        <p
          className="score-ring"
          style={{ '--pct': scenario.score * 10 }}
          role="img"
          aria-label={`Overall score ${fmt(scenario.score)} out of 10`}
        >
          <strong>{fmt(scenario.score)}</strong>
          <span>/10</span>
        </p>
        <Ratings label="Criterion scores" items={scenario.scores.map((s) => [s.criterion, s.score])} />
      </div>

      <details className="inspect-details">
        <summary>Persona profile</summary>
        <Ratings label="Persona ratings" items={scenario.persona.attributes.map((a) => [a.criterion, a.rating])} />
        <pre>{scenario.persona.prompt}</pre>
      </details>
      <details className="inspect-details">
        <summary>Agent prompt</summary>
        <pre>{scenario.prompt}</pre>
      </details>
    </section>
  )
}

export default function InspectPage() {
  const [trainingId, setTrainingId] = useState(null)
  const [round, setRound] = useState(null)
  const [scenarioId, setScenarioId] = useState(null)
  const [version, setVersion] = useState(0)
  const [downloadError, setDownloadError] = useState('')
  const roundsRef = useRef(null)
  const scenariosRef = useRef(null)

  const trainings = useLoad(api.listTrainings, 'all', version)
  const rounds = useLoad(() => api.listRounds(trainingId), trainingId, version)
  const detail = useLoad(() => api.getTraining(trainingId), trainingId, version)
  const scenarios = useLoad(
    () => api.listScenarios(trainingId, round),
    round == null ? null : `${trainingId}/${round}`,
    version,
  )
  const scenario = scenarios.data?.find((s) => s.id === scenarioId)
  const selected = trainings.data?.find((t) => t.id === trainingId)

  // While any training is in progress, reload everything 2 s after each trainings list arrives.
  const active = trainings.data?.some((t) => t.status === 'pending' || t.status === 'running')
  useEffect(() => {
    if (!active) return
    const timer = setTimeout(() => setVersion((v) => v + 1), 2000)
    return () => clearTimeout(timer)
  }, [trainings, active])

  useEffect(() => reveal(roundsRef.current), [trainingId])
  useEffect(() => reveal(scenariosRef.current), [round])

  const save = (fetchBlob, filename) => {
    setDownloadError('')
    fetchBlob(trainingId).then(
      (blob) => saveBlob(blob, filename),
      (e) => setDownloadError(e.message),
    )
  }
  const pickTraining = (id) => {
    setTrainingId(id)
    setRound(null)
    setScenarioId(null)
    setDownloadError('')
  }
  const pickRound = (r) => {
    setRound(r)
    setScenarioId(null)
  }

  return (
    <div className="stack" data-testid="inspect-page">
      <h1 className="page-title">Inspect</h1>
      <div className="inspect-cols">
        <Column
          title="Trainings"
          state={trainings}
          loadingText="Loading trainings…"
          emptyText={
            <>
              No trainings yet — start one on the <Link to="/train">Train page</Link>
            </>
          }
        >
          {trainings.data?.map((t) => (
            <Option key={t.id} selected={t.id === trainingId} onSelect={() => pickTraining(t.id)}>
              <span className="option-main">
                #{t.id} · {t.base_model}
              </span>
              <span className={`badge badge-${t.status}`}>{t.status}</span>
              <span className="option-meta">{new Date(t.created_at).toLocaleDateString()}</span>
              <span className="option-score" title="Best score">
                {fmt(t.best_score)}
              </span>
            </Option>
          ))}
        </Column>

        {trainingId != null && (
          <Column ref={roundsRef} title="Rounds" state={rounds} loadingText="Loading rounds…" emptyText="No rounds finished yet.">
            {rounds.data?.map((r) => (
              <Option key={r.round} selected={r.round === round} onSelect={() => pickRound(r.round)}>
                <span className="option-main">Round {r.round}</span>
                <span className="option-meta">{r.scenarios} scenarios</span>
                <span className="option-score" title="Mean score">
                  {fmt(r.mean_score)}
                </span>
              </Option>
            ))}
          </Column>
        )}

        {round != null && (
          <Column ref={scenariosRef} title="Scenarios" state={scenarios} loadingText="Loading scenarios…" emptyText="No scenarios in this round.">
            {scenarios.data?.map((s) => (
              <Option key={s.id} selected={s.id === scenarioId} onSelect={() => setScenarioId(s.id)}>
                <span className="option-main">Scenario {s.scenario_number}</span>
                <Bar value={s.score} />
                <span className="option-score">{fmt(s.score)}</span>
              </Option>
            ))}
          </Column>
        )}
      </div>

      {trainingId != null && <Prompts state={detail} />}

      {selected?.status === 'completed' && (
        <div className="row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => save(api.downloadPrompt, `training-${trainingId}-prompt.txt`)}
          >
            Download prompt
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => save(api.downloadWeights, `training-${trainingId}-lora.zip`)}
          >
            Download weights
          </button>
        </div>
      )}
      {downloadError && (
        <p className="error" role="alert">
          {downloadError}
        </p>
      )}

      {scenario && <Transcript scenario={scenario} />}
    </div>
  )
}
