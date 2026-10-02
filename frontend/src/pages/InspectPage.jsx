import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import './InspectPage.css'

const fmt = (score) => (score == null ? '—' : String(Math.round(score * 10) / 10))

// Loads data for `key`; results for a previous key are ignored, so fast clicking never shows stale lists.
function useLoad(load, key) {
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
  }, [key])
  return state.key === key ? state : { loading: key != null }
}

function Column({ title, state, loadingText, emptyText, children }) {
  return (
    <section className="inspect-col glass" aria-label={title}>
      <h2 className="inspect-col-title">{title}</h2>
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
      {state.data && (state.data.length ? <ul className="inspect-list">{children}</ul> : <p className="muted">{emptyText}</p>)}
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
  return (
    <section className="inspect-transcript glass card" aria-label={`Scenario ${scenario.scenario_number}`}>
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

  const trainings = useLoad(api.listTrainings, 'all')
  const rounds = useLoad(() => api.listRounds(trainingId), trainingId)
  const scenarios = useLoad(() => api.listScenarios(trainingId, round), round == null ? null : `${trainingId}/${round}`)
  const scenario = scenarios.data?.find((s) => s.id === scenarioId)

  const pickTraining = (id) => {
    setTrainingId(id)
    setRound(null)
    setScenarioId(null)
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
          <Column title="Rounds" state={rounds} loadingText="Loading rounds…" emptyText="No rounds finished yet.">
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
          <Column title="Scenarios" state={scenarios} loadingText="Loading scenarios…" emptyText="No scenarios in this round.">
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

      {scenario && <Transcript scenario={scenario} />}
    </div>
  )
}
