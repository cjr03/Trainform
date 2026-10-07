import { CircleAlert, Download, Loader2, Pause, Play, RotateCcw, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import { BattleResult } from '@/components/battles'
import { AgentImage, Overall } from '@/components/shared'
import { cn } from '@/lib/utils'
import { api, saveBlob } from '../api'
import './InspectPage.css'

const ACTIVE = ['pending', 'running', 'pausing'] // in progress: can't be resumed or deleted
const HARD = 7 // a persona rating at or above this pushes the agent hard on that criterion
const fmt = (score) => (score == null ? '—' : String(Math.round(score * 10) / 10))
// Below 860px the panels stack, so a newly opened one can start below the fold. Returns nothing: effects call it,
// and current Chromium's scrollIntoView returns a promise, which React would take for the effect's cleanup.
const reveal = (el) => {
  el?.scrollIntoView({ block: 'nearest' })
}

const TITLE = 'm-0 text-sm tracking-wider text-muted-foreground'
const PRE = 'm-0 rounded-md border bg-[rgb(5_6_15/0.6)] px-3.5 py-3 text-[0.85rem] whitespace-pre-wrap wrap-anywhere'
const PROMPT_PRE = cn(PRE, 'max-h-[360px] overflow-y-auto')
const OPTION_MAIN = 'font-semibold wrap-anywhere'
const OPTION_META = 'text-[0.8rem] text-muted-foreground'
const OPTION_SCORE = 'col-start-2 row-span-2 row-start-1 font-display text-lg text-brand'
const BUBBLE = 'max-w-[min(72ch,85%)] rounded-[18px] border px-4 py-3 max-[860px]:max-w-[92%]'
const SAID = 'mt-1 mb-0 whitespace-pre-wrap wrap-anywhere'
const DETAILS = 'border-t pt-3 [&[open]>summary]:mb-3'

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

function ErrorAlert({ children }) {
  return (
    <Alert variant="destructive">
      <CircleAlert aria-hidden="true" />
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}

function LoadStatus({ state, loadingText }) {
  return (
    <>
      {state.loading && (
        <p className="m-0 flex items-center gap-2 text-muted-foreground" role="status">
          <Loader2 className="size-4 animate-spin" aria-hidden="true" /> {loadingText}
        </p>
      )}
      {state.error && <ErrorAlert>{state.error.message}</ErrorAlert>}
    </>
  )
}

function Column({ ref, title, state, loadingText, emptyText, children }) {
  return (
    <Card ref={ref} role="region" aria-label={title} className="min-h-30 gap-3 p-4">
      <h2 className={TITLE}>{title}</h2>
      <LoadStatus state={state} loadingText={loadingText} />
      {state.data &&
        (state.data.length ? (
          <ul className="m-0 flex max-h-[420px] list-none flex-col gap-1.5 overflow-y-auto p-0">{children}</ul>
        ) : (
          <p className="m-0 text-muted-foreground">{emptyText}</p>
        ))}
    </Card>
  )
}

// The training's starting prompt beside the prompt of its best-scoring round (null until a round finishes).
function Prompts({ state }) {
  const t = state.data
  return (
    <Card role="region" aria-label="Prompts" className="gap-3 p-4">
      <h2 className={TITLE}>Prompts</h2>
      <LoadStatus state={state} loadingText="Loading prompts…" />
      {t && (
        <div className="grid gap-4 min-[860px]:grid-cols-2">
          <section aria-label="Original prompt" className="min-w-0">
            <h3 className="mb-2 text-[0.95rem]">Original prompt</h3>
            <pre tabIndex={0} className={PROMPT_PRE}>
              {t.initial_prompt}
            </pre>
          </section>
          <section aria-label="Highest-scoring prompt" className="min-w-0">
            <h3 className="mb-2 flex items-baseline justify-between gap-3 text-[0.95rem]">
              Highest-scoring prompt{' '}
              {t.best_prompt != null && <span className="font-display text-brand">Score {fmt(t.best_score)}/10</span>}
            </h3>
            {t.best_prompt != null ? (
              <pre tabIndex={0} className={cn(PROMPT_PRE, 'border-border-strong shadow-glow')}>
                {t.best_prompt}
              </pre>
            ) : (
              <p className="m-0 text-muted-foreground">No finished round yet</p>
            )}
          </section>
        </div>
      )}
    </Card>
  )
}

function Option({ selected, onSelect, children }) {
  return (
    <li>
      <button
        type="button"
        className="grid w-full cursor-pointer grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 rounded-md border border-l-3 border-transparent bg-white/3 px-3 py-2.5 text-left transition-colors hover:bg-white/6 aria-pressed:border-border-strong aria-pressed:border-l-brand aria-pressed:bg-brand/8 aria-pressed:shadow-glow"
        aria-pressed={selected}
        onClick={onSelect}
      >
        {children}
      </button>
    </li>
  )
}

function Bar({ value, className }) {
  return (
    <span className={cn('block h-1.5 overflow-hidden rounded-full bg-white/7', className)} aria-hidden="true">
      <span className="block h-full bg-(image:--gradient)" style={{ width: `${value * 10}%` }} />
    </span>
  )
}

function Ratings({ label, items }) {
  return (
    <ul className="m-0 flex list-none flex-col gap-2.5 p-0" aria-label={label}>
      {items.map(([name, value]) => (
        // Name and score share a row on narrow screens, with the bar spanning the row below.
        <li
          key={name}
          className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1 min-[860px]:grid-cols-[minmax(90px,1fr)_2fr_auto]"
        >
          <span className="wrap-anywhere">{name}</span>
          <Bar value={value} className="max-[860px]:col-span-full max-[860px]:row-start-2" />
          <span className="text-muted-foreground tabular-nums">{value}/10</span>
        </li>
      ))}
    </ul>
  )
}

// The criteria this persona pushes hardest, strongest first: what the scenario is really testing.
function Targets({ attributes }) {
  const hard = attributes.filter((a) => a.rating >= HARD).sort((a, b) => b.rating - a.rating)
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-muted-foreground">Tests</span>
      {hard.length ? (
        <ul className="m-0 flex list-none flex-wrap gap-1.5 p-0" aria-label="Criteria this persona tests">
          {hard.map((a) => (
            <li key={a.criterion}>
              <Badge variant="outline" className="border-brand-2/45 whitespace-normal text-brand-2">
                {a.criterion}
              </Badge>
            </li>
          ))}
        </ul>
      ) : (
        <span>This persona keeps every criterion easy.</span>
      )}
    </div>
  )
}

function Transcript({ scenario }) {
  const ref = useRef(null)
  useEffect(() => {
    reveal(ref.current)
    ref.current.focus({ preventScroll: true })
  }, [scenario.id])
  return (
    <Card
      ref={ref}
      tabIndex={-1}
      role="region"
      className="gap-4 p-6"
      aria-label={`Scenario ${scenario.scenario_number}`}
    >
      <h2 className={TITLE}>Scenario {scenario.scenario_number}</h2>
      <Targets attributes={scenario.persona.attributes} />
      <div className="mb-3 flex flex-col gap-3.5">
        <article className={cn(BUBBLE, 'self-end rounded-br-sm border-brand-2/35 bg-brand-2/12')} aria-label="Persona">
          <span className="block text-right text-xs font-semibold text-muted-foreground">Persona</span>
          <p className={SAID}>{scenario.input}</p>
        </article>
        <article className={cn(BUBBLE, 'self-start rounded-bl-sm border-border-strong bg-brand/8')} aria-label="Agent">
          <span className="block text-xs font-semibold text-brand">Agent</span>
          <p className={SAID}>{scenario.output}</p>
        </article>
      </div>

      <div className="grid items-center gap-7 border-t py-5 min-[860px]:grid-cols-[auto_1fr]">
        <p
          className="score-ring justify-self-center min-[860px]:justify-self-auto"
          style={{ '--pct': scenario.score * 10 }}
          role="img"
          aria-label={`Overall score ${fmt(scenario.score)} out of 10`}
        >
          <strong>{fmt(scenario.score)}</strong>
          <span>/10</span>
        </p>
        <Ratings label="Criterion scores" items={scenario.scores.map((s) => [s.criterion, s.score])} />
      </div>

      <details className={DETAILS}>
        <summary className="cursor-pointer font-semibold">Persona profile</summary>
        <Ratings label="Persona ratings" items={scenario.persona.attributes.map((a) => [a.criterion, a.rating])} />
        <pre className={cn(PRE, 'mt-3')}>{scenario.persona.prompt}</pre>
      </details>
      <details className={DETAILS}>
        <summary className="cursor-pointer font-semibold">Agent prompt</summary>
        <pre className={PRE}>{scenario.prompt}</pre>
      </details>
    </Card>
  )
}

export default function InspectPage() {
  // The agent lives in the URL (?agent=<id>), so the Agents page can link straight to its trainings.
  const [params, setParams] = useSearchParams()
  const agentId = params.get('agent') == null ? null : Number(params.get('agent'))
  const [trainingId, setTrainingId] = useState(null)
  const [round, setRound] = useState(null)
  const [scenarioId, setScenarioId] = useState(null)
  // A different agent (e.g. the Inspect link drops ?agent=) clears the selection under it.
  const [shownAgent, setShownAgent] = useState(agentId)
  if (agentId !== shownAgent) {
    setShownAgent(agentId)
    setTrainingId(null)
    setRound(null)
    setScenarioId(null)
  }
  const [version, setVersion] = useState(0)
  const [actionError, setActionError] = useState('')
  const [busy, setBusy] = useState(false)
  const navigate = useNavigate()
  const trainingsRef = useRef(null)
  const roundsRef = useRef(null)
  const scenariosRef = useRef(null)

  const agents = useLoad(api.listAgents, 'all', version)
  const trainings = useLoad(() => api.listTrainings(agentId), agentId, version)
  const rounds = useLoad(() => api.listRounds(trainingId), trainingId, version)
  const detail = useLoad(() => api.getTraining(trainingId), trainingId, version)
  const scenarios = useLoad(
    () => api.listScenarios(trainingId, round),
    round == null ? null : `${trainingId}/${round}`,
    version,
  )
  const scenario = scenarios.data?.find((s) => s.id === scenarioId)
  const selected = trainings.data?.find((t) => t.id === trainingId)
  const status = selected?.status
  const kind = selected?.battle ? 'battle' : 'training'

  // While any training is in progress, reload everything 2 s after each trainings list arrives.
  const active = trainings.data?.some((t) => ACTIVE.includes(t.status))
  useEffect(() => {
    if (!active) return
    const timer = setTimeout(() => setVersion((v) => v + 1), 2000)
    return () => clearTimeout(timer)
  }, [trainings, active])

  useEffect(() => reveal(trainingsRef.current), [agentId])
  useEffect(() => reveal(roundsRef.current), [trainingId])
  useEffect(() => reveal(scenariosRef.current), [round])

  const save = (fetchBlob, filename) => {
    setActionError('')
    fetchBlob(trainingId).then(
      (blob) => saveBlob(blob, filename),
      (e) => setActionError(e.message),
    )
  }
  const pickTraining = (id) => {
    setTrainingId(id)
    setRound(null)
    setScenarioId(null)
    setActionError('')
  }
  const act = async (fn) => {
    setActionError('')
    setBusy(true)
    try {
      await fn()
    } catch (e) {
      setActionError(e.message)
    } finally {
      setBusy(false)
    }
  }
  const pause = () =>
    act(async () => {
      await api.pauseTraining(trainingId)
      setVersion((v) => v + 1)
    })
  // The Train (or Battle) page opens the training in progress, so the resumed one is shown there.
  const resume = () =>
    act(async () => {
      const t = await api.resumeTraining(trainingId)
      navigate(t.battle ? '/battle' : '/train')
    })
  // Runs from the confirmation dialog's Delete button.
  const remove = () =>
    act(async () => {
      await api.deleteTraining(trainingId)
      pickTraining(null)
      setVersion((v) => v + 1)
    })
  const pickAgent = (id) => {
    setParams({ agent: id }, { replace: true })
    pickTraining(null)
  }
  const pickRound = (r) => {
    setRound(r)
    setScenarioId(null)
  }

  return (
    <div className="flex flex-col gap-3.5" data-testid="inspect-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Inspect</h1>
      <div className="grid items-start gap-4 min-[860px]:grid-cols-2 xl:grid-cols-4">
        <Column
          title="Agents"
          state={agents}
          loadingText="Loading agents…"
          emptyText={
            <>
              No agents yet — make one on the <Link to="/agents">Agents page</Link>
            </>
          }
        >
          {agents.data?.map((a) => (
            <Option key={a.id} selected={a.id === agentId} onSelect={() => pickAgent(a.id)}>
              <span className={cn(OPTION_MAIN, 'flex items-center gap-2')}>
                <AgentImage svg={a.image} className="size-6 rounded-md" />
                {a.name}
              </span>
              <span className={OPTION_META}>
                {a.base_model === 'claude' ? 'Claude' : a.base_model} · {a.trainings}{' '}
                {a.trainings === 1 ? 'training' : 'trainings'}
              </span>
              <Overall value={a.overall} className="col-start-2 row-span-2 row-start-1" />
            </Option>
          ))}
        </Column>

        {agentId != null && (
          <div className="grid gap-4">
            {/* One list, split: Trainings shows its loading and errors, Battles just its share once it's loaded. */}
            {[
              ['Trainings', (t) => !t.battle, trainingsRef, trainings, <>No trainings yet — start one on the <Link to="/train">Train page</Link></>],
              ['Battles', (t) => t.battle, null, {}, <>No battles yet — start one on the <Link to="/battle">Battle page</Link></>],
            ].map(([title, holds, ref, state, emptyText]) => (
              <Column
                key={title}
                ref={ref}
                title={title}
                state={{ ...state, data: trainings.data?.filter(holds) }}
                loadingText={`Loading ${title.toLowerCase()}…`}
                emptyText={emptyText}
              >
                {trainings.data?.filter(holds).map((t) => (
                  <Option key={t.id} selected={t.id === trainingId} onSelect={() => pickTraining(t.id)}>
                    <span className={OPTION_MAIN}>
                      #{t.id} · {t.battle ? `vs ${t.battle.opponent?.name ?? 'challenger'}` : t.base_model}
                    </span>
                    <Badge variant="outline" className={`badge-${t.status} justify-self-start tracking-wider uppercase`}>
                      {t.status}
                    </Badge>
                    <span className={OPTION_META}>
                      {new Date(t.created_at).toLocaleDateString()}
                      {t.helper && ` · ${t.helper === 'claude' ? 'Claude' : 'Local model'}`}
                    </span>
                    {t.battle?.grades ? (
                      <span className={OPTION_SCORE} title="Your agent's grade – its opponent's">
                        {fmt(t.battle.grades[0])}–{fmt(t.battle.grades[1])}
                      </span>
                    ) : (
                      <span className={OPTION_SCORE} title="Best score">
                        {fmt(t.best_score)}
                      </span>
                    )}
                  </Option>
                ))}
              </Column>
            ))}
          </div>
        )}

        {trainingId != null && (
          <Column ref={roundsRef} title="Rounds" state={rounds} loadingText="Loading rounds…" emptyText="No rounds finished yet.">
            {rounds.data?.map((r) => (
              <Option key={r.round} selected={r.round === round} onSelect={() => pickRound(r.round)}>
                <span className={OPTION_MAIN}>Round {r.round}</span>
                <span className={OPTION_META}>
                  {r.scenarios} scenarios · practice {fmt(r.mean_score)}
                </span>
                <span className={OPTION_SCORE} title="Score on the eval users, the same every round">
                  {fmt(r.eval_score)}
                </span>
              </Option>
            ))}
          </Column>
        )}

        {round != null && (
          <Column ref={scenariosRef} title="Scenarios" state={scenarios} loadingText="Loading scenarios…" emptyText="No scenarios in this round.">
            {scenarios.data?.map((s) => (
              <Option key={s.id} selected={s.id === scenarioId} onSelect={() => setScenarioId(s.id)}>
                <span className={OPTION_MAIN}>Scenario {s.scenario_number}</span>
                <Bar value={s.score} />
                <span className={OPTION_SCORE}>{fmt(s.score)}</span>
              </Option>
            ))}
          </Column>
        )}
      </div>

      {trainingId != null && <Prompts state={detail} />}
      {selected?.battle && detail.data?.battle?.grades && (
        <Card role="region" aria-label="Head to head" className="gap-3 p-4">
          <h2 className={TITLE}>Head to head</h2>
          <BattleResult battle={detail.data.battle} rounds={detail.data.rounds} />
        </Card>
      )}

      {status === 'failed' && detail.data && (
        <ErrorAlert>{detail.data.error || `The ${kind} stopped without an error message.`}</ErrorAlert>
      )}
      {selected && (
        <div className="flex flex-wrap items-center gap-3">
          {(status === 'pending' || status === 'running') && (
            <Button variant="outline" disabled={busy} onClick={pause}>
              <Pause aria-hidden="true" /> Pause {kind}
            </Button>
          )}
          {status === 'paused' && (
            <Button disabled={busy} onClick={resume}>
              <Play aria-hidden="true" /> Resume {kind}
            </Button>
          )}
          {status === 'failed' && (
            <Button disabled={busy} onClick={resume}>
              <RotateCcw aria-hidden="true" /> Retry {kind}
            </Button>
          )}
          {status === 'completed' && (
            <>
              <Button onClick={() => save(api.downloadPrompt, `${kind}-${trainingId}-prompt.txt`)}>
                <Download aria-hidden="true" /> Download prompt
              </Button>
              {selected.base_model !== 'claude' && (
                <Button onClick={() => save(api.downloadWeights, `${kind}-${trainingId}-lora.zip`)}>
                  <Download aria-hidden="true" /> Download weights
                </Button>
              )}
            </>
          )}
          {!ACTIVE.includes(status) && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="outline" className="border-destructive/50 text-destructive" disabled={busy}>
                  <Trash2 aria-hidden="true" /> Delete {kind}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    Delete {kind} #{trainingId}?
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    This permanently deletes all its rounds and scenarios, its prompts and its trained weights.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction variant="destructive" onClick={remove}>
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
        </div>
      )}
      {actionError && <ErrorAlert>{actionError}</ErrorAlert>}

      {scenario && <Transcript scenario={scenario} />}
    </div>
  )
}
