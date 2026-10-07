import { Download, Pause, Play, Plus, RotateCcw, Sparkles, Swords, Trash2 } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Progress } from '@/components/ui/progress'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { BattleResult, PastBattles } from '@/components/battles'
import { AgentImage, Choice, ErrorAlert, Overall, Spinner } from '@/components/shared'
import { FindAgent } from '@/components/social'
import { cn, useAction } from '@/lib/utils'
import { api, saveBlob } from '../api'
import { CLAUDE_NOTE } from './AgentsPage'
import './TrainPage.css'

const STEPS = [
  ['Agent', 'Pick the agent to train, and whether Claude or your local model runs the helper agents.'],
  ['Rubric', 'List the criteria every simulated conversation is scored against, 1 to 10.'],
  ['Scenarios', 'Choose how many simulated personas each round tests, and how many rounds to train.'],
  ['Training', 'Each round tests the agent against new personas, scores it and trains a better version. Pause stops at once, dropping the step in progress.'],
  ['Complete', 'Training finished. Take the best prompt and LoRA weights with you.'],
]
// A battle sets up like a training; its rounds train a challenger (or it challenges a published agent), which then
// meets the agent head to head.
const BATTLE_STEPS = [
  ['Agent', 'Pick your agent, the champion, its opponent, and whether Claude or your local model runs the helper agents.'],
  STEPS[1],
  ['Scenarios', 'Choose how many personas each round tests, how many rounds train the challenger, and how many rounds the battle has. Each battle round meets that many new users.'],
  ['Battle', 'The rounds train a challenger from your agent. Then both answer the same new users and a judge picks each bout. Your agent only changes if the challenger wins.'],
  ['Result', 'The winner keeps the agent. Take its prompt and LoRA weights with you.'],
]
const WINDOWS = [
  ['n', 'Total runs'],
  ['n/2', 'Last 50%'],
  ['n/4', 'Last 25%'],
  ['last', 'Last run'],
]
const inRange = (n, max) => Number.isInteger(n) && n >= 1 && n <= max
const score = (s) => (s == null ? '—' : s.toFixed(1))
const ACTIVE = ['pending', 'running', 'pausing'] // a user has at most one training in these

// The setup and the training on screen, kept for this tab so leaving the page or reloading loses neither.
// Signing in or out clears it (clearToken). Storage can be unavailable (e.g. blocked); the page then just forgets.
const draftKey = (battle) => (battle ? 'trainform_battle_draft' : 'trainform_train_draft')
const loadDraft = (key) => {
  try {
    return JSON.parse(sessionStorage.getItem(key)) ?? {}
  } catch {
    return {}
  }
}
const storeDraft = (key, draft) => {
  try {
    if (draft) sessionStorage.setItem(key, JSON.stringify(draft))
    else sessionStorage.removeItem(key)
  } catch {
    // Unavailable or full: nothing to keep.
  }
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
        <p className="m-0 text-sm text-muted-foreground">
          Paused. The step in progress was dropped; resuming redoes it from the saved progress.
        </p>
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
  const { id, battle } = training
  const save = (fetchBlob, filename) => run(async () => saveBlob(await fetchBlob(id), filename))
  return (
    <div className="grid gap-4">
      {battle?.grades ? ( // battles from before grading (judged bouts) show as trainings
        <BattleResult battle={battle} rounds={training.rounds} />
      ) : (
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
      )}
      <h3 className="m-0 text-base">{battle?.grades ? 'Winning prompt' : 'Best prompt'}</h3>
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
        {training.base_model !== 'claude' && (
          <Button type="button" disabled={busy} onClick={() => save(api.downloadWeights, `training-${id}-lora.zip`)}>
            <Download />
            Download weights
          </Button>
        )}
        <Button type="button" variant="ghost" onClick={onReset}>
          {battle ? 'Start a new battle' : 'Start a new training'}
        </Button>
      </div>
    </div>
  )
}

// A battle's opponent: a challenger trained from your agent, another of your agents, or a published agent picked
// from your feed or a Trainformer's profile. Only one on your agent's base model fits: the winner's weights go to
// your agent.
function OpponentStep({ agent, agents, value, onChange }) {
  const [mode, setMode] = useState(value ? (value.agent_id ? 'mine' : 'challenge') : 'train')
  const pick = (m) => {
    setMode(m)
    onChange(null)
  }
  const others = agents?.filter((a) => a.id !== agent?.id) ?? []
  return (
    <div className="grid gap-3">
      <span id="battle-opponent" className="text-sm font-medium">
        Opponent
      </span>
      <Choice
        aria-labelledby="battle-opponent"
        value={mode}
        onValueChange={pick}
        options={[
          ['train', 'Train a challenger'],
          ['mine', 'Battle my own agent'],
          ['challenge', 'Challenge an agent'],
        ]}
      />
      {mode === 'mine' &&
        (others.length ? (
          <select
            aria-label="Your opponent"
            className="h-9 rounded-md border border-input bg-input/30 px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [&_option]:bg-background"
            value={value?.agent_id ?? ''}
            onChange={(e) => {
              const a = others.find((o) => o.id === Number(e.target.value))
              onChange(a && { agent_id: a.id, name: a.name, base_model: a.base_model })
            }}
          >
            <option value="">Pick one of your agents</option>
            {others.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name} ({a.base_model === 'claude' ? 'Claude' : a.base_model})
              </option>
            ))}
          </select>
        ) : (
          <p className="m-0 text-sm text-muted-foreground">You need a second agent to battle your own.</p>
        ))}
      {mode === 'mine' && value && agent && value.base_model !== agent.base_model && (
        <p role="status" className="m-0 text-sm text-warning">
          Only an agent on {agent.base_model} can battle yours: the winner's weights must fit it.
        </p>
      )}
      {mode === 'challenge' &&
        (value ? (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2.5 text-sm">
            <AgentImage svg={value.image} className="size-8" />
            <span className="mr-auto grid min-w-0 gap-0.5">
              <span className="font-semibold wrap-anywhere">
                {value.name} by {value.author}
              </span>
              <span className="text-xs text-muted-foreground wrap-anywhere">
                {value.base_model === 'claude' ? 'Claude' : value.base_model}
              </span>
            </span>
            <Button type="button" variant="ghost" size="sm" onClick={() => onChange(null)}>
              Change
            </Button>
            {agent && value.base_model !== agent.base_model && (
              <p role="status" className="m-0 w-full text-sm text-warning">
                Only an agent on {agent.base_model} can battle yours: the winner's weights must fit it.
              </p>
            )}
          </div>
        ) : (
          <FindAgent onPick={onChange} pickLabel="Challenge" />
        ))}
    </div>
  )
}

// "Use my own" picks one of your agents; "Find an agent" copies a published one into them and picks it.
function AgentStep({ agents, error, value, onChange, onCopied }) {
  const [source, setSource] = useState('own')
  const agent = agents?.find((a) => a.id === value)
  const link = (
    <Link to="/agents" className="text-primary underline-offset-4 hover:underline">
      Agents page
    </Link>
  )
  return (
    <div className="grid gap-3">
      <span id="train-agents" className="text-sm font-medium">
        Agent
      </span>
      <Choice
        aria-label="Agent source"
        value={source}
        onValueChange={setSource}
        options={[
          ['own', 'Use my own'],
          ['find', 'Find an agent'],
        ]}
      />
      {source === 'find' ? (
        <FindAgent
          onCopied={(a) => {
            setSource('own')
            onCopied(a)
          }}
        />
      ) : (
        <>
          {!agents && !error && <Spinner />}
          <ErrorAlert>{error}</ErrorAlert>
          {agents?.length === 0 && <p className="m-0 text-sm text-muted-foreground">No agents yet. Make one on the {link}.</p>}
          <RadioGroup
            aria-labelledby="train-agents"
            value={value == null ? '' : String(value)}
            onValueChange={(v) => onChange(Number(v))}
            className="grid-cols-[repeat(auto-fill,minmax(min(260px,100%),1fr))] gap-2"
          >
            {agents?.map((a) => (
              <label
                key={a.id}
                className="flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 text-sm transition-colors hover:bg-accent/50 has-data-[state=checked]:border-border-strong has-data-[state=checked]:bg-primary/10"
              >
                <RadioGroupItem value={String(a.id)} />
                <AgentImage svg={a.image} className="size-8" />
                <span className="mr-auto grid min-w-0 gap-0.5">
                  <span className="font-semibold wrap-anywhere">{a.name}</span>
                  <span className="text-xs text-muted-foreground wrap-anywhere">
                    {a.base_model === 'claude' ? 'Claude' : a.base_model}
                  </span>
                </span>
                <Overall value={a.overall} />
              </label>
            ))}
          </RadioGroup>
          {agent?.model_available === false && (
            <p role="status" className="m-0 text-sm text-warning">
              {agent.base_model} isn't downloaded yet. Download it from the Hub (Agents page, New agent) before training.
            </p>
          )}
          {agents?.length > 0 && <p className="m-0 text-sm text-muted-foreground">Make or edit agents on the {link}.</p>}
        </>
      )}
    </div>
  )
}

function Dashboard({ battle, onReset }) {
  const [draft] = useState(() => loadDraft(draftKey(battle)))
  const steps = battle ? BATTLE_STEPS : STEPS
  const [step, setStep] = useState(draft.step ?? 0)
  const [agentId, setAgentId] = useState(draft.agentId ?? null)
  const [agents, setAgents] = useState(null)
  const [agentsError, setAgentsError] = useState('')
  const [helper, setHelper] = useState(draft.helper ?? null) // 'claude' | 'local'; null: the server picks
  const [claudeAvailable, setClaudeAvailable] = useState(null)
  const [rubric, setRubric] = useState(draft.rubric ?? [])
  const [scenarios, setScenarios] = useState(draft.scenarios ?? '5')
  const [rounds, setRounds] = useState(draft.rounds ?? '3')
  const [battleRounds, setBattleRounds] = useState(draft.battleRounds ?? '1')
  const [opponent, setOpponent] = useState(draft.opponent ?? null) // a battle's rival agent; null: train one
  // A running one is refreshed by the effects below; a finished one no longer changes.
  const [training, setTraining] = useState(draft.training ?? null)
  const [pollError, setPollError] = useState('')
  const [runStart, starting, startError] = useAction()
  const [runControl, controlling, controlError] = useAction()
  const slides = useRef([])
  const shown = useRef(step)

  const agent = agents?.find((a) => a.id === agentId)
  const claudeAgent = agent?.base_model === 'claude'
  const criteria = rubric.map((c) => c.trim()).filter(Boolean)
  const valid = [
    // A Claude agent needs Claude as the helper; a copied agent may need its model downloaded first.
    Boolean(agent) && !(claudeAgent && helper === 'local') && agent.model_available !== false &&
      (!opponent || (opponent.base_model === agent.base_model && opponent.agent_id !== agent.id)),
    criteria.length > 0,
    inRange(Number(scenarios), 100) && (Boolean(opponent) || inRange(Number(rounds), 50)),
  ]
  const completed = training?.status === 'completed'
  const reachable = (i) => (training ? i === 3 || (i === 4 && completed) : i < 3 && valid.slice(0, i).every(Boolean))
  // Scenarios has valid defaults, so it only counts as done once training starts.
  const done = (i) => (training ? i < 3 || (i === 3 && completed) : i < 2 && i !== step && valid[i] && reachable(i))
  const state = (i) => (i === step ? 'current' : done(i) ? 'done' : reachable(i) ? 'todo' : 'locked')

  useEffect(
    () => storeDraft(draftKey(battle), { step, agentId, helper, rubric, scenarios, rounds, battleRounds, opponent, training }),
    [battle, step, agentId, helper, rubric, scenarios, rounds, battleRounds, opponent, training],
  )

  useEffect(() => {
    api.listAgents().then(setAgents, (e) => setAgentsError(e.message))
  }, [])

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
        const id = list.find((t) => ACTIVE.includes(t.status) && Boolean(t.battle) === battle)?.id ?? draft.training?.id
        if (id == null) return
        const t = await api.getTraining(id)
        if (!live) return
        setTraining(t)
        // A finished training opens on Complete, unless it was already finished and shown on Training.
        setStep((s) => (t.status !== 'completed' ? 3 : draft.training?.status === 'completed' ? s : 4))
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
        if (t.status === 'completed') setStep(4)
      } catch (e) {
        if (live) setPollError(e.message)
      }
      if (live) timer = setTimeout(poll, delay)
    }
    const delay = status === 'pausing' ? 300 : 2000 // a pause lands within a second: show it as soon as it does
    timer = setTimeout(poll, delay)
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
        agent_id: agentId,
        rubric: criteria,
        scenarios: Number(scenarios),
        rounds: opponent ? 1 : Number(rounds), // a challenge has no rounds; the hidden field may hold anything
        helper,
        battle,
        opponent_post_id: opponent?.id ?? null,
        opponent_agent_id: opponent?.agent_id ?? null,
        ...(battle && { battle_rounds: Number(battleRounds) }),
      }
      // 409: one is already in progress (e.g. started in another tab), so open it as the resume effect does.
      const t = await api.startTraining(config).catch(async (e) => {
        if (e.status !== 409) throw e
        const active = (await api.listTrainings()).find((t) => ACTIVE.includes(t.status))
        // Finished in the meantime, or the other kind (a battle on Train, a training on Battle): say it's busy.
        if (!active || Boolean(active.battle) !== battle) throw e
        return api.getTraining(active.id)
      })
      setTraining(t)
      setStep(3)
    })

  const content = [
    <div key="agent" className="grid gap-6">
      <AgentStep
        agents={agents}
        error={agentsError}
        value={agentId}
        onChange={setAgentId}
        onCopied={(a) => {
          api.listAgents().then(setAgents, (e) => setAgentsError(e.message))
          setAgentId(a.id)
        }}
      />
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
      {claudeAgent && (
        <p className="m-0 text-sm text-muted-foreground">
          {CLAUDE_NOTE} Claude agents need Claude as the helper.
        </p>
      )}
      {battle && <OpponentStep agent={agent} agents={agents} value={opponent} onChange={setOpponent} />}
    </div>,
    <RubricStep
      key="rubric"
      model={claudeAgent ? null : agent?.base_model}
      helper={helper}
      prompt={agent?.prompt ?? ''}
      criteria={rubric}
      onChange={setRubric}
    />,
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
        <div className="grid gap-2" hidden={Boolean(opponent)}>
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
        {battle && (
          <div className="grid gap-2">
            <Label htmlFor="battle-rounds">Battle rounds</Label>
            <select
              id="battle-rounds"
              className="h-9 rounded-md border border-input bg-input/30 px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [&_option]:bg-background"
              value={battleRounds}
              onChange={(e) => setBattleRounds(e.target.value)}
            >
              {['1', '3', '5', '7'].map((n) => (
                <option key={n} value={n}>
                  {n === '1' ? '1 round' : `Best of ${n}`}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
      {!valid[2] && <p className="m-0 text-sm text-muted-foreground">Scenarios must be 1 to 100 and rounds 1 to 50.</p>}
      {opponent && <p className="m-0 text-sm text-muted-foreground">A challenge trains nothing: each battle round, both agents answer this many new users.</p>}
      <dl className="m-0 grid grid-cols-3 gap-2 rounded-lg border p-4">
        {[
          ['Agent', agent?.name ?? '—'],
          ['Criteria', criteria.length],
          opponent
            ? ['Opponent', opponent.name]
            : ['Total runs', valid[2] ? Number(scenarios) * Number(rounds) : '—', 'total-runs'],
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
      onPause={() => {
        setTraining((t) => ({ ...t, status: 'pausing' }))
        runControl(async () => setTraining(await api.pauseTraining(id)))
      }}
      onResume={() => runControl(async () => setTraining(await api.resumeTraining(id)))}
      busy={controlling}
      error={controlError}
    />,
    <CompleteStep key="complete" training={training} onReset={onReset} />,
  ]

  return (
    <div className="train-page" data-testid="train-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">{battle ? 'Battle your agent' : 'Train an agent'}</h1>

      <ol className="stepper" aria-label={battle ? 'Battle workflow' : 'Training workflow'} style={{ '--progress': step / (steps.length - 1) }}>
        {steps.map(([name], i) => (
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
          {steps.map(([name, blurb], i) => (
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

      {step < 3 && (
        <div className="flex justify-between">
          <Button type="button" variant="ghost" disabled={step === 0} onClick={() => setStep(step - 1)}>
            Back
          </Button>
          {step < 2 ? (
            <Button type="button" disabled={!valid[step]} onClick={() => setStep(step + 1)}>
              Next
            </Button>
          ) : (
            <Button type="button" disabled={!valid.every(Boolean) || starting} onClick={start}>
              {starting ? <Spinner /> : battle ? <Swords /> : <Play />}
              {battle ? 'Start battle' : 'Start training'}
            </Button>
          )}
        </div>
      )}

      {/* On the first and result slides only: opening a past battle replaces the one on screen. */}
      {battle && (step === 0 || step === 4) && (
        <PastBattles
          status={status}
          onOpen={(t) => {
            setTraining(t)
            setStep(4)
          }}
        />
      )}
    </div>
  )
}

export default function TrainPage({ battle = false }) {
  const [run, setRun] = useState(0)
  const reset = () => {
    storeDraft(draftKey(battle), null)
    setRun((r) => r + 1)
  }
  return <Dashboard key={run} battle={battle} onReset={reset} />
}
