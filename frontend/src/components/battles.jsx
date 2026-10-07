import { useEffect, useState } from 'react'
import { ErrorAlert } from '@/components/shared'
import { cn, useAction } from '@/lib/utils'
import { api } from '../api'

const score = (s) => (s == null ? '—' : s.toFixed(1))
const count = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`
const roundsText = (battle, rounds) =>
  `${count(battle.rounds ?? 1, 'battle round')} · ${
    battle.opponent ? 'a challenge, no training' : `challenger trained for ${count(rounds, 'round')}`
  }`

// Both sides' grades on the battle's new users. Each battle round is won by the higher mean on its users, and the side
// that wins more rounds keeps the agent (a tie keeps yours). Battles from before rounds were one round.
export function BattleResult({ battle, rounds }) {
  const rival = battle.opponent ? `${battle.opponent.name} by ${battle.opponent.author}` : 'The challenger'
  const [mine, theirs] = battle.grades
  const roundGrades = battle.round_grades ?? [battle.grades]
  const perRound = battle.bouts.length / roundGrades.length
  const wins = (side) => roundGrades.filter((g) => g[side] > g[1 - side]).length
  return (
    <div className="grid gap-3">
      <div className="flex flex-col items-center gap-1 pt-3 pb-2">
        <span className="text-sm text-muted-foreground wrap-anywhere">Your agent vs {battle.opponent ? rival : 'challenger'}</span>
        <span
          className="font-display text-6xl leading-tight font-bold tabular-nums text-gradient sm:text-7xl"
          data-testid="battle-score"
        >
          {score(mine)} – {score(theirs)}
        </span>
        <span className="text-sm text-muted-foreground">Mean grade out of 10 over {battle.bouts.length} new users</span>
        {roundGrades.length > 1 && (
          <span className="font-semibold tabular-nums" data-testid="rounds-won">
            Rounds won {wins(0)} – {wins(1)}
          </span>
        )}
        <span className="text-sm text-muted-foreground">{roundsText(battle, rounds)}</span>
        <p role="status" className="m-0 pt-2 text-center font-semibold">
          {battle.winner === 'challenger'
            ? `${rival} wins. Your agent now has its prompt and weights.`
            : 'Your agent holds. It keeps its prompt and weights.'}
        </p>
      </div>
      <details className="rounded-lg border p-3">
        <summary className="cursor-pointer text-sm font-medium">Scenarios</summary>
        <ol className="m-0 mt-3 grid gap-3 p-0">
          {battle.bouts.map((b, i) => (
            <li key={i} className="grid gap-1.5 border-t pt-3 text-sm [overflow-wrap:anywhere] first:border-t-0 first:pt-0">
              {roundGrades.length > 1 && i % perRound === 0 && (
                <span className="font-display text-base">
                  Round {i / perRound + 1}: {score(roundGrades[i / perRound][0])} – {score(roundGrades[i / perRound][1])}
                </span>
              )}
              <span className="font-semibold">
                Scenario {i + 1}: {score(b.grades[0])} – {score(b.grades[1])}
              </span>
              <span className="text-muted-foreground">User: {b.message}</span>
              <span>Your agent: {b.champion}</span>
              <span>{battle.opponent?.name ?? 'Challenger'}: {b.challenger}</span>
            </li>
          ))}
        </ol>
      </details>
    </div>
  )
}

// Finished battles (of one agent, or all), newest first, refetched when `status` (the battle on screen's) changes;
// picking one passes its full training to `onOpen`. `empty` shows when there are none.
export function PastBattles({ agentId, status, onOpen, Heading = 'h2', empty = null }) {
  const [battles, setBattles] = useState(null)
  const [error, setError] = useState('')
  const [run, busy, openError] = useAction()
  useEffect(() => {
    api.listTrainings(agentId).then((ts) => setBattles(ts.filter((t) => t.battle?.grades)), (e) => setError(e.message))
  }, [agentId, status])
  if (!battles?.length && !error && !empty) return null
  return (
    <section aria-labelledby="past-battles" className="grid gap-3">
      <Heading id="past-battles" className={cn('m-0', Heading === 'h2' ? 'text-xl' : 'text-base')}>
        Past battles
      </Heading>
      <ErrorAlert>{error || openError}</ErrorAlert>
      {battles && !battles.length && empty}
      {battles?.length > 0 && (
        <ul className="m-0 grid list-none gap-2 p-0">
          {battles.map(({ id, agent_name, created_at, battle, rounds }) => {
            const rival = battle.opponent?.name ?? 'challenger'
            return (
              <li key={id}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => run(async () => onOpen(await api.getTraining(id)))}
                  className="grid w-full grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 rounded-lg border bg-card p-3 text-left hover:bg-accent"
                >
                  <span className="font-medium wrap-anywhere">
                    {agent_name ?? 'Your agent'} vs {rival}
                  </span>
                  <span className="font-semibold tabular-nums">
                    {score(battle.grades[0])} – {score(battle.grades[1])}
                  </span>
                  <span className="col-span-2 text-sm text-muted-foreground">
                    {new Date(created_at).toLocaleDateString()} · {roundsText(battle, rounds)} ·{' '}
                    {battle.winner === 'challenger' ? `${rival} won` : 'Your agent held'}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
