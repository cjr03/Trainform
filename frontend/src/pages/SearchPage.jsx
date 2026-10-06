import { Search } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Avatar, ErrorAlert, Spinner } from '@/components/shared'
import { FollowButton, LINK } from '@/components/social'
import { cn, profilePath, useAction } from '@/lib/utils'
import { api } from '../api'

const plural = (n, word) => `${n} ${n === 1 ? word : `${word}s`}`

export default function SearchPage() {
  const [q, setQ] = useState('')
  const [users, setUsers] = useState(null)
  const [run, busy, error] = useAction()
  const search = (e) => {
    e.preventDefault()
    run(async () => setUsers(await api.searchUsers(q.trim())))
  }

  return (
    <div className="mx-auto grid w-full max-w-2xl gap-5" data-testid="search-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Find Trainformers</h1>
      <form role="search" className="grid grid-cols-[1fr_auto] gap-2" onSubmit={search}>
        <Input
          type="search"
          aria-label="Search Trainformers"
          placeholder="Username"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <Button disabled={!q.trim() || busy}>
          {busy ? <Spinner /> : <Search aria-hidden="true" />}
          Search
        </Button>
      </form>
      <ErrorAlert>{error}</ErrorAlert>
      {users?.length === 0 && <p className="m-0 text-center text-muted-foreground">No Trainformers match that name.</p>}
      {users?.length > 0 && (
        <ul aria-label="Trainformers" className="m-0 grid list-none gap-2 p-0">
          {users.map((u) => (
            <li key={u.username} className="flex items-center gap-3 rounded-xl border bg-card p-3">
              <Avatar url={u.avatar_url} className="size-10" />
              <span className="mr-auto grid min-w-0 gap-0.5">
                <Link to={profilePath(u.username)} className={cn(LINK, 'font-semibold wrap-anywhere')}>
                  {u.username}
                </Link>
                <span className="text-xs text-muted-foreground">
                  {plural(u.agents, 'agent')} · {plural(u.followers, 'follower')}
                </span>
              </span>
              <FollowButton user={u} onChange={(n) => setUsers((list) => list.map((x) => (x.username === n.username ? n : x)))} />
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
