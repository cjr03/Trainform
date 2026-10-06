import { ArrowBigDown, ArrowBigUp, Copy, Plus, Search, Send, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Dialog } from 'radix-ui'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Textarea } from '@/components/ui/textarea'
import { AgentImage, Avatar, ErrorAlert, Overall, Spinner } from '@/components/shared'
import { cn, profilePath, replacePost, useAction } from '@/lib/utils'
import { api } from '../api'

export const LINK = 'text-primary underline-offset-4 hover:underline'

// A published agent: its author, image, overall, caption and prompt, votes, and a copy (or, if `mine`, delete) button.
export function PostCard({ post, mine, copyLabel = 'Copy to my agents', onChange, onDelete, onCopied }) {
  const [run, busy, error] = useAction()
  const [copied, setCopied] = useState(null)
  const vote = (v) => run(async () => onChange({ ...post, ...(await api.vote(post.id, post.my_vote === v ? 0 : v)) }))
  const copy = () =>
    run(async () => {
      const agent = await api.copyPost(post.id)
      setCopied(agent)
      onCopied?.(agent)
    })
  const remove = () =>
    window.confirm('Delete this post?') &&
    run(async () => {
      await api.deletePost(post.id)
      onDelete(post)
    })
  const voteButton = (value, Icon, label) => (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      className="rounded-full aria-pressed:text-brand"
      aria-label={label}
      aria-pressed={post.my_vote === value}
      disabled={busy || mine}
      onClick={() => vote(value)}
    >
      <Icon className={cn('size-5', post.my_vote === value && 'fill-current')} />
    </Button>
  )

  return (
    <article aria-labelledby={`post-${post.id}`} className="grid min-w-0 gap-3 rounded-xl border bg-card p-4">
      <div className="flex items-center gap-2 text-sm">
        <Link to={profilePath(post.author)} className={cn(LINK, 'flex min-w-0 items-center gap-2 font-medium')}>
          <Avatar url={post.author_avatar_url} />
          <span className="wrap-anywhere">{post.author}</span>
        </Link>
        <time dateTime={post.created_at} className="ml-auto shrink-0 text-muted-foreground">
          {new Date(post.created_at).toLocaleDateString()}
        </time>
      </div>
      <div className="flex items-center gap-3">
        <AgentImage svg={post.image} className="size-16" />
        <div className="mr-auto grid min-w-0 gap-0.5">
          <h3 id={`post-${post.id}`} className="m-0 font-sans text-base tracking-normal wrap-anywhere">
            {post.name}
          </h3>
          <span className="text-xs text-muted-foreground wrap-anywhere">
            {post.base_model === 'claude' ? 'Claude' : post.base_model}
          </span>
        </div>
        <Overall value={post.overall} />
      </div>
      {post.caption && <p className="m-0 text-sm whitespace-pre-wrap wrap-anywhere">{post.caption}</p>}
      <details className="text-sm [&[open]>summary]:mb-2">
        <summary className="cursor-pointer text-muted-foreground">Prompt</summary>
        <pre className="m-0 max-h-60 overflow-auto rounded-lg border bg-background/60 p-3 whitespace-pre-wrap wrap-anywhere">
          {post.prompt}
        </pre>
      </details>
      <ErrorAlert>{error}</ErrorAlert>
      {copied && (
        <p role="status" className="m-0 text-sm text-success">
          Copied to your{' '}
          <Link to="/agents" className={LINK}>
            agents
          </Link>{' '}
          as {copied.name}.{copied.model_available === false && ` Download ${copied.base_model} before training it.`}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Votes" className="flex items-center rounded-full border bg-background/40">
          {voteButton(1, ArrowBigUp, 'Upvote')}
          <span className="min-w-6 text-center font-display text-sm tabular-nums">
            {post.score}
          </span>
          {voteButton(-1, ArrowBigDown, 'Downvote')}
        </div>
        {mine ? (
          <Button type="button" variant="outline" className="ml-auto border-destructive/50 text-destructive" disabled={busy} onClick={remove}>
            <Trash2 aria-hidden="true" />
            Delete post
          </Button>
        ) : (
          <Button type="button" variant="outline" className="ml-auto" disabled={busy} onClick={copy}>
            {busy ? <Spinner /> : <Copy aria-hidden="true" />}
            {copyLabel}
          </Button>
        )}
      </div>
    </article>
  )
}

// Follows or unfollows `user` ({username, followers, following}), handing back the updated user.
export function FollowButton({ user, onChange }) {
  const [run, busy, error] = useAction()
  const toggle = () =>
    run(async () => {
      await (user.following ? api.unfollow : api.follow)(user.username)
      onChange({ ...user, following: !user.following, followers: user.followers + (user.following ? -1 : 1) })
    })
  return (
    <span className="grid justify-items-end gap-1">
      <Button
        type="button"
        size="sm"
        variant={user.following ? 'outline' : 'default'}
        aria-label={`${user.following ? 'Unfollow' : 'Follow'} ${user.username}`}
        disabled={busy}
        onClick={toggle}
      >
        {busy && <Spinner />}
        {user.following ? 'Unfollow' : 'Follow'}
      </Button>
      {error && <span className="text-xs text-destructive">{error}</span>}
    </span>
  )
}

export function PostList({ posts, ...props }) {
  return (
    <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(min(320px,100%),1fr))] gap-3 p-0">
      {posts.map((p) => (
        <li key={p.id} className="min-w-0">
          <PostCard post={p} {...props} />
        </li>
      ))}
    </ul>
  )
}

// "Find an agent": your feed, or a Trainformer found by search; picking a post copies it into your agents.
export function FindAgent({ onCopied }) {
  const [from, setFrom] = useState('') // '': your feed, else that Trainformer's posts
  const [posts, setPosts] = useState(null)
  const [q, setQ] = useState('')
  const [users, setUsers] = useState(null)
  const [run, busy, error] = useAction()
  const show = (username) =>
    run(async () => {
      setFrom(username)
      setPosts(null)
      setPosts(username ? (await api.getProfile(username)).posts : await api.feed())
    })
  useEffect(() => {
    show('')
    // Loads the feed once, on opening.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const search = (e) => {
    e.preventDefault()
    run(async () => setUsers(await api.searchUsers(q.trim())))
  }

  return (
    <div className="grid gap-4">
      <form role="search" className="grid gap-2 sm:grid-cols-[1fr_auto]" onSubmit={search}>
        <Input
          type="search"
          aria-label="Search Trainformers"
          placeholder="Search Trainformers by username"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <Button variant="outline" disabled={!q.trim() || busy}>
          <Search aria-hidden="true" />
          Search
        </Button>
      </form>
      {users?.length === 0 && <p className="m-0 text-sm text-muted-foreground">No Trainformers match that name.</p>}
      {users?.length > 0 && (
        <ul aria-label="Trainformers" className="m-0 flex list-none flex-wrap gap-2 p-0">
          {users.map((u) => (
            <li key={u.username}>
              <Button type="button" variant="outline" size="sm" aria-pressed={from === u.username} onClick={() => show(u.username)}>
                <Avatar url={u.avatar_url} className="size-5" />
                {u.username} · {u.agents}
              </Button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="m-0 font-sans text-base tracking-normal wrap-anywhere">
          {from ? `${from}'s agents` : 'From Trainformers you follow'}
        </h3>
        {from && (
          <Button type="button" variant="ghost" size="sm" onClick={() => show('')}>
            Back to your feed
          </Button>
        )}
      </div>
      <ErrorAlert>{error}</ErrorAlert>
      {!posts && !error && <Spinner />}
      {posts?.length === 0 && (
        <p className="m-0 text-sm text-muted-foreground">
          {from ? 'No published agents yet.' : 'Your feed is empty. Search for a Trainformer to find their agents.'}
        </p>
      )}
      {posts?.length > 0 && (
        <PostList posts={posts} copyLabel="Use this agent" onChange={(p) => setPosts(replacePost(p))} onCopied={onCopied} />
      )}
    </div>
  )
}

// The header's Plus: publish one of your agents (with an optional caption) to your followers' feeds.
export function PublishDialog({ username }) {
  const [open, setOpen] = useState(false)
  const [agents, setAgents] = useState(null)
  const [agentId, setAgentId] = useState('')
  const [caption, setCaption] = useState('')
  const [done, setDone] = useState(false)
  const [run, busy, error] = useAction()
  const toggle = (o) => {
    setOpen(o)
    if (!o) return
    setDone(false)
    setAgentId('')
    setCaption('')
    run(async () => setAgents(await api.listAgents()))
  }
  const publish = (e) => {
    e.preventDefault()
    run(async () => {
      await api.publishPost(Number(agentId), caption.trim() || null)
      setDone(true)
    })
  }

  return (
    <Dialog.Root open={open} onOpenChange={toggle}>
      <Dialog.Trigger asChild>
        <Button variant="ghost" size="icon-sm" aria-label="Publish an agent" title="Publish an agent">
          <Plus aria-hidden="true" />
        </Button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/50" />
        <Dialog.Content className="fixed top-[50%] left-[50%] z-50 grid max-h-[90dvh] w-full max-w-[calc(100%-2rem)] translate-x-[-50%] translate-y-[-50%] gap-4 overflow-y-auto rounded-lg border bg-background p-6 shadow-lg sm:max-w-lg">
          <Dialog.Title className="m-0 text-lg font-semibold">Publish an agent</Dialog.Title>
          <Dialog.Description className="m-0 text-sm text-muted-foreground">
            Share one of your agents on the feed of Trainformers who follow you.
          </Dialog.Description>
          {done ? (
            <p role="status" className="m-0 text-sm text-success">
              Published! See it on{' '}
              <Link to={profilePath(username ?? '')} className={LINK} onClick={() => setOpen(false)}>
                your profile
              </Link>
              .
            </p>
          ) : (
            <form className="grid gap-4" onSubmit={publish}>
              {!agents && !error && <Spinner />}
              {agents?.length === 0 && (
                <p className="m-0 text-sm text-muted-foreground">No agents yet. Make one on the Agents page first.</p>
              )}
              <RadioGroup aria-label="Agent" value={agentId} onValueChange={setAgentId} className="gap-1">
                {agents?.map((a) => (
                  <label
                    key={a.id}
                    className="flex cursor-pointer items-center gap-3 rounded-md border border-transparent px-3 py-2 text-sm transition-colors hover:bg-accent/50 has-data-[state=checked]:border-border-strong has-data-[state=checked]:bg-primary/10"
                  >
                    <RadioGroupItem value={String(a.id)} />
                    <AgentImage svg={a.image} className="size-8" />
                    <span className="mr-auto font-semibold wrap-anywhere">{a.name}</span>
                    <Overall value={a.overall} />
                  </label>
                ))}
              </RadioGroup>
              <div className="grid gap-2">
                <Label htmlFor="publish-caption">Caption (optional)</Label>
                <Textarea id="publish-caption" maxLength={500} value={caption} onChange={(e) => setCaption(e.target.value)} />
              </div>
              <ErrorAlert>{error}</ErrorAlert>
              <div className="flex justify-end gap-2">
                <Dialog.Close asChild>
                  <Button type="button" variant="ghost">
                    Cancel
                  </Button>
                </Dialog.Close>
                <Button disabled={!agentId || busy}>
                  {busy ? <Spinner /> : <Send aria-hidden="true" />}
                  Publish
                </Button>
              </div>
            </form>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
