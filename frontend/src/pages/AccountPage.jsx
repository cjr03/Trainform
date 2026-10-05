import { CircleAlert, Laptop, Loader2, Sparkles } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { cn } from '@/lib/utils'
import { api } from '../api'

const spinner = <Loader2 className="animate-spin" aria-hidden="true" />

function ErrorAlert({ children }) {
  return (
    <Alert variant="destructive">
      <CircleAlert aria-hidden="true" />
      <AlertDescription>{children}</AlertDescription>
    </Alert>
  )
}

function Success({ children }) {
  return (
    <p className="m-0 text-sm text-success" role="status">
      {children}
    </p>
  )
}

function Field({ label, ...props }) {
  return (
    <div className="grid gap-2">
      <Label htmlFor={props.id}>{label}</Label>
      <Input {...props} />
    </div>
  )
}

export default function AccountPage() {
  const [user, setUser] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.me().then(setUser, (err) => setError(err.message))
  }, [])

  return (
    <div data-testid="account-page">
      <h1 className="m-0 mb-3.5 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Account</h1>
      {error ? (
        <ErrorAlert>{error}</ErrorAlert>
      ) : !user ? (
        <p className="flex items-center gap-2 text-muted-foreground" role="status">
          {spinner}
          Loading account…
        </p>
      ) : (
        <div className="grid gap-4 min-[880px]:grid-cols-2">
          <Card aria-labelledby="profile-title" role="region">
            <CardHeader>
              <CardTitle>
                <h2 id="profile-title" className="m-0 text-lg">Profile</h2>
              </CardTitle>
              <CardDescription>Passwords are stored hashed, so they can't be displayed.</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-5 gap-y-3">
                <dt className="text-muted-foreground">Username</dt>
                <dd className="m-0 [overflow-wrap:anywhere]">{user.username}</dd>
                <dt className="text-muted-foreground">Email</dt>
                <dd className="m-0 [overflow-wrap:anywhere]">{user.email}</dd>
                <dt className="text-muted-foreground">Password</dt>
                <dd className="m-0 tracking-[0.2em]">••••••••</dd>
              </dl>
            </CardContent>
          </Card>
          <PasswordForm />
          <ClaudeLink user={user} setUser={setUser} />
        </div>
      )}
    </div>
  )
}

const EMPTY = { current: '', next: '', confirm: '' }

function PasswordForm() {
  const [form, setForm] = useState(EMPTY)
  const [error, setError] = useState('')
  const [done, setDone] = useState(false)
  const [busy, setBusy] = useState(false)

  const bind = (name) => ({
    id: `pw-${name}`,
    value: form[name],
    onChange: (e) => {
      const { value } = e.target
      setForm((f) => ({ ...f, [name]: value }))
    },
  })

  async function submit(e) {
    e.preventDefault()
    setDone(false)
    if (form.next !== form.confirm) return setError('New passwords do not match.')
    setBusy(true)
    setError('')
    try {
      await api.changePassword(form.current, form.next, form.confirm)
      setForm(EMPTY)
      setDone(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card aria-labelledby="password-title" role="region">
      <CardHeader>
        <CardTitle>
          <h2 id="password-title" className="m-0 text-lg">Change password</h2>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form className="grid gap-4" onSubmit={submit}>
          <Field label="Current password" {...bind('current')} type="password" autoComplete="current-password" required />
          <Field label="New password" {...bind('next')} type="password" autoComplete="new-password" minLength={8} required />
          <Field
            label="Confirm new password"
            {...bind('confirm')}
            type="password"
            autoComplete="new-password"
            minLength={8}
            required
          />
          {error && <ErrorAlert>{error}</ErrorAlert>}
          {done && <Success>Password changed.</Success>}
          <Button className="justify-self-start" disabled={busy}>
            {busy && spinner}
            Change password
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function ClaudeLink({ user, setUser }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [done, setDone] = useState('')
  const [busy, setBusy] = useState(false)
  const linked = user.claude_linked

  async function run(call, message) {
    setBusy(true)
    setError('')
    setDone('')
    try {
      setUser(await call())
      setKey('')
      setDone(message)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card
      aria-labelledby="claude-title"
      role="region"
      className={cn('transition-[border-color,box-shadow] duration-400 min-[880px]:col-span-2', linked && 'border-success/45 shadow-glow')}
    >
      <CardHeader>
        <CardTitle>
          <h2 id="claude-title" className="m-0 text-lg">Claude linkage</h2>
        </CardTitle>
        <CardDescription>Your Anthropic API key is stored encrypted and never shown again.</CardDescription>
        <CardAction>
          <Badge variant="outline" className={cn(linked && 'border-success/45 text-success')}>
            {linked ? 'Linked' : 'Not linked'}
          </Badge>
        </CardAction>
      </CardHeader>
      <CardContent className="grid gap-6">
        <ul className="m-0 grid list-none gap-3 p-0 text-sm sm:grid-cols-2">
          <li className="flex gap-3">
            <Sparkles className="mt-0.5 size-4 shrink-0 text-brand" aria-hidden="true" />
            <span>
              <strong className="font-medium">With Claude:</strong>{' '}
              <span className="text-muted-foreground">
                stronger prompts, rubrics, personas and grading, so each round improves your agent more.
              </span>
            </span>
          </li>
          <li className="flex gap-3">
            <Laptop className="mt-0.5 size-4 shrink-0 text-brand-2" aria-hidden="true" />
            <span>
              <strong className="font-medium">Without a key:</strong>{' '}
              <span className="text-muted-foreground">
                training still works, with the helpers running locally on the model you train. Results are weaker.
              </span>
            </span>
          </li>
        </ul>
        <form
          className="grid max-w-[520px] gap-4"
          onSubmit={(e) => {
            e.preventDefault()
            run(() => api.setClaudeKey(key.trim()), 'Key saved.')
          }}
        >
          <Field
            label="Anthropic API key"
            id="claude-key"
            type="password"
            autoComplete="off"
            placeholder="sk-ant-…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
          <a className="text-sm" href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer">
            Get a key from the Anthropic Console
          </a>
          {error && <ErrorAlert>{error}</ErrorAlert>}
          {done && <Success>{done}</Success>}
          <div className="flex flex-wrap gap-2">
            <Button disabled={busy || !key.trim()}>{linked ? 'Replace key' : 'Link key'}</Button>
            {linked && (
              <Button type="button" variant="destructive" onClick={() => run(api.removeClaudeKey, 'Key removed.')} disabled={busy}>
                Unlink
              </Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
