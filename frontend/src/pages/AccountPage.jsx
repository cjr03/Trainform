import { useEffect, useState } from 'react'
import { api } from '../api'
import './AccountPage.css'

export default function AccountPage() {
  const [user, setUser] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api.me().then(setUser, (err) => setError(err.message))
  }, [])

  return (
    <div className="account" data-testid="account-page">
      <h1 className="page-title">Account</h1>
      {error ? (
        <p className="error" role="alert">{error}</p>
      ) : !user ? (
        <p className="row muted" role="status">
          <span className="spinner" aria-hidden="true" />
          Loading account…
        </p>
      ) : (
        <div className="account-grid">
          <section className="glass card" aria-labelledby="profile-title">
            <h2 id="profile-title">Profile</h2>
            <dl className="facts">
              <dt>Username</dt>
              <dd>{user.username}</dd>
              <dt>Email</dt>
              <dd>{user.email}</dd>
              <dt>Password</dt>
              <dd className="masked">••••••••</dd>
            </dl>
            <p className="muted account-note">Passwords are stored hashed, so they can't be displayed.</p>
          </section>
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
    <section className="glass card" aria-labelledby="password-title">
      <h2 id="password-title">Change password</h2>
      <form className="stack" onSubmit={submit}>
        <div className="field">
          <label className="label" htmlFor="pw-current">Current password</label>
          <input className="input" {...bind('current')} type="password" autoComplete="current-password" required />
        </div>
        <div className="field">
          <label className="label" htmlFor="pw-next">New password</label>
          <input className="input" {...bind('next')} type="password" autoComplete="new-password" minLength={8} required />
        </div>
        <div className="field">
          <label className="label" htmlFor="pw-confirm">Confirm new password</label>
          <input className="input" {...bind('confirm')} type="password" autoComplete="new-password" minLength={8} required />
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        {done && <p className="success" role="status">Password changed.</p>}
        <button className="btn btn-primary" disabled={busy}>
          {busy && <span className="spinner" aria-hidden="true" />}
          Change password
        </button>
      </form>
    </section>
  )
}

function ClaudeLink({ user, setUser }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const linked = user.claude_linked

  async function run(call) {
    setBusy(true)
    setError('')
    try {
      setUser(await call())
      setKey('')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className={`glass card claude${linked ? ' card-glow' : ''}`} aria-labelledby="claude-title">
      <div className="claude-head">
        <h2 id="claude-title">Claude linkage</h2>
        <span className={`badge${linked ? ' badge-linked' : ''}`}>{linked ? 'Linked' : 'Not linked'}</span>
      </div>
      <p className="muted account-note">
        Trainform's helper agents run on your own Anthropic API key. It is stored encrypted and never shown again.
      </p>
      <form
        className="stack"
        onSubmit={(e) => {
          e.preventDefault()
          run(() => api.setClaudeKey(key))
        }}
      >
        <div className="field">
          <label className="label" htmlFor="claude-key">Anthropic API key</label>
          <input
            id="claude-key"
            className="input"
            type="password"
            autoComplete="off"
            placeholder="sk-ant-…"
            value={key}
            onChange={(e) => setKey(e.target.value)}
          />
        </div>
        {error && <p className="error" role="alert">{error}</p>}
        <div className="row">
          <button className="btn btn-primary" disabled={busy || !key}>
            {linked ? 'Replace key' : 'Link key'}
          </button>
          {linked && (
            <button type="button" className="btn btn-danger" onClick={() => run(api.removeClaudeKey)} disabled={busy}>
              Unlink
            </button>
          )}
        </div>
      </form>
    </section>
  )
}
