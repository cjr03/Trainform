import { useState } from 'react'
import { api } from '../api'
import './AuthPage.css'

const MODES = [
  ['login', 'Sign in'],
  ['signup', 'Create account'],
]
const STEPS = ['Identity', 'Security', 'Link Claude']

export default function AuthPage({ onAuth }) {
  const [mode, setMode] = useState('login')
  const [phase, setPhase] = useState(0)
  const [dir, setDir] = useState('forward')
  const [form, setForm] = useState({ email: '', username: '', password: '', password_confirm: '', key: '' })
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const bind = (name) => ({
    id: `auth-${name}`,
    value: form[name],
    onChange: (e) => {
      const { value } = e.target
      setForm((f) => ({ ...f, [name]: value }))
    },
  })

  const switchMode = (m) => {
    if (m === mode) return
    setDir(m === 'signup' ? 'forward' : 'back')
    setMode(m)
    setForm((f) => ({ ...f, password: '', password_confirm: '' }))
    setError('')
  }
  const go = (p) => {
    setDir(p > phase ? 'forward' : 'back')
    setPhase(p)
    setError('')
  }

  async function submit(call) {
    setBusy(true)
    setError('')
    try {
      onAuth(await call())
    } catch (err) {
      if (err.status === 409 || (err.status === 422 && /email|pattern/i.test(err.message))) go(0)
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const login = (e) => {
    e.preventDefault()
    submit(() => api.login(form.username, form.password))
  }

  const signup = (key) => {
    const { email, username, password, password_confirm } = form
    submit(() => api.signup({ email, username, password, password_confirm, anthropic_api_key: key?.trim() || null }))
  }

  const next = (e) => {
    e.preventDefault()
    if (phase === 0) {
      if (!e.currentTarget.checkValidity()) return setError('Enter a valid email and a username of 3-32 letters, digits, _ . or -.')
      go(1)
    } else if (phase === 1) {
      if (form.password.length < 8) return setError('Use a password of at least 8 characters.')
      if (form.password !== form.password_confirm) return setError('Passwords do not match.')
      go(2)
    } else signup(form.key)
  }

  const onTabKey = (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    const other = mode === 'login' ? 'signup' : 'login'
    switchMode(other)
    document.getElementById(`tab-${other}`).focus()
  }

  const spinner = busy && <span className="spinner" aria-hidden="true" />

  return (
    <div className="auth" data-testid="auth-page">
      <header className="auth-head">
        <h1 className="brand auth-brand">TRAINFORM</h1>
        <p className="muted">Shape an agent's prompt and weights against simulated users.</p>
      </header>

      <div className="glass card card-glow auth-card">
        <div className="segmented" role="tablist" aria-label="Account access" data-mode={mode} onKeyDown={onTabKey}>
          {MODES.map(([m, label]) => (
            <button
              key={m}
              id={`tab-${m}`}
              type="button"
              role="tab"
              aria-selected={mode === m}
              aria-controls="auth-panel"
              tabIndex={mode === m ? 0 : -1}
              onClick={() => switchMode(m)}
            >
              {label}
            </button>
          ))}
        </div>

        <div id="auth-panel" role="tabpanel" aria-labelledby={`tab-${mode}`}>
          {mode === 'signup' && (
            <ol className="steps" aria-label="Sign-up progress" style={{ '--progress': phase / 2 }}>
              {STEPS.map((step, i) => (
                <li key={step} className={i < phase ? 'done' : undefined} aria-current={i === phase ? 'step' : undefined}>
                  <span className="step-dot">{i + 1}</span>
                  {step}
                </li>
              ))}
            </ol>
          )}

          <div key={mode === 'login' ? 'login' : phase} className="auth-pane" data-dir={dir}>
            {mode === 'login' ? (
              <form className="stack" onSubmit={login}>
                <div className="field">
                  <label className="label" htmlFor="auth-username">Username</label>
                  <input className="input" {...bind('username')} autoComplete="username" required />
                </div>
                <div className="field">
                  <label className="label" htmlFor="auth-password">Password</label>
                  <input className="input" {...bind('password')} type="password" autoComplete="current-password" required />
                </div>
                {error && <p className="error" role="alert">{error}</p>}
                <button className="btn btn-primary" disabled={busy}>
                  {spinner}Sign in
                </button>
              </form>
            ) : (
              <form className="stack" onSubmit={next} noValidate>
                {phase === 0 && (
                  <>
                    <div className="field">
                      <label className="label" htmlFor="auth-email">Email</label>
                      <input className="input" {...bind('email')} type="email" autoComplete="email" required autoFocus={dir === 'back'} />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="auth-username">Username</label>
                      <input
                        className="input"
                        {...bind('username')}
                        autoComplete="username"
                        pattern="[A-Za-z0-9_.\-]{3,32}"
                        maxLength={32}
                        required
                      />
                    </div>
                  </>
                )}
                {phase === 1 && (
                  <>
                    <div className="field">
                      <label className="label" htmlFor="auth-password">Password</label>
                      <input className="input" {...bind('password')} type="password" autoComplete="new-password" maxLength={128} autoFocus />
                    </div>
                    <div className="field">
                      <label className="label" htmlFor="auth-password_confirm">Confirm password</label>
                      <input className="input" {...bind('password_confirm')} type="password" autoComplete="new-password" maxLength={128} />
                    </div>
                    <p className="muted auth-note">At least 8 characters.</p>
                  </>
                )}
                {phase === 2 && (
                  <>
                    <p className="auth-note">
                      Trainform's helper agents write prompts, build rubrics, play personas and score conversations. They
                      run on your own Claude key, stored encrypted, or without one on the model you train, with much weaker
                      results. You can link a key later from Account.
                    </p>
                    <div className="field">
                      <label className="label" htmlFor="auth-key">Anthropic API key</label>
                      <input className="input" {...bind('key')} type="password" autoComplete="off" placeholder="sk-ant-…" autoFocus />
                    </div>
                    <a className="auth-note" href="https://console.anthropic.com/settings/keys" target="_blank" rel="noreferrer">
                      Get a key from the Anthropic Console
                    </a>
                  </>
                )}
                {error && <p className="error" role="alert">{error}</p>}
                <div className="auth-actions">
                  {phase > 0 && (
                    <button type="button" className="btn btn-ghost" onClick={() => go(phase - 1)} disabled={busy}>
                      Back
                    </button>
                  )}
                  {phase < 2 ? (
                    <button className="btn btn-primary">Continue</button>
                  ) : (
                    <>
                      <button type="button" className="btn" onClick={() => signup(null)} disabled={busy}>
                        Skip for now
                      </button>
                      <button className="btn btn-primary auth-final" disabled={busy || !form.key.trim()}>
                        {spinner}Link &amp; create account
                      </button>
                    </>
                  )}
                </div>
              </form>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
