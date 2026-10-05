import { CircleAlert, Download, Eye, Laptop, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { cn } from '@/lib/utils'
import { api } from '../api'

const MODES = [
  ['login', 'Sign in'],
  ['signup', 'Create account'],
]
const STEPS = ['Identity', 'Security', 'Link Claude']

const HOW = [
  ['Set the goal', 'A system prompt and a rubric of what every good reply does. Write them or have them drafted.'],
  ['Simulate', 'Realistic customers talk to your agent, and every reply is scored against the rubric.'],
  ['Improve', 'The prompt is rewritten after each conversation and a local model is fine-tuned on the best replies.'],
  ['Take it with you', 'Download the best prompt and LoRA weights. Every conversation and score stays in Inspect.'],
]
const BENEFITS = [
  [Download, 'Yours to keep', 'Download the best prompt and LoRA weights for an open-weight model.'],
  [Laptop, 'Local or Claude', 'Run the helper agents on Claude with your API key, or keep everything on your machine.'],
  [Eye, 'Nothing hidden', 'Inspect every simulated conversation and every score.'],
]

function Field({ label, ...props }) {
  return (
    <div className="grid gap-2">
      <Label htmlFor={props.id}>{label}</Label>
      <Input {...props} />
    </div>
  )
}

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

  const spinner = busy && <Loader2 className="animate-spin" aria-hidden="true" />
  const errorAlert = error && (
    <Alert variant="destructive">
      <CircleAlert aria-hidden="true" />
      <AlertDescription>{error}</AlertDescription>
    </Alert>
  )
  const pane = cn('grid gap-4 animate-in fade-in duration-300', dir === 'back' ? 'slide-in-from-left-4' : 'slide-in-from-right-4')

  return (
    <div
      className="mx-auto grid min-h-screen w-full max-w-6xl content-center gap-x-16 gap-y-10 px-4 py-10 lg:grid-cols-[minmax(0,1fr)_420px]"
      data-testid="auth-page"
    >
      <header className="grid gap-5">
        <span className="font-brand text-lg font-bold tracking-[0.3em] text-gradient">TRAINFORM</span>
        <h1 className="m-0 text-4xl leading-tight text-balance sm:text-5xl">
          Describe your agent. <span className="text-gradient">Get back a better one.</span>
        </h1>
        <p className="m-0 max-w-xl text-lg text-muted-foreground">
          Trainform turns a plain-language description into a tuned system prompt and fine-tuned local model weights,
          tested against simulated customers and graded reply by reply. No test cases to write.
        </p>
      </header>

      <Card className="w-full gap-0 border-border-strong py-6 shadow-glow lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:self-start">
        <CardContent>
          <Tabs value={mode} onValueChange={switchMode} className="gap-6">
            <TabsList aria-label="Account access" className="grid h-10 w-full grid-cols-2">
              {MODES.map(([m, label]) => (
                <TabsTrigger key={m} value={m} className="data-[state=active]:text-primary">
                  {label}
                </TabsTrigger>
              ))}
            </TabsList>

            <TabsContent value="login">
              <form className={pane} onSubmit={login}>
                <Field label="Username" {...bind('username')} autoComplete="username" required />
                <Field
                  label="Password"
                  {...bind('password')}
                  type="password"
                  autoComplete="current-password"
                  required
                />
                {errorAlert}
                <Button size="lg" disabled={busy}>
                  {spinner}Sign in
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="signup">
              <ol aria-label="Sign-up progress" className="m-0 mb-6 flex list-none gap-2 p-0">
                {STEPS.map((step, i) => (
                  <li
                    key={step}
                    aria-current={i === phase ? 'step' : undefined}
                    className="group flex flex-1 flex-col gap-2 text-xs text-muted-foreground aria-[current=step]:text-foreground"
                  >
                    <span
                      className={cn(
                        'h-1 rounded-full bg-muted transition-colors duration-300',
                        i <= phase && 'bg-primary',
                      )}
                    />
                    <span>
                      <span className="sr-only">Step </span>
                      {i + 1}. {step}
                    </span>
                  </li>
                ))}
              </ol>

              <form key={phase} className={pane} onSubmit={next} noValidate>
                {phase === 0 && (
                  <>
                    <Field
                      label="Email"
                      {...bind('email')}
                      type="email"
                      autoComplete="email"
                      required
                      autoFocus={dir === 'back'}
                    />
                    <Field
                      label="Username"
                      {...bind('username')}
                      autoComplete="username"
                      pattern="[A-Za-z0-9_.\-]{3,32}"
                      maxLength={32}
                      required
                    />
                  </>
                )}
                {phase === 1 && (
                  <>
                    <Field
                      label="Password"
                      {...bind('password')}
                      type="password"
                      autoComplete="new-password"
                      maxLength={128}
                      aria-describedby="auth-password-hint"
                      autoFocus
                    />
                    <p id="auth-password-hint" className="m-0 text-sm text-muted-foreground">
                      At least 8 characters.
                    </p>
                    <Field
                      label="Confirm password"
                      {...bind('password_confirm')}
                      type="password"
                      autoComplete="new-password"
                      maxLength={128}
                    />
                  </>
                )}
                {phase === 2 && (
                  <>
                    <p className="m-0 text-sm text-muted-foreground">
                      Trainform's helper agents write prompts, build rubrics, play personas and score conversations. They
                      run on your own Claude key, stored encrypted, or without one on the model you train, with much weaker
                      results. You can link a key later from Account.
                    </p>
                    <Field
                      label="Anthropic API key"
                      {...bind('key')}
                      type="password"
                      autoComplete="off"
                      placeholder="sk-ant-…"
                      autoFocus
                    />
                    <a
                      className="text-sm"
                      href="https://console.anthropic.com/settings/keys"
                      target="_blank"
                      rel="noreferrer"
                    >
                      Get a key from the Anthropic Console
                    </a>
                  </>
                )}
                {errorAlert}
                <div className="flex flex-wrap gap-2">
                  {phase === 2 && (
                    <Button size="lg" className="w-full" disabled={busy || !form.key.trim()}>
                      {spinner}Link &amp; create account
                    </Button>
                  )}
                  {phase > 0 && (
                    <Button type="button" variant="ghost" onClick={() => go(phase - 1)} disabled={busy}>
                      Back
                    </Button>
                  )}
                  {phase < 2 ? (
                    <Button size="lg" className="ml-auto min-w-36">
                      Continue
                    </Button>
                  ) : (
                    <Button type="button" variant="outline" className="ml-auto" onClick={() => signup(null)} disabled={busy}>
                      Skip for now
                    </Button>
                  )}
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <section aria-labelledby="how-title" className="grid gap-8">
        <div>
          <h2 id="how-title" className="mb-4 text-sm font-semibold tracking-[0.2em] text-muted-foreground uppercase">
            How it works
          </h2>
          <ol className="m-0 grid list-none gap-4 p-0 sm:grid-cols-2">
            {HOW.map(([title, text], i) => (
              <li key={title} className="flex gap-3">
                <span
                  aria-hidden="true"
                  className="grid size-7 shrink-0 place-items-center rounded-full border border-primary/40 font-display text-sm text-primary"
                >
                  {i + 1}
                </span>
                <div>
                  <h3 className="mb-1 text-base">{title}</h3>
                  <p className="m-0 text-sm text-muted-foreground">{text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
        <ul aria-label="What you get" className="m-0 grid list-none gap-3 border-t p-0 pt-8 sm:grid-cols-3">
          {BENEFITS.map(([Icon, title, text]) => (
            <li key={title} className="grid content-start gap-1">
              <span className="flex items-center gap-2 font-medium">
                <Icon className="size-4 text-brand-2" aria-hidden="true" />
                {title}
              </span>
              <span className="text-sm text-muted-foreground">{text}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  )
}
