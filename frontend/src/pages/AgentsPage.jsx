import { Download, Plus, Sparkles, Trash2, Upload } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
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
import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Textarea } from '@/components/ui/textarea'
import { Choice, ErrorAlert, Overall, Spinner } from '@/components/shared'
import { cn, useAction } from '@/lib/utils'
import { api } from '../api'

export const CLAUDE_NOTE = "Claude's weights can't be fine-tuned, so training will only improve its prompt."
const compact = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 })
const score = (s) => (s == null ? '—' : s.toFixed(1))
const modelName = (m) => (m === 'claude' ? 'Claude' : m)
const count = (n) => `${n} ${n === 1 ? 'training' : 'trainings'}`
const ACTIVE = ['pending', 'running', 'pausing']

// Pick a local model, downloading one from the Hugging Face Hub or uploading a zipped one first.
function ModelPicker({ value, onChange }) {
  const [tab, setTab] = useState('hub')
  const [models, setModels] = useState(null)
  const [listError, setListError] = useState('')
  const [search, setSearch] = useState('')
  const [sort, setSort] = useState('downloads')
  const [results, setResults] = useState(null)
  const [downloading, setDownloading] = useState('')
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [runHub, hubBusy, hubError] = useAction()
  const [runUpload, uploading, uploadError] = useAction()

  const refresh = async () => {
    setModels(await api.listLocalModels())
    setListError('')
  }
  useEffect(() => {
    api.listLocalModels().then(setModels, (e) => setListError(e.message))
  }, [])

  const onSearch = (e) => {
    e.preventDefault()
    runHub(async () => setResults(await api.searchHubModels(search.trim(), sort)))
  }
  const download = async (id) => {
    setDownloading(id)
    await runHub(async () => {
      const model = await api.downloadHubModel(id)
      await refresh()
      onChange(model.name)
    })
    setDownloading('')
  }
  const upload = (e) => {
    e.preventDefault()
    runUpload(async () => {
      const model = await api.uploadModel(file, name.trim())
      await refresh()
      onChange(model.name)
    })
  }

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div className="grid gap-4">
        <Choice
          aria-label="Model source"
          value={tab}
          onValueChange={setTab}
          options={[
            ['hub', 'Hugging Face Hub'],
            ['upload', 'Upload'],
          ]}
        />

        <div hidden={tab !== 'hub'} className="grid gap-4">
          <form role="search" className="grid gap-2 sm:grid-cols-[1fr_auto_auto]" onSubmit={onSearch}>
            <Input
              type="search"
              aria-label="Search models"
              placeholder="Search models, e.g. Qwen"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <select
              aria-label="Sort by"
              className="h-9 rounded-md border border-input bg-input/30 px-3 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [&_option]:bg-background"
              value={sort}
              onChange={(e) => setSort(e.target.value)}
            >
              <option value="downloads">Most downloaded</option>
              <option value="lastModified">Latest</option>
            </select>
            <Button disabled={hubBusy}>
              {hubBusy && !downloading && <Spinner />}Search
            </Button>
          </form>
          <ErrorAlert>{hubError}</ErrorAlert>
          {results === null ? (
            <p className="m-0 text-sm text-muted-foreground">
              Search the Hub to find a base model. Small instruct models train fastest.
            </p>
          ) : results.length === 0 ? (
            <p className="m-0 text-sm text-muted-foreground">No models match that search. Try a broader name.</p>
          ) : (
            <ul className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(min(230px,100%),1fr))] gap-3 p-0">
              {results.map((m) => (
                <li key={m.id} className="flex flex-col gap-3 rounded-lg border bg-background/40 p-4">
                  <h3 className="m-0 font-sans text-sm font-semibold tracking-normal [overflow-wrap:anywhere]">
                    {m.id}
                  </h3>
                  <dl className="m-0 grid grid-cols-3 gap-2 text-sm tabular-nums">
                    {[
                      ['Downloads', compact.format(m.downloads ?? 0)],
                      ['Likes', compact.format(m.likes ?? 0)],
                      ['Updated', m.last_modified ? new Date(m.last_modified).toLocaleDateString() : '—'],
                    ].map(([term, detail]) => (
                      <div key={term}>
                        <dt className="text-xs text-muted-foreground">{term}</dt>
                        <dd className="m-0">{detail}</dd>
                      </div>
                    ))}
                  </dl>
                  <Button
                    type="button"
                    variant="outline"
                    className="mt-auto"
                    aria-label={`Download ${m.id}`}
                    disabled={Boolean(downloading)}
                    onClick={() => download(m.id)}
                  >
                    {downloading === m.id ? (
                      <>
                        <Spinner />
                        Downloading…
                      </>
                    ) : (
                      <>
                        <Download />
                        Download
                      </>
                    )}
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <form hidden={tab !== 'upload'} className="grid gap-4" onSubmit={upload}>
          <p className="m-0 text-sm text-muted-foreground">
            Zip a Hugging Face model folder (config, tokenizer and weights) and upload it.
          </p>
          <div className="grid gap-2">
            <Label htmlFor="agent-upload-file">Model archive (.zip)</Label>
            <Input
              id="agent-upload-file"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setFile(e.target.files[0] ?? null)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="agent-upload-name">Model name</Label>
            <Input id="agent-upload-name" value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <ErrorAlert>{uploadError}</ErrorAlert>
          <div>
            <Button disabled={!file || !name.trim() || uploading}>
              {uploading ? <Spinner /> : <Upload />}Upload model
            </Button>
          </div>
        </form>
      </div>

      <div className="grid gap-3 rounded-lg border bg-background/40 p-4">
        <span id="agent-local-models" className="text-sm font-medium">
          Local models
        </span>
        {!models && !listError && <Spinner />}
        <ErrorAlert>{listError}</ErrorAlert>
        {models?.length === 0 && (
          <p className="m-0 text-sm text-muted-foreground">
            No local models yet. Download one from the Hub or upload a zip.
          </p>
        )}
        <RadioGroup aria-labelledby="agent-local-models" value={value} onValueChange={onChange} className="gap-1">
          {models?.map((m) => (
            <label
              key={m.name}
              className="flex cursor-pointer items-center gap-3 rounded-md border border-transparent px-3 py-2.5 text-sm transition-colors hover:bg-accent/50 has-data-[state=checked]:border-border-strong has-data-[state=checked]:bg-primary/10"
            >
              <RadioGroupItem value={m.name} />
              <span className="mr-auto [overflow-wrap:anywhere]">{m.name}</span>
              {m.source && <Badge variant="outline">{m.source}</Badge>}
            </label>
          ))}
        </RadioGroup>
      </div>
    </div>
  )
}

// The agent's system prompt: typed, read from a text file, or drafted by the helper from a description.
// `model` is the agent's base model, which drafts it when Claude isn't linked.
function PromptEditor({ value, onChange, model }) {
  const id = useId()
  const file = useRef(null)
  const [mode, setMode] = useState('write')
  const [description, setDescription] = useState('')
  const [run, busy, error] = useAction()
  const replace = () => !value.trim() || window.confirm('Replace your prompt?')
  // Without a helper choice here, the server drafts with Claude when linked, else on this local model.
  const request = () =>
    run(async () => onChange((await api.generatePrompt(description.trim(), model === 'claude' ? null : model || null, null)).prompt))
  const generate = () => replace() && request()
  const upload = (e) =>
    run(async () => {
      const picked = e.target.files[0]
      e.target.value = '' // so picking the same file again still uploads it
      if (picked && replace()) onChange(await picked.text())
    })

  return (
    <div className="grid gap-4">
      <Choice
        aria-label="How to create the prompt"
        value={mode}
        onValueChange={setMode}
        options={[
          ['write', 'Write my own'],
          ['describe', 'Describe my agent'],
        ]}
      />
      {mode === 'describe' && (
        <div className="grid gap-2">
          <Label htmlFor={`${id}-description`}>Agent description</Label>
          <Textarea
            id={`${id}-description`}
            placeholder="A support agent for a bike shop that handles returns and repair bookings."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" disabled={!description.trim() || busy} onClick={generate}>
              {busy ? <Spinner /> : <Sparkles />}Generate prompt
            </Button>
            {error && (
              <Button type="button" variant="ghost" disabled={!description.trim() || busy} onClick={request}>
                Retry
              </Button>
            )}
          </div>
        </div>
      )}
      <div className="grid gap-2">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <Label htmlFor={`${id}-prompt`}>Agent prompt</Label>
          <Button type="button" variant="outline" size="sm" onClick={() => file.current.click()}>
            <Upload />
            Upload prompt
          </Button>
          <input
            ref={file}
            type="file"
            accept=".txt,.md,text/plain,text/markdown"
            aria-label="Prompt file"
            hidden
            onChange={upload}
          />
        </div>
        <Textarea
          id={`${id}-prompt`}
          className="min-h-60 font-mono"
          placeholder="You are a friendly support agent for…"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
      <ErrorAlert>{error}</ErrorAlert>
    </div>
  )
}

function NewAgent({ claudeAvailable, onCreated, onCancel }) {
  const [name, setName] = useState('')
  const [kind, setKind] = useState('local')
  const [model, setModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [run, busy, error] = useAction()
  const baseModel = kind === 'claude' ? 'claude' : model
  const create = () =>
    run(async () => onCreated(await api.createAgent({ name: name.trim(), base_model: baseModel, prompt: prompt.trim() })))

  return (
    <Card className="gap-5 max-sm:py-5">
      <CardHeader className="max-sm:px-5">
        <h2 className="m-0 text-xl">New agent</h2>
        <CardDescription className="max-w-prose">
          Name it, choose the model it runs on and write the system prompt it starts from.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-6 max-sm:px-5">
        <div className="grid gap-2">
          <Label htmlFor="agent-name">Name</Label>
          <Input id="agent-name" maxLength={100} value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="grid gap-3">
          <span className="text-sm font-medium" id="agent-model">
            Model
          </span>
          <Choice
            aria-labelledby="agent-model"
            value={kind}
            onValueChange={setKind}
            options={[
              ['local', 'Local model'],
              [
                'claude',
                'Claude',
                { disabled: !claudeAvailable, 'aria-describedby': claudeAvailable === false ? 'agent-claude-hint' : undefined },
              ],
            ]}
          />
          {claudeAvailable === false && (
            <p id="agent-claude-hint" className="m-0 text-sm text-muted-foreground">
              Link your Claude API key on the{' '}
              <Link to="/account" className="text-primary underline-offset-4 hover:underline">
                Account
              </Link>{' '}
              page to make a Claude agent.
            </p>
          )}
          {kind === 'claude' ? (
            <p className="m-0 text-sm text-muted-foreground">{CLAUDE_NOTE}</p>
          ) : (
            <ModelPicker value={model} onChange={setModel} />
          )}
        </div>
        <PromptEditor value={prompt} onChange={setPrompt} model={baseModel} />
        <ErrorAlert>{error}</ErrorAlert>
        {/* Not a form: the model picker holds forms of its own. */}
        <div className="flex flex-wrap justify-end gap-2">
          <Button type="button" variant="ghost" onClick={onCancel}>
            Cancel
          </Button>
          <Button type="button" disabled={!name.trim() || !baseModel || !prompt.trim() || busy} onClick={create}>
            {busy ? <Spinner /> : <Plus />}Create agent
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

function AgentDetail({ id, onChanged, onDeleted }) {
  const [agent, setAgent] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [draft, setDraft] = useState(null) // the prompt being edited
  const [run, busy, error] = useAction()

  useEffect(() => {
    let live = true
    api.getAgent(id).then(
      (a) => live && setAgent(a),
      (e) => live && setLoadError(e.message),
    )
    return () => {
      live = false
    }
  }, [id])

  if (loadError) return <ErrorAlert>{loadError}</ErrorAlert>
  if (!agent) return <Spinner />
  const active = agent.trainings.some((t) => ACTIVE.includes(t.status))
  const save = () =>
    run(async () => {
      setAgent({ ...agent, prompt: (await api.updateAgent(id, { prompt: draft.trim() })).prompt })
      setDraft(null)
      onChanged()
    })
  const remove = () =>
    run(async () => {
      await api.deleteAgent(id)
      onDeleted()
    })

  return (
    <Card role="region" aria-labelledby="agent-detail-name" className="gap-5 max-sm:py-5">
      <CardContent className="grid gap-5 max-sm:px-5">
        <div className="flex items-center gap-4">
          <Overall value={agent.overall} large />
          <div className="grid min-w-0 gap-1">
            <h2 id="agent-detail-name" className="m-0 text-2xl wrap-anywhere">
              {agent.name}
            </h2>
            <span className="text-sm text-muted-foreground wrap-anywhere">
              {modelName(agent.base_model)} · {count(agent.trainings.length)}
            </span>
          </div>
        </div>
        {agent.base_model === 'claude' && <p className="m-0 text-sm text-muted-foreground">{CLAUDE_NOTE}</p>}

        <section aria-label="Prompt" className="grid gap-2">
          <h3 className="m-0 text-base">Prompt</h3>
          {draft == null ? (
            <>
              <pre className="m-0 max-h-80 overflow-auto rounded-lg border bg-background/60 p-4 text-sm whitespace-pre-wrap [overflow-wrap:anywhere]">
                {agent.prompt}
              </pre>
              <div>
                <Button type="button" variant="outline" disabled={active} onClick={() => setDraft(agent.prompt)}>
                  Edit prompt
                </Button>
              </div>
            </>
          ) : (
            <>
              <PromptEditor value={draft} onChange={setDraft} model={agent.base_model} />
              <div className="flex flex-wrap gap-2">
                <Button type="button" disabled={!draft.trim() || busy} onClick={save}>
                  {busy && <Spinner />}Save prompt
                </Button>
                <Button type="button" variant="ghost" onClick={() => setDraft(null)}>
                  Cancel
                </Button>
              </div>
            </>
          )}
        </section>

        <section aria-label="Past trainings" className="grid gap-2">
          <h3 className="m-0 text-base">Past trainings</h3>
          {agent.trainings.length ? (
            <ul className="m-0 grid list-none gap-1.5 p-0">
              {agent.trainings.map((t) => (
                <li key={t.id} className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border bg-background/40 px-3 py-2 text-sm">
                  <span className="font-semibold">#{t.id}</span>
                  <Badge variant="outline" className={`badge-${t.status} tracking-wider uppercase`}>
                    {t.status}
                  </Badge>
                  <span className="mr-auto text-muted-foreground">
                    {new Date(t.created_at).toLocaleDateString()} · {t.rounds} rounds × {t.scenarios_per_round} scenarios
                  </span>
                  <span className="font-display text-brand tabular-nums" title="Best score">
                    {score(t.best_score)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="m-0 text-sm text-muted-foreground">
              No trainings yet. Train it on the{' '}
              <Link to="/train" className="text-primary underline-offset-4 hover:underline">
                Train page
              </Link>
              .
            </p>
          )}
        </section>

        <ErrorAlert>{error}</ErrorAlert>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline">
            <Link to={`/inspect?agent=${id}`}>Open in Inspect</Link>
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" className="border-destructive/50 text-destructive" disabled={busy || active}>
                <Trash2 aria-hidden="true" /> Delete agent
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete {agent.name}?</AlertDialogTitle>
                <AlertDialogDescription>
                  This permanently deletes the agent, its weights and all its trainings.
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
        </div>
      </CardContent>
    </Card>
  )
}

export default function AgentsPage() {
  const [agents, setAgents] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [selected, setSelected] = useState(null)
  const [creating, setCreating] = useState(false)
  const [claudeAvailable, setClaudeAvailable] = useState(null)

  const reload = () =>
    api.listAgents().then(
      (list) => {
        setAgents(list)
        setLoadError('')
      },
      (e) => setLoadError(e.message),
    )
  useEffect(() => {
    reload()
    api.me().then((me) => setClaudeAvailable(me.claude_available), () => {})
  }, [])

  const created = (agent) => {
    setCreating(false)
    setSelected(agent.id)
    reload()
  }
  const deleted = () => {
    setSelected(null)
    reload()
  }

  return (
    <div className="mx-auto grid w-full max-w-5xl gap-5" data-testid="agents-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Agents</h1>
      {!creating && (
        <div className="flex justify-end">
          <Button type="button" onClick={() => setCreating(true)}>
            <Plus />
            New agent
          </Button>
        </div>
      )}
      {creating && <NewAgent claudeAvailable={claudeAvailable} onCreated={created} onCancel={() => setCreating(false)} />}

      <ErrorAlert>{loadError}</ErrorAlert>
      {!agents && !loadError && <Spinner />}
      {agents?.length === 0 && <p className="m-0 text-center text-muted-foreground">No agents yet. Make your first one.</p>}
      {agents?.length > 0 && (
        <ul aria-label="Your agents" className="m-0 grid list-none grid-cols-[repeat(auto-fill,minmax(min(260px,100%),1fr))] gap-3 p-0">
          {agents.map((a) => (
            <li key={a.id}>
              <button
                type="button"
                aria-pressed={a.id === selected}
                onClick={() => setSelected(a.id)}
                className={cn(
                  'flex w-full cursor-pointer items-center gap-4 rounded-xl border bg-card p-4 text-left transition-colors hover:bg-white/6',
                  'aria-pressed:border-border-strong aria-pressed:bg-brand/8 aria-pressed:shadow-glow',
                )}
              >
                <Overall value={a.overall} />
                <span className="grid min-w-0 gap-1">
                  <span className="font-semibold wrap-anywhere">{a.name}</span>
                  <span className="text-[0.8rem] text-muted-foreground wrap-anywhere">
                    {modelName(a.base_model)} · {count(a.trainings)}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {selected != null && <AgentDetail key={selected} id={selected} onChanged={reload} onDeleted={deleted} />}
    </div>
  )
}
