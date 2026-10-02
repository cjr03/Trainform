import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, clearToken, getToken, request, saveBlob, setToken } from './api'

function respond(status, body, contentType = 'application/json') {
  return vi.fn(
    async () =>
      new Response(body === null ? null : typeof body === 'string' ? body : JSON.stringify(body), {
        status,
        headers: body === null ? {} : { 'Content-Type': contentType },
      }),
  )
}

beforeEach(() => clearToken())
afterEach(() => vi.unstubAllGlobals())

describe('request', () => {
  it('sends JSON with the bearer token and returns parsed JSON', async () => {
    const fetch = respond(200, { ok: true })
    vi.stubGlobal('fetch', fetch)
    setToken('tok')
    expect(await request('/thing', { method: 'POST', body: { a: 1 } })).toEqual({ ok: true })
    const [url, init] = fetch.mock.calls[0]
    expect(url).toBe('/api/thing')
    expect(init.method).toBe('POST')
    expect(init.headers).toEqual({ Authorization: 'Bearer tok', 'Content-Type': 'application/json' })
    expect(init.body).toBe('{"a":1}')
  })

  it('omits auth and content-type when there is no token or body', async () => {
    const fetch = respond(200, [])
    vi.stubGlobal('fetch', fetch)
    await request('/thing')
    expect(fetch.mock.calls[0][1].headers).toEqual({})
  })

  it('throws ApiError with the FastAPI detail string', async () => {
    vi.stubGlobal('fetch', respond(409, { detail: 'Username taken' }))
    const err = await request('/x').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.status).toBe(409)
    expect(err.message).toBe('Username taken')
  })

  it('joins validation error messages', async () => {
    vi.stubGlobal('fetch', respond(422, { detail: [{ msg: 'bad email' }, { msg: 'too short' }] }))
    await expect(request('/x')).rejects.toThrow('bad email; too short')
  })

  it('clears the token and announces logout on 401', async () => {
    vi.stubGlobal('fetch', respond(401, { detail: 'Not authenticated' }))
    const onLogout = vi.fn()
    window.addEventListener('trainform:logout', onLogout)
    setToken('stale')
    await expect(request('/me')).rejects.toThrow('Not authenticated')
    expect(getToken()).toBeNull()
    expect(onLogout).toHaveBeenCalledOnce()
    window.removeEventListener('trainform:logout', onLogout)
  })

  it('returns null for 204 and a Blob for files', async () => {
    vi.stubGlobal('fetch', respond(204, null))
    expect(await request('/x')).toBeNull()
    vi.stubGlobal('fetch', respond(200, 'PROMPT', 'text/plain'))
    const blob = await request('/x')
    expect(blob).toBeInstanceOf(Blob)
    expect(await blob.text()).toBe('PROMPT')
  })
})

describe('api', () => {
  it.each([
    ['signup', [{ username: 'ada' }], 'POST', '/auth/signup', { username: 'ada' }],
    ['login', ['ada', 'pw'], 'POST', '/auth/login', { username: 'ada', password: 'pw' }],
    ['me', [], 'GET', '/me'],
    ['changePassword', ['a', 'b', 'b'], 'POST', '/me/password',
      { current_password: 'a', new_password: 'b', new_password_confirm: 'b' }],
    ['setClaudeKey', ['sk'], 'PUT', '/me/claude-key', { api_key: 'sk' }],
    ['removeClaudeKey', [], 'DELETE', '/me/claude-key'],
    ['generatePrompt', ['d'], 'POST', '/agents/prompt', { description: 'd' }],
    ['generateRubric', ['p'], 'POST', '/agents/rubric', { agent_prompt: 'p' }],
    ['downloadHubModel', ['org/m'], 'POST', '/hub/download', { repo_id: 'org/m' }],
    ['listLocalModels', [], 'GET', '/models'],
    ['startTraining', [{ rounds: 1 }], 'POST', '/trainings', { rounds: 1 }],
    ['listTrainings', [], 'GET', '/trainings'],
    ['getTraining', [3], 'GET', '/trainings/3'],
    ['listRounds', [3], 'GET', '/trainings/3/rounds'],
    ['downloadPrompt', [3], 'GET', '/trainings/3/download/prompt'],
    ['downloadWeights', [3], 'GET', '/trainings/3/download/weights'],
  ])('%s sends %s %s', async (name, args, method, path, body) => {
    const fetch = respond(200, {})
    vi.stubGlobal('fetch', fetch)
    await api[name](...args)
    const [url, init] = fetch.mock.calls[0]
    expect([url, init.method]).toEqual([`/api${path}`, method])
    expect(init.body).toBe(body === undefined ? undefined : JSON.stringify(body))
  })

  it('uploads models as multipart form data', async () => {
    const fetch = respond(201, { name: 'mine', source: 'upload' })
    vi.stubGlobal('fetch', fetch)
    const file = new File(['zip'], 'm.zip')
    await api.uploadModel(file, 'mine')
    const init = fetch.mock.calls[0][1]
    expect(init.body).toBeInstanceOf(FormData)
    expect(init.body.get('name')).toBe('mine')
    expect(init.headers['Content-Type']).toBeUndefined()
  })

  it('builds query strings and nested paths', async () => {
    const fetch = respond(200, [])
    vi.stubGlobal('fetch', fetch)
    await api.searchHubModels('qwen', 'lastModified')
    await api.listScenarios(7, 2)
    expect(fetch.mock.calls.map((c) => c[0])).toEqual([
      '/api/hub/models?search=qwen&sort=lastModified',
      '/api/trainings/7/rounds/2/scenarios',
    ])
  })

  it('sends each wrapper to its route with the right method and body', async () => {
    const fetch = respond(200, {})
    vi.stubGlobal('fetch', fetch)
    const config = { base_model: 'm', prompt: 'p', rubric: ['c'], scenarios: 1, rounds: 1 }
    await api.downloadHubModel('org/tiny')
    await api.changePassword('old', 'new', 'new2')
    await api.setClaudeKey('sk-ant')
    await api.removeClaudeKey()
    await api.startTraining(config)
    await api.listRounds(7)
    await api.downloadPrompt(7)
    await api.downloadWeights(7)
    const calls = fetch.mock.calls.map(([url, init]) => [init.method, url, init.body && JSON.parse(init.body)])
    expect(calls).toEqual([
      ['POST', '/api/hub/download', { repo_id: 'org/tiny' }],
      ['POST', '/api/me/password', { current_password: 'old', new_password: 'new', new_password_confirm: 'new2' }],
      ['PUT', '/api/me/claude-key', { api_key: 'sk-ant' }],
      ['DELETE', '/api/me/claude-key', undefined],
      ['POST', '/api/trainings', config],
      ['GET', '/api/trainings/7/rounds', undefined],
      ['GET', '/api/trainings/7/download/prompt', undefined],
      ['GET', '/api/trainings/7/download/weights', undefined],
    ])
  })
})

it('saveBlob downloads the blob under the filename and revokes its URL', () => {
  const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function () {
    expect([this.href, this.download]).toEqual(['blob:fake', 'prompt.txt'])
  })
  URL.createObjectURL = vi.fn(() => 'blob:fake')
  URL.revokeObjectURL = vi.fn()
  const blob = new Blob(['hi'])
  saveBlob(blob, 'prompt.txt')
  expect(URL.createObjectURL).toHaveBeenCalledWith(blob)
  expect(click).toHaveBeenCalledOnce()
  expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake')
  click.mockRestore()
})
