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

// What the API's auth check returns; a wrong password at sign-in is a 401 without the WWW-Authenticate header.
const unauthenticated = () =>
  vi.fn(async () => new Response('{"detail":"Not authenticated"}', { status: 401, headers: { 'WWW-Authenticate': 'Bearer' } }))

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

  it('keeps sending the token it signed in with even if storage loses it mid-session', async () => {
    // The macOS app's WebKit storage has dropped the token right after sign-in, signing the user out on their next click.
    const fetch = respond(200, {})
    vi.stubGlobal('fetch', fetch)
    setToken('tok')
    localStorage.clear()
    await request('/thing')
    expect(fetch.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer tok' })
    expect(getToken()).toBe('tok')
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

  it('clears the token and tab session and announces logout on 401', async () => {
    vi.stubGlobal('fetch', unauthenticated())
    const onLogout = vi.fn()
    window.addEventListener('trainform:logout', onLogout)
    setToken('stale')
    sessionStorage.setItem('trainform_train_draft', '{}')
    await expect(request('/me')).rejects.toThrow('Not authenticated')
    expect(getToken()).toBeNull()
    expect(sessionStorage.length).toBe(0) // the next user in this tab gets none of it
    expect(onLogout).toHaveBeenCalledOnce()
    window.removeEventListener('trainform:logout', onLogout)
  })

  it('signs out a tab whose token was cleared elsewhere, such as by signing out in another tab', async () => {
    vi.stubGlobal('fetch', unauthenticated())
    const onLogout = vi.fn()
    window.addEventListener('trainform:logout', onLogout)
    await expect(request('/me')).rejects.toThrow('Not authenticated')
    expect(onLogout).toHaveBeenCalledOnce()
    window.removeEventListener('trainform:logout', onLogout)
  })

  it('does not sign out on a wrong password at sign-in', async () => {
    vi.stubGlobal('fetch', respond(401, { detail: 'Invalid username or password' }))
    const onLogout = vi.fn()
    window.addEventListener('trainform:logout', onLogout)
    await expect(request('/auth/login', { method: 'POST', body: {} })).rejects.toThrow('Invalid username or password')
    expect(onLogout).not.toHaveBeenCalled()
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
    ['generatePrompt', ['d', 'm', 'local'], 'POST', '/agents/prompt', { description: 'd', model: 'm', helper: 'local' }],
    ['generateRubric', ['p', 'm', 'claude'], 'POST', '/agents/rubric', { agent_prompt: 'p', model: 'm', helper: 'claude' }],
    ['downloadHubModel', ['org/m'], 'POST', '/hub/download', { repo_id: 'org/m' }],
    ['listLocalModels', [], 'GET', '/models'],
    ['startTraining', [{ rounds: 1 }], 'POST', '/trainings', { rounds: 1 }],
    ['listTrainings', [], 'GET', '/trainings'],
    ['listTrainings', [4], 'GET', '/trainings?agent_id=4'],
    ['listAgents', [], 'GET', '/agents'],
    ['getAgent', [4], 'GET', '/agents/4'],
    ['createAgent', [{ name: 'A', base_model: 'claude', prompt: 'p' }], 'POST', '/agents',
      { name: 'A', base_model: 'claude', prompt: 'p' }],
    ['updateAgent', [4, { prompt: 'p' }], 'PATCH', '/agents/4', { prompt: 'p' }],
    ['deleteAgent', [4], 'DELETE', '/agents/4'],
    ['getTraining', [3], 'GET', '/trainings/3'],
    ['listRounds', [3], 'GET', '/trainings/3/rounds'],
    ['downloadPrompt', [3], 'GET', '/trainings/3/download/prompt'],
    ['downloadWeights', [3], 'GET', '/trainings/3/download/weights'],
    ['pauseTraining', [3], 'POST', '/trainings/3/pause'],
    ['resumeTraining', [3], 'POST', '/trainings/3/resume'],
    ['deleteTraining', [3], 'DELETE', '/trainings/3'],
    ['generateAgentImage', [4, 'local'], 'POST', '/agents/4/image/generate', { helper: 'local' }],
    ['generateAgentImage', [4], 'POST', '/agents/4/image/generate', {}],
    ['setAgentImage', [4, '<svg/>'], 'PUT', '/agents/4/image', { svg: '<svg/>' }],
    ['setAgentImage', [4, null], 'PUT', '/agents/4/image', { svg: null }],
    ['generateAvatar', ['claude'], 'POST', '/me/avatar/generate', { helper: 'claude' }],
    ['removeAvatar', [], 'DELETE', '/me/avatar'],
    ['avatar', ['/social/users/ada/avatar?v=2'], 'GET', '/social/users/ada/avatar?v=2'],
    ['avatar', ['/api/social/users/ada/avatar'], 'GET', '/social/users/ada/avatar'],
    ['searchUsers', ['a b'], 'GET', '/social/users?q=a+b'],
    ['getProfile', ['a/b'], 'GET', '/social/users/a%2Fb'],
    ['follow', ['ada'], 'POST', '/social/users/ada/follow'],
    ['unfollow', ['ada'], 'DELETE', '/social/users/ada/follow'],
    ['publishPost', [4, 'Hi'], 'POST', '/social/posts', { agent_id: 4, caption: 'Hi' }],
    ['feed', [], 'GET', '/social/feed?limit=20'],
    ['feed', [9, 5], 'GET', '/social/feed?before=9&limit=5'],
    ['deletePost', [9], 'DELETE', '/social/posts/9'],
    ['vote', [9, -1], 'PUT', '/social/posts/9/vote', { value: -1 }],
    ['copyPost', [9], 'POST', '/social/posts/9/copy'],
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

  it('uploads a profile picture as a multipart PNG', async () => {
    const fetch = respond(200, { avatar_url: '/social/users/ada/avatar' })
    vi.stubGlobal('fetch', fetch)
    expect(await api.uploadAvatar(new Blob(['png'], { type: 'image/png' }))).toEqual({ avatar_url: '/social/users/ada/avatar' })
    const [url, init] = fetch.mock.calls[0]
    expect([url, init.method]).toEqual(['/api/me/avatar', 'PUT'])
    expect(init.body.get('file').name).toBe('avatar.png')
  })

  it('keeps the session with the fresh token a password change returns', async () => {
    vi.stubGlobal('fetch', respond(200, { token: 'fresh' }))
    setToken('old')
    await api.changePassword('a', 'b', 'b')
    expect(getToken()).toBe('fresh')
  })

  it('revokes the session on logout', async () => {
    const fetch = respond(204, null)
    vi.stubGlobal('fetch', fetch)
    setToken('tok')
    await api.logout()
    const [url, init] = fetch.mock.calls[0]
    expect([url, init.method, init.headers.Authorization]).toEqual(['/api/auth/logout', 'POST', 'Bearer tok'])
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
