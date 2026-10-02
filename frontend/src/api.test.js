import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { api, ApiError, clearToken, getToken, request, setToken } from './api'

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
