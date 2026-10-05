const TOKEN_KEY = 'trainform_token'

// The session's token lives here; localStorage only keeps it for the next launch. The macOS app's WebKit storage has
// lost it mid-session, which sent requests without it and signed the user out on their next click.
let token = localStorage.getItem(TOKEN_KEY)

export const getToken = () => token
export const setToken = (value) => {
  token = value
  localStorage.setItem(TOKEN_KEY, value)
}
// Signing out also forgets what this tab kept for the user, such as the Train page's draft.
export const clearToken = () => {
  token = null
  localStorage.removeItem(TOKEN_KEY)
  try {
    sessionStorage.clear()
  } catch {
    // Storage blocked: nothing was kept.
  }
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

const detailMessage = (detail) => (Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail)

export async function request(path, { method = 'GET', body, form } = {}) {
  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  const res = await fetch(`/api${path}`, {
    method,
    headers,
    body: form ?? (body === undefined ? undefined : JSON.stringify(body)),
  })
  if (!res.ok) {
    // The API's auth check (unlike a wrong password at sign-in) says WWW-Authenticate. It signs this tab out even
    // without a token, which another tab clears by signing out, so the page doesn't stay stuck on "Not authenticated".
    if (res.status === 401 && res.headers.has('WWW-Authenticate')) {
      clearToken()
      window.dispatchEvent(new Event('trainform:logout'))
    }
    const data = await res.json().catch(() => ({}))
    throw new ApiError(res.status, detailMessage(data.detail) || res.statusText || `HTTP ${res.status}`)
  }
  if (res.status === 204) return null
  return (res.headers.get('Content-Type') ?? '').includes('application/json') ? res.json() : res.blob()
}

const post = (path, body) => request(path, { method: 'POST', body })

export const api = {
  signup: (data) => post('/auth/signup', data),
  login: (username, password) => post('/auth/login', { username, password }),
  me: () => request('/me'),
  changePassword: (current_password, new_password, new_password_confirm) =>
    post('/me/password', { current_password, new_password, new_password_confirm }).then(({ token }) => setToken(token)),
  logout: () => post('/auth/logout'),
  setClaudeKey: (api_key) => request('/me/claude-key', { method: 'PUT', body: { api_key } }),
  removeClaudeKey: () => request('/me/claude-key', { method: 'DELETE' }),

  // `helper` ('claude' or 'local'; null: Claude if linked) runs the agents; `model` is the local model they'd use.
  generatePrompt: (description, model, helper) => post('/agents/prompt', { description, model, helper }),
  generateRubric: (agent_prompt, model, helper) => post('/agents/rubric', { agent_prompt, model, helper }),

  searchHubModels: (search = '', sort = 'downloads') => request(`/hub/models?${new URLSearchParams({ search, sort })}`),
  downloadHubModel: (repo_id) => post('/hub/download', { repo_id }),
  uploadModel: (file, name) => {
    const form = new FormData()
    form.append('file', file)
    form.append('name', name)
    return request('/hub/upload', { method: 'POST', form })
  },
  listLocalModels: () => request('/models'),

  startTraining: (config) => post('/trainings', config),
  listTrainings: () => request('/trainings'),
  getTraining: (id) => request(`/trainings/${id}`),
  pauseTraining: (id) => post(`/trainings/${id}/pause`),
  // Continues a paused or failed training from its saved progress (also the Retry after a failure).
  resumeTraining: (id) => post(`/trainings/${id}/resume`),
  deleteTraining: (id) => request(`/trainings/${id}`, { method: 'DELETE' }),
  listRounds: (id) => request(`/trainings/${id}/rounds`),
  listScenarios: (id, round) => request(`/trainings/${id}/rounds/${round}/scenarios`),
  downloadPrompt: (id) => request(`/trainings/${id}/download/prompt`),
  downloadWeights: (id) => request(`/trainings/${id}/download/weights`),
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
