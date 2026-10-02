const TOKEN_KEY = 'trainform_token'

export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (token) => localStorage.setItem(TOKEN_KEY, token)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

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
    if (res.status === 401 && token) {
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
    post('/me/password', { current_password, new_password, new_password_confirm }),
  setClaudeKey: (api_key) => request('/me/claude-key', { method: 'PUT', body: { api_key } }),
  removeClaudeKey: () => request('/me/claude-key', { method: 'DELETE' }),

  generatePrompt: (description) => post('/agents/prompt', { description }),
  generateRubric: (agent_prompt) => post('/agents/rubric', { agent_prompt }),

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
