import { useState } from 'react'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}

// Runs an async action, tracking whether it is in progress and the message of its last failure.
export function useAction() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (fn) => {
    setBusy(true)
    setError('')
    try {
      await fn()
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }
  return [run, busy, error]
}

export const profilePath = (username) => `/u/${encodeURIComponent(username)}`
// Swaps the updated post into a list of posts.
export const replacePost = (post) => (posts) => posts.map((p) => (p.id === post.id ? post : p))
