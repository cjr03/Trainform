import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { api, clearToken, getToken, setToken } from './api'
import AccountPage from './pages/AccountPage'
import AuthPage from './pages/AuthPage'
import InspectPage from './pages/InspectPage'
import TrainPage from './pages/TrainPage'

export default function App() {
  const [authed, setAuthed] = useState(() => Boolean(getToken()))
  const navigate = useNavigate()

  useEffect(() => {
    const onLogout = () => setAuthed(false)
    window.addEventListener('trainform:logout', onLogout)
    return () => window.removeEventListener('trainform:logout', onLogout)
  }, [])

  const onAuth = ({ token }) => {
    setToken(token)
    setAuthed(true)
    navigate('/train')
  }
  const logout = () => {
    api.logout().catch(() => {}) // revokes the token server-side; it is read before clearToken runs
    clearToken()
    setAuthed(false)
  }

  if (!authed) return <AuthPage onAuth={onAuth} />

  return (
    <div className="app-shell">
      <nav className="nav glass">
        <span className="brand">TRAINFORM</span>
        <div className="nav-links">
          <NavLink to="/train">Train</NavLink>
          <NavLink to="/inspect">Inspect</NavLink>
          <NavLink to="/account">Account</NavLink>
        </div>
        <button className="btn btn-ghost" onClick={logout}>
          Sign out
        </button>
      </nav>
      <main className="page">
        <Routes>
          <Route path="/train" element={<TrainPage />} />
          <Route path="/inspect" element={<InspectPage />} />
          <Route path="/account" element={<AccountPage />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </main>
    </div>
  )
}
