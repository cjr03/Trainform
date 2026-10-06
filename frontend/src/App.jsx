import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { LogOut } from 'lucide-react'
import { api, clearToken, getToken, setToken } from './api'
import { Button } from '@/components/ui/button'
import AccountPage from './pages/AccountPage'
import AgentsPage from './pages/AgentsPage'
import AuthPage from './pages/AuthPage'
import InspectPage from './pages/InspectPage'
import TrainPage from './pages/TrainPage'

const NAV_LINK =
  'flex-1 rounded-md px-3 py-1.5 text-center text-sm font-medium text-muted-foreground no-underline transition-colors hover:bg-accent hover:text-foreground sm:flex-none aria-[current=page]:bg-primary/15 aria-[current=page]:text-primary'

export default function App() {
  const [authed, setAuthed] = useState(() => Boolean(getToken()))
  const navigate = useNavigate()

  useEffect(() => {
    const onLogout = () => setAuthed(false)
    window.addEventListener('trainform:logout', onLogout)
    return () => window.removeEventListener('trainform:logout', onLogout)
  }, [])

  const onAuth = ({ token }) => {
    clearToken() // a draft left in this tab by whoever signed in before (e.g. signed out from another tab)
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
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-lg">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-4 py-2.5">
          <span className="font-brand text-base font-bold tracking-[0.3em] text-gradient">TRAINFORM</span>
          <nav aria-label="Main" className="order-last flex w-full gap-1 sm:order-none sm:mr-auto sm:w-auto">
            {['Train', 'Agents', 'Inspect', 'Account'].map((name) => (
              <NavLink key={name} to={`/${name.toLowerCase()}`} className={NAV_LINK}>
                {name}
              </NavLink>
            ))}
          </nav>
          <Button variant="ghost" size="sm" onClick={logout}>
            <LogOut aria-hidden="true" />
            Sign out
          </Button>
        </div>
      </header>
      <main className="w-full px-4 pt-8 pb-16">
        <Routes>
          <Route path="/train" element={<TrainPage />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/inspect" element={<InspectPage />} />
          <Route path="/account" element={<AccountPage />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </main>
    </div>
  )
}
