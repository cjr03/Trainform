import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { Gem, Hammer, LogOut, Map as MapIcon, Search, Swords, UserRound } from 'lucide-react'
import { api, clearToken, getToken, setToken } from './api'
import KnightHelmet from '@/components/KnightHelmet'
import { Avatar } from '@/components/shared'
import { PublishDialog } from '@/components/social'
import { profilePath } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import AccountPage from './pages/AccountPage'
import AgentsPage from './pages/AgentsPage'
import AuthPage from './pages/AuthPage'
import FeedPage from './pages/FeedPage'
import InspectPage from './pages/InspectPage'
import ProfilePage from './pages/ProfilePage'
import SearchPage from './pages/SearchPage'
import TrainPage from './pages/TrainPage'

const NAV = [
  ['Agents', Hammer],
  ['Inspect', KnightHelmet],
  ['Train', Swords],
  ['Battle', Gem],
]
const NAV_LINK =
  'inline-flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-center text-sm font-medium text-muted-foreground no-underline transition-colors hover:bg-accent hover:text-foreground lg:flex-none aria-[current=page]:bg-primary/15 aria-[current=page]:text-primary'

export default function App() {
  const [authed, setAuthed] = useState(() => Boolean(getToken()))
  const [me, setMe] = useState(null)
  const navigate = useNavigate()

  // Your username and picture, for the header.
  useEffect(() => {
    if (authed) api.me().then(setMe, () => {})
  }, [authed])

  useEffect(() => {
    const onLogout = () => {
      setAuthed(false)
      setMe(null)
    }
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
    setMe(null)
  }

  if (!authed) return <AuthPage onAuth={onAuth} />

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-lg">
        {/* The logo in the middle column; the tabs and icons share the right one on wide screens, and below lg the
            tabs take a second row (the wrapper's `contents` lets them join the outer grid). */}
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-x-4 gap-y-2 px-4 py-2.5">
          <span className="col-start-2 row-start-1 font-brand text-base font-bold tracking-[0.3em] text-gradient">
            TRAINFORM
          </span>
          <div className="contents lg:col-start-3 lg:row-start-1 lg:flex lg:items-center lg:justify-end lg:gap-1">
            <nav aria-label="Main" className="col-span-3 row-start-2 flex gap-1">
              {NAV.map(([name, Icon]) => (
                <NavLink key={name} to={`/${name.toLowerCase()}`} className={NAV_LINK}>
                  <Icon aria-hidden="true" className="size-4" />
                  {name}
                </NavLink>
              ))}
            </nav>
            <div className="col-start-3 row-start-1 flex items-center justify-self-end">
              <PublishDialog username={me?.username} />
              <Button asChild variant="ghost" size="icon-sm" className="text-foreground aria-[current=page]:text-primary">
                <NavLink to="/feed" aria-label="Feed" title="Feed">
                  <MapIcon aria-hidden="true" />
                </NavLink>
              </Button>
              <Button asChild variant="ghost" size="icon-sm" className="text-foreground aria-[current=page]:text-primary">
                <NavLink to="/search" aria-label="Search Trainformers" title="Search Trainformers">
                  <Search aria-hidden="true" />
                </NavLink>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" aria-label="Account menu" title="Account">
                    <Avatar url={me?.avatar_url} />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="max-w-[calc(100vw-2rem)]">
                  {me && (
                    <DropdownMenuItem onSelect={() => navigate(profilePath(me.username))}>
                      <Avatar url={me.avatar_url} />
                      <span className="font-semibold wrap-anywhere">{me.username}</span>
                    </DropdownMenuItem>
                  )}
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => navigate('/account')}>
                    <UserRound aria-hidden="true" />
                    Account
                  </DropdownMenuItem>
                  <DropdownMenuItem onSelect={logout}>
                    <LogOut aria-hidden="true" />
                    Sign out
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </div>
      </header>
      <main className="w-full px-4 pt-8 pb-16">
        <Routes>
          <Route path="/train" element={<TrainPage />} />
          <Route path="/battle" element={<TrainPage key="battle" battle />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/inspect" element={<InspectPage />} />
          <Route path="/account" element={<AccountPage onUser={setMe} />} />
          <Route path="/feed" element={<FeedPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/u/:username" element={<ProfilePage me={me?.username} />} />
          <Route path="*" element={<Navigate to="/train" replace />} />
        </Routes>
      </main>
    </div>
  )
}
