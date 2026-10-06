import { useEffect, useState } from 'react'
import { Navigate, NavLink, Route, Routes, useNavigate } from 'react-router-dom'
import { Gem, Hammer, LogOut, Map as MapIcon, Search, Swords, UserRound } from 'lucide-react'
import { api, clearToken, getToken, setToken } from './api'
import KnightHelmet from '@/components/KnightHelmet'
import { Avatar, Hint } from '@/components/shared'
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
  ['Agents', KnightHelmet],
  ['Inspect', Gem],
  ['Train', Hammer],
  ['Battle', Swords],
]
// Icon-only tabs: a full-height cell of the phone's bottom bar (a 48px touch target), a 36px button from sm up.
const NAV_LINK =
  'inline-flex h-12 flex-1 items-center justify-center rounded-md text-muted-foreground transition-colors outline-none hover:bg-accent hover:text-foreground focus-visible:ring-[3px] focus-visible:ring-ring/50 sm:size-9 sm:flex-none aria-[current=page]:bg-primary/15 aria-[current=page]:text-primary'
const ACTION = 'text-foreground sm:size-9 aria-[current=page]:text-primary'

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
      {/* Tabs left, logo centred, actions right: one row from sm up. On phones the tabs move to a bottom tab bar, in
          thumb reach, and the logo goes left (centred, it would collide with the actions). The header blurs only from
          sm up: a backdrop filter would make it the fixed bar's containing block. */}
      <header className="sticky top-0 z-10 border-b bg-background sm:bg-background/80 sm:backdrop-blur-lg">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-x-2 px-3 py-2 sm:gap-x-4 sm:px-4">
          <nav
            aria-label="Main"
            className="fixed inset-x-0 bottom-0 z-10 flex border-t bg-background px-2 pb-[env(safe-area-inset-bottom)] sm:static sm:col-start-1 sm:row-start-1 sm:gap-1 sm:border-0 sm:bg-transparent sm:p-0"
          >
            {NAV.map(([name, Icon]) => (
              <Hint key={name} label={name}>
                <NavLink to={`/${name.toLowerCase()}`} aria-label={name} className={NAV_LINK}>
                  <Icon aria-hidden="true" className="size-5" />
                </NavLink>
              </Hint>
            ))}
          </nav>
          <span className="col-start-1 row-start-1 justify-self-start font-brand text-sm font-bold tracking-[0.2em] text-gradient sm:col-start-2 sm:justify-self-auto sm:text-base sm:tracking-[0.3em]">
            TRAINFORM
          </span>
          <div className="col-start-3 row-start-1 flex items-center gap-0.5 justify-self-end sm:gap-1">
            <PublishDialog username={me?.username} />
            <Hint label="Feed">
              <Button asChild variant="ghost" size="icon-sm" className={ACTION}>
                <NavLink to="/feed" aria-label="Feed">
                  <MapIcon aria-hidden="true" className="size-5" />
                </NavLink>
              </Button>
            </Hint>
            <Hint label="Search Trainformers">
              <Button asChild variant="ghost" size="icon-sm" className={ACTION}>
                <NavLink to="/search" aria-label="Search Trainformers">
                  <Search aria-hidden="true" className="size-5" />
                </NavLink>
              </Button>
            </Hint>
            <DropdownMenu>
              <Hint label="Account">
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon-sm" className={ACTION} aria-label="Account menu">
                    <Avatar url={me?.avatar_url} />
                  </Button>
                </DropdownMenuTrigger>
              </Hint>
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
      </header>
      <main className="w-full px-4 pt-8 pb-28 sm:pb-16">
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
