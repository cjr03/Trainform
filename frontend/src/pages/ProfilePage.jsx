import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { Avatar, ErrorAlert, Spinner } from '@/components/shared'
import { FollowButton, PostList } from '@/components/social'
import { replacePost } from '@/lib/utils'
import { api } from '../api'

// A Trainformer's page: follower counts, a follow button and their published agents. `me` is your username.
export default function ProfilePage({ me }) {
  const { username } = useParams()
  const [state, setState] = useState({}) // {username, profile | error}, so another profile's never shows

  useEffect(() => {
    let live = true
    api.getProfile(username).then(
      (profile) => live && setState({ username, profile }),
      (e) => live && setState({ username, error: e.message }),
    )
    return () => {
      live = false
    }
  }, [username])

  if (state.username !== username) return <Spinner />
  if (state.error) return <ErrorAlert>{state.error}</ErrorAlert>
  const { profile } = state
  const mine = username === me
  const setProfile = (p) => setState({ username, profile: p })
  const setPosts = (update) => setState((s) => ({ ...s, profile: { ...s.profile, posts: update(s.profile.posts) } }))

  return (
    <div className="mx-auto grid w-full max-w-5xl gap-5" data-testid="profile-page">
      <div className="flex flex-wrap items-center gap-4">
        <Avatar url={profile.avatar_url} alt={`${username}'s profile picture`} className="size-20" />
        <div className="mr-auto grid min-w-0 gap-1">
          <h1 className="m-0 text-[clamp(1.6rem,3vw,2.4rem)] wrap-anywhere">{username}</h1>
          <span className="text-sm text-muted-foreground tabular-nums">
            {profile.followers} {profile.followers === 1 ? 'follower' : 'followers'} · {profile.following_count} following
          </span>
        </div>
        {!mine && <FollowButton user={profile} onChange={setProfile} />}
      </div>
      {profile.posts.length ? (
        <PostList
          posts={profile.posts}
          mine={mine}
          onChange={(p) => setPosts(replacePost(p))}
          onDelete={(p) => setPosts((posts) => posts.filter((x) => x.id !== p.id))}
        />
      ) : (
        <p className="m-0 text-center text-muted-foreground">No published agents yet.</p>
      )}
    </div>
  )
}
