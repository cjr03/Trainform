import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { ErrorAlert, Spinner } from '@/components/shared'
import { LINK, PostList } from '@/components/social'
import { replacePost, useAction } from '@/lib/utils'
import { api } from '../api'

const PAGE = 20

export default function FeedPage() {
  const [posts, setPosts] = useState(null)
  const [more, setMore] = useState(false)
  const [run, busy, error] = useAction()
  const load = (before) =>
    run(async () => {
      const page = await api.feed(before, PAGE)
      setPosts((p) => [...(before == null ? [] : p), ...page])
      setMore(page.length === PAGE)
    })
  useEffect(() => {
    load()
    // Loads the first page once.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="mx-auto grid w-full max-w-5xl gap-5" data-testid="feed-page">
      <h1 className="m-0 text-center text-[clamp(1.6rem,3vw,2.4rem)]">Feed</h1>
      <ErrorAlert>{error}</ErrorAlert>
      {!posts && !error && <Spinner />}
      {posts?.length === 0 && (
        <p className="m-0 text-center text-muted-foreground">
          No agents here yet.{' '}
          <Link to="/search" className={LINK}>
            Search for Trainformers
          </Link>{' '}
          to follow and their agents show up here.
        </p>
      )}
      {posts?.length > 0 && <PostList posts={posts} onChange={(p) => setPosts(replacePost(p))} />}
      {more && (
        <Button type="button" variant="outline" className="justify-self-center" disabled={busy} onClick={() => load(posts.at(-1).id)}>
          {busy && <Spinner />}Load more
        </Button>
      )}
    </div>
  )
}
