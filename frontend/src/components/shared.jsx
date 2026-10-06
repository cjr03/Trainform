import { CircleAlert, CircleUser, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Tooltip } from 'radix-ui'
import KnightHelmet from '@/components/KnightHelmet'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { cn } from '@/lib/utils'
import { api } from '../api'

export const Spinner = () => <Loader2 className="animate-spin" />

// The text an icon-only control shows on hover or keyboard focus; the control carries the same name as aria-label.
export function Hint({ label, children }) {
  return (
    <Tooltip.Provider delayDuration={150}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            sideOffset={6}
            className="z-50 rounded-md bg-foreground px-2 py-1 text-xs font-medium text-background shadow-md"
          >
            {label}
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  )
}

export const ErrorAlert = ({ children }) =>
  children ? (
    <Alert variant="destructive">
      <CircleAlert />
      <AlertDescription className="text-destructive">{children}</AlertDescription>
    </Alert>
  ) : null

// Pill-shaped choices: one radio group with a visible radio in each pill.
export function Choice({ options, ...props }) {
  return (
    <RadioGroup className="flex flex-wrap gap-2" {...props}>
      {options.map(([key, text, props]) => (
        <label
          key={key}
          className="flex cursor-pointer items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-colors has-disabled:cursor-not-allowed has-disabled:opacity-50 has-data-[state=checked]:border-primary has-data-[state=checked]:bg-primary/10 has-data-[state=checked]:text-primary"
        >
          <RadioGroupItem value={key} {...props} />
          {text}
        </label>
      ))}
    </RadioGroup>
  )
}

// An agent's overall rating, Madden style: its best training score × 10, or Unranked before any.
export function Overall({ value, large, className }) {
  return (
    <span
      role="img"
      aria-label={value == null ? 'Unranked' : `Overall ${value}`}
      className={cn(
        'grid shrink-0 place-content-center justify-items-center rounded-xl border border-border-strong bg-brand/8 leading-none shadow-glow',
        large ? 'size-24' : 'size-14',
        className,
      )}
    >
      {value == null ? (
        <span className={cn('font-semibold text-muted-foreground uppercase', large ? 'text-xs tracking-wider' : 'text-[0.6rem]')}>
          --
        </span>
      ) : (
        <>
          <span className={cn('font-display font-bold tabular-nums text-gradient', large ? 'text-5xl' : 'text-2xl')}>
            {value}
          </span>
          <span className="mt-1 text-[0.6rem] font-semibold tracking-[0.2em] text-muted-foreground">OVR</span>
        </>
      )}
    </span>
  )
}

// An agent's SVG image, only ever as an <img> data URL so no script in it can run. The knight helmet stands in without one.
export function AgentImage({ svg, alt = '', className }) {
  const box = cn('size-12 shrink-0 rounded-lg', className)
  return svg ? (
    <img src={`data:image/svg+xml;utf8,${encodeURIComponent(svg)}`} alt={alt} className={cn(box, 'bg-background/40')} />
  ) : (
    <KnightHelmet className={cn(box, 'border bg-brand/8 p-1.5 text-brand')} />
  )
}

// A Trainformer's profile picture: `url` is an API path fetched with the token, so it shows from an object URL.
export function Avatar({ url, alt = '', className }) {
  const [shown, setShown] = useState({})
  useEffect(() => {
    if (!url) return
    let live = true
    let src
    api.avatar(url).then(
      (blob) => live && setShown({ url, src: (src = URL.createObjectURL(blob)) }),
      () => {}, // the icon stays
    )
    return () => {
      live = false
      if (src) URL.revokeObjectURL(src)
    }
  }, [url])
  const box = cn('size-6 shrink-0 rounded-full', className)
  return url && shown.url === url ? (
    <img src={shown.src} alt={alt} className={cn(box, 'object-cover')} />
  ) : (
    <CircleUser aria-hidden="true" className={box} />
  )
}
