import { CircleAlert, Loader2 } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { cn } from '@/lib/utils'

export const Spinner = () => <Loader2 className="animate-spin" />

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
          Unranked
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
