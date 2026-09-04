import {
  Construction,
  Flame,
  HardHat,
  Layers,
  MoveDown,
  ShieldOff,
  Truck,
  Wind,
  Zap,
  CircleHelp,
  Crosshair,
  ClipboardCheck,
} from "lucide-react"

/** One icon per Life-Saving Rule, so a rule is never identified by color alone. */
export const LSR_ICON = {
  "energy-isolation": Zap,
  "hot-work": Flame,
  "confined-space": Wind,
  "line-of-fire": Crosshair,
  "working-at-height": MoveDown,
  "safe-mechanical-lifting": Layers,
  "bypassing-safety-controls": ShieldOff,
  "work-authorisation": ClipboardCheck,
  driving: Truck,
  unmapped: CircleHelp,
}

export const HAZARD_ICON = {
  gravity: MoveDown,
  pressure: Construction,
  electrical: Zap,
  thermal: Flame,
  mechanical: HardHat,
  chemical: Wind,
  motion: Truck,
}

export function LsrIcon({ id, size = 13, ...rest }) {
  const Icon = LSR_ICON[id] ?? CircleHelp
  return <Icon size={size} strokeWidth={2} aria-hidden {...rest} />
}
