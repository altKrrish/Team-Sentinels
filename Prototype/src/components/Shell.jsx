import {
  FileText,
  LayoutDashboard,
  ListChecks,
  Moon,
  Radar,
  ShieldAlert,
  Sun,
} from "lucide-react"
import { Badge } from "./kit/index.jsx"

export const SCREENS = [
  { key: "dashboard", label: "Dashboard", Icon: LayoutDashboard },
  { key: "triage", label: "Triage", Icon: ListChecks },
  { key: "reports", label: "Reports", Icon: FileText },
  { key: "patterns", label: "Patterns", Icon: Radar },
]

function SourceBadge({ source, error }) {
  if (source === "live") return <Badge tone="good">Live model</Badge>
  if (source === "fallback")
    return (
      <Badge tone="critical" title={error}>
        Model unreachable · demo data
      </Badge>
    )
  return <Badge tone="warning">Demo data</Badge>
}

export default function Shell({ screen, onScreen, source, error, theme, onTheme, children }) {
  return (
    <div className="flex h-full min-h-0">
      {/* sidebar */}
      <aside
        className="hidden w-[212px] shrink-0 flex-col lg:flex"
        style={{ borderRight: "1px solid var(--border)", background: "var(--surface-1)" }}
      >
        <div className="flex items-center gap-2 px-4 py-4">
          <ShieldAlert size={18} strokeWidth={2.2} style={{ color: "var(--status-critical)" }} />
          <div className="min-w-0">
            <div className="text-[14px] leading-tight font-semibold tracking-[-0.01em]">
              CloseCall
            </div>
            <div className="text-[10.5px] leading-tight text-[var(--text-muted)]">
              Oil India Limited · HSSE
            </div>
          </div>
        </div>

        <nav className="flex flex-col gap-0.5 px-2 py-2">
          {SCREENS.map(({ key, label, Icon }) => {
            const active = screen === key
            return (
              <button
                key={key}
                onClick={() => onScreen(key)}
                aria-current={active ? "page" : undefined}
                className="flex h-9 items-center gap-2.5 rounded-lg px-2.5 text-[13px] font-medium transition-colors"
                style={{
                  background: active ? "var(--surface-2)" : "transparent",
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                <Icon size={15} strokeWidth={2} aria-hidden />
                {label}
              </button>
            )
          })}
        </nav>

        <div className="mt-auto px-4 py-4">
          <p className="text-[11px] leading-relaxed text-[var(--text-muted)]">
            Flags reports carrying genuine fatal potential, separately from severity triage.
            Benchmark: 20–25% of reports.
          </p>
        </div>
      </aside>

      {/* main column */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <header
          className="flex shrink-0 items-center gap-3 px-4 py-3"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--surface-1)" }}
        >
          <div className="flex min-w-0 items-center gap-2 lg:hidden">
            <ShieldAlert size={17} strokeWidth={2.2} style={{ color: "var(--status-critical)" }} />
            <span className="text-[14px] font-semibold">CloseCall</span>
          </div>

          <div className="hidden min-w-0 lg:block">
            <h1 className="truncate text-[14px] font-semibold tracking-[-0.01em]">
              {SCREENS.find((s) => s.key === screen)?.label}
            </h1>
            <p className="text-[11.5px] text-[var(--text-muted)]">
              SIF precursor detection across unsafe acts, unsafe conditions and near-misses
            </p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <SourceBadge source={source} error={error} />
            <button
              onClick={() => onTheme(theme === "dark" ? "light" : "dark")}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              title={theme === "dark" ? "Light theme" : "Dark theme"}
              className="grid size-8 place-items-center rounded-lg transition-colors hover:bg-[var(--surface-2)]"
              style={{ border: "1px solid var(--border-strong)", color: "var(--text-secondary)" }}
            >
              {theme === "dark" ? (
                <Sun size={14.5} strokeWidth={2} aria-hidden />
              ) : (
                <Moon size={14.5} strokeWidth={2} aria-hidden />
              )}
            </button>
          </div>
        </header>

        {/* mobile nav */}
        <nav
          className="flex shrink-0 gap-1 overflow-x-auto px-2 py-1.5 lg:hidden"
          style={{ borderBottom: "1px solid var(--border)", background: "var(--surface-1)" }}
        >
          {SCREENS.map(({ key, label, Icon }) => {
            const active = screen === key
            return (
              <button
                key={key}
                onClick={() => onScreen(key)}
                aria-current={active ? "page" : undefined}
                className="flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-[12.5px] font-medium transition-colors"
                style={{
                  background: active ? "var(--surface-2)" : "transparent",
                  color: active ? "var(--text-primary)" : "var(--text-secondary)",
                }}
              >
                <Icon size={13.5} strokeWidth={2} aria-hidden />
                {label}
              </button>
            )
          })}
        </nav>

        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </div>
    </div>
  )
}
