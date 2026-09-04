import { useEffect, useRef } from "react"
import { AnimatePresence, motion } from "framer-motion"
import { X } from "lucide-react"

export default function Drawer({ open, onClose, title, subtitle, children, footer }) {
  const closeRef = useRef(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === "Escape" && onClose()
    document.addEventListener("keydown", onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = "hidden"
    closeRef.current?.focus()
    return () => {
      document.removeEventListener("keydown", onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40"
            style={{ background: "rgba(8,8,8,0.42)" }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            aria-label={title}
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[540px] flex-col"
            style={{
              background: "var(--surface-1)",
              borderLeft: "1px solid var(--border-strong)",
              boxShadow: "var(--shadow-pop)",
            }}
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 420, damping: 38 }}
          >
            <header
              className="flex items-start justify-between gap-3 px-4 py-3"
              style={{ borderBottom: "1px solid var(--border)" }}
            >
              <div className="min-w-0">
                <h2 className="truncate text-[14px] font-semibold tracking-[-0.005em]">{title}</h2>
                {subtitle && (
                  <p className="mt-0.5 text-[12px] text-[var(--text-secondary)]">{subtitle}</p>
                )}
              </div>
              <button
                ref={closeRef}
                onClick={onClose}
                aria-label="Close"
                className="grid size-7 shrink-0 place-items-center rounded-lg transition-colors hover:bg-[var(--surface-2)]"
                style={{ color: "var(--text-secondary)" }}
              >
                <X size={15} strokeWidth={2.2} aria-hidden />
              </button>
            </header>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">{children}</div>

            {footer && (
              <footer className="px-4 py-3" style={{ borderTop: "1px solid var(--border)" }}>
                {footer}
              </footer>
            )}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  )
}
