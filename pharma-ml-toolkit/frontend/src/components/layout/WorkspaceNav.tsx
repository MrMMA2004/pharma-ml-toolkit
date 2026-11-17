import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import { WORKSPACES, type WorkspaceId } from "@/app/workspaces";

/**
 * The horizontal workspace navigation — part11 §10.1.
 *
 * Requirements implemented here: exactly nine primary entries in the
 * exact order, Introduction selected at startup, a visually clear
 * active state, mouse AND keyboard operation (the buttons are real
 * <button>s in a `<nav role=tablist>`-free landmark — arrow-key
 * roving follows in the interaction-polish row), clean small-window
 * behavior (the strip scrolls horizontally rather than wrapping or
 * dropping entries), and no duplicate page instances (the parent
 * mounts one page per workspace and preserves state).
 */
export function WorkspaceNav({
  active,
  onSelect,
}: {
  active: WorkspaceId;
  onSelect: (id: WorkspaceId) => void;
}) {
  return (
    <nav
      aria-label="Workspaces"
      className="shrink-0 border-b border-[var(--color-border)] bg-[var(--color-card)]"
    >
      <div className="mx-auto flex max-w-7xl items-center gap-1 overflow-x-auto px-4 py-2">
        {WORKSPACES.map((workspace) => {
          const isActive = workspace.id === active;
          return (
            <button
              key={workspace.id}
              type="button"
              onClick={() => onSelect(workspace.id)}
              aria-current={isActive ? "page" : undefined}
              className={cn(
                "relative shrink-0 rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--color-ring)]",
                isActive
                  ? "text-[var(--color-primary)]"
                  : "text-[var(--color-muted-foreground)] hover:bg-[var(--color-accent)] hover:text-[var(--color-accent-foreground)]",
              )}
            >
              {workspace.label}
              {isActive ? (
                <motion.span
                  layoutId="workspace-nav-underline"
                  className="absolute inset-x-2 -bottom-[9px] h-0.5 rounded-full bg-[var(--color-primary)]"
                  transition={{ type: "spring", stiffness: 500, damping: 40 }}
                />
              ) : null}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
