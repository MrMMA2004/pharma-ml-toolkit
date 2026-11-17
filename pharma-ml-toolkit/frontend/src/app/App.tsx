import { useCallback, useState } from "react";
import { WorkspaceNav } from "@/components/layout/WorkspaceNav";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";
import { IntroductionPage } from "@/features/introduction/IntroductionPage";
import { INITIAL_WORKSPACE, WORKSPACES, type WorkspaceId } from "@/app/workspaces";

/**
 * The application shell — part11 §10.
 *
 * One page instance per workspace (state preserved across switches;
 * no duplicate mounts), Introduction selected at startup, the exact
 * §10.1 navigation order. The queued workspace rows render the honest
 * shell; the workflow rows connect in their campaign rounds.
 */
export function App() {
  const [active, setActive] = useState<WorkspaceId>(INITIAL_WORKSPACE);

  const handleSelect = useCallback((id: WorkspaceId) => {
    setActive(id);
  }, []);

  const entry = WORKSPACES.find((w) => w.id === active);
  if (!entry) {
    // unreachable while WORKSPACES covers every WorkspaceId — the
    // exhaustive registry is drift-guarded by the pytest contract test
    throw new Error(`unknown workspace: ${active}`);
  }

  return (
    <div className="flex min-h-screen flex-col bg-[var(--color-background)]">
      <header className="shrink-0 border-b border-[var(--color-border)] bg-[var(--color-card)]">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--color-primary)] text-[var(--color-primary-foreground)]">
            <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
              <path
                fill="currentColor"
                d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6V3z"
                opacity="0.9"
              />
            </svg>
          </div>
          <div>
            <div className="text-sm font-semibold leading-tight text-[var(--color-foreground)]">
              PharmaML Studio
            </div>
            <div className="text-[11px] leading-tight text-[var(--color-muted-foreground)]">
              Offline computational pharmacy workbench
            </div>
          </div>
        </div>
      </header>
      <WorkspaceNav active={active} onSelect={handleSelect} />
      <main className="flex-1">
        {entry.id === "introduction" ? (
          <IntroductionPage />
        ) : (
          <WorkspaceShell entry={entry} />
        )}
      </main>
    </div>
  );
}
