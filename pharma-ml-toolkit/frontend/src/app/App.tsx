import { useCallback, useState } from "react";
import { WorkspaceNav } from "@/components/layout/WorkspaceNav";
import { WorkspaceShell } from "@/components/layout/WorkspaceShell";
import { IntroductionPage } from "@/features/introduction/IntroductionPage";
import { DatasetsPage } from "@/features/datasets/DatasetsPage";
import { MLPage } from "@/features/ml/MLPage";
import { DLPage } from "@/features/dl/DLPage";
import { GenerativePage } from "@/features/generative/GenerativePage";
import { DockingPage } from "@/features/docking/DockingPage";
import { PKPDPage } from "@/features/pkpd/PKPDPage";
import { MDPage } from "@/features/md/MDPage";
import { VisualizationPage } from "@/features/visualization/VisualizationPage";
import { StartupGate } from "@/app/StartupGate";
import { INITIAL_WORKSPACE, WORKSPACES, type WorkspaceId } from "@/app/workspaces";

/**
 * The application shell — part11 §10.
 *
 * The whole shell mounts inside the §14.1 StartupGate: no page
 * renders before the backend handshake completes, health is
 * verified, and the event stream opens.
 *
 * One page instance per workspace (state preserved across switches;
 * no duplicate mounts), Introduction selected at startup, the exact
 * §10.1 navigation order. The connected workflow rows render their
 * real API-driven pages; the queued rows keep the honest shell (the
 * campaign ledger owns the queue).
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
    <StartupGate>
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
          ) : entry.id === "datasets" ? (
            <DatasetsPage />
          ) : entry.id === "machine-learning" ? (
            <MLPage />
          ) : entry.id === "deep-learning" ? (
            <DLPage />
          ) : entry.id === "generative-ai" ? (
            <GenerativePage />
          ) : entry.id === "molecular-docking" ? (
            <DockingPage />
          ) : entry.id === "pkpd" ? (
            <PKPDPage />
          ) : entry.id === "molecular-dynamics" ? (
            <MDPage />
          ) : entry.id === "visualization" ? (
            <VisualizationPage />
          ) : (
            <WorkspaceShell entry={entry} />
          )}
        </main>
      </div>
    </StartupGate>
  );
}
