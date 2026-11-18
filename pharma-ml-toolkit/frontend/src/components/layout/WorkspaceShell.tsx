import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { WorkspaceEntry } from "@/app/workspaces";

/**
 * The honest shell for a workspace whose workflow row is queued: the
 * route, identity and §11 workflow summary exist; no fake controls,
 * no fabricated data, no disabled buttons pretending at parity. The
 * campaign ledger (docs/MIGRATION_TAURI2.md §3) owns the row queue.
 */
export function WorkspaceShell({ entry }: { entry: WorkspaceEntry }) {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <h1 className="text-2xl font-bold tracking-tight text-[var(--color-foreground)]">
        {entry.label}
      </h1>
      <Card className="mt-6 max-w-3xl">
        <CardHeader>
          <CardTitle className="text-base">The sequential workflow</CardTitle>
          <CardDescription>
            The §11 workflow this workspace will implement, connected to
            the real backend API in its campaign row:
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-[var(--color-muted-foreground)]">
            {entry.summary}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
