import { FlaskConical, HardDrive, WifiOff, Workflow } from "lucide-react";
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { WORKSPACES } from "@/app/workspaces";

/**
 * The Introduction page — part11 §10.2.
 *
 * Professional and informative; explains what PharmaML Studio is, its
 * computational pharmaceutical purpose, the supported scientific
 * areas, the general workflow, the offline nature and the local
 * backend architecture. It must NOT become a second dashboard of nine
 * large duplicate workspace cards — the workspace list below is a
 * concise directory (label + one line), not a card grid.
 */
export function IntroductionPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <div className="max-w-3xl">
        <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-3 py-1 text-xs font-medium text-[var(--color-muted-foreground)]">
          <WifiOff aria-hidden className="h-3.5 w-3.5" />
          Fully offline — your data never leaves this machine
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-[var(--color-foreground)]">
          PharmaML Studio
        </h1>
        <p className="mt-4 text-lg leading-relaxed text-[var(--color-muted-foreground)]">
          An offline desktop workbench for computational pharmacy: QSAR
          machine learning over molecular descriptors and fingerprints,
          deep learning, semi-supervised and generative molecular design,
          docking, PK/PD, molecular-dynamics analysis, and a
          visualization workspace — over one Python backend, driven
          through a typed local REST + WebSocket API.
        </p>
      </div>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <FlaskConical aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">The scientific backend stays Python</CardTitle>
            <CardDescription>
              Chemistry, ML, DL, generative, docking, PK/PD, MD analysis
              and applicability-domain logic run in the packaged Python
              backend — the UI never reimplements science.
            </CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <Workflow aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">Sequential workflows, real jobs</CardTitle>
            <CardDescription>
              Every workspace follows its end-to-end workflow; long
              computations run as managed background jobs with live
              progress over WebSocket and honest cancellation.
            </CardDescription>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <HardDrive aria-hidden className="h-5 w-5 text-[var(--color-primary)]" />
            <CardTitle className="text-base">Local, durable, yours</CardTitle>
            <CardDescription>
              Datasets, models, artifacts and job history persist in a
              local SQLite database plus a backend-managed file vault —
              with backup and restore built in.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>

      <div className="mt-12">
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">
          The nine workspaces
        </h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--color-muted-foreground)]">
          Navigation runs left-to-right in a fixed order — the workflow
          from data to insight. Each workspace exposes its complete
          backend configuration; work continues in later campaign rows.
        </p>
        <ol className="mt-6 divide-y divide-[var(--color-border)] overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-card)]">
          {WORKSPACES.map((workspace, index) => (
            <li key={workspace.id} className="flex gap-4 px-5 py-4">
              <span className="mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
                {index + 1}
              </span>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-medium text-[var(--color-foreground)]">
                    {workspace.label}
                  </span>
                  {workspace.state === "shell" ? (
                    <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--color-muted-foreground)]">
                      workflow row queued
                    </span>
                  ) : null}
                </div>
                <p className="mt-1 text-sm leading-relaxed text-[var(--color-muted-foreground)]">
                  {workspace.summary}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
