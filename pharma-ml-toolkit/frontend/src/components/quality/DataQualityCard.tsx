/**
 * part16 §15.4 — the Data Quality card (the Phase-12 frontend lane).
 *
 * The §15.4 "Data Quality and Scientific Validation" surface over one
 * registered dataset, mounted in the Datasets workspace beside the
 * profile lane: the "Run data-quality scan" button submits to
 * `POST /api/v1/datasets/{id}/quality`, then the card renders the
 * severity summary (errors / warnings / info), the findings list
 * (the severity badge, the pinned code, the message, the affected
 * columns) and the executed-check list (the §15.4 vocabulary with
 * the honest executed/skipped statuses — the chemistry pair skipped
 * when no SMILES column is detected).
 *
 * Advisory by construction: the card renders findings, it never
 * offers a fix/repair/apply control — §15.4's "never silently alter
 * user-provided scientific data" is the contract, and the advisory
 * line rides every report. Nothing is fabricated: absent evidence
 * renders as the honest dash, the pre-scan state says so, and a
 * failed request surfaces as the error banner.
 */

import { useCallback, useState } from "react";
import { LoaderCircle, ShieldCheck, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { qualityApi } from "@/services/quality/client";
import {
  QUALITY_SEVERITIES,
  type DataQualityReport,
  type QualityFinding,
  type QualitySeverity,
} from "@/services/quality/types";

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** The severity badge vocabulary — one chip per honest finding class. */
const SEVERITY_BADGE: Record<QualitySeverity, string> = {
  error:
    "border-[var(--color-destructive)]/40 bg-[var(--color-destructive)]/10 text-[var(--color-destructive)]",
  warning:
    "border-yellow-600/40 bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
  info:
    "border-[var(--color-border)] bg-[var(--color-muted)] text-[var(--color-muted-foreground)]",
};

function FindingRow({ finding }: { finding: QualityFinding }) {
  return (
    <li
      className="rounded-md border border-[var(--color-border)] p-2"
      aria-label={`${finding.severity}: ${finding.code}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${SEVERITY_BADGE[finding.severity]}`}
        >
          {finding.severity}
        </span>
        <span className="font-mono text-[11px] text-[var(--color-muted-foreground)]">
          {finding.code}
        </span>
        <span className="text-[10px] uppercase tracking-wide text-[var(--color-muted-foreground)]">
          {finding.category}
        </span>
      </div>
      <p className="mt-1 text-sm text-[var(--color-foreground)]">
        {finding.message}
      </p>
      {finding.columns.length > 0 && (
        <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
          columns:{" "}
          <span className="font-mono text-[var(--color-foreground)]">
            {finding.columns.join(", ")}
          </span>
        </p>
      )}
    </li>
  );
}

/**
 * The §15.4 data-quality card. Self-contained by design: it loads its
 * own data through `qualityApi`, no global state, no context — the
 * host stage only positions it and passes the selected dataset.
 */
export function DataQualityCard({
  datasetId,
  target,
}: {
  datasetId: string | null;
  target?: string | null;
}) {
  const [report, setReport] = useState<DataQualityReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runScan = useCallback(async () => {
    if (datasetId === null || busy) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const scanned = await qualityApi.run(datasetId, {
        target: target ?? undefined,
      });
      setReport(scanned);
    } catch (requestError) {
      setError(describeError(requestError));
    } finally {
      setBusy(false);
    }
  }, [busy, datasetId, target]);

  return (
    <Card id="data-quality-card">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <ShieldCheck aria-hidden className="size-4" />
          Data Quality — Scientific Validation
        </CardTitle>
        <CardDescription>
          The §15.4 read-only scan: SMILES/structure validation, duplicate
          detection, the leakage family, class imbalance, the pinned unit
          ranges.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center gap-2">
          <Button
            onClick={runScan}
            disabled={datasetId === null || busy}
            aria-label="Run the data-quality scan"
          >
            {busy ? (
              <>
                <LoaderCircle aria-hidden className="size-4 animate-spin" />
                Scanning…
              </>
            ) : (
              <>
                <ShieldAlert aria-hidden className="size-4" />
                Run data-quality scan
              </>
            )}
          </Button>
          {datasetId === null && (
            <span className="text-xs text-[var(--color-muted-foreground)]">
              Select a dataset first.
            </span>
          )}
        </div>

        {error !== null && (
          <div
            role="alert"
            className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm"
          >
            <span className="font-medium">Scan failed:</span>{" "}
            <span className="text-[var(--color-muted-foreground)]">{error}</span>
          </div>
        )}

        {report !== null && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-2 py-0.5 text-[var(--color-muted-foreground)]">
                {report.rows} rows × {report.columns} columns
              </span>
              {report.target !== null && (
                <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-2 py-0.5 text-[var(--color-muted-foreground)]">
                  target: <span className="font-mono">{report.target}</span>
                </span>
              )}
              {report.smiles_column !== null && (
                <span className="rounded-full border border-[var(--color-border)] bg-[var(--color-muted)] px-2 py-0.5 text-[var(--color-muted-foreground)]">
                  SMILES:{" "}
                  <span className="font-mono">{report.smiles_column}</span>
                </span>
              )}
              {QUALITY_SEVERITIES.map((severity) => (
                <span
                  key={severity}
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${SEVERITY_BADGE[severity]}`}
                >
                  {report.counts[severity]} {severity}
                  {report.counts[severity] === 1 ? "" : "s"}
                </span>
              ))}
            </div>

            {report.findings.length === 0 ? (
              <p
                role="status"
                className="rounded-md border border-[var(--color-border)] bg-[var(--color-muted)] p-3 text-sm text-[var(--color-muted-foreground)]"
              >
                No issues detected — every executed §15.4 check returned clean.
              </p>
            ) : (
              <ul className="space-y-2">
                {report.findings.map((finding, index) => (
                  <FindingRow
                    key={`${finding.code}-${index}`}
                    finding={finding}
                  />
                ))}
              </ul>
            )}

            <details className="rounded-md border border-[var(--color-border)] p-2">
              <summary className="cursor-pointer text-xs text-[var(--color-muted-foreground)]">
                The executed §15.4 checks ({report.checks.length})
              </summary>
              <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto text-xs">
                {report.checks.map((entry) => (
                  <li
                    key={entry.check}
                    className="flex flex-wrap items-baseline gap-2"
                  >
                    <span
                      className={
                        entry.status === "executed"
                          ? "text-[var(--color-primary)]"
                          : "text-[var(--color-muted-foreground)]"
                      }
                    >
                      {entry.status === "executed" ? "✓" : "—"}
                    </span>
                    <span className="text-[var(--color-foreground)]">
                      {entry.label}
                    </span>
                    <span className="text-[var(--color-muted-foreground)]">
                      {entry.detail}
                    </span>
                  </li>
                ))}
              </ul>
            </details>

            <p className="text-[11px] text-[var(--color-muted-foreground)]">
              {report.advisory}
            </p>
          </>
        )}

        {report === null && error === null && (
          <p className="text-xs text-[var(--color-muted-foreground)]">
            Not scanned yet — run the scan to see the §15.4 report.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
