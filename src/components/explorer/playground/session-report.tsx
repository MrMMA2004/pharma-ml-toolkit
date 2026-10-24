"use client";

import { useMemo, useState } from "react";
import { useTheme } from "next-themes";
import { FileDown, Loader2, MonitorSmartphone } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToolkitStore } from "@/lib/store";
import {
  collectSessionReport,
  sessionReportCensus,
  type SessionReportResponse,
} from "@/lib/session-report";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Session report (100x-127) — the web twin of the desktop playground's */
/* File ▸ Save session report (HTML)…                                   */
/*                                                                      */
/* The session is the persisted playground state (last ML result, the  */
/* learning-curve run, the always-live PK/PD regimen, NCA rows, run    */
/* history); the collector turns it into a section payload, the        */
/* /api/report/session bridge re-composes REAL matplotlib figures      */
/* server-side and assembles them through the SAME self-contained      */
/* themed HTMLReport the desktop saves — theme toggle included.        */
/* ------------------------------------------------------------------ */

export function SessionReport() {
  const mlResult = useToolkitStore((s) => s.mlResult);
  const lcResult = useToolkitStore((s) => s.lcResult);
  const mlHistory = useToolkitStore((s) => s.mlHistory);
  const pkParams = useToolkitStore((s) => s.pkParams);
  const pdParams = useToolkitStore((s) => s.pdParams);
  const ncaState = useToolkitStore((s) => s.ncaState);
  const { resolvedTheme } = useTheme();
  const [busy, setBusy] = useState(false);

  const snapshot = useMemo(
    () => ({ mlResult, lcResult, mlHistory, pkParams, pdParams, ncaState }),
    [mlResult, lcResult, mlHistory, pkParams, pdParams, ncaState],
  );
  const census = useMemo(() => sessionReportCensus(snapshot), [snapshot]);

  async function download() {
    setBusy(true);
    try {
      // the desktop twin bakes the APP's theme radios; the web twin
      // bakes the app's resolved palette (light/dark — "auto" stays a
      // reader-side choice via the report's own toggle).
      const payload = collectSessionReport(snapshot);
      payload.theme = resolvedTheme === "dark" ? "dark" : "light";
      const res = await fetch("/api/report/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? String((data as { error: unknown }).error)
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const report = data as SessionReportResponse;
      const blob = new Blob([report.html], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "pharmaml-session-report.html";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast.success(
        `Session report saved — ${report.figures} figure(s), ${report.tables} table(s), ` +
          `opens ${report.theme} (the button in its top-right corner cycles it per reader).`,
        { duration: 7000 },
      );
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "The session report failed.",
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <CardTitle className="flex flex-wrap items-center gap-2 text-base">
          <FileDown className="size-4 text-primary" aria-hidden />
          Session report
          <Badge variant="secondary" className="ml-1 font-normal">
            <MonitorSmartphone className="mr-1 size-3" aria-hidden />
            opens {resolvedTheme === "dark" ? "dark" : "light"}
          </Badge>
        </CardTitle>
        <CardDescription className="text-xs">
          Every chart and table this session still holds — one self-contained
          HTML file (figures embedded, no external assets). The toggle in its
          top-right corner cycles the theme per reader.
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center justify-between gap-3 py-3">
        <p className="text-xs text-muted-foreground" aria-live="polite">
          Ready to ship:{" "}
          <span className="font-medium text-foreground">
            {census.figures} figure{census.figures === 1 ? "" : "s"}
          </span>{" "}
          ·{" "}
          <span className="font-medium text-foreground">
            {census.tables} table{census.tables === 1 ? "" : "s"}
          </span>{" "}
          ·{" "}
          <span className="font-medium text-foreground">
            {census.metrics} metric card{census.metrics === 1 ? "" : "s"}
          </span>
          {mlResult ? " · last ML run included" : " · train a model to add its figures"}
          {lcResult ? " · learning curve included" : ""}
        </p>
        <Button
          onClick={download}
          disabled={busy}
          className="min-h-11 shrink-0"
          aria-label="Download the session report as a self-contained HTML file"
        >
          {busy ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <FileDown className="size-4" aria-hidden />
          )}
          Download report (HTML)
        </Button>
      </CardContent>
    </Card>
  );
}
