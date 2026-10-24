"use client";

import { useEffect, useMemo, useState } from "react";
import { useTheme } from "next-themes";
import {
  Activity,
  Check,
  Dna,
  FileDown,
  History,
  Loader2,
  MonitorSmartphone,
  Table2,
  TrendingUp,
  Waves,
  X,
} from "lucide-react";
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
  sessionGroupsAvailable,
  sessionReportCensus,
  type SessionReportOptions,
  type SessionReportResponse,
} from "@/lib/session-report";
import { toast } from "sonner";

/* ------------------------------------------------------------------ */
/* Session report (100x-127) — the web twin of the desktop playground's */
/* File ▸ Save session report (HTML)…                                   */
/*                                                                      */
/* The session is the persisted playground state (last ML result, the   */
/* learning-curve run, the always-live PK/PD regimen, NCA rows, run     */
/* history — and, since 100x-131, the last molecule-surgery run); the   */
/* collector turns it into a section payload, the                      */
/* /api/report/session bridge re-composes REAL matplotlib figures      */
/* server-side and assembles them through the SAME self-contained      */
/* themed HTMLReport the desktop saves — theme toggle included.        */
/*                                                                      */
/* (100x-131) Per-section control: every group the session actually    */
/* has content for gets a toggle chip — skip the PK figures when only   */
/* the ML run matters, ship only the surgery grid, whatever the         */
/* session needs.  The census line and the download both walk the       */
/* SAME filtered collector, so "what will ship" is always the truth.    */
/* The toggle set persists in localStorage (tolerant restore).          */
/* ------------------------------------------------------------------ */

/** localStorage key for the per-group toggle set (100x-131). */
const SECTION_TOGGLE_STORAGE_KEY = "pharmaml.reportSections";

type GroupId = keyof SessionReportOptions;

/** One toggle chip's presentation — label + icon + what the group adds. */
const GROUPS: Array<{
  id: GroupId;
  label: string;
  icon: typeof Activity;
  hint: string;
}> = [
  {
    id: "ml",
    label: "ML training",
    icon: TrendingUp,
    hint: "the training metrics card + the parity / decision figure",
  },
  {
    id: "learningCurve",
    label: "Learning curve",
    icon: Activity,
    hint: "the curve's metrics + the mean ± std band figure",
  },
  {
    id: "pkpd",
    label: "PK / PD",
    icon: Waves,
    hint: "the regimen concentration profile + the PD effect overlay",
  },
  {
    id: "nca",
    label: "NCA data",
    icon: Table2,
    hint: "the observed-data rows as a real text table",
  },
  {
    id: "history",
    label: "Run history",
    icon: History,
    hint: "the training-run table (when, task, model, metric)",
  },
  {
    id: "surgery",
    label: "Surgery",
    icon: Dna,
    hint: "the surgery run's metrics + SMILES table + structure grid",
  },
];

/** Read the persisted toggle set (client only, failure-tolerant):
 * anything absent from the stored object defaults to shipped, and a
 * corrupt payload resets to "everything on" — never a broken report. */
function loadSectionToggles(): SessionReportOptions {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(SECTION_TOGGLE_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return {};
    const stored = parsed as Record<string, unknown>;
    const options: SessionReportOptions = {};
    for (const group of GROUPS) {
      const value = stored[group.id];
      if (typeof value === "boolean" && value === false) options[group.id] = false;
    }
    return options;
  } catch {
    return {};
  }
}

/** Persist the toggle set (client only, best-effort). */
function saveSectionToggles(options: SessionReportOptions): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      SECTION_TOGGLE_STORAGE_KEY,
      JSON.stringify(options),
    );
  } catch {
    /* a blocked localStorage just loses the preference */
  }
}

export function SessionReport() {
  const mlResult = useToolkitStore((s) => s.mlResult);
  const lcResult = useToolkitStore((s) => s.lcResult);
  const mlHistory = useToolkitStore((s) => s.mlHistory);
  const pkParams = useToolkitStore((s) => s.pkParams);
  const pdParams = useToolkitStore((s) => s.pdParams);
  const ncaState = useToolkitStore((s) => s.ncaState);
  const surgeryResult = useToolkitStore((s) => s.surgeryResult);
  const { resolvedTheme } = useTheme();
  const [busy, setBusy] = useState(false);
  const [options, setOptions] = useState<SessionReportOptions>({});

  // restore once on mount (the tolerant read — defaults ship everything)
  useEffect(() => {
    setOptions(loadSectionToggles());
  }, []);

  const snapshot = useMemo(
    () => ({
      mlResult,
      lcResult,
      mlHistory,
      pkParams,
      pdParams,
      ncaState,
      surgery: surgeryResult,
    }),
    [mlResult, lcResult, mlHistory, pkParams, pdParams, ncaState, surgeryResult],
  );
  const available = useMemo(() => sessionGroupsAvailable(snapshot), [snapshot]);
  const census = useMemo(
    () => sessionReportCensus(snapshot, options),
    [snapshot, options],
  );
  const shippable =
    census.figures + census.tables + census.metrics + census.images > 0;

  function toggleGroup(id: GroupId) {
    setOptions((prev) => {
      const next = { ...prev, [id]: !(prev[id] !== false) };
      // never persist "true" entries — absence means shipped, so the
      // stored object stays minimal and future groups default on
      const clean: SessionReportOptions = {};
      for (const group of GROUPS) {
        if (next[group.id] === false) clean[group.id] = false;
      }
      saveSectionToggles(clean);
      return clean;
    });
  }

  async function download() {
    setBusy(true);
    try {
      // the desktop twin bakes the APP's theme radios; the web twin
      // bakes the app's resolved palette (light/dark — "auto" stays a
      // reader-side choice via the report's own toggle).
      const payload = collectSessionReport(snapshot, options);
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
      const bits = [`${report.figures} figure(s)`, `${report.tables} table(s)`];
      if (census.images > 0) bits.push(`${census.images} structure grid(s)`);
      toast.success(
        `Session report saved — ${bits.join(", ")}; ` +
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
      <CardContent className="space-y-3 py-3">
        {/* (100x-131) per-section control — one chip per group the
            session actually has content for; a group with nothing to
            ship renders no chip (there is nothing to skip). */}
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-foreground">Include in report</p>
          <div
            className="flex flex-wrap gap-1.5"
            role="group"
            aria-label="Report section groups"
          >
            {GROUPS.filter((group) => available[group.id]).map((group) => {
              const on = options[group.id] !== false;
              const Icon = group.icon;
              return (
                <button
                  key={group.id}
                  type="button"
                  aria-pressed={on}
                  onClick={() => toggleGroup(group.id)}
                  title={
                    on
                      ? `Included: ${group.hint} — click to skip it`
                      : `Skipped: ${group.hint} — click to include it`
                  }
                  className={`inline-flex min-h-9 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    on
                      ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/15"
                      : "border-dashed border-muted-foreground/30 bg-muted/30 text-muted-foreground hover:border-muted-foreground/50 hover:text-foreground"
                  }`}
                >
                  {on ? (
                    <Check className="size-3.5" aria-hidden />
                  ) : (
                    <X className="size-3.5" aria-hidden />
                  )}
                  <Icon className="size-3.5" aria-hidden />
                  {group.label}
                </button>
              );
            })}
          </div>
          <p className="text-[11px] text-muted-foreground">
            Skipped groups stay skipped on this browser; groups with nothing
            to ship never show a chip.
          </p>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-muted-foreground" aria-live="polite">
            {shippable ? (
              <>
                Ready to ship:{" "}
                <span className="font-medium text-foreground">
                  {census.figures} figure{census.figures === 1 ? "" : "s"}
                </span>
                {census.images > 0 ? (
                  <>
                    {" · "}
                    <span className="font-medium text-foreground">
                      {census.images} structure grid{census.images === 1 ? "" : "s"}
                    </span>
                  </>
                ) : null}
                {" · "}
                <span className="font-medium text-foreground">
                  {census.tables} table{census.tables === 1 ? "" : "s"}
                </span>
                {" · "}
                <span className="font-medium text-foreground">
                  {census.metrics} metric card{census.metrics === 1 ? "" : "s"}
                </span>
                {surgeryResult
                  ? options.surgery === false
                    ? " · surgery run skipped"
                    : " · surgery grid included"
                  : ""}
              </>
            ) : (
              "Nothing would ship — enable at least one group above."
            )}
          </p>
          <Button
            onClick={download}
            disabled={busy || !shippable}
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
        </div>
      </CardContent>
    </Card>
  );
}
