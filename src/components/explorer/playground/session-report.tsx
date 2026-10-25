"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useTheme } from "next-themes";
import {
  Activity,
  Check,
  ChevronDown,
  ChevronUp,
  Dna,
  FileDown,
  GripVertical,
  History,
  Loader2,
  MonitorSmartphone,
  RotateCcw,
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
  SESSION_GROUP_ORDER,
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
/*                                                                      */
/* (100x-132) The chips REORDER: drag one (or ride its arrows) and the  */
/* report walks its groups in that order — the reader's narrative, not  */
/* the code's.  The order persists per browser (tolerant restore), a    */
/* dashed "Reset walking order" chip appears whenever it drifted from   */
/* the canonical one, and the collector sanitizes whatever it is        */
/* handed (unknown/duplicate ids dropped, missing groups still ship).   */
/* ------------------------------------------------------------------ */

/** localStorage key for the per-group toggle set (100x-131). */
const SECTION_TOGGLE_STORAGE_KEY = "pharmaml.reportSections";
/** localStorage key for the group walking order (100x-132). */
const SECTION_ORDER_STORAGE_KEY = "pharmaml.reportSectionOrder";

type GroupId = (typeof SESSION_GROUP_ORDER)[number];

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

/** (100x-132) Read the persisted walking order (client only,
 * failure-tolerant): only known group ids survive, duplicates collapse
 * (first wins), and anything empty/corrupt falls back to the canonical
 * order — the collector's own sanitizer double-checks whatever it is
 * handed, so a stale or tampered payload can never break a report. */
function loadSectionOrder(): GroupId[] {
  if (typeof window === "undefined") return [...SESSION_GROUP_ORDER];
  try {
    const raw = window.localStorage.getItem(SECTION_ORDER_STORAGE_KEY);
    if (!raw) return [...SESSION_GROUP_ORDER];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [...SESSION_GROUP_ORDER];
    const seen = new Set<GroupId>();
    const order: GroupId[] = [];
    for (const item of parsed) {
      const id = item as GroupId;
      if (SESSION_GROUP_ORDER.includes(id) && !seen.has(id)) {
        seen.add(id);
        order.push(id);
      }
    }
    // a stored order that predates a new group still ships it (last)
    for (const id of SESSION_GROUP_ORDER) {
      if (!seen.has(id)) order.push(id);
    }
    return order;
  } catch {
    return [...SESSION_GROUP_ORDER];
  }
}

/** (100x-132) Persist the walking order — only while it differs from
 * the canonical one (the stored payload stays minimal; equality means
 * the key is REMOVED, so a fresh default never writes storage). */
function saveSectionOrder(order: GroupId[]): void {
  if (typeof window === "undefined") return;
  try {
    if (order.join() === SESSION_GROUP_ORDER.join()) {
      window.localStorage.removeItem(SECTION_ORDER_STORAGE_KEY);
    } else {
      window.localStorage.setItem(
        SECTION_ORDER_STORAGE_KEY,
        JSON.stringify(order),
      );
    }
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
  // (100x-132) the walking order — starts canonical, restored on mount
  const [order, setOrder] = useState<GroupId[]>([...SESSION_GROUP_ORDER]);
  const [dragging, setDragging] = useState<GroupId | null>(null);
  const [dropBefore, setDropBefore] = useState<GroupId | null>(null);
  const dragId = useRef<GroupId | null>(null);

  // restore once on mount (the tolerant reads — defaults ship everything)
  useEffect(() => {
    setOptions(loadSectionToggles());
    setOrder(loadSectionOrder());
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
    () => sessionReportCensus(snapshot, options, order),
    [snapshot, options, order],
  );
  const shippable =
    census.figures + census.tables + census.metrics + census.images > 0;
  const customOrder = order.join() !== SESSION_GROUP_ORDER.join();
  // the chips render in the walking order, only the shippable groups
  const orderedGroups = useMemo(
    () =>
      order
        .map((id) => GROUPS.find((group) => group.id === id))
        .filter((group): group is (typeof GROUPS)[number] => group !== undefined)
        .filter((group) => available[group.id]),
    [order, available],
  );

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

  /** (100x-132) Move one group one VISIBLE slot (the chips' arrow
   * buttons — the keyboard/pointer path alongside drag-and-drop).  The
   * swap targets the nearest AVAILABLE neighbour in the direction, so
   * a click always moves the chip the eye can see (an unavailable
   * group between two chips is skipped, never blocking the move). */
  function moveGroup(id: GroupId, delta: number) {
    setOrder((prev) => {
      const visible = prev.filter((groupId) => available[groupId]);
      const target = visible[visible.indexOf(id) + delta];
      if (target === undefined) return prev;
      const from = prev.indexOf(id);
      const to = prev.indexOf(target);
      if (from < 0 || to < 0) return prev;
      const next = [...prev];
      next.splice(from, 1);
      next.splice(to, 0, id);
      saveSectionOrder(next);
      return next;
    });
  }

  /** Drop the dragged chip before the target chip (insert-before). */
  function dropOn(target: GroupId) {
    const id = dragId.current;
    if (id === null || id === target) return;
    setOrder((prev) => {
      const from = prev.indexOf(id);
      let to = prev.indexOf(target);
      if (from < 0 || to < 0) return prev;
      if (from < to) to -= 1; // removing the dragged chip shifts the target left
      const next = [...prev];
      next.splice(from, 1);
      next.splice(to, 0, id);
      saveSectionOrder(next);
      return next;
    });
  }

  function clearDrag() {
    dragId.current = null;
    setDragging(null);
    setDropBefore(null);
  }

  /** Restore the canonical walking order (the dashed reset chip). */
  function resetOrder() {
    setOrder([...SESSION_GROUP_ORDER]);
    saveSectionOrder([...SESSION_GROUP_ORDER]);
  }

  async function download() {
    setBusy(true);
    try {
      // the desktop twin bakes the APP's theme radios; the web twin
      // bakes the app's resolved palette (light/dark — "auto" stays a
      // reader-side choice via the report's own toggle).
      const payload = collectSessionReport(snapshot, options, order);
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
            ship renders no chip (there is nothing to skip).
            (100x-132) the chips reorder: drag (or the arrows on each
            chip's right) walk the groups into any order. */}
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-foreground">Include in report</p>
          <div
            className="flex flex-wrap gap-1.5"
            role="list"
            aria-label="Report section groups — drag or arrow to reorder"
          >
            {orderedGroups.map((group, index) => {
              const on = options[group.id] !== false;
              const Icon = group.icon;
              const isDragging = dragging === group.id;
              const showCaret =
                dragging !== null && dropBefore === group.id && !isDragging;
              return (
                <span
                  key={group.id}
                  role="listitem"
                  aria-roledescription="sortable chip"
                  draggable
                  onDragStart={(event) => {
                    dragId.current = group.id;
                    setDragging(group.id);
                    event.dataTransfer.effectAllowed = "move";
                    event.dataTransfer.setData("text/plain", group.id);
                  }}
                  onDragOver={(event) => {
                    if (dragId.current === null || dragId.current === group.id)
                      return;
                    event.preventDefault();
                    event.dataTransfer.dropEffect = "move";
                    setDropBefore(group.id);
                  }}
                  onDrop={(event) => {
                    event.preventDefault();
                    dropOn(group.id);
                    clearDrag();
                  }}
                  onDragEnd={clearDrag}
                  className="group/chip relative inline-flex items-center"
                >
                  {showCaret ? (
                    <span
                      aria-hidden
                      className="absolute -left-1 top-1 bottom-1 w-0.5 rounded-full bg-primary"
                    />
                  ) : null}
                  <button
                    type="button"
                    aria-pressed={on}
                    onClick={() => toggleGroup(group.id)}
                    title={
                      on
                        ? `Included: ${group.hint} — click to skip it`
                        : `Skipped: ${group.hint} — click to include it`
                    }
                    className={`inline-flex min-h-11 cursor-grab items-center gap-1.5 rounded-full border px-3.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring active:cursor-grabbing ${
                      on
                        ? "border-primary/40 bg-primary/10 text-primary hover:bg-primary/15"
                        : "border-dashed border-muted-foreground/30 bg-muted/30 text-muted-foreground hover:border-muted-foreground/50 hover:text-foreground"
                    } ${isDragging ? "opacity-40" : ""}`}
                  >
                    <GripVertical
                      className="size-3 text-muted-foreground/70"
                      aria-hidden
                    />
                    {on ? (
                      <Check className="size-3.5" aria-hidden />
                    ) : (
                      <X className="size-3.5" aria-hidden />
                    )}
                    <Icon className="size-3.5" aria-hidden />
                    {group.label}
                  </button>
                  {/* the micro arrows — hover/focus reveals them; they
                      are the keyboard/touch path for the same reorder
                      the drag performs (each carries an extended hit
                      area via its ::before) */}
                  <span className="ml-0.5 inline-flex flex-col opacity-0 transition-opacity focus-within:opacity-100 group-hover/chip:opacity-100">
                    <button
                      type="button"
                      aria-label={`Move ${group.label} earlier in the report`}
                      title={`Move ${group.label} earlier (slot ${index + 1} of ${orderedGroups.length})`}
                      disabled={index === 0}
                      onClick={() => moveGroup(group.id, -1)}
                      className="relative flex size-5 items-center justify-center rounded text-muted-foreground before:absolute before:-inset-1.5 before:content-[''] hover:text-foreground disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      <ChevronUp className="size-3" aria-hidden />
                    </button>
                    <button
                      type="button"
                      aria-label={`Move ${group.label} later in the report`}
                      title={`Move ${group.label} later (slot ${index + 1} of ${orderedGroups.length})`}
                      disabled={index === orderedGroups.length - 1}
                      onClick={() => moveGroup(group.id, 1)}
                      className="relative flex size-5 items-center justify-center rounded text-muted-foreground before:absolute before:-inset-1.5 before:content-[''] hover:text-foreground disabled:cursor-not-allowed disabled:opacity-30"
                    >
                      <ChevronDown className="size-3" aria-hidden />
                    </button>
                  </span>
                </span>
              );
            })}
            {/* the dashed reset — only while the order drifted from the
                canonical one (the finders' Clear-chip convention) */}
            {customOrder ? (
              <button
                type="button"
                onClick={resetOrder}
                title="Restore the default walking order (models → curve → regimen → data tables → surgery)"
                className="inline-flex min-h-11 items-center gap-1.5 rounded-full border border-dashed border-muted-foreground/30 bg-muted/30 px-3.5 text-xs font-medium text-muted-foreground transition-colors hover:border-muted-foreground/50 hover:text-foreground"
              >
                <RotateCcw className="size-3.5" aria-hidden />
                Reset walking order
              </button>
            ) : null}
          </div>
          <p className="text-[11px] text-muted-foreground">
            Skipped groups stay skipped on this browser; groups with nothing
            to ship never show a chip. Drag a chip (or use its arrows) to
            set the report&apos;s walking order — it persists per browser.
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
                {customOrder ? " · in your chosen walking order" : ""}
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
