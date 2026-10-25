"use client";

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type KeyboardEvent } from "react";
import { useTheme } from "next-themes";
import {
  Activity,
  Check,
  ChevronDown,
  ChevronUp,
  Dna,
  Eye,
  FileDown,
  Gauge,
  GripVertical,
  History,
  Image as ImageIcon,
  LineChart,
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useToolkitStore } from "@/lib/store";
import {
  pinWasTouched,
  pinnedTraceSnapshot,
  savePinnedTrace,
  subscribeToPinnedTraces,
} from "@/lib/pinned-trace-store";
import {
  collectSessionReport,
  contentsCensusLegs,
  contentsCensusText,
  figureSketchBounds,
  orderConfessionText,
  PREVIEW_SCATTER_POINTS,
  PREVIEW_TABLE_ROWS,
  traceInspectText,
  SESSION_GROUP_ORDER,
  sessionContentsCensus,
  sessionGroupsAvailable,
  sessionReportCensus,
  sessionReportContents,
  type ReportContentsEntry,
  type ReportFigureSection,
  type ReportSection,
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
/*                                                                      */
/* (100x-133) The PREVIEW: a dialog renders the ordered sections before */
/* the download — the actual walking order made visible, section by     */
/* section (metric cards, table heads, figure SKETCHES drawn from the   */
/* same trace data, the surgery grids' real PNGs).  The sketches are    */
/* approximations and say so; the downloaded report renders every       */
/* figure through matplotlib on the bridge.                             */
/*                                                                      */
/* (100x-137) The preview gained the file's CONTENTS twin: a sticky     */
/* strip above the section list listing exactly what the downloaded     */
/* file's header Contents block lists (every table/figure/image page    */
/* in walking order — metrics are title-page material and never         */
/* listed), each line a button that JUMPS the dialog to its section      */
/* (smooth scroll + a brief accent ring on the arrival, the HTML        */
/* twin's :target ring mirrored inside the dialog).                     */
/*                                                                      */
/* (100x-139) The strip grew two honest affordances: the census line     */
/* became CLICKABLE LEGS (each kind's count is a button that            */
/* SPOTLIGHTS that kind's lines — the others dim, never hide, they       */
/* stay fully clickable; clicking the active leg again clears it;        */
/* the spotlight resets when the dialog closes so a fresh open starts   */
/* fresh), and the strip's confession names the ORDER's provenance      */
/* (the web twin of the desktop picklist's memory label — "your last     */
/* saved walking order was applied" when the browser's persisted         */
/* order was applied, and the canonical order confesses nothing, the     */
/* desktop's "nothing remembered: stay quiet" rule).                     */
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

/** (100x-136) The hook a sketch reads its pin through — subscribe to
 * the external store (src/lib/pinned-trace-store.ts, the probe-pinned
 * twin of the desktop's session-file key), snapshot THIS figure's
 * entry, and hydrate against null. */
function usePinnedTrace(title: string): number | null {
  return useSyncExternalStore(
    subscribeToPinnedTraces,
    () => pinnedTraceSnapshot(title),
    () => null,
  );
}

/* ------------------------------------------------------------------ */
/* The preview's section renderers (100x-133)                           */
/* ------------------------------------------------------------------ */

/** The sketch's viewBox geometry — one place so the math stays legible. */
const SKETCH_W = 320;
const SKETCH_H = 150;
const SKETCH_PAD = { left: 6, right: 6, top: 6, bottom: 18 };

/** (100x-133) A figure section's preview sketch — an inline SVG drawn
 *  from the SAME trace data the bridge hands matplotlib: lines become
 *  polylines, bands become filled polygons with their mean line on
 *  top, class scatters become dots (stride-capped, the shape survives
 *  the cap).  It is an honest approximation and the dialog says so —
 *  the downloaded report renders the real figure.  (100x-134) Every
 *  trace group now carries a hover INSPECT: a native SVG <title> with
 *  the trace's honest numbers (point count, x/y ranges, the band's
 *  center mean, the scatter's class split — `traceInspectText`), and
 *  the trace under the pointer thickens (group-hover stroke/fill) so
 *  WHICH trace the tooltip describes is never ambiguous.
 *  (100x-135) The inspect grew a KEYBOARD path: every trace group is
 *  focusable (Tab reaches it, role="group" + the inspect text as its
 *  aria-label, so screen readers announce the same numbers), focus
 *  thickens the trace exactly like hover does and shows a readout
 *  strip under the sketch carrying the FULL inspect text (the native
 *  <title> tooltip never appears on keyboard focus — a visible strip
 *  is the honest replacement), and Enter PINS the readout so it stays
 *  readable while focus walks on (Esc unpins, Enter again unpins too —
 *  the pin is a toggle, and only one trace is pinned at a time).
 *  (100x-136) The pin PERSISTS per browser: the pin lives in a small
 *  external store over localStorage (useSyncExternalStore — no restore
 *  effect anywhere), every pin/unpin writes through, a stored pin is
 *  validated against the live trace count at READ time (a stale index
 *  pins nothing), and a restored pin confesses its provenance in the
 *  readout — the desktop's report_page_order memory, mirrored exactly
 *  (its session-file key traded for this browser's storage). */
function FigureSketch({ section }: { section: ReportFigureSection }) {
  const [focusedTrace, setFocusedTrace] = useState<number | null>(null);
  // (100x-136) the pin rides the external store — the stored value,
  // validated against the live trace list at read time (a figure that
  // changed shape since the pin landed pins nothing, the honest
  // fallback), and the confession marks a pin this browser session
  // never wrote (it arrived from storage, the reader's last visit).
  const storedPin = usePinnedTrace(section.title);
  const pinnedTrace =
    storedPin !== null && storedPin < section.traces.length ? storedPin : null;
  const pinRestored = pinnedTrace !== null && !pinWasTouched(section.title);
  const bounds = figureSketchBounds(section.traces);
  const spanX = bounds.xMax - bounds.xMin;
  const spanY = bounds.yMax - bounds.yMin;
  const px = (x: number) =>
    SKETCH_PAD.left +
    ((x - bounds.xMin) / spanX) * (SKETCH_W - SKETCH_PAD.left - SKETCH_PAD.right);
  const py = (y: number) =>
    SKETCH_PAD.top +
    (1 - (y - bounds.yMin) / spanY) *
      (SKETCH_H - SKETCH_PAD.top - SKETCH_PAD.bottom);
  // the readout: the pinned trace wins over the focused one, so a pin
  // stays readable while focus walks elsewhere (the pin is deliberate)
  const readoutIndex = pinnedTrace ?? focusedTrace;
  const readoutTrace =
    readoutIndex === null ? null : (section.traces[readoutIndex] ?? null);
  /** (100x-135) one shared handler for every trace group's keyboard
   *  contract: Enter or Space toggles the pin on THIS trace, Escape
   *  clears a pin THIS trace holds.  Space is prevented so the page
   *  never scrolls while toggling. */
  const traceKeyDown =
    (traceIndex: number) =>
    (event: KeyboardEvent<SVGGElement>): void => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        // (100x-136) the pin writes through the store — the toggle AND
        // the memory are one call, and the notification re-renders.
        savePinnedTrace(
          section.title,
          pinnedTrace === traceIndex ? null : traceIndex,
        );
      } else if (event.key === "Escape" && pinnedTrace === traceIndex) {
        // the pin CONSUMES the Escape: clear the readout and stop the
        // event here, so the dialog's own close-on-escape never fires
        // on the same keystroke -- a reader clearing a pin sees the
        // cleared state, and only a SECOND Escape closes the dialog.
        // (100x-136) the clear writes through too — forgetting is a
        // stored decision, not a lost one.
        event.preventDefault();
        event.stopPropagation();
        savePinnedTrace(section.title, null);
      }
    };
  return (
    <>
      <svg
        viewBox={`0 0 ${SKETCH_W} ${SKETCH_H}`}
        role="img"
        aria-label={`Sketch of ${section.title} — ${section.xLabel} vs ${section.yLabel}`}
        className="h-36 w-full rounded-md border bg-background"
      >
      {section.traces.map((trace, traceIndex) => {
        if (trace.style === "band") {
          const band = trace.points.filter(
            (p): p is [number, number, number, number] =>
              p.length >= 4 && Number.isFinite(p[2]) && Number.isFinite(p[3]),
          );
          if (band.length < 2) return null;
          const forward = band
            .map((p) => `${px(p[0]).toFixed(1)},${py(p[2]).toFixed(1)}`)
            .join(" ");
          const backward = [...band]
            .reverse()
            .map((p) => `${px(p[0]).toFixed(1)},${py(p[3]).toFixed(1)}`)
            .join(" ");
          return (
            <g
              key={traceIndex}
              className="group cursor-help"
              tabIndex={0}
              role="group"
              aria-label={traceInspectText(trace)}
              onFocus={() => setFocusedTrace(traceIndex)}
              onBlur={() =>
                setFocusedTrace((prev) => (prev === traceIndex ? null : prev))
              }
              onKeyDown={traceKeyDown(traceIndex)}
            >
              <title>{traceInspectText(trace)}</title>
              <polygon
                points={`${forward} ${backward}`}
                className="fill-primary/15 transition-[fill] group-hover:fill-primary/25 group-focus-visible:fill-primary/25"
              />
              <polyline
                points={band
                  .map((p) => `${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`)
                  .join(" ")}
                className="fill-none stroke-primary transition-[stroke-width] group-hover:stroke-2 group-focus-visible:stroke-2"
                strokeWidth={1.5}
              />
            </g>
          );
        }
        if (trace.style === "scatter") {
          const total = trace.points.length;
          const stride = Math.max(1, Math.ceil(total / PREVIEW_SCATTER_POINTS));
          const dots: Array<{ cx: number; cy: number; fill: string; opacity: number }> = [];
          for (let i = 0; i < total; i += stride) {
            const point = trace.points[i];
            const klass = trace.classes?.[i] ?? 1;
            const proba = trace.probas?.[i];
            dots.push({
              cx: px(point[0]),
              cy: py(point[1]),
              fill: klass === 1 ? "currentColor" : "var(--muted-foreground)",
              opacity:
                proba === undefined ? 0.85 : Math.max(0.25, Math.abs(proba - 0.5) * 2),
            });
          }
          return (
            <g
              key={traceIndex}
              className="group cursor-help text-primary"
              tabIndex={0}
              role="group"
              aria-label={traceInspectText(trace)}
              onFocus={() => setFocusedTrace(traceIndex)}
              onBlur={() =>
                setFocusedTrace((prev) => (prev === traceIndex ? null : prev))
              }
              onKeyDown={traceKeyDown(traceIndex)}
            >
              <title>{traceInspectText(trace)}</title>
              {dots.map((dot, dotIndex) => (
                <circle
                  key={dotIndex}
                  cx={dot.cx.toFixed(1)}
                  cy={dot.cy.toFixed(1)}
                  r={1.6}
                  fill={dot.fill}
                  opacity={dot.opacity}
                />
              ))}
            </g>
          );
        }
        if (trace.points.length < 2) return null;
        return (
          <g
            key={traceIndex}
            className="group cursor-help"
            tabIndex={0}
            role="group"
            aria-label={traceInspectText(trace)}
            onFocus={() => setFocusedTrace(traceIndex)}
            onBlur={() =>
              setFocusedTrace((prev) => (prev === traceIndex ? null : prev))
            }
            onKeyDown={traceKeyDown(traceIndex)}
          >
            <title>{traceInspectText(trace)}</title>
            <polyline
              points={trace.points
                .map((p) => `${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`)
                .join(" ")}
              className="fill-none stroke-primary transition-[stroke-width] group-hover:stroke-2 group-focus-visible:stroke-2"
              strokeWidth={1.5}
            />
          </g>
        );
      })}
      <text
        x={SKETCH_W / 2}
        y={SKETCH_H - 4}
        textAnchor="middle"
        fontSize={8}
        className="fill-muted-foreground"
      >
        {section.xLabel}
      </text>
      <text x={2} y={10} fontSize={8} className="fill-muted-foreground">
        {section.yLabel}
      </text>
      </svg>
      {/* (100x-135) the keyboard readout strip — the FULL inspect text
        *  for the focused (or pinned) trace.  The native <title>
        *  tooltip never shows on keyboard focus, so this visible strip
        *  is the honest replacement; a pin appends its own confession
        *  so the reader knows Esc clears it.  role="status" makes it a
        *  polite live region: the numbers are announced as they land. */}
      {readoutTrace ? (
        <p
          role="status"
          data-pinned={pinnedTrace === readoutIndex ? "true" : undefined}
          className="rounded-md border bg-muted/30 px-2 py-1 text-[11px] text-muted-foreground"
        >
          {traceInspectText(readoutTrace)}
          {pinnedTrace === readoutIndex
            ? pinRestored
              ? " — pinned from your last visit (Esc clears, Enter toggles)"
              : " — pinned (Esc clears, Enter toggles)"
            : ""}
        </p>
      ) : null}
    </>
  );
}

/** One section's preview body — the mini rendering per kind. */
function SectionPreviewBody({ section }: { section: ReportSection }) {
  if (section.kind === "metrics") {
    return (
      <dl className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
        {section.items.map(([key, value]) => (
          <div
            key={key}
            className="rounded-md border bg-muted/30 px-2 py-1.5 text-xs"
          >
            <dt className="text-muted-foreground">{key}</dt>
            <dd className="truncate font-medium" title={String(value)}>
              {String(value)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  if (section.kind === "table") {
    const shown = section.rows.slice(0, PREVIEW_TABLE_ROWS);
    const hidden = section.rows.length - shown.length;
    return (
      <div className="space-y-1">
        <div className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                {section.columns.map((column) => (
                  <TableHead key={column} className="h-7 text-[11px]">
                    {column}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <TableCell key={cellIndex} className="py-1 text-[11px]">
                      {String(cell)}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        {hidden > 0 ? (
          <p className="text-[11px] text-muted-foreground">
            + {hidden} more row{hidden === 1 ? "" : "s"} in the download — the
            preview caps at {PREVIEW_TABLE_ROWS} rows.
          </p>
        ) : null}
      </div>
    );
  }
  if (section.kind === "figure") {
    return (
      <div className="space-y-1.5">
        <FigureSketch section={section} />
        <p className="text-[11px] text-muted-foreground">
          {section.traces.length} trace{section.traces.length === 1 ? "" : "s"}:{" "}
          {section.traces.map((trace) => trace.name).join(" · ")}
          — a quick sketch; hover a trace (or Tab to it) to inspect its
          numbers, Enter pins the readout — a pin is remembered per
          browser; the download renders this figure through matplotlib.
        </p>
      </div>
    );
  }
  return (
    <img
      src={section.dataUri}
      alt={section.title}
      className="max-h-56 w-full rounded-md border bg-background object-contain"
    />
  );
}

/** The kind badge a preview entry carries — icon + label. */
function sectionKindBadge(section: ReportSection): {
  icon: typeof Activity;
  label: string;
} {
  if (section.kind === "metrics") return { icon: Gauge, label: "metrics" };
  if (section.kind === "table") return { icon: Table2, label: "table" };
  if (section.kind === "figure") return { icon: LineChart, label: "figure" };
  return { icon: ImageIcon, label: "structure grid" };
}

/** (100x-137) The icon a Contents strip line carries for its kind —
 * the same icons the preview entries below wear, so a line and its
 * target are recognizably the same animal. */
function contentsKindIcon(kind: ReportContentsEntry["kind"]): typeof Activity {
  if (kind === "table") return Table2;
  if (kind === "figure") return LineChart;
  return ImageIcon;
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
  // (100x-133) the preview dialog — the walking order made visible
  const [previewOpen, setPreviewOpen] = useState(false);
  // (100x-137) the Contents strip's jump target — the section the last
  // jump landed on, held briefly so the arrival ring fades on its own
  const [jumpedTo, setJumpedTo] = useState<number | null>(null);
  // (100x-139) the strip's SPOTLIGHT — the census-leg kind whose lines
  // stay lit while the others dim (null = no spotlight, every line at
  // full strength; the dimmed lines stay fully clickable — dimmed is
  // never hidden).  Cleared when the dialog closes so a fresh open
  // starts fresh — a spotlight is a view aid, never a filter on what
  // the report ships.
  const [spotlight, setSpotlight] = useState<ReportContentsEntry["kind"] | null>(null);
  const dragId = useRef<GroupId | null>(null);
  const jumpTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // the jump timer is cleaned up when the component goes (a plain
  // unmount cleanup — never a restore effect)
  useEffect(
    () => () => {
      if (jumpTimer.current) clearTimeout(jumpTimer.current);
    },
    [],
  );

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
  // (100x-132) the order-drift flag — the persisted browser order vs
  // the canonical one (the dashed Reset chip appears while this holds;
  // the strip's order confession keys off it too)
  const customOrder = order.join() !== SESSION_GROUP_ORDER.join();
  // (100x-133) the exact payload the download would POST — the preview
  // renders THIS (same options, same walking order), so what the dialog
  // shows is what the file ships, section for section.
  const payload = useMemo(
    () => collectSessionReport(snapshot, options, order),
    [snapshot, options, order],
  );
  // (100x-137) the Contents lines the strip shows — the SAME list the
  // downloaded file's header block prints (the pure derivation, so
  // the probe can pin it; a metrics-only report yields [] and the
  // strip hides itself, the file twin's noise rule)
  const contents = useMemo(
    () => sessionReportContents(payload.sections),
    [payload],
  );
  // (100x-138) the strip's KIND CENSUS — the counts of exactly what
  // it lists ("2 figures · 1 table · 1 structure grid"), derived from
  // the Contents entries themselves so metrics are excluded by
  // construction; the footer's own census counts metric cards, this
  // one never does — the strip confesses only its own lines
  const contentsCensusLine = useMemo(
    () => contentsCensusText(sessionContentsCensus(contents)),
    [contents],
  );
  // (100x-139) the census's LEGS — the clickable form of the same line
  // (each kind's count a button that spotlights its lines); a leg's
  // label is the same text the plain line prints, so the visual census
  // never drifts from the aria one
  const contentsLegs = useMemo(
    () => contentsCensusLegs(sessionContentsCensus(contents)),
    [contents],
  );
  // (100x-139) the strip's ORDER confession — the web twin of the
  // desktop picklist's memory label: the browser's persisted walking
  // order applied ("your last saved order") vs the canonical order's
  // honest silence (the pure derivation, probe-pinned)
  const orderConfession = useMemo(
    () => orderConfessionText(customOrder),
    [customOrder],
  );
  const shippable =
    census.figures + census.tables + census.metrics + census.images > 0;
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

  /** (100x-137) Jump the preview's list to one section — the Contents
   *  strip's click contract: smooth-scroll the entry into view and
   *  hold a brief accent ring on it (the HTML twin's :target ring,
   *  mirrored inside the dialog; the ring fades on its own timer). */
  function jumpToSection(index: number) {
    const target = document.getElementById(`preview-section-${index}`);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    setJumpedTo(index);
    if (jumpTimer.current) clearTimeout(jumpTimer.current);
    jumpTimer.current = setTimeout(() => setJumpedTo(null), 1600);
  }

  async function download() {
    setBusy(true);
    try {
      // the desktop twin bakes the APP's theme radios; the web twin
      // bakes the app's resolved palette (light/dark — "auto" stays a
      // reader-side choice via the report's own toggle).
      const body = {
        ...payload,
        theme: resolvedTheme === "dark" ? "dark" : "light",
      };
      const res = await fetch("/api/report/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="outline"
              onClick={() => setPreviewOpen(true)}
              disabled={!shippable}
              className="min-h-11 shrink-0"
              aria-label="Preview the report's ordered sections before downloading"
            >
              <Eye className="size-4" aria-hidden />
              Preview order
            </Button>
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
        </div>
      </CardContent>

      {/* (100x-133) the PREVIEW — the walking order made visible: the
          exact payload the download would POST, rendered section by
          section in a dialog BEFORE any file is written.  Metric cards,
          table heads (capped, with the honest "+N more rows"), figure
          sketches drawn from the same trace data, and the surgery
          grids' real PNGs — numbered, in the chosen walking order. */}
      <Dialog
        open={previewOpen}
        onOpenChange={(open) => {
          setPreviewOpen(open);
          // (100x-139) a closed dialog forgets its spotlight — a fresh
          // open starts fresh (the view aid never leaks into the next
          // session's reading)
          if (!open) setSpotlight(null);
        }}
      >
        {/* (100x-135) the ESCAPE LAYERING: Radix closes the dialog on
            Escape at the document-capture phase — before any trace
            group's own keydown could stop it — so the dialog itself
            must hold the door while a pinned readout is showing: the
            first Escape clears the pin (the group's handler), the
            dialog stays; only a second Escape (no pin held) closes
            it.  The pin marks itself with data-pinned for this query
            — one attribute, no lifted state. */}
        <DialogContent
          className="sm:max-w-2xl"
          onEscapeKeyDown={(event) => {
            if (document.querySelector("p[data-pinned='true']")) {
              event.preventDefault();
            }
          }}
        >
          <DialogHeader>
            <DialogTitle className="flex flex-wrap items-center gap-2 text-base">
              <Eye className="size-4 text-primary" aria-hidden />
              Report preview
              <Badge variant="secondary" className="ml-1 font-normal">
                {payload.sections.length} section
                {payload.sections.length === 1 ? "" : "s"}
                {customOrder ? " · your walking order" : " · default order"}
              </Badge>
            </DialogTitle>
            <DialogDescription className="text-xs">
              The exact walking order the downloaded file ships — the same
              sections, the same order, nothing sent anywhere.  Figure
              sketches are quick approximations; the download renders them
              through matplotlib.  It opens {resolvedTheme === "dark" ? "dark" : "light"}.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[62vh] overflow-y-auto pr-1">
            {/* (100x-137) the file's CONTENTS twin — a sticky strip at the
                top of the scroll area listing exactly what the downloaded
                file's header block lists (metrics never listed, the noise
                rule), each line jumping the dialog to its section.  It
                stays pinned while the list scrolls — a dialog is small,
                a contents list that scrolls away is a contents list you
                can only use once. */}
            {contents.length > 0 ? (
              <nav
                aria-label="Contents — the downloaded file's header block, mirrored"
                className="sticky top-0 z-10 mb-2.5 rounded-lg border bg-card/95 px-3 py-2 shadow-sm backdrop-blur"
              >
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <p className="text-xs font-medium text-foreground">Contents</p>
                  {/* (100x-138) the KIND CENSUS — the counts of exactly
                      what the strip lists (metrics excluded, the strip's
                      own noise rule), the footer's singular/plural
                      discipline.  (100x-139) the census became CLICKABLE
                      LEGS: each kind's count spotlights that kind's lines
                      (the others dim — never hide, they stay fully
                      clickable); clicking the active leg again clears
                      it; a kind with zero lines renders as plain text
                      (nothing to spotlight — an honest no-op avoided). */}
                  <p
                    className="text-[10px] tabular-nums text-muted-foreground"
                    aria-label="The contents census — the kinds and counts the strip lists; click a kind to spotlight its lines"
                    title={`${contentsCensusLine} — click a kind to spotlight its lines`}
                  >
                    {contentsLegs.map((leg, legIndex) => (
                      <span key={leg.kind}>
                        {legIndex > 0 ? <span aria-hidden> · </span> : null}
                        {leg.count > 0 ? (
                          <button
                            type="button"
                            aria-pressed={spotlight === leg.kind}
                            onClick={() =>
                              setSpotlight(spotlight === leg.kind ? null : leg.kind)
                            }
                            title={
                              spotlight === leg.kind
                                ? `Spotlight on — click to clear (every line back at full strength)`
                                : `Spotlight the ${leg.count} ${leg.kind === "image" ? "structure grid" : leg.kind}${leg.count === 1 ? "" : "s"} — the other kinds dim, never hide`
                            }
                            className={`rounded px-0.5 -mx-0.5 tabular-nums underline-offset-2 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                              spotlight === leg.kind
                                ? "font-medium text-primary underline decoration-dotted"
                                : "text-muted-foreground hover:text-primary hover:underline decoration-dotted"
                            }`}
                          >
                            {leg.label}
                          </button>
                        ) : (
                          <span className="tabular-nums">{leg.label}</span>
                        )}
                      </span>
                    ))}
                  </p>
                </div>
                <ul className="mt-1 flex flex-wrap gap-1" role="list">
                  {contents.map((entry) => {
                    const KindIcon = contentsKindIcon(entry.kind);
                    // (100x-139) the SPOTLIGHT — a line outside the
                    // spotlighted kind DIMS (opacity, never display: the
                    // line stays exactly as clickable as before — dimmed
                    // is not hidden, and the jump keeps working)
                    const dimmed = spotlight !== null && entry.kind !== spotlight;
                    return (
                      <li
                        key={entry.index}
                        role="listitem"
                        className={dimmed ? "opacity-40 transition-opacity" : "transition-opacity"}
                      >
                        <button
                          type="button"
                          onClick={() => jumpToSection(entry.index)}
                          aria-label={`Jump to section ${entry.index + 1}: ${entry.title}`}
                          title={`${entry.title} — jump to its entry (the downloaded file's Contents links to the page itself)`}
                          className="inline-flex max-w-56 items-center gap-1.5 rounded-full border bg-muted/40 px-2.5 py-1 text-[11px] text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        >
                          <span
                            className="tabular-nums font-semibold"
                            aria-hidden
                          >
                            {entry.index + 1}
                          </span>
                          <KindIcon className="size-3 shrink-0" aria-hidden />
                          <span className="truncate">{entry.title}</span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
                <p className="mt-1 text-[10px] text-muted-foreground">
                  The same block the downloaded file&apos;s header carries — click
                  a line to jump to it.
                  {/* (100x-139) the ORDER confession — the web twin of the
                      desktop picklist's memory label: the persisted order
                      applied says so and names the way out; the canonical
                      order stays quiet (nothing was remembered). */}
                  {orderConfession ? ` ${orderConfession}` : ""}
                </p>
              </nav>
            ) : null}
            {payload.sections.length > 0 ? (
              <ol className="space-y-2.5" aria-label="The report's sections in walking order">
                {payload.sections.map((section, index) => {
                  const badge = sectionKindBadge(section);
                  const BadgeIcon = badge.icon;
                  return (
                    <li
                      key={`${section.kind}-${index}-${section.title}`}
                      id={`preview-section-${index}`}
                      className={`scroll-mt-[84px] rounded-lg border bg-card p-3 transition-shadow ${
                        jumpedTo === index
                          ? "ring-2 ring-primary ring-offset-1 ring-offset-card"
                          : ""
                      }`}
                    >
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span
                          className="flex size-5 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold tabular-nums text-muted-foreground"
                          aria-hidden
                        >
                          {index + 1}
                        </span>
                        <Badge variant="secondary" className="gap-1 font-normal">
                          <BadgeIcon className="size-3" aria-hidden />
                          {badge.label}
                        </Badge>
                        <span className="min-w-0 flex-1 truncate text-sm font-medium">
                          {section.title}
                        </span>
                      </div>
                      <SectionPreviewBody section={section} />
                    </li>
                  );
                })}
              </ol>
            ) : (
              <p className="rounded-md border border-dashed p-4 text-center text-sm text-muted-foreground">
                Nothing would ship — enable at least one group on the card.
              </p>
            )}
          </div>
          <DialogFooter className="flex-col items-stretch gap-2 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-xs text-muted-foreground" aria-live="polite">
              {shippable
                ? `Ready to ship: ${census.figures} figure${census.figures === 1 ? "" : "s"} · ${census.tables} table${census.tables === 1 ? "" : "s"} · ${census.metrics} metric card${census.metrics === 1 ? "" : "s"}${census.images > 0 ? ` · ${census.images} structure grid${census.images === 1 ? "" : "s"}` : ""}`
                : "Nothing would ship yet."}
            </p>
            <Button
              onClick={() => {
                setPreviewOpen(false);
                download();
              }}
              disabled={busy || !shippable}
              className="min-h-11"
            >
              {busy ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <FileDown className="size-4" aria-hidden />
              )}
              Download this report (HTML)
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
