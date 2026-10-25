/**
 * 100x-137 — the preview's CONTENTS twin, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's pure derivation — `sessionReportContents` — the exact
 * list the downloaded file's header Contents block prints:
 *
 *  1. the walking order — every table/figure/image section's title in
 *     payload order, metrics sections NEVER listed (title-page
 *     material, the file twin's discipline);
 *  2. the indices are the OVERALL payload indices (0-based, gaps where
 *     metrics sat) — the preview's own numbered entries use the same
 *     coordinates, so the strip's numbers and the list's numbers never
 *     disagree;
 *  3. the kinds ride along (table / figure / image — images list like
 *     figures, the add_figure contract they ride in the file);
 *  4. the noise rule — a metrics-only payload yields [] (the strip
 *     hides itself), and so does an empty one;
 *  5. the file-twin parity — over the same mixed fixture, the derived
 *     lines are exactly the titles the BRIDGE would hand add_table /
 *     add_figure (every non-metrics section's title, in order).
 *
 * Emits /tmp/sr_ts_137.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_137.ts
 */
import { writeFileSync } from "fs";
import {
  sessionReportContents,
  type ReportSection,
} from "../src/lib/session-report";

const out: Record<string, unknown> = {};

/* -- the mixed fixture: every kind, metrics interleaved ----------------- */
const mixed: ReportSection[] = [
  {
    kind: "metrics",
    title: "ML training",
    items: [
      ["model", "ridge"],
      ["R²", "0.87"],
    ],
  },
  {
    kind: "figure",
    title: "Predicted vs observed",
    xLabel: "observed",
    yLabel: "predicted",
    traces: [
      { name: "model prediction", style: "line", points: [[0, 0], [1, 1]] },
    ],
  },
  {
    kind: "table",
    title: "NCA observed data",
    columns: ["#", "time (h)"],
    rows: [[1, "0.5"]],
  },
  {
    kind: "image",
    title: "Ring topology mutants — 3 structure panel(s)",
    dataUri: "data:image/png;base64,AAAA",
  },
  {
    kind: "metrics",
    title: "Learning curve",
    items: [["folds", "5"]],
  },
  {
    kind: "figure",
    title: "PK regimen — concentration vs time",
    xLabel: "time (h)",
    yLabel: "concentration",
    traces: [
      { name: "regimen", style: "line", points: [[0, 0], [1, 2]] },
    ],
  },
];

/* -- 1. the walking order, metrics never listed -------------------------- */
const lines = sessionReportContents(mixed);
out.mixedTitles = lines.map((line) => line.title);
out.metricsNeverListed = !lines.some((line) => line.title.startsWith("ML") || line.title.startsWith("Learning"));

/* -- 2. the indices are the OVERALL payload indices ---------------------- */
out.mixedIndices = lines.map((line) => line.index);
out.indicesAreOverall = out.mixedIndices.every(
  (index) => mixed[index as number].title === lines.find((l) => l.index === index)?.title,
);

/* -- 3. the kinds ride along --------------------------------------------- */
out.mixedKinds = lines.map((line) => line.kind);

/* -- 4. the noise rule ---------------------------------------------------- */
const metricsOnly: ReportSection[] = [
  { kind: "metrics", title: "just cards", items: [["a", "1"]] },
];
out.metricsOnlyYieldsEmpty = sessionReportContents(metricsOnly).length === 0;
out.emptyYieldsEmpty = sessionReportContents([]).length === 0;

/* -- 5. the file-twin parity --------------------------------------------- */
// every non-metrics section's title, in payload order — exactly the
// lines the bridge hands add_table / add_figure (the file's Contents)
const expected = mixed
  .filter((section) => section.kind !== "metrics")
  .map((section) => section.title);
out.fileTwinParity =
  JSON.stringify(lines.map((line) => line.title)) === JSON.stringify(expected);

/* -- the honest verdicts -------------------------------------------------- */
if (!out.metricsNeverListed) throw new Error("a metrics section was listed");
if (!out.indicesAreOverall) throw new Error("indices are not the overall ones");
if (!out.metricsOnlyYieldsEmpty) throw new Error("metrics-only did not yield []");
if (!out.emptyYieldsEmpty) throw new Error("empty payload did not yield []");
if (!out.fileTwinParity) throw new Error("the derivation drifted from the file twin");

writeFileSync("/tmp/sr_ts_137.json", JSON.stringify(out, null, 2));
console.log("PROBE137_OK — contents lines:", lines.length);
