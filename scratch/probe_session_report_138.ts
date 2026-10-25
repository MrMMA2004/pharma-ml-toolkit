/**
 * 100x-138 — the preview's Contents strip KIND CENSUS, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's pure derivations — `sessionContentsCensus` (the counts of
 * exactly what the strip lists) and `contentsCensusText` (the readable
 * line):
 *
 *  1. the census counts the STRIP'S OWN lines — figures / tables /
 *     images, metrics excluded by construction (the footer's census
 *     counts metric cards too; this one never does);
 *  2. the wording — "2 figures · 1 table · 1 structure grid", the
 *     footer's singular/plural discipline (images read as "structure
 *     grid(s)", the established wording);
 *  3. a census with no images drops the leg entirely (the footer's
 *     images-conditional, mirrored);
 *  4. singular forms ("1 figure · 1 table" — no plural s);
 *  5. the strip's noise rule holds upstream — a metrics-only payload
 *     yields [] and the strip (census included) never renders.
 *
 * Emits /tmp/sr_ts_138.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_138.ts
 */
import { writeFileSync } from "fs";
import {
  contentsCensusText,
  sessionContentsCensus,
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

/* -- 1/2. the census over the strip's own lines + the wording ----------- */
const lines = sessionReportContents(mixed);
const census = sessionContentsCensus(lines);
out.mixedCensus = census;
out.mixedText = contentsCensusText(census);
out.metricsExcluded =
  census.figures + census.tables + census.images === lines.length;
out.metricsCounted = "metrics" in census;

/* -- 3. no images -> the leg drops --------------------------------------- */
const noImages: ReportSection[] = [
  {
    kind: "figure",
    title: "A",
    xLabel: "x",
    yLabel: "y",
    traces: [{ name: "t", style: "line", points: [[0, 0]] }],
  },
  {
    kind: "table",
    title: "B",
    columns: ["c"],
    rows: [[1]],
  },
];
out.noImagesText = contentsCensusText(
  sessionContentsCensus(sessionReportContents(noImages)),
);

/* -- 4. singular forms ---------------------------------------------------- */
const singular: ReportSection[] = [
  {
    kind: "figure",
    title: "only one",
    xLabel: "x",
    yLabel: "y",
    traces: [{ name: "t", style: "line", points: [[0, 0]] }],
  },
];
out.singularText = contentsCensusText(
  sessionContentsCensus(sessionReportContents(singular)),
);
const oneOfEach: ReportSection[] = [
  {
    kind: "figure",
    title: "f",
    xLabel: "x",
    yLabel: "y",
    traces: [{ name: "t", style: "line", points: [[0, 0]] }],
  },
  { kind: "table", title: "t", columns: ["c"], rows: [[1]] },
  { kind: "image", title: "g", dataUri: "data:image/png;base64,AAAA" },
];
out.oneOfEachText = contentsCensusText(
  sessionContentsCensus(sessionReportContents(oneOfEach)),
);

/* -- 5. the noise rule upstream ------------------------------------------ */
const metricsOnly: ReportSection[] = [
  { kind: "metrics", title: "just cards", items: [["a", "1"]] },
];
out.metricsOnlyYieldsEmpty =
  sessionReportContents(metricsOnly).length === 0 &&
  contentsCensusText(sessionContentsCensus(sessionReportContents(metricsOnly))) ===
    "0 figures · 0 tables";

/* -- the honest verdicts -------------------------------------------------- */
if (census.figures !== 2 || census.tables !== 1 || census.images !== 1) {
  throw new Error("the census miscounted the mixed fixture");
}
if (out.mixedText !== "2 figures · 1 table · 1 structure grid") {
  throw new Error(`the census text drifted: ${String(out.mixedText)}`);
}
if (!out.metricsExcluded) throw new Error("the census counted something else");
if (out.metricsCounted === true) throw new Error("the census carries metrics");
if (out.noImagesText !== "1 figure · 1 table") {
  throw new Error(`the no-images leg did not drop: ${String(out.noImagesText)}`);
}
if (out.singularText !== "1 figure · 0 tables") {
  throw new Error(`the singular form drifted: ${String(out.singularText)}`);
}
if (out.oneOfEachText !== "1 figure · 1 table · 1 structure grid") {
  throw new Error(`the one-of-each text drifted: ${String(out.oneOfEachText)}`);
}
if (!out.metricsOnlyYieldsEmpty) {
  throw new Error("the noise rule broke for the census");
}

writeFileSync("/tmp/sr_ts_138.json", JSON.stringify(out, null, 2));
console.log("PROBE138_OK — census:", JSON.stringify(census), "text:", out.mixedText);
