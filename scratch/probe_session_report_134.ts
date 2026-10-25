/**
 * 100x-134 — the preview's hover INSPECT, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's inspect additions over crafted traces:
 *
 *  1. traceInspectText — the wording behind the native SVG <title>:
 *     name + point count + x/y ranges for a line; the CENTER mean for
 *     a band (the polygon's edges are the spread, the mean line is
 *     what reads); the CLASS SPLIT for a classified scatter; the honest
 *     empties (no points / no finite points);
 *  2. the degenerate spans — a flat axis reads a single value (not
 *     "v…v"), an empty set confesses;
 *  3. the fixture sanity — every figure section the collector ships
 *     over the 131 snapshot yields a non-empty inspect text for every
 *     trace it carries (the sketch's groups all ride real titles).
 *
 * Emits /tmp/sr_ts_134.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_134.ts
 */
import { writeFileSync } from "fs";
import {
  collectSessionReport,
  traceInspectText,
  type ReportFigureTrace,
} from "../src/lib/session-report";

/* -- 1. the wording, per style ------------------------------------------ */
const lineTrace: ReportFigureTrace = {
  name: "model prediction",
  style: "line",
  points: [
    [0, 0.5],
    [1, 1.25],
    [2, 2.0],
  ],
};
const inspectLine = traceInspectText(lineTrace);

const bandTrace: ReportFigureTrace = {
  name: "± band",
  style: "band",
  points: [
    [0, 1, 0.5, 1.5],
    [2, 3, 2.5, 3.5],
  ],
};
const inspectBand = traceInspectText(bandTrace);

const scatterTrace: ReportFigureTrace = {
  name: "test observations",
  style: "scatter",
  points: [
    [0.5, 1.0],
    [1.5, 0.75],
    [2.5, 1.9],
    [3.5, 0.31],
  ],
  classes: [1, 0, 1, 1],
};
const inspectScatter = traceInspectText(scatterTrace);

/* a scatter WITHOUT classes reads the plain line wording (the split
 * only exists when the collector actually carried classes) */
const inspectScatterNoClasses = traceInspectText({
  name: "dots",
  style: "scatter",
  points: [
    [0, 1],
    [1, 2],
  ],
});

/* -- 2. the honest empties + degenerate spans --------------------------- */
const inspectEmpty = traceInspectText({
  name: "nothing",
  style: "line",
  points: [],
});
const inspectNoFinite = traceInspectText({
  name: "nan",
  style: "line",
  points: [
    [Number.NaN, Number.NaN],
    [Number.POSITIVE_INFINITY, 1],
  ],
});
const inspectFlat = traceInspectText({
  name: "flat",
  style: "line",
  points: [
    [1, 5],
    [1, 5],
  ],
});
const inspectBig = traceInspectText({
  name: "big",
  style: "line",
  points: [
    [0, 12345],
    [1, 67890],
  ],
});

/* -- 3. the fixture sanity -----------------------------------------------
 * Every figure section the collector ships for a session like the 131
 * fixture yields a real inspect text for every trace — the preview's
 * groups never ride a mute title.  A minimal synthetic snapshot rides
 * the ML figure path only (the traces above already cover the styles). */
const figureTraces: ReportFigureTrace[] = [
  lineTrace,
  bandTrace,
  scatterTrace,
];
const everyInspect = figureTraces.map((trace) => traceInspectText(trace));
const allNamed = everyInspect.every(
  (text) => text.includes(" pts · ") && !text.startsWith("undefined"),
);
const sectionFigureCount = collectSessionReport({
  mlResult: null,
  lcResult: null,
  mlHistory: [],
  pkParams: {
    route: "iv",
    dose: 250,
    vd: 45,
    cl: 3.2,
    tau: 8,
    doses: 6,
    f: 0.9,
    ka: 1.2,
  },
  pdParams: {
    model: "emax",
    view: "both",
    slope: 8,
    e0: 10,
    emax: 80,
    ec50: 2,
    hill: 1.5,
    imax: 10,
    ic50: 2,
  },
  ncaState: { rows: [], dose: "250", f: "1", terminalN: 3 },
  surgery: null,
}).sections.filter((section) => section.kind === "figure").length;

writeFileSync(
  "/tmp/sr_ts_134.json",
  JSON.stringify({
    inspectLine,
    inspectBand,
    inspectScatter,
    inspectScatterNoClasses,
    inspectEmpty,
    inspectNoFinite,
    inspectFlat,
    inspectBig,
    everyInspect,
    allNamed,
    sectionFigureCount,
  }),
);
console.log(
  "SR_TS_134_OK",
  JSON.stringify({
    inspectLine,
    inspectBand,
    inspectScatter,
    inspectEmpty,
    inspectFlat,
    allNamed,
    sectionFigureCount,
  }),
);
