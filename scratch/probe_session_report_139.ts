/**
 * 100x-139 — the strip's SPOTLIGHT legs + the ORDER confession, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's pure derivations:
 *
 *  1. `orderConfessionText` — the web twin of the desktop picklist's
 *     memory label: a drifted (persisted) order confesses "Your last
 *     saved walking order was applied — Reset walking order forgets
 *     it."; the canonical order returns null (nothing remembered:
 *     stay quiet, the desktop's own noise rule);
 *  2. `contentsCensusLegs` — the census's CLICKABLE form: one leg per
 *     kind the strip lists, each carrying kind + label + count, so a
 *     click can spotlight exactly that kind's lines;
 *  3. the legs' discipline against `contentsCensusText` — the legs
 *     joined with " · " are the plain line, byte for byte (the visual
 *     census never drifts from the aria one);
 *  4. a zero-count images leg never renders (the plain line's own
 *     conditional, mirrored) while zero-count figures/tables legs DO
 *     (the plain line prints them — "1 figure · 0 tables");
 *  5. the legs' kinds are only figure/table/image — metrics can never
 *     be a spotlight target (excluded by construction upstream).
 *
 * Emits /tmp/sr_ts_139.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_139.ts
 */
import { writeFileSync } from "fs";
import {
  contentsCensusLegs,
  contentsCensusText,
  orderConfessionText,
  sessionContentsCensus,
  sessionReportContents,
  type ReportSection,
} from "../src/lib/session-report";

const out: Record<string, unknown> = {};

/* -- the mixed fixture: every kind, metrics interleaved ----------------- */
const mixed: ReportSection[] = [
  { kind: "metrics", title: "ML training", items: [["model", "ridge"]] },
  {
    kind: "figure",
    title: "Predicted vs observed",
    xLabel: "observed",
    yLabel: "predicted",
    traces: [{ name: "model prediction", style: "line", points: [[0, 0], [1, 1]] }],
  },
  { kind: "table", title: "NCA observed data", columns: ["#", "time (h)"], rows: [[1, "0.5"]] },
  {
    kind: "image",
    title: "Ring topology mutants — 3 structure panel(s)",
    dataUri: "data:image/png;base64,AAAA",
  },
  { kind: "metrics", title: "Learning curve", items: [["folds", "5"]] },
  {
    kind: "figure",
    title: "PK regimen — concentration vs time",
    xLabel: "time (h)",
    yLabel: "concentration",
    traces: [{ name: "regimen", style: "line", points: [[0, 0], [1, 2]] }],
  },
];

/* -- 1. the ORDER confession ---------------------------------------------- */
const CONFESSION =
  "Your last saved walking order was applied — Reset walking order forgets it.";
out.driftedConfession = orderConfessionText(true);
out.canonicalConfession = orderConfessionText(false);
out.confessionNamesTheWayOut = orderConfessionText(true)?.includes(
  "Reset walking order forgets it",
);

/* -- 2. the LEGS over the mixed fixture ------------------------------------ */
const legs = contentsCensusLegs(sessionContentsCensus(sessionReportContents(mixed)));
out.mixedLegs = legs;

/* -- 3. the legs ARE the plain line ---------------------------------------- */
out.legsJoined = legs.map((leg) => leg.label).join(" · ");
out.plainLine = contentsCensusText(sessionContentsCensus(sessionReportContents(mixed)));
out.legsMatchLine = out.legsJoined === out.plainLine;

/* -- 4. zero-count discipline ---------------------------------------------- */
const figureOnly: ReportSection[] = [
  {
    kind: "figure",
    title: "only one",
    xLabel: "x",
    yLabel: "y",
    traces: [{ name: "t", style: "line", points: [[0, 0]] }],
  },
];
const figureLegs = contentsCensusLegs(
  sessionContentsCensus(sessionReportContents(figureOnly)),
);
out.figureOnlyLegs = figureLegs;
out.zeroTableLegRides = figureLegs.some(
  (leg) => leg.kind === "table" && leg.count === 0 && leg.label === "0 tables",
);
out.noImagesLegWithoutImages = !figureLegs.some((leg) => leg.kind === "image");

/* -- 5. metrics can never be a spotlight target ---------------------------- */
out.metricsNeverALeg = legs.every(
  (leg) => leg.kind === "figure" || leg.kind === "table" || leg.kind === "image",
);

/* -- the honest verdicts ---------------------------------------------------- */
if (out.driftedConfession !== CONFESSION) {
  throw new Error(`the drifted confession drifted: ${String(out.driftedConfession)}`);
}
if (out.canonicalConfession !== null) {
  throw new Error("the canonical order confessed something — the noise rule broke");
}
if (out.confessionNamesTheWayOut !== true) {
  throw new Error("the confession lost its way out");
}
if (legs.length !== 3) throw new Error(`the legs miscounted: ${legs.length}`);
if (legs[0].kind !== "figure" || legs[0].count !== 2 || legs[0].label !== "2 figures") {
  throw new Error("the figure leg drifted");
}
if (legs[1].kind !== "table" || legs[1].count !== 1 || legs[1].label !== "1 table") {
  throw new Error("the table leg drifted");
}
if (
  legs[2].kind !== "image" ||
  legs[2].count !== 1 ||
  legs[2].label !== "1 structure grid"
) {
  throw new Error("the image leg drifted");
}
if (!out.legsMatchLine) {
  throw new Error("the legs and the plain line disagree — the census drifted");
}
if (!out.zeroTableLegRides) throw new Error("the zero-count table leg vanished");
if (!out.noImagesLegWithoutImages) throw new Error("an images leg rode without images");
if (!out.metricsNeverALeg) throw new Error("a metrics leg appeared — impossible");

writeFileSync("/tmp/sr_ts_139.json", JSON.stringify(out, null, 2));
console.log(
  "PROBE139_OK — legs:",
  JSON.stringify(legs),
  "confession:",
  out.driftedConfession === CONFESSION ? "pinned" : "DRIFTED",
);
