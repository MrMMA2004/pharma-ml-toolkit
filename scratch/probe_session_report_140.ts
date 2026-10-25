/**
 * 100x-140 — the census's KEYBOARD path, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's pure derivation:
 *
 *  1. `spotlightKeyAction` — the keyboard twin of clicking a leg:
 *     digits "1"/"2"/"3" toggle the census line's 1st/2nd/3rd leg
 *     (figure / table / structure grid — the digits count the line's
 *     legs, clickable or plain, so the mapping is stable whatever the
 *     session lists), "0" clears, every other key PASSES untouched;
 *  2. the zero-count leg's honest no-op — a kind with nothing to
 *     spotlight (its leg renders as plain text) swallows its digit
 *     without toggling anything;
 *  3. the out-of-range digit passes — a digit beyond the legs the
 *     line lists is not a spotlight key;
 *  4. non-digits always pass (Tab, Escape, Enter, letters, arrows —
 *     the keydown lets them go untouched);
 *  5. the mapping rides the SAME legs the click path consumes
 *     (contentsCensusLegs) — the two paths cannot disagree.
 *
 * Emits /tmp/sr_ts_140.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_140.ts
 */
import { writeFileSync } from "fs";
import {
  contentsCensusLegs,
  sessionContentsCensus,
  sessionReportContents,
  spotlightKeyAction,
  type ReportSection,
} from "../src/lib/session-report";

const out: Record<string, unknown> = {};

/* -- the mixed fixture: every kind rides (legs = figure/table/image) ----- */
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
  {
    kind: "figure",
    title: "PK regimen — concentration vs time",
    xLabel: "time (h)",
    yLabel: "concentration",
    traces: [{ name: "regimen", style: "line", points: [[0, 0], [1, 2]] }],
  },
];
const mixedLegs = contentsCensusLegs(sessionContentsCensus(sessionReportContents(mixed)));
if (mixedLegs.length !== 3) {
  throw new Error(`the mixed fixture lost a leg: ${mixedLegs.length}`);
}

/* -- 1. the digits toggle the census line's legs --------------------------- */
out.digit1 = spotlightKeyAction("1", mixedLegs);
out.digit2 = spotlightKeyAction("2", mixedLegs);
out.digit3 = spotlightKeyAction("3", mixedLegs);
out.digit0 = spotlightKeyAction("0", mixedLegs);

/* -- 2. the zero-count leg's honest no-op ----------------------------------- */
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
out.zeroCountLegDigit = spotlightKeyAction("2", figureLegs); // "0 tables" leg
out.zeroCountFigureDigit = spotlightKeyAction("1", figureLegs); // the real leg

/* -- 3. the out-of-range digit passes --------------------------------------- */
out.outOfRangeDigit = spotlightKeyAction("3", figureLegs); // no images leg
out.farOutOfRangeDigit = spotlightKeyAction("9", mixedLegs);

/* -- 4. non-digits always pass ---------------------------------------------- */
const nonDigits = ["Tab", "Enter", "Escape", " ", "a", "ArrowDown", "F1", "", "10"];
out.nonDigitsPass = nonDigits.map((key) => spotlightKeyAction(key, mixedLegs));
out.allNonDigitsPassed = out.nonDigitsPass.every(
  (verdict) => (verdict as { action: string }).action === "pass",
);

/* -- the honest verdicts ------------------------------------------------------ */
if (out.digit1.action !== "toggle" || out.digit1.kind !== "figure") {
  throw new Error(`digit 1 mis-mapped: ${JSON.stringify(out.digit1)}`);
}
if (out.digit2.action !== "toggle" || out.digit2.kind !== "table") {
  throw new Error(`digit 2 mis-mapped: ${JSON.stringify(out.digit2)}`);
}
if (out.digit3.action !== "toggle" || out.digit3.kind !== "image") {
  throw new Error(`digit 3 mis-mapped: ${JSON.stringify(out.digit3)}`);
}
if (out.digit0.action !== "clear") {
  throw new Error(`digit 0 lost the clear: ${JSON.stringify(out.digit0)}`);
}
if (out.zeroCountLegDigit.action !== "noop") {
  throw new Error(
    `the zero-count leg's digit should be a no-op: ${JSON.stringify(out.zeroCountLegDigit)}`,
  );
}
if (out.zeroCountFigureDigit.action !== "toggle" || out.zeroCountFigureDigit.kind !== "figure") {
  throw new Error("the figure-only leg's digit mis-mapped");
}
if (out.outOfRangeDigit.action !== "pass") {
  throw new Error("a digit beyond the legs should pass");
}
if (out.farOutOfRangeDigit.action !== "pass") {
  throw new Error("a far digit should pass");
}
if (!out.allNonDigitsPassed) {
  throw new Error("a non-digit was hijacked — the keydown owns only the census digits");
}

writeFileSync("/tmp/sr_ts_140.json", JSON.stringify(out, null, 2));
console.log(
  "PROBE140_OK — digits:",
  JSON.stringify([out.digit1, out.digit2, out.digit3, out.digit0]),
  "zero-count no-op:",
  out.zeroCountLegDigit.action,
  "non-digits pass:",
  out.allNonDigitsPassed,
);
