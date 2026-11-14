/**
 * 100x-127 — session-report CONTRACT parity guard, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (the module the
 * SessionReport card's collector lives in, not a mirror) and emits the
 * twin JSON (/tmp/sr_ts_127.json): the census over a FIXED fixture
 * snapshot plus the full request payload
 * (/tmp/sr_payload_127.json) the Python side pipes through the repo
 * root's scripts/report_bridge.py — the suite's
 * TestSessionReportContract asserts the round trip: the collector's
 * payload composes into a REAL pharmaml.viz.HTMLReport (theme baked,
 * toggle present, figures embedded, self-contained).
 *
 * The fixture mirrors the store's defaults where they matter (PK/PD
 * regimen, emax model) so the census is the one a fresh session with
 * one trained model + one learning curve + the default NCA rows
 * produces: 4 figures (parity, learning curve, PK, PD), 2 tables
 * (NCA rows, run history), 2 metric cards.
 *
 * Run: bun run scratch/probe_session_report_127.ts
 */
import { writeFileSync } from "fs";
import {
  collectSessionReport,
  sessionReportCensus,
  type SessionSnapshot,
} from "../src/lib/session-report";

/* -- the fixture (a fresh session after: train ridge, plot the curve) -- */
const snapshot: SessionSnapshot = {
  mlResult: {
    task: "regression",
    model: "ridge",
    metrics: { "R²": 0.9123, RMSE: 0.2111 },
    series: {
      test: [
        { x: 0.1, y: 0.12 },
        { x: 0.5, y: 0.55 },
        { x: 0.9, y: 0.88 },
      ],
      train: [{ x: 0.2, y: 0.18 }],
      curve: [
        { x: 0, yhat: 0.01, lo: -0.05, hi: 0.07 },
        { x: 1, yhat: 1.02, lo: 0.96, hi: 1.08 },
      ],
    },
    datasetMeta: {
      nTotal: 20,
      nTrain: 16,
      nTest: 4,
      noise: 0.1,
      seed: 42,
      features: ["x"],
    },
  },
  lcResult: {
    task: "regression",
    model: "ridge",
    metricName: "R²",
    folds: 5,
    fractions: [0.1, 0.5, 1.0],
    points: [
      { size: 2, trainMean: 0.5, trainStd: 0.05, valMean: 0.45, valStd: 0.04 },
      { size: 8, trainMean: 0.7, trainStd: 0.03, valMean: 0.66, valStd: 0.03 },
      { size: 16, trainMean: 0.91, trainStd: 0.02, valMean: 0.88, valStd: 0.02 },
    ],
    gap: 0.03,
    verdict: "curves converging — additional data has diminishing returns",
    datasetMeta: { nTotal: 20, nFeatures: 1, kind: "synthetic" },
  },
  mlHistory: [
    {
      id: 1730000000000,
      at: "2026-10-24T10:00:00.000Z",
      task: "regression",
      model: "ridge",
      keyMetric: "R²",
      keyValue: 0.9123,
      config: {} as never, // unused by the collector's table
      result: {} as never, // unused by the collector's table
    },
  ],
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
  ncaState: {
    rows: [
      { id: 1, t: "0.5", c: "3.2" },
      { id: 2, t: "1", c: "2.1" },
    ],
    dose: "250",
    f: "1",
    terminalN: 3,
  },
  // (100x-131) the collector's snapshot gained the surgery slot; this
  // legacy fixture pins the NO-surgery walk (the 127 contract: same
  // census as before) — the surgery-on round trip lives in
  // scratch/probe_session_report_131.ts.
  surgery: null,
  ligands: null,
};

const payload = collectSessionReport(snapshot);
const census = sessionReportCensus(snapshot);
const theme = "dark";
writeFileSync("/tmp/sr_payload_127.json", JSON.stringify({ ...payload, theme }));
writeFileSync(
  "/tmp/sr_ts_127.json",
  JSON.stringify({
    census,
    sectionTitles: payload.sections.map((s) => s.title),
    firstMetrics: payload.sections.find((s) => s.kind === "metrics")
      ?.items ?? [],
    parityTraceStyles: (
      payload.sections.find(
        (s) => s.kind === "figure" && s.title === "Predicted vs observed",
      ) as { traces: Array<{ style: string; name: string }> } | undefined
    )?.traces.map((t) => `${t.name}:${t.style}`) ?? [],
  }),
);
console.log("SR_TS_127_OK", JSON.stringify(census));
