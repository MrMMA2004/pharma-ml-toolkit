/**
 * 100x-131 — the session report's SECTION CONTROL + SURGERY feed,
 * TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and
 * pins the round's two collector additions over a FIXED fixture:
 *
 *  1. the surgery walk — a topology run in the snapshot adds the
 *     metrics card + the SMILES table + the structure-grid image
 *     section (kind "image", the bridge's new full-bleed PNG embed);
 *  2. the per-group options — `{ pkpd: false }` drops the two PK/PD
 *     figures without touching anything else; `{ surgery: false }`
 *     drops the three surgery sections; a surgery-only report ships
 *     exactly the three surgery sections;
 *  3. sessionGroupsAvailable — the honest source the UI's toggle
 *     chips render from (a fresh session shows only pkpd).
 *
 * Emits /tmp/sr_payload_131.json (the all-on payload WITH surgery —
 * the Python side pipes it through scripts/report_bridge.py and
 * asserts the image section composes into a real embedded figure)
 * and /tmp/sr_ts_131.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_131.ts
 */
import { writeFileSync } from "fs";
import {
  collectSessionReport,
  sessionGroupsAvailable,
  sessionReportCensus,
  type SessionSnapshot,
} from "../src/lib/session-report";

/* A tiny valid PNG data URI (a 72x36 solid swatch, matplotlib-rendered
 * at 110 dpi — the dpi figure_to_base64 renders at, so the bridge's
 * full-bleed re-composition round trips it 1:1).  The bridge decodes +
 * PNG-magic-checks it, so it must be a real PNG. */
const TINY_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAABCCAYAAADjVADoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAABDrAAAQ6wFQlOh8AAAAq0lEQVR4nO3bsRHCQAwAQb+HXmiANlyt23ADVPMEXAEm8OBgN1aguVGqMeecC8v67wXuQogIESEiRISIEBEiQkSIPH4Zfr+2q/a4xPPYT8+6iAgRISJEhIgQESJCRIgIESEiRISIEBEiQkSICBEhIkSEiBARIkJEiAgRISJEhIgQESJCRIgIESEiRISIEBEiQkSICBEhMrw7frmICBEhIkSEiBARIkJEiAiRDweoC300y07zAAAAAElFTkSuQmCC";

/* -- the fixture: the 127 session (ridge + curve + regimen + NCA +
 * history) PLUS one topology surgery run ------------------------------ */
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
  surgery: {
    mode: "topology",
    source: "C1CCNCC1",
    mutants: ["C1CCCC1", "C1CCCNCC1", "C1CCNC1"],
    summary: "3 distinct one-step ring-size mutant(s) of C1CCNCC1",
    gridPng: TINY_PNG,
    gridPanels: 3,
    elapsedSeconds: 0.4,
    rdkitVersion: "2026.09.1",
  },
  ligands: null,
};

const emptySnapshot: SessionSnapshot = {
  ...snapshot,
  mlResult: null,
  lcResult: null,
  mlHistory: [],
  ncaState: { rows: [], dose: "250", f: "1", terminalN: 3 },
  surgery: null,
  ligands: null,
};

const allOn = collectSessionReport(snapshot);
const allOnCensus = sessionReportCensus(snapshot);
const noPkpd = sessionReportCensus(snapshot, { pkpd: false });
const noSurgery = sessionReportCensus(snapshot, { surgery: false });
const surgeryOnly = collectSessionReport(snapshot, {
  ml: false,
  learningCurve: false,
  pkpd: false,
  nca: false,
  history: false,
});
const surgeryOnlyCensus = sessionReportCensus(snapshot, {
  ml: false,
  learningCurve: false,
  pkpd: false,
  nca: false,
  history: false,
});

writeFileSync(
  "/tmp/sr_payload_131.json",
  JSON.stringify({ ...allOn, theme: "light" }),
);
writeFileSync(
  "/tmp/sr_ts_131.json",
  JSON.stringify({
    allOnCensus,
    noPkpd,
    noSurgery,
    surgeryOnlyCensus,
    available: sessionGroupsAvailable(snapshot),
    availableEmpty: sessionGroupsAvailable(emptySnapshot),
    surgeryTitles: allOn.sections
      .filter((s) => s.kind === "image" || s.title.startsWith("Ring topology") || s.title.startsWith("Surgery"))
      .map((s) => s.title),
    surgeryOnlyTitles: surgeryOnly.sections.map((s) => s.title),
    imageSection: allOn.sections.find((s) => s.kind === "image") ?? null,
    mutantRows:
      allOn.sections.find((s) => s.title === "Ring topology mutants")?.kind ===
      "table"
        ? (
            allOn.sections.find(
              (s) => s.title === "Ring topology mutants",
            ) as { rows: Array<Array<string | number>> }
          ).rows
        : [],
  }),
);
console.log("SR_TS_131_OK", JSON.stringify({
  allOnCensus,
  noPkpd,
  noSurgery,
  surgeryOnlyCensus,
}));
