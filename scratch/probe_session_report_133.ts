/**
 * 100x-133 — the session report's PREVIEW, TS side.
 *
 * Imports the SHIPPED src/lib/session-report.ts (not a mirror) and pins
 * the round's preview additions over the SAME fixed 131 fixture the
 * 132 probe rides:
 *
 *  1. figureSketchBounds — the sketch's mapping arithmetic: band edges
 *     (lo/hi) participate in the y bounds, a degenerate (flat) axis
 *     pads itself ±1 so a flat line still draws inside the viewBox,
 *     and a trace set with no finite points falls back to the unit
 *     box;
 *  2. the preview caps — PREVIEW_TABLE_ROWS / PREVIEW_SCATTER_POINTS,
 *     the honest "+N more rows" and stride-sampled dot contracts;
 *  3. the preview's section stream — the collector's sections over a
 *     custom walking order, as (kind, title) pairs: exactly what the
 *     preview dialog renders, in the order it renders them.
 *
 * Emits /tmp/sr_ts_133.json (the assertion data the suite pins).
 *
 * Run: bun run scratch/probe_session_report_133.ts
 */
import { writeFileSync } from "fs";
import {
  collectSessionReport,
  figureSketchBounds,
  PREVIEW_SCATTER_POINTS,
  PREVIEW_TABLE_ROWS,
  type ReportFigureTrace,
  type SessionGroupId,
  type SessionSnapshot,
} from "../src/lib/session-report";

/* The same tiny valid PNG data URI the 132 probe rides (the surgery
 * grid -- the preview renders it as a real <img>). */
const TINY_PNG =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEIAAABCCAYAAADjVADoAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAABDrAAAQ6wFQlOh8AAAAq0lEQVR4nO3bsRHCQAwAQb+HXmiANlyt23ADVPMEXAEm8OBgN1aguVGqMeecC8v67wXuQogIESEiRISIEBEiQkSIPH4Zfr+2q/a4xPPYT8+6iAgRISJEhIgQESJCRIgIESEiRISIEBEiQkSICBEhIkSEiBARIkJEiAgRISJEhIgQESJCRIgIESEiRISIEBEiQkSICBEhMrw7frmICBEhIkSEiBARIkJEiAiRDweoC300y07zAAAAAElFTkSuQmCC";

/* -- the fixture: the 131 session (the 132 probe's own) ----------------- */
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

/* -- 1. the sketch bounds arithmetic ------------------------------------ */
const bandTrace: ReportFigureTrace = {
  name: "band",
  style: "band",
  // y rides 1..3 but the band edges stretch the axis to 0.5..3.5
  points: [
    [0, 1, 0.5, 1.5],
    [2, 3, 2.5, 3.5],
  ],
};
const boundsBand = figureSketchBounds([bandTrace]);
const boundsFlat = figureSketchBounds([
  { name: "flat", style: "line", points: [[1, 5], [1, 5]] },
]);
const boundsEmpty = figureSketchBounds([]);
const boundsScatter = figureSketchBounds([
  {
    name: "dots",
    style: "scatter",
    points: [
      [-1, 2],
      [3, -4],
    ],
  },
]);
// a mixed set: the union of every trace's extents
const boundsMixed = figureSketchBounds([
  bandTrace,
  { name: "line", style: "line", points: [[-2, 10], [4, 0]] },
]);

/* -- 2. the preview caps ------------------------------------------------- */
const previewRows = PREVIEW_TABLE_ROWS;
const previewScatter = PREVIEW_SCATTER_POINTS;

/* -- 3. the preview's section stream -------------------------------------
 * The same custom order the 132 probe rode (surgery first, then ML):
 * the preview dialog renders exactly these sections, in this order. */
const surgeryFirst: SessionGroupId[] = ["surgery", "ml"];
const ordered = collectSessionReport(snapshot, {}, surgeryFirst);
const sectionStream = ordered.sections.map((section) => ({
  kind: section.kind,
  title: section.title,
}));

writeFileSync(
  "/tmp/sr_ts_133.json",
  JSON.stringify({
    boundsBand,
    boundsFlat,
    boundsEmpty,
    boundsScatter,
    boundsMixed,
    previewRows,
    previewScatter,
    sectionStream,
  }),
);
console.log(
  "SR_TS_133_OK",
  JSON.stringify({
    boundsBand,
    boundsFlat,
    boundsEmpty,
    previewRows,
    previewScatter,
    sections: sectionStream.length,
  }),
);
