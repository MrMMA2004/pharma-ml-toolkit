/**
 * Session report collector (100x-127) — the web twin of the desktop
 * playground's `File ▸ Save session report (HTML)…`.
 *
 * The desktop report walks its live session (every figure on screen +
 * every filled table) into pharmaml.viz.HTMLReport.  The web twin's
 * "session" is the persisted playground state: the last ML training
 * result (metrics + chart series), the last learning-curve run, the
 * PK/PD regimen (always live — the regimen card always renders), the
 * NCA observed-data rows and the ML run history.  This module is the
 * PURE collector: it turns that state into the section payload the
 * `/api/report/session` route forwards to the Python bridge, which
 * composes real matplotlib figures and the SAME self-contained themed
 * HTMLReport the desktop saves (theme toggle included).
 *
 * Trace contract (mirrors what the web charts draw):
 *  - `line`   → solid polyline (model curve / PK profile)
 *  - `band`   → mean line + ±1 std fill (learning curve, GP band)
 *  - `scatter` with `classes` → class-coloured dots (classification
 *    test points / boundary grid), `probas` fades opacity by
 *    |p − 0.5| exactly the way the web boundary dots do
 */

import { concentrationAt, derivePk, pdEffect } from "@/lib/pk-pd";
import type {
  MLLearningCurveResponse,
  MLTrainResponse,
  SurgeryResponse,
} from "@/lib/toolkit-types";
import type {
  MlRunRecord,
  NcaState,
  PdParams,
  PkParams,
} from "@/lib/store";

/* ------------------------------------------------------------------ */
/* Payload types — the bridge contract                                 */
/* ------------------------------------------------------------------ */

export type ReportTheme = "auto" | "light" | "dark";

/** One drawable series inside a figure section. */
export interface ReportFigureTrace {
  name: string;
  style: "line" | "band" | "scatter";
  /**
   * `[x, y]` pairs; `band` traces carry `[x, y, lo, hi]` (mean ± band
   * edges).  Numbers stay raw — the bridge's matplotlib does the
   * formatting, exactly like the desktop figures.
   */
  points: Array<[number, number] | [number, number, number, number]>;
  /** Per-point class (0/1) for classification scatters — color split. */
  classes?: number[];
  /** Per-point class-1 probability — opacity = |p − 0.5| × 2. */
  probas?: number[];
}

export interface ReportFigureSection {
  kind: "figure";
  title: string;
  xLabel: string;
  yLabel: string;
  traces: ReportFigureTrace[];
}

export interface ReportTableSection {
  kind: "table";
  title: string;
  columns: string[];
  rows: Array<Array<string | number>>;
}

/** (100x-131) A pre-rasterised PNG section — the surgery structure
 * grids arrive from the server bridge as data URIs; the report bridge
 * decodes them onto a full-bleed matplotlib figure and embeds it through
 * the same `add_figure` contract every other figure rides. */
export interface ReportImageSection {
  kind: "image";
  title: string;
  dataUri: string;
}

export interface ReportMetricsSection {
  kind: "metrics";
  title: string;
  /** Metric cards — string values (the bridge escapes them). */
  items: Array<[string, string]>;
}

export type ReportSection =
  | ReportFigureSection
  | ReportTableSection
  | ReportMetricsSection
  | ReportImageSection;

/** POST /api/report/session request body. */
export interface SessionReportRequest {
  theme: ReportTheme;
  subtitle: string;
  sections: ReportSection[];
}

/** POST /api/report/session response (the bridge's census). */
export interface SessionReportResponse {
  html: string;
  figures: number;
  tables: number;
  bytes: number;
  theme: string;
}

/** The persisted playground state the collector walks. */
export interface SessionSnapshot {
  mlResult: MLTrainResponse | null;
  lcResult: MLLearningCurveResponse | null;
  mlHistory: MlRunRecord[];
  pkParams: PkParams;
  pdParams: PdParams;
  ncaState: NcaState;
  /** (100x-131) the last molecule-surgery run (topology or breed — the
   *  latest run wins, exactly like mlResult); null = no surgery yet. */
  surgery: SurgeryResponse | null;
}

/** (100x-131) Per-group section control — the report card's toggles.
 *  Every group defaults to shipped; the collector silently skips a
 *  group the snapshot has no content for either way, so the toggle set
 *  the UI renders comes from `sessionGroupsAvailable`. */
export interface SessionReportOptions {
  ml?: boolean;
  learningCurve?: boolean;
  pkpd?: boolean;
  nca?: boolean;
  history?: boolean;
  surgery?: boolean;
}

/** Which groups would actually ship something for this snapshot — the
 * honest source for the UI: a group with no content renders no toggle
 * (nothing to skip), so a fresh session shows only "PK / PD" and the
 * buttons appear as the session fills. */
export function sessionGroupsAvailable(
  snapshot: SessionSnapshot,
): Record<keyof SessionReportOptions, boolean> {
  const rows = snapshot.ncaState.rows.filter(
    (r) => r.t.trim() !== "" && r.c.trim() !== "",
  );
  return {
    ml: snapshot.mlResult !== null,
    learningCurve: snapshot.lcResult !== null,
    pkpd: true, // the regimen card is always live in the web twin
    nca: rows.length > 0,
    history: snapshot.mlHistory.length > 0,
    surgery: snapshot.surgery !== null,
  };
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Compact number formatting for table cells (≤ 4 significant digits,
 *  trailing zeros trimmed — the fmtNum convention of the web cards). */
export function fmtCell(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001) return value.toExponential(3);
  const fixed = abs >= 100 ? 1 : abs >= 1 ? 3 : 4;
  return String(Number(value.toFixed(fixed)));
}

/** The PK concentration series the regimen card draws (same sampling:
 *  3 half-lives or 2 intervals of tail, 420 samples + the endpoints). */
function pkSeries(params: PkParams): Array<[number, number]> {
  const derived = derivePk(params);
  const tail = Math.max(3 * derived.halfLife, 2 * params.tau);
  const total = (params.doses - 1) * params.tau + tail;
  const samples = 420;
  const step = total / samples;
  const points: Array<[number, number]> = [];
  for (let i = 0; i <= samples; i += 1) {
    const t = i * step;
    points.push([Number(t.toFixed(2)), concentrationAt(t, params, derived.ke)]);
  }
  return points;
}

/* ------------------------------------------------------------------ */
/* The collector                                                       */
/* ------------------------------------------------------------------ */

/** Build the report payload from the session snapshot.  Pure: no DOM,
 *  no store subscription — the UI component reads the store and hands
 *  the pieces in, so the collector is trivially unit-testable.
 *
 * Sections, in walking order (the desktop report walks its cards the
 * same way — models before curves, tables last):
 *
 *  1. ML training — metrics card + the parity/decision figure
 *     (only when a result exists)
 *  2. Learning curve — metrics + the mean ± std band figure
 *     (only when a curve was computed)
 *  3. PK regimen — the concentration profile (the regimen card is
 *     always live in the web twin) + the PD effect overlay figure
 *  4. NCA observed data — the rows table (only when filled)
 *  5. ML run history — the runs table (only when a run exists)
 *  6. Molecule surgery (100x-131) — the run's metrics + SMILES table
 *     + the structure-grid image (only when a run is in the session)
 *
 * (100x-131) `options` turns whole groups off — the report card's
 * per-section toggles: skip the PK figures when only the ML run
 * matters, ship only the surgery grid, whatever the session needs.
 * A group with no content is skipped regardless, so the census the
 * UI shows always matches what would ship.
 */
export function collectSessionReport(
  snapshot: SessionSnapshot,
  options: SessionReportOptions = {},
): SessionReportRequest {
  const enabled = {
    ml: options.ml !== false,
    learningCurve: options.learningCurve !== false,
    pkpd: options.pkpd !== false,
    nca: options.nca !== false,
    history: options.history !== false,
    surgery: options.surgery !== false,
  };
  const sections: ReportSection[] = [];

  /* -- 1. ML training ------------------------------------------------ */
  const ml = snapshot.mlResult;
  if (ml && enabled.ml) {
    const metrics: Array<[string, string]> = [
      ["model", ml.model],
      ["task", ml.task],
    ];
    for (const [key, value] of Object.entries(ml.metrics)) {
      metrics.push([key, fmtCell(value)]);
    }
    metrics.push(["rows", `${ml.datasetMeta.nTotal}`]);
    metrics.push(["features", String(ml.datasetMeta.features.length)]);
    if (ml.datasetMeta.source) {
      metrics.push(["source", ml.datasetMeta.source.fileName ?? "csv"]);
    }
    sections.push({ kind: "metrics", title: "ML training", items: metrics });

    if (ml.task === "regression") {
      const traces: ReportFigureTrace[] = [
        {
          name: "model prediction",
          style: "line",
          points: (ml.series.curve ?? []).map((p) => [p.x, p.yhat] as [number, number]),
        },
        {
          name: "test observations",
          style: "scatter",
          points: ml.series.test.map((p) => [p.x, p.y] as [number, number]),
        },
      ];
      if (ml.series.train && ml.series.train.length > 0) {
        traces.push({
          name: "train observations",
          style: "scatter",
          points: ml.series.train.map((p) => [p.x, p.y] as [number, number]),
        });
      }
      const bandTrace = bandFromCurve(ml.series.curve ?? []);
      if (bandTrace) traces.splice(1, 0, bandTrace);
      sections.push({
        kind: "figure",
        title: "Predicted vs observed",
        xLabel: "observed",
        yLabel: "predicted",
        traces,
      });
    } else {
      const traces: ReportFigureTrace[] = [];
      if (ml.series.boundary && ml.series.boundary.length > 0) {
        traces.push({
          name: "decision regions",
          style: "scatter",
          points: ml.series.boundary.map(
            (p) => [p.x, p.y] as [number, number],
          ),
          classes: ml.series.boundary.map((p) => Math.round(p.pred)),
          probas: ml.series.boundary.map((p) => p.proba ?? Math.round(p.pred)),
        });
      }
      traces.push({
        name: "test observations",
        style: "scatter",
        points: ml.series.test.map((p) => [p.x, p.y] as [number, number]),
        classes: ml.series.test.map((p) => Math.round(p.label ?? 0)),
      });
      sections.push({
        kind: "figure",
        title: "Decision map",
        xLabel: "feature 1",
        yLabel: "feature 2",
        traces,
      });
    }
  }

  /* -- 2. Learning curve ---------------------------------------------- */
  const lc = snapshot.lcResult;
  if (lc && enabled.learningCurve) {
    sections.push({
      kind: "metrics",
      title: "Learning curve",
      items: [
        ["model", lc.model],
        ["metric", lc.metricName],
        ["folds", `${lc.folds}`],
        ["gap", fmtCell(lc.gap)],
        ["dataset", lc.datasetMeta.kind === "csv"
          ? (lc.datasetMeta.fileName ?? "csv")
          : "synthetic"],
      ],
    });
    sections.push({
      kind: "figure",
      title: `Learning curve — ${lc.metricName} vs training samples`,
      xLabel: "training samples",
      yLabel: lc.metricName,
      traces: [
        {
          name: `training ${lc.metricName}`,
          style: "band",
          points: lc.points.map(
            (p) => [p.size, p.trainMean, p.trainMean - p.trainStd, p.trainMean + p.trainStd] as [number, number, number, number],
          ),
        },
        {
          name: `CV ${lc.metricName}`,
          style: "band",
          points: lc.points.map(
            (p) => [p.size, p.valMean, p.valMean - p.valStd, p.valMean + p.valStd] as [number, number, number, number],
          ),
        },
      ],
    });
  }

  /* -- 3. PK regimen (always live) ------------------------------------ */
  if (enabled.pkpd) {
    const conc = pkSeries(snapshot.pkParams);
    sections.push({
      kind: "figure",
      title: "PK regimen — concentration vs time",
      xLabel: "time (h)",
      yLabel: "concentration (mg/L)",
      traces: [
        {
          name: `${snapshot.pkParams.route === "iv" ? "IV bolus" : "oral"} regimen`,
          style: "line",
          points: conc,
        },
      ],
    });
    const effect = conc.map(
      ([t, c]) =>
        [t, pdEffect(c, snapshot.pdParams.model, snapshot.pdParams)] as [number, number],
    );
    sections.push({
      kind: "figure",
      title: `PD effect — ${snapshot.pdParams.model} model`,
      xLabel: "time (h)",
      yLabel: "effect",
      traces: [
        { name: "effect", style: "line", points: effect },
      ],
    });
  }

  /* -- 4. NCA observed data ------------------------------------------- */
  const rows = snapshot.ncaState.rows.filter((r) => r.t.trim() !== "" && r.c.trim() !== "");
  if (rows.length > 0 && enabled.nca) {
    sections.push({
      kind: "table",
      title: "NCA observed data",
      columns: ["#", "time (h)", "concentration (mg/L)"],
      rows: rows.map((r, i) => [i + 1, r.t, r.c]),
    });
  }

  /* -- 5. ML run history ---------------------------------------------- */
  if (snapshot.mlHistory.length > 0 && enabled.history) {
    sections.push({
      kind: "table",
      title: "ML run history",
      columns: ["when", "task", "model", "metric"],
      rows: snapshot.mlHistory.map((run) => [
        new Date(run.at).toLocaleString(),
        run.task,
        run.model,
        `${run.keyMetric} ${fmtCell(run.keyValue)}`,
      ]),
    });
  }

  /* -- 6. Molecule surgery (100x-131) ---------------------------------- */
  const surgery = snapshot.surgery;
  if (surgery && enabled.surgery) {
    if (surgery.mode === "topology") {
      sections.push({
        kind: "metrics",
        title: "Ring topology moves",
        items: [
          ["source", surgery.source],
          ["distinct mutants", `${surgery.mutants.length}`],
          ["compute time", `${fmtCell(surgery.elapsedSeconds)} s`],
          ["RDKit", surgery.rdkitVersion],
        ],
      });
      sections.push({
        kind: "table",
        title: "Ring topology mutants",
        columns: ["#", "mutant SMILES"],
        rows: surgery.mutants.map((mutant, i) => [i + 1, mutant]),
      });
      sections.push({
        kind: "image",
        title: `Ring topology mutants — ${surgery.mutants.length} structure panel(s)`,
        dataUri: surgery.gridPng,
      });
    } else {
      sections.push({
        kind: "metrics",
        title: "Scaffold-hop crossover",
        items: [
          ["parent A", surgery.parentA],
          ["parent B", surgery.parentB],
          ["viable children", `${surgery.viable}/${surgery.children}`],
          ["seed", `${surgery.seed}`],
          ["compute time", `${fmtCell(surgery.elapsedSeconds)} s`],
          ["RDKit", surgery.rdkitVersion],
        ],
      });
      sections.push({
        kind: "table",
        title: "Crossover litter",
        columns: ["draw", "child SMILES"],
        rows: surgery.litter.map((child, i) => [
          i + 1,
          child ?? "(no viable crossover)",
        ]),
      });
      sections.push({
        kind: "image",
        title: `Parents & children — ${surgery.gridPanels} structure panel(s)`,
        dataUri: surgery.gridPng,
      });
    }
  }

  return {
    theme: "auto",
    subtitle: "web playground session",
    sections,
  };
}

/** Turn a curve with optional lo/hi into a band trace (GP-style). */
function bandFromCurve(
  curve: Array<{ x: number; yhat: number; lo?: number; hi?: number }>,
): ReportFigureTrace | null {
  const banded = curve.filter((p) => p.lo !== undefined && p.hi !== undefined);
  if (banded.length < 2) return null;
  return {
    name: "± band",
    style: "band",
    points: banded.map(
      (p) => [p.x, p.yhat, p.lo as number, p.hi as number] as [number, number, number, number],
    ),
  };
}

/** The live census for the UI button ("what will ship"): figure and
 *  table counts over the same walk the collector performs — image
 *  sections count separately (the surgery grids) so the button line
 *  can say "2 figures · 1 structure grid" honestly. */
export function sessionReportCensus(
  snapshot: SessionSnapshot,
  options: SessionReportOptions = {},
): {
  figures: number;
  tables: number;
  metrics: number;
  images: number;
} {
  const { sections } = collectSessionReport(snapshot, options);
  let figures = 0;
  let tables = 0;
  let metrics = 0;
  let images = 0;
  for (const section of sections) {
    if (section.kind === "figure") figures += 1;
    else if (section.kind === "table") tables += 1;
    else if (section.kind === "image") images += 1;
    else metrics += 1;
  }
  return { figures, tables, metrics, images };
}
