/**
 * Shared ML dataset resolution — the single source of the "which rows,
 * which features, which split seed" contract for the tune and compare
 * routes. Extracted from the tune route (which itself mirrored the train
 * route) so a third consumer could join without a third copy of the
 * CSV→matrix / generator→points plumbing.
 *
 * The train route deliberately keeps its own resolution: its legacy
 * single-feature path (linear/polynomial/ridge) keeps its own call shape,
 * and its csvData is consumed by the run* functions differently.
 *
 * LEAKAGE RULE (100x-101): resolved rows are in RAW feature units. The
 * model fitters (ml-math.ts / ml-models.ts) standardize on the statistics
 * of the rows they are GIVEN — the train split or the CV fold's train
 * part — never on the full column. Scaling here would fold held-out rows
 * into the kNN metric / RBF kernel geometry: train/test leakage.
 */

import {
  hashCsvSeed,
  numericColumnNames,
  parseCsv,
  resolveClassification,
  resolveMultiRegression,
  resolveRegression,
} from "@/lib/csv";
import { makeMoons, makeRegressionData, mulberry32 } from "@/lib/ml-math";
import type { CsvSourceMeta } from "@/lib/toolkit-types";

/** The dataset half of a train/tune/compare request (all CSV fields
 *  optional — the seeded generator fills in when `csv` is absent). */
export interface DatasetRequest {
  task: "regression" | "classification";
  samples?: number;
  noise?: number;
  seed: number;
  csv?: string;
  csvName?: string;
  /* Regression: multi-feature picks (≥2 → the multivariate matrix). */
  featureColumns?: string[];
  featureColumn?: string;
  /* Classification: the two feature-column picks. */
  featureColumn2?: string;
  targetColumn?: string;
}

/** A resolved dataset: row/target accessors over n rows plus the
 *  display labels, provenance and the content-derived split seed. */
export interface ResolvedMlDataset {
  task: "regression" | "classification";
  n: number;
  row(i: number): number[];
  target(i: number): number;
  /** Display labels (raw column names — features stay in raw units). */
  features: string[];
  /** Raw column names (feature-importance keys). */
  featureNames: string[];
  source?: CsvSourceMeta;
  /** The effective split seed (req.seed, or the CSV-content hash). */
  splitSeed: number;
}

/** Either a resolved dataset or an actionable 400 payload. */
export type DatasetResult =
  | { ok: true; data: ResolvedMlDataset }
  | { ok: false; error: string; detail?: string[] };

/**
 * Resolve a request into the shared row/target accessor pair.
 *
 * Rules (identical to the routes this was extracted from):
 * - `csv` present → parse + resolve columns (multivariate regression when
 *   ≥2 feature picks or auto with ≥3 numeric candidates; classification
 *   always takes two feature columns). Same file → same split via the
 *   content hash seed.
 * - otherwise → the seeded generators (makeRegressionData / makeMoons).
 * - bad files/picks surface as `{ ok: false }` with an actionable message.
 */
export function resolveMlDataset(req: DatasetRequest): DatasetResult {
  let rows: number[][] | null = null;
  let ys: number[] = [];
  let pts: Array<{ x: number; y: number; label: number }> = [];
  let source: CsvSourceMeta | undefined;
  let features: string[] = [];
  let featureNames: string[] = [];
  let splitSeed = req.seed;

  if (req.csv !== undefined) {
    const table = parseCsv(req.csv);
    if (table.rows.length === 0) {
      return { ok: false, error: "The uploaded CSV has no data rows below the header." };
    }
    try {
      if (req.task === "regression") {
        const multi =
          (req.featureColumns?.length ?? 0) >= 2 ||
          (req.featureColumns === undefined &&
            !req.featureColumn &&
            numericColumnNames(table).length >= 3);
        if (multi) {
          const csv = resolveMultiRegression(table, req.featureColumns, req.targetColumn);
          rows = csv.X;
          ys = csv.y;
          features = csv.features;
          featureNames = csv.features;
          source = {
            kind: "csv",
            fileName: req.csvName,
            columns: csv.columns,
            featureColumns: csv.features,
            targetColumn: csv.target,
            nRows: csv.y.length,
            droppedRows: csv.droppedRows,
          };
        } else {
          const csv = resolveRegression(
            table,
            req.featureColumn ?? req.featureColumns?.[0],
            req.targetColumn
          );
          /* Raw units — the fitters standardize on the rows they are
           * given (the train folds / train split), never on the full
           * column. Pre-zscoring here was train/test leakage (the rar's
           * signature bug, audited out). */
          rows = csv.x.map((x) => [x]);
          ys = csv.y; // target kept in original units
          features = [csv.feature];
          featureNames = [csv.feature];
          source = {
            kind: "csv",
            fileName: req.csvName,
            columns: csv.columns,
            featureColumns: [csv.feature],
            targetColumn: csv.target,
            nRows: csv.x.length,
            droppedRows: csv.droppedRows,
          };
        }
      } else {
        const csv = resolveClassification(
          table,
          req.featureColumn,
          req.featureColumn2,
          req.targetColumn
        );
        /* Raw units — same leakage rule as the regression paths: the
         * fitters standardize on their train rows internally. */
        const { x1, x2, label } = csv;
        pts = x1.map((x, i) => ({ x, y: x2[i] ?? 0, label: label[i] ?? 0 }));
        features = [csv.feature1, csv.feature2];
        featureNames = [csv.feature1, csv.feature2];
        source = {
          kind: "csv",
          fileName: req.csvName,
          columns: csv.columns,
          featureColumns: [csv.feature1, csv.feature2],
          targetColumn: csv.target,
          nRows: csv.x1.length,
          droppedRows: csv.droppedRows,
          labelMap: csv.labelMap,
        };
      }
    } catch (err) {
      return {
        ok: false,
        error: "Could not use the uploaded CSV.",
        detail: [err instanceof Error ? err.message : String(err)],
      };
    }
    /* Same file → same split: derive the split seed from the content. */
    splitSeed = hashCsvSeed(req.csv);
  } else if (req.task === "regression") {
    const rand = mulberry32(req.seed);
    const { xs, ys: y } = makeRegressionData(req.samples ?? 300, req.noise ?? 0.4, rand);
    rows = xs.map((x) => [x]);
    ys = y;
    features = ["x"];
    featureNames = ["x"];
  } else {
    const rand = mulberry32(req.seed);
    pts = makeMoons(req.samples ?? 300, req.noise ?? 0.4, rand);
    features = ["x1", "x2"];
    featureNames = ["x1", "x2"];
  }

  const row = (i: number): number[] =>
    rows
      ? (rows[i] ?? [])
      : [pts[i]?.x ?? 0, pts[i]?.y ?? 0];
  const target = (i: number): number =>
    rows ? (ys[i] ?? 0) : (pts[i]?.label ?? 0);
  const n = rows ? ys.length : pts.length;

  return {
    ok: true,
    data: { task: req.task, n, row, target, features, featureNames, source, splitSeed },
  };
}
