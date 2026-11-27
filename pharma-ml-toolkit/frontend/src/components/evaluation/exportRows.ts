/**
 * part21 §3.4 — the one export-row builder pair for ML and DL.
 *
 * Both workspaces' prediction exports flow through the SAME
 * `/api/v1/visualization/table` artifact lane with rows built by one
 * shared implementation: the prediction builder (moved from the ML
 * workspace's own copy — no duplicate survives) and the evaluation
 * builder (the aligned actual/predicted/residual/SMILES sample the
 * charts plot). Reuses the backend artifact contract (digest-verified
 * downloads, provenance) exactly as before — the export lane was
 * already unified server-side; this file unifies the client-side row
 * construction.
 */
import type { PredictResponse } from "@/services/ml/client";

/** One row per prediction: the prediction plus probability columns. */
export function buildPredictionExportRows(
  result: PredictResponse,
): { rows: Record<string, unknown>[]; columns: string[] } {
  const columns: string[] = ["prediction"];
  const probabilities = result.probabilities;
  if (probabilities !== null) {
    const used = new Set<string>(columns);
    for (const cls of probabilities.classes) {
      let label = `probability_${String(cls)}`;
      while (used.has(label)) {
        label += "_";
      }
      used.add(label);
      columns.push(label);
    }
  }
  const probLabels = columns.slice(1);
  const rows: Record<string, unknown>[] = [];
  for (const [i, prediction] of result.predictions.entries()) {
    const row: Record<string, unknown> = { prediction };
    if (probabilities !== null) {
      const probRow = probabilities.rows[i];
      if (probRow !== undefined) {
        for (const [j, label] of probLabels.entries()) {
          row[label] = probRow[j] ?? null;
        }
      }
    }
    rows.push(row);
  }
  return { rows, columns };
}

/** One row per evaluation sample: the aligned pair + residual + SMILES. */
export function buildEvaluationExportRows(
  result: PredictResponse,
): { rows: Record<string, unknown>[]; columns: string[] } {
  const evaluation = result.evaluation;
  const columns = ["row", "actual", "predicted", "residual", "smiles"];
  if (evaluation === null) {
    return { rows: [], columns };
  }
  const rows = evaluation.rows.map((row) => ({
    row: row.name ?? "",
    actual: row.actual ?? null,
    predicted: row.predicted ?? null,
    residual: row.residual,
    smiles: row.smiles ?? null,
  }));
  return { rows, columns };
}
