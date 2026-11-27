/**
 * part21 §3.1 — the shared evaluation-visualization layer.
 *
 * The extracted-and-extended home of the ML workspace's proven
 * evaluation charts (the parity scatter, the confusion-matrix heatmap,
 * the click-through sample inspector — §22.5's identity contract) plus
 * the part21 additions driven by the backend's own sample-derived
 * diagnostics (the residual histogram, the normal Q–Q points, the
 * per-class metrics table, the OvR ROC/PR curves, the binary
 * reliability diagram). BOTH the ML and the DL workspaces render
 * this one component over the SAME `PredictResponse` contract —
 * `POST /api/v1/ml/predict`'s dataset-mode evaluation block — so no
 * duplicate parity/confusion implementation survives anywhere.
 *
 * The scientific rails (part21 §4): the component decides what to
 * render from the task type and the fields the backend actually
 * attached; a missing diagnostic renders an explanatory state (never
 * a fabricated curve); graph/sequence/autoencoder models never flow
 * through this contract at all (the route 422s them to their own
 * lanes); and every chart's data is the evaluation SAMPLE the parity
 * lane already plots, with the sample bound reported on the surface.
 */
import { useMemo, useState } from "react";

import type {
  EvaluationBlock,
  EvaluationRow,
  PredictResponse,
} from "@/services/ml/client";

import { CalibrationChart } from "./CalibrationChart";
import { ClassMetricsTable } from "./ClassMetricsTable";
import { ConfusionChart } from "./ConfusionChart";
import { EvalRowInspector } from "./EvalRowInspector";
import { ParityChart } from "./ParityChart";
import { PrCurveChart } from "./PrCurveChart";
import { QqPlotChart } from "./QqPlotChart";
import { ResidualHistogramChart } from "./ResidualHistogramChart";
import { RocCurveChart } from "./RocCurveChart";

export { formatEvalValue as formatValue } from "./formatValue";
export {
  buildEvaluationExportRows,
  buildPredictionExportRows,
} from "./exportRows";

export interface EvaluationChartsProps {
  readonly result: PredictResponse;
  readonly modelId: string;
  readonly datasetId: string | null;
}

/** One bordered chart panel with its own caption line. */
function ChartPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
      <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
        {title}
      </p>
      {children}
    </div>
  );
}

/**
 * The shared ML/DL evaluation visualization (part21 §3.2).
 *
 * Regression answers render the parity scatter plus — when the
 * backend attached them — the residual histogram, the Q–Q points,
 * and the residual summary line. Classification answers render the
 * confusion matrix (raw/normalized) plus the per-class metrics table
 * and, when the pass carried probabilities, the OvR ROC/PR curves
 * and the binary reliability diagram.
 */
export function EvaluationCharts({
  result,
  modelId,
  datasetId,
}: EvaluationChartsProps) {
  const evaluation: EvaluationBlock | null = result.evaluation;
  const [selected, setSelected] = useState<EvaluationRow | null>(null);
  const numericRows = useMemo(
    () =>
      evaluation === null
        ? []
        : evaluation.rows.filter(
            (row) =>
              typeof row.actual === "number" &&
              typeof row.predicted === "number",
          ),
    [evaluation],
  );

  if (evaluation === null) {
    return (
      <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
        No evaluation block in this answer — inline-row predictions carry
        no aligned actuals to chart (the dataset-mode lane plots them).
      </p>
    );
  }

  const isRegression =
    evaluation.task === "regression" && numericRows.length >= 2;
  const hasClasses =
    evaluation.task === "classification" && evaluation.rows.length > 0;
  const boundNote =
    evaluation.n_rows_available !== evaluation.n_rows_evaluated
      ? ` · ${evaluation.n_rows_evaluated} of ${evaluation.n_rows_available} rows plotted (the route's own sample bound)`
      : ` · ${evaluation.n_rows_evaluated} rows plotted`;

  return (
    <div className="mt-4" data-testid="evaluation-charts">
      <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
        Shared evaluation charts (Plotly) — {isRegression
          ? "regression diagnostics"
          : hasClasses
            ? "classification diagnostics"
            : "no plottable pairs"}
        {boundNote}
      </p>
      {isRegression ? (
        <>
          <ParityChart rows={numericRows} onRowSelect={setSelected} />
          {evaluation.residual_diagnostics !== null ? (
            <>
              <ChartPanel title="Residual distribution (the sample's own residuals)">
                <ResidualHistogramChart
                  diagnostics={evaluation.residual_diagnostics}
                />
              </ChartPanel>
              {evaluation.residual_diagnostics.qq !== null ? (
                <ChartPanel title="Normal Q–Q of the residuals">
                  <QqPlotChart
                    qq={evaluation.residual_diagnostics.qq}
                  />
                </ChartPanel>
              ) : null}
              <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
                Residual summary — mean{" "}
                {evaluation.residual_diagnostics.stats.mean.toFixed(4)} ·
                median{" "}
                {evaluation.residual_diagnostics.stats.median.toFixed(4)} ·
                std {evaluation.residual_diagnostics.stats.std.toFixed(4)} ·
                range [{evaluation.residual_diagnostics.stats.min.toFixed(4)},{" "}
                {evaluation.residual_diagnostics.stats.max.toFixed(4)}] over{" "}
                {evaluation.residual_diagnostics.stats.n} numeric residuals
              </p>
            </>
          ) : null}
        </>
      ) : hasClasses ? (
        <>
          <ConfusionChart rows={evaluation.rows} />
          {evaluation.class_metrics !== null ? (
            <ChartPanel title="Per-class metrics over the sample">
              <ClassMetricsTable metrics={evaluation.class_metrics} />
            </ChartPanel>
          ) : null}
          {evaluation.roc_curves !== null ? (
            <ChartPanel title="ROC curves (one-vs-rest, per class)">
              <RocCurveChart curves={evaluation.roc_curves} />
            </ChartPanel>
          ) : null}
          {evaluation.pr_curves !== null ? (
            <ChartPanel title="Precision–Recall curves (one-vs-rest, per class)">
              <PrCurveChart curves={evaluation.pr_curves} />
            </ChartPanel>
          ) : null}
          {evaluation.calibration !== null ? (
            <ChartPanel title="Reliability diagram (positive class)">
              <CalibrationChart calibration={evaluation.calibration} />
            </ChartPanel>
          ) : null}
          {evaluation.roc_curves === null ? (
            <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
              No ROC/PR/calibration charts — this answer carried no
              probability block (ask for probabilities on the predict
              call; the curves need the real class scores).
            </p>
          ) : null}
        </>
      ) : (
        <p className="rounded-md border border-dashed border-[var(--color-border)] px-3 py-4 text-xs text-[var(--color-muted-foreground)]">
          No plottable numeric pairs in the evaluation sample — the parity
          lane needs numeric actual/predicted values.
        </p>
      )}
      <EvalRowInspector
        row={selected}
        target={evaluation.target}
        task={evaluation.task}
        modelId={modelId}
        datasetId={datasetId}
        onClear={() => setSelected(null)}
      />
    </div>
  );
}
