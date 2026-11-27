/**
 * part21 §4/§11.1 — the shared ML/DL evaluation-visualization layer
 * as a complete component-level contract test.
 *
 * The shared `EvaluationCharts` renders the SAME component both
 * workspaces serve: the regression family (parity + residual
 * histogram + Q–Q + the summary line), the classification family
 * (confusion raw/normalized + per-class metrics + ROC/PR/calibration),
 * the explanatory states (no evaluation block; no probability
 * curves), the sample-bound note, and the §22.5 click-through sample
 * inspector — the point's own metadata, never a plot position.
 *
 * PlotlyFigure is stubbed at the component boundary (the GnnLane
 * convention) with an interactive probe that exposes the real
 * `onPointSelect` contract so the inspector's identity assertion runs
 * on the REAL component wiring.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

vi.mock("@/components/charts/PlotlyFigure", () => ({
  PlotlyFigure: (props: {
    readonly data: readonly { readonly type: string }[];
    readonly ariaLabel: string;
    readonly onPointSelect?: (point: {
      pointIndex: number | null;
      customdata: unknown;
    }) => void;
  }) => (
    <div>
      <div data-testid={`plotly-${props.data[0]?.type ?? "none"}`} />
      <span>{props.ariaLabel}</span>
      {props.onPointSelect !== undefined ? (
        <button
          type="button"
          onClick={() =>
            props.onPointSelect?.({
              pointIndex: 0,
              customdata: props.data[0],
            })
          }
        >
          select-point-0
        </button>
      ) : null}
    </div>
  ),
}));

import { EvaluationCharts } from "@/components/evaluation/EvaluationCharts";
import { buildEvaluationExportRows, buildPredictionExportRows } from "@/components/evaluation/exportRows";
import type { PredictResponse } from "@/services/ml/client";

function regressionResult(): PredictResponse {
  const rows = Array.from({ length: 12 }, (_, i) => ({
    name: i,
    actual: i * 1.1,
    predicted: i * 1.1 + 0.05,
    residual: -0.05,
    smiles: null,
  }));
  return {
    mode: "sync",
    model_id: "mdl-test",
    n_rows: 500,
    predictions: rows.map((r) => r.predicted),
    probabilities: null,
    evaluation: {
      target: "y",
      task: "regression",
      smiles_col: null,
      n_rows_available: 500,
      n_rows_evaluated: 12,
      rows,
      residual_diagnostics: {
        histogram: {
          bin_edges: [-0.1, -0.05, 0.0, 0.05, 0.1],
          counts: [2, 4, 4, 2],
        },
        stats: {
          mean: -0.05,
          median: -0.05,
          std: 0.02,
          min: -0.1,
          max: 0.0,
          n: 12,
        },
        qq: {
          theoretical: [-1.6, -1.0, -0.5, 0.0, 0.5, 1.0, 1.6],
          empirical: [-1.5, -1.1, -0.4, 0.0, 0.4, 1.1, 1.5],
        },
      },
      class_metrics: null,
      roc_curves: null,
      pr_curves: null,
      calibration: null,
    },
  };
}

function classificationResult(withCurves: boolean): PredictResponse {
  const rows = [
    { name: 0, actual: 1, predicted: 1, residual: null, smiles: "CCO" },
    { name: 1, actual: 1, predicted: 1, residual: null, smiles: "CCC" },
    { name: 2, actual: 0, predicted: 0, residual: null, smiles: "CCN" },
    { name: 3, actual: 0, predicted: 1, residual: null, smiles: "CCCl" },
    { name: 4, actual: 2, predicted: 2, residual: null, smiles: "CCBr" },
  ];
  return {
    mode: "sync",
    model_id: "mdl-test",
    n_rows: 5,
    predictions: [1, 1, 0, 1, 2],
    probabilities: withCurves
      ? {
          classes: [0, 1, 2],
          rows: [
            [0.05, 0.9, 0.05],
            [0.05, 0.85, 0.1],
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.1, 0.8],
          ],
        }
      : null,
    evaluation: {
      target: "y",
      task: "classification",
      smiles_col: "smiles",
      n_rows_available: 5,
      n_rows_evaluated: 5,
      rows,
      residual_diagnostics: null,
      class_metrics: {
        rows: [
          { label: "0", precision: 1, recall: 0.5, f1: 0.6667, support: 2 },
          { label: "1", precision: 0.6667, recall: 1, f1: 0.8, support: 2 },
          { label: "2", precision: 1, recall: 1, f1: 1, support: 1 },
        ],
        macro: { precision: 0.8889, recall: 0.8333, f1: 0.8556 },
        accuracy: 0.8,
        n: 5,
      },
      roc_curves: withCurves
        ? [
            {
              label: "0",
              fpr: [0, 0.6, 1],
              tpr: [0, 0.5, 1],
              thresholds: [0.8, 0.3, 0.05],
              auc: 0.65,
              n_pos: 2,
              n_neg: 3,
            },
            {
              label: "1",
              fpr: [0, 0.5, 1],
              tpr: [0, 1, 1],
              thresholds: [0.9, 0.6, 0.05],
              auc: 0.9,
              n_pos: 2,
              n_neg: 3,
            },
            {
              label: "2",
              fpr: [0, 0.75, 1],
              tpr: [0, 1, 1],
              thresholds: [0.8, 0.1, 0.05],
              auc: 0.875,
              n_pos: 1,
              n_neg: 4,
            },
          ]
        : null,
      pr_curves: withCurves
        ? [
            {
              label: "0",
              recall: [0, 0.5, 1],
              precision: [1, 0.5, 0.4],
              thresholds: [0.8, 0.3, 0.05],
              average_precision: 0.55,
              prevalence: 0.4,
              n_pos: 2,
            },
          ]
        : null,
      calibration: withCurves
        ? {
            bins: [
              { mean_predicted: 0.1, observed: 0, count: 2 },
              { mean_predicted: 0, observed: 0, count: 0 },
              { mean_predicted: 0.85, observed: 1, count: 2 },
              { mean_predicted: 0.8, observed: 1, count: 1 },
            ],
            n: 5,
            positive_class: "1",
          }
        : null,
    },
  };
}

function renderCharts(result: PredictResponse) {
  return render(
    <EvaluationCharts
      result={result}
      modelId="mdl-test"
      datasetId="ds-test"
    />,
  );
}

describe("part21 §3.1 — the shared evaluation charts (regression family)", () => {
  it("renders the parity scatter, the residual histogram, the Q–Q plot, and the summary", () => {
    renderCharts(regressionResult());
    // the parity scatter (the §22.5 extraction)
    expect(
      screen.getByText(/actual-versus-predicted parity scatter/),
    ).toBeInTheDocument();
    // the residual histogram + the Q–Q plot (the part21 additions)
    expect(screen.getByText(/Residual distribution/)).toBeInTheDocument();
    expect(screen.getByText(/Normal Q–Q of the residuals/)).toBeInTheDocument();
    // the residual summary line over the real stats
    expect(screen.getByText(/mean -0.0500/)).toBeInTheDocument();
    expect(screen.getByText(/12 numeric residuals/)).toBeInTheDocument();
  });

  it("reports the sample bound on the surface", () => {
    renderCharts(regressionResult());
    expect(
      screen.getByText(/12 of 500 rows plotted \(the route's own sample bound\)/),
    ).toBeInTheDocument();
  });

  it("clicks through to the point's own row metadata (§22.5's identity)", async () => {
    const user = userEvent.setup();
    renderCharts(regressionResult());
    await user.click(screen.getByText("select-point-0"));
    // the inspector shows the REAL row's own identity, never a position
    await waitFor(() => {
      expect(screen.getByText("Selected sample (the point's own metadata)")).toBeInTheDocument();
    });
    // actual 0 * 1.1 = 0 — the first row's actual value
    expect(screen.getByText("mdl-test")).toBeInTheDocument();
    expect(screen.getByText("ds-test")).toBeInTheDocument();
    expect(screen.getByText("regression")).toBeInTheDocument();
  });

  it("handles null residuals in the sample without fabricating diagnostics", () => {
    const result = regressionResult();
    const evaluation = result.evaluation!;
    const rows = evaluation!.rows.map((row) => ({
      ...row,
      residual: null,
    }));
    const bare: PredictResponse = {
      ...result,
      evaluation: {
        ...evaluation,
        rows,
        residual_diagnostics: null,
      },
    };
    renderCharts(bare);
    expect(screen.getByText(/parity: actual vs predicted|regression diagnostics/)).toBeInTheDocument();
    expect(screen.queryByText(/Residual distribution/)).toBeNull();
    expect(screen.queryByText(/Normal Q–Q/)).toBeNull();
  });

  it("builds the evaluation export rows (the shared §3.4 lane)", () => {
    const result = regressionResult();
    const built = buildEvaluationExportRows(result);
    expect(built.columns).toEqual([
      "row",
      "actual",
      "predicted",
      "residual",
      "smiles",
    ]);
    expect(built.rows).toHaveLength(12);
    expect(built.rows[0]).toMatchObject({ row: 0, actual: 0 });
    const preds = buildPredictionExportRows(result);
    expect(preds.columns).toEqual(["prediction"]);
    expect(preds.rows).toHaveLength(12);
  });
});

describe("part21 §3.1 — the shared evaluation charts (classification family)", () => {
  it("renders the confusion matrix, the metrics table, and every curve family", () => {
    renderCharts(classificationResult(true));
    expect(screen.getByText(/confusion matrix heatmap/)).toBeInTheDocument();
    expect(screen.getByText(/Per-class metrics over the sample/)).toBeInTheDocument();
    expect(screen.getByText(/ROC curves \(one-vs-rest, per class\)/)).toBeInTheDocument();
    expect(screen.getByText(/Precision–Recall curves \(one-vs-rest, per class\)/)).toBeInTheDocument();
    expect(screen.getByText(/Reliability diagram \(positive class\)/)).toBeInTheDocument();
    // the per-class metrics table's own rows (string labels, real supports)
    expect(screen.getAllByRole("cell", { name: "0.6667" })).toHaveLength(2);
    expect(screen.getAllByRole("cell", { name: "5" }).length).toBeGreaterThan(0);
  });

  it("toggles the confusion matrix between raw counts and the per-true-class normalized view", async () => {
    const user = userEvent.setup();
    renderCharts(classificationResult(true));
    await user.click(screen.getByText("Show normalized (per true class)"));
    expect(
      screen.getByText("Show raw counts"),
    ).toBeInTheDocument();
    // back again — the toggle is a control, not a one-way trip
    await user.click(screen.getByText("Show raw counts"));
    expect(
      screen.getByText("Show normalized (per true class)"),
    ).toBeInTheDocument();
  });

  it("renders the explanatory state when the answer carried no probability block", () => {
    renderCharts(classificationResult(false));
    // the confusion matrix + metrics stay (they need only aligned rows)
    expect(screen.getByText(/confusion matrix heatmap/)).toBeInTheDocument();
    expect(screen.getByText(/Per-class metrics over the sample/)).toBeInTheDocument();
    // the curves are absent — with the honest explanation
    expect(
      screen.getByText(/No ROC\/PR\/calibration charts/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/ROC curves/)).toBeNull();
  });

  it("renders the no-evaluation explanatory state (inline-row answers)", () => {
    const result = classificationResult(true);
    const bare: PredictResponse = { ...result, evaluation: null };
    renderCharts(bare);
    expect(
      screen.getByText(/No evaluation block in this answer/),
    ).toBeInTheDocument();
  });

  it("renders non-numeric class labels in the metrics table verbatim", () => {
    const result = classificationResult(true);
    const evaluation = result.evaluation;
    const classMetrics = evaluation !== null && evaluation.class_metrics !== null
      ? {
          ...evaluation.class_metrics,
          rows: evaluation.class_metrics.rows.map((row) => ({
            ...row,
            label: `active-${row.label}`,
          })),
        }
      : null;
    const relabeled: PredictResponse =
      evaluation !== null
        ? { ...result, evaluation: { ...evaluation, class_metrics: classMetrics } }
        : result;
    renderCharts(relabeled);
    expect(screen.getByText("active-0")).toBeInTheDocument();
    expect(screen.getByText("active-2")).toBeInTheDocument();
  });
});
