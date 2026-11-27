/** The per-class one-vs-rest Precision–Recall curves.
 *
 * part21 §3.1: one line per class over the backend-computed
 * recall/precision walks, each carrying its own average precision and
 * — the imbalance-honest rail — its PREVALENCE baseline as the dashed
 * horizontal (a class present in 10% of the sample anchors its no-skill
 * line at 0.10, never at ROC's 0.5): on imbalanced data the PR curve's
 * baseline is the honest reference, not the ROC diagonal.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { PrCurveSeries } from "@/services/ml/client";

const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
] as const;

export interface PrCurveChartProps {
  readonly curves: readonly PrCurveSeries[];
}

export function PrCurveChart({ curves }: PrCurveChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const series: PlotTrace[] = [];
    curves.forEach((curve, i) => {
      const color = SERIES_COLORS[i % SERIES_COLORS.length];
      series.push({
        type: "scatter",
        mode: "lines",
        name: `${curve.label} (AP ${curve.average_precision.toFixed(3)})`,
        x: curve.recall,
        y: curve.precision,
        customdata: curve.recall.map((rec, j) => [
          curve.label,
          rec,
          curve.precision[j],
          curve.thresholds[j],
        ]),
        hovertemplate:
          "%{customdata[0]}<br>recall %{customdata[1]:.4g}" +
          "<br>precision %{customdata[2]:.4g}" +
          "<br>threshold ≥ %{customdata[3]:.4g}<extra></extra>",
        line: { color, width: 2 },
      });
      series.push({
        type: "scatter",
        mode: "lines",
        name: `${curve.label} prevalence`,
        x: [0, 1],
        y: [curve.prevalence, curve.prevalence],
        showlegend: false,
        hovertemplate:
          "%{customdata[0]}<br>prevalence %{y:.4g}<extra></extra>",
        customdata: curve.recall.map(() => [curve.label]),
        line: { color, width: 1, dash: "dot" },
      });
    });
    return series;
  }, [curves]);
  return (
    <PlotlyFigure
      data={traces}
      layout={{
        xaxis: { title: { text: "recall" }, range: [0, 1] },
        yaxis: { title: { text: "precision" }, range: [0, 1] },
        margin: { l: 56, r: 12, t: 8, b: 40 },
      }}
      height={280}
      ariaLabel="the one-vs-rest precision-recall curves per class over the evaluation sample"
      emptyHint="No PR curves in this answer."
    />
  );
}
