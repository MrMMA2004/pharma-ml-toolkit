/** The per-class one-vs-rest ROC curves.
 *
 * part21 §3.1: one line per class over the backend-computed FPR/TPR
 * walks (each class colored from the design-token palette —
 * colorblind-aware by construction), the chance diagonal as the
 * dashed anchor, and the class's own AUC + sample counts in the
 * legend/hover. Only the classes with both positives and negatives
 * in the sample are here — the backend omits the rest honestly.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { RocCurveSeries } from "@/services/ml/client";

const SERIES_COLORS = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
] as const;

export interface RocCurveChartProps {
  readonly curves: readonly RocCurveSeries[];
}

export function RocCurveChart({ curves }: RocCurveChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const series: PlotTrace[] = curves.map((curve, i) => ({
      type: "scatter",
      mode: "lines",
      name: `${curve.label} (AUC ${curve.auc.toFixed(3)})`,
      x: curve.fpr,
      y: curve.tpr,
      customdata: curve.fpr.map((fpr, j) => [
        curve.label,
        fpr,
        curve.tpr[j],
        curve.thresholds[j],
      ]),
      hovertemplate:
        "%{customdata[0]}<br>FPR %{customdata[1]:.4g}" +
        "<br>TPR %{customdata[2]:.4g}" +
        "<br>threshold ≥ %{customdata[3]:.4g}<extra></extra>",
      line: {
        color: SERIES_COLORS[i % SERIES_COLORS.length],
        width: 2,
      },
    }));
    series.push({
      type: "scatter",
      mode: "lines",
      name: "chance",
      x: [0, 1],
      y: [0, 1],
      showlegend: false,
      hoverinfo: "skip",
      line: { color: "var(--chart-5)", width: 1, dash: "dot" },
    });
    return series;
  }, [curves]);
  return (
    <PlotlyFigure
      data={traces}
      layout={{
        xaxis: { title: { text: "false-positive rate" }, range: [0, 1] },
        yaxis: { title: { text: "true-positive rate" }, range: [0, 1] },
        margin: { l: 56, r: 12, t: 8, b: 40 },
      }}
      height={280}
      ariaLabel="the one-vs-rest ROC curves per class over the evaluation sample"
      emptyHint="No ROC curves in this answer."
    />
  );
}
