/** The residual distribution over the sample's own residuals.
 *
 * part21 §3.1: a bar chart over the backend-computed histogram (the
 * bin edges and counts the evaluation block attached) — the frontend
 * renders what the backend computed, never re-binning in TypeScript.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { ResidualDiagnostics } from "@/services/ml/client";

export interface ResidualHistogramChartProps {
  readonly diagnostics: ResidualDiagnostics;
}

export function ResidualHistogramChart({
  diagnostics,
}: ResidualHistogramChartProps) {
  const { traces, layout } = useMemo(() => {
    const edges = diagnostics.histogram.bin_edges;
    const counts = diagnostics.histogram.counts;
    const centers: number[] = [];
    const widths: number[] = [];
    for (let i = 0; i < counts.length; i++) {
      const lo = edges[i] ?? 0;
      const hi = edges[i + 1] ?? lo + 1;
      centers.push((lo + hi) / 2);
      widths.push(Math.max(hi - lo, 1e-9));
    }
    const traces: readonly PlotTrace[] = [
      {
        type: "bar",
        name: "residuals",
        x: centers,
        y: counts,
        width: widths,
        customdata: centers.map((_, i) => [edges[i], edges[i + 1], counts[i]]),
        hovertemplate:
          "residual bin [%{customdata[0]:.4g}, %{customdata[1]:.4g})" +
          "<br>count %{customdata[2]}<extra></extra>",
        marker: { color: "var(--chart-2)" },
      },
    ];
    return {
      traces,
      layout: {
        xaxis: { title: { text: "residual (actual − predicted)" } },
        yaxis: { title: { text: "count" } },
        bargap: 0.05,
        margin: { l: 56, r: 12, t: 8, b: 40 },
      },
    };
  }, [diagnostics]);
  return (
    <PlotlyFigure
      data={traces}
      layout={layout}
      height={240}
      ariaLabel="the residual distribution histogram over the evaluation sample"
      emptyHint="No residuals to histogram."
    />
  );
}
