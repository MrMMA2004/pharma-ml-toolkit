/** The normal Q–Q plot of the sample's residuals.
 *
 * part21 §3.1: the backend computed the Blom-position theoretical
 * quantiles and the standardized empirical order statistics — the
 * frontend draws exactly those pairs against the identity line. A
 * heavy-tailed deviation from the line is the honest signal it shows;
 * the reference line is the visual anchor, never a fitted claim.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { QQPoints } from "@/services/ml/client";

export interface QqPlotChartProps {
  readonly qq: QQPoints;
}

export function QqPlotChart({ qq }: QqPlotChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const xs = qq.theoretical;
    const ys = qq.empirical;
    const lo = Math.min(...xs, ...ys);
    const hi = Math.max(...xs, ...ys);
    const pad = (hi - lo) * 0.05 || 0.5;
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "ordered residuals",
        x: xs,
        y: ys,
        customdata: xs.map((x, i) => [x, ys[i]]),
        hovertemplate:
          "theoretical %{customdata[0]:.4g}" +
          "<br>empirical %{customdata[1]:.4g}<extra></extra>",
        marker: { color: "var(--chart-4)", size: 6, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "y = x",
        x: [lo - pad, hi + pad],
        y: [lo - pad, hi + pad],
        showlegend: false,
        hoverinfo: "skip",
        line: { color: "var(--chart-5)", width: 1, dash: "dot" },
      },
    ];
  }, [qq]);
  return (
    <PlotlyFigure
      data={traces}
      layout={{
        xaxis: { title: { text: "theoretical normal quantile" } },
        yaxis: { title: { text: "residual (standardized)" } },
        margin: { l: 56, r: 12, t: 8, b: 40 },
      }}
      height={240}
      ariaLabel="the normal quantile-quantile plot of the evaluation sample's residuals"
      emptyHint="No residual quantile pairs to plot."
    />
  );
}
