/** The binary reliability diagram (the calibration view).
 *
 * part21 §3.1: the observed positive fraction against the mean
 * predicted probability per bin — the backend computed both over the
 * sample's positive-class scores; empty bins stay count-0 points
 * (honest gaps, never dropped to fake density); the dashed identity
 * line is the perfectly-calibrated anchor, and each hover carries
 * the bin's own count so a 1-sample bin never reads as a strong
 * estimate.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { CalibrationDiagram } from "@/services/ml/client";

export interface CalibrationChartProps {
  readonly calibration: CalibrationDiagram;
}

export function CalibrationChart({ calibration }: CalibrationChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const occupied = calibration.bins.filter((bin) => bin.count > 0);
    const series: PlotTrace[] = [
      {
        type: "scatter",
        mode: "lines+markers",
        name: `observed vs predicted (${calibration.positive_class})`,
        x: occupied.map((bin) => bin.mean_predicted),
        y: occupied.map((bin) => bin.observed),
        customdata: occupied.map((bin) => [bin.count]),
        hovertemplate:
          "mean predicted %{x:.4g}<br>observed %{y:.4g}" +
          "<br>count %{customdata[0]}<extra></extra>",
        line: { color: "var(--chart-3)", width: 2 },
        marker: { size: 7 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "perfect calibration",
        x: [0, 1],
        y: [0, 1],
        showlegend: false,
        hoverinfo: "skip",
        line: { color: "var(--chart-5)", width: 1, dash: "dot" },
      },
    ];
    return series;
  }, [calibration]);
  return (
    <div>
      <PlotlyFigure
        data={traces}
        layout={{
          xaxis: { title: { text: "mean predicted probability" }, range: [0, 1] },
          yaxis: { title: { text: "observed positive fraction" }, range: [0, 1] },
          margin: { l: 56, r: 12, t: 8, b: 40 },
        }}
        height={240}
        ariaLabel="the reliability diagram of the positive-class probabilities over the evaluation sample"
        emptyHint="No calibration bins in this answer."
      />
      <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
        Over {calibration.n} samples, positive class{" "}
        {calibration.positive_class}; empty bins are honest gaps (count 0),
        not zero-observation estimates.
      </p>
    </div>
  );
}
