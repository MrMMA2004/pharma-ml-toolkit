/** The regression parity scatter with the identity reference line.
 *
 * Extracted from the ML workspace verbatim (part21 §3.1: the proven
 * §22.5 implementation becomes the shared layer's regression anchor):
 * two traces over the aligned sample (markers + the dashed y=x line),
 * every point carrying the row's own metadata in `customdata`, and
 * the click-through sample inspector riding `onRowSelect`.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { EvaluationRow } from "@/services/ml/client";

export interface ParityChartProps {
  readonly rows: readonly {
    readonly name: unknown;
    readonly actual: unknown;
    readonly predicted: unknown;
    readonly residual: number | null;
    readonly smiles: string | null;
  }[];
  readonly onRowSelect: (row: EvaluationRow) => void;
}

export function ParityChart({
  rows,
  onRowSelect,
}: ParityChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const xs = rows.map((row) => row.actual as number);
    const ys = rows.map((row) => row.predicted as number);
    const lo = Math.min(...xs, ...ys);
    const hi = Math.max(...xs, ...ys);
    const pad = (hi - lo) * 0.05 || 0.5;
    const x0 = lo - pad;
    const x1 = hi + pad;
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "samples",
        x: xs,
        y: ys,
        customdata: rows.map((row) => [
          row.name,
          row.smiles ?? "—",
          row.actual,
          row.predicted,
          row.residual ?? "—",
        ]),
        hovertemplate:
          "row %{customdata[0]}<br>actual %{customdata[2]:.4g}" +
          "<br>predicted %{customdata[3]:.4g}" +
          "<br>residual %{customdata[4]:.4g}" +
          "<br>%{customdata[1]}<extra></extra>",
        marker: { color: "var(--chart-1)", size: 7, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "y = x",
        x: [x0, x1],
        y: [x0, x1],
        showlegend: false,
        hoverinfo: "skip",
        line: { color: "var(--chart-5)", width: 1, dash: "dot" },
      },
    ];
  }, [rows]);
  const layout = useMemo(
    () => ({
      xaxis: { title: { text: "actual" }, scaleanchor: "y", scaleratio: 1 },
      yaxis: { title: { text: "predicted" } },
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [],
  );
  return (
    <PlotlyFigure
      data={traces}
      layout={layout}
      height={300}
      ariaLabel="the actual-versus-predicted parity scatter over the evaluation sample"
      emptyHint="No numeric actual/predicted pairs to plot."
      onPointSelect={(point) => {
        const index = point.pointIndex ?? null;
        if (index === null || index >= rows.length) return;
        onRowSelect(rows[index] as EvaluationRow);
      }}
    />
  );
}
