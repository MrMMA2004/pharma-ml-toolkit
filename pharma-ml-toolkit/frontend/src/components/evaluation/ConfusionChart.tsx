/** The classification confusion-matrix heatmap over the sample rows.
 *
 * Extracted from the ML workspace (part21 §3.1) and extended with the
 * part21 §3.1 raw/normalized toggle: the counts mode shows absolute
 * tallies; the normalized mode divides each cell by its ACTUAL-class
 * row total (the per-true-class recall view), reporting the share of
 * the true class each predicted class received — a row the sample
 * never carried stays all-zero rather than a fabricated 0/0 division.
 */
import { useMemo, useState } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import { Button } from "@/components/ui/button";
import type { EvaluationRow } from "@/services/ml/client";

export interface ConfusionChartProps {
  readonly rows: readonly EvaluationRow[];
}

export function ConfusionChart({ rows }: ConfusionChartProps) {
  const [normalized, setNormalized] = useState(false);
  const { traces, layout, classes } = useMemo(() => {
    const classSet = new Set<string>();
    for (const row of rows) {
      classSet.add(String(row.actual));
      classSet.add(String(row.predicted));
    }
    const classes = Array.from(classSet).sort();
    const index = new Map(classes.map((cls, i) => [cls, i]));
    const z = classes.map(() => classes.map(() => 0));
    for (const row of rows) {
      const i = index.get(String(row.actual));
      const j = index.get(String(row.predicted));
      if (i === undefined || j === undefined) continue;
      z[i]![j]! += 1;
    }
    const rowTotals = z.map((row) => row.reduce((a, b) => a + b, 0));
    const display = normalized
      ? z.map((row, i) =>
          row.map(
            (count) =>
              rowTotals[i]! > 0
                ? Math.round((count / rowTotals[i]!) * 1000) / 1000
                : 0,
          ),
        )
      : z;
    const annotations = [];
    for (let i = 0; i < classes.length; i++) {
      for (let j = 0; j < classes.length; j++) {
        annotations.push({
          x: classes[j],
          y: classes[i],
          text: normalized
            ? String(display[i]![j]! === 0 ? "0" : display[i]![j]!.toFixed(2))
            : String(z[i]![j]!),
          showarrow: false,
          font: { size: 11 },
        });
      }
    }
    const traces: readonly PlotTrace[] = [
      {
        type: "heatmap",
        z: display,
        x: classes,
        y: classes,
        hovertemplate: normalized
          ? "actual %{y}<br>predicted %{x}<br>share %{z}<extra></extra>"
          : "actual %{y}<br>predicted %{x}<br>count %{z}<extra></extra>",
        colorscale: [
          [0, "rgba(47, 125, 95, 0.06)"],
          [1, "rgba(47, 125, 95, 0.92)"],
        ],
        showscale: false,
      },
    ];
    return {
      traces,
      layout: {
        xaxis: { title: { text: "predicted class" }, constrain: "domain" },
        yaxis: { title: { text: "actual class" }, autorange: "reversed" },
        annotations,
        margin: { l: 64, r: 12, t: 8, b: 48 },
      },
      classes,
    };
  }, [rows, normalized]);
  return (
    <div>
      <div className="mb-2 flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          aria-pressed={normalized}
          onClick={() => setNormalized((prev) => !prev)}
        >
          {normalized ? "Show raw counts" : "Show normalized (per true class)"}
        </Button>
      </div>
      <PlotlyFigure
        data={traces}
        layout={layout}
        height={300}
        ariaLabel={`the confusion matrix heatmap over ${rows.length} evaluation rows and ${classes.length} classes`}
        emptyHint="No class pairs to chart."
      />
    </div>
  );
}
