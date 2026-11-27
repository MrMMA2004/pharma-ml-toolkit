/**
 * The shared training-history chart (part21 §3.3).
 *
 * Extracted from the Deep Learning workspace (the live loss/validation
 * curve) as the reusable training visualization: a real Plotly line
 * chart drawn from the REAL per-epoch telemetry points (no chart
 * fabrication: two lines over the points the backend actually
 * published, the missing points of either series skipped, axes
 * labelled honestly; §22.2's learning-curve form, §22.4's per-point
 * hover without any mouse-following crosshair).
 *
 * The component displays only the signals the backend supplies — the
 * `TrainingMetricsPoint` contract (train_loss, val_loss, best_val,
 * stale) — and never invents accuracy, learning-rate, or epoch values
 * that no lane publishes. Additional series (train/val accuracy,
 * learning-rate schedule, epoch duration) attach here the day their
 * lanes genuinely emit them.
 */
import { useMemo } from "react";

import { PlotlyFigure } from "@/components/charts/PlotlyFigure";
import type { PlotTrace } from "@/components/charts/PlotlyFigure";
import type { TrainingMetricsPoint } from "@/services/dl/client";

export interface LossCurveChartProps {
  readonly points: readonly TrainingMetricsPoint[];
}

export function LossCurveChart({ points }: LossCurveChartProps) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const series = (key: "train_loss" | "val_loss"): readonly {
      epoch: number;
      value: number;
    }[] =>
      points
        .filter((p) => {
          const value = p[key];
          return value !== null && Number.isFinite(value);
        })
        .map((p) => ({ epoch: p.epoch, value: p[key] as number }));
    const train = series("train_loss");
    const val = series("val_loss");
    return [
      {
        type: "scatter",
        mode: "lines+markers",
        name: "train loss",
        x: train.map((p) => p.epoch),
        y: train.map((p) => p.value),
        customdata: train.map((p) => [p.epoch]),
        hovertemplate: "epoch %{customdata[0]} · train %{y:.4g}<extra></extra>",
        line: { color: "var(--chart-1)", width: 2 },
        marker: { size: 4 },
      },
      {
        type: "scatter",
        mode: "lines+markers",
        name: "validation loss",
        x: val.map((p) => p.epoch),
        y: val.map((p) => p.value),
        customdata: val.map((p) => [p.epoch]),
        hovertemplate: "epoch %{customdata[0]} · val %{y:.4g}<extra></extra>",
        line: { color: "var(--chart-3)", width: 2, dash: "dash" },
        marker: { size: 4 },
      },
    ];
  }, [points]);
  if (points.length < 2) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Waiting for at least two epochs of telemetry — the backend
        publishes one `training.metrics` envelope per epoch.
      </p>
    );
  }
  return (
    <PlotlyFigure
      data={traces}
      layout={{
        xaxis: { title: { text: "epoch" } },
        yaxis: { title: { text: "loss" } },
        margin: { l: 56, r: 12, t: 8, b: 40 },
      }}
      height={260}
      ariaLabel="the live train and validation loss per epoch"
    />
  );
}
