/** The per-class metrics table over the evaluation sample.
 *
 * part21 §3.1: precision / recall / F1 / support per class — the
 * backend's own counts over the bounded sample, in the model's own
 * class ordering — with the macro averages and the sample accuracy in
 * the footer. Zero-division metrics are 0.0 with the support column
 * exposed so a zero reads as "no samples in the sample", never as
 * performance.
 */
import type { ClassMetrics } from "@/services/ml/client";

import { formatEvalValue } from "./formatValue";

export interface ClassMetricsTableProps {
  readonly metrics: ClassMetrics;
}

export function ClassMetricsTable({ metrics }: ClassMetricsTableProps) {
  return (
    <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
      <div className="max-h-72 overflow-auto">
        <table className="w-full border-collapse text-[11px]">
          <thead className="sticky top-0 bg-[var(--color-muted)]">
            <tr>
              <th scope="col" className="px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                class
              </th>
              <th scope="col" className="px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                precision
              </th>
              <th scope="col" className="px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                recall
              </th>
              <th scope="col" className="px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                F1
              </th>
              <th scope="col" className="px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                support
              </th>
            </tr>
          </thead>
          <tbody>
            {metrics.rows.map((row) => (
              <tr
                key={row.label}
                className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
              >
                <td className="px-3 py-1 font-medium text-[var(--color-foreground)]">
                  {formatEvalValue(row.label)}
                </td>
                <td className="px-3 py-1 text-[var(--color-muted-foreground)]">
                  {row.precision.toFixed(4)}
                </td>
                <td className="px-3 py-1 text-[var(--color-muted-foreground)]">
                  {row.recall.toFixed(4)}
                </td>
                <td className="px-3 py-1 text-[var(--color-muted-foreground)]">
                  {row.f1.toFixed(4)}
                </td>
                <td className="px-3 py-1 text-[var(--color-muted-foreground)]">
                  {row.support}
                </td>
              </tr>
            ))}
            <tr className="border-t border-[var(--color-border)] bg-[var(--color-muted)] font-medium">
              <td className="px-3 py-1 text-[var(--color-foreground)]">macro avg</td>
              <td className="px-3 py-1 text-[var(--color-foreground)]">
                {metrics.macro.precision.toFixed(4)}
              </td>
              <td className="px-3 py-1 text-[var(--color-foreground)]">
                {metrics.macro.recall.toFixed(4)}
              </td>
              <td className="px-3 py-1 text-[var(--color-foreground)]">
                {metrics.macro.f1.toFixed(4)}
              </td>
              <td className="px-3 py-1 text-[var(--color-foreground)]">
                {metrics.n}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="border-t border-[var(--color-border)] px-3 py-1.5 text-[11px] text-[var(--color-muted-foreground)]">
        Sample accuracy {metrics.accuracy.toFixed(4)} over{" "}
        {metrics.n} rows — the same bounded sample the charts plot
        (the train answer&apos;s cv block stays the authoritative
        cross-validated quality lane).
      </p>
    </div>
  );
}
