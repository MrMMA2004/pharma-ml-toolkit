/** §22.5's sample inspector — the clicked row's own metadata.
 *
 * Extracted from the ML workspace verbatim (part21 §3.1): the point's
 * own row identity, SMILES when the dataset carried structures,
 * actual, predicted, residual, and the model/dataset context — never
 * a plot position or an array index.
 */
import { Button } from "@/components/ui/button";
import type { EvaluationRow } from "@/services/ml/client";

import { formatEvalValue } from "./formatValue";

export interface EvalRowInspectorProps {
  readonly row: EvaluationRow | null;
  readonly target: string;
  readonly task: string;
  readonly modelId: string;
  readonly datasetId: string | null;
  readonly onClear: () => void;
}

export function EvalRowInspector({
  row,
  target,
  task,
  modelId,
  datasetId,
  onClear,
}: EvalRowInspectorProps) {
  if (row === null) {
    return (
      <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
        Click a point in the parity chart to inspect that sample&apos;s own
        metadata (row identity, SMILES when the dataset carried them,
        actual, predicted, residual).
      </p>
    );
  }
  const fields: readonly [string, string][] = [
    ["row", formatEvalValue(row.name)],
    ["SMILES", row.smiles ?? "—"],
    [`actual ${target}`, formatEvalValue(row.actual)],
    ["predicted", formatEvalValue(row.predicted)],
    ["residual", row.residual === null ? "—" : row.residual.toFixed(4)],
    ["task", task],
    ["model", modelId],
    ["dataset", datasetId ?? "—"],
  ];
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted)] p-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-[var(--color-foreground)]">
          Selected sample (the point&apos;s own metadata)
        </p>
        <Button size="sm" variant="outline" onClick={onClear}>
          Clear
        </Button>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
        {fields.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="text-[10px] uppercase tracking-wide text-[var(--color-muted-foreground)]">
              {label}
            </dt>
            <dd className="truncate text-[11px] font-medium text-[var(--color-foreground)]">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
