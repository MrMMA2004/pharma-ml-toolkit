/**
 * The TS twin of pharmaml.biology.most_variable_genes (the 100x-124
 * web mirror of the desktop's "Rank variable genes (MAD)" button).
 *
 * Variability is the median absolute deviation (MAD) of each gene's
 * values across the panel — the robust workhorse for RNA-seq feature
 * selection. The ranking is deterministic: genes sort by MAD
 * descending, ties broken by gene name ascending, exactly the library
 * contract, so the parity guard (scratch/probe_mad_parity_124.{ts,py})
 * can compare the two implementations row-by-row.
 */

export interface VariableGeneRow {
  /** 1-based rank by MAD descending. */
  rank: number;
  gene: string;
  /** Median absolute deviation across the panel (log2 units). */
  mad: number;
}

/**
 * Median with the pandas/numpy convention: the average of the two
 * middle values for even-length samples. Sorts a copy — the input
 * array is never mutated.
 */
export function median(values: readonly number[]): number {
  if (values.length === 0) throw new Error("median of an empty sample");
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Median absolute deviation: median(|x − median(x)|). */
export function mad(values: readonly number[]): number {
  const center = median(values);
  return median(values.map((v) => Math.abs(v - center)));
}

/**
 * Rank a gene-by-cell-line matrix by MAD, most variable first.
 *
 * @param matrix gene → values across the panel (one array per gene)
 * @param n number of rows to return (clipped to the gene count)
 * @returns ranked rows, MAD descending then gene ascending
 */
export function rankVariableGenes(
  matrix: ReadonlyMap<string, readonly number[]>,
  n: number
): VariableGeneRow[] {
  if (n < 1) {
    throw new Error(`n must be >= 1, got ${n}`);
  }
  const rows: Array<{ gene: string; mad: number }> = [];
  for (const [gene, values] of matrix) {
    if (values.length === 0) continue;
    rows.push({ gene, mad: mad(values) });
  }
  rows.sort((a, b) => {
    if (a.mad !== b.mad) return b.mad - a.mad;
    return a.gene < b.gene ? -1 : a.gene > b.gene ? 1 : 0;
  });
  return rows.slice(0, Math.min(n, rows.length)).map((row, i) => ({
    rank: i + 1,
    gene: row.gene,
    mad: row.mad,
  }));
}
