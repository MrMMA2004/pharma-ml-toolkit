/**
 * Hyperband rung-scatter row builder — pure, self-contained.
 *
 * Extracted from the HyperbandRungScatter component (100x-117; the fold
 * itself landed in 100x-115/116) so the shipped bracket-best logic can
 * be probed directly (scratch/probe_rung_fold_117.ts via bun — the
 * probe runs the code the browser runs, not a mirror), following the
 * bayes-acquisition.ts precedent from 100x-114.
 *
 * Contract with the tune API: trials arrive ranked best-first
 * (score desc, then trial asc on ties), so the first EVALUABLE row
 * (finite score + numeric resource) is the global winner — that is
 * the `isWinner: i === 0` pin. Per-bracket best uses strict `>` so a
 * tie keeps the FIRST occurrence in arrival order (deterministic like
 * every chart in the house). Missing Hyperband fields default to the
 * RandomSearch-equivalent values (bracket 0 / rung 0 / not promoted).
 *
 * One deliberate fix over the 100x-115 component fold, caught by the
 * probe this round: `typeof resource === "number"` let NaN resources
 * through (typeof NaN === "number"), which would draw a dot at
 * x = NaN; the filter now requires a FINITE resource.
 */
import type { MLTuneTrial } from "./toolkit-types";

/** One Hyperband evaluation: a dot at (resource, CV score). */
export interface RungPoint {
  /** The resource the config was actually scored at (X axis). */
  x: number;
  /** The mean CV score (Y axis). */
  y: number;
  /** 1-based trial number, in the order the configs were drawn. */
  trial: number;
  /** The bracket the trial ran in (0 = full-budget bracket). */
  bracket: number;
  /** The rung index within the bracket (0 = cheapest). */
  rung: number;
  /** True when the config survived promotion to the next rung. */
  promoted: boolean;
  /** The global winner (first evaluable row of the ranked trials). */
  isWinner: boolean;
  /** The highest-scoring evaluation inside this point's bracket —
   *  exactly one dot per bracket carries the flag. */
  isBracketBest: boolean;
}

/**
 * Fold the ranked tuner trials into rung-scatter rows.
 *
 * Filters to evaluable rows (FINITE `resource` — the NaN fix — and
 * finite `score`), then single-passes once over them to find each
 * bracket's own best (the O(n) replacement for the original quadratic
 * `find`-in-loop; behavior-identical: strict `>`, ties keep the first
 * occurrence), then maps to the row shape the chart, the tooltip, the
 * facets and the ARIA labels all consume.
 */
export function buildRungPoints(trials: MLTuneTrial[]): RungPoint[] {
  const evaluated = trials.filter(
    (t) =>
      typeof t.resource === "number" &&
      Number.isFinite(t.resource) &&
      Number.isFinite(t.score)
  );
  const bestByBracket = new Map<number, { trial: number; score: number }>();
  for (const t of evaluated) {
    const b = t.bracket ?? 0;
    const best = bestByBracket.get(b);
    if (best === undefined || t.score > best.score) {
      bestByBracket.set(b, { trial: t.trial, score: t.score });
    }
  }
  return evaluated.map((t, i) => ({
    x: t.resource as number,
    y: t.score,
    trial: t.trial,
    bracket: t.bracket ?? 0,
    rung: t.rung ?? 0,
    promoted: t.promoted ?? false,
    isWinner: i === 0,
    isBracketBest: bestByBracket.get(t.bracket ?? 0)?.trial === t.trial,
  }));
}
