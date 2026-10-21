/**
 * 100x-117 — Hyperband bracket-best fold guard, TS side.
 *
 * Imports the SHIPPED src/lib/hyperband-rungs.ts (extracted from
 * ml-playground.tsx this round — that is the whole point: the probe
 * runs the code the browser runs, not a mirror) and pins the fold's
 * contract:
 *
 *   1. winner-first  — the first evaluable row of the API's ranked
 *      (best-first) trials is the ONE isWinner dot;
 *   2. per-bracket best — exactly one isBracketBest dot per bracket,
 *      and it owns the bracket's max score;
 *   3. ties keep the FIRST occurrence (strict >, deterministic);
 *   4. the winner may ALSO be its bracket's best (both flags true —
 *      rungDot renders winner first by design);
 *   5. non-finite scores / missing resources are dropped, and pure
 *      RandomSearch trials (no resource at all) produce no rows —
 *      INCLUDING the NaN-resource fix this round: the original fold's
 *      `typeof resource === "number"` let NaN through (a dot at
 *      x = NaN); the shipped module now requires a FINITE resource;
 *   6. missing bracket/rung/promoted default to 0/0/false;
 *   7. the shipped O(n) single-pass fold is IDENTICAL to the original
 *      quadratic find-in-loop algorithm it replaced, on a seeded
 *      240-trial randomized grid (the parity proof).
 *
 * Run:
 *   bun run scratch/probe_rung_fold_117.ts
 * Exits non-zero on the first broken contract with a named message.
 */
import { buildRungPoints, type RungPoint } from "../src/lib/hyperband-rungs";
import type { MLTuneTrial } from "../src/lib/toolkit-types";

function fail(name: string, detail: string): never {
  console.error(`FAIL [${name}] ${detail}`);
  process.exit(1);
}

function expect(name: string, cond: boolean, detail: string): void {
  if (!cond) fail(name, detail);
}

/* ---- 1-4: the pinned E2E-shaped case (mirrors the 100x-116 browser
 * evidence: 18 evaluations, 3 brackets, winner in bracket 2) --------- */
const RANKED: MLTuneTrial[] = [
  // ranked best-first, exactly as /api/ml/tune returns them
  { trial: 14, params: { estimators: 50 }, score: 0.9139, bracket: 2, rung: 2, resource: 50, promoted: true },
  { trial: 5, params: { estimators: 25 }, score: 0.9128, bracket: 0, rung: 0, resource: 3, promoted: false },
  { trial: 12, params: { estimators: 50 }, score: 0.9060, bracket: 2, rung: 1, resource: 16, promoted: true },
  { trial: 2, params: { estimators: 3 }, score: 0.9020, bracket: 0, rung: 0, resource: 3, promoted: false },
  { trial: 9, params: { estimators: 9 }, score: 0.9015, bracket: 1, rung: 0, resource: 4, promoted: true },
  { trial: 8, params: { estimators: 9 }, score: 0.9015, bracket: 1, rung: 0, resource: 4, promoted: true }, // tie with trial 9 — the bracket-1 MAX
  { trial: 11, params: { estimators: 50 }, score: 0.9010, bracket: 1, rung: 1, resource: 16, promoted: true },
  { trial: 15, params: { estimators: 50 }, score: 0.8970, bracket: 2, rung: 0, resource: 4, promoted: false },
  { trial: 1, params: { estimators: 3 }, score: 0.8900, bracket: 0, rung: 0, resource: 3, promoted: false },
  { trial: 6, params: { estimators: 25 }, score: 0.8850, bracket: 0, rung: 0, resource: 3, promoted: false },
  { trial: 10, params: { estimators: 9 }, score: 0.8820, bracket: 1, rung: 0, resource: 4, promoted: false },
  { trial: 13, params: { estimators: 16 }, score: 0.8780, bracket: 2, rung: 0, resource: 4, promoted: false },
];

const points: RungPoint[] = buildRungPoints(RANKED);

expect("shape", points.length === 12, `expected 12 rows, got ${points.length}`);
expect("row-fields", points[0]!.x === 50 && points[0]!.y === 0.9139
  && points[0]!.trial === 14 && points[0]!.bracket === 2
  && points[0]!.rung === 2 && points[0]!.promoted === true,
  `row 0 field mapping wrong: ${JSON.stringify(points[0])}`);

// 1 — winner-first, exactly one
const winners = points.filter((p) => p.isWinner);
expect("winner-first", winners.length === 1 && winners[0]!.trial === 14,
  `expected exactly trial 14 as winner, got ${JSON.stringify(winners.map((w) => w.trial))}`);

// 2 — exactly one bracket-best per bracket, owning the max
for (const b of [0, 1, 2]) {
  const rows = points.filter((p) => p.bracket === b);
  const bests = rows.filter((p) => p.isBracketBest);
  const maxScore = Math.max(...rows.map((r) => r.y));
  expect("one-best-per-bracket", bests.length === 1,
    `bracket ${b}: expected exactly 1 bracket-best, got ${bests.length}`);
  expect("best-owns-max", bests[0]!.y === maxScore,
    `bracket ${b}: bracket-best score ${bests[0]!.y} != bracket max ${maxScore}`);
}
// bracket 2's best IS the global winner (0.9139, trial 14)
const b2best = points.find((p) => p.bracket === 2 && p.isBracketBest)!;
expect("winner-also-bracket-best", b2best.isWinner,
  "the winner in bracket 2 should carry BOTH flags");

// 3 — ties keep the FIRST occurrence: trials 9 and 8 tie at 0.9015 in
// bracket 1; trial 9 appears first in arrival order
const b1best = points.find((p) => p.bracket === 1 && p.isBracketBest)!;
expect("tie-first-occurrence", b1best.trial === 9,
  `bracket 1 tie should keep trial 9 (first occurrence), got ${b1best.trial}`);

/* ---- 5: filtering (incl. the NaN-resource fix) ------------------- */
const withJunk: MLTuneTrial[] = [
  RANKED[0]!,
  { trial: 99, params: {}, score: Number.NaN, bracket: 0, rung: 0, resource: 5 },
  { trial: 98, params: {}, score: 0.5, bracket: 0, rung: 0 }, // no resource
  { trial: 97, params: {}, score: 0.4, resource: Number.NaN }, // NaN resource — the 100x-117 fix drops it
];
const filtered = buildRungPoints(withJunk);
expect("filtering", filtered.length === 1 && filtered[0]!.trial === 14,
  `junk rows (NaN score / no resource / NaN resource) must all drop, expected only trial 14, got ${JSON.stringify(filtered.map((p) => p.trial))}`);
// pure RandomSearch trials (resource never set) → no rows at all
const randomSearch: MLTuneTrial[] = [
  { trial: 1, params: {}, score: 0.9 },
  { trial: 2, params: {}, score: 0.8 },
];
expect("random-search-empty", buildRungPoints(randomSearch).length === 0,
  "RandomSearch trials without resource must produce zero rows");

/* ---- 6: defaults ------------------------------------------------- */
const bare: MLTuneTrial[] = [{ trial: 1, params: {}, score: 0.5, resource: 10 }];
const barePoints = buildRungPoints(bare);
expect("defaults", barePoints.length === 1 && barePoints[0]!.bracket === 0
  && barePoints[0]!.rung === 0 && barePoints[0]!.promoted === false,
  `missing hyperband fields must default to 0/0/false: ${JSON.stringify(barePoints[0])}`);

/* ---- 7: parity with the ORIGINAL quadratic fold ------------------ */
/** The pre-100x-117 algorithm, verbatim from ml-playground.tsx: the
 *  quadratic find-in-loop over the same filtered rows. */
function originalFold(trials: MLTuneTrial[]): RungPoint[] {
  const evaluated = trials.filter(
    (t) => typeof t.resource === "number" && Number.isFinite(t.score)
  );
  const bestTrialPerBracket = new Map<number, number>();
  for (const t of evaluated) {
    const b = t.bracket ?? 0;
    const best = bestTrialPerBracket.get(b);
    const bestScore = evaluated.find((e) => e.trial === best)?.score;
    if (best === undefined || bestScore === undefined || t.score > bestScore) {
      bestTrialPerBracket.set(b, t.trial);
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
    isBracketBest: bestTrialPerBracket.get(t.bracket ?? 0) === t.trial,
  }));
}

/* Deterministic seeded generator (mulberry32): 240 trials, 4 brackets,
 * heavy ties (scores quantised to 2 decimals so the strict-> first-
 * occurrence path is exercised ~always), shuffled arrival order, then
 * re-ranked best-first the way the API does. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(0x117);
const seeded: MLTuneTrial[] = Array.from({ length: 240 }, (_, i) => ({
  trial: i + 1,
  params: { estimators: 1 + Math.floor(rng() * 60) },
  score: Math.round(rng() * 100) / 100, // quantised → ties everywhere
  bracket: Math.floor(rng() * 4),
  rung: Math.floor(rng() * 3),
  resource: 1 + Math.floor(rng() * 51),
  promoted: rng() < 0.3,
}));
// one NaN-SCORE row: both algorithms drop it via the score check, so
// it stays inside the parity grid (unlike the NaN-resource row, whose
// drop is the deliberate 100x-117 fix pinned in section 5 above)
seeded.push({ trial: 241, params: {}, score: Number.NaN, bracket: 1, rung: 0, resource: 5 });
const seededRanked = [...seeded].sort(
  (a, b) => b.score - a.score || a.trial - b.trial
);
const shipped = buildRungPoints(seededRanked);
const original = originalFold(seededRanked);
expect("parity", JSON.stringify(shipped) === JSON.stringify(original),
  `shipped O(n) fold diverged from the original quadratic fold on the seeded grid `
    + `(${shipped.length} vs ${original.length} rows)`);

console.log(
  `PASS probe_rung_fold_117 — ${points.length} pinned rows (winner 14, ties keep 9), `
    + `NaN-resource fix pinned, filtering/defaults proven, and the shipped O(n) fold is `
    + `row-identical to the original quadratic algorithm on a ${seededRanked.length}-trial seeded grid`
);
