/**
 * Web-console learning-curve bookkeeping that borrows chart conventions
 * from `pharmaml.viz.plot_learning_curve`: the default fraction schedule
 * (10 % … 100 % in eight steps), gap-verdict threshold and wording, and
 * mean ± population-std fold summaries. The route uses independent
 * TypeScript fitters and rounds sizes separately within each fold; it does
 * not reproduce sklearn's absolute training-size schedule or scores. The
 * parity probe pins only the stated shared conventions, not full numeric
 * parity or a shared UI/feature set.
 */

/**
 * The library's default schedule: linspace(0.1, 1.0, 8) — step
 * 0.9/7 ≈ 0.128571…, the LAST element pinned exactly to 1.0 (both
 * numpy and this mirror set the final element by construction, never
 * by accumulation).
 */
export const LC_FRACTIONS: readonly number[] = linspace(0.1, 1.0, 8);

/** numpy.linspace's arithmetic: start + i·step, last pinned to stop. */
function linspace(start: number, stop: number, num: number): number[] {
  if (num < 2) throw new Error("linspace needs at least 2 points");
  const step = (stop - start) / (num - 1);
  const out: number[] = [];
  for (let i = 0; i < num - 1; i++) out.push(start + step * i);
  out.push(stop);
  return out;
}

/** Train/CV gap above which the verdict blames overfitting, not data. */
export const LC_GAP_THRESHOLD = 0.15;

/**
 * The verdict line the library prints under the curve title. The gap
 * is formatted with two decimals exactly like the Python ``f"gap ≈
 * {gap:.2f} …"`` template.
 */
export function lcVerdict(gap: number): string {
  return gap > LC_GAP_THRESHOLD
    ? `gap ≈ ${gap.toFixed(2)} — overfitting dominates, more data alone will not close it`
    : "curves converging — additional data has diminishing returns";
}

/** One schedule point, already reduced across folds. */
export interface LcPoint {
  /** Largest fold-specific training-subset size at this point. */
  size: number;
  trainMean: number;
  trainStd: number;
  valMean: number;
  valStd: number;
}

/** What the engine needs from the caller per (fold, size) pair. */
export interface LcFoldInputs {
  /** Validation-fold row indices (scored as the CV side). */
  fold: number[];
  /** This fold's full training-complement indices, in fold order. */
  trainIdx: number[];
}

export interface LcEngineInput {
  /** Pre-split fold inputs (the route's kFoldSplit output). */
  folds: LcFoldInputs[];
  /**
   * Fit on the given training rows and score BOTH sides: returns
   * [trainScore, valScore] where trainScore is measured on the very
   * rows the fit saw (the training curve) and valScore on the fold
   * (the CV curve). ``slot`` is the 1-based fold index — the caller
   * derives its per-fold RNG stream from it (the train route's runCv
   * convention), so identical requests stay byte-identical. Pure from
   * the engine's perspective — the caller owns the model construction
   * and metric.
   */
  fitAndScore: (trainRows: number[], valRows: number[], slot: number) => [number, number];
  /** Size schedule override; defaults to the library's 8 steps. */
  fractions?: readonly number[];
}

/**
 * The learning-curve reduction: for every fraction, each fold's
 * training complement is cut to a leading-row prefix (no re-shuffle),
 * then both scores reduce to mean ± population std across folds. This
 * prefix convention resembles sklearn's, but sizing differs: this web
 * helper independently uses ``max(1, round(fraction × fold-train rows))``
 * per fold, whereas sklearn converts fractions against the smallest
 * training split and uses the resulting common absolute size.
 *
 * Deterministic: no RNG of its own. The reported ``size`` is the
 * largest fold-specific subset for that fraction, so the x axis shows
 * an upper bound on the rows used at that point, not a common size.
 */
export function learningCurvePoints(input: LcEngineInput): LcPoint[] {
  const fractions = input.fractions ?? LC_FRACTIONS;
  if (fractions.length === 0) throw new Error("learning curve needs a non-empty size schedule");
  for (const f of fractions) {
    if (!(f > 0) || f > 1) {
      throw new Error(`learning-curve fractions must be in (0, 1], got ${f}`);
    }
  }
  if (input.folds.length === 0) throw new Error("learning curve needs at least one fold");
  const points: LcPoint[] = [];
  for (const fraction of fractions) {
    const trainScores: number[] = [];
    const valScores: number[] = [];
    let maxSize = 0;
    for (let foldIndex = 0; foldIndex < input.folds.length; foldIndex++) {
      const foldInput = input.folds[foldIndex];
      const size = Math.max(1, Math.round(fraction * foldInput.trainIdx.length));
      maxSize = Math.max(maxSize, size);
      const sub = foldInput.trainIdx.slice(0, size);
      const [trainScore, valScore] = input.fitAndScore(sub, foldInput.fold, foldIndex + 1);
      trainScores.push(trainScore);
      valScores.push(valScore);
    }
    points.push({
      size: maxSize,
      trainMean: mean(trainScores),
      trainStd: populationStd(trainScores),
      valMean: mean(valScores),
      valStd: populationStd(valScores),
    });
  }
  return points;
}

/** Arithmetic mean (the fold-reduction the train route's CV uses). */
function mean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (const v of values) sum += v;
  return sum / values.length;
}

/** Population std (ddof = 0), matching the toolkit's CV convention. */
function populationStd(values: readonly number[]): number {
  if (values.length <= 1) return 0;
  const m = mean(values);
  let acc = 0;
  for (const v of values) acc += (v - m) * (v - m);
  return Math.sqrt(acc / values.length);
}
