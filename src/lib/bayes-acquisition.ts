/**
 * Bayesian-optimisation acquisition math — pure, self-contained.
 *
 * Extracted verbatim from the GP band chart overlay (100x-113) so the
 * shipped formulas can be probed directly against scipy.stats.norm
 * (scratch/verify_acquisition_parity_114.{py,ts} on both sides).
 *
 * Maximisation convention throughout: the "best observed" value f* is
 * the MAX of the train targets, and both acquisitions reward points
 * expected to EXCEED it.
 */

/** Standard-normal CDF via the Abramowitz–Stegun 26.2.17 polynomial
 *  approximation of Φ (|ε| < 7.5e-8 — plenty for a chart overlay).
 *  Note: this is the direct Φ polynomial (t = 1/(1 + 0.2316419|z|),
 *  five odd-mirror coefficients), NOT the 7.1.26 erf rational form. */
export function stdNormalCdf(z: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly =
    t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const p = 1 - (Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)) * poly;
  return z >= 0 ? p : 1 - p;
}

/** Standard-normal PDF φ(z). */
export function stdNormalPdf(z: number): number {
  return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
}

/** Expected Improvement (maximisation) at one posterior point:
 *  EI = (μ − f*)·Φ(z) + σ·φ(z) with z = (μ − f*)/σ.
 *  Degenerate σ (≤ 1e-12) collapses to the deterministic improvement
 *  max(0, μ − f*) — the same convention the scipy reference probe
 *  pins, so the guard branch is parity-checked too. */
export function expectedImprovement(mu: number, sigma: number, fStar: number): number {
  if (sigma <= 1e-12) return Math.max(0, mu - fStar);
  const z = (mu - fStar) / sigma;
  return (mu - fStar) * stdNormalCdf(z) + sigma * stdNormalPdf(z);
}

/** Upper Confidence Bound (maximisation): μ + κ·σ. */
export function upperConfidenceBound(mu: number, sigma: number, kappa: number): number {
  return mu + kappa * sigma;
}
