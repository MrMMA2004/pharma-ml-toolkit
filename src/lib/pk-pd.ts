/**
 * PK/PD math shared by the playground PK/PD Explorer.
 *
 * Two layers, both mirroring the `pharmaml.pharma` Python package
 * (`pharma-ml-toolkit/src/pharmaml/pharma/pk.py` + `pd.py`):
 *
 *  1. PK regimen — one-compartment superposition (moved here from
 *     pk-simulator.tsx so the PD panel can reuse the exact same curve);
 *  2. PD effect models — the four models of pharmaml.pharma.pd;
 *  3. NCA over observed data — trapezoid AUC, log-linear terminal fit,
 *     extrapolation, CL/Vz (mirrors auc_trapezoid / nca_summary /
 *     clearance / volume_of_distribution semantics, extended with a
 *     selectable terminal-point count + regression r²).
 *
 * ─── Cross-validation vs pharmaml (Python 3, /home/z/.venv/bin/python) ───
 * Probe: scratch/probe_pkpd_20b.py — same inputs, pharmaml.pharma functions.
 *
 *  ORAL_CLEAN  t=[0,.25,.5,1,2,4,6,8,12,24,36,48]
 *              c=[0,1.28,2.21,3.33,4.08,3.85,3.31,2.82,2.05,.79,.30,.12]
 *    auc_trapezoid            = 62.746250          (TS: 62.74625)
 *    nca_summary Cmax/Tmax    = 4.08 / 2.0
 *    ke  last-2 (Python)      = 0.076358 → t½ 9.077650   (TS N=2 identical)
 *    ke  last-3 LS            = 0.078523 → t½ 8.827365, r² 0.999747
 *    Clast 0.12 → AUCextrap   = 1.528223, AUC∞ 64.274473, %extrap 2.3777
 *    CL (F·D/AUC∞, D=250,F=.9)= 3.500612, Vz = CL/ke = 44.580978
 *  IV_BOLUS    t=[0,.5,1,2,4,6,8,10,12]
 *              c=[12,10.59,9.35,7.28,4.41,2.68,1.62,.99,.60]
 *    auc_trapezoid            = 46.227500
 *    Cmax/Tmax                = 12.00 / 0.0
 *    ke last-2 (Python)       = 0.250388 → t½ 2.768296 (TS N=2 identical)
 *    ke last-3 LS             = 0.248313 → t½ 2.791426, r² 0.999977
 *    AUCextrap 2.416306, AUC∞ 48.643806, %extrap 4.9673
 *    CL (D=500, F=1)          = 10.278801, Vz = 41.394544
 *  NOISY_ORAL  t=[0,.25,.5,1,2,4,6,8,12,24,36,48]
 *              c=[0,1.35,2.05,3.45,3.98,4.12,3.18,2.90,1.98,.85,.34,.08]
 *    auc_trapezoid            = 63.563750
 *    Cmax/Tmax                = 4.12 / 4.0   (noise flips the peak to t=4)
 *    ke last-2 (Python)       = 0.120577 → t½ 5.748605 (TS N=2 identical)
 *    ke last-3 LS             = 0.098467 → t½ 7.039380, r² 0.983472
 *    AUCextrap 0.812454, AUC∞ 64.376204, %extrap 1.2620
 *    CL (D=250, F=.9)         = 3.495080, Vz = 35.494914
 *
 *  PD spot-checks (pharmaml.pharma.pd):
 *    linear slope=8:            C=1.28→10.24, 4.08→32.64, 12.8→102.4
 *    emax E0=10 Emax=80 EC50=2: C=1.28→41.2195, 4.08→63.6842, 12.8→79.1892
 *    sigmoid (+h=1.5):          C=1.28→37.0899, 4.08→69.5590, 12.8→85.3464
 *    inhibitory E0=10 Imax=10 IC50=2: C=1.28→6.0976, 4.08→3.2895, 12.8→1.3514
 */

/* ------------------------------------------------------------------ */
/* PK regimen: one compartment, first-order elimination, superposition */
/* ------------------------------------------------------------------ */

export interface PkParamsLike {
  route: "iv" | "oral";
  dose: number; // mg
  vd: number; // L
  cl: number; // L/h
  tau: number; // h
  doses: number;
  f: number; // bioavailability (oral)
  ka: number; // 1/h (oral)
}

export interface PkDerived {
  ke: number; // 1/h
  halfLife: number; // h
  accumulation: number; // R_ac
  cssAvg: number; // mg/L
  cMaxFirst: number; // mg/L
  cMaxSS: number; // mg/L (NaN when not yet near steady state)
}

/** Concentration at time t — exact superposition of every dose so far. */
export function concentrationAt(
  t: number,
  p: PkParamsLike,
  ke: number,
): number {
  let c = 0;
  for (let i = 0; i < p.doses; i++) {
    const since = t - i * p.tau;
    if (since < 0) break;
    if (p.route === "iv") {
      c += (p.dose / p.vd) * Math.exp(-ke * since);
    } else {
      const denom = p.vd * (p.ka - ke);
      if (Math.abs(denom) < 1e-9) continue;
      c +=
        ((p.f * p.dose * p.ka) / denom) *
        (Math.exp(-ke * since) - Math.exp(-p.ka * since));
    }
  }
  return c;
}

/** Derived regimen quantities (ke, t½, accumulation, Css,avg, Cmax). */
export function derivePk(p: PkParamsLike): PkDerived {
  const ke = p.cl / p.vd;
  const halfLife = Math.log(2) / ke;
  const accumulation = 1 / (1 - Math.exp(-ke * p.tau));
  const cssAvg = (p.route === "iv" ? 1 : p.f) * (p.dose / (p.cl * p.tau));

  const endOfFirst = p.route === "iv" ? 0 : Math.log(p.ka / ke) / (p.ka - ke);
  const cMaxFirst = concentrationAt(endOfFirst, { ...p, doses: 1 }, ke);
  const lastDose = (p.doses - 1) * p.tau;
  const cMaxSS =
    p.doses > 4
      ? concentrationAt(lastDose + endOfFirst, p, ke)
      : Number.NaN; // not yet near steady state — don't mislead
  return { ke, halfLife, accumulation, cssAvg, cMaxFirst, cMaxSS };
}

/* ------------------------------------------------------------------ */
/* PD effect models (pharmaml.pharma.pd mirror)                        */
/* ------------------------------------------------------------------ */

export type PdModelId = "linear" | "emax" | "sigmoid" | "inhibitory";

export const PD_MODELS: Array<{
  id: PdModelId;
  label: string;
  equation: string;
  /** Potency parameter (mg/L) — null for the linear model. */
  potency: "ec50" | "ic50" | null;
}> = [
  {
    id: "linear",
    label: "Linear",
    equation: "E(C) = Slope·C",
    potency: null,
  },
  {
    id: "emax",
    label: "Emax",
    equation: "E(C) = E0 + Emax·C/(EC50 + C)",
    potency: "ec50",
  },
  {
    id: "sigmoid",
    label: "Sigmoid Emax",
    equation: "E(C) = E0 + Emax·C^h/(EC50^h + C^h)",
    potency: "ec50",
  },
  {
    id: "inhibitory",
    label: "Inhibitory Emax",
    equation: "E(C) = E0 − Imax·C/(IC50 + C)",
    potency: "ic50",
  },
];

export interface PdParamsLike {
  slope: number;
  e0: number;
  emax: number;
  ec50: number;
  hill: number;
  imax: number;
  ic50: number;
}

/**
 * Effect at one concentration for the selected model.
 *
 * Degenerate-parameter guards (the sliders normally prevent these, but
 * persisted/programmatic values must never reach NaN on screen):
 *  - negative C is clamped to 0 (concentrations are non-negative);
 *  - EC50/IC50 ≤ 0 is clamped to a tiny epsilon — the model limit as the
 *    potency parameter → 0⁺ is well defined ("infinitely potent": any
 *    C > 0 gives the full Emax term);
 *  - Hill h ≤ 0 is clamped to a tiny epsilon (h→0⁺ degenerates to a
 *    constant half-fraction; never a division by zero).
 */
export function pdEffect(
  c: number,
  model: PdModelId,
  p: PdParamsLike,
): number {
  const conc = Number.isFinite(c) && c > 0 ? c : 0;
  switch (model) {
    case "linear":
      return p.slope * conc;
    case "emax":
    case "sigmoid": {
      const ec50 = p.ec50 > 1e-9 ? p.ec50 : 1e-9;
      if (model === "emax") {
        return p.e0 + (p.emax * conc) / (ec50 + conc);
      }
      const h = p.hill > 1e-6 ? p.hill : 1e-6;
      const powered = Math.pow(conc, h);
      return p.e0 + (p.emax * powered) / (Math.pow(ec50, h) + powered);
    }
    case "inhibitory": {
      const ic50 = p.ic50 > 1e-9 ? p.ic50 : 1e-9;
      return p.e0 - (p.imax * conc) / (ic50 + conc);
    }
  }
}

/** Half-maximum effect on the effect axis (Emax family only). */
export function pdHalfMaxEffect(
  model: PdModelId,
  p: PdParamsLike,
): number | null {
  if (model === "emax" || model === "sigmoid") return p.e0 + p.emax / 2;
  if (model === "inhibitory") return p.e0 - p.imax / 2;
  return null;
}

/** Potency concentration (EC50/IC50) in mg/L — null for the linear model. */
export function pdPotency(model: PdModelId, p: PdParamsLike): number | null {
  if (model === "emax" || model === "sigmoid") return p.ec50 > 0 ? p.ec50 : null;
  if (model === "inhibitory") return p.ic50 > 0 ? p.ic50 : null;
  return null;
}

/**
 * Total time the concentration stays above a threshold, integrated over a
 * sampled series with linear interpolation of the crossing points (exact
 * for piecewise-linear concentration profiles).
 */
export function timeAbove(
  series: Array<{ t: number; c: number }>,
  threshold: number,
): number {
  let total = 0;
  for (let i = 1; i < series.length; i++) {
    const a = series[i - 1];
    const b = series[i];
    const dt = b.t - a.t;
    if (dt <= 0) continue;
    const aAbove = a.c > threshold;
    const bAbove = b.c > threshold;
    if (aAbove && bAbove) {
      total += dt;
    } else if (aAbove !== bAbove) {
      // linear crossing: fraction of the interval below→above
      const frac = (threshold - a.c) / (b.c - a.c);
      total += aAbove ? dt * (1 - frac) : dt * frac;
    }
  }
  return total;
}

/* ------------------------------------------------------------------ */
/* NCA over observed data                                              */
/* ------------------------------------------------------------------ */

export interface NcaPoint {
  t: number;
  c: number;
}

export interface NcaWarning {
  level: "amber" | "destructive";
  text: string;
}

export interface NcaFit {
  /** Number of terminal points actually used (≤ requested N). */
  n: number;
  /** Terminal rate constant λz (1/h); null when no valid fit. */
  ke: number | null;
  /** Terminal half-life (h); null when no valid fit. */
  halfLife: number | null;
  /** r² of the log-linear regression; null when not computable. */
  r2: number | null;
  /** Time of the first terminal point used (h). */
  from: number | null;
  /** Fitted tail amplitude: C(t) = a0·e^(−ke·t). */
  a0: number | null;
}

export interface NcaResult {
  /** Sorted-by-time observation points (chart order). */
  points: NcaPoint[];
  /** Indices (into `points`) of the terminal-fit points. */
  terminalIdx: number[];
  fit: NcaFit;
  /** Linear trapezoid AUC over the observed window (mg·h/L). */
  aucLast: number | null;
  cMax: number | null;
  tMax: number | null;
  /** Last quantifiable (positive) concentration + its time. */
  clast: number | null;
  tlast: number | null;
  /** Clast/ke extrapolated area (mg·h/L). */
  aucExtrap: number | null;
  /** AUC0–∞ = AUC0–last + extrapolated (mg·h/L). */
  aucInf: number | null;
  /** Extrapolated fraction of AUC∞ (%). */
  pctExtrap: number | null;
  /** CL = F·Dose/AUC∞ (L/h) — only with a supplied dose. */
  cl: number | null;
  /** Vz = CL/ke (L) — terminal-phase volume. */
  vz: number | null;
  /** Tail endpoint for the chart (where the fit hits 0.01·Cmax). */
  tailEnd: number | null;
  warnings: NcaWarning[];
}

/** Ordinary least squares of y on x — {slope, intercept, r²}. */
function linregress(
  xs: number[],
  ys: number[],
): { slope: number; intercept: number; r2: number | null } {
  const n = xs.length;
  if (n === 0) return { slope: 0, intercept: 0, r2: null };
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  if (sxx <= 0) {
    // all x identical (duplicate timestamps): slope undefined
    return { slope: 0, intercept: my, r2: null };
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  if (syy <= 0) return { slope, intercept, r2: 1 }; // perfect flat fit
  const r2 = Math.max(0, Math.min(1, (sxy * sxy) / (sxx * syy)));
  return { slope, intercept, r2 };
}

export interface NcaOptions {
  /** Terminal points for the log-linear fit (clamped to what's available). */
  terminalN: number;
  /** Administered dose (mg); null/≤0 hides CL/Vz. */
  dose: number | null;
  /** Absolute bioavailability fraction (oral); 1 for IV. */
  f: number;
}

/**
 * Non-compartmental analysis of one observed profile.
 *
 * Semantics mirror pharmaml.pharma: AUC by linear trapezoid
 * (auc_trapezoid), terminal ke by log-linear regression on the last
 * positive points (nca_summary uses the last 2; here N is selectable —
 * N=2 reproduces nca_summary exactly), t½ = ln2/ke, AUC∞ = AUClast +
 * Clast/ke, CL = F·Dose/AUC∞ (clearance), Vz = CL/ke
 * (volume_of_distribution). Terminal candidates are positive points
 * strictly after Tmax; degenerate inputs produce warnings + null metrics
 * instead of NaN.
 */
export function computeNca(
  raw: NcaPoint[],
  opts: NcaOptions,
): NcaResult {
  const warnings: NcaWarning[] = [];

  // Sort by time (with a notice) so trapezoids never see negative dt.
  let points = raw;
  const unsorted = raw.some((p, i) => i > 0 && p.t < raw[i - 1].t);
  if (unsorted) {
    points = [...raw].sort((a, b) => a.t - b.t);
    warnings.push({
      level: "amber",
      text: "Times were not ascending — rows were auto-sorted by time before analysis (order in the table is preserved for editing).",
    });
  }

  // Duplicate timestamps poison the terminal regression (Δt = 0). Keep the
  // last occurrence for the maths and warn.
  const seen = new Map<number, NcaPoint>();
  let duplicates = 0;
  for (const p of points) {
    if (seen.has(p.t)) duplicates += 1;
    seen.set(p.t, p); // last occurrence wins
  }
  const dedup = [...seen.values()];
  if (duplicates > 0) {
    warnings.push({
      level: "amber",
      text: `${duplicates} duplicate timestamp${duplicates === 1 ? "" : "s"} — the last concentration at each time is used for the analysis.`,
    });
  }

  const base: NcaResult = {
    points,
    terminalIdx: [],
    fit: { n: 0, ke: null, halfLife: null, r2: null, from: null, a0: null },
    aucLast: null,
    cMax: null,
    tMax: null,
    clast: null,
    tlast: null,
    aucExtrap: null,
    aucInf: null,
    pctExtrap: null,
    cl: null,
    vz: null,
    tailEnd: null,
    warnings,
  };

  if (dedup.length < 2) {
    warnings.push({
      level: "destructive",
      text: "At least two valid observations (time + concentration) are required for NCA.",
    });
    return base;
  }

  // AUC0–last: linear trapezoid over the deduplicated, sorted series
  // (zero-width duplicate slivers contribute nothing, matching Python).
  let auc = 0;
  for (let i = 1; i < dedup.length; i++) {
    auc += (0.5 * (dedup[i - 1].c + dedup[i].c) * (dedup[i].t - dedup[i - 1].t));
  }
  const cMax = Math.max(...dedup.map((p) => p.c));
  const tMax = dedup.reduce((best, p) => (p.c > best.c ? p : best), dedup[0]).t;

  // Last quantifiable (positive) concentration.
  let tlast: number | null = null;
  let clast: number | null = null;
  for (let i = dedup.length - 1; i >= 0; i--) {
    if (dedup[i].c > 0) {
      clast = dedup[i].c;
      tlast = dedup[i].t;
      break;
    }
  }
  if (clast === null || tlast === null) {
    warnings.push({
      level: "destructive",
      text: "No positive concentrations — AUC/Cmax exist but the terminal phase cannot be characterised.",
    });
    return { ...base, aucLast: auc, cMax, tMax };
  }
  if (tlast < dedup[dedup.length - 1].t) {
    warnings.push({
      level: "amber",
      text: "Trailing non-positive observations are ignored for the terminal fit and extrapolation (last quantifiable C at t = " +
        `${tlast.toFixed(2)} h).`,
    });
  }

  // Terminal candidates: positive points strictly after Tmax.
  const cand = dedup
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p.c > 0 && p.t > tMax);
  const nAvail = cand.length;
  const wanted = Math.max(2, Math.min(opts.terminalN, 6));
  const chosen = cand.slice(-wanted);
  const terminalIdx = chosen.map(({ i }) => points.indexOf(dedup[i]));

  const result: NcaResult = {
    ...base,
    aucLast: auc,
    cMax,
    tMax,
    clast,
    tlast,
    terminalIdx,
  };

  if (nAvail < 2) {
    warnings.push({
      level: "destructive",
      text:
        nAvail === 0 && dedup.every((p, i) => i === 0 || p.c >= dedup[i - 1].c)
          ? "Concentrations only increase — there is no terminal (declining) phase to fit."
          : "Fewer than 2 positive observations after Tmax — the terminal phase cannot be characterised.",
    });
    return result;
  }
  if (chosen.length < wanted) {
    warnings.push({
      level: "amber",
      text: `Only ${chosen.length} terminal point${chosen.length === 1 ? "" : "s"} available after Tmax (requested ${wanted}).`,
    });
  }

  // Log-linear regression on the terminal points.
  const xs = chosen.map(({ p }) => p.t);
  const ys = chosen.map(({ p }) => Math.log(p.c));
  const { slope, intercept, r2 } = linregress(xs, ys);
  const declining = slope < 0 && Number.isFinite(slope);
  if (!declining) {
    warnings.push({
      level: "destructive",
      text: "The selected terminal points do not decline (log-linear slope ≥ 0) — ke/t½ are not reported.",
    });
    return {
      ...result,
      fit: { n: chosen.length, ke: null, halfLife: null, r2, from: xs[0], a0: null },
    };
  }

  const ke = -slope;
  const halfLife = Math.log(2) / ke;
  const a0 = Math.exp(intercept);
  const aucExtrap = clast / ke;
  const aucInf = auc + aucExtrap;
  const pctExtrap = (100 * aucExtrap) / aucInf;

  if (r2 !== null && r2 < 0.9) {
    warnings.push({
      level: "amber",
      text: `Terminal fit r² = ${r2.toFixed(3)} — the last ${chosen.length} points are not log-linear (noisy tail?). Try a different terminal-point count.`,
    });
  }
  if (pctExtrap > 20) {
    warnings.push({
      level: "amber",
      text: `${pctExtrap.toFixed(1)}% of AUC∞ is extrapolated (> 20%) — sampling does not capture the tail well; treat AUC∞, CL and Vz with caution.`,
    });
  }

  // Chart tail: extend the fitted exponential to where C ≈ 0.01·Cmax.
  const tailEnd = Math.log(a0 / (0.01 * Math.max(cMax, 1e-12))) / ke;

  let cl: number | null = null;
  let vz: number | null = null;
  if (opts.dose !== null && opts.dose > 0 && Number.isFinite(opts.dose)) {
    const f = opts.f > 0 ? Math.min(opts.f, 1) : 1;
    cl = (f * opts.dose) / aucInf; // pharmaml.pharma.clearance semantics
    vz = cl / ke; // volume_of_distribution(CL, ke)
  }

  return {
    ...result,
    fit: { n: chosen.length, ke, halfLife, r2, from: xs[0], a0 },
    aucExtrap,
    aucInf,
    pctExtrap,
    cl,
    vz,
    tailEnd: Number.isFinite(tailEnd) && tailEnd > (tlast ?? 0) ? tailEnd : null,
  };
}

/* ------------------------------------------------------------------ */
/* NCA preset datasets (numbers pinned above — see the probe header)   */
/* ------------------------------------------------------------------ */

export interface NcaPreset {
  id: string;
  label: string;
  description: string;
  dose: string;
  f: string;
  rows: Array<[string, string]>;
}

export const NCA_PRESETS: NcaPreset[] = [
  {
    id: "oral",
    label: "Clean oral",
    description: "1-cmt oral (ka 1.2, ke 0.08) — 12 tidy samples to 48 h",
    dose: "250",
    f: "0.9",
    rows: [
      ["0", "0"],
      ["0.25", "1.28"],
      ["0.5", "2.21"],
      ["1", "3.33"],
      ["2", "4.08"],
      ["4", "3.85"],
      ["6", "3.31"],
      ["8", "2.82"],
      ["12", "2.05"],
      ["24", "0.79"],
      ["36", "0.30"],
      ["48", "0.12"],
    ],
  },
  {
    id: "iv",
    label: "IV bolus",
    description: "Pure exponential decay (ke 0.25) from C0 = 12 mg/L",
    dose: "500",
    f: "1",
    rows: [
      ["0", "12"],
      ["0.5", "10.59"],
      ["1", "9.35"],
      ["2", "7.28"],
      ["4", "4.41"],
      ["6", "2.68"],
      ["8", "1.62"],
      ["10", "0.99"],
      ["12", "0.60"],
    ],
  },
  {
    id: "noisy",
    label: "Noisy oral",
    description: "Same oral profile + assay noise — peak flips to t = 4 h",
    dose: "250",
    f: "0.9",
    rows: [
      ["0", "0"],
      ["0.25", "1.35"],
      ["0.5", "2.05"],
      ["1", "3.45"],
      ["2", "3.98"],
      ["4", "4.12"],
      ["6", "3.18"],
      ["8", "2.90"],
      ["12", "1.98"],
      ["24", "0.85"],
      ["36", "0.34"],
      ["48", "0.08"],
    ],
  },
];
