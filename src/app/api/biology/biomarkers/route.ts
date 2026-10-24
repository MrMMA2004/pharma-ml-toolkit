import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { gaussian, mulberry32 } from "@/lib/ml-math";
import { rankVariableGenes } from "@/lib/variable-genes";
import type {
  BiomarkerRow,
  BiomarkerScatter,
  BiomarkersResponse,
  BiologyCompound,
} from "@/lib/toolkit-types";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/* ------------------------------------------------------------------ */
/* Synthetic GDSC/CCLE panel — mirrors examples/make_synthetic_cellline */
/* _data.py from the Python package, so the web playground shows the   */
/* same planted-EGFR story the library demo recovers.                  */
/* ------------------------------------------------------------------ */

/** Canonical panel: (cell line, tissue), shared by screen + expression. */
const CELL_LINES: Array<[string, string]> = [
  ["A549", "lung"],
  ["NCI-H1299", "lung"],
  ["NCI-H1975", "lung"],
  ["MCF-7", "breast"],
  ["MDA-MB-231", "breast"],
  ["T-47D", "breast"],
  ["HCT-116", "colon"],
  ["HT-29", "colon"],
  ["SW-480", "colon"],
  ["HL-60", "blood"],
  ["K-562", "blood"],
  ["A375", "skin"],
  ["SK-MEL-28", "skin"],
  ["PANC-1", "pancreas"],
  ["MIA-PaCa-2", "pancreas"],
  ["Hep-G2", "liver"],
  ["Huh-7", "liver"],
  ["OVCAR-3", "ovary"],
  ["SK-OV-3", "ovary"],
  ["PC-3", "prostate"],
  ["DU-145", "prostate"],
  ["U-87-MG", "brain"],
  ["U-251", "brain"],
  ["SH-SY5Y", "brain"],
];

/** Per-compound LN_IC50 baselines (natural log of µM IC50). */
const COMPOUNDS: Record<BiologyCompound, number> = {
  erlotinib: 1.9,
  gefitinib: 2.1,
  cisplatin: 3.4,
  doxorubicin: 1.6,
  paclitaxel: 2.0,
  "5-fluorouracil": 3.9,
  gemcitabine: 2.8,
  sorafenib: 3.1,
};

const COMPOUND_IDS = Object.keys(COMPOUNDS) as BiologyCompound[];

/** EGFR inhibitors: higher EGFR → lower LN_IC50 (the planted biomarker). */
const EGFR_DRUGS: Partial<Record<BiologyCompound, number>> = {
  erlotinib: -0.75,
  gefitinib: -0.55,
};

/** Compounds with a tissue-specific potency shift. */
const TISSUE_EFFECTS: Partial<Record<BiologyCompound, Record<string, number>>> = {
  cisplatin: { blood: -0.6, skin: 0.4 },
  paclitaxel: { prostate: -0.5, breast: -0.3 },
  gemcitabine: { pancreas: -0.7 },
  sorafenib: { liver: -0.5 },
  "5-fluorouracil": { colon: -0.4 },
};

/** EGFR levels are a property of the cell line, drawn from their OWN fixed
 * seed so the response screen and the expression matrix agree on them
 * regardless of the caller's seed (same trick as the Python generator). */
const EGFR_SEED = 2024;

/** AUC tracks LN_IC50 through this sigmoid: sensitive (low IC50) → low AUC. */
function aucFromLnIc50(lnIc50: number): number {
  return 0.95 - 0.55 / (1.0 + Math.exp(-(3.2 - lnIc50)));
}

/* ------------------------------------------------------------------ */
/* Statistics helpers (Spearman + BH FDR)                              */
/* ------------------------------------------------------------------ */

/**
 * Average ranks (1..n) with ties sharing their mean rank — the standard
 * midrank convention that makes Pearson-on-ranks equal Spearman's rho.
 */
function averageRanks(values: number[]): number[] {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(values.length);
  let i = 0;
  while (i < order.length) {
    let j = i;
    while (j + 1 < order.length && order[j + 1].v === order[i].v) j++;
    const avg = (i + j) / 2 + 1; // ranks are 1-based
    for (let k = i; k <= j; k++) ranks[order[k].i] = avg;
    i = j + 1;
  }
  return ranks;
}

/** Mean and population std with a zero-variance guard. */
function meanStd(values: number[]): { mu: number; sigma: number } {
  const n = values.length;
  const mu = values.reduce((a, b) => a + b, 0) / n;
  const sigma = Math.sqrt(values.reduce((a, b) => a + (b - mu) ** 2, 0) / n);
  return { mu, sigma: sigma > 1e-12 ? sigma : 1 };
}

/** Pearson correlation; 0 when either side is constant. */
function pearson(xs: number[], ys: number[]): number {
  const { mu: mx, sigma: sx } = meanStd(xs);
  const { mu: my, sigma: sy } = meanStd(ys);
  let cov = 0;
  for (let i = 0; i < xs.length; i++) cov += (xs[i] - mx) * (ys[i] - my);
  const denom = (xs.length - 1) * sx * sy;
  return denom > 1e-12 ? cov / denom : 0;
}

/** Σ (t³ − t) / (n³ − n) over tie groups — the classic tie correction term. */
function tieTerm(values: number[]): number {
  const n = values.length;
  const counts = new Map<number, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let sum = 0;
  for (const t of counts.values()) sum += t ** 3 - t;
  return sum / (n ** 3 - n);
}

/** Φ(|z|) via the Abramowitz–Stegun 7.1.26 erf approximation (|ε| < 1.5e-7). */
function normalCdfAbs(z: number): number {
  const t = 1 / (1 + 0.2316419 * z);
  const poly =
    t *
    (0.319381530 +
      t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return 1 - (Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)) * poly;
}

/**
 * Two-sided Spearman p-value, normal approximation with tie correction.
 *
 * Under H0 the asymptotic variance of rho is
 *   Var = (1/(n−1)) · (1 − Tx) · (1 − Ty),
 * where T = Σ(t³−t)/(n³−n) sums over the tie groups of each variable
 * (Kendall, Advanced Theory of Statistics). With no ties this reduces to
 * the familiar 1/(n−1). z = rho/√Var and p = 2·(1 − Φ(|z|)). For n ≈ 24
 * this is slightly conservative versus an exact permutation or t-based p,
 * which is the safe direction for a discovery table.
 */
function spearmanTwoSidedP(rho: number, xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 4) return 1;
  const varR = (1 / (n - 1)) * (1 - tieTerm(xs)) * (1 - tieTerm(ys));
  if (varR <= 1e-12) return 1;
  const z = Math.abs(rho) / Math.sqrt(varR);
  if (!Number.isFinite(z)) return 0;
  return Math.min(1, Math.max(0, 2 * (1 - normalCdfAbs(z))));
}

/**
 * Benjamini–Hochberg step-up FDR: adjusted p for each gene, enforcing the
 * monotonicity fdr_(i) = min(fdr_(i+1), p_(i) · m / i) from the largest
 * p down (the standard cumulative-min construction), clipped to [0, 1].
 * Returns adjusted values in the ORIGINAL input order.
 */
function benjaminiHochberg(pValues: number[]): number[] {
  const m = pValues.length;
  const order = pValues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p);
  const adjusted = new Array<number>(m);
  let runningMin = 1;
  for (let k = m - 1; k >= 0; k--) {
    const rank = k + 1;
    const raw = (order[k].p * m) / rank;
    runningMin = Math.min(runningMin, raw, 1);
    adjusted[order[k].i] = runningMin;
  }
  return adjusted;
}

/**
 * Rank-space fitted trend, mapped back to value space for the chart:
 * least squares of rank(y) on rank(x), then each predicted rank is
 * converted to the empirical y-quantile at that fractional position —
 * a monotone curve showing the fitted rank relationship on value axes.
 */
function rankTrendLine(
  points: Array<{ value: number; lnIc50: number }>
): Array<{ x: number; y: number }> {
  const n = points.length;
  if (n < 2) return [];
  const rx = averageRanks(points.map((p) => p.value));
  const ry = averageRanks(points.map((p) => p.lnIc50));
  const { mu: mx } = meanStd(rx);
  const { mu: my } = meanStd(ry);
  let sxy = 0;
  let sxx = 0;
  for (let i = 0; i < n; i++) {
    sxy += (rx[i] - mx) * (ry[i] - my);
    sxx += (rx[i] - mx) ** 2;
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = my - slope * mx;

  const sortedY = points.map((p) => p.lnIc50).sort((a, b) => a - b);
  const quantile = (rank: number): number => {
    const pos = Math.min(Math.max(rank - 1, 0), n - 1);
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sortedY[lo];
    return sortedY[lo] + (sortedY[hi] - sortedY[lo]) * (pos - lo);
  };

  return points
    .map((p, i) => ({ x: p.value, y: quantile(intercept + slope * rx[i]) }))
    .sort((a, b) => a.x - b.x);
}

/* ------------------------------------------------------------------ */
/* Generators (deterministic, mulberry32-seeded)                       */
/* ------------------------------------------------------------------ */

/** EGFR log2-expression per cell line (lung lines run hot), fixed seed. */
function egfrLevels(): Map<string, number> {
  const rand = mulberry32(EGFR_SEED);
  const levels = new Map<string, number>();
  for (const [name, tissue] of CELL_LINES) {
    levels.set(name, tissue === "lung" ? 9.0 + 0.8 * gaussian(rand) : 5.5 + 0.9 * gaussian(rand));
  }
  return levels;
}

interface ScreenRow {
  cellLine: string;
  tissue: string;
  lnIc50: number;
  auc: number;
}

/**
 * Response rows for the selected compound. Noise is drawn cell-outer /
 * compound-inner across the full 8-compound screen (the Python loop
 * order), so any single compound's numbers are stable and consistent
 * with a full-screen generation at the same seed.
 */
function makeScreen(
  compound: BiologyCompound,
  panel: Array<[string, string]>,
  seed: number,
  egfr: Map<string, number>
): ScreenRow[] {
  const rand = mulberry32(seed);
  const egfrValues = panel.map(([name]) => egfr.get(name) ?? 5.5);
  const { mu, sigma } = meanStd(egfrValues);
  const egfrZ = egfrValues.map((v) => (v - mu) / sigma);
  const egfrCoef = EGFR_DRUGS[compound] ?? 0;
  const tissueEffect = TISSUE_EFFECTS[compound] ?? {};

  const rows: ScreenRow[] = [];
  for (let cellIdx = 0; cellIdx < panel.length; cellIdx++) {
    const [cell, tissue] = panel[cellIdx];
    for (const id of COMPOUND_IDS) {
      const baseline = COMPOUNDS[id];
      let lnIc50 = baseline + (tissueEffect[tissue] ?? 0);
      if (id === compound) lnIc50 += egfrCoef * egfrZ[cellIdx];
      lnIc50 += 0.25 * gaussian(rand);
      // AUC noise is consumed in the same loop order as the Python screen.
      const auc = Math.min(1, Math.max(0, aucFromLnIc50(lnIc50) + 0.03 * gaussian(rand)));
      if (id === compound) {
        rows.push({ cellLine: cell, tissue, lnIc50, auc });
      }
    }
  }
  return rows;
}

/**
 * Expression matrix: rows = cell lines, columns = genes. Noise genes
 * GENE_001..GENE_{n−2} are normal(6, 1) drawn cell-major (the numpy
 * matrix fill order), EGFR comes from the fixed levels, and KRT8 is
 * breast-hot — exactly the Python column semantics.
 */
function makeExpression(
  panel: Array<[string, string]>,
  nGenes: number,
  seed: number,
  egfr: Map<string, number>
): Map<string, number[]> {
  const rand = mulberry32((seed + 1) >>> 0);
  const noiseCount = nGenes - 2;
  const noiseNames = Array.from(
    { length: noiseCount },
    (_, g) => `GENE_${String(g + 1).padStart(3, "0")}`
  );
  const genes = new Map<string, number[]>();
  for (const name of noiseNames) {
    genes.set(name, new Array<number>(panel.length));
  }
  for (let cellIdx = 0; cellIdx < panel.length; cellIdx++) {
    for (const name of noiseNames) {
      genes.get(name)![cellIdx] = 6.0 + 1.0 * gaussian(rand);
    }
  }
  genes.set(
    "EGFR",
    panel.map(([name]) => egfr.get(name) ?? 5.5)
  );
  genes.set(
    "KRT8",
    panel.map(([, tissue]) =>
      tissue === "breast" ? 9.0 + 0.7 * gaussian(rand) : 5.0 + 0.8 * gaussian(rand)
    )
  );
  return genes;
}

/* ------------------------------------------------------------------ */
/* Route                                                               */
/* ------------------------------------------------------------------ */

const bodySchema = z.object({
  compound: z.enum([
    "erlotinib",
    "gefitinib",
    "cisplatin",
    "doxorubicin",
    "paclitaxel",
    "5-fluorouracil",
    "gemcitabine",
    "sorafenib",
  ]),
  genes: z.number().int().min(60).max(600).default(300),
  seed: z.number().int().min(0).max(2_147_483_647).default(42),
  nCellLines: z.number().int().min(6).max(24).default(24),
});

/** Rows shown in the ranked biomarker table. */
const TOP_N = 15;

export async function POST(request: NextRequest) {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const parsed = bodySchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json(
      {
        error: "Invalid request body.",
        detail: parsed.error.issues.map((i) => `${i.path.join(".") || "body"}: ${i.message}`),
      },
      { status: 400 }
    );
  }
  const req = parsed.data;
  const started = Date.now();

  try {
    const panel = CELL_LINES.slice(0, req.nCellLines);
    const egfr = egfrLevels();

    /* Screen rows for the selected compound (deterministic in the seed). */
    const screen = makeScreen(req.compound, panel, req.seed, egfr);
    const lnIc50 = screen.map((r) => r.lnIc50);

    /* Expression matrix — independent stream (seed + 1), Python-style. */
    const expression = makeExpression(panel, req.genes, req.seed, egfr);

    /* Per-gene Spearman vs the compound's LN_IC50. */
    const geneNames = [...expression.keys()];
    const rhos: number[] = [];
    const pValues: number[] = [];
    for (const gene of geneNames) {
      const values = expression.get(gene);
      if (!values) continue;
      const rho = pearson(averageRanks(values), averageRanks(lnIc50));
      rhos.push(rho);
      pValues.push(spearmanTwoSidedP(rho, values, lnIc50));
    }

    /* BH FDR across all screened genes. */
    const fdrs = benjaminiHochberg(pValues);

    const rows: BiomarkerRow[] = geneNames.map((gene, i) => ({
      rank: 0,
      gene,
      rho: rhos[i] ?? 0,
      pValue: pValues[i] ?? 1,
      fdr: fdrs[i] ?? 1,
      significant: (fdrs[i] ?? 1) < 0.05,
    }));
    rows.sort((a, b) => {
      const aAbs = Math.abs(a.rho);
      const bAbs = Math.abs(b.rho);
      if (aAbs !== bAbs) return bAbs - aAbs;
      if (a.pValue !== b.pValue) return a.pValue - b.pValue;
      return a.gene.localeCompare(b.gene);
    });
    rows.forEach((r, i) => {
      r.rank = i + 1;
    });

    const egfrRow = rows.find((r) => r.gene === "EGFR") ?? null;
    const topRow = rows[0];
    if (!topRow) throw new Error("Expression matrix has no genes.");

    const scatterFor = (row: BiomarkerRow): BiomarkerScatter => {
      const values = expression.get(row.gene) ?? [];
      return {
        gene: row.gene,
        rho: row.rho,
        significant: row.significant,
        points: screen.map((r, i) => ({
          cellLine: r.cellLine,
          tissue: r.tissue,
          value: values[i] ?? 0,
          lnIc50: r.lnIc50,
          auc: r.auc,
        })),
        trend: rankTrendLine(
          screen.map((r, i) => ({ value: values[i] ?? 0, lnIc50: r.lnIc50 }))
        ),
      };
    };

    const tissues: string[] = [];
    for (const [, tissue] of panel) {
      if (!tissues.includes(tissue)) tissues.push(tissue);
    }

    const result: BiomarkersResponse = {
      biomarkers: rows.slice(0, TOP_N),
      /* (100x-124) the MAD variability pre-filter — the TS twin of
       * pharmaml.biology.most_variable_genes, computed over the SAME
       * expression matrix (parity-guarded row-by-row). */
      variableGenes: rankVariableGenes(expression, TOP_N),
      egfr: egfrRow,
      topHit: scatterFor(topRow),
      egfrScatter: egfrRow ? scatterFor(egfrRow) : null,
      tissues,
      meta: {
        nGenes: req.genes,
        nCellLines: panel.length,
        compound: req.compound,
        plantedHit: egfrRow !== null && egfrRow.rank === 1 && egfrRow.significant ? "EGFR" : null,
        nSignificant: rows.filter((r) => r.significant).length,
        strongestRho: Math.abs(topRow.rho),
        seed: req.seed,
        runtimeMs: Date.now() - started,
      },
    };
    return NextResponse.json(result);
  } catch (err) {
    return NextResponse.json(
      {
        error: "Biomarker discovery failed.",
        detail: err instanceof Error ? err.message : String(err),
      },
      { status: 500 }
    );
  }
}
