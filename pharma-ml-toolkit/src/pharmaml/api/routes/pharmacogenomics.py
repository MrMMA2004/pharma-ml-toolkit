"""``/api/v1/pharmacogenomics`` — the pharmacogenomic-association lane
over the REAL in-tree backend (part13 Phase 5's next remainder item —
the pharmacogenomics API lane over :mod:`pharmaml.ml.pharmacogenomics`).

* ``GET /api/v1/pharmacogenomics/registry`` — the module's own
  vocabulary verbatim: the classic GDSC ANOVA design matrix ``[1 |
  tissue dummies (reference = first sorted level) | msi | feature]``
  with its sequential QR F-test and the feasibility gates (a drug
  needs ``>= 6`` non-NA IC50s, a feature ``>= 3`` positives AND ``>= 3``
  negatives among the drug's screened lines, the MSI factor ``>= 2``
  each side), the effect-size contract (Cohen's d on the pooled
  ddof=1 variance, Glass Δ, the Δ-mean, the equal-variance t-test),
  the multiple-testing layer (Benjamini-Hochberg step-up FDR +
  Storey-Tibshirani q-values over the λ grid, gdsctools' **percent**
  convention), the volcano transforms with gdsctools' default hit
  rule, the drevalpy evaluation guards (a constant *prediction*
  scores 0, a constant *target* or fewer than two pairs is NaN) and
  the additive mean-effects ruler with the *normalized* metric (the
  score on the residuals beyond the naive ruler), the cBioPortal
  vocabularies (the discrete CNA codes ``-2``..``+2``, the
  diploid-reference z-scores, the co-expression closed form with the
  0.3 default |rho| cutoff, the enrichment dispatchers' exact
  branching with the NaN → 1 convention), and the honest provenance
  block: five repositories reviewed file by file, zero real
  GDSC/CCLE/CTRP/DepMap tables ship, and the served screen is the
  backend's own SYNTHETIC seeded demo with planted associations.
* ``POST /api/v1/pharmacogenomics/analyze`` — the full honest round
  trip over the real backend: the seeded synthetic screen, the real
  GDSC ANOVA over every feasible drug x feature pair, the
  Storey-Tibshirani q-value + BH FDR columns, the volcano verdicts,
  the top associations table, and the mean-effects ruler evaluated
  through the guarded metric family with the normalized score beyond
  the naive drug-mean ruler.  Sync-only light compute (a 40x12
  screen, dense small matrices); every p-value and effect size is a
  statistic over the synthetic screen — one response scale only (the
  natural-log µM scale, GDSC's ``LN_IC50``; ``IC50 = exp(LN_IC50)``
  µM at readout), never a clinical recommendation.

The honest notes everywhere: the associations are triage hypotheses
on a synthetic stand-in screen; no real cell line, expression panel,
mutation panel or drug-response table ships with this route.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from pharmaml.ml.pharmacogenomics import (
    CNA_VALUES,
    MeanEffectsPredictor,
    demo_pharmacogenomics,
    gdsc_anova,
    guarded_kendall,
    guarded_pearson,
    guarded_spearman,
    naive_group_means,
    normalized_metric,
    volcano_points,
)

router = APIRouter(prefix="/pharmacogenomics", tags=["pharmacogenomics"])

_MAX_TOP_K = 20
_MAX_SEED = 2147483647


class PharmacogenomicsAnalyzeRequest(BaseModel):
    """One seeded synthetic screen + ANOVA + baseline-evaluation request.

    The screen geometry is the backend's own demo contract (40 cell
    lines x 12 drugs x 10 binary features, ~15 % untested); the seed
    is the only generative dial.  ``include_msi`` mirrors
    ``gdsc_anova``'s own parameter (the MSI factor rides the design
    matrix when both sides carry ``>= 2`` lines).
    """

    seed: int = Field(default=42, ge=0, le=_MAX_SEED)
    top_k: int = Field(default=5, ge=1, le=_MAX_TOP_K)
    include_msi: bool = Field(default=True)


def _provenance_block() -> dict[str, Any]:
    """The provenance + limitations block (the §11.1-honest state)."""
    return {
        "backend": "pharmaml.ml.pharmacogenomics",
        "source": (
            "five repositories reviewed file by file — CancerRxGene/"
            "gdsctools (the classic GDSC ANOVA, the qvalue procedure, "
            "the IC50Cluster replicate rule, the volcano transforms); "
            "TangYiChing/GDSC_DataParser (the GDSC2 EXP1/EXP2 "
            "dose-range branching, the replicate-mean collapse, the "
            "IC50 = exp(LN_IC50) recovery); daisybio/drevalpy (the "
            "metric guards, the naive mean-effects predictors, the "
            "normalized metrics, the GroupMeanCenterer); CBIIT/"
            "cellminercdb (the pattern-comparison and per-tissue "
            "correlation queries with BH FDR over the whole table); "
            "cBioPortal/cbioportal (the discrete CNA vocabulary, the "
            "diploid-reference z-scores, the co-expression closed "
            "form, the enrichment dispatchers)"
        ),
        "n_repositories": 5,
        "shipped_tables": 0,
        "shipped_tables_note": (
            "no real GDSC, CCLE, CTRP or DepMap expression, mutation "
            "or drug-response table ships with the toolkit — every "
            "constant the module transcribes is a documented "
            "transcription, never a bundled dataset"
        ),
        "shipped_checkpoints": 0,
        "shipped_checkpoints_note": (
            "no pretrained predictor ships — this lane ships the "
            "statistical association layer itself (the ANOVA + the "
            "multiple-testing + the guards), which runs in-tree over "
            "the seeded screen; there is no learned-model surface to "
            "checkpoint"
        ),
        "screen": (
            "the served screen is the backend's own SYNTHETIC seeded "
            "demo (demo_pharmacogenomics: 40 cell lines x 12 drugs on "
            "the natural-log µM scale, ~15 % untested, ten binary "
            "features at ~40 % positivity with TP53_mut (-0.8 on five "
            "drugs) and KRAS_mut (-0.4 on three) planted, one "
            "resistant +0.8 plant, three tissues with a lung effect, "
            "MSI in ~10 % of lines) — the stand-in for a real "
            "GDSC-sized screen"
        ),
        "disclaimer": (
            "p-values, q-values, effect sizes and rankings are "
            "statistics over the synthetic screen — associations are "
            "triage hypotheses, never clinical recommendations and "
            "never real epidemiology"
        ),
    }


def _finite(value: float) -> float | None:
    """JSON-safe float (NaN/inf metrics read as null, never as lies)."""
    number = float(value)
    return number if math.isfinite(number) else None


@router.get("/registry")
def registry(request: Request) -> dict:
    return {
        "associations": {
            "family": "the classic GDSC ANOVA (gdsctools' anova.py)",
            "design_matrix": (
                "[1 | tissue dummies (reference = first sorted level) "
                "| msi | feature] — the sequential QR F-test: Q, _ = "
                "qr(X); the factor sum-of-squares is the sum of squared "
                "projected effects over its columns; F = MS / (SSR / "
                "df_resid)"
            ),
            "feasibility_gates": {
                "drug": ">= 6 non-NA IC50 values (MINIMUM_NONNA_IC50)",
                "feature": (
                    ">= 3 positives AND >= 3 negatives among the "
                    "drug's screened lines (FEATURE_FACTOR_THRESHOLD)"
                ),
                "msi": (
                    "the MSI factor rides only when both sides carry "
                    ">= 2 lines (MSI_FACTOR_THRESHOLD); otherwise it "
                    "is dropped from the design"
                ),
                "tissue": (
                    "the tissue factor is dropped when a single "
                    "tissue remains among the screened lines"
                ),
            },
            "row_schema": [
                "drug", "feature", "n_pos", "n_neg", "mean_pos",
                "mean_neg", "delta_mean", "sd_pos", "sd_neg",
                "effect_size", "glass_pos", "glass_neg", "t_pvalue",
                "pvalue", "tissue_pvalue", "msi_pvalue", "qvalue",
                "fdr_percent",
            ],
            "ordering": "sorted by pvalue ascending, stable (ties keep the walk order)",
        },
        "effect_sizes": {
            "keys": ["effect_size", "glass_pos", "glass_neg", "delta_mean", "t_pvalue"],
            "note": (
                "Cohen's d on the pooled ddof=1 variance, Glass Δ "
                "against each side's reference sd, the raw Δ-mean, "
                "and the equal-variance t-test p-value "
                "(EQUAL_VAR_TTEST = True — gdsctools' default)"
            ),
        },
        "multiple_testing": {
            "methods": ["bh_fdr", "qvalue"],
            "bh_fdr": (
                "Benjamini-Hochberg step-up FDR: sort ascending, the "
                "running threshold is rank/m · alpha with the "
                "monotone back-substitution, clipped to [0, 1]"
            ),
            "qvalue": (
                "Storey-Tibshirani q-values: the λ grid "
                "arange(0, 0.9 + 1e-8, 0.05) (19 values), "
                "pi0(λ) = mean(p >= λ)/(1 - λ) per λ, a cubic spline "
                "(df = 3) extrapolated to λ = 0.9, clipped to "
                "[1/m, 1]; the q-values are the BH back-substitution "
                "scaled by pi0"
            ),
            "percent_convention": (
                "gdsctools' convention: FDR_THRESHOLD = 25 means 25 % "
                "— the fdr_percent column carries the BH FDR in "
                "percent"
            ),
        },
        "volcano": {
            "transforms": (
                "x = sign(delta_mean) · effect_size (the signed "
                "Cohen's d); y = -log10(max(pvalue, 1e-300))"
            ),
            "hit_rule": (
                "gdsctools' default conjunction: pvalue <= 0.001 "
                "AND fdr_percent <= 25 AND |signed effect| >= 0"
            ),
            "verdicts": (
                "sensitive (green) when the signed effect is "
                "negative, resistant (red) when positive, otherwise "
                "'not significant'"
            ),
            "additional_fdr_lines": [0.01, 0.1, 10.0],
        },
        "evaluation": {
            "ruler": (
                "drevalpy's NaiveMeanEffectsPredictor: y ~ overall + "
                "tissue + cell-residual + drug, where tissue_effect = "
                "mean(y|t) - overall, cell_line_residual = mean(y|c) - "
                "mean(y|tissue(c)) (the residual after tissue — no "
                "double counting) and drug_effect = mean(y|d) - "
                "overall; unknown entities contribute 0"
            ),
            "guarded_metrics": ["pearson", "spearman", "kendall"],
            "guards": (
                "drevalpy's degenerate-input semantics: a constant "
                "PREDICTION is an uninformative model and scores 0.0; "
                "a constant TARGET or fewer than two pairs is "
                "undefined (NaN); the tolerance is 1e-6 — a constant "
                "target never reads as a perfect score and a constant "
                "prediction never reads as an error"
            ),
            "normalized_metric": (
                "metric(y_true - y_naive, y_pred - y_naive) — the "
                "model scored on what it predicts BEYOND the naive "
                "ruler; the analyze pass reports it with the naive "
                "drug-mean ruler as the baseline"
            ),
        },
        "vocabulary": {
            "cna_values": {str(k): v for k, v in CNA_VALUES.items()},
            "diploid_zscores": (
                "per gene the reference population is the diploid "
                "samples; empty or zero-variance references map to NaN"
            ),
            "coexpression": (
                "the closed form t = r·sqrt((n-2)/(1-r²)) with the "
                "0.3 default |rho| cutoff (COEXPRESSION_THRESHOLD)"
            ),
            "enrichment_dispatchers": (
                "two groups → Fisher exact / t-test / Mann-Whitney; "
                "more → chi-square / ANOVA / Kruskal-Wallis; a NaN "
                "p-value reads as 1, never as a discovery"
            ),
        },
        "screen": {
            "geometry": {"n_cells": 40, "n_drugs": 12, "n_features": 10},
            "planted_associations": (
                "TP53_mut (-0.8 on five of the twelve drugs), KRAS_mut "
                "(-0.4 on three others), one resistant +0.8 plant; "
                "the remaining eight features are pure noise"
            ),
            "label": (
                "SYNTHETIC — the backend's own seeded demo screen "
                "(demo_pharmacogenomics), the demo/test stand-in for "
                "a real GDSC-sized screen, not real pharmacology"
            ),
            "scale": (
                "one response scale only — the natural-log µM scale "
                "(GDSC's LN_IC50); IC50 = exp(LN_IC50) µM at readout "
                "(ic50_from_ln); incompatible response scales are "
                "never mixed"
            ),
            "missing": "~15 % of the (cell, drug) pairs are untested (NaN)",
        },
        "provenance": _provenance_block(),
        "analyze_url": "/api/v1/pharmacogenomics/analyze",
        "note": (
            "sync-only light compute — a 40x12 screen with small "
            "dense matrices; every p-value and effect size is a "
            "statistic over the synthetic screen, and the seed is the "
            "only generative dial (the geometry is the backend's own "
            "demo contract)"
        ),
    }


@router.post("/analyze")
def analyze(req: PharmacogenomicsAnalyzeRequest, request: Request) -> dict:
    # the real backend round trip: seeded screen -> the classic GDSC
    # ANOVA over every feasible drug x feature pair -> the volcano
    # verdicts -> the top associations -> the mean-effects ruler
    # evaluated through the guarded family + the normalized score
    screen = demo_pharmacogenomics(seed=req.seed)
    results = gdsc_anova(
        screen.ic50,
        screen.features,
        screen.tissue,
        screen.msi,
        include_msi=req.include_msi,
    )
    volcano = volcano_points(results)

    # the volcano verdict census over the FULL association table
    verdicts = volcano["verdict"].tolist() if not volcano.empty else []
    n_sensitive = sum(1 for v in verdicts if v == "sensitive")
    n_resistant = sum(1 for v in verdicts if v == "resistant")
    n_not_significant = sum(1 for v in verdicts if v == "not significant")

    # the top associations table (the full gdsctools row schema)
    top_rows: list[dict[str, Any]] = []
    volcano_x = volcano["x"].tolist() if not volcano.empty else []
    volcano_y = volcano["y"].tolist() if not volcano.empty else []
    for i in range(min(req.top_k, len(results))):
        row = results.iloc[i]
        top_rows.append(
            {
                "drug": str(row["drug"]),
                "feature": str(row["feature"]),
                "n_pos": int(row["n_pos"]),
                "n_neg": int(row["n_neg"]),
                "mean_pos": _finite(row["mean_pos"]),
                "mean_neg": _finite(row["mean_neg"]),
                "delta_mean": _finite(row["delta_mean"]),
                "effect_size": _finite(row["effect_size"]),
                "glass_pos": _finite(row["glass_pos"]),
                "glass_neg": _finite(row["glass_neg"]),
                "t_pvalue": _finite(row["t_pvalue"]),
                "pvalue": _finite(row["pvalue"]),
                "qvalue": _finite(row["qvalue"]),
                "fdr_percent": _finite(row["fdr_percent"]),
                "tissue_pvalue": _finite(row["tissue_pvalue"]),
                "msi_pvalue": _finite(row["msi_pvalue"]),
                "volcano_x": _finite(volcano_x[i]) if i < len(volcano_x) else None,
                "volcano_y": _finite(volcano_y[i]) if i < len(volcano_y) else None,
                "verdict": verdicts[i] if i < len(verdicts) else None,
            }
        )

    # the mean-effects ruler over the observed entries: the long form
    # (cell, drug, ln_ic50) with the tissue factor, fit -> predict ->
    # the guarded family + the normalized score beyond the naive
    # drug-mean ruler
    long_frame = screen.ic50.stack().dropna().rename("y").reset_index()
    long_frame.columns = ["cell", "drug", "y"]
    tissue_map = screen.tissue.to_dict()
    long_frame["tissue"] = long_frame["cell"].map(tissue_map)
    y = long_frame["y"].to_numpy(dtype=float)
    drugs = long_frame["drug"].tolist()
    cells = long_frame["cell"].tolist()
    tissues = long_frame["tissue"].tolist()

    ruler = MeanEffectsPredictor().fit(y, drugs, cells, tissues)
    ruler_pred = ruler.predict(drugs, cells)

    drug_means = naive_group_means(y, drugs)
    cell_means = naive_group_means(y, cells)
    drug_mean_pred = np.asarray(
        [drug_means[str(d)] for d in drugs], dtype=float
    )
    cell_mean_pred = np.asarray(
        [cell_means[str(c)] for c in cells], dtype=float
    )

    guarded = {
        "pearson": guarded_pearson(ruler_pred, y),
        "spearman": guarded_spearman(ruler_pred, y),
        "kendall": guarded_kendall(ruler_pred, y),
    }
    naive_drug_guarded = {
        "pearson": guarded_pearson(drug_mean_pred, y),
        "spearman": guarded_spearman(drug_mean_pred, y),
        "kendall": guarded_kendall(drug_mean_pred, y),
    }
    naive_cell_guarded = {
        "pearson": guarded_pearson(cell_mean_pred, y),
        "spearman": guarded_spearman(cell_mean_pred, y),
        "kendall": guarded_kendall(cell_mean_pred, y),
    }
    normalized = {
        "pearson": normalized_metric(y, ruler_pred, drug_mean_pred, guarded_pearson),
        "spearman": normalized_metric(
            y, ruler_pred, drug_mean_pred, guarded_spearman
        ),
    }

    observed_entries = int(y.size)
    total_entries = int(screen.ic50.size)
    tissues_census = screen.tissue.value_counts().to_dict()
    msi_census = {
        "msi_positive": int((screen.msi > 0).sum()),
        "msi_negative": int((screen.msi == 0).sum()),
    }

    return {
        "request": {
            "seed": req.seed,
            "top_k": req.top_k,
            "include_msi": req.include_msi,
        },
        "screen": {
            "n_cells": int(screen.ic50.shape[0]),
            "n_drugs": int(screen.ic50.shape[1]),
            "n_features": int(screen.features.shape[1]),
            "drugs": list(screen.ic50.columns),
            "features": list(screen.features.columns),
            "tissues": {str(k): int(v) for k, v in tissues_census.items()},
            "msi": msi_census,
            "n_observed": observed_entries,
            "n_entries": total_entries,
            "observed_fraction": (
                observed_entries / total_entries if total_entries else 0.0
            ),
            "label": "SYNTHETIC seeded screen — not real pharmacology",
        },
        "associations": {
            "n_rows": len(results),
            "n_drugs_tested": int(results["drug"].nunique()) if len(results) else 0,
            "n_features_tested": (
                int(results["feature"].nunique()) if len(results) else 0
            ),
            "top_k": req.top_k,
            "top": top_rows,
            "ordering": (
                "sorted by pvalue ascending, stable — the top-k slice "
                "of the full feasible table"
            ),
        },
        "volcano": {
            "n_sensitive": n_sensitive,
            "n_resistant": n_resistant,
            "n_not_significant": n_not_significant,
            "hit_rule": (
                "pvalue <= 0.001 AND fdr_percent <= 25 AND |signed "
                "effect| >= 0 — sensitive when the signed effect is "
                "negative, resistant when positive"
            ),
        },
        "evaluation": {
            "n_observed": observed_entries,
            "ruler": (
                "the tissue-aware mean-effects ruler: overall + tissue "
                "+ cell-residual + drug (no double counting)"
            ),
            "guarded": {key: _finite(v) for key, v in guarded.items()},
            "naive_drug_mean": {
                key: _finite(v) for key, v in naive_drug_guarded.items()
            },
            "naive_cell_mean": {
                key: _finite(v) for key, v in naive_cell_guarded.items()
            },
            "normalized_beyond_drug_mean": {
                key: _finite(v) for key, v in normalized.items()
            },
            "guards_note": (
                "a constant prediction scores 0, a constant target or "
                "fewer than two pairs reads as null — never as a "
                "perfect score and never as zero"
            ),
            "normalized_note": (
                "the ruler scored on what it predicts beyond the "
                "naive drug-mean baseline: metric(y_true - y_naive, "
                "y_pred - y_naive)"
            ),
        },
        "provenance": _provenance_block(),
        "note": (
            "p-values, q-values, effect sizes and rankings are "
            "statistics over the synthetic screen — associations are "
            "triage hypotheses, never clinical recommendations; no "
            "real GDSC/CCLE/CTRP/DepMap table ships with the toolkit"
        ),
    }
