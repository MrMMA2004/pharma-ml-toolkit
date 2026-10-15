"""Generate synthetic GDSC/CCLE-style cell line datasets for demos.

The biology domain works on two tables: a long-format **response
screen** (one row per cell line x compound pair with ``LN_IC50`` /
``AUC`` metrics) and a wide **expression matrix** (one row per cell
line, one column per gene).  Real panels ship as multi-gigabyte
downloads, so this generator produces a structurally honest stand-in:

* 24 canonical cell lines across 10 tissues,
* 8 compounds with per-compound potency baselines and tissue effects,
* a **planted EGFR biomarker**: EGFR expression (high in lung lines)
  drives sensitivity to the two EGFR inhibitors (erlotinib,
  gefitinib) — exactly the association the biomarker discovery
  workflow should recover,
* AUC correlated with ``LN_IC50`` (lower IC50 -> lower AUC ->
  more sensitive), and
* optional planted hygiene issues (one duplicate pair, one missing
  ``LN_IC50``) so ``pharmaml biology response-audit`` has something
  to find in demos.

Both tables share the same cell-line universe, so they join cleanly
for the correlation and modeling steps.

Usage::

    python examples/make_synthetic_cellline_data.py
    python examples/make_synthetic_cellline_data.py --out-dir data --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Canonical panel: (cell line, tissue).  Shared by both tables so the
#: response screen and the expression matrix always join.
CELL_LINES: list[tuple[str, str]] = [
    ("A549", "lung"),
    ("NCI-H1299", "lung"),
    ("NCI-H1975", "lung"),
    ("MCF-7", "breast"),
    ("MDA-MB-231", "breast"),
    ("T-47D", "breast"),
    ("HCT-116", "colon"),
    ("HT-29", "colon"),
    ("SW-480", "colon"),
    ("HL-60", "blood"),
    ("K-562", "blood"),
    ("A375", "skin"),
    ("SK-MEL-28", "skin"),
    ("PANC-1", "pancreas"),
    ("MIA-PaCa-2", "pancreas"),
    ("Hep-G2", "liver"),
    ("Huh-7", "liver"),
    ("OVCAR-3", "ovary"),
    ("SK-OV-3", "ovary"),
    ("PC-3", "prostate"),
    ("DU-145", "prostate"),
    ("U-87-MG", "brain"),
    ("U-251", "brain"),
    ("SH-SY5Y", "brain"),
]

#: Compounds with a per-compound LN_IC50 baseline (micromolar log scale).
COMPOUNDS: dict[str, float] = {
    "erlotinib": 1.9,
    "gefitinib": 2.1,
    "cisplatin": 3.4,
    "doxorubicin": 1.6,
    "paclitaxel": 2.0,
    "5-fluorouracil": 3.9,
    "gemcitabine": 2.8,
    "sorafenib": 3.1,
}

#: EGFR inhibitors: higher EGFR expression -> lower LN_IC50 (more
#: sensitive).  This is the planted biomarker the demo recovers.
_EGFR_DRUGS: dict[str, float] = {"erlotinib": -0.75, "gefitinib": -0.55}

#: Compounds with a tissue-specific potency shift (docetaxel-ish
#: taxane behaviour in prostate lines, platinum hypersensitivity in
#: blood lines, ...).
_TISSUE_EFFECTS: dict[str, dict[str, float]] = {
    "cisplatin": {"blood": -0.6, "skin": +0.4},
    "paclitaxel": {"prostate": -0.5, "breast": -0.3},
    "gemcitabine": {"pancreas": -0.7},
    "sorafenib": {"liver": -0.5},
    "5-fluorouracil": {"colon": -0.4},
}


#: EGFR levels are a property of the *cell line*, not of either table,
#: so they are drawn from their own fixed seed: the response screen and
#: the expression matrix then agree on them regardless of the caller's
#: ``--seed`` and the planted biomarker stays recoverable.
_EGFR_SEED = 2024


def _egfr_levels() -> dict[str, float]:
    """EGFR log2-expression per cell line (lung lines run hot)."""
    rng = np.random.default_rng(_EGFR_SEED)
    return {
        name: float(rng.normal(9.0, 0.8) if tissue == "lung" else rng.normal(5.5, 0.9))
        for name, tissue in CELL_LINES
    }


def make_response_screen(
    seed: int = 42,
    plant_issues: bool = True,
    n_cell_lines: int | None = None,
) -> pd.DataFrame:
    """Synthetic long-format response screen (GDSC/CCLE layout).

    Columns: ``cell_line``, ``compound``, ``tissue``, ``LN_IC50``,
    ``AUC``.  ``LN_IC50`` is the natural log of the IC50 in micromolar
    (lower = more sensitive); ``AUC`` follows it (lower = more
    sensitive) with measurement noise, clipped to ``(0, 1)``.

    plant_issues:
        Insert one duplicate (cell, compound) row and one missing
        ``LN_IC50`` value so the audit workflow has something to
        report.  ``False`` gives a clean screen.
    n_cell_lines:
        Restrict the panel to the first ``n_cell_lines`` canonical
        lines (>= 6 so every workflow still has data); ``None`` keeps
        the full 24-line panel.
    """
    if n_cell_lines is None:
        panel = list(CELL_LINES)
    else:
        if n_cell_lines < 6:
            raise ValueError(f"n_cell_lines must be >= 6, got {n_cell_lines}")
        if n_cell_lines > len(CELL_LINES):
            raise ValueError(
                f"n_cell_lines must be <= {len(CELL_LINES)}, got {n_cell_lines}"
            )
        panel = CELL_LINES[:n_cell_lines]

    rng = np.random.default_rng(seed)
    egfr = _egfr_levels()
    egfr_values = np.array([egfr[name] for name, _ in panel])
    egfr_z = (egfr_values - egfr_values.mean()) / egfr_values.std(ddof=0)

    rows: list[dict[str, object]] = []
    for cell_idx, (cell, tissue) in enumerate(panel):
        for compound, baseline in COMPOUNDS.items():
            ln_ic50 = baseline
            ln_ic50 += _TISSUE_EFFECTS.get(compound, {}).get(tissue, 0.0)
            ln_ic50 += _EGFR_DRUGS.get(compound, 0.0) * float(egfr_z[cell_idx])
            ln_ic50 += float(rng.normal(0.0, 0.25))
            # AUC tracks potency: sensitive (low ln_ic50) -> low AUC.
            auc = 0.95 - 0.55 / (1.0 + np.exp(-(3.2 - ln_ic50)))
            auc = float(np.clip(auc + rng.normal(0.0, 0.03), 0.0, 1.0))
            rows.append(
                {
                    "cell_line": cell,
                    "compound": compound,
                    "tissue": tissue,
                    "LN_IC50": round(float(ln_ic50), 4),
                    "AUC": round(auc, 4),
                }
            )

    frame = pd.DataFrame(rows)

    if plant_issues:
        # One replicate (a duplicate pair the audit flags and
        # collapse_replicates resolves by median) ...
        duplicate = frame.iloc[[0]].copy()
        duplicate["LN_IC50"] = round(float(frame.iloc[0]["LN_IC50"]) + 0.2, 4)
        frame = pd.concat([frame, duplicate], ignore_index=True)
        # ... and one missing potency value.
        frame.loc[len(frame) // 2, "LN_IC50"] = np.nan

    return frame


def make_expression_matrix(
    seed: int = 43,
    n_genes: int = 300,
    n_cell_lines: int | None = None,
) -> pd.DataFrame:
    """Synthetic expression matrix: rows = cell lines, columns = genes.

    Values are raw RNA-seq-like counts on a log2 scale.  ``EGFR`` is
    the planted biomarker (lung lines run hot, mirroring
    :func:`make_response_screen`); ``KRT8`` marks breast lines; the
    remaining ``GENE_001`` .. ``GENE_{n-2}`` columns are noise so
    feature selection has real work to do.
    """
    if n_cell_lines is None:
        panel = list(CELL_LINES)
    else:
        if n_cell_lines < 6:
            raise ValueError(f"n_cell_lines must be >= 6, got {n_cell_lines}")
        if n_cell_lines > len(CELL_LINES):
            raise ValueError(
                f"n_cell_lines must be <= {len(CELL_LINES)}, got {n_cell_lines}"
            )
        panel = CELL_LINES[:n_cell_lines]
    if n_genes < 3:
        raise ValueError(f"n_genes must be >= 3, got {n_genes}")

    rng = np.random.default_rng(seed)
    egfr = _egfr_levels()

    noise_genes = [f"GENE_{i:03d}" for i in range(1, n_genes - 1)]
    genes = noise_genes + ["EGFR", "KRT8"]

    frame = pd.DataFrame(
        rng.normal(6.0, 1.0, size=(len(panel), len(noise_genes))),
        index=[name for name, _ in panel],
        columns=noise_genes,
    )
    frame["EGFR"] = [egfr[name] for name, _ in panel]
    frame["KRT8"] = [
        float(rng.normal(9.0, 0.7) if tissue == "breast" else rng.normal(5.0, 0.8))
        for _, tissue in panel
    ]
    return frame[genes]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic cell line response + expression tables"
    )
    parser.add_argument("--out-dir", default="data", help="output directory")
    parser.add_argument("--screen", default="synthetic_screen.csv")
    parser.add_argument("--expression", default="synthetic_expression.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean", action="store_true",
        help="skip the planted duplicate/missing-value hygiene issues",
    )
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    screen = make_response_screen(seed=args.seed, plant_issues=not args.clean)
    screen_path = out / args.screen
    screen.to_csv(screen_path, index=False)
    print(f"response screen : {screen_path} ({len(screen)} rows, "
          f"{screen['cell_line'].nunique()} cell lines, "
          f"{screen['compound'].nunique()} compounds)")

    expression = make_expression_matrix(seed=args.seed + 1)
    expr_path = out / args.expression
    expression.to_csv(expr_path)
    print(f"expression      : {expr_path} "
          f"({expression.shape[0]} cell lines x {expression.shape[1]} genes)")


if __name__ == "__main__":
    main()
