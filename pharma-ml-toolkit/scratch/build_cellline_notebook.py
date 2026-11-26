"""Build ``examples/complete_cellline_workflow.ipynb`` — the "from 0 to
100" cell line data analysis workflow (user request, round 100x-17).

Sections mirror the three existing Zero-to-One-Hundred notebooks:

* §1  data entry        — a GDSC-style drug-response screen, loaded and
                          validated through ``load_response_table``
* §2  QC audit          — ``audit_response_table`` + replicate collapse
* §3  normalization     — ``ln_ic50_to_pic50`` + distribution figure
* §4  response matrix   — cell x compound pivot + per-compound potency
* §5  sensitivity calls — sensitive / intermediate / resistant quantiles
* §6  omics features    — expression matrix, log2+z, MAD gene selection
* §7  biomarkers        — ``sensitivity_associations`` with BH-FDR
* §8  response model    — predict a drug's pIC50 from expression
* §9  deep learning     — the new ``dnn`` / ``dnn_residual`` torch models
                          in the same ``compare_models`` leaderboard
* §10 recap

Synthetic data is planted with real structure: EGFR expression drives
erlotinib/gefitinib sensitivity, so §7 actually discovers the planted
biomarker with FDR-significant rho, and §8/§9 models have genuine signal
to fit.  One duplicate pair and one missing value are planted on purpose
so the §2 audit has something honest to report.

Run from the toolkit root:

    python scratch/build_cellline_notebook.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_cellline_workflow.ipynb"

CELLS: list[tuple[str, str]] = [
    ("markdown", """\
# PharmaML — Cell Line Data Analysis (Zero to One Hundred)

One notebook, every stage of a cell-line pharmacology project, in the order
you would actually run it — from a raw drug-response screen to biomarkers
and a deep-learning response model:

| § | Stage | Toolkit |
|---|---|---|
| 1 | **Data entry** — GDSC-style response screen in, validated | `pharmaml.biology` |
| 2 | **QC audit** — missing values, duplicates, replicate collapse | `pharmaml.biology` |
| 3 | Normalization — LN_IC50 → pIC50, distributions | `pharmaml.biology` + `viz` |
| 4 | Response matrix — cell × compound potency landscape | `pharmaml.biology` |
| 5 | Sensitive vs resistant — quantile calls per drug | `pharmaml.biology` |
| 6 | Omics features — expression matrix, log2+z, variable genes | `pharmaml.biology` |
| 7 | **Biomarker discovery** — expression ↔ response with FDR | `pharmaml.biology` |
| 8 | Response model — predict pIC50 from expression (classical ML) | `pharmaml.ml` |
| 9 | **Deep learning** — `dnn` / `dnn_residual` in the leaderboard | `pharmaml.ml` |
| 10 | Recap | — |

The demo data is synthetic but structurally honest: EGFR expression is
planted to drive erlotinib/gefitinib sensitivity, so the biomarker call
in §7 genuinely recovers it. Point the loaders at your own GDSC/CCLE
exports (same columns) and every cell works unchanged."""),
    ("code", """\
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pharmaml.biology import (
    load_response_table, audit_response_table, collapse_replicates,
    ln_ic50_to_pic50, response_matrix, summarize_compounds,
    call_sensitive_resistant, normalize_expression, impute_expression,
    most_variable_genes, sensitivity_associations, top_biomarkers,
    direction_of_association,
)

WORK = Path(tempfile.mkdtemp(prefix="cellline_"))
rng = np.random.default_rng(42)
pd.set_option("display.width", 120)

print("python:", sys.version.split()[0], "| working dir:", WORK)"""),
    ("markdown", """\
## 1 · Data entry — a GDSC-style response screen

The screen arrives (or is exported) in **long format**: one row per
(cell line, compound) pair with a potency metric — here `LN_IC50`,
the natural log of the IC50 in µM, exactly as GDSC ships it.
`load_response_table` validates the columns, auto-detects metric
columns and refuses mistyped inputs with a message that names the
offending column.

The synthetic panel: **24 cell lines** across 10 tissues, **8
compounds**, with one deliberate duplicate pair and one missing value
(§2 needs something honest to audit)."""),
    ("code", """\
# --- DEMO DATA (synthetic, seeded) ---------------------------------------
CELL_LINES = {
    # name: tissue
    "MCF-7": "breast", "T-47D": "breast", "BT-549": "breast",
    "A549": "lung", "H1299": "lung", "H1975": "lung",
    "HepG2": "liver", "Huh7": "liver",
    "K-562": "haematopoietic", "Jurkat": "haematopoietic",
    "HCT-116": "colon", "HT-29": "colon", "DLD-1": "colon",
    "U-87 MG": "CNS", "U-251": "CNS",
    "PC-3": "prostate", "DU-145": "prostate",
    "MIA PaCa-2": "pancreas", "PANC-1": "pancreas",
    "SK-OV-3": "ovary", "OVCAR-3": "ovary",
    "HeLa": "cervix", "Saos-2": "bone",
}
DRUGS = {
    # name: (base LN_IC50, tissue sensitivity bonus, egfr coefficient)
    "erlotinib":  (3.10, {"lung": -0.45, "haematopoietic": +0.30}, -0.90),
    "gefitinib":  (3.30, {"lung": -0.30}, -0.55),
    "everolimus": (2.60, {"haematopoietic": -0.35, "breast": -0.15}, 0.0),
    "dasatinib":  (2.80, {"haematopoietic": -0.50}, 0.0),
    "sorafenib":  (3.90, {"liver": -0.25}, 0.0),
    "crizotinib": (3.60, {"lung": -0.20}, -0.25),
    "paclitaxel": (2.30, {"breast": -0.20, "ovary": -0.20}, 0.0),
    "cisplatin":  (2.90, {"CNS": +0.20}, 0.0),
}

# EGFR expression per cell line (log2 scale): lung lines run hot.
egfr_level = {
    name: rng.normal(9.0, 0.8) if tissue == "lung" else rng.normal(5.5, 0.9)
    for name, tissue in CELL_LINES.items()
}
egfr_z = (pd.Series(egfr_level) - pd.Series(egfr_level).mean()) / pd.Series(egfr_level).std()

rows = []
for cell, tissue in CELL_LINES.items():
    for drug, (base, tissue_map, egfr_coef) in DRUGS.items():
        ln_ic50 = base + tissue_map.get(tissue, 0.0) + egfr_coef * egfr_z[cell]
        ln_ic50 += rng.normal(0.0, 0.18)
        rows.append((cell, tissue, drug, ln_ic50))

screen_frame = pd.DataFrame(rows, columns=["cell_line", "tissue", "compound", "LN_IC50"])

# One duplicate pair (assay repeat) and one missing value, planted for §2.
screen_frame = pd.concat(
    [screen_frame, screen_frame.iloc[[5]]], ignore_index=True
)  # duplicate of row 5
screen_frame.loc[100, "LN_IC50"] = np.nan

screen_path = WORK / "screen.csv"
screen_frame.to_csv(screen_path, index=False)
print(f"{len(screen_frame)} rows written -> {screen_path}")"""),
    ("code", """\
# --- THE ACTUAL IMPORT (works unchanged for your GDSC/CCLE export) --------
screen = load_response_table(screen_path, tissue_column="tissue")
print(screen.summary())"""),
    ("markdown", """\
## 2 · QC audit — find what the screen is hiding

`audit_response_table` returns one row per issue class: missing metric
values, duplicated (cell line, compound) pairs, cell lines mapping to
more than one tissue, and compounds tested on too few lines. Then
`collapse_replicates` resolves duplicates the GDSC way — median per
pair — leaving a clean screen behind."""),
    ("code", """\
audit = audit_response_table(screen)
audit"""),
    ("code", """\
clean = collapse_replicates(screen)
print(f"{len(screen.frame)} rows -> {len(clean.frame)} rows after replicate collapse")
audit_response_table(clean)"""),
    ("markdown", """\
## 3 · Normalization — LN_IC50 → pIC50

GDSC's `LN_IC50` is ln(IC50 in µM); pharmacology prefers pIC50 =
−log10(IC50 in M) — bigger is *more* potent. `ln_ic50_to_pic50` does the
exact conversion (pIC50 = 6 − LN_IC50 / ln 10) and the histogram grid
shows every compound's potency distribution across the panel."""),
    ("code", """\
pic50_frame = clean.frame.copy()
pic50_frame["pIC50"] = ln_ic50_to_pic50(pic50_frame["LN_IC50"])
screen_p = load_response_table(pic50_frame, tissue_column="tissue", metrics=["pIC50"])

# Hand-check: 1 µM IC50 (LN_IC50 = 0) must be exactly pIC50 6.
print("ln_ic50_to_pic50(0.0) =", round(float(ln_ic50_to_pic50(0.0)), 6))

pic50_frame[["cell_line", "compound", "LN_IC50", "pIC50"]].head(8)"""),
    ("code", """\
from pharmaml.viz import plot_descriptor_distributions

fig_dist = plot_descriptor_distributions(
    response_matrix(screen_p, "pIC50"),
    title="pIC50 distributions across the panel (per compound)",
)
fig_dist""",
     ),
    ("markdown", """\
## 4 · Response matrix — the cell × compound potency landscape

`response_matrix` pivots the long table into the classic screening
matrix (cell lines × compounds, NaN where untested);
`summarize_compounds` ranks the panel per compound — median potency,
spread across the panel, and the most sensitive line."""),
    ("code", """\
matrix = response_matrix(screen_p, "pIC50")
print(f"matrix: {matrix.shape[0]} cell lines x {matrix.shape[1]} compounds")
matrix.round(2)"""),
    ("code", """\
summary = summarize_compounds(screen_p, "pIC50")
summary.round(3).sort_values("median", ascending=False)"""),
    ("markdown", """\
## 5 · Sensitive vs resistant — quantile calls

For one drug, `call_sensitive_resistant` labels each cell line by
response quantiles (default 25th/75th). pIC50 is
`lower_is_sensitive=False` — *high* pIC50 means *low* IC50 means
sensitive."""),
    ("code", """\
erlo = matrix["erlotinib"].dropna()
labels = call_sensitive_resistant(erlo, lower_is_sensitive=False)
print(labels.value_counts().to_string())
labels[labels != "intermediate"]"""),
    ("markdown", """\
## 6 · Omics features — expression in, genes selected

The expression matrix: **rows = cell lines, columns = genes** — the same
orientation as every `X` in `pharmaml.ml`, so it feeds models directly.
`normalize_expression` does log2(x+1) + per-gene z-score;
`most_variable_genes` picks the MAD-highest genes (the ones that carry
panel structure — 300 random genes are planted, plus EGFR with its
lung-line signal, and another gene carrying a breast signature so
selection isn't a one-gene show)."""),
    ("code", """\
# --- DEMO EXPRESSION MATRIX (synthetic, seeded) ---------------------------
genes = [f"GENE_{i:03d}" for i in range(1, 299)] + ["EGFR", "KRT8"]
base = rng.normal(6.0, 1.0, size=(len(CELL_LINES), len(genes) - 2))
expression = pd.DataFrame(base, index=list(CELL_LINES), columns=genes[:-2])
expression["EGFR"] = [egfr_level[c] for c in CELL_LINES]        # planted biomarker
expression["KRT8"] = [                                        # planted breast marker
    rng.normal(9.0, 0.7) if t == "breast" else rng.normal(5.0, 0.8)
    for t in CELL_LINES.values()
]
expression.index.name = "cell_line"

expr_norm = normalize_expression(expression)
top_genes = most_variable_genes(expr_norm, n=40)
print(f"expression: {expr_norm.shape[0]} cell lines x {expr_norm.shape[1]} genes (normalized)")
print("top variable genes (first 6):", top_genes[:6])
print("EGFR rank:", top_genes.index("EGFR") + 1 if "EGFR" in top_genes else "not in top 40")"""),
    ("markdown", """\
## 7 · Biomarker discovery — which genes explain sensitivity?

`sensitivity_associations` correlates **every gene** against one drug's
response vector across the panel (Spearman by default), then adjusts the
p-values for 300 simultaneous tests with Benjamini–Hochberg FDR. This is
the classic "omics → biomarker" step — and it genuinely recovers the
planted EGFR → erlotinib association here."""),
    ("code", """\
associations = sensitivity_associations(
    expr_norm, erlo, method="spearman", fdr_alpha=0.05
)
hits = top_biomarkers(associations, n=8)
hits"""),
    ("code", """\
best = associations.loc[associations["gene"] == "EGFR"].iloc[0]
print(f"EGFR vs erlotinib: rho={best['rho']:+.3f}  p={best['p_value']:.2e}  "
      f"FDR={best['fdr']:.2e}  significant={best['significant']}")
print(direction_of_association(best["rho"], lower_is_sensitive=False))

# Same story, weaker coefficient, second EGFR inhibitor:
gefi = matrix["gefitinib"].dropna().loc[erlo.index]
assoc_gefi = sensitivity_associations(expr_norm, gefi.reindex(erlo.index))
best_gefi = assoc_gefi.loc[assoc_gefi["gene"] == "EGFR"].iloc[0]
print(f"\\nEGFR vs gefitinib: rho={best_gefi['rho']:+.3f}  "
      f"FDR={best_gefi['fdr']:.2e}  (same target, weaker planted effect)")"""),
    ("markdown", """\
## 8 · Response model — predict pIC50 from expression

Classical ML on the top-40 variable genes: stratified split, elastic net
vs random forest, honest held-out metrics. The signal is EGFR, so the
models have something real to learn."""),
    ("code", """\
from pharmaml.ml import make_train_test_split, train_model, evaluate_model

X = expr_norm.loc[erlo.index, top_genes]
y = erlo

Xtr, Xte, ytr, yte = make_train_test_split(X, y, test_size=0.25, random_state=42)

en = train_model("elastic_net", "regression", Xtr, ytr, tune=True)
en_report = evaluate_model(en.model, Xte, yte, "regression", keep_predictions=True)
print(f"elastic_net  rmse={en_report.rmse:.3f}  r2={en_report.r2:.3f}")

rf = train_model("random_forest", "regression", Xtr, ytr, tune=False)
rf_report = evaluate_model(rf.model, Xte, yte, "regression", keep_predictions=True)
print(f"random_forest rmse={rf_report.rmse:.3f}  r2={rf_report.r2:.3f}")"""),
    ("code", """\
from pharmaml.viz import plot_predicted_vs_actual

fig_parity = plot_predicted_vs_actual(
    yte, rf_report.predictions,
    metrics={"rmse": rf_report.rmse, "r2": rf_report.r2},
    title="Erlotinib pIC50 — predicted vs actual (random forest)",
)
fig_parity""",
     ),
    ("markdown", """\
## 9 · Deep learning — `dnn` and `dnn_residual` join the leaderboard

The registry's two PyTorch families (deep feed-forward nets with
early stopping, and residual nets with learned skip projections) plug
into the *same* `compare_models` call as every classical model —
cross-validated, tuned and scored identically. From the
`pharma-ml-toolkit/` package root, install the local extra with
`python -m pip install -e '.[deep]'`; without torch they show as
unavailable instead of breaking anything."""),
    ("code", """\
from pharmaml.ml import compare_models, list_models

deep_rows = [m for m in list_models(task="regression") if m["name"] in ("dnn", "dnn_residual")]
for m in deep_rows:
    print(f"{m['name']:<14} available={m['available']}  {m['notes']}")"""),
    ("code", """\
leaderboard = compare_models(
    X, y, task="regression",
    models=["elastic_net", "random_forest", "ann", "dnn", "dnn_residual"],
    cv=3,
)
leaderboard""",
     ),
    ("code", """\
from pharmaml.viz import plot_model_comparison

fig_board = plot_model_comparison(leaderboard, title="Erlotinib response — CV leaderboard (classical + deep)")
fig_board""",
     ),
    ("markdown", """\
## 10 · Recap

| Stage | One-liner |
|---|---|
| Load screen | `load_response_table("screen.csv", tissue_column="tissue")` |
| Audit | `audit_response_table(screen)` |
| Collapse replicates | `collapse_replicates(screen)` |
| Normalize | `ln_ic50_to_pic50(frame["LN_IC50"])` |
| Response matrix | `response_matrix(screen, "pIC50")` |
| Rank compounds | `summarize_compounds(screen, "pIC50")` |
| Call sensitivity | `call_sensitive_resistant(series, lower_is_sensitive=False)` |
| Normalize omics | `normalize_expression(expression)` |
| Select genes | `most_variable_genes(expr, n=40)` |
| Find biomarkers | `sensitivity_associations(expr, response)` |
| Classical model | `train_model("elastic_net", "regression", Xtr, ytr)` |
| Deep learning | `compare_models(..., models=[..., "dnn", "dnn_residual"])` |

Next steps with real data: point §1 at your GDSC/CCLE export, §6 at a
CCLE expression matrix, and re-run — every stage is the same call."""),
]


def main() -> int:
    nb = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell(src) if kind == "markdown"
               else nbformat.v4.new_code_cell(src)
               for kind, src in CELLS],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )

    os.environ.update(
        {
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT / "examples")}},
    ).execute()
    nbformat.write(nb, NOTEBOOK)

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [
        out
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    figures = sum(
        1
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "display_data"
        and "image/png" in out.get("data", {})
    )
    print(
        f"notebook executed: {len(nb.cells)} cells, {len(code_cells)} code, "
        f"{figures} figures, {len(errors)} error outputs"
    )
    if errors:
        for err in errors[:3]:
            print("ERROR:", err.get("ename"), err.get("evalue"))
        return 1

    # refresh the no-output twin (drift-guard parity)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
