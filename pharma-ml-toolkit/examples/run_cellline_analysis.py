"""End-to-end cell line analysis example (the biology domain demo).

Runs the full production flow — on **your own files** or on synthetic
data (mirroring ``run_classification.py``):

1. load the response screen + expression matrix (``--screen`` /
   ``--expression``) or generate them deterministically in memory,
2. QC audit (duplicate pairs, missing values, hygiene issues),
3. replicate collapse by median,
4. LN_IC50 -> pIC50 + per-compound potency ranking,
5. sensitive / intermediate / resistant calls,
6. expression normalization + MAD feature selection,
7. biomarker discovery — the synthetic data has a **planted EGFR
   biomarker** for the EGFR inhibitors, so a correct run must surface
   EGFR at the top of the table,
8. response modeling with the ML registry leaderboard (elastic net vs
   random forest; ``--deep`` adds the two PyTorch families when torch
   is installed).

Unlike ``run_classification.py`` the synthetic tables are **never
cached to disk** — they are regenerated deterministically from the
seed on every run, so there is no stale-cache failure mode.

Usage::

    python examples/run_cellline_analysis.py
    python examples/run_cellline_analysis.py --compound erlotinib --top 5
    python examples/run_cellline_analysis.py --deep
    python examples/run_cellline_analysis.py --screen my_screen.csv \\
        --expression my_expr.csv --compound cisplatin
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pharmaml.biology import (
    audit_response_table,
    call_sensitive_resistant,
    collapse_replicates,
    load_response_table,
    ln_ic50_to_pic50,
    response_matrix,
    sensitivity_associations,
    summarize_compounds,
    top_biomarkers,
)
from pharmaml.biology.omics import most_variable_genes, normalize_expression
from pharmaml.logger import configure_logging
from pharmaml.ml import evaluate_model, make_train_test_split, train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end cell line / biomarker analysis demo"
    )
    parser.add_argument(
        "--screen", default=None,
        help="your own long-format response CSV/TSV/Excel (default: synthetic data)",
    )
    parser.add_argument(
        "--expression", default=None,
        help="your own expression matrix CSV (rows = cell lines, default: synthetic)",
    )
    parser.add_argument(
        "--compound", default="erlotinib",
        help="compound to model + mine biomarkers for (default: erlotinib)",
    )
    parser.add_argument("--metric", default="LN_IC50", help="response metric column")
    parser.add_argument("--top", type=int, default=5, help="biomarkers to list")
    parser.add_argument(
        "--n-genes", type=int, default=40,
        help="features for the response model (top MAD genes + planted EGFR)",
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="add the dnn / dnn_residual deep-learning models to the leaderboard "
        "(requires torch: pip install 'pharmaml[deep]')",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()

    # 1. Data: the user's files when given, otherwise the deterministic
    #    synthetic generator (make_synthetic_cellline_data.py).
    if args.screen is not None or args.expression is not None:
        if args.screen is None or args.expression is None:
            parser.error("--screen and --expression must be given together")
        screen_table = args.screen
        expression_table = args.expression
        print(f"Using user files: {args.screen} + {args.expression}")
    else:
        from make_synthetic_cellline_data import (
            make_expression_matrix,
            make_response_screen,
        )

        screen_table = make_response_screen(seed=args.seed, plant_issues=True)
        expression_table = make_expression_matrix(seed=args.seed + 1)
        print(
            "Using synthetic data: 24 cell lines x 8 compounds "
            "(planted EGFR biomarker, one duplicate + one missing value)"
        )

    screen = _load_screen(screen_table)
    print("\n=== Screen summary ===")
    print(screen.summary())

    # 2. QC audit -- the hygiene report before any statistics.
    audit = audit_response_table(screen)
    print("\n=== QC audit ===")
    print(audit.to_string(index=False))

    # 4. pIC50 + per-compound potency ranking.  The collapsed frame gains
    #    a pIC50 column, so it is reloaded as a screen that knows the new
    #    metric (screens are immutable -- metric lists never mutate).
    collapsed = collapse_replicates(screen).frame.copy()
    collapsed["pIC50"] = ln_ic50_to_pic50(collapsed[args.metric])
    clean = load_response_table(
        collapsed,
        cell_column=screen.cell_col,
        compound_column=screen.compound_col,
        tissue_column=screen.tissue_col,
        metrics=screen.metrics + ["pIC50"],
    )
    compounds = summarize_compounds(clean, metric="pIC50")
    print(f"\nReplicate collapse: {len(screen.frame)} -> {len(collapsed)} row(s)")
    print("\n=== Per-compound potency (pIC50: higher = more potent) ===")
    print(compounds.to_string())  # compound is the index -- keep it visible

    # 5. Sensitive / resistant calls for the chosen compound.
    matrix = response_matrix(clean, metric=args.metric)
    if args.compound not in matrix.columns:
        available = ", ".join(sorted(matrix.columns))
        parser.error(
            f"compound {args.compound!r} not in the screen; available: {available}"
        )
    compound_response = matrix[args.compound]
    labels = call_sensitive_resistant(compound_response)
    print(f"\n=== Sensitivity calls: {args.compound} ({args.metric}) ===")
    print(labels.to_string())

    # 6. Expression normalization + MAD feature selection.
    from pharmaml.utils.io import read_table as _read_table

    if isinstance(expression_table, (str, Path)):
        raw = _read_table(expression_table)
        expression = raw.set_index(raw.columns[0])  # first column = cell line
    else:
        expression = expression_table
    expr_norm = normalize_expression(expression)
    top_genes = most_variable_genes(expr_norm, n=args.n_genes)
    if "EGFR" in expr_norm.columns and "EGFR" not in top_genes:
        top_genes = ["EGFR"] + top_genes[:-1]  # keep the planted biomarker in play
    print(f"\nTop {len(top_genes)} variable genes: {', '.join(top_genes[:8])} ...")

    # 7. Biomarker discovery for the chosen compound.
    response = clean.frame[clean.frame["compound"] == args.compound].set_index(
        clean.cell_col
    )[args.metric]
    associations = sensitivity_associations(
        expr_norm.loc[response.index.intersection(expr_norm.index)], response
    )
    markers = top_biomarkers(associations, n=args.top)
    print(f"\n=== Top biomarkers for {args.compound} ===")
    print(markers.to_string(index=False))

    # 8. Response modeling: predict the compound's pIC50 from expression.
    #    Cell lines with a missing response (the audit already flagged
    #    them) cannot train a model -- dropped here, visibly.  Cell lines
    #    absent from the expression matrix cannot either.
    response = response.dropna()
    usable = response.index.intersection(expr_norm.index)
    if len(usable) < len(response):
        print(
            f"Note: {len(response) - len(usable)} responded cell line(s) have "
            "no expression profile -- excluded from the model"
        )
    response = response.loc[usable]
    X = expr_norm.loc[response.index, top_genes]
    y = ln_ic50_to_pic50(response)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=0.25, random_state=args.seed
    )
    print(f"\n=== Response model: pIC50({args.compound}) from expression ===")
    print(f"{len(X_train)} train / {len(X_test)} test cell lines, "
          f"{X.shape[1]} gene features")

    results = {}
    # elastic_net is tuned (its regularization is what tames wide, 40-feature
    # screens); the tree ensemble is strong enough untuned on 24-row panels.
    for name, tune in (("elastic_net", True), ("random_forest", False)):
        trained = train_model(name, "regression", X_train, y_train, tune=tune)
        report = evaluate_model(trained.model, X_test, y_test, "regression")
        results[name] = (trained, report)
        print(f"{name:<15} rmse={report.rmse:.3f}  r2={report.r2:+.3f}")

    if args.deep:
        from pharmaml.ml.models import list_models

        deep_available = {
            m["name"]: m["available"]
            for m in list_models(task="regression")
            if m["name"] in ("dnn", "dnn_residual")
        }
        for name, available in deep_available.items():
            if not available:
                print(f"{name:<15} unavailable (install torch, or "
                      "'pip install pharmaml[deep]')")
                continue
            # Single-combination "grid" pins demo-sized hyperparameters
            # (train_model has no model_kwargs; one combo = one fit path).
            # 200 epochs sounds large but early stopping (patience 15)
            # picks the real stopping point.
            trained = train_model(
                name, "regression", X_train, y_train, tune=True, cv=3,
                param_grid={"epochs": [200], "hidden_layers": [(32, 16)]},
            )
            report = evaluate_model(trained.model, X_test, y_test, "regression")
            results[name] = (trained, report)
            print(f"{name:<15} rmse={report.rmse:.3f}  r2={report.r2:+.3f}")

    best = max(results, key=lambda k: results[k][1].r2)
    print(f"\nBest model: {best} (r2={results[best][1].r2:+.3f})")
    print(
        "\nRecap: audit -> collapse -> pIC50 -> sensitivity calls -> "
        "biomarkers -> response model. See "
        "examples/complete_cellline_workflow.ipynb for the notebook version."
    )


def _load_screen(screen_table):
    """Load a response screen from a frame or a path, sniffing the tissue column."""
    if hasattr(screen_table, "columns"):  # DataFrame
        tissue = "tissue" if "tissue" in screen_table.columns else None
        return load_response_table(screen_table, tissue_column=tissue)
    tissue = "tissue"  # CSVs from the generator always carry the annotation
    return load_response_table(screen_table, tissue_column=tissue)



if __name__ == "__main__":
    main()
