"""End-to-end classification pipeline example (upgraded RF.py workflow).

Runs the full production flow — on **your own file** or on synthetic data:

1. load the descriptor table (``--data my_compounds.csv``) or generate it,
2. inspect + clean + stratified split,
3. cross-validated model comparison across the registry,
4. tune + train the winner,
5. evaluate on the held-out set,
6. persist the model with provenance metadata.

Classification targets with **two or more classes** are fully supported:
``--classes 3`` generates a 3-class table, and a user CSV whose target
column holds 2+ distinct values is treated as multiclass automatically.

Usage::

    python examples/run_classification.py
    python examples/run_classification.py --classes 3
    python examples/run_classification.py --data my_compounds.csv --target activity
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pharmaml.config import PipelineConfig
from pharmaml.data import load_dataset
from pharmaml.logger import configure_logging
from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.models import list_models
from pharmaml.ml.persistence import save_model
from pharmaml.ml.preprocessing import make_train_test_split
from pharmaml.ml.training import compare_models, train_model

DATA = Path(__file__).parent / "data" / "synthetic_classification.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end classification demo")
    parser.add_argument(
        "--data", default=None,
        help="your own CSV/TSV/Excel file (default: synthetic data)",
    )
    parser.add_argument(
        "--target", default=None,
        help="target column name (auto-detected when omitted)",
    )
    parser.add_argument(
        "--classes", type=int, default=2, metavar="K",
        help="classes in the synthetic table when --data is not given (>= 2)",
    )
    parser.add_argument("--model", default=None, help="train this model instead of the winner")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    configure_logging()
    config = PipelineConfig(
        output_dir=Path("artifacts"), test_size=args.test_size, cv_folds=args.cv
    )

    # 1. Data: the user's file when given, otherwise the synthetic generator
    #    (make_synthetic_qsar_data.py).  load_dataset() reads any
    #    CSV/TSV/Excel file, auto-detects the target column and infers the
    #    task — including multiclass (2+ distinct target values).
    if args.data is not None:
        dataset = load_dataset(args.data, target=args.target)
        print(f"Loaded user file '{args.data}': {dataset.n_samples} rows, "
              f"{dataset.n_features} features, task={dataset.task}")
    else:
        if not DATA.is_file():
            from make_synthetic_qsar_data import make_classification_table

            frame = make_classification_table(n_classes=args.classes)
            DATA.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(DATA, index=False)
        dataset = load_dataset(DATA, target="Target")
        print(f"Loaded synthetic table: {dataset.n_samples} rows, "
              f"{dataset.n_features} features, task={dataset.task}")

    if dataset.class_counts:
        print(f"Class balance: {dataset.class_counts}")
    print(dataset.summary())

    # 2. Stratified split (keeps every class in both train and test).
    X_train, X_test, y_train, y_test = make_train_test_split(
        dataset.X, dataset.y, test_size=config.test_size, random_state=config.random_state
    )

    # 3. Cross-validated leaderboard (replaces the 16 legacy
    #    classification scripts, one per algorithm).
    leaderboard = compare_models(dataset.X, dataset.y, task="classification", cv=config.cv_folds)
    print("\n=== Model leaderboard (CV ROC-AUC) ===")
    print(leaderboard.to_string(index=False))

    # 4. Tune + train the best available model (or the requested one).
    winner = args.model or str(leaderboard.iloc[0]["model"])
    print(f"\nTraining winner: {winner}")
    result = train_model(winner, "classification", X_train, y_train, cv=config.cv_folds)

    # 5. Held-out evaluation — multiclass-aware (per-class metrics,
    #    one-vs-rest ROC-AUC, k x k confusion matrix).
    report = evaluate_model(result.model, X_test, y_test, "classification")
    print(f"\n=== Held-out performance ===\n{report}")
    print(f"best params: {result.best_params}")

    # 6. Persistence with provenance.
    model_path = config.output_path(f"{winner}_classification.joblib")
    save_model(
        result.model,
        model_path,
        name=winner,
        task="classification",
        features=dataset.feature_names,
        best_params=result.best_params,
        metrics=report.as_dict(),
    )
    print(f"\nSaved model -> {model_path}")
    print("Available models next run:", [m["name"] for m in list_models()])


if __name__ == "__main__":
    main()
