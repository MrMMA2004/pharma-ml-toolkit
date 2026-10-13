"""End-to-end classification pipeline example (upgraded RF.py workflow).

Runs the full production flow on synthetic data:

1. generate/load the descriptor table,
2. clean + split,
3. cross-validated model comparison across the registry,
4. tune + train the winner,
5. evaluate on the held-out set,
6. persist the model with provenance metadata.

Usage::

    python examples/run_classification.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pharmaml.config import PipelineConfig
from pharmaml.logger import configure_logging
from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.models import list_models
from pharmaml.ml.persistence import save_model
from pharmaml.ml.preprocessing import (
    clean_numeric_frame,
    make_train_test_split,
    split_features_target,
)
from pharmaml.ml.training import compare_models, train_model
from pharmaml.utils.io import read_table

DATA = Path(__file__).parent / "data" / "synthetic_classification.csv"


def main() -> None:
    configure_logging()
    config = PipelineConfig(output_dir=Path("artifacts"))

    # 1. Data (synthetic generator from make_synthetic_qsar_data.py).
    if DATA.is_file():
        frame = read_table(DATA)
    else:
        from make_synthetic_qsar_data import make_classification_table

        frame = make_classification_table()
        frame.to_csv(DATA, index=False)

    # 2. Clean and split.
    frame = clean_numeric_frame(frame, target_col="Target")
    X, y = split_features_target(frame, target_col="Target")
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    # 3. Cross-validated leaderboard (replaces the 16 legacy
    #    classification scripts, one per algorithm).
    leaderboard = compare_models(X, y, task="classification", cv=config.cv_folds)
    print("\n=== Model leaderboard (CV ROC-AUC) ===")
    print(leaderboard.to_string(index=False))

    # 4. Tune + train the best available model.
    winner = str(leaderboard.iloc[0]["model"])
    print(f"\nTraining winner: {winner}")
    result = train_model(winner, "classification", X_train, y_train, cv=config.cv_folds)

    # 5. Held-out evaluation.
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
        features=list(X.columns),
        best_params=result.best_params,
        metrics=report.as_dict(),
    )
    print(f"\nSaved model -> {model_path}")
    print("Available models next run:", [m["name"] for m in list_models()])


if __name__ == "__main__":
    main()
