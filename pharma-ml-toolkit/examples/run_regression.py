"""End-to-end regression pipeline example (upgrades Regression/RF.py et al.).

Trains a tuned random forest on a synthetic pIC50 dataset, reports
RMSE/MAE/R2 on the held-out set and persists the model bundle.

Usage::

    python examples/run_regression.py
"""

from __future__ import annotations

from pathlib import Path

from pharmaml.config import PipelineConfig
from pharmaml.logger import configure_logging
from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.persistence import save_model
from pharmaml.ml.preprocessing import (
    clean_numeric_frame,
    make_train_test_split,
    split_features_target,
)
from pharmaml.ml.training import train_model
from pharmaml.utils.io import read_table

DATA = Path(__file__).parent / "data" / "synthetic_regression.csv"


def main() -> None:
    configure_logging()
    config = PipelineConfig(output_dir=Path("artifacts"))

    if DATA.is_file():
        frame = read_table(DATA)
    else:
        from make_synthetic_qsar_data import make_regression_table

        frame = make_regression_table()
        frame.to_csv(DATA, index=False)

    frame = clean_numeric_frame(frame, target_col="pIC50")
    X, y = split_features_target(frame, target_col="pIC50")
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    result = train_model(
        "random_forest", "regression", X_train, y_train, cv=config.cv_folds
    )
    report = evaluate_model(result.model, X_test, y_test, "regression")

    print(f"model:  {result.name}")
    print(f"CV:     rmse={-result.cv_mean:.3f} +/- {result.cv_std:.3f}" if result.cv_mean else "CV: n/a")
    print(f"test:   {report}")
    print(f"params: {result.best_params}")

    model_path = config.output_path("random_forest_regression.joblib")
    save_model(
        result.model,
        model_path,
        name="random_forest",
        task="regression",
        features=list(X.columns),
        best_params=result.best_params,
        metrics=report.as_dict(),
    )
    print(f"\nSaved model -> {model_path}")


if __name__ == "__main__":
    main()
