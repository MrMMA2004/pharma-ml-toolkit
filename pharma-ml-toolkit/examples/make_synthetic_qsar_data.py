"""Generate synthetic QSAR datasets for demos and smoke tests.

The original scripts required real datasets (``1000selected_features_
classification.csv``, ``BBB.xlsx``) that cannot ship with a toolkit.
This generator produces realistic stand-ins: informative + noise
features, class imbalance, and a pIC50-like continuous endpoint.

Usage::

    python examples/make_synthetic_qsar_data.py --out data --n 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_classification_table(
    n_samples: int = 400,
    n_informative: int = 8,
    n_noise: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic descriptor table with a binary ``Target`` column."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative + n_noise,
        n_informative=n_informative,
        n_redundant=0,
        weights=[0.55, 0.45],
        random_state=seed,
    )
    feature_names = [f"descr_{i:03d}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=feature_names)
    frame["Target"] = y
    return frame


def make_regression_table(
    n_samples: int = 400,
    n_informative: int = 8,
    n_noise: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic descriptor table with a continuous ``pIC50`` target."""
    rng = np.random.default_rng(seed)
    informative = rng.normal(size=(n_samples, n_informative))
    noise = rng.normal(scale=2.0, size=(n_samples, n_noise))
    true_weight = rng.normal(scale=1.5, size=n_informative)
    pic50 = 6.0 + informative @ true_weight + rng.normal(scale=0.5, size=n_samples)

    data = np.hstack([informative, noise])
    columns = [f"descr_{i:03d}" for i in range(data.shape[1])]
    frame = pd.DataFrame(data, columns=columns)
    frame["pIC50"] = pic50
    return frame


def make_pk_profile(ka: float = 1.2, ke: float = 0.35, f: float = 0.9,
                    dose: float = 250.0, vd: float = 35.0,
                    duration_h: float = 24.0, step_h: float = 0.5,
                    error_pct: float = 0.05, seed: int = 7) -> pd.DataFrame:
    """Synthetic oral concentration-time profile with assay noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(0.0, duration_h + step_h, step_h)
    coefficient = f * dose * ka / (vd * (ka - ke))
    conc = coefficient * (np.exp(-ke * times) - np.exp(-ka * times))
    conc *= 1.0 + rng.normal(scale=error_pct, size=times.size)
    conc = np.clip(conc, 0.0, None)
    return pd.DataFrame({"time_h": times, "conc_ng_ml": conc})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--n", type=int, default=400, help="samples per table")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cls_table = make_classification_table(args.n, seed=args.seed)
    cls_table.to_csv(out_dir / "synthetic_classification.csv", index=False)

    reg_table = make_regression_table(args.n, seed=args.seed)
    reg_table.to_csv(out_dir / "synthetic_regression.csv", index=False)

    pk_table = make_pk_profile()
    pk_table.to_csv(out_dir / "synthetic_pk_profile.csv", index=False)

    print(f"Wrote {len(cls_table)} classification rows, {len(reg_table)} "
          f"regression rows and a {len(pk_table)}-point PK profile to '{out_dir}'.")


if __name__ == "__main__":
    main()
