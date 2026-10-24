"""100x-125 — learning-curve CONTRACT parity guard, Python side.

The desktop ML card's "Learning curve" button runs
``pharmaml.viz.plot_learning_curve``; the web twin's card runs the TS
engine in ``src/lib/learning-curve.ts`` + ``/api/ml/learning-curve``.
The per-point numbers are engine-specific by design (sklearn vs the TS
twin), so this guard pins the SHARED CONTRACT — the pieces that must
never drift between the two cards:

* the default size schedule: ``linspace(0.1, 1.0, 8)`` (fractions),
* the gap threshold ``0.15``,
* the two verdict strings, byte-mirrored (``f"gap ≈ {gap:.2f} …"`` on
  the Python side, ``toFixed(2)`` on the TS side).

The verdicts here are read off REAL figures the shipped library
produced (the title's second line) for one overfitting setup and one
converging setup, with the gap parsed back out — the TS twin then
formats the SAME gaps and the suite compares the strings byte-for-byte.

Emits /tmp/lc_ref_125.json.  Run from the toolkit root:
python scratch/probe_lc_parity_125.py [extra gaps...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import os

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

OUTPUT = Path("/tmp/lc_ref_125.json")

# The library's default schedule, computed by the same numpy call the
# shipped code uses (step arithmetic + last element pinned to stop).
FRACTIONS = np.linspace(0.1, 1.0, 8)


def _verdict_of(model: object, X: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Call the SHIPPED plot_learning_curve and read the title back."""
    from pharmaml.viz import plot_learning_curve

    figure = plot_learning_curve(model, X, y, cv=5, title="probe")
    title = figure.axes[0].get_title()
    figure.clear()
    # the title is f"{title}\n{verdict}" — the verdict is the 2nd line
    verdict = title.split("\n", 1)[1]
    match = re.search(r"gap ≈ ([0-9.]+)", verdict)
    if match is None:  # the converging branch carries no number
        return -1.0, verdict
    return float(match.group(1)), verdict


def main() -> None:
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor

    # Case 1 — a deep tree on noisy data: train R^2 ~ 1, CV far below,
    # the gap comfortably over the 0.15 cut (deterministic seeds).
    X_over, y_over = make_regression(
        n_samples=160, n_features=6, noise=18.0, random_state=0
    )
    gap_over, verdict_over = _verdict_of(
        DecisionTreeRegressor(max_depth=None, random_state=0), X_over, y_over
    )

    # Case 2 — plain OLS on a nearly noiseless line: both curves hug
    # 1.0, the gap far under the cut.
    rng = np.random.default_rng(1)
    X_lin = rng.uniform(0.0, 10.0, size=(120, 1))
    y_lin = (3.0 * X_lin[:, 0] + 2.0).ravel() + rng.normal(0.0, 0.05, size=120)
    gap_conv, verdict_conv = _verdict_of(LinearRegression(), X_lin, y_lin)

    # The threshold literal, read from the shipped source (the boundary
    # both sides agree on).
    source = (
        Path(__file__).resolve().parents[1] / "src" / "pharmaml" / "viz" / "ml_plots.py"
    ).read_text(encoding="utf-8")
    match = re.search(r"if gap > ([0-9.]+)", source)
    if match is None:
        raise SystemExit("the gap threshold literal was not found in ml_plots.py")
    threshold = float(match.group(1))

    cases = []
    if gap_over >= 0:
        cases.append({"gap": gap_over, "verdict": verdict_over})
    cases.append({"gap": -1.0 if gap_conv < 0 else gap_conv, "verdict": verdict_conv})
    # extra gaps from the caller (the suite passes none; the boundary
    # grid is the TS probe's own default)
    for extra in sys.argv[1:]:
        cases.append({"gap": float(extra), "verdict": None})

    payload = {
        "fractions": [float(v) for v in FRACTIONS],
        "threshold": threshold,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1))
    print(f"lc_ref_125: {len(cases)} case(s), threshold {threshold}")
    print(f"  overfit : gap={gap_over:.4f} {verdict_over}")
    print(f"  converge: gap={gap_conv:.4f} {verdict_conv}")


if __name__ == "__main__":
    main()
