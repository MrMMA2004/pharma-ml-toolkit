"""PROBE audit_154_c -- the GUI heatmap join's ndarray fallback (100x-154).

Round-36's F1 fix made the correlation-heatmap target join swap-proof
(the run's OWN y_test).  The Series path (``y_test.loc[frame.index]``)
fails honestly (KeyError -> not joined).  The ndarray FALLBACK,
though, slices to fit:

    values = np.asarray(y_test, dtype=float).reshape(-1)
    frame[target] = values[: len(frame)]

A y_test LONGER than the frame silently TRUNCATES -- pairing the
first len(frame) values with x rows they may not belong to (a partial
truth the house standard rejects).  A y_test SHORTER raises inside
pandas (length mismatch) -> caught -> not joined (honest).  The probe
drives the exact work() body headlessly with all three shapes.

Run:  DISPLAY=:99 python scratch/probe_audit_154_c.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def join_like_the_gui(x_test: pd.DataFrame, y_test) -> tuple[pd.DataFrame, bool]:
    """The handler's exact join block (playground.py _ml_plot_correlation_heatmap)."""
    target = "y"
    frame = x_test.copy()
    joined = False
    try:
        if isinstance(y_test, pd.Series):
            frame[target] = y_test.loc[frame.index].to_numpy()
        else:
            values = np.asarray(y_test, dtype=float).reshape(-1)
            frame[target] = values[: len(frame)]
        joined = True
    except (KeyError, IndexError, ValueError, TypeError):
        joined = False
    return frame, joined


def main() -> None:
    rng = np.random.default_rng(7)
    x_test = pd.DataFrame(
        {"a": rng.normal(size=10), "b": rng.normal(size=10)},
        index=list(range(100, 110)),
    )

    # 1) the aligned Series (the normal path)
    y_series = pd.Series(rng.normal(size=10), index=list(range(100, 110)), name="y")
    frame, joined = join_like_the_gui(x_test, y_series)
    print(f"series aligned       joined={joined}")

    # 2) Series whose index does NOT cover the frame's labels
    y_off = pd.Series(rng.normal(size=10), index=list(range(200, 210)), name="y")
    frame, joined = join_like_the_gui(x_test, y_off)
    print(f"series non-covering  joined={joined} (must be False -- honest KeyError)")

    # 3) ndarray EXACT length
    y_arr = rng.normal(size=10)
    frame, joined = join_like_the_gui(x_test, y_arr)
    print(f"ndarray exact        joined={joined}")

    # 4) ndarray LONGER -- the silent truncation suspicion
    y_long = rng.normal(size=25)
    frame, joined = join_like_the_gui(x_test, y_long)
    truncated = joined and len(frame) == 10
    print(f"ndarray longer (25)  joined={joined} frame_rows={len(frame)} "
          f"silent_truncation={truncated}")

    # 5) ndarray SHORTER -- the honest length-mismatch
    y_short = rng.normal(size=6)
    frame, joined = join_like_the_gui(x_test, y_short)
    print(f"ndarray shorter (6)  joined={joined} (must be False -- honest ValueError)")

    print("PROBE_C_DONE")


if __name__ == "__main__":
    main()
