"""Probe G2 (GUI): the correlation-heatmap target join under a dataset swap.

The handler's own comment claims: "the target joins only when the session
frame holds it AND its rows still align with the split -- a swapped
dataset makes the join dishonest, so it is skipped, not faked."

This probe tests the NATURAL swap flow: train on dataset A, load dataset B
(same shape / same target name / RangeIndex), then click Correlation
heatmap.  Does the join get skipped (honest) or does it silently pair
dataset A's features with dataset B's target (dishonest)?

Round-36 audit (100x-153). Needs Xvfb; deterministic; no network.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"

# bring up a private Xvfb
DISPLAY = ":97"
proc = subprocess.Popen(
    ["Xvfb", DISPLAY, "-screen", "0", "1280x800x24"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
os.environ["DISPLAY"] = DISPLAY
deadline = time.time() + 10
while time.time() < deadline:
    try:
        os.listdir(f"/tmp/.X11-unix")
        if f"X{DISPLAY[1:]}" in os.listdir("/tmp/.X11-unix"):
            break
    except OSError:
        pass
    time.sleep(0.2)

spec = importlib.util.spec_from_file_location("playground_probe", PLAYGROUND_PATH)
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 60

# dataset A: y = 10 * a  (strong correlation between feature a and target)
a = rng.uniform(size=N)
frame_a = pd.DataFrame({
    "a": a,
    "b": rng.uniform(size=N),
    "c": rng.uniform(size=N),
    "y": 10.0 * a,
})
# dataset B (the swap): same shape, same column names, but y is noise
frame_b = pd.DataFrame({
    "a": rng.uniform(size=N),
    "b": rng.uniform(size=N),
    "c": rng.uniform(size=N),
    "y": rng.uniform(size=N),
})

app = pg.PlaygroundApp(persist_session=False)

# record dialogs
seen: list[tuple[str, tuple]] = []
for name in ("showerror", "showwarning", "showinfo"):
    setattr(
        pg.messagebox, name,
        lambda *args, _seen=seen, _n=name, **k: _seen.append((_n, args)),
    )

from sklearn.linear_model import LinearRegression
from pharmaml.ml.evaluation import RegressionReport

# emulate the natural flow's run tuple (exactly what the train render sets)
y_a = frame_a["y"]
x_all = frame_a[["a", "b", "c"]]
idx = rng.permutation(N)
test_idx = idx[:12]
train_idx = idx[12:]
x_test = x_all.iloc[test_idx]
y_test = y_a.iloc[test_idx]
model = LinearRegression().fit(x_all.iloc[train_idx], y_a.iloc[train_idx])
report = RegressionReport(
    rmse=0.1, mae=0.1, r2=0.9, max_error=0.2, n_samples=len(test_idx),
)
trained = type("T", (), {})()
trained.name = "ridge"
app._ml_last_run = (
    model, report, "regression",
    x_all.iloc[train_idx], y_a.iloc[train_idx], x_test, y_test, "y",
)

# the natural adoption path for dataset B (the loader's shared body)
app._ml_adopt_frame(frame_b, Path("dataset_b.csv"))

def drain(timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False

app._ml_plot_correlation_heatmap()
ok = drain()
note = app._ml_result_note.cget("text")
print("job drained:", ok)
print("NOTE:", note)
print("dialogs:", [n for n, _ in seen])

joined = "top 10 |r| + y" in note
print()
print("VERDICT: the target join", "SILENTLY PAIRED dataset B's y against dataset A's features"
      if joined else "was skipped (honest)")

# direct evidence: what correlation does the drawn matrix report for a vs y?
# dataset A alone: corr(a, y) ~= 1.0.  With B's y noise: ~= 0.
if app._ml_host._pg_figure is not None:
    fig = app._ml_host._pg_figure
    texts = [t.get_text() for t in fig.axes[0].texts]
    print("cell annotations:", texts[:8])

app.destroy()
proc.terminate()
print("PROBE_G2_DONE")
