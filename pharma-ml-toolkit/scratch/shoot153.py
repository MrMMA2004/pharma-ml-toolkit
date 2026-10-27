"""shoot153 — the swap-proof correlation heatmap, desktop artifact.

The round-36 audit's RED finding made visible: train on dataset A
(y = 10*a), adopt a same-shaped dataset B (y = noise), click
"Correlation heatmap" — the matrix must show A's OWN a-y = +1.00 (the
run's y_test), never B's noise pairing. The note line and the cell
annotations are the evidence.

Deterministic (seed 42); needs Xvfb; no network.
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
OUT = TOOLKIT_ROOT / "download" / "pg153-ml-swap-proof.png"

DISPLAY = ":97"
proc = subprocess.Popen(
    ["Xvfb", DISPLAY, "-screen", "0", "1600x1000x24"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
os.environ["DISPLAY"] = DISPLAY
deadline = time.time() + 10
while time.time() < deadline:
    try:
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
n = 60
a = rng.uniform(size=n)
frame_a = pd.DataFrame(
    {
        "a": a,
        "b": rng.uniform(size=n),
        "c": rng.uniform(size=n),
        "y": 10.0 * a,
    }
)
frame_b = pd.DataFrame(
    {
        "a": rng.uniform(size=n),
        "b": rng.uniform(size=n),
        "c": rng.uniform(size=n),
        "y": rng.uniform(size=n),
    }
)

app = pg.PlaygroundApp(persist_session=False)
try:
    app.geometry("1500x950+40+30")
except Exception:  # noqa: BLE001 - unmapped
    pass

# the ML tab must be SELECTED for the host to be mapped on screen
# (the black-frame lesson: an unmapped widget grabs as empty).
for widget in app._walk_widgets(app):
    if widget.winfo_class() == "TNotebook":
        for tab_id in widget.tabs():
            if "ML" in str(widget.tab(tab_id, "text")):
                widget.select(tab_id)
                break
for _ in range(30):
    app.update()
    time.sleep(0.02)
app.update_idletasks()

from sklearn.linear_model import LinearRegression

from pharmaml.ml.evaluation import RegressionReport

x_all = frame_a[["a", "b", "c"]]
y_a = frame_a["y"]
idx = rng.permutation(n)
test_idx, train_idx = idx[:12], idx[12:]
model = LinearRegression().fit(x_all.iloc[train_idx], y_a.iloc[train_idx])
report = RegressionReport(
    rmse=0.1, mae=0.1, r2=0.9, max_error=0.2, n_samples=len(test_idx)
)
trained = type("Trained", (), {})()
trained.name = "ridge"
app._ml_last_run = (
    model, report, "regression",
    x_all.iloc[train_idx], y_a.iloc[train_idx],
    x_all.iloc[test_idx], y_a.iloc[test_idx], "y",
)
app._ml_adopt_frame(frame_b, Path("swapped_dataset.csv"))

app._ml_plot_correlation_heatmap()


def drain(timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


assert drain(), "heatmap job hung"
time.sleep(0.6)
for _ in range(20):
    app.update()

figure = app._ml_host._pg_figure
assert figure is not None, "no figure rendered"
texts = [t.get_text() for t in figure.axes[0].texts]
# row-major 4x4 over [a, b, c, y]: the a-y cell (index 3) must be A's
# own +1.00 -- the swap-proof join (the audit's red finding, fixed).
assert texts[3] == "+1.00", f"a-y cell is {texts[3]!r}, expected '+1.00'"

note = app._ml_result_note.cget("text")
assert "top 10 |r| + y" in note, note

try:
    # jump the Results card into view (the shoot152 recipe) then grab
    app._jump_to_card(5, "Results")
    app._ml_host.update_idletasks()
    app.update_idletasks()
    app.update()
    time.sleep(0.4)
    for _ in range(10):
        app.update()
    from PIL import ImageGrab

    # grab the WHOLE app window (the shoot152 recipe — the host's own
    # bbox can sit inside an unpainted region when TkAgg renders the
    # figure as a PNG photo)
    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(OUT)
except Exception as exc:  # noqa: BLE001 - grab diagnostics
    print("grab failed:", exc)
    raise

print("NOTE:", note)
print("cells:", texts[:8])
print("SHOOT153_OK ->", OUT.name)

app.destroy()
proc.terminate()
