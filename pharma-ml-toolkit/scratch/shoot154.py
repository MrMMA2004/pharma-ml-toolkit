"""shoot154 — the SHAP beeswarm + top-20 bar, desktop artifact.

The round's feature made visible: a GradientBoostingRegressor trained
on a 3-feature frame (y driven by 'strong'), the Results card's
fourth row clicked -- the BEESWARM first (every (sample, feature)
point coloured by the feature's own value, the amber->teal ramp), then
the TOP-20 BAR (the compact sibling). The note lines (top-3 leaders,
base value, held-out row count) are the evidence.

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
OUT_SWARM = TOOLKIT_ROOT / "download" / "pg154-ml-shap-beeswarm.png"
OUT_BAR = TOOLKIT_ROOT / "download" / "pg154-ml-shap-top20-bar.png"

DISPLAY = ":96"
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
n = 120
frame = pd.DataFrame(
    {
        "strong": rng.uniform(-2.0, 2.0, size=n),
        "weak": rng.uniform(-1.0, 1.0, size=n),
        "noise": rng.uniform(-1.0, 1.0, size=n),
    }
)
frame["y"] = 4.0 * frame["strong"] + 0.6 * frame["weak"]

app = pg.PlaygroundApp(persist_session=False)
try:
    app.geometry("1500x950+40+30")
except Exception:  # noqa: BLE001 - unmapped
    pass

# the ML tab must be SELECTED for the host to be mapped on screen
# (the black-frame lesson).
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

from sklearn.ensemble import GradientBoostingRegressor

from pharmaml.ml.evaluation import RegressionReport

x_all = frame[["strong", "weak", "noise"]]
y = frame["y"]
idx = rng.permutation(n)
test_idx, train_idx = idx[:24], idx[24:]
model = GradientBoostingRegressor(random_state=42).fit(
    x_all.iloc[train_idx], y.iloc[train_idx]
)
report = RegressionReport(
    rmse=0.4, mae=0.3, r2=0.97, max_error=1.2, n_samples=len(test_idx)
)
app._ml_last_run = (
    model, report, "regression",
    x_all.iloc[train_idx], y.iloc[train_idx],
    x_all.iloc[test_idx], y.iloc[test_idx], "y",
)


def drain(timeout: float = 120.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def grab(out: Path) -> None:
    app._jump_to_card(5, "Results")
    app._ml_host.update_idletasks()
    app.update_idletasks()
    app.update()
    time.sleep(0.4)
    for _ in range(10):
        app.update()
    from PIL import ImageGrab

    x, y_, w, h = app.winfo_rootx(), app.winfo_rooty(), app.winfo_width(), app.winfo_height()
    ImageGrab.grab(bbox=(x, y_, x + w, y_ + h)).save(out)


# -- the beeswarm -----------------------------------------------------
app._ml_plot_shap_beeswarm()
assert drain(), "SHAP beeswarm job hung"
time.sleep(0.6)
for _ in range(20):
    app.update()

figure = app._ml_host._pg_figure
assert figure is not None, "no beeswarm rendered"
labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
assert labels[0] == "noise" and labels[1] == "weak" and labels[2] == "strong", labels
note = app._ml_result_note.cget("text")
assert "SHAP beeswarm" in note and "24 held-out row(s)" in note, note
assert "strong" in note and "base value" in note, note
grab(OUT_SWARM)

# -- the top-20 bar ---------------------------------------------------
app._ml_plot_shap_importance()
assert drain(), "SHAP bar job hung"
time.sleep(0.6)
for _ in range(20):
    app.update()

figure = app._ml_host._pg_figure
assert figure is not None, "no bar rendered"
labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
assert labels[0] == "noise" and labels[1] == "weak" and labels[2] == "strong", labels
note = app._ml_result_note.cget("text")
assert "top-20 bar" in note and "12" not in note.split("·")[1], note
grab(OUT_BAR)

print("NOTE(swarm):", note)
print("labels:", labels)
print("SHOOT154_OK ->", OUT_SWARM.name, "+", OUT_BAR.name)

app.destroy()
proc.terminate()
sys.exit(0)
