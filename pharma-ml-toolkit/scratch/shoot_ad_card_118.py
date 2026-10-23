"""Shoot the applicability-domain card with real analysis results (100x-118)."""
from __future__ import annotations

import subprocess
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, ".")
import playground

app = playground.PlaygroundApp()


def pump(timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


app._ml_load_path(Path("examples/data/library_with_descriptors.csv"))
app._model_combo.set("logistic_regression")
app._ml_train()
assert pump(), "train hung"
app._ad_boot_spin.set("8")
app._ml_ad_run()
assert pump(), "AD hung"
assert app._ml_ad_result is not None

out = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/ad-card-r118.png"
app._book.select(5)  # Data & ML tab
app.update()
page = app._book.nametowidget(app._book.tabs()[5])
page._canvas.yview_moveto(1.0)  # scroll to the bottom (the AD card)
for _ in range(80):
    app.update()
    time.sleep(0.02)
subprocess.run(["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1600x1000",
                "-i", os.environ.get("DISPLAY", ":99"), "-frames:v", "1", out],
               capture_output=True)
print("saved", out, os.path.getsize(out) if os.path.exists(out) else 0)
app.destroy()
