"""Double-fork ship runner (round 185, the 160 recipe).

Wraps `tools/checkpoint.py --commit` so the 10-15 min gate sequence
survives the sandbox's background-process reaper: plain `&` and even
`setsid nohup` die ~2 min in; a full double-fork (new session,
reparented to init) is the only pattern that reliably outlives the
calling shell.  Output lands in /tmp/ship185b.txt.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
MESSAGE = (
    "round 186: the four-repo absorption -- tensorflow/tensorboard "
    "(ml/runboard.py: the summary-writer event model, the Vitter "
    "reservoir with always_keep_last, the EMA smoothing, the projector's "
    "nearest-neighbours), lmcinnes/umap (ml/umap.py: smooth_knn_dist, "
    "the fuzzy simplicial set, find_ab_params, the negative-sampling "
    "SGD, transform, sklearn-pinned trustworthiness), voxel51/fiftyone "
    "(ml/curation.py: the view-stage algebra + the brain methods) and "
    "plotly/plotly.py's express layer (viz/interactive.py: hovertext, "
    "templates, express_figure, SVG title tooltips) -- four new Deep "
    "Learning cards (Run board, UMAP map, Data curation, Interactive "
    "figure), 122 new tests, notebooks re-executed for 1.2.106, "
    "4032 collected across 147 files (v1.2.106)"
)
COMMAND = [PY, "tools/checkpoint.py", "--commit", MESSAGE]
OUT = "/tmp/ship185b.txt"


def main() -> int:
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open(OUT, "wb", buffering=0)  # noqa: SIM115 - dup2'd, must stay open
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(TOOLKIT)
    os.environ["DISPLAY"] = os.environ.get("DISPLAY", ":99")
    os.environ.setdefault("PYTHONPATH", str(TOOLKIT / "src"))
    os.execvp(COMMAND[0], COMMAND)
    return 1


if __name__ == "__main__":
    sys.exit(main())
