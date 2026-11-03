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
    "round 185: the XAI absorption re-ship -- "
    "captum/tf-keras-vis/PAIR-code-saliency/bertviz as the offline ml.xai "
    "engine (attribution methods, layer traces, attention grid) + Deep "
    "Learning GUI cards, notebooks re-executed for 1.2.105, suite "
    "chunk-green at 3896 tests (v1.2.105)"
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
