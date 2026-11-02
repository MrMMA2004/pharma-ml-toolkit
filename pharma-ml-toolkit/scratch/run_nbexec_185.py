"""Double-fork notebook re-execution runner (round 185, the 160 recipe).

Re-executes every example notebook so their committed version prints
match the shipped tree (1.2.105), then strips the no-output twins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
COMMAND = [
    str(Path.home() / ".venv" / "bin" / "python"),
    "scratch/reexecute_all_notebooks_64.py",
]
OUT = "/tmp/nbexec185b.txt"


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
    os.execvp(COMMAND[0], COMMAND)
    return 1


if __name__ == "__main__":
    sys.exit(main())
