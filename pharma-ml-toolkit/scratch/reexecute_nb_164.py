"""Detached driver for the round-164 notebook refresh (the standing
double-fork recipe; the underlying logic is reexecute_all_notebooks_64)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
COMMAND = [
    str(Path.home() / ".venv" / "bin" / "python"),
    str(TOOLKIT / "scratch" / "reexecute_all_notebooks_64.py"),
]
OUT = "/tmp/nbexec164.txt"


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
