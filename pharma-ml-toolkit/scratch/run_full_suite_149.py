"""Double-fork full-suite runner (the standing recipe, junit149 edition).

Home-directory literals are forbidden in the shipped tree (the leakage
guard): paths are derived from this file's location instead.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
COMMAND = [
    str(Path.home() / ".venv" / "bin" / "python"),
    "-m",
    "pytest",
    "-p",
    "no:warnings",
    "--tb=line",
    "-q",
    "--junitxml=scratch/junit149.xml",
]
OUT = "/tmp/full149.txt"


def main() -> int:
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open(OUT, "wb", buffering=0)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(TOOLKIT)
    os.execvp(COMMAND[0], COMMAND)
    return 1


if __name__ == "__main__":
    sys.exit(main())
