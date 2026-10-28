"""Detached playground-suite runner (the standing double-fork recipe)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
COMMAND = [
    str(Path.home() / ".venv" / "bin" / "python"),
    "-m",
    "pytest",
    "tests/test_playground.py",
    "-p",
    "no:warnings",
    "--tb=line",
    "-q",
    "--junitxml=scratch/junit164_playground.xml",
]
OUT = "/tmp/pg164.txt"


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
    os.environ["DISPLAY"] = ":98"
    os.execvp(COMMAND[0], COMMAND)
    return 1


if __name__ == "__main__":
    sys.exit(main())
