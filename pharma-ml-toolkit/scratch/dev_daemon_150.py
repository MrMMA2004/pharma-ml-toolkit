#!/usr/bin/env python3
"""Double-fork dev-server daemon (the standing recipe, 100x-150 edition).

The sandbox reaps processes left in a tool session's process group, so
`setsid bun run dev &` dies with the session; a double-fork orphan
survives exactly the way the full-suite runner does.  The log goes to
the repo-root dev.log the .zscripts convention already uses.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COMMAND = ["bun", "run", "dev"]
OUT = str(REPO / "dev.log")


def main() -> int:
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open(OUT, "ab", buffering=0)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(str(REPO))
    os.execvp(COMMAND[0], COMMAND)
    return 1


if __name__ == "__main__":
    sys.exit(main())
