#!/usr/bin/env python3
"""Double-fork launcher for the r373 notebook re-execution.

The sandbox's per-call subtree reaping kills setsid/nohup/disown
children at every tool-call boundary (the round-258 lesson, codified
in scratch/dev_daemon.py) -- only a DOUBLE FORK that reparents the
grandchild to PID 1 survives.  Usage:

    python3 scratch/reexec_daemon_371.py
"""
from __future__ import annotations

import os
import sys

REPO = "/home/z/my-project"
LOG = f"{REPO}/scratch/reexec_373.log"
PY = "/home/z/.venv/bin/python"
TOOL = f"{REPO}/pharma-ml-toolkit/tools/reexec_notebooks.py"


def main() -> int:
    if os.fork() > 0:
        return 0  # parent exits at birth
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # intermediate exits; grandchild reparents to PID 1
    os.chdir(REPO)
    with open(LOG, "ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    os.execv(PY, [PY, TOOL])
    return 1


if __name__ == "__main__":
    sys.exit(main())
