"""Double-fork dev-server launcher (the standing recipe).

The sandbox reaps every process still attached to the tool command's
session; a plain `nohup ... &` dies between commands.  Two forks +
setsid fully orphan the server so it survives -- the ONLY launch that
does (see the worklog's standing traps).
"""

from __future__ import annotations

import os
import sys

PROJECT = "/home/z/my-project"
LOG = os.path.join(PROJECT, "dev.log")


def main() -> int:
    if os.fork() > 0:
        return 0  # parent exits immediately
    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # intermediate exits -- the grandchild is orphaned
    log = open(LOG, "ab", buffering=0)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(PROJECT)
    os.execvp("bun", ["bun", "run", "dev"])
    return 1  # never reached


if __name__ == "__main__":
    sys.exit(main())
