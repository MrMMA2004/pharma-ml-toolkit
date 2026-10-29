"""Round-166 detached launcher: all 8 notebooks in the background.

reexecute_nb_client_164.py re-executes every shipped notebook with
fresh outputs; this double-fork launcher detaches it (with DISPLAY=:98
and single-thread env pins) so the shell returns immediately while the
~1h run continues.  Log: /tmp/nbexec166.txt.  After it finishes, run
scratch/strip_notebook_outputs.py to refresh the no-output twins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
COMMAND = [
    str(Path.home() / ".venv" / "bin" / "python"),
    str(TOOLKIT / "scratch" / "reexecute_nb_client_164.py"),
]
OUT = "/tmp/nbexec166.txt"


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
