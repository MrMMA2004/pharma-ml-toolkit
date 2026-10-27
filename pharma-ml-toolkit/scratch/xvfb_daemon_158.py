#!/usr/bin/env python3
"""Double-fork Xvfb daemon (the standing orphan recipe, 100x-158 edition).

The sandbox reaps processes left in a tool session's process group, so
`Xvfb :99 &` (even with setsid/nohup) dies with the session; a
double-fork orphan survives exactly the way the dev-server daemon and
the full-suite runner do.  Idempotent: refuses to start when :99
already answers.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

DISPLAY = ":99"
SOCKET = f"/tmp/.X11-unix/X{DISPLAY[1:]}"


def alive() -> bool:
    probe = socket.socket(socket.AF_UNIX)
    try:
        probe.connect(SOCKET)
        return True
    except OSError:
        return False
    finally:
        probe.close()


def main() -> int:
    if alive():
        print(f"Xvfb {DISPLAY} already answering")
        return 0
    for stale in (f"/tmp/.X{DISPLAY[1:]}-lock", SOCKET):
        try:
            os.unlink(stale)
        except OSError:
            pass
    if os.fork() > 0:
        time.sleep(1.5)
        print("Xvfb answering" if alive() else "Xvfb FAILED to answer")
        return 0 if alive() else 1
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open("/tmp/xvfb-158.log", "ab", buffering=0)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.execvp("Xvfb", ["Xvfb", DISPLAY, "-screen", "0", "1920x1080x24"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
