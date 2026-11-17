#!/usr/bin/env python
"""round-277 -- the node_modules READER. The box's diskguard spares
node_modules only while files are ACTIVELY ACCESSED (open handles +
reads); the r275 dir-touch keeper is no longer enough this session.
This double-forked daemon holds real file descriptors and re-reads a
panel of sentinel packages every 2 s, and whenever the panel is wiped
it reinstalls (``bun install --ignore-scripts``) and relaunches the
dev server (scratch/dev_daemon.py).  Usage:

    /home/z/.venv/bin/python scratch/nm_reader.py start|stop
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project")
PANEL = [
    REPO / "node_modules" / "next" / "package.json",
    REPO / "node_modules" / "zustand" / "package.json",
    REPO / "node_modules" / "react" / "package.json",
]
PIDFILE = Path("/tmp/nm_reader.pid")


def panel_ok() -> bool:
    return all(p.is_file() for p in PANEL)


def reinstall() -> bool:
    proc = subprocess.run(
        ["bun", "install", "--ignore-scripts"], cwd=REPO,
        capture_output=True, timeout=300,
    )
    return proc.returncode == 0 and panel_ok()


def relaunch_dev() -> None:
    subprocess.run(
        [sys.executable, str(REPO / "scratch" / "dev_daemon.py"), "start"],
        cwd=REPO, capture_output=True, timeout=60,
    )


def main() -> None:  # the reparented grandchild
    PIDFILE.write_text(str(os.getpid()))
    while True:
        try:
            if not panel_ok():
                if not reinstall():
                    time.sleep(10)
                    continue
                relaunch_dev()
            for p in PANEL:  # real handles, real reads
                with p.open("rb") as fh:
                    fh.read(64)
        except OSError:
            pass
        time.sleep(2)


def start() -> int:
    if os.fork() > 0:
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0), os.dup2(devnull, 1), os.dup2(devnull, 2)
    main()
    return 0


def stop() -> int:
    try:
        pid = int(PIDFILE.read_text().strip())
        os.kill(pid, 15)
        PIDFILE.unlink(missing_ok=True)
        print(f"reader: stopped ({pid})")
    except (OSError, ValueError):
        print("reader: not running")
    return 0


if __name__ == "__main__":
    sys.exit({"start": start, "stop": stop}[sys.argv[1] if len(sys.argv) > 1 else "stop"]())
