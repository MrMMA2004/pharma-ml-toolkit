#!/usr/bin/env python3
"""Double-fork the Next.js dev server (port 3000) so it survives the
sandbox's per-call subtree reaping (the e2e291_daemons pattern)."""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

REPO = Path("/home/z/my-project")
LOG = REPO / "dev.log"

def _forked(child):
    if os.fork() > 0:
        return
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    child()
    os._exit(0)

def _dev():
    os.chdir(REPO)
    log = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(log, 1); os.dup2(log, 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    subprocess.run(["bun", "run", "dev"])

def up():
    return subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "--max-time", "3", "http://localhost:3000/"],
        capture_output=True, text=True,
    ).stdout.strip() == "200"

def start():
    if up():
        print("devserver: already up (GET / 200), nothing to do")
        return 0
    _forked(_dev)
    for _ in range(60):
        time.sleep(2)
        if up():
            print("devserver: up (GET / 200)")
            return 0
    print("devserver: failed to become ready in 120 s (tail dev.log)")
    return 1

def stop():
    subprocess.run(["pkill", "-f", "next dev"], check=False)
    subprocess.run(["pkill", "-f", "next-server"], check=False)
    print("devserver: stopped")
    return 0

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    sys.exit({"start": start, "stop": stop}[cmd]())
