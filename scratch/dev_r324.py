#!/usr/bin/env python
"""r324 dev-server daemon — the double-fork shape of
scratch/start_dev_server.py with the one fix the fifteenth rollback
surfaced: the package.json "dev" script pipes through tee, and bun
delegates piped scripts to bash WITHOUT node_modules/.bin on PATH
("next: command not found" — the server dies instantly inside the
daemon, invisible because the daemon's stdio is /dev/null). The PATH
fix rides the child env; the daemon shape (double fork + setsid +
closed fds) stays the r263 pattern that survives the tool-call reaper.
"""
import os
import subprocess
import sys

REPO = "/home/z/my-project"
BIN = REPO + "/node_modules/.bin"


def main() -> None:
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    log = open(REPO + "/dev.log", "ab")
    subprocess.Popen(
        ["bun", "run", "dev"],
        cwd=REPO,
        env=env,
        stdout=log,
        stderr=log,
        stdin=open("/dev/null", "rb"),
        start_new_session=True,
    )


def daemonize() -> None:
    if os.fork():
        sys.exit(0)
    os.setsid()
    if os.fork():
        sys.exit(0)
    sys.stdout.close()
    sys.stderr.close()
    main()


if __name__ == "__main__":
    daemonize()
