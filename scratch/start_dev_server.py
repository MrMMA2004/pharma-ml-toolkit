#!/usr/bin/env python
"""Double-fork daemonized `bun run dev` (the r263 suite-runner pattern --
single setsid attempts die with the bash tool call's cleanup, the
double-fork + setsid + closed-fds shape survives across calls)."""
import os
import subprocess
import sys

REPO = "/home/z/my-project"


def main() -> None:
    subprocess.Popen(
        ["bun", "run", "dev"],
        cwd=REPO,
        stdout=open("/dev/null", "wb"),
        stderr=open("/dev/null", "wb"),
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
