#!/usr/bin/env python3
"""r288 follow-up: re-run chunk 1 (a-l) ALONE after its first attempt was
SIGTERM-killed at ~77% by concurrent memory pressure (the zip build + audit
running alongside it on the 4-GB box). Same env/envvars as the r288 runner."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite288_chunk1.txt"

PATTERNS = ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
            "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
            "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
            "tests/test_l*.py"]
BASE = [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts="]


def expand(patterns: list[str]) -> list[str]:
    out = []
    for pattern in patterns:
        if "*" in pattern:
            out.extend(sorted(str(p) for p in TOOLKIT.glob(pattern)))
        else:
            out.append(pattern)
    return out


def main() -> int:
    if os.fork() > 0:
        print("chunk1 rerun forked")
        return 0
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    out = open(OUT, "wb", buffering=0)
    os.dup2(out.fileno(), 1)
    os.dup2(out.fileno(), 2)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.chdir(TOOLKIT)
    os.environ["DISPLAY"] = ":99"
    os.environ["PYTHONPATH"] = str(TOOLKIT / "src")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    done = subprocess.run(BASE + expand(PATTERNS), cwd=str(TOOLKIT),
                          capture_output=True, text=True, check=False)
    tail = "\n".join(done.stdout.strip().splitlines()[-3:])
    print(f"CHUNK 1 RERUN exit={done.returncode}: {tail}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
