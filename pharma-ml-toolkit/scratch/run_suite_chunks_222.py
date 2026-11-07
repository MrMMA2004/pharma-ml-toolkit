"""Round-222 chunked suite runner (the 185b double-fork pattern).

Chunks the 4,790 collected tests + 785 doctests into six batches so no
single pytest process has to hold the whole tree (the OOM lesson).
Chunk 4 (tests/ui) and chunk 5 (playground, Tk under Xvfb) need
DISPLAY=:99 -- the runner exports it.  Writes /tmp/suite222.txt.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite222.txt"

BASE = [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"]


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            out.extend(sorted(glob.glob(str(TOOLKIT / pattern))) or [pattern])
        else:
            out.append(pattern)
    return out


CHUNKS: list[tuple[str, list[str]]] = [
    ("1 (a-l)", ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
                 "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
                 "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
                 "tests/test_l*.py"]),
    ("2 (m-p)", ["tests/test_m*.py", "tests/test_notebook_variants.py",
                 "tests/test_p*.py"]),
    ("3 (r-z)", ["tests/test_r*.py", "tests/test_s*.py", "tests/test_t*.py",
                 "tests/test_u*.py", "tests/test_v*.py", "tests/test_w*.py"]),
    ("4 (ui)", ["tests/ui"]),
    ("5 (playground Tk)", ["tests/test_playground.py"]),
    ("6 (src doctests)", ["src/pharmaml", "--doctest-modules"]),
]


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
    os.environ["DISPLAY"] = ":99"
    os.environ["PYTHONPATH"] = str(TOOLKIT / "src")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    failures = 0
    for label, patterns in CHUNKS:
        command = BASE + expand(patterns)
        done = subprocess.run(command, cwd=str(TOOLKIT), capture_output=True, text=True)
        failures += done.returncode != 0
        tail = "\n".join(done.stdout.strip().splitlines()[-2:])
        print(f"CHUNK {label} exit={done.returncode}: {tail}", flush=True)
    print(f"SUITE222 DONE: {failures} failing chunk(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
