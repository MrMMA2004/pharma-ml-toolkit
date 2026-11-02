"""Double-fork follow-up runner (round 185): the file chunks the first
runner passed as unexpanded globs, plus the doctest re-run.  Globs are
expanded in Python (subprocess does not expand them)."""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]
PY = str(Path.home() / ".venv" / "bin" / "python")
OUT = "/tmp/suite185b.txt"

BASE = [PY, "-m", "pytest", "-q", "-p", "no:warnings", "--tb=line"]


def expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            out.extend(sorted(glob.glob(str(TOOLKIT / pattern)))
                       or [pattern])
        else:
            out.append(pattern)
    return out


CHUNKS: list[tuple[str, list[str]]] = [
    ("1 (a-l)", ["tests/test_a*.py", "tests/test_b*.py", "tests/test_c*.py",
                 "tests/test_d*.py", "tests/test_e*.py", "tests/test_f*.py",
                 "tests/test_g*.py", "tests/test_h*.py", "tests/test_i*.py",
                 "tests/test_l*.py"]),
    ("2 (m-p + notebooks)", ["tests/test_m*.py", "tests/test_notebook_variants.py",
                             "tests/test_p*.py", "--ignore=tests/test_playground.py"]),
    ("3 (r-z)", ["tests/test_r*.py", "tests/test_s*.py", "tests/test_t*.py",
                 "tests/test_u*.py", "tests/test_v*.py", "tests/test_w*.py"]),
    ("4 (src doctests)", ["src/pharmaml", "--doctest-modules"]),
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

    failures = 0
    for label, patterns in CHUNKS:
        command = BASE + expand(patterns)
        done = subprocess.run(command, cwd=str(TOOLKIT), capture_output=True, text=True)
        failures += done.returncode != 0
        tail = "\n".join(done.stdout.strip().splitlines()[-2:])
        print(f"CHUNK {label} exit={done.returncode}: {tail}", flush=True)
    print(f"SUITE185B DONE: {failures} failing chunk(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
