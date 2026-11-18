#!/usr/bin/env python3
"""round 210: the fallback-stats census — the toolkit-fs.ts-exact walk.

Replicates ``src/lib/toolkit-fs.ts``'s ``buildTree`` stats exactly, so
the web panel's ``FALLBACK_STATS`` (used when the live tree walk is
unavailable) stays the honest number:

* ``files`` counts EVERY file in the walk — zips, PNGs, CSVs, PDBs
  included; only ``EXCLUDED_DIRS`` (caches, venvs, ``.git`` …) and
  ``*.egg-info`` directories are skipped;
* ``lines`` counts ONLY viewable text files (``.py .toml .txt .md
  .yml .yaml .cfg .ini .ipynb .json`` plus the ``Dockerfile`` /
  ``.dockerignore`` / ``.gitignore`` basenames), one line per
  ``"\\n"`` split segment (the TS ``countLines`` semantics); data
  nodes (CSV/PDB/…) are never line-counted;
* read failures skip the line count, exactly like the TS ``catch``.

Committed this round (the round-210 lesson: the round-209 census.py
died untracked with the sandbox rollback).  Run from the toolkit
root; prints ``files/lines`` in the FALLBACK_STATS shape.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRS = {
    ".pytest_cache", "__pycache__", "node_modules", "catboost_info",
    "artifacts", ".venv", "venv", "env", ".git", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
}
ALLOWED_EXTENSIONS = {
    ".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".cfg",
    ".ini", ".ipynb", ".json",
}
ALLOWED_BASENAMES = {"Dockerfile", ".dockerignore", ".gitignore"}


def count_lines(content: str) -> int:
    if len(content) == 0:
        return 0
    return content.count("\n") + 1  # the TS split("\n").length semantics


def main() -> int:
    files = 0
    lines = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d for d in sorted(dirnames)
            if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")
        ]
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_file():
                files += 1
                ext = path.suffix.lower()
                if ext in ALLOWED_EXTENSIONS or name in ALLOWED_BASENAMES:
                    try:
                        lines += count_lines(path.read_text(encoding="utf8"))
                    except (OSError, UnicodeDecodeError):
                        pass  # the TS catch: unreadable files skip the count
    print(f"{files}/{lines:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
