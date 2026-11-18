#!/usr/bin/env python
"""scratch/census.py -- replicate toolkit-fs.ts's census walk exactly.

files  = every file under pharma-ml-toolkit/ outside EXCLUDED_DIRS
         (+ *.egg-info); the .zip homes inside the tree count as files
         (the round-201 lesson: the census moved when zip homes landed).
lines  = only viewable source files (.py/.toml/.txt/.md/.yml/.yaml/
         .cfg/.ini/.ipynb/.json + Dockerfile/.dockerignore/.gitignore);
         data nodes (.csv/.pdb/.dlg/.glg/.gpf/.dpf) are never counted.

Usage:  python scratch/census.py [tree-root]   (default: the live toolkit)
"""

from __future__ import annotations

import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "catboost_info",
    "artifacts",
    ".venv",
    "venv",
    "env",
    ".git",
    ".ruff_cache",
    ".mypy_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".cache",
}
ALLOWED_EXTENSIONS = {
    ".py",
    ".toml",
    ".txt",
    ".md",
    ".yml",
    ".yaml",
    ".cfg",
    ".ini",
    ".ipynb",
    ".json",
}
ALLOWED_BASENAMES = {"Dockerfile", ".dockerignore", ".gitignore"}
DATA_EXTENSIONS = {".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"}


def count_lines(content: str) -> int:
    if len(content) == 0:
        return 0
    return content.count("\n") + 1  # split("\n").length, the TS exact


def census(root: Path) -> tuple[int, int]:
    files = 0
    lines = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        # a file inside an excluded dir (or .egg-info) never counts
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel_parts[:-1]):
            continue
        files += 1
        name = path.name
        ext = path.suffix.lower()
        if name in ALLOWED_BASENAMES:
            viewable = True
        else:
            viewable = ext in ALLOWED_EXTENSIONS
        if viewable and ext not in DATA_EXTENSIONS:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            lines += count_lines(text)
    return files, lines


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
    files, lines = census(root)
    print(f"census: {files} files / {lines:,} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
