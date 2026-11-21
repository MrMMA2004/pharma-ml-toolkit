#!/usr/bin/env python3
"""R319 housekeeping: census over the TRACKED tree only (git ls-files),
mirroring census_worktree.py rules exactly (the /api/toolkit/tree walk).

The dirty-worktree walk (1,683/562,262) disagrees with the shipped pins
(1,672/561,916) -- hypothesis: the untracked home-zip strays + the
untracked .github/ dir + the inner scratch counter pollute the walk.
This script isolates the tracked set to verify or refute the pins.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/z/my-project")
TOOLKIT = "pharma-ml-toolkit"

EXCLUDED_DIRS = {
    ".pytest_cache", "__pycache__", "node_modules", "catboost_info",
    ".venv", "venv", "env", ".git", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
}
EXCLUDED_REL_DIRS = {"artifacts", "examples/artifacts"}
ALLOWED_EXTENSIONS = {".py", ".toml", ".txt", ".md", ".yml", ".yaml",
                      ".cfg", ".ini", ".ipynb", ".json"}
ALLOWED_BASENAMES = {"Dockerfile", ".dockerignore", ".gitignore"}
TREE_DATA_EXTENSIONS = {".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"}


def is_viewable(rel: str) -> bool:
    p = Path(rel)
    if p.name in ALLOWED_BASENAMES:
        return True
    return p.suffix.lower() in ALLOWED_EXTENSIONS


def excluded(rel: str) -> bool:
    parts = Path(rel).parts
    if any(part in EXCLUDED_DIRS for part in parts[:-1]):
        return True
    return any("/".join(parts[:i]) in EXCLUDED_REL_DIRS
               for i in range(1, len(parts)))


files = 0
lines = 0
delta_files = []
for rel in subprocess.run(
        ["git", "ls-files", TOOLKIT], cwd=REPO, capture_output=True,
        text=True, check=True).stdout.splitlines():
    if not rel or excluded(rel):
        continue
    files += 1
    if is_viewable(rel) and Path(rel).suffix.lower() not in TREE_DATA_EXTENSIONS:
        content = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        lines += len(content.split("\n"))
    else:
        delta_files.append(rel)

print(f"tracked census: files={files} lines={lines}")
print(f"pin:            files=1672 lines=561916")
print(f"verdict: {'PINS-TRUE' if (files, lines) == (1672, 561916) else 'PINS-DRIFT'}")

# what the dirty-worktree walk adds (the stray set)
tracked = set(subprocess.run(
    ["git", "ls-files", TOOLKIT], cwd=REPO, capture_output=True,
    text=True, check=True).stdout.splitlines())
from pathlib import Path as P
strays = []
for p in (REPO / TOOLKIT).rglob("*"):
    if p.is_file() and not excluded(str(p.relative_to(REPO))):
        rel = str(p.relative_to(REPO))
        if rel not in tracked:
            strays.append(rel)
print(f"untracked strays inside the toolkit tree: {len(strays)}")
for s in sorted(strays)[:20]:
    print(f"  + {s}")
