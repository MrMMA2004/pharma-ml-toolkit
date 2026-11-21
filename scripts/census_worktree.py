#!/usr/bin/env python3
"""Faithful Python replica of the web app's /api/toolkit/tree walk
(src/lib/toolkit-fs.ts buildTree) so a CLEAN WORKTREE census can be
computed without routing the walk through the main checkout's dev
server (which walks the possibly-dirty main tree, not the worktree).

Rules mirrored exactly:
- EXCLUDED_DIRS at any depth + *dir*.egg-info pruned;
- every regular file counts once toward `files`;
- line counts ONLY for viewable files (allowed extensions + the three
  allowed basenames), split("\n").length, 0 for empty content;
- tree-data extensions (.csv/.pdb/.dlg/.glg/.gpf/.dpf) are never
  line-counted (the LOC stat stays source-only);
- totalBytes sums every file's size.

Usage: census_worktree.py <toolkit-root>
"""
from __future__ import annotations

import sys
from pathlib import Path

EXCLUDED_DIRS = {
    ".pytest_cache", "__pycache__", "node_modules", "catboost_info",
    ".venv", "venv", "env", ".git", ".ruff_cache",
    ".mypy_cache", ".tox", ".idea", ".vscode", ".cache",
}
# the r302 rule (toolkit-fs.ts EXCLUDED_REL_DIRS): artifacts is excluded
# only at these relative depths -- "artifacts" as a NAME hid the real
# frontend/src/services/artifacts/ source tree (the r302 find)
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


def census(root: Path) -> dict[str, int]:
    files = dirs = total_lines = total_bytes = 0
    for dirpath, dirnames, filenames in root.walk():
        folder = Path(dirpath)
        rel_dir = folder.relative_to(root).as_posix()
        dirnames[:] = sorted(
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.endswith(".egg-info")
            and (rel_dir + "/" + d if rel_dir != "." else d) not in EXCLUDED_REL_DIRS
        )
        for d in dirnames:
            dirs += 1
        for name in sorted(filenames):
            path = folder / name
            if not path.is_file() or path.is_symlink():
                continue
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            files += 1
            total_bytes += path.stat().st_size
            ext = path.suffix.lower()
            if ext in TREE_DATA_EXTENSIONS:
                continue
            if is_viewable(rel):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                total_lines += (len(text.split("\n")) if text else 0)
    return {"files": files, "dirs": dirs, "totalLines": total_lines,
            "totalBytes": total_bytes}


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    s = census(root)
    print(f"files={s['files']} dirs={s['dirs']} "
          f"totalLines={s['totalLines']} totalBytes={s['totalBytes']}")
