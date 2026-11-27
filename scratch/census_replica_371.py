#!/usr/bin/env python3
"""r371 offline census replica — the exact /api/toolkit/tree?fresh=1 walk
semantics replicated in Python (the r366 precedent: when the dev-server
tree route is wedged, the offline replica is the canonical measurement).

Semantics mirrored from src/lib/toolkit-fs.ts:
  * EXCLUDED_DIRS (any depth) + .egg-info suffix + EXCLUDED_REL_DIRS
    (toolkit-root-anchored: artifacts, examples/artifacts,
    frontend/dist, build/theme-shots);
  * every regular (non-symlink) file counts: files += 1, bytes += size;
  * lines counted ONLY for viewable files (extension in the allowed set
    or the allowed basenames), EXCEPT tree-data nodes (.csv/.pdb/.dlg/
    .glg/.gpf/.dpf) which are never line-counted;
  * Node's readFile(utf8) replaces invalid bytes (errors="replace");
  * countLines: content.split("\\n").length — N newlines + 1 if
    non-empty, 0 for empty;
  * dirs += 1 for each non-excluded child directory (root not counted).
"""
from __future__ import annotations

import os
import sys

TOOLKIT = "/home/z/my-project/pharma-ml-toolkit"

EXCLUDED_DIRS = {
    ".pytest_cache", "__pycache__", "node_modules", "catboost_info",
    ".venv", "venv", "env", ".git", ".ruff_cache", ".mypy_cache",
    ".tox", ".idea", ".vscode", ".cache",
}
EXCLUDED_REL_DIRS = {
    "artifacts", "examples/artifacts", "frontend/dist", "build/theme-shots",
}
ALLOWED_EXTENSIONS = {
    ".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".cfg", ".ini",
    ".ipynb", ".json",
}
ALLOWED_BASENAMES = {"Dockerfile", ".dockerignore", ".gitignore"}
TREE_DATA_EXTENSIONS = {".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"}

files = 0
dirs = 0
lines = 0
total_bytes = 0


def count_lines(content: str) -> int:
    if len(content) == 0:
        return 0
    return content.count("\n") + 1


def walk(dir_abs: str, rel_dir: str) -> None:
    global files, dirs, lines, total_bytes
    for name in sorted(os.listdir(dir_abs)):
        rel = f"{rel_dir}/{name}" if rel_dir else name
        abs_path = os.path.join(dir_abs, name)
        if os.path.islink(abs_path):
            continue  # Node's Dirent.isFile()/isDirectory() are false for symlinks
        if os.path.isdir(abs_path):
            if (name in EXCLUDED_DIRS or name.endswith(".egg-info")
                    or rel in EXCLUDED_REL_DIRS):
                continue
            dirs += 1
            walk(abs_path, rel)
        elif os.path.isfile(abs_path):
            st = os.stat(abs_path)
            files += 1
            total_bytes += st.st_size
            ext = os.path.splitext(name)[1].lower()
            if ext in TREE_DATA_EXTENSIONS:
                continue  # tree-data node: language marker only, never counted
            viewable = (
                name in ALLOWED_BASENAMES
                or ext in ALLOWED_EXTENSIONS
            )
            if viewable:
                with open(abs_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                lines += count_lines(content)


def main() -> int:
    walk(TOOLKIT, "")
    print(f"files={files} dirs={dirs} lines={lines} bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
