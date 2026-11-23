#!/usr/bin/env python3
"""r340 offline census — an exact replica of src/lib/toolkit-fs.ts's walk()
(the same rules the /api/toolkit/tree route serves): the same excluded-dir
sets, the same allowed-extension/basename line-counting lane, the same JS
split("\n") semantics, the same data-node never-counted rule. Used when the
console lane is uneconomical (the sandbox reaper's node_modules wipe)."""
import os
import sys

ROOT = "/home/z/my-project/pharma-ml-toolkit"
EXCLUDED_DIRS = {".pytest_cache", "__pycache__", "node_modules", "catboost_info",
                 ".venv", "venv", "env", ".git", ".ruff_cache", ".mypy_cache",
                 ".tox", ".idea", ".vscode", ".cache"}
EXCLUDED_REL_DIRS = {"artifacts", "examples/artifacts", "frontend/dist", "build/theme-shots"}
ALLOWED_EXT = {".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".cfg", ".ini", ".ipynb", ".json"}
ALLOWED_BASE = {"Dockerfile", ".dockerignore", ".gitignore"}
DATA_EXT = {".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"}


def count_lines(content: str) -> int:
    """JS semantics: ''.split('\\n').length == 1 handled by the empty guard."""
    if len(content) == 0:
        return 0
    return content.count("\n") + 1


def walk(abs_dir: str, rel_dir: str):
    files = dirs = lines = nbytes = 0
    entries = sorted(os.scandir(abs_dir), key=lambda e: (not e.is_dir(follow_symlinks=False), e.name))
    for entry in entries:
        rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
        if entry.is_dir(follow_symlinks=False):
            if (entry.name in EXCLUDED_DIRS or entry.name.endswith(".egg-info")
                    or rel in EXCLUDED_REL_DIRS):
                continue
            dirs += 1
            f, d, l, b = walk(entry.path, rel)
            files += f; dirs += d; lines += l; nbytes += b
        elif entry.is_file(follow_symlinks=False):
            files += 1
            st = entry.stat(follow_symlinks=False)
            nbytes += st.st_size
            base = entry.name
            ext = os.path.splitext(entry.name)[1].lower()
            if ext in DATA_EXT:
                continue  # data nodes carry a language marker, never counted
            if base in ALLOWED_BASE or ext in ALLOWED_EXT:
                try:
                    with open(entry.path, "r", encoding="utf8", newline="") as fh:
                        content = fh.read()
                    lines += count_lines(content)
                except (UnicodeDecodeError, OSError):
                    pass  # unreadable-as-utf8: lines undefined, never counted
    return files, dirs, lines, nbytes


def read_version() -> str:
    try:
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf8") as fh:
            in_project = False
            for raw in fh:
                line = raw.strip()
                if line.startswith("["):
                    in_project = line == "[project]"
                    continue
                if not in_project:
                    continue
                if line.startswith("version") and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf8") as fh:
            return fh.read().strip()
    except OSError:
        return "0.0.0"


if __name__ == "__main__":
    f, d, l, b = walk(ROOT, "")
    print(f"OFFLINE-WALK: {f} files / {d} dirs / {l} lines / {b} bytes / v{read_version()}")
