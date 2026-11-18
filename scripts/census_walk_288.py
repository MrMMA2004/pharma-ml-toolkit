#!/usr/bin/env python3
"""Faithful Python replica of src/lib/toolkit-fs.ts walk() census.
Reports files / dirs / totalLines / totalBytes exactly as the live
/api/toolkit/tree walk computes them (the 71dcc06a authority).
Prints numbers only; no file contents."""
import pathlib, sys

ROOT = pathlib.Path("/home/z/my-project/pharma-ml-toolkit")
EXCLUDED_DIRS = {".pytest_cache", "__pycache__", "node_modules", "catboost_info",
                 "artifacts", ".venv", "venv", "env", ".git", ".ruff_cache",
                 ".mypy_cache", ".tox", ".idea", ".vscode", ".cache"}
ALLOWED_EXTENSIONS = {".py", ".toml", ".txt", ".md", ".yml", ".yaml", ".cfg",
                      ".ini", ".ipynb", ".json"}
ALLOWED_BASENAMES = {"Dockerfile", ".dockerignore", ".gitignore"}
TREE_DATA_EXTENSIONS = {".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"}

files = dirs = total_lines = 0
total_bytes = 0

def ext_of(name: str) -> str:
    dot = name.rfind(".")
    if dot <= 0:
        return ""
    return name[dot:].lower()

for path in ROOT.rglob("*"):
    rel = path.relative_to(ROOT)
    # skip anything under an excluded dir (rglob yields them; prune cheaply)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
        continue
    if path.is_dir():
        if path.name in EXCLUDED_DIRS or path.name.endswith(".egg-info"):
            continue
        dirs += 1
        continue
    if not path.is_file():
        continue
    files += 1
    total_bytes += path.stat().st_size
    ext = ext_of(path.name)
    if ext in TREE_DATA_EXTENSIONS:
        continue
    if path.name in ALLOWED_BASENAMES or ext in ALLOWED_EXTENSIONS:
        try:
            content = path.read_text(encoding="utf8")
        except (UnicodeDecodeError, OSError):
            continue
        total_lines += 0 if content == "" else len(content.split("\n"))

print(f"files={files} dirs={dirs} lines={total_lines} bytes={total_bytes:,}")
