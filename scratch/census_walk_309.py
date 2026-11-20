#!/usr/bin/env python3
"""Faithful Python replica of src/lib/toolkit-fs.ts walk() census with
the round-302 SCOPED exclusion semantics (B-1 fix) — round-309 trued:

  * EMPTY-FILE LINE SEMANTICS trued to the JS carrier: countLines()
    returns 0 when content.length === 0 (in place since 2026-10-13);
    the r303-era replica counted len("".split("\\n")) == 1 per empty
    file — a systematic +4 bias over the four 0-byte __init__.py files
    (tests/api, ui/pages, ui/widgets, ui/workers). Every census since
    carried it; the round-309 find trues it (the r307-era pin 553,002
    answered a live walk of 552,998).

Line counting is the JS semantics: len(content.split('\\n')) over the
viewable extension set ONLY (.py/.toml/.txt/.md/.yml/.yaml/.cfg/.ini/
.ipynb/.json + the three basenames) — ts/tsx/css/html are viewable tree
nodes but never counted. Empty files count 0 lines. Prints numbers
only; no file contents."""
import pathlib

ROOT = pathlib.Path("/home/z/my-project/pharma-ml-toolkit")
EXCLUDED_DIRS = {".pytest_cache", "__pycache__", "node_modules", "catboost_info",
                 ".venv", "venv", "env", ".git", ".ruff_cache",
                 ".mypy_cache", ".tox", ".idea", ".vscode", ".cache"}
EXCLUDED_REL_DIRS = {"artifacts", "examples/artifacts", "frontend/dist",
                     "build/theme-shots"}
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


def excluded_rel(rel) -> bool:
    parts = rel.parts
    if len(parts) >= 1 and "/".join(parts[:1]) in EXCLUDED_REL_DIRS:
        return True
    if len(parts) >= 2 and "/".join(parts[:2]) in EXCLUDED_REL_DIRS:
        return True
    return False


def viewable(rel) -> bool:
    name = rel.parts[-1]
    if name in ALLOWED_BASENAMES:
        return True
    return ext_of(name) in ALLOWED_EXTENSIONS


for path in ROOT.rglob("*"):
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]):
        continue
    if path.is_dir():
        if path.name in EXCLUDED_DIRS or path.name.endswith(".egg-info"):
            continue
        if excluded_rel(rel):
            continue
        dirs += 1
        continue
    if not path.is_file():
        continue
    if excluded_rel(rel):
        continue
    files += 1
    total_bytes += path.stat().st_size
    if viewable(rel) and ext_of(path.name) not in TREE_DATA_EXTENSIONS:
        content = path.read_text(encoding="utf-8", errors="replace")
        if content:  # the JS carrier: empty file -> 0 lines
            total_lines += len(content.split("\n"))

print(f"files={files} dirs={dirs} lines={total_lines} bytes={total_bytes}")
