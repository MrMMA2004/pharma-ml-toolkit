"""Generate docs/INDEX.md from the package's public API (module __all__ exports)."""
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path("/home/z/my-project/pharma-ml-toolkit")
sys.path.insert(0, str(ROOT / "src"))

MODULES = [
    ("pharmaml", "Package root: exceptions + version"),
    ("pharmaml.data", "Bring your own data: load user CSV/TSV/Excel files"),
    ("pharmaml.chem", "Cheminformatics: converters, descriptors, similarity"),
    ("pharmaml.ml", "QSAR machine learning: preprocessing → training → evaluation → persistence"),
    ("pharmaml.pharma", "Pharmaceutical calculations: units, PK, PD, dosage"),
    ("pharmaml.docking", "Molecular docking: preparation, workflow, parsing, scoring"),
    ("pharmaml.utils.io", "Safe tabular I/O helpers"),
    ("pharmaml.utils.validation", "Shared input-validation layer"),
]

lines = [
    "# Module index",
    "",
    "Auto-derived from each module's `__all__` exports — the authoritative quick",
    "reference for the public API. Re-run `python scratch/build_docs_index.py`",
    "after adding new public functions.",
    "",
]


def entry(kind: str, name: str, doc: str) -> str:
    doc = (doc or "").strip().splitlines()[0] if doc else ""
    return f"| `{name}` | {kind} | {doc} |"


for mod_name, blurb in MODULES:
    mod = importlib.import_module(mod_name)
    lines += [f"## `{mod_name}`", "", f"*{blurb}*", "", "| Symbol | Kind | Summary |", "|---|---|---|"]
    if hasattr(mod, "__all__"):
        exported = list(mod.__all__)
    else:
        # No explicit __all__: fall back to the public dir() names so the
        # section is never silently empty (mirrors the drift test in
        # tests/test_docs_index.py).  Note that this includes imported
        # names, so modules should declare __all__ explicitly.
        exported = [name for name in dir(mod) if not name.startswith("_")]
    for name in exported:
        obj = getattr(mod, name)
        if inspect.isclass(obj):
            kind = "class"
            doc = inspect.getdoc(obj)
        elif inspect.isroutine(obj) or inspect.isfunction(inspect.unwrap(obj)):
            # isroutine + unwrap covers plain functions, builtins and
            # wrappers such as functools.lru_cache (previously these fell
            # through to the "constant" branch and were misclassified).
            kind = "function"
            doc = inspect.getdoc(obj)
        elif inspect.ismodule(obj):
            kind = "module"
            doc = inspect.getdoc(obj)
        else:
            kind = "constant"
            doc = None
        lines.append(entry(kind, name, doc))
    lines.append("")

lines += [
    "## Command-line interface",
    "",
    "```text",
    "pharmaml models | data inspect | ml train | ml compare | chem pic50",
    "             | chem descriptors | pharma pk | pharma dose | docking scan",
    "             | docking check",
    "```",
    "",
    "See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.",
    "",
]

out = ROOT / "docs" / "INDEX.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
print(f"written {out} ({len(lines)} lines, {len(MODULES)} modules)")
