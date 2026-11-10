#!/usr/bin/env python
"""Regenerate ``docs/INDEX.md`` from the live ``pharmaml`` package.

Run from the repository root::

    python scratch/build_docs_index.py

The generator walks every public module's ``__all__``, classifies each
symbol (class / function / module / constant) via introspection and
takes the first line of its docstring as the summary.  ``tests/
test_docs_index.py`` guards the generated file against drift, so this
script must be re-run whenever a public symbol is added, removed or
renamed.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

#: Every module that gets a section in the index, in output order.
MODULES: tuple[str, ...] = (
    "pharmaml",
    "pharmaml.data",
    "pharmaml.chem",
    "pharmaml.ml",
    "pharmaml.generative",
    "pharmaml.biology",
    "pharmaml.pharma",
    "pharmaml.docking",
    "pharmaml.md",
    "pharmaml.viz",
    "pharmaml.utils.io",
    "pharmaml.utils.validation",
    "pharmaml.utils.checkpoints",
)

HEADER = """\
# Module index

Auto-derived from each module's `__all__` exports — the authoritative quick
reference for the public API. Re-run `python scratch/build_docs_index.py`
after adding new public functions.
"""

CLI_SECTION = """\
## Command-line interface

```text
pharmaml models | data inspect | data split | data convert | ml train
             | ml compare | chem pic50 | chem descriptors | pharma pk
             | pharma dose | docking prep-ligands | docking prep-protein
             | docking scan | docking check | generative run
             | generative corpus
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
"""

#: Maximum characters of a docstring's first line kept in the summary cell.
MAX_SUMMARY = 80


def classify(obj: Any) -> str:
    """Classify a symbol exactly the way tests/test_docs_index.py does."""
    if inspect.isclass(obj):
        return "class"
    if inspect.isroutine(obj) or inspect.isfunction(inspect.unwrap(obj)):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    return "constant"


def first_doc_line(obj: Any) -> str:
    """First non-blank line of ``obj``'s docstring ("" when absent)."""
    doc = inspect.getdoc(obj)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def truncate(text: str, width: int = MAX_SUMMARY) -> str:
    """Cut a summary at a word boundary so table cells stay readable."""
    if len(text) <= width:
        return text
    cut = text[:width]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(",;:")


def render_index() -> str:
    parts: list[str] = [HEADER]
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        exported = list(getattr(module, "__all__", None) or [
            name for name in dir(module) if not name.startswith("_")
        ])
        blurb = first_doc_line(module) or module_name
        parts.append(f"## `{module_name}`\n\n*{blurb}*\n\n")
        parts.append("| Symbol | Kind | Summary |\n|---|---|---|\n")
        for symbol in exported:
            obj = getattr(module, symbol)
            kind = classify(obj)
            summary = "" if kind == "constant" else truncate(first_doc_line(obj))
            parts.append(f"| `{symbol}` | {kind} | {summary} |\n")
        parts.append("\n")
    parts.append(CLI_SECTION)
    return "".join(parts)


def main() -> int:
    out_path = PACKAGE_ROOT / "docs" / "INDEX.md"
    content = render_index()
    out_path.write_text(content, encoding="utf-8")
    sections = content.count("\n## `")
    symbols = sum(1 for line in content.splitlines() if line.startswith("| `"))
    print(f"wrote {out_path.relative_to(PACKAGE_ROOT)} "
          f"({sections} modules, {symbols} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
