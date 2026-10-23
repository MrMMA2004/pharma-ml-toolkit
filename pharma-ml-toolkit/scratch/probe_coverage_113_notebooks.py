#!/usr/bin/env python3
"""Notebook axis of the per-file coverage contract (100x-113).

The 100x-111 audit greps playground.py per rar file (the GUI axis);
this companion greps the executed notebooks' CODE cells for the same
per-file symbol mapping (markdown prose does NOT count — an API tour
means import/call surface, not a mention).  Verdict per file:

  NOTEBOOK-COVERED   >=1 code-cell hit for >=1 mapped symbol
  NOTEBOOK-ABSENT    0 hits (the family is GUI+library-only today)
  ABSORBED/EMPTY     the documented non-goals (same list as 100x-111)

The GUI axis remains the binding contract (0 zero-hit); the notebook
axis is reported honestly so a family that loses its notebook tour
shows up here instead of silently rotting.

Run after any notebook or mapping change:
    python scratch/probe_coverage_113_notebooks.py

100x-118 extension — the uploads axis: the 2026-10 upload
Applicability_domain_analysis.py joins the notebook contract too (the
ML workflow's §14 Trust section carries its import/call surface).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "scratch"))

# reuse the canonical per-file mapping (single source of truth)
from probe_coverage_111_perfile import (  # noqa: E402
    ABSORBED,
    FILE_MAP,
    RAR_ROOT,
    UPLOAD_MAP,
)

NB_TEXTS: list[tuple[str, str]] = []
for nb_path in sorted(EXAMPLES.glob("*.ipynb")):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in data["cells"]
        if cell["cell_type"] == "code"
    )
    NB_TEXTS.append((nb_path.name, code))
ALL_CODE = "\n".join(text for _, text in NB_TEXTS)


def count_hits(symbol: str) -> int:
    return len(re.findall(r"\b" + re.escape(symbol) + r"\b", ALL_CODE))


def main() -> int:
    actual = sorted(
        str(p.relative_to(RAR_ROOT)) for p in RAR_ROOT.rglob("*") if p.is_file()
    )
    if not actual:
        print("ERROR: fresh rar extraction missing at", RAR_ROOT)
        print("       re-extract first (see the 100x-111 audit contract)")
        return 1

    rows: list[tuple[str, str, str]] = []
    for rel in actual:
        syms = FILE_MAP.get(rel)
        if syms is None:
            rows.append((rel, "NO-MAPPING", ""))
            continue
        if not syms:
            note = "absorbed (documented non-goal)"
            rows.append((rel, "ABSORBED" if rel in ABSORBED else "EMPTY", note))
            continue
        hits = {s: count_hits(s) for s in syms}
        if any(h > 0 for h in hits.values()):
            detail = " ".join(f"{s}:{h}" for s, h in hits.items() if h > 0)
            rows.append((rel, "NOTEBOOK-COVERED", detail))
        else:
            rows.append((rel, "NOTEBOOK-ABSENT", ",".join(syms)))

    # 100x-118: the uploads axis — an uploaded file's notebook surface
    upload_rows: list[tuple[str, str, str]] = []
    for rel, syms in sorted(UPLOAD_MAP.items()):
        hits = {s: count_hits(s) for s in syms}
        if any(h > 0 for h in hits.values()):
            detail = " ".join(f"{s}:{h}" for s, h in hits.items() if h > 0)
            upload_rows.append((rel, "NOTEBOOK-COVERED", detail))
        else:
            upload_rows.append((rel, "NOTEBOOK-ABSENT", ",".join(syms)))

    print("=== NOTEBOOK AXIS (code cells of examples/*.ipynb) ===")
    for rel, status, detail in rows:
        print(f"{status:17s} {rel}  {detail}")
    print("=== UPLOADS AXIS (100x-118) ===")
    for rel, status, detail in upload_rows:
        print(f"{status:17s} {rel}  {detail}")

    n_cov = sum(1 for r in rows if r[1] == "NOTEBOOK-COVERED")
    n_abs = sum(1 for r in rows if r[1] in ("ABSORBED", "EMPTY"))
    n_absent = sum(1 for r in rows if r[1] == "NOTEBOOK-ABSENT")
    n_nomap = sum(1 for r in rows if r[1] == "NO-MAPPING")
    print(f"\ntotal={len(rows)} notebook-covered={n_cov} "
          f"absorbed/empty={n_abs} notebook-absent={n_absent} "
          f"no-mapping={n_nomap}")
    u_cov = sum(1 for r in upload_rows if r[1] == "NOTEBOOK-COVERED")
    u_absent = sum(1 for r in upload_rows if r[1] == "NOTEBOOK-ABSENT")
    print(f"uploads: notebook-covered={u_cov} notebook-absent={u_absent}")
    if u_absent:
        print("\nUPLOAD NOTEBOOK-ABSENT — the uploads axis is BINDING "
              "(a file the user handed over must tour in a notebook):")
        for rel, _, syms in upload_rows:
            if rel:
                print(f"  - {rel} (mapped: {syms})")
        return 1

    absent = [r for r in rows if r[1] == "NOTEBOOK-ABSENT"]
    if absent:
        print(f"\nNOTEBOOK-ABSENT FILES ({len(absent)}) — honest report, "
              f"the GUI axis is the binding contract:")
        for rel, _, syms in absent:
            print(f"  - {rel} (mapped: {syms})")
    print("\nVERDICT: notebook axis reported; "
          "the binding per-file contract stays playground.py (100x-111).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
