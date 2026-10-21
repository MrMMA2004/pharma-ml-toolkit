"""100x-115/116 — the provenance round-trip closure, one-shot auditor.

The standing 100x-104 open item ("every CSV the NOTEBOOKS load == the
file the builder writes") closed with three legs, each re-runnable here:

  1. notebook-source scan  — every DATA_DIR reference in the EXECUTED
     notebooks resolves to a top-level examples/data entry (no stale
     paths), and only code cells count;
  2. no orphans            — every top-level examples/data entry is
     loaded by at least one executed notebook (15/15 today);
  3. builder --check       — the non-mutating drift audit: byte-identity
     of every owned file, the missing-file branch, the orphan scan, the
     PDB size pins, and the verify() content pins — plus a planted-drift
     self-test proving the gate fires (exactly one drift line, exit 1,
     and the planted file is never repaired).

100x-116 adds two extensions, re-run here with the same three-leg rig:

  4. documentation surfaces — every examples/data/... path the CLI help
     text (src/pharmaml/cli.py) and the README promise must resolve AND
     be builder-owned (prose promises bind like code blocks);
  5. drift attribution      — every drift line names the owning builder
     function (owner: build_*) and STATS["owners"] maps every owned path
     to its function, so a drift report is an actionable pointer.

The pytest guards live in tests/test_data_provenance.py (the binding
contract); this probe is the human-readable evidence run + the
worklog/COVERAGE documentation source.

Run:  python scratch/probe_provenance_roundtrip_115.py
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../pharma-ml-toolkit
DATA = ROOT / "examples" / "data"
EXAMPLES = ROOT / "examples"
BUILDER = ROOT / "scratch" / "build_example_data.py"

EXECUTED_NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "complete_tuning_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
)

FAILURES: list[str] = []


def _ok(condition: bool, message: str) -> None:
    tag = "OK " if condition else "FAIL"
    print(f"  [{tag}] {message}")
    if not condition:
        FAILURES.append(message)


def leg_notebook_scan() -> dict[str, set[str]]:
    print("\n--- leg 1: notebook-source scan (code cells only) ---")
    pattern = re.compile(r'DATA_DIR\s*/\s*["\']([^"\']+)["\']')
    references: dict[str, set[str]] = {}
    for name in EXECUTED_NOTEBOOKS:
        notebook = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
        found: set[str] = set()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                found.update(pattern.findall("".join(cell["source"])))
        references[name] = found
        print(f"  {name:<44} {len(found)} data reference(s)")
    stale = [
        (nb, ref)
        for nb, refs in references.items()
        for ref in sorted(refs)
        if not (DATA / ref).exists()
    ]
    deep = [
        (nb, ref) for nb, refs in references.items() for ref in sorted(refs) if "/" in ref
    ]
    _ok(not stale, f"every reference resolves on disk (stale: {stale})")
    _ok(not deep, f"no references below the top level (deep: {deep})")
    return references


def leg_no_orphans(references: dict[str, set[str]]) -> None:
    print("\n--- leg 2: no orphans (disk -> notebooks) ---")
    loaded: set[str] = set()
    for refs in references.values():
        loaded.update(refs)
    shipped = {p.name for p in DATA.iterdir()}
    orphans = sorted(shipped - loaded)
    unshipped = sorted(loaded - shipped)
    print(f"  shipped top-level entries : {len(shipped)}")
    print(f"  loaded by notebooks       : {len(loaded & shipped)}")
    _ok(not orphans, f"no shipped entry goes unloaded (orphans: {orphans})")
    _ok(not unshipped, f"no loaded name is unshipped (unshipped: {unshipped})")


def leg_builder_check() -> None:
    print("\n--- leg 3: builder --check (non-mutating round-trip) ---")
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_example_data", BUILDER)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    # clean run: byte-identity + orphans + verify pins + the ownership map
    builder.main(["--check"])
    _ok(builder.STATS["drift"] == [], "clean run: zero drift")
    _ok(builder.STATS["written"] == 0, "clean run: --check wrote nothing")
    _ok(
        builder.STATS["unchanged"] >= 42,
        f"clean run: {builder.STATS['unchanged']} owned files byte-identical",
    )
    owned = {Path(p) for p in builder.STATS["paths"]}
    disk = {p for p in DATA.rglob("*") if p.is_file()}
    _ok(
        {p for p in owned if p.parent != builder.TESTS_DATA_DIR} == disk,
        f"clean run: owned set == disk set ({len(disk)} files, no orphans)",
    )
    owners = builder.STATS["owners"]
    _ok(
        len(owners) == len(builder.STATS["paths"])
        and all(str(v).startswith(("build_", "ensure_pdbs")) for v in owners.values()),
        f"attribution: all {len(owners)} owned paths map to a builder function",
    )
    distinct = sorted(set(owners.values()))
    print(f"    owners: {len(distinct)} functions -> {', '.join(distinct)}")

    # planted drift: one flipped byte -> exactly one drift line, exit 1,
    # and the audit never repairs the planted file
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "data"
        shutil.copytree(DATA, sandbox)
        builder.main_reset = None  # (STATS is reset inside main)
        builder.DATA_DIR = sandbox  # type: ignore[assignment]
        victim = sandbox / "qsar_regression.csv"
        original = victim.read_bytes()
        victim.write_bytes(original + b"drift\n")
        try:
            builder.main(["--check"])
            gate_fired = False
        except SystemExit as exc:
            gate_fired = exc.code == 1
        _ok(gate_fired, "planted drift: gate fired with exit 1")
        _ok(
            len(builder.STATS["drift"]) == 1
            and "qsar_regression.csv" in builder.STATS["drift"][0]
            and "differs from" in builder.STATS["drift"][0]
            and "owner: build_qsar_multiclass_and_regression" in builder.STATS["drift"][0],
            f"planted drift: exactly one ATTRIBUTED drift line ({builder.STATS['drift']})",
        )
        _ok(
            victim.read_bytes() == original + b"drift\n",
            "planted drift: the audit never repaired the file",
        )
        # empty-dir branch: every owned file reads as missing
        empty = Path(tmp) / "empty"
        empty.mkdir()
        builder.DATA_DIR = empty  # type: ignore[assignment]
        try:
            builder.main(["--check"])
            missing_fired = False
        except SystemExit as exc:
            missing_fired = exc.code == 1
        missing = [d for d in builder.STATS["drift"] if "is missing from" in d]
        _ok(
            missing_fired and len(missing) >= 40,
            f"empty-dir branch: {len(missing)} missing-file drift lines, exit 1",
        )
        _ok(
            all("(owner:" in d for d in missing),
            "empty-dir branch: every missing line carries its owner",
        )


def leg_docs_surfaces() -> None:
    """100x-116 leg 4: the CLI + README sides of the contract."""
    print("\n--- leg 4: documentation surfaces (cli.py + README) ---")
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_example_data", BUILDER)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.main(["--check"])

    owned: set[str] = set()
    for p in builder.STATS["paths"]:
        try:
            owned.add(p.relative_to(DATA).as_posix())
        except ValueError:
            continue  # tests/data — outside the docs inventory
    owned_dirs: set[str] = set()
    for name in owned:
        parts = name.split("/")
        if len(parts) > 1:
            owned_dirs.add("/".join(parts[:-1]))

    pattern = re.compile(r"examples/data/([A-Za-z0-9_./-]+)")
    for surface in ("src/pharmaml/cli.py", "README.md"):
        text = (ROOT / surface).read_text(encoding="utf-8")
        names = {n.rstrip("/.") for n in pattern.findall(text)}
        names.discard("")
        stale = sorted(n for n in names if not (DATA / n).exists())
        unowned = sorted(n for n in names if n not in owned and n not in owned_dirs)
        print(f"  {surface:<24} {len(names)} example-data reference(s)")
        _ok(bool(names), f"{surface}: the scan still finds references (pattern rot?)")
        _ok(not stale, f"{surface}: every reference resolves (stale: {stale})")
        _ok(
            not unowned,
            f"{surface}: every reference is builder-owned (unowned: {unowned})",
        )


def main() -> None:
    print("PROVENANCE ROUND-TRIP AUDIT (100x-115, extended 100x-116)")
    references = leg_notebook_scan()
    leg_no_orphans(references)
    leg_builder_check()
    leg_docs_surfaces()
    print()
    if FAILURES:
        print(f"PROVENANCE ROUND-TRIP: FAIL ({len(FAILURES)} problem(s))")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print(
        "PROVENANCE ROUND-TRIP: PASS — every notebook load resolves, every "
        "shipped file is loaded, examples/data is byte-identical to the "
        "builder output (no orphans, drift gate proven to fire), every "
        "CLI/README promise resolves and is builder-owned, and drift lines "
        "name their owning builder function"
    )


if __name__ == "__main__":
    main()
