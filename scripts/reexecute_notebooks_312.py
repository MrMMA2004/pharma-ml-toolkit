#!/usr/bin/env python3
"""Round 312: re-execute the 8 notebooks at 1.2.234 (foreground) with the
canonical builder-path scrubber INTEGRATED before write-back.

Same contract as the round-311 driver (resumable: a notebook that already
prints 1.2.234 is skipped; the «venv»/lib + «home» replacement keeps the
zip leak scan at real leaks 0 by construction). The no-output twins ride
the CANONICAL tools/strip_notebook_outputs.py after execution (the r310
defect class: this driver's own strip_twin lost the variant-note cell).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

VERSION = "1.2.234"
EXAMPLES = Path("/home/z/my-project/pharma-ml-toolkit/examples")
WORK = Path("/tmp/nbwork_312")

EXECUTED = (
    "complete_ml_workflow",
    "complete_docking_workflow",
    "complete_pk_workflow",
    "complete_cellline_workflow",
    "complete_generative_workflow",
    "complete_deep_learning_workflow",
    "complete_tuning_workflow",
    "pharmaml_walkthrough",
)
VARIANTS = tuple(n for n in EXECUTED if n != "pharmaml_walkthrough")

_HOME = Path.home()
_VENV_SITE = str(_HOME / ".venv" / "lib")


def _scrub_text(text: str) -> str:
    return text.replace(_VENV_SITE, "«venv»/lib").replace(str(_HOME), "«home»")


def _scrub_payload(value: Any) -> Any:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, list):
        return _scrub_text("".join(value)).splitlines(keepends=True)
    return value


def _scrub_notebook(nb) -> int:
    """Scrub outputs in place; return how many home-path survivors remain."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if "text" in out:
                out["text"] = _scrub_payload(out["text"])
            data = out.get("data")
            if data:
                for key in list(data.keys()):
                    data[key] = _scrub_payload(data[key])
    return nbformat.writes(nb).count(str(_HOME))


def already_fresh(name: str) -> bool:
    src = EXAMPLES / f"{name}.ipynb"
    if not src.exists():
        return False
    try:
        nb = nbformat.read(src, as_version=4)
    except Exception:
        return False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = ""
            if out.output_type == "stream":
                text = out.get("text", "")
            elif out.output_type in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
            if VERSION in text:
                return True
    return False


def run_one(name: str) -> dict:
    src = EXAMPLES / f"{name}.ipynb"
    work_nb = WORK / f"{name}.ipynb"
    shutil.copy2(src, work_nb)
    nb = nbformat.read(work_nb, as_version=4)
    started = time.monotonic()
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(WORK)}},
    )
    error = ""
    try:
        client.execute()
    except Exception as exc:  # noqa: BLE001 - report and continue counting
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.monotonic() - started
    survivors = _scrub_notebook(nb) if not error else -1
    if not error:
        nbformat.write(nb, src)  # scrubbed executed result -> examples/
    prints_version = False
    error_outputs = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = ""
            if out.output_type == "stream":
                text = out.get("text", "")
            elif out.output_type in ("execute_result", "display_data"):
                text = out.get("data", {}).get("text/plain", "")
            elif out.output_type == "error":
                error_outputs += 1
            if VERSION in text:
                prints_version = True
    return {
        "name": name,
        "seconds": seconds,
        "error": error,
        "error_outputs": error_outputs,
        "prints_version": prints_version,
        "survivors": survivors,
        "nb": nb,
    }


def strip_twin(nb) -> object:
    """The no-output twin: same cells, outputs stripped, counts cleared."""
    twin = nbformat.v4.new_notebook()
    twin.metadata = nb.metadata
    for cell in nb.cells:
        if cell.cell_type == "code":
            fresh = nbformat.v4.new_code_cell(cell.source)
        elif cell.cell_type == "markdown":
            fresh = nbformat.v4.new_markdown_cell(cell.source)
        else:
            fresh = nbformat.v4.new_raw_cell(cell.source)
        fresh.id = cell.id
        twin.cells.append(fresh)
    return twin


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "data").exists():
        shutil.copytree(EXAMPLES / "data", WORK / "data")
    total_started = time.monotonic()
    results = []
    for name in EXECUTED:
        if already_fresh(name):
            print(f"[SKIP] {name}: already prints {VERSION}")
            continue
        result = run_one(name)
        results.append(result)
        flag = "OK" if not result["error"] and not result["error_outputs"] else "FAIL"
        print(
            f"[{flag}] {result['name']}: {result['seconds']:.0f}s, "
            f"error_outputs={result['error_outputs']}, prints_{VERSION}={result['prints_version']}, "
            f"home-path survivors={result['survivors']}"
            + (f" -- {result['error'][:200]}" if result["error"] else "")
        )
    print(f"total: {time.monotonic() - total_started:.0f}s")
    failures = [r for r in results if r["error"] or r["error_outputs"]]
    print(f"failures: {len(failures)} / {len(results)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
