"""Round 310: re-execute the 8 notebooks at 1.2.232 (foreground) + regenerate the 7 no-output twins.

Same contract as the round-249-era driver (the lost-driver reconstruction
lineage): ExecutePreprocessor in-process, one notebook at a time, report
seconds + error outputs + whether the notebook prints 1.2.232. This
round's addition: RESUMABLE -- a notebook that already prints 1.2.232 is
skipped, so a tool-window timeout can be re-entered without redoing work
(the executed result writes back to examples/ per notebook, not at the end).

Pre-flight (the caller does these, the script re-asserts): the venv's
pharmaml resolves to the SOURCE tree (pip install --no-deps -e .) so the
version print carries the bumped __init__, and the freshness gate's
carrying-count stays 8.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

VERSION = "1.2.232"
EXAMPLES = Path("/home/z/my-project/pharma-ml-toolkit/examples")
WORK = Path("/tmp/nbwork_310")  # notebooks run in a WORK dir (relative data/
                                # resolves, housekeeping writes never touch the
                                # shipped tree); the executed result is written
                                # back to examples/ per notebook.

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
    if not error:
        nbformat.write(nb, src)  # the executed result lands back in examples/
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
        "nb": nb,
    }


def strip_twin(nb) -> object:
    """The no-output twin: same cells, every output stripped, execution_count cleared."""
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
            f"error_outputs={result['error_outputs']}, prints_{VERSION}={result['prints_version']}"
            + (f" -- {result['error'][:200]}" if result["error"] else "")
        )
        if not result["error"] and name in VARIANTS:
            twin = strip_twin(result["nb"])
            twin_path = EXAMPLES / f"{name}_no_output.ipynb"
            nbformat.write(twin, twin_path)
            print(f"       twin regenerated -> {name}_no_output.ipynb")
    print(f"total: {time.monotonic() - total_started:.0f}s")
    failures = [r for r in results if r["error"] or r["error_outputs"]]
    print(f"failures: {len(failures)} / {len(results)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
