"""Round 249: re-execute the 8 notebooks at 1.2.161 (foreground) + regenerate the 7 no-output twins.

Lost-driver reconstruction (repo-root scratch/ rolled back). Same contract as
the round-64-era driver: ExecutePreprocessor in-process, one notebook at a
time, report seconds + error outputs + whether the notebook prints 1.2.161.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

EXAMPLES = Path("/home/z/my-project/pharma-ml-toolkit/examples")
WORK = Path("/tmp/nbwork_249")  # the lost driver's discipline: notebooks run in a WORK dir
                                # (relative data/ resolves, housekeeping writes never touch the
                                # shipped tree); the executed result is written back to examples/.

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
        nbformat.write(nb, src)  # the executed result lands back in examples/ (nbclient mutates the in-memory nb; the work FILE was never written)
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
            if "1.2.161" in text:
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
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    shutil.copytree(EXAMPLES / "data", WORK / "data")
    total_started = time.monotonic()
    results = []
    for name in EXECUTED:
        result = run_one(name)
        results.append(result)
        flag = "OK" if not result["error"] and not result["error_outputs"] else "FAIL"
        print(
            f"[{flag}] {result['name']}: {result['seconds']:.0f}s, "
            f"error_outputs={result['error_outputs']}, prints_1.2.161={result['prints_version']}"
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


if __name__ == "__main__":
    main()
