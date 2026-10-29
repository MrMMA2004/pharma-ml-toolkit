"""Round-166 notebook re-execution: the two stale notebooks only.

complete_ml_workflow.ipynb and pharmaml_walkthrough.ipynb shipped
stale outputs (printed 1.2.98 under the 1.2.99 tree -- the 1.2.99
round's own debt, caught by test_executed_outputs_print_shipped_version).
This driver re-executes exactly those two through the jupyter_client
direct path -- the reply-first, iopub-drain-later kernel driving that
reexecute_nb_client_164.py proved under this sandbox's memory pressure
-- keeping its site-packages scrub contract intact.

Run: python scratch/reexecute_nb_166.py
"""
from __future__ import annotations

import os
import sys
import sysconfig
import time
from pathlib import Path

for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.pop("MPLBACKEND", None)

import nbformat
from jupyter_client.manager import KernelManager

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
_SITE_PACKAGES = sysconfig.get_paths()["purelib"]
NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
)


def _execute_notebook(path: Path) -> tuple[int, int, int]:
    """Run every code cell in one fresh kernel, outputs into the nb.

    Returns ``(code cells, executed, error outputs)``.
    """
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(path.parent))
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=120)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            print(f"  cell: {(cell.source or '').splitlines()[0][:70]}", flush=True)
            msg_id = kc.execute(cell.source)
            outputs: list[dict] = []
            execution_count = None
            # the AUTHORITATIVE completion signal is the shell reply --
            # the iopub SUB socket can DROP the idle status under a big
            # PNG burst (ZMQ PUB/SUB high-water marks); wait for the
            # reply FIRST, then drain iopub best-effort.
            reply = kc.get_shell_msg(timeout=3600)
            while reply.get("parent_header", {}).get("msg_id") != msg_id:
                reply = kc.get_shell_msg(timeout=3600)
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=1.0)
                except Exception:  # noqa: BLE001 - queue drained
                    if time.monotonic() > deadline:
                        break
                    continue
                if msg["parent_header"].get("msg_id") != msg_id:
                    deadline = time.monotonic() + 5.0
                    continue
                kind = msg["msg_type"]
                content = msg["content"]
                if kind == "status":
                    if content["execution_state"] == "idle":
                        break
                    continue
                deadline = time.monotonic() + 5.0
                if kind == "execute_input":
                    execution_count = content["execution_count"]
                elif kind == "error":
                    outputs.append({
                        "output_type": "error",
                        "ename": content["ename"],
                        "evalue": content["evalue"],
                        "traceback": content.get("traceback", []),
                    })
                elif kind == "stream":
                    outputs.append({
                        "output_type": "stream",
                        "name": content["name"],
                        "text": content["text"],
                    })
                elif kind in ("execute_result", "display_data"):
                    entry = {
                        "output_type": kind,
                        "data": content["data"],
                        "metadata": content.get("metadata", {}),
                    }
                    if kind == "execute_result":
                        entry["execution_count"] = execution_count
                    outputs.append(entry)
            cell.outputs = [nbformat.from_dict(o) for o in outputs]
            if execution_count is not None:
                cell.execution_count = execution_count
        # the site-packages scrub (the 100x-101 rule)
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    raw = output.get("text", "")
                    if isinstance(raw, str):
                        raw = [raw]
                    joined = "".join(
                        t.replace(_SITE_PACKAGES + "/", "site-packages/")
                        for t in raw
                    )
                    output["text"] = joined.splitlines(keepends=True)
        nbformat.write(nb, path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [
            o for c in code_cells for o in c.get("outputs", [])
            if o.get("output_type") == "error"
        ]
        return len(code_cells), executed, len(errors)
    finally:
        try:
            kc.stop_channels()
            km.shutdown_kernel(now=True)
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            print(f"kernel teardown note: {exc}", flush=True)


def main() -> int:
    failures = 0
    total = time.perf_counter()
    for name in NOTEBOOKS:
        path = EXAMPLES / name
        print(f"== {name}", flush=True)
        started = time.perf_counter()
        try:
            cells, executed, errors = _execute_notebook(path)
            print(f"   {executed}/{cells} cells, {errors} error(s), "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
            if errors or executed != cells:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {exc}", flush=True)
            failures += 1
    print(f"reexecute_nb_166: {'FAILURES' if failures else 'ALL OK'} "
          f"({time.perf_counter() - total:.0f}s total)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
