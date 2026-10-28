"""Round-164 notebook re-execution: the jupyter_client direct driver.

The nbclient path died on the ML workflow notebook under this sandbox's
memory pressure (DeadKernelError at ~64 s, peak kernel RSS ~1.1 GB);
driving the SAME kernel through jupyter_client directly -- one cell at
a time, outputs collected manually -- completes every cell.  This
driver reproduces reexecute_all_notebooks_64.py's contract (fresh
outputs, execution counts, the site-packages scrub, the no-output
twins) through that lighter path.
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
    "complete_docking_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "complete_tuning_workflow.ipynb",
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
            print(f"  cell starting ({cell.source.splitlines()[0][:60] if cell.source else ''})", flush=True)
            msg_id = kc.execute(cell.source)
            outputs: list[dict] = []
            execution_count = None
            # the AUTHORITATIVE completion signal is the shell reply --
            # the iopub SUB socket can DROP the idle status under a big
            # PNG burst (ZMQ PUB/SUB high-water marks), which stalled the
            # first driver for an hour.  Wait for the reply FIRST, then
            # drain iopub best-effort.
            reply = kc.get_shell_msg(timeout=3600)
            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                # a stale reply from an earlier cell (never observed,
                # but never fatal): keep only matching ones
                while reply.get("parent_header", {}).get("msg_id") != msg_id:
                    reply = kc.get_shell_msg(timeout=3600)
            # drain the iopub stream: everything the kernel published
            # for THIS cell (skipping other parents), stopping at idle
            # or after a short tail grace period once the queue empties
            import time as _time

            deadline = _time.monotonic() + 5.0
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=1.0)
                except Exception:  # noqa: BLE001 - queue drained
                    if _time.monotonic() > deadline:
                        break
                    continue
                if msg["parent_header"].get("msg_id") != msg_id:
                    deadline = _time.monotonic() + 5.0
                    continue
                kind = msg["msg_type"]
                content = msg["content"]
                if kind == "status":
                    if content["execution_state"] == "idle":
                        break
                    continue
                deadline = _time.monotonic() + 5.0
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
    total = time.perf_counter()
    for name in NOTEBOOKS:
        path = EXAMPLES / name
        started = time.perf_counter()
        n_cells, executed, n_errors = _execute_notebook(path)
        if n_errors or executed != n_cells:
            print(f"FAIL {name}: {n_errors} error(s), {executed}/{n_cells} cells",
                  flush=True)
            return 1
        print(f"executed {name}: {n_cells} cells, "
              f"{time.perf_counter() - started:.0f} s", flush=True)
    print(f"all notebooks refreshed in {time.perf_counter() - total:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
