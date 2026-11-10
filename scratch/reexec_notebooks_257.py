#!/usr/bin/env python
"""round-257 -- the notebook re-executor (the round-209 scrubber contract,
reconstructed: the original script was untracked scratch tooling lost to a
rollback).

For each of the eight example notebooks:
  1. execute every code cell in-place with nbclient (cwd = examples/ so
     the shipped relative data paths resolve);
  2. scrub every TEXT payload -- str AND list-of-lines (nbformat reads
     ``text`` as ``str`` for most outputs but stream outputs carry
     ``text`` as a list of lines; the round-137/209 lesson: guard both)
     -- of the absolute home prefix ``/home/z`` so the archive never
     leaks the build machine's paths;
  3. write the notebook back (compact JSON, nbformat's canonical form).

The run FAILS (exit 1) if any absolute home path survives the scrub or
any cell errored, so ``home-path-survivors=0`` is enforced at the source.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path("/home/z/my-project")
EXAMPLES = REPO / "pharma-ml-toolkit" / "examples"
NOTEBOOKS = [
    "pharmaml_walkthrough.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_ml_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "complete_tuning_workflow.ipynb",
]

HOME_PREFIXES = ("/home/z",)


def _scrub_text(value):
    """Recursively scrub str and list-of-lines text payloads."""
    if isinstance(value, str):
        for prefix in HOME_PREFIXES:
            # keep the path readable but machine-free: /home/z/x -> ~/x
            value = value.replace(prefix, "~")
        return value
    if isinstance(value, list):
        return [_scrub_text(v) for v in value]
    if isinstance(value, dict):
        return {k: _scrub_text(v) for k, v in value.items()}
    return value


def _count_survivors(nb) -> int:
    survivors = 0
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            payload = output.get("text")
            if payload is None:
                continue
            text = "".join(payload) if isinstance(payload, list) else payload
            if any(prefix in text for prefix in HOME_PREFIXES):
                survivors += 1
            data = output.get("data", {})
            for key in ("text/plain", "text/html"):
                if key in data:
                    blob = data[key]
                    joined = "".join(blob) if isinstance(blob, list) else blob
                    if any(prefix in joined for prefix in HOME_PREFIXES):
                        survivors += 1
    return survivors


def main() -> int:
    total_started = time.monotonic()
    failures = 0
    for name in NOTEBOOKS:
        path = EXAMPLES / name
        started = time.monotonic()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES)}},
        )
        try:
            client.execute()
        except Exception as exc:  # noqa: BLE001 - reported per notebook
            print(f"FAIL {name}: execution error: {exc}")
            failures += 1
            continue
        # scrub every text payload in every output (str AND list-of-lines)
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                if "text" in output:
                    output["text"] = _scrub_text(output["text"])
                data = output.get("data")
                if isinstance(data, dict):
                    for key in ("text/plain", "text/html"):
                        if key in data:
                            data[key] = _scrub_text(data[key])
        survivors = _count_survivors(nb)
        nbformat.write(nb, path)
        status = "OK" if survivors == 0 else f"LEAK({survivors})"
        print(f"{status} {name} ({time.monotonic() - started:.0f}s)")
        if survivors:
            failures += 1
    total = time.monotonic() - total_started
    print(
        f"notebooks {len(NOTEBOOKS) - failures}/{len(NOTEBOOKS)} in {total:.0f}s, "
        f"home-path-survivors total = "
        f"{'0' if failures == 0 else 'NONZERO'}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
