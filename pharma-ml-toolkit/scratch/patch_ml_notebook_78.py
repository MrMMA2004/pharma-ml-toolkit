"""78 driver: the §12 persistence audit cells (notebook side of 100x-77).

Round 100x-77 shipped ``validate_bundle_input`` (library) and ``ml
check`` (CLI) — the inference-time contract audit for saved model
bundles.  This driver lands the notebook axis: the ML workflow's §12
(save/load) now *demonstrates* the audit on a drifted copy of its own
shipped CSV, so readers see the verdict instead of reading about it.

1. INSERT four cells after the "score five compounds" cell (anchor:
   ``# Score five compounds the model has never seen.``, the tail of
   §12) and BEFORE ``## 13 · Publish``:
   [markdown intro: the production failure mode] + [code: the OK audit
   on the training frame, then the drifted frame — drop one feature,
   rename another, add a brand-new column — with the FAIL verdict and
   ``ok: False``] + [code: the reorder trap — same names, different
   order, sklearn's subtlest rejection] + [markdown: the ``ml check``
   CLI twin with ``--target`` and exit-1-for-CI].
2. PATCH the §12 intro markdown: the deploy paragraph gains the audit
   sentence.
3. PATCH the §14 recap: the Deploy row gains ``validate_bundle_input``.
4. EXECUTE the ML notebook with nbclient (cwd=examples/, MPLBACKEND
   popped so inline PNGs survive — the 100x-43 lesson).  The new cells
   only ADD output: no existing name is rebound (AST-audited —
   validate_bundle_input / drifted / audit_drift / reordered are bound
   by no other cell and mentioned nowhere else), so existing pins
   cannot shift.
5. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
6. VERIFY banned fabrication tokens absent + pinned values present.

Pins (deterministic, probed against examples/data/qsar_classification.csv
before writing — the drift demo derives from the shipped frame, never
fabricates one):

  OK audit    : audit: OK
  drift audit : audit: FAIL
                - missing 2 feature(s): descr_00, descr_01
                - 2 extra column(s) the model does not know:
                  renamed_feature, batch_id
                ok: False
  reorder     : audit: FAIL
                - feature columns are in a different order than
                  trained on (expected: descr_00, ...)
  (the full seven-notebook re-execution follows this driver; the other
  six still print 1.2.27 until reexecute_all_notebooks_64.py runs.)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "tests"))  # BANNED_FABRICATION_TOKENS

from test_notebook_variants import BANNED_FABRICATION_TOKENS

WORKER_ENV = {
    **os.environ,
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

ML_NOTEBOOK = "complete_ml_workflow.ipynb"

AUDIT_MARKER = "Score five compounds the model has never seen"  # cell 47

AUDIT_MD_INTRO = """\
### Audit the input before you trust the prediction

The production failure mode is quiet: the model was saved months ago,
the CSV got regenerated, a column was renamed, a new one appeared — and
instead of a prediction you get a cryptic sklearn feature-count crash
deep inside `predict`. `validate_bundle_input` checks any new frame
against the bundle's recorded contract *before* the first call and
names every difference — the inference-time twin of the provenance
`save_model` wrote.
"""

AUDIT_CODE_DRIFT = '''\
from pharmaml.ml import validate_bundle_input

# The frame the model was actually trained on: the contract holds.
print(validate_bundle_input(MODEL_PATH, ds.X))
print()

# The months-later CSV: one feature dropped, one renamed, one brand-new
# column -- the three ways real data drifts, all named in one verdict.
drifted = ds.X.drop(columns=[ds.feature_names[0]]).rename(
    columns={ds.feature_names[1]: "renamed_feature"}
)
drifted["batch_id"] = 1.0

audit_drift = validate_bundle_input(MODEL_PATH, drifted)
print(audit_drift)
print("ok:", audit_drift.ok)
'''

AUDIT_CODE_REORDER = '''\
# The subtlest drift: the same columns in a different order.  sklearn
# rejects the frame even though the *set* of names matches -- the audit
# catches it before predict() does.
reordered = ds.X[[ds.feature_names[1], ds.feature_names[0]] + ds.feature_names[2:]]
print(validate_bundle_input(MODEL_PATH, reordered))
'''

AUDIT_MD_CLI = """\
The same audit ships from the terminal — `pharmaml ml check` prints the
bundle's provenance block plus the verdict and **exits 1 on failure**
(CI-friendly); `--target` drops the training label when the CSV still
carries it:

```text
pharmaml ml check --model rf_activity.joblib --data new.csv --target activity
```
"""

# --- cell replacements (old snippet -> new snippet) ------------------------

DEPLOY_OLD = (
    "`save_model` writes a joblib bundle *with provenance metadata*; `load_model`\n"
    "round-trips it. New, unseen rows are scored with one `predict` call."
)
DEPLOY_NEW = (
    "`save_model` writes a joblib bundle *with provenance metadata*; `load_model`\n"
    "round-trips it. New, unseen rows are scored with one `predict` call — and\n"
    "`validate_bundle_input` audits any new frame against the bundle's recorded\n"
    "contract *before* the first predict, naming every way the data has drifted."
)

RECAP_OLD = "| Deploy | `save_model(...)` / `load_model(path)` |"
RECAP_NEW = (
    "| Deploy | `save_model(...)` / `load_model(path)` / "
    "`validate_bundle_input(path, X)` |"
)

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    ML_NOTEBOOK: (
        AUDIT_MARKER,
        [
            ("markdown", AUDIT_MD_INTRO),
            ("code", AUDIT_CODE_DRIFT),
            ("code", AUDIT_CODE_REORDER),
            ("markdown", AUDIT_MD_CLI),
        ],
    ),
}

REPLACEMENTS = {
    ML_NOTEBOOK: [
        (DEPLOY_OLD, DEPLOY_NEW),
        (RECAP_OLD, RECAP_NEW),
    ],
}

PINS_OUTPUT = {
    ML_NOTEBOOK: [
        # -- the three audit verdicts (this round's cells) ------------------
        "audit: OK",
        "audit: FAIL",
        "missing 2 feature(s): descr_00, descr_01",
        "2 extra column(s) the model does not know: renamed_feature, batch_id",
        "ok: False",
        "feature columns are in a different order than trained on",
        # -- §12 save/load pins (must survive) ------------------------------
        "reloaded: random_forest | trained:",
    ],
}

PINS_SOURCE = {
    ML_NOTEBOOK: [
        # -- the audit cells -------------------------------------------------
        "from pharmaml.ml import validate_bundle_input",
        "drifted = ds.X.drop(columns=[ds.feature_names[0]])",
        'drifted["batch_id"] = 1.0',
        "The subtlest drift: the same columns in a different order",
        "pharmaml ml check --model rf_activity.joblib",
        "exits 1 on failure",
        # -- the deploy + recap edits -----------------------------------------
        "`validate_bundle_input` audits any new frame",
        "`validate_bundle_input(path, X)` |",
        # -- §12 source pins (must survive untouched) -------------------------
        "from pharmaml.ml import save_model, load_model",
        "MODEL_PATH = WORK / \"rf_activity.joblib\"",
        "## 13 · Publish",
    ],
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def insert_cells() -> None:
    import nbformat

    for name, (marker, cells) in INSERTIONS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        if any(AUDIT_MARKER in "".join(c.source) and "validate_bundle_input" in "".join(
            c.source
        ) for c in nb.cells):
            print(f"insert: {name} already carries the audit cells")
            continue
        anchor = next(
            (i for i, c in enumerate(nb.cells) if marker in "".join(c.source)),
            None,
        )
        if anchor is None:
            _fail(f"{name}: anchor marker {marker!r} not found")
        for offset, (kind, source) in enumerate(cells, start=1):
            if kind == "markdown":
                nb.cells.insert(
                    anchor + offset, nbformat.v4.new_markdown_cell(source)
                )
            else:
                nb.cells.insert(anchor + offset, nbformat.v4.new_code_cell(source))
        nbformat.write(nb, path)
        print(f"inserted {len(cells)} cell(s) into {name} after cell {anchor}")


def replace_cells() -> None:
    import nbformat

    for name, pairs in REPLACEMENTS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        changed = 0
        for old, new in pairs:
            applied = False
            for cell in nb.cells:
                src = "".join(cell.source)
                if new in src:
                    applied = True  # idempotent re-run: already applied
                    break
                if old in src:
                    cell.source = src.replace(old, new, 1)
                    applied = True
                    changed += 1
                    break
            if not applied:
                _fail(f"{name}: replacement anchor not found:\n{old[:80]}...")
        if changed:
            nbformat.write(nb, path)
        print(f"{name}: {changed} cell replacement(s) applied")


def execute_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

    # The kernel must NOT inherit MPLBACKEND=Agg (the repo's pytest
    # default): with it, matplotlib_inline never activates and every
    # figure cell commits a text/plain-only repr -- the inline PNGs the
    # notebook guards pin would silently vanish.  Unset it for the
    # kernel spawn so ipykernel picks the inline backend.
    os.environ.pop("MPLBACKEND", None)
    for name in (ML_NOTEBOOK,):  # only the ML notebook this round
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=2400,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES)}},
        )
        client.execute()
        nbformat.write(nb, path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [
            o
            for c in code_cells
            for o in c.get("outputs", [])
            if o.output_type == "error"
        ]
        if errors or executed != len(code_cells):
            _fail(
                f"{name}: {len(errors)} error output(s), "
                f"{executed}/{len(code_cells)} cells executed"
            )
        print(
            f"executed {name}: {len(code_cells)} code cells, "
            f"{time.perf_counter() - started:.1f}s"
        )


def regenerate_twin() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        capture_output=True,
        text=True,
        check=False,
        env=WORKER_ENV,
    )
    if result.returncode != 0:
        _fail(f"strip_notebook_outputs.py failed: {result.stderr[-400:]}")
    print("twins regenerated (idempotent)")


def _notebook_text(path: Path) -> str:
    """Source + output text of a notebook, JSON-decoded (raw file text
    escapes embedded quotes, which would break every pin containing one)."""
    import nbformat

    nb = nbformat.read(path, as_version=4)
    chunks: list[str] = []
    for cell in nb.cells:
        chunks.append("".join(cell.source))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                chunks.append("".join(output.get("text", [])))
            elif output.get("output_type") in ("execute_result", "display_data"):
                text = output.get("data", {}).get("text/plain", [])
                chunks.append("".join(text) if isinstance(text, list) else str(text))
    return "\n".join(chunks)


def verify() -> None:
    for name, source_pins in PINS_SOURCE.items():
        raw = _notebook_text(EXAMPLES / name)
        # the executed notebook carries source + outputs; the twin carries
        # source only -- check BOTH so the narrative cannot drift from the
        # artifact.
        twin = _notebook_text(
            EXAMPLES / name.replace(".ipynb", "_no_output.ipynb")
        )
        for pin in source_pins:
            if pin not in raw:
                _fail(f"{name}: source pin missing from executed notebook: {pin[:70]}")
            if pin not in twin:
                _fail(f"twin: source pin missing: {pin[:70]}")
        for pin in PINS_OUTPUT[name]:
            if pin not in raw:
                _fail(f"{name}: output pin missing from executed notebook: {pin[:70]}")
        raw_file = (EXAMPLES / name).read_text(encoding="utf-8")
        twin_file = (EXAMPLES / name.replace(".ipynb", "_no_output.ipynb")).read_text(
            encoding="utf-8"
        )
        for token in BANNED_FABRICATION_TOKENS:
            if token in raw_file or token in twin_file:
                _fail(f"{name}: banned fabrication token present: {token!r}")
        print(
            f"verified {name}: {len(source_pins)} source pins + "
            f"{len(PINS_OUTPUT[name])} output pins + fabrication gate"
        )


def main() -> None:
    insert_cells()
    replace_cells()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("driver 78 complete")


if __name__ == "__main__":
    main()
