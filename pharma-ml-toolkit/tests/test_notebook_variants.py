"""Guards for the notebook variants and the no-inline-fabrication contract.

Five guard families live here:

* ``TestNoOutputVariants`` — the ``*_no_output.ipynb`` twins are derived
  artifacts (scratch/strip_notebook_outputs.py): identical cell sources to
  the executed original, every output stripped, plus one inserted
  explanation note.  These tests pin all three properties so the twins can
  never silently fork from the notebooks readers actually read.
* ``test_no_inline_fabrication`` — the user's standing requirement (round
  24): "Jupyter files must work entirely based on the files I give them
  and not create their own sample data."  Every executed notebook loads
  its demo data from the shipped ``examples/data/`` inventory (built
  byte-identically by ``scratch/build_example_data.py``, whose provenance
  map names the exact notebook cell each file came from) — the parametrized
  guard below keeps it that way.
* ``test_executed_outputs_print_shipped_version`` — the freshness guard
  (round 100x-64): every executed notebook's setup cell prints
  ``pharmaml <version>`` into the committed outputs; the guard pins that
  print to the *shipped* ``__version__`` so committed outputs can never
  silently predate the tree that ships them (the 1.0.0-era drift this
  round cleaned up).
* ``test_ml_workflow_persistence_audit_is_committed`` — the §12 audit
  demo guard (round 100x-78): the ML workflow demonstrates
  ``validate_bundle_input`` on a drifted copy of its own shipped frame;
  the guard pins the committed verdicts so the demo can never silently
  become stale or un-executed.
* ``test_docking_workflow_consensus_breakdown_is_committed`` — the §9
  breakdown demo guard (round 100x-83): the docking workflow
  demonstrates ``consensus_breakdown`` (the introspection twin of
  ``consensus_rescore`` from 100x-82) on its own rescoring frame; the
  guard pins the committed driving-evidence verdict so the demo can
  never silently become stale or un-executed.
* ``test_tuning_workflow_quartet_is_committed`` — the four-feature
  guard (round 100x-89): the tuning workflow demonstrates the whole
  1.2.36 quartet (3D-RISM-KH descriptors, HSVR, the tuner, batchnorm);
  the guard pins the committed verdicts — the HSVR-vs-global R² pair,
  the tuner's gain line, Hyperband's resource ladder and savings, and
  the batchnorm A/B — so the demo can never silently become stale or
  un-executed.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

VARIANTS = (
    "complete_ml_workflow",
    "complete_docking_workflow",
    "complete_pk_workflow",
    "complete_cellline_workflow",
    "complete_generative_workflow",
    "complete_deep_learning_workflow",
    "complete_tuning_workflow",
)

EXECUTED_NOTEBOOKS = VARIANTS + ("pharmaml_walkthrough",)


def _read(name: str):
    path = EXAMPLES / f"{name}.ipynb"
    assert path.is_file(), f"examples/{name}.ipynb missing"
    return nbformat.read(path, as_version=4)


@pytest.mark.parametrize("name", VARIANTS)
class TestNoOutputVariants:
    def test_variant_exists(self, name: str):
        assert (EXAMPLES / f"{name}_no_output.ipynb").is_file()

    def test_outputs_stripped(self, name: str):
        nb = _read(f"{name}_no_output")
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        assert code_cells, "variant has no code cells"
        for cell in code_cells:
            assert cell.get("outputs", []) == [], "output was not stripped"
            assert cell.get("execution_count") is None

    def test_note_cell_inserted(self, name: str):
        nb = _read(f"{name}_no_output")
        note = nb.cells[1]
        assert note.cell_type == "markdown"
        assert "No-output variant" in note.source
        assert name in note.source, "note must name the full twin"

    def test_sources_match_original(self, name: str):
        """The one guarantee that matters: no drift between the twins."""
        original = _read(name)
        variant = _read(f"{name}_no_output")
        original_sources = [c.source for c in original.cells]
        variant_sources = [
            c.source for c in variant.cells if c.get("id") != "variant-note"
        ]
        assert variant_sources == original_sources, (
            f"{name}_no_output.ipynb drifted from {name}.ipynb — "
            "re-run scratch/strip_notebook_outputs.py"
        )

    def test_variant_is_actually_smaller(self, name: str):
        full = (EXAMPLES / f"{name}.ipynb").stat().st_size
        slim = (EXAMPLES / f"{name}_no_output.ipynb").stat().st_size
        assert slim < full * 0.5, (
            f"variant ({slim} B) is not meaningfully smaller than the "
            f"original ({full} B)"
        )


@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_executed_outputs_print_shipped_version(name: str):
    """Committed outputs must come from the code that ships with them.

    Every executed notebook prints ``pharmaml <version>`` in its setup
    cell; the committed stream output records the version that actually
    produced the outputs.  A mismatch means the outputs predate the
    shipped tree (e.g. 1.0.0-era outputs shipped with 1.2.x code) — after
    any version bump, re-run scratch/reexecute_all_notebooks_64.py and
    regenerate the twins so the committed evidence stays honest.
    """
    from pharmaml import __version__

    nb = _read(name)
    printed: set[str] = set()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", [])) or "".join(
                out.get("data", {}).get("text/plain", [])
            )
            printed.update(
                m.group(1)
                for m in re.finditer(r"pharmaml (\d+\.\d+\.\d+)", text)
            )
    assert printed, (
        f"{name}: no committed output prints the pharmaml version — "
        "every executed notebook's setup cell should print it"
    )
    stale = printed - {__version__}
    assert not stale, (
        f"{name}: committed outputs print pharmaml {sorted(stale)} but "
        f"the shipped tree is {__version__} — re-run "
        "scratch/reexecute_all_notebooks_64.py (then the twins via "
        "scratch/strip_notebook_outputs.py)"
    )


def test_ml_workflow_persistence_audit_is_committed():
    """The §12 persistence-audit demo (round 100x-78) must stay executed.

    ``validate_bundle_input`` — the inference-time contract audit from
    100x-77 (library + the ``ml check`` CLI) — is demonstrated in the ML
    workflow's §12 on a drifted copy of the notebook's own shipped frame
    (one feature dropped, one renamed, one brand-new column) plus the
    reorder trap.  The committed stream outputs are the evidence the
    demo actually ran: an OK verdict on the training frame, a FAIL
    verdict naming the exact drift, and the reorder rejection.  A cell
    edit nobody re-executed, or a verdict that changed shape, fails here.
    """
    nb = _read("complete_ml_workflow")
    sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from pharmaml.ml import validate_bundle_input" in sources, (
        "complete_ml_workflow lost the §12 audit demo"
    )
    assert "drifted = ds.X.drop(columns=[ds.feature_names[0]])" in sources

    outputs = "\n".join(
        "".join(out.get("text", []))
        for c in nb.cells
        if c.cell_type == "code"
        for out in c.get("outputs", [])
    )
    assert "audit: OK" in outputs
    assert "missing 2 feature(s): descr_00, descr_01" in outputs
    assert (
        "2 extra column(s) the model does not know: renamed_feature, batch_id"
        in outputs
    ), "the drift demo's named extras changed — re-execute the notebook"
    assert "feature columns are in a different order than trained on" in outputs


def test_docking_workflow_consensus_breakdown_is_committed():
    """The §9 consensus-breakdown demo (round 100x-83) must stay executed.

    ``consensus_breakdown`` — the introspection twin of
    ``consensus_rescore`` from 100x-82 (library + the ``docking rescore``
    CLI) — is demonstrated in the docking workflow's §9 on the
    notebook's own rescoring frame: the explainable table (one ``_norm``
    unit per evidence column next to the consensus) and the
    driving-evidence verdict for the top-ranked ligand.  The committed
    outputs are the evidence the demo actually ran: the top ligand's row
    of 0–1 units and the argmax verdict naming the evidence that carried
    the rank.  A cell edit nobody re-executed, or a verdict that changed
    shape, fails here.
    """
    nb = _read("complete_docking_workflow")
    sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from pharmaml.docking import consensus_breakdown" in sources, (
        "complete_docking_workflow lost the §9 breakdown demo"
    )
    assert 'breakdown = consensus_breakdown(' in sources
    assert (
        'ok, columns=["best_lowest_energy", "best_num_in_cluster",'
        ' "ligand_efficiency"]' in sources
    )
    assert 'driver = top_row.drop("consensus").idxmax()' in sources

    outputs = "\n".join(
        "".join(out.get("text", []))
        or "".join(out.get("data", {}).get("text/plain", []))
        for c in nb.cells
        if c.cell_type == "code"
        for out in c.get("outputs", [])
    )
    assert "folder  best_lowest_energy_norm  best_num_in_cluster_norm" in (
        outputs
    ), "the explainable table is gone — re-execute the notebook"
    assert "07_metformin tops the table (consensus 0.782):" in outputs
    assert "  ligand_efficiency_norm     1.000" in outputs
    assert "driving evidence: ligand_efficiency_norm = 1.000" in outputs, (
        "the driving-evidence verdict changed — re-execute the notebook"
    )


def test_tuning_workflow_quartet_is_committed():
    """The four 1.2.36-feature demos (round 100x-89) must stay executed.

    ``complete_tuning_workflow`` is the executed-notebook face of the
    user's four-feature request: 3D-RISM-KH descriptors (§2),
    HierarchicalSVR vs the global SVR (§3), RandomSearch + Hyperband
    (§4-§5) and the batchnorm A/B (§6).  The committed stream outputs
    are the evidence every demo actually ran: the RISM table for the
    12-drug set, the hsvr/global R² pair, the tuner's defaults-vs-best
    gain line, Hyperband's resource ladder with the epoch savings, and
    the two batchnorm A/B lines.  A cell edit nobody re-executed, or a
    verdict that changed shape, fails here.
    """
    nb = _read("complete_tuning_workflow")
    sources = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "from pharmaml.chem.rism import compute_rism_descriptors" in sources, (
        "complete_tuning_workflow lost the §2 RISM demo"
    )
    assert "from pharmaml.ml.hierarchical import HierarchicalSVR" in sources
    assert "from pharmaml.ml.tuner import Hyperband, build_deep_regressor" in (
        sources
    )
    assert "def fit_variant(batchnorm: bool)" in sources

    outputs = "\n".join(
        "".join(out.get("text", []))
        or "".join(out.get("data", {}).get("text/plain", []))
        for c in nb.cells
        if c.cell_type == "code"
        for out in c.get("outputs", [])
    )
    # §2 — the drug set computed, Gsolv decomposition table committed.
    assert "12 molecules: aspirin" in outputs
    assert "RISM_Gsolv_kcalmol" in outputs
    # §3 — the regime verdicts.
    assert "global SVR : rmse=" in outputs
    assert "hsvr       : rmse=" in outputs
    # §4 — the tuner's apples-to-apples gain line.
    assert "defaults: 0.623 +/- 0.025  ->  best 0.820  (gain +0.197 R2)" in (
        outputs
    ), "the tuner gain line changed — re-execute the notebook"
    # §5 — the resource ladder and the epoch economics.
    assert "epoch budget spent: 783 of a naive 1863 (58% saved)" in outputs, (
        "the Hyperband economics changed — re-execute the notebook"
    )
    # §6 — the A/B lines.
    assert "plain      R2 0.941" in outputs
    assert "batchnorm  R2 0.916" in outputs


# The no-inline-fabrication contract (round 24).  Each token below is a
# code shape that ONLY ever appeared in a real inline data-fabrication
# cell pre-conversion (verified against the round-23 git HEAD state,
# 2026-10-16 — 22 cell-hits total); none of them matches legitimate
# seeded MODELLING code (GA/RL runs and train/test splits take a ``seed``
# argument and never construct their own data frames):
#
#   token                                          pre-conversion cells
#   ---------------------------------------------- ------------------------------
#   from sklearn.datasets import make_classification  walkthrough 5, 10
#   np.random.default_rng(                         ml 3, 11 · cellline 1 · pk 4 ·
#                                                    docking 29 · walkthrough 3
#   rng.normal(                                    ml 3, 11 · cellline 3, 17 ·
#                                                    pk 4 · walkthrough 3
#   rng.choice(                                    ml 3 · walkthrough 3
#   LIBRARY = [                                    ml 6   (79-molecule inline list)
#   "SMILES": [                                    docking 3 (10-row inline table)
#   CAMPAIGN_LIGANDS = [                           docking 29 (14-ligand manifest)
#   SEED_TABLE = {                                 generative 8 (12-drug dict)
#   from make_synthetic_qsar_data import           ml 31, 35
#
# Post-conversion each of those cells loads the corresponding shipped file
# instead (e.g. ml cell 3 -> data/qsar_classification.csv, docking cell 29
# -> data/docking_campaign/ + its ligands.csv manifest, generative cell 8
# -> data/seed_molecules.csv) — see scratch/build_example_data.py for the
# full provenance map and scratch/reexecute_notebooks_26.py for the
# conversion driver.  ZERO allowlist entries: a hit is always a failure.
BANNED_FABRICATION_TOKENS = (
    "from sklearn.datasets import make_classification",
    "np.random.default_rng(",
    "rng.normal(",
    "rng.choice(",
    "LIBRARY = [",
    '"SMILES": [',
    "CAMPAIGN_LIGANDS = [",
    "SEED_TABLE = {",
    "from make_synthetic_qsar_data import",
)


@pytest.mark.parametrize("token", BANNED_FABRICATION_TOKENS)
@pytest.mark.parametrize("name", EXECUTED_NOTEBOOKS)
def test_no_inline_fabrication(name: str, token: str):
    """Executed notebooks must load data from files, never fabricate it.

    8 notebooks x 9 banned tokens = 72 assertions, zero allowlist entries.
    The no-output twins are covered transitively: their sources are pinned
    to these originals by ``TestNoOutputVariants.test_sources_match_original``.
    """
    nb = _read(name)
    hits = [
        i for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and token in cell.source
    ]
    assert not hits, (
        f"{name}.ipynb fabricates input data inline (token {token!r} in "
        f"cell {hits}) — load the corresponding shipped file from "
        "examples/data/ instead (see scratch/build_example_data.py)"
    )
