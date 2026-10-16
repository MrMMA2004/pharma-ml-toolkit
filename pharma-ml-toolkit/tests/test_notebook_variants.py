"""Guards for the notebook variants and the no-inline-fabrication contract.

Two guard families live here:

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
"""

from __future__ import annotations

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

    6 notebooks x 9 banned tokens = 54 assertions, zero allowlist entries.
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
