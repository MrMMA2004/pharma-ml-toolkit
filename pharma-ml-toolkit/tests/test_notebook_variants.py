"""Guards for the no-output notebook variants.

The ``*_no_output.ipynb`` twins are derived artifacts (scratch/
strip_notebook_outputs.py): identical cell sources to the executed
original, every output stripped, plus one inserted explanation note.
These tests pin all three properties so the twins can never silently
fork from the notebooks readers actually read.
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
)


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
