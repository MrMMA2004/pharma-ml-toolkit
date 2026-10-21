"""Data-provenance round-trip guards (100x-115).

The 100x-104 open item, finally closed: "every CSV the NOTEBOOKS load ==
the file the builder writes".  Three legs, each pinned here:

1. **Notebook-source scan** — every ``DATA_DIR / "<name>"`` reference in
   the EXECUTED notebooks' code cells must resolve to something on disk
   (catches a notebook pointing at a stale path after a rename).
2. **No orphans** — every top-level entry under ``examples/data`` must be
   loaded by at least one notebook (catches data the demos forgot).
3. **Builder round-trip** — ``scratch/build_example_data.py --check``
   re-generates the whole inventory in memory and proves the files on
   disk are byte-identical to the builder output, with no unclaimed
   orphans; the planted-drift test proves the gate actually fires.

Before this, leg 3 was guaranteed only by running the mutating builder
after every sandbox reset — a silent fix, not a failing check.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import ClassVar

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
DATA = TOOLKIT_ROOT / "examples" / "data"
EXAMPLES = TOOLKIT_ROOT / "examples"
BUILDER = TOOLKIT_ROOT / "scratch" / "build_example_data.py"

# The eight EXECUTED notebooks (the _no_output twins carry identical
# source and would only double-count).
EXECUTED_NOTEBOOKS = (
    "complete_ml_workflow.ipynb",
    "complete_tuning_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
)


def _notebook_data_references() -> dict[str, set[str]]:
    """Map notebook name -> the DATA_DIR-relative names its code loads.

    Only CODE cells count (markdown prose mentioning a file is not a
    load), and only ``DATA_DIR / "..."`` references — scratch paths under
    ``WORK`` are legitimately per-run temporaries, never shipped data.
    """
    import re

    pattern = re.compile(r'DATA_DIR\s*/\s*["\']([^"\']+)["\']')
    references: dict[str, set[str]] = {}
    for name in EXECUTED_NOTEBOOKS:
        path = EXAMPLES / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        found: set[str] = set()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            found.update(pattern.findall("".join(cell["source"])))
        references[name] = found
    return references


def _load_builder():
    """Import scratch/build_example_data.py as a module (main() is __main__-guarded)."""
    spec = importlib.util.spec_from_file_location("build_example_data", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNotebookDataReferences:
    """Leg 1 + 2: the notebook side of the round-trip."""

    REFERENCES: ClassVar[dict[str, set[str]]] = {}

    @classmethod
    def setup_class(cls) -> None:
        cls.REFERENCES = _notebook_data_references()

    def test_every_reference_resolves(self) -> None:
        """A notebook must never point at a stale data path."""
        stale = [
            (notebook, name)
            for notebook, names in self.REFERENCES.items()
            for name in sorted(names)
            if not (DATA / name).exists()
        ]
        assert not stale, f"notebook references that do not resolve: {stale}"

    def test_every_reference_is_a_top_level_entry(self) -> None:
        """References name top-level inventory entries (never deep paths).

        Deep reads go through scratch copies (``WORK / ...``) by design;
        a ``DATA_DIR / "sub/file"`` reference would bypass the inventory
        contract this module pins.
        """
        deep = [
            (notebook, name)
            for notebook, names in self.REFERENCES.items()
            for name in sorted(names)
            if "/" in name or "\\" in name
        ]
        assert not deep, f"references below the top level: {deep}"

    def test_notebooks_load_the_full_inventory(self) -> None:
        """No orphans: every shipped data entry is loaded by >= 1 notebook."""
        loaded: set[str] = set()
        for names in self.REFERENCES.values():
            loaded.update(names)
        shipped = {p.name for p in DATA.iterdir()}
        orphans = sorted(shipped - loaded)
        assert not orphans, (
            f"examples/data entries no executed notebook loads: {orphans} "
            "(ship them, load them, or remove them — the audit does not guess)"
        )


class TestBuilderRoundTrip:
    """Leg 3: the builder side — disk == generator output, byte for byte."""

    def test_check_mode_reports_no_drift(self) -> None:
        builder = _load_builder()
        builder.main(["--check"])  # raises SystemExit(1) on any drift
        assert builder.STATS["drift"] == []
        assert builder.STATS["written"] == 0  # --check never mutates
        owned_in_data = {p for p in builder.STATS["paths"] if p.parent != builder.TESTS_DATA_DIR}
        disk = {p for p in DATA.rglob("*") if p.is_file()}
        assert owned_in_data == disk, "owned set != disk set (orphan scan desynced)"

    def test_check_mode_flags_planted_drift(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One flipped byte must produce exactly one drift line and exit 1.

        Runs against a COPY under tmp_path — the real tree is never
        touched, and the missing-file branch is exercised by pointing
        --check at a fresh empty directory first.
        """
        builder = _load_builder()
        sandbox = tmp_path / "data"
        shutil.copytree(DATA, sandbox)

        # empty-dir leg: every owned file reads as missing
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(builder, "DATA_DIR", empty)
        with pytest.raises(SystemExit) as empty_excinfo:
            builder.main(["--check"])
        assert empty_excinfo.value.code == 1
        missing = [d for d in builder.STATS["drift"] if "is missing from" in d]
        assert missing, "empty data dir must report missing files"

        # byte-flip leg: exactly one drift entry
        monkeypatch.setattr(builder, "DATA_DIR", sandbox)
        victim = sandbox / "qsar_regression.csv"
        original = victim.read_bytes()
        victim.write_bytes(original + b"drift\n")
        with pytest.raises(SystemExit) as excinfo:
            builder.main(["--check"])
        assert excinfo.value.code == 1
        drift = builder.STATS["drift"]
        assert len(drift) == 1, f"expected exactly the planted drift, got {drift}"
        assert "qsar_regression.csv" in drift[0]
        assert "differs from" in drift[0]

        # and the audit never repaired the planted file
        assert victim.read_bytes() == original + b"drift\n"
