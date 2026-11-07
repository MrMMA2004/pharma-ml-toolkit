"""Tests for the feature-recalculation roundtrip (round 204).

The owner's catalog continuation ask: use the metadata the feature
factory stores to RECALCULATE the selected features.  Two layers:

* library -- :func:`pharmaml.chem.features.recalculate_selected_features`:
  the recorded columns in the recorded rank order, only the needed
  families computed, stale metadata failing loudly (fingerprint-size
  contradictions, vanished features, internally inconsistent ledgers);
* GUI -- the Chemistry tab's "Feature recalculation (from metadata)"
  card: the three adoption paths (the bundled example, a saved JSON
  file, the factory's in-session selection), the worker-thread run,
  the molecule-x-feature tree render, and the CSV/Excel save arms.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"
EXAMPLE_METADATA = TOOLKIT_ROOT / "examples" / "data" / "selection_metadata.json"


def _load_playground() -> Any:
    """Import playground.py by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("playground_recl_under_test", PLAYGROUND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys

    sys.modules["playground_recl_under_test"] = module
    return module


@pytest.fixture(scope="module")
def playground() -> Any:
    return _load_playground()


def _xvfb_available() -> bool:
    return shutil.which("Xvfb") is not None


XVFB_SKIP = pytest.mark.skipif(
    importlib.util.find_spec("tkinter") is None or not _xvfb_available(),
    reason="needs tkinter + Xvfb for a headless display",
)


def _selection_fixture(
    smiles: list[str], families: tuple[str, ...] = ("ecfp", "descriptors"), k: int = 6
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Compute + select + record: the roundtrip's factory half."""
    from pharmaml.chem.features import (
        compute_feature_frame,
        rank_features,
        select_top_features,
        selection_metadata,
    )

    frame, report = compute_feature_frame(smiles, families=families)
    scores = rank_features(frame, None, method="variance")
    selection = select_top_features(frame, scores.reindex(frame.columns), k)
    metadata = selection_metadata(selection, report, k=k)
    return frame, report, selection, metadata


# --------------------------------------------------------------------- #
# the library roundtrip
# --------------------------------------------------------------------- #
class TestRecalculateLibrary:
    def test_roundtrip_columns_in_recorded_rank_order(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        _f, _r, _s, metadata = _selection_fixture(["CCO", "CCC", "CCN"], k=6)
        frame, report = recalculate_selected_features(["CCO", "c1ccccc1"], metadata)
        assert list(frame.columns) == metadata["selected_features"]
        assert len(report.smiles) == 2 and report.dropped == []

    def test_recalculated_values_match_a_direct_compute(self) -> None:
        """The restriction is a pure projection -- no value drift."""
        from pharmaml.chem.features import compute_feature_frame, recalculate_selected_features

        _f, _r, _s, metadata = _selection_fixture(["CCO", "CCC", "CCN"], k=6)
        new_batch = ["CCO", "c1ccccc1", "CCN"]
        frame, _report = recalculate_selected_features(new_batch, metadata)
        direct, _direct_report = compute_feature_frame(new_batch, families=("ecfp", "descriptors"))
        for column in metadata["selected_features"]:
            assert frame[column].tolist() == direct[column].tolist()

    def test_only_the_needed_families_are_computed(self) -> None:
        """A descriptors-only selection never touches the fingerprint
        families -- the recalculation's efficiency promise."""
        from pharmaml.chem.features import recalculate_selected_features

        metadata = {
            "selected_features": ["desc_amw", "desc_tpsa", "desc_exactmw"],
            "family_counts": {"descriptors": 3},
            "fingerprint_sizes": {"ecfp": 2048, "descriptors": 43},
            "families_computed": {"descriptors": 43, "ecfp": 2048},
            "k_requested": 3,
            "method": "variance",
        }
        _frame, report = recalculate_selected_features(["CCO", "CCC"], metadata)
        assert set(report.families) == {"descriptors"}

    def test_dropped_molecules_ride_the_report(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        _f, _r, _s, metadata = _selection_fixture(["CCO", "CCC", "CCN"], k=4)
        frame, report = recalculate_selected_features(
            ["CCO", "not-a-molecule", "CCN"], metadata
        )
        assert report.dropped == [(1, "not-a-molecule")]
        assert list(frame.index) == [0, 2]  # original positions, gap kept
        assert len(report.smiles) == 2

    def test_progress_callback_fires(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        _f, _r, _s, metadata = _selection_fixture(["CCO", "CCC"], k=4)
        seen: list[tuple[int, int | None, str]] = []
        _frame, _report = recalculate_selected_features(
            ["CCO", "CCN"], metadata, progress=lambda d, t, n: seen.append((d, t, n))
        )
        assert seen and seen[-1][0] == 2  # every molecule completed

    def test_bare_string_smiles_rejected(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="not a single string"):
            recalculate_selected_features("CCO", {"selected_features": ["desc_amw"]})

    def test_metadata_must_be_a_mapping(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="must be a mapping"):
            recalculate_selected_features(["CCO"], "not a dict")

    def test_empty_selection_rejected(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="carries no selected_features"):
            recalculate_selected_features(["CCO"], {"selected_features": []})

    def test_missing_selection_key_rejected(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="carries no selected_features"):
            recalculate_selected_features(["CCO"], {"k_requested": 5})

    def test_duplicate_columns_rejected(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="duplicates"):
            recalculate_selected_features(
                ["CCO"], {"selected_features": ["desc_amw", "desc_amw"]}
            )

    def test_unknown_column_prefix_rejected(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="not a column this module can compute"):
            recalculate_selected_features(["CCO"], {"selected_features": ["SMILES"]})

    def test_fingerprint_size_contradiction_is_loud(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="says 1024 bits, but this build computes 2048"):
            recalculate_selected_features(
                ["CCO"],
                {"selected_features": ["ecfp_bit_17"], "fingerprint_sizes": {"ecfp": 1024}},
            )

    def test_families_computed_contradiction_is_loud(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="internally inconsistent"):
            recalculate_selected_features(
                ["CCO"],
                {"selected_features": ["ecfp_bit_17"], "families_computed": {"descriptors": 43}},
            )

    def test_vanished_feature_is_the_honest_stop(self) -> None:
        """A descriptor this build no longer offers cannot come back NaN."""
        from pharmaml.chem.features import recalculate_selected_features

        with pytest.raises(ValueError, match="no longer exist"):
            recalculate_selected_features(
                ["CCO"],
                {
                    "selected_features": ["desc_definitely_not_a_property"],
                    "families_computed": {"descriptors": 43},
                },
            )

    def test_exported_from_pharmaml_chem(self) -> None:
        import pharmaml.chem

        assert pharmaml.chem.recalculate_selected_features is not None


# --------------------------------------------------------------------- #
# the bundled example itself is honest
# --------------------------------------------------------------------- #
class TestBundledExample:
    def test_example_file_exists_and_roundtrips(self) -> None:
        from pharmaml.chem.features import recalculate_selected_features

        assert EXAMPLE_METADATA.is_file()
        metadata = json.loads(EXAMPLE_METADATA.read_text(encoding="utf-8"))
        selected = metadata["selected_features"]
        assert len(selected) == 24
        assert metadata["family_counts"] == {"descriptors": 22, "ecfp": 2}
        frame, report = recalculate_selected_features(["CCO", "c1ccccc1"], metadata)
        assert list(frame.columns) == selected
        assert len(report.smiles) == 2

    def test_example_mentions_its_own_smiles(self) -> None:
        metadata = json.loads(EXAMPLE_METADATA.read_text(encoding="utf-8"))
        smiles = metadata["example_smiles"]
        assert len(smiles) == 8 and "CC(=O)Oc1ccccc1C(=O)O" in smiles


# --------------------------------------------------------------------- #
# the GUI card (Xvfb)
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestRecalculationGui:
    # (round 209) the display fixture moved to tests/conftest.py as ONE
    # session-scoped Xvfb -- per-class fresh servers + a stale display
    # connection from a destroyed app = the next Tk event pump hits the
    # broken connection and Tk 9's X error handler exits the whole
    # process (the round-209 collapse forensics).

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        """Neutralise modal dialogs so failures assert instead of blocking."""
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 120.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the card, its description and its workflow grouping ---------- #

    def test_card_is_described_and_in_the_descriptors_workflow(
        self, app: Any, playground: Any
    ) -> None:
        titles = [title for title, _subtitle, _shell in playground._CARD_INDEX]
        assert "Feature recalculation (from metadata)" in titles
        subtitle = next(
            sub
            for t, sub, _s in playground._CARD_INDEX
            if t == "Feature recalculation (from metadata)"
        )
        assert "metadata" in subtitle and "NEW molecules" in subtitle

    def test_workflow_registry_carries_the_new_step(self, playground: Any) -> None:
        workflow = next(w for w in playground.WORKFLOWS if w["name"] == "Descriptors & features")
        step_cards = [card for _tab, card, _blurb in workflow["steps"]]
        assert step_cards == [
            "CSV \u2192 canonical SMILES (the standardization lab)",
            "SMILES inspector",
            "Feature factory",
            "Feature recalculation (from metadata)",
            "Library similarity scan",
        ]
        assert playground.workflow_for_card(
            "Chemistry", "Feature recalculation (from metadata)"
        ) == "Descriptors & features"

    def test_badge_names_the_step_number(self, app: Any, playground: Any) -> None:
        badge = playground._WORKFLOW_BADGES[
            ("Descriptors & features", "Feature recalculation (from metadata)")
        ]
        assert badge == "Descriptors & features \u00b7 step 4/5"

    # -- the three adoption paths ------------------------------------- #

    def test_example_adoption(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._recl_load_example()
        assert app._recl_metadata is not None
        assert len(app._recl_metadata["selected_features"]) == 24
        info = str(app._recl_meta_info.cget("text"))
        assert "24 selected features" in info and "the bundled example" in info

    def test_json_file_adoption(
        self, app: Any, playground: Any, tmp_path: Path, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        payload = {
            "selected_features": ["desc_amw", "desc_tpsa"],
            "family_counts": {"descriptors": 2},
            "k_requested": 2,
            "method": "variance",
        }
        target = tmp_path / "selection_metadata.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **kw: str(target)
        )
        app._recl_load_json()
        assert app._recl_metadata == payload

    def test_factory_relay_adoption(self, app: Any, quiet_dialogs: list[Any]) -> None:
        from pharmaml.chem.features import compute_feature_frame, rank_features, select_top_features

        frame, report = compute_feature_frame(["CCO", "CCC"], families=("descriptors",))
        scores = rank_features(frame, None, method="variance")
        selection = select_top_features(frame, scores.reindex(frame.columns), 3)
        app._feat_selection = selection
        app._feat_report = report
        app._feat_k = 3
        app._recl_adopt_factory()
        assert app._recl_metadata is not None
        assert app._recl_metadata["selected_features"] == selection.selected
        assert app._recl_source_note == "the factory's current selection"

    def test_relay_without_a_factory_selection_warns(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._feat_selection = None
        app._feat_report = None
        app._recl_adopt_factory()
        # the honest stop is the "No factory selection" info dialog
        infos = [call for name, call in quiet_dialogs if name == "showinfo"]
        assert infos and "factory" in str(infos[0])
        assert app._recl_metadata is None

    def test_non_metadata_file_warns(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._recl_adopt_metadata({"k_requested": 5}, "a bad file")
        assert app._recl_metadata is None
        warnings = [call for name, call in quiet_dialogs if name == "showwarning"]
        assert warnings and "selected_features" in str(warnings[0])

    # -- the run, the render, the save -------------------------------- #

    def test_run_renders_the_value_matrix(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._recl_load_example()
        app._recl_run()
        assert self._drain(app), "the recalculation job did not drain"
        assert app._recl_frame is not None
        assert app._recl_frame.shape == (3, 24)  # the 3 default pasted molecules
        assert list(app._recl_frame.columns) == app._recl_metadata["selected_features"]
        rows = app._recl_tree.get_children()
        assert len(rows) == 3
        first = app._recl_tree.item(rows[0], "values")
        assert first[1] == "CCO"  # the SMILES column
        assert app._recl_tree.heading("v1", "text") == "desc_amw"
        summary = str(app._recl_summary.cget("text"))
        assert "24 features recalculated for 3 molecules" in summary

    def test_run_without_metadata_warns(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._recl_run()
        warnings = [call for name, call in quiet_dialogs if name == "showwarning"]
        assert warnings and "metadata" in str(warnings[0]).lower()

    def test_stale_metadata_fails_loudly_in_the_gui(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._recl_adopt_metadata(
            {"selected_features": ["ecfp_bit_17"], "fingerprint_sizes": {"ecfp": 1024}},
            "a stale file",
        )
        app._recl_run()
        assert self._drain(app)
        errors = [call for name, call in quiet_dialogs if name == "showerror"]
        assert errors and "fingerprint_sizes" in str(errors[0])

    def test_save_writes_xlsx_and_reads_back(
        self, app: Any, playground: Any, tmp_path: Path, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        from pharmaml.utils.io import read_table

        app._recl_load_example()
        app._recl_run()
        assert self._drain(app)
        target = tmp_path / "recalculated.xlsx"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **kw: str(target)
        )
        app._recl_save()
        assert target.is_file()
        back = read_table(target)
        assert list(back.columns) == ["SMILES", *app._recl_metadata["selected_features"]]
        assert len(back) == 3
        assert back["SMILES"].tolist() == ["CCO", "c1ccccc1", "CC(=O)O"]

    def test_save_before_run_warns(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._recl_save()
        infos = [call for name, call in quiet_dialogs if name == "showinfo"]
        assert infos and "Recalculate features first" in str(infos[0])
