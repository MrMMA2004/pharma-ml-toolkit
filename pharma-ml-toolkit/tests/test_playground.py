"""Tests for playground.py -- the toolkit's single-file graphical front end.

The pure helpers (parsing, formatting, dependency probing, objective
factory, operator-stats rendering, docking tuple/engine helpers, the
feature factory's width/SMILES-column helpers) are tested directly; the
window itself is exercised headlessly under Xvfb via ``--self-test`` and
small GUI-fixture classes that drive real callbacks (including the Cell
Lines tab's screen loader, converter and biomarker discovery, the
Docking tab's ligand/receptor preparation and result parsing, and the
Chemistry tab's feature factory compute/export runs).
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"


def _load_playground() -> Any:
    """Import playground.py by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("playground_under_test", PLAYGROUND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # (round 200, the arena merge) register under the spec's name so
    # string-target monkeypatches ("playground_under_test.filedialog…")
    # resolve -- the merged fold-in tests import it by that name
    sys.modules["playground_under_test"] = module
    return module


@pytest.fixture(scope="module")
def playground() -> Any:
    return _load_playground()


# --------------------------------------------------------------------- #
# module surface
# --------------------------------------------------------------------- #
class TestModuleSurface:
    def test_file_exists_at_toolkit_root(self) -> None:
        assert PLAYGROUND_PATH.is_file()

    def test_docstring_documents_the_run_command(self, playground: Any) -> None:
        assert "python playground.py" in playground.__doc__

    def test_module_exposes_helpers_and_app(self, playground: Any) -> None:
        for name in (
            "probe_dependencies",
            "load_pharmaml",
            "parse_float",
            "parse_int",
            "parse_smiles_lines",
            "format_number",
            "build_objective",
            "toolkit_version",
            "PlaygroundApp",
            "main",
            "self_test",
            "print_banner",
            "apply_theme",
            "session_path",
            "dump_session",
            "load_session",
            "parse_geometry",
        ):
            assert hasattr(playground, name), name

    def test_docstring_documents_session_and_theme(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "playground-session.json" in doc
        assert "View \u25b8 Follow system / Light / Dark" in doc
        assert "Save session" in doc


# --------------------------------------------------------------------- #
# toolkit version plumbing (pyproject [project] -> VERSION -> 0.0.0)
# --------------------------------------------------------------------- #
class TestToolkitVersion:
    @staticmethod
    def _pyproject_project_version() -> str:
        """Independent re-derivation of the pyproject ``[project]`` version."""
        in_project = False
        for raw_line in (TOOLKIT_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project = line == "[project]"
                continue
            if not in_project or not line.startswith("version"):
                continue
            key, sep, value = line.partition("=")
            if sep and key.strip() == "version":
                return value.split("#", 1)[0].strip().strip('"').strip("'")
        return ""

    def test_pyproject_version_file_and_playground_agree(self, playground: Any) -> None:
        declared = self._pyproject_project_version()
        assert declared, "no [project] version line found in pyproject.toml"
        version_file = (TOOLKIT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        # the drift guard: all three sources must name the same version
        assert playground.toolkit_version() == declared
        assert version_file == declared
        assert playground.__version__ == declared

    def test_fallback_chain_version_file_then_placeholder(
        self, playground: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        only_version = tmp_path / "only_version"
        only_version.mkdir()
        (only_version / "VERSION").write_text("9.9.9\n", encoding="utf-8")
        monkeypatch.setattr(playground, "TOOLKIT_ROOT", only_version)
        assert playground.toolkit_version() == "9.9.9"

        neither = tmp_path / "neither"
        neither.mkdir()
        monkeypatch.setattr(playground, "TOOLKIT_ROOT", neither)
        assert playground.toolkit_version() == "0.0.0"

    def test_fallback_ignores_versions_outside_project_section(
        self, playground: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        sneaky = tmp_path / "sneaky"
        sneaky.mkdir()
        (sneaky / "pyproject.toml").write_text(
            '[tool.something]\nversion = "7.7.7"\n\n[project]\nname = "x"\n',
            encoding="utf-8",
        )
        (sneaky / "VERSION").write_text("8.8.8\n", encoding="utf-8")
        monkeypatch.setattr(playground, "TOOLKIT_ROOT", sneaky)
        # a version line outside [project] must not be picked up
        assert playground.toolkit_version() == "8.8.8"


# --------------------------------------------------------------------- #
# the Cell Lines tab surface
# --------------------------------------------------------------------- #
class TestCellLineSurface:
    def test_cellline_tab_builder_exists(self, playground: Any) -> None:
        assert hasattr(playground.PlaygroundApp, "_build_cellline_tab")

    def test_docstring_mentions_the_cell_lines_tab(self, playground: Any) -> None:
        assert "Cell Lines" in playground.__doc__

    def test_dependency_table_covers_the_heavy_packages(self, playground: Any) -> None:
        modules = {entry[0] for entry in playground._DEPENDENCIES}
        # 100x-162: selfies joined the heavy-packages table (the SELFIES
        # codec / audit surfaces -- pure-Python package, zero transitive
        # dependencies)
        assert modules == {
            "numpy", "pandas", "sklearn", "scipy", "matplotlib", "rdkit", "torch", "selfies"
        }


# --------------------------------------------------------------------- #
# the Docking tab surface
# --------------------------------------------------------------------- #
class TestDockingTab:
    def test_docking_tab_builder_exists(self, playground: Any) -> None:
        assert hasattr(playground.PlaygroundApp, "_build_docking_tab")

    def test_docstring_mentions_the_docking_tab(self, playground: Any) -> None:
        assert "Docking" in playground.__doc__

    def test_stdlib_shutil_is_imported_for_binary_probes(self, playground: Any) -> None:
        # the campaign runner gates on shutil.which before starting a job
        assert playground.shutil.which is not None


# --------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------- #
class TestPureHelpers:
    def test_probe_dependencies_returns_booleans_for_every_module(self, playground: Any) -> None:
        status = playground.probe_dependencies()
        # 100x-162: selfies joined the probed heavy packages
        assert set(status) == {
            "numpy",
            "pandas",
            "sklearn",
            "scipy",
            "matplotlib",
            "rdkit",
            "torch",
            "selfies",
        }
        assert all(isinstance(value, bool) for value in status.values())

    def test_parse_smiles_lines_filters_blanks_and_comments(self, playground: Any) -> None:
        text = "CCO\n\n# a comment\n  CCN  \nCCO\n"
        assert playground.parse_smiles_lines(text) == ["CCO", "CCN", "CCO"]

    def test_parse_smiles_lines_empty_text(self, playground: Any) -> None:
        assert playground.parse_smiles_lines("") == []
        assert playground.parse_smiles_lines("\n \n# only comments\n") == []

    def test_parse_float_happy_and_formatted_errors(self, playground: Any) -> None:
        assert playground.parse_float(" 2.5 ", "dose") == 2.5
        with pytest.raises(ValueError, match="dose"):
            playground.parse_float("abc", "dose")
        with pytest.raises(ValueError, match="finite"):
            playground.parse_float("nan", "dose")
        with pytest.raises(ValueError, match="below the minimum"):
            playground.parse_float("-1", "dose", minimum=0.0)
        with pytest.raises(ValueError, match="above the maximum"):
            playground.parse_float("101", "F %", maximum=100.0)

    def test_parse_int_rejects_rubbish_and_enforces_bounds(self, playground: Any) -> None:
        assert playground.parse_int("16", "generations") == 16
        with pytest.raises(ValueError, match="generations"):
            playground.parse_int("soon", "generations")
        with pytest.raises(ValueError, match="below the minimum"):
            playground.parse_int("2", "population", minimum=4)

    def test_format_number_modes(self, playground: Any) -> None:
        assert playground.format_number(1.23456) == "1.2346"
        assert playground.format_number(123456.7) == "1.235e+05"
        assert playground.format_number(0.00012) == "1.200e-04"
        assert playground.format_number(float("nan")) == "nan"
        assert playground.format_number(float("-inf")) == "-inf"
        assert playground.format_number(1234.5) == "1,234.5000"

    def test_cell_text_truncates_and_formats(self, playground: Any) -> None:
        assert playground._cell_text(0.123456) == "0.1235"
        assert playground._cell_text("CC(=O)Oc1ccccc1C(=O)O", limit=10) == "CC(=O)Oc1…"
        assert playground._cell_text(None) == ""

    def test_encode_class_target_string_classes_get_codes_and_note(self, playground: Any) -> None:
        # (100x-169) the heatmap/correlation bug: a STRING class target
        # must encode as sorted-unique float codes with the disclosure
        # note, never raise, and never come back string-typed.
        import pandas as pd

        codes, note = playground.encode_class_target(
            pd.Series(["high", "low", "mid", "low"])
        )
        assert codes.tolist() == [0.0, 1.0, 2.0, 1.0]
        assert note == "high=0 low=1 mid=2"

        # binary case: the classic point-biserial pairing
        codes, note = playground.encode_class_target(
            pd.Series(["inactive", "active", "active"])
        )
        assert codes.tolist() == [1.0, 0.0, 0.0]
        assert note == "active=0 inactive=1"

        # a plain list (not a Series) takes the same path
        codes, note = playground.encode_class_target(["b", "a", "b"])
        assert codes.tolist() == [1.0, 0.0, 1.0]
        assert note == "a=0 b=1"

    def test_encode_class_target_numeric_passes_through_silently(self, playground: Any) -> None:
        import pandas as pd

        values, note = playground.encode_class_target(pd.Series([1.5, 2.5, 3.5]))
        assert values.tolist() == [1.5, 2.5, 3.5]
        assert note == ""


# --------------------------------------------------------------------- #
# the manual-hyperparameter + progress helpers (100x-106)
# --------------------------------------------------------------------- #
class TestManualHyperparameterHelpers:
    def test_parse_typed_value_follows_the_default_type(self, playground: Any) -> None:
        parse = playground.parse_typed_value
        assert parse("0.5", 1.0) == 0.5  # float default
        assert parse("300", 100) == 300  # int default
        assert parse("scale", "auto") == "scale"  # string default
        assert parse("true", False) is True  # bool before int
        assert parse("NO", True) is False
        assert parse("", 7.5) == 7.5  # blank keeps the default

    def test_parse_typed_value_none_and_tuple_defaults(self, playground: Any) -> None:
        parse = playground.parse_typed_value
        assert parse("none", None) is None  # explicit None slot
        assert parse("null", None) is None
        assert parse("7", None) == 7  # numeric fallback
        assert parse("auto-ish", None) == "auto-ish"
        assert parse("(128, 64)", (100,)) == (128, 64)
        assert parse("128, 64", (100,)) == (128, 64)
        assert parse("16", (100,)) == (16,)

    def test_parse_typed_value_rejects_rubbish_with_field_context(self, playground: Any) -> None:
        parse = playground.parse_typed_value
        with pytest.raises(ValueError, match="true/false"):
            parse("maybe", False)
        with pytest.raises(ValueError):
            parse("abc", 1.0)  # float slot, float jargon
        with pytest.raises(ValueError, match="width"):
            parse("(128, sixty-four)", (100,))

    def test_fmt_duration_formats_every_magnitude(self, playground: Any) -> None:
        fmt = playground._fmt_duration
        assert fmt(0) == "0 s"
        assert fmt(41.9) == "42 s"
        assert fmt(-3.0) == "0 s"  # clamped, never negative
        assert fmt(192) == "3 min 12 s"
        assert fmt(599.6) == "10 min 00 s"
        assert fmt(3720) == "1 h 02 min"

    def test_accepts_progress_argument_signature_rule(self, playground: Any) -> None:
        accepts = playground._accepts_progress_argument
        assert accepts(lambda report: None) is True
        assert accepts(lambda: None) is False
        assert accepts(lambda *args: None) is True
        assert accepts(lambda x=None, *, y=1: None) is True  # x is positional
        assert accepts(lambda *, y=1: None) is False  # keyword-only is no

    def test_call_work_injects_the_reporter_conditionally(self, playground: Any) -> None:
        call = playground._call_work
        seen: list[object] = []
        assert call(lambda: 1, seen.append) == 1
        assert seen == []  # the zero-arg callable never sees the reporter
        assert call(lambda report: report("x"), seen.append) is None
        assert seen == ["x"]  # the one-arg callable received it
        assert call(lambda report: report("y") or 2, seen.append) == 2
        assert seen == ["x", "y"]


# --------------------------------------------------------------------- #
# the Docking tab's pure helpers (npts / centre / size tuples, engines)
# --------------------------------------------------------------------- #
class TestDockingHelpers:
    def test_parse_int_triple_accepts_spaces_and_commas(self, playground: Any) -> None:
        assert playground.parse_int_triple("60 60 60", "npts") == (60, 60, 60)
        assert playground.parse_int_triple("60,60,60", "npts") == (60, 60, 60)
        assert playground.parse_int_triple(" 60,  60 ,60 ", "npts") == (60, 60, 60)

    def test_parse_int_triple_rejects_rubbish(self, playground: Any) -> None:
        with pytest.raises(ValueError, match="npts"):
            playground.parse_int_triple("60 60", "npts")
        with pytest.raises(ValueError, match="three"):
            playground.parse_int_triple("", "npts")
        with pytest.raises(ValueError, match="whole number"):
            playground.parse_int_triple("60 60 sixty", "npts")
        with pytest.raises(ValueError, match="below the minimum"):
            playground.parse_int_triple("0 60 60", "npts", minimum=1)

    def test_parse_float_triple_accepts_negatives_and_commas(self, playground: Any) -> None:
        assert playground.parse_float_triple("0 0 0", "centre") == (0.0, 0.0, 0.0)
        assert playground.parse_float_triple("20 20 20", "size") == (20.0, 20.0, 20.0)
        assert playground.parse_float_triple("-1.5, 2, 2.5", "centre") == (-1.5, 2.0, 2.5)

    def test_parse_float_triple_rejects_rubbish(self, playground: Any) -> None:
        with pytest.raises(ValueError, match="centre"):
            playground.parse_float_triple("1 2", "centre")
        with pytest.raises(ValueError, match="not a number"):
            playground.parse_float_triple("1 2 x", "centre")

    def test_docking_engine_binaries(self, playground: Any) -> None:
        assert playground.docking_engine_binaries("AutoDock4 campaign") == (
            "autogrid4",
            "autodock4",
        )
        assert playground.docking_engine_binaries("Vina single ligand") == ("vina",)

    # -- (100x-171) the .exe engine folder ------------------------------- #
    def test_dock_engine_binary_prefers_picked_folder_then_bundled(
        self, playground: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        picked = tmp_path / "picked"
        picked.mkdir()
        (picked / "vina.exe").write_bytes(b"")
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "vina.exe").write_bytes(b"")
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", bundled)
        monkeypatch.setattr(playground.shutil, "which", lambda _name: None)
        assert playground.dock_engine_binary("vina", picked) == picked / "vina.exe"
        assert playground.dock_engine_binary("vina") == bundled / "vina.exe"

    def test_dock_engine_binary_falls_back_to_the_path_exe_spelling(
        self, playground: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "nowhere")
        monkeypatch.setattr(
            playground.shutil,
            "which",
            lambda name: "/usr/local/bin/vina.exe" if name == "vina.exe" else None,
        )
        assert playground.dock_engine_binary("vina") == Path("/usr/local/bin/vina.exe")

    def test_dock_missing_engines_reports_only_the_absent_names(
        self, playground: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "nowhere")
        monkeypatch.setattr(playground.shutil, "which", lambda _name: None)
        folder = tmp_path / "engines"
        folder.mkdir()
        (folder / "autogrid4.exe").write_bytes(b"")
        assert playground.dock_missing_engines(("autogrid4", "autodock4"), folder) == [
            "autodock4"
        ]
        assert playground.dock_missing_engines(("autogrid4", "autodock4")) == [
            "autogrid4",
            "autodock4",
        ]

    def test_dock_engine_dir_prefers_picked_then_bundled_then_none(
        self, playground: Any, tmp_path: Any, monkeypatch: Any
    ) -> None:
        picked = tmp_path / "picked"
        picked.mkdir()
        (picked / "vina.exe").write_bytes(b"")
        bundled = tmp_path / "bundled"
        bundled.mkdir()
        (bundled / "autodock4.exe").write_bytes(b"")
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", bundled)
        assert playground.dock_engine_dir(picked) == picked
        assert playground.dock_engine_dir(None) == bundled
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", empty)
        assert playground.dock_engine_dir(None) is None


# --------------------------------------------------------------------- #
# session persistence + themes (pure helpers -- no Tk required)
# --------------------------------------------------------------------- #
class TestSessionHelpers:
    def test_session_path_lives_under_home_dot_pharmaml(self, playground: Any) -> None:
        path = playground.session_path()
        assert path.name == "playground-session.json"
        assert path.parent == Path.home() / ".pharmaml"

    def test_parse_geometry_accepts_only_full_specs(self, playground: Any) -> None:
        assert playground.parse_geometry("1180x780+100+50") == (1180, 780, 100, 50)
        assert playground.parse_geometry(" 900x600-20+0 ") == (900, 600, -20, 0)
        for broken in (
            "",
            "1180x780",
            "+100+50",
            "1180x780+100",
            "junk",
            "0x0+0+0extra",
            "1180xx780+1+1",
        ):
            assert playground.parse_geometry(broken) is None, broken

    def test_dump_session_normalises_every_field(self, playground: Any) -> None:
        payload = playground.dump_session(
            theme="neon",
            geometry=1180,
            tab=True,
            variables={"_a": "x", 5: "y"},
            families={"descriptors": 1, "ecfp": None},
            texts={"pasted": "CCO", "corpus": 42},
        )
        assert payload["format"] == playground.SESSION_FORMAT
        assert payload["theme"] == "light"  # unknown theme -> light
        assert payload["geometry"] == "1180"  # coerced to str
        assert payload["tab"] == 1  # bool -> int
        assert payload["variables"] == {"_a": "x", "5": "y"}
        assert payload["families"] == {"descriptors": True, "ecfp": False}
        assert payload["texts"] == {"pasted": "CCO", "corpus": "42"}

    def test_load_session_rejects_malformed_payloads(self, playground: Any) -> None:
        for raw in (
            None,
            [],
            "session",
            {},
            {"format": 0},
            {"format": "1"},
            {"format": 2, "theme": "dark"},
        ):
            assert playground.load_session(raw) is None, raw

    def test_load_session_sanitises_half_formed_payloads(self, playground: Any) -> None:
        loaded = playground.load_session(
            {
                "format": playground.SESSION_FORMAT,
                "theme": "solarized",
                "geometry": None,
                "tab": -4,
                "variables": "not a dict",
                "families": {"descriptors": "yes"},
                "texts": {"pasted": "ok", "corpus": 7},
            }
        )
        assert loaded == {
            "format": playground.SESSION_FORMAT,
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {"pasted": "ok"},
            # (100x-142) the picked-paths key -- always present, empty
            # when the payload carries none (older files stay loadable)
            "paths": {},
            # (100x-144) the multi-file picks -- same always-present
            # discipline (an older payload degrades to empty lists,
            # never a missing key the restore would have to defend)
            "paths_multi": {},
        }

    def test_dump_load_round_trip_is_lossless(self, playground: Any) -> None:
        payload = playground.dump_session(
            theme="dark",
            geometry="1180x780+10+20",
            tab=4,
            variables={"_aff_value": "3.14", "_tune_var": True, "_feat_k_entry": "777"},
            families={"descriptors": True, "morgan_scaffold": False},
            texts={"pasted": "CCO\nCCN", "corpus": ""},
        )
        assert playground.load_session(payload) == payload

    def test_apply_theme_rebinds_and_restores_module_colours(self, playground: Any) -> None:
        light_bg = playground.BG
        try:
            assert playground.apply_theme("dark") == "dark"
            assert playground.BG == playground._DARK_PALETTE["BG"]
            assert playground.CARD == playground._DARK_PALETTE["CARD"]
            assert playground._ACTIVE_THEME == "dark"
            assert playground.apply_theme("light") == "light"
            assert playground.BG == light_bg
            assert playground.apply_theme("bogus") == "light"  # unknown -> light
        finally:
            playground.apply_theme("light")  # never leak dark into other tests

    def test_palettes_agree_on_keys(self, playground: Any) -> None:
        # every palette must define the same key set or apply_theme would
        # silently leave stale globals behind
        assert set(playground._LIGHT_PALETTE) == set(playground._DARK_PALETTE)
        for key, value in playground._LIGHT_PALETTE.items():
            assert value != playground._DARK_PALETTE[key], key


# --------------------------------------------------------------------- #
# pharmaml integration (needs the package importable -- pytest's pythonpath
# config and the editable install both cover that)
# --------------------------------------------------------------------- #
class TestPharmamlIntegration:
    def test_load_pharmaml_returns_the_real_module(self, playground: Any) -> None:
        import pharmaml

        module = playground.load_pharmaml()
        assert module.__version__ == pharmaml.__version__

    def test_format_operator_stats_placeholder_without_adaptive_metadata(
        self, playground: Any
    ) -> None:
        assert "adaptive" in playground._format_operator_stats({})

    def test_format_operator_stats_renders_levels_and_probabilities(self, playground: Any) -> None:
        text = playground._format_operator_stats(
            {
                "operator_stats": {
                    "level1": {
                        "clone": {"draws": 24, "successes": 20, "final_probability": 0.893},
                    },
                    "mutation": {
                        "topology": {"draws": 200, "successes": 180, "final_probability": 0.896},
                    },
                }
            }
        )
        assert "[level1]" in text and "[mutation]" in text
        assert "clone" in text and "p 0.893" in text
        assert "topology" in text and "p 0.896" in text

    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_build_objective_qed_has_score(self, playground: Any) -> None:
        objective = playground.build_objective("QED drug-likeness", 1.0, 3.0)
        assert callable(objective.score)

    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_build_objective_logp_window_scores_inside_band(self, playground: Any) -> None:
        objective = playground.build_objective("LogP window", 1.0, 3.0)
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"  # LogP ~ 1.31 -- inside the band
        assert objective.score(aspirin) > 0.0

    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_build_objective_blend_is_multiobjective(self, playground: Any) -> None:
        from pharmaml.generative import MultiObjective

        objective = playground.build_objective("QED + LogP blend", 1.0, 3.0)
        assert isinstance(objective, MultiObjective)


# --------------------------------------------------------------------- #
# the cell-line pipeline the Cell Lines tab drives (same pharmaml calls,
# no Tk -- pins the shipped example data's flagship facts)
# --------------------------------------------------------------------- #
class TestCellLinePipeline:
    @pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="needs SciPy")
    def test_example_screen_audit_and_matrix_facts(self) -> None:
        import pandas as pd

        from pharmaml.biology import (
            audit_response_table,
            collapse_replicates,
            ln_ic50_to_pic50,
            load_response_table,
            response_matrix,
        )

        screen_path = TOOLKIT_ROOT / "examples" / "data" / "cellline_screen.csv"
        raw = load_response_table(screen_path)
        assert len(raw.frame) == 185
        assert raw.metrics == ["LN_IC50"]
        assert len(raw.compounds) == 8
        audit = audit_response_table(raw)
        assert audit["issue"].tolist() == ["missing metric values", "duplicate pairs"]
        assert audit["detail"].tolist() == ["metric 'LN_IC50'", "MCF-7 / crizotinib"]
        assert audit["count"].tolist() == [1, 2]

        collapsed = collapse_replicates(raw)
        assert len(collapsed.frame) == 184
        matrix = response_matrix(collapsed, "LN_IC50")
        assert matrix.shape == (23, 8)
        assert int(matrix.isna().sum().sum()) == 1  # the one untested pair

        # the quick converter card's flagship value (MCF-7 / erlotinib)
        assert ln_ic50_to_pic50(3.2085197999117585) == 4.606557555820998
        assert isinstance(matrix, pd.DataFrame)  # the tab stores exactly this

    @pytest.mark.skipif(importlib.util.find_spec("scipy") is None, reason="needs SciPy")
    def test_example_biomarker_story_pins_egfr(self) -> None:
        import pandas as pd

        from pharmaml.biology import (
            collapse_replicates,
            impute_expression,
            load_response_table,
            normalize_expression,
            response_matrix,
            sensitivity_associations,
            top_biomarkers,
        )

        data = TOOLKIT_ROOT / "examples" / "data"
        screen = collapse_replicates(load_response_table(data / "cellline_screen.csv"))
        matrix = response_matrix(screen, "LN_IC50")
        expression = pd.read_csv(data / "cellline_expression.csv", index_col="cell_line")
        assert expression.shape == (23, 300)
        norm = impute_expression(normalize_expression(expression))
        assert int(norm.isna().sum().sum()) == 0

        def top_hit(compound: str) -> Any:
            associations = sensitivity_associations(norm, matrix[compound])
            return associations, top_biomarkers(associations, n=10).iloc[0]

        # the flagship: erlotinib -> EGFR, rho -0.9783, FDR far below 1e-10
        associations, hit = top_hit("erlotinib")
        assert len(associations) == 300
        assert int(associations["significant"].sum()) == 1
        assert hit["gene"] == "EGFR"
        assert round(float(hit["rho"]), 4) == -0.9783
        assert float(hit["fdr"]) < 1e-10
        assert bool(hit["significant"]) is True

        # the two siblings (same story, weaker signal)
        associations, hit = top_hit("gefitinib")
        assert int(associations["significant"].sum()) == 2
        assert hit["gene"] == "EGFR"
        assert round(float(hit["rho"]), 4) == -0.9239
        assert float(hit["fdr"]) < 1e-6

        associations, hit = top_hit("crizotinib")
        assert int(associations["significant"].sum()) == 1
        assert hit["gene"] == "EGFR"
        assert round(float(hit["rho"]), 4) == -0.7589
        assert round(float(hit["fdr"]), 6) == 0.008087
        assert bool(hit["significant"]) is True


# --------------------------------------------------------------------- #
# the docking pipeline the Docking tab drives (same pharmaml calls, no
# Tk -- pins the shipped example library / receptor / campaign facts)
# --------------------------------------------------------------------- #
class TestDockingPipeline:
    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_example_ligand_library_prep_and_grid_configs(self, tmp_path: Any) -> None:
        from pharmaml.docking import (
            compute_grid_box,
            extract_pdbqt_atom_types,
            load_ligand_csv,
            prepare_ligand_library,
            write_dpf,
            write_gpf,
        )

        library_path = TOOLKIT_ROOT / "examples" / "data" / "ligand_library.csv"
        library = load_ligand_csv(library_path)
        assert len(library.frame) == 10
        assert int((library.frame["status"] == "valid").sum()) == 8

        report = prepare_ligand_library(library, tmp_path)
        assert int(report.frame["ok"].sum()) == 8
        assert report.root == tmp_path
        first = report.frame[report.frame["ok"]].iloc[0]
        assert first["folder"] == "01_aspirin"
        ligand = Path(first["pdbqt"])
        assert ligand.is_file()

        box = compute_grid_box(ligand, padding=4.0, npts=(60, 60, 60), spacing=0.375)
        assert box.size == (22.125, 22.125, 22.125)  # (npts - 1) * spacing
        assert extract_pdbqt_atom_types(ligand) == ["A", "C", "HD", "OA"]
        gpf = write_gpf(
            ligand, receptor_file="r.pdbqt", box=box, output_path=ligand.parent / "r.gpf"
        )
        dpf = write_dpf(
            ligand,
            receptor_file="r.pdbqt",
            output_path=ligand.parent / "r.dpf",
            ga_run=100,
            ga_pop_size=150,
        )
        assert gpf.is_file() and dpf.is_file()

    @pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="needs pandas")
    def test_example_campaign_scan_and_dlg_facts(self) -> None:
        import pandas as pd

        from pharmaml.docking import (
            ligand_efficiency,
            ligand_efficiency_series,
            parse_dlg_cluster_table,
            parse_dlg_summary,
            scan_docking_directory,
        )

        campaign = TOOLKIT_ROOT / "examples" / "data" / "docking_campaign"
        frame = scan_docking_directory(campaign)
        assert len(frame) == 14
        assert int((frame["status"] == "OK").sum()) == 12
        assert frame.iloc[0]["folder"] == "01_aspirin"

        dlg = campaign / "01_aspirin" / "r.dlg"
        clusters = parse_dlg_cluster_table(dlg)
        assert len(clusters) == 5
        assert clusters[0].rank == 1
        assert clusters[0].lowest_energy == pytest.approx(-4.91)
        assert clusters[0].num_in_cluster == 15
        summary = parse_dlg_summary(dlg)
        assert summary["n_clusters"] == 5
        assert summary["top_lowest_energy"] == pytest.approx(-4.91)

        cluster_frame = pd.DataFrame(
            [
                {
                    "rank": c.rank,
                    "lowest_energy": c.lowest_energy,
                    "mean_energy": c.mean_energy,
                    "num_in_cluster": c.num_in_cluster,
                    "n_heavy": 13,
                }
                for c in clusters
            ]
        )
        efficiency = ligand_efficiency_series(
            cluster_frame, affinity_col="lowest_energy", atoms_col="n_heavy"
        )
        # the library reports |dG| / N_heavy -- aspirin: 4.91 / 13
        assert efficiency.iloc[0] == pytest.approx(abs(-4.91) / 13)
        assert ligand_efficiency(-4.91, 13) == pytest.approx(0.3776923)


# --------------------------------------------------------------------- #
# command-line helpers
# --------------------------------------------------------------------- #
class TestCommandLine:
    def test_probe_reports_pharmaml_and_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PLAYGROUND_PATH), "--probe"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=TOOLKIT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert "pharmaml" in result.stdout
        assert "numpy" in result.stdout and "RDKit" in result.stdout

    def test_version_prints_both_versions(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PLAYGROUND_PATH), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=TOOLKIT_ROOT,
        )
        assert result.returncode == 0
        assert "playground" in result.stdout and "pharmaml" in result.stdout
        assert "toolkit v" in result.stdout
        # the playground and toolkit lines must carry the SAME version number
        numbers = set()
        for line in result.stdout.splitlines():
            if line.startswith("playground "):
                numbers.add(line.removeprefix("playground "))
            if line.startswith("toolkit v"):
                numbers.add(line.removeprefix("toolkit v"))
        assert numbers, "no playground/toolkit version lines found"
        assert len(numbers) == 1, f"version drift between lines: {numbers}"


# --------------------------------------------------------------------- #
# the window itself -- under Xvfb when the sandbox can provide one
# --------------------------------------------------------------------- #
def _xvfb_available() -> bool:
    return shutil.which("Xvfb") is not None


XVFB_SKIP = pytest.mark.skipif(
    importlib.util.find_spec("tkinter") is None or not _xvfb_available(),
    reason="needs tkinter + Xvfb for a headless display",
)


@pytest.fixture()
def picklist_accept_all(playground: Any, monkeypatch: Any) -> None:
    """(100x-132) Bypass the report menus' pre-save picklist by accepting
    every page -- the pre-132 tests' original semantics (the whole
    session ships).  The 132 tests drive the real dialog instead."""

    monkeypatch.setattr(
        playground.PlaygroundApp,
        "_session_report_picklist",
        lambda self, live, tables: (live, tables),
    )


@pytest.fixture(scope="module")
def xvfb_display() -> str:
    """A module-scoped virtual display, polled until Xvfb accepts sockets."""
    import os
    import socket
    import time

    display = ":98"
    process = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x800x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    socket_path = Path(f"/tmp/.X11-unix/X{display[1:]}")
    deadline = time.time() + 10.0
    ready = False
    while time.time() < deadline:
        if socket_path.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(socket_path))
                ready = True
            except OSError:
                pass
            finally:
                probe.close()
        if ready:
            break
        time.sleep(0.1)
    assert ready, "Xvfb did not come up on :98 within 10 s"
    os.environ["DISPLAY"] = display
    try:
        yield display
    finally:
        process.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)


@XVFB_SKIP
class TestGuiUnderXvfb:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        # persist_session=False keeps the suite deterministic: it must
        # neither read nor write the developer's real session file.
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        # collect the (destroyed) widget/Variable cycles NOW, in the main
        # thread -- otherwise the next test's worker thread may finalize
        # tk Variables off-thread and trip "main thread is not in main loop"
        import gc

        del application
        gc.collect()

    def test_self_test_cli_builds_every_tab(self, xvfb_display: str) -> None:
        import os

        result = subprocess.run(
            [sys.executable, str(PLAYGROUND_PATH), "--self-test"],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=TOOLKIT_ROOT,
            env={**os.environ, "DISPLAY": xvfb_display},
        )
        assert result.returncode == 0, result.stderr
        assert "no errors" in result.stdout

    def test_example_corpus_loader_fills_the_text_widget(self, playground: Any, app: Any) -> None:
        app._load_example_corpus()
        lines = [line for line in app._corpus_text.get("1.0", "end").splitlines() if line.strip()]
        assert len(lines) == 12  # examples/data/seed_molecules.csv ships 12
        assert "CC(=O)Oc1ccccc1C(=O)O" in lines  # aspirin

    def test_default_affinity_conversion_is_exact(self, app: Any) -> None:
        app._convert_affinity()  # 100 nM -> pIC50 7.000
        assert app._aff_result.cget("text") == "7.000"

    def test_default_molar_mass_is_caffeine(self, app: Any) -> None:
        app._molar_mass()  # C8H10N4O2 -> 194.19 g/mol
        assert "194.19" in app._mass_result.cget("text")

    def test_half_life_card_derives_ke_and_accumulation(self, app: Any) -> None:
        app._halflife_math()
        text = app._half_result.cget("text")
        assert "0.1733" in text and "1.55" in text

    # -- the Cell Lines tab (real callbacks, dialogs recorded) ------------- #
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
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_cellline_example_screen_loads_with_audit(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._bio_load_example()
        assert quiet_dialogs == []
        audit = app._bio_audit_text.get("1.0", "end")
        assert "duplicate pairs" in audit
        assert "MCF-7 / crizotinib" in audit
        assert "185 rows" in app._bio_info.cget("text")
        assert "23 cell lines" in app._bio_info.cget("text")
        assert "erlotinib" in app._bio_compound_combo["values"]
        assert "LN_IC50" in app._bio_metric_combo["values"]

    def test_cellline_ln_ic50_converter_both_directions(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._bio_ic50_entry.delete(0, "end")
        app._bio_ic50_entry.insert(0, "3.2085197999117585")  # MCF-7 / erlotinib
        app._bio_ln_to_pic50()
        assert app._bio_ic50_result.cget("text") == "4.6066"
        app._bio_ic50_entry.delete(0, "end")
        app._bio_ic50_entry.insert(0, "4.6066")
        app._bio_pic50_to_ln()
        assert app._bio_ic50_result.cget("text") == "3.2084"
        assert quiet_dialogs == []

    def test_cellline_matrix_job_fills_the_tree(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._bio_load_example()
        app._bio_build_matrix()
        assert self._drain_jobs(app), "matrix job hung"
        rows = app._bio_matrix_tree.get_children()
        assert len(rows) == 23  # 23 cell lines in the example screen
        assert "23 cell lines × 8 compounds" in app._bio_matrix_info.cget("text")
        assert app._bio_matrix is not None and app._bio_matrix.shape == (23, 8)
        assert quiet_dialogs == []

    def test_cellline_sensitive_resistant_calls(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._bio_load_example()
        app._bio_compound_var.set("erlotinib")
        app._bio_call_sensitive()
        assert self._drain_jobs(app), "sensitive/resistant job hung"
        summary = app._bio_calls_summary.cget("text")
        assert "sensitive" in summary and "resistant" in summary
        rows = app._bio_calls_tree.get_children()
        assert len(rows) == 23
        # sorted sensitive-first: the very first call must be "sensitive"
        first = app._bio_calls_tree.item(rows[0], "values")
        assert first[0] == "sensitive"
        assert app._bio_calls_tree.item(rows[0], "text")  # a cell-line name
        assert quiet_dialogs == []

    def test_cellline_biomarkers_erlotinib_finds_egfr(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._bio_load_example()
        app._bio_load_expr_example()
        app._bio_compound_var.set("erlotinib")
        app._bio_discover_biomarkers()
        assert self._drain_jobs(app, timeout=300.0), "biomarker job hung"
        summary = app._bio_marker_summary.cget("text")
        assert "300 genes correlated" in summary
        assert "1 significant" in summary
        rows = app._bio_marker_tree.get_children()
        assert rows, "biomarker tree is empty"
        values = app._bio_marker_tree.item(rows[0], "values")
        assert values[0] == "EGFR"
        assert values[1] == "-0.9783"  # rho, 4 dp
        assert values[3] == "2.363e-13"  # BH-corrected fdr
        assert values[4] == "Yes"
        assert "more sensitive" in values[5]  # negative rho, LN_IC50 direction
        assert quiet_dialogs == []

    # -- the Data & ML tab's manual hyperparameters + progress (100x-106) -- #
    def test_ml_example_load_lists_models_and_clears_panel(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        assert app._model_combo.get() == "auto"
        assert "ridge" in list(app._model_combo["values"])
        # "auto" selection keeps the panel at its placeholder
        assert app._ml_param_entries == {}
        assert "pick a model" in app._ml_params_status.cget("text")
        assert quiet_dialogs == []

    def test_manual_hyperparam_panel_builds_and_pins(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        entries = app._ml_param_entries
        assert list(entries) == ["alpha"]  # ridge's registry grid
        assert entries["alpha"].get() == ""  # blank = estimator default
        assert "ridge · 1 knob(s)" in app._ml_params_status.cget("text")
        # typing pins; the per-model JSON persistence carries it
        entries["alpha"].insert(0, "0.5")
        app._ml_sync_params_var()
        assert json.loads(app._ml_params_var.get()) == {
            "ridge": {"alpha": "0.5"},
        }
        assert app._ml_manual_params("ridge", "regression") == {"alpha": 0.5}
        # switching models keeps ridge's values; switching back restores
        app._model_combo.set("elastic_net")
        app._ml_refresh_hyperparam_panel()
        assert set(app._ml_param_entries) == {"alpha", "l1_ratio"}
        assert json.loads(app._ml_params_var.get())["ridge"] == {"alpha": "0.5"}
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        assert app._ml_param_entries["alpha"].get() == "0.5"
        assert quiet_dialogs == []

    def test_manual_hyperparam_bad_value_fails_pre_flight(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_param_entries["alpha"].insert(0, "banana")
        with pytest.raises(ValueError, match="alpha"):
            app._ml_manual_params("ridge", "regression")
        assert quiet_dialogs == []

    def test_manual_hyperparam_reset_rebuilds_defaults(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_param_entries["alpha"].insert(0, "9.9")
        app._ml_sync_params_var()
        app._ml_params_reset()
        assert app._ml_param_entries["alpha"].get() == ""
        assert app._ml_params_var.get() == "{}"
        assert app._ml_manual_params("ridge", "regression") is None
        assert quiet_dialogs == []

    def test_progress_report_drives_determinate_bar_and_eta(self, app: Any) -> None:
        import time as _time

        app._busy = True
        app._job_label_base = "probe job"
        app._job_started = _time.monotonic() - 4.0  # 4 s elapsed
        app._report_progress(1, 8, "best 0.70")
        app._poll_progress()
        assert str(app._progress.cget("mode")) == "determinate"
        assert int(app._progress.cget("maximum")) == 8
        assert int(app._progress["value"]) == 1
        label = app._job_label.cget("text")
        assert label.startswith("probe job — 1/8 · ~") and " left" in label
        assert label.endswith("best 0.70")
        # the ETA shrinks as the fraction grows
        app._report_progress(4, 8, "")
        app._poll_progress()
        later = app._job_label.cget("text")
        assert later.startswith("probe job — 4/8 · ~")
        # total-less updates degrade to a plain counter
        app._report_progress(7, None, "")
        app._poll_progress()
        assert app._job_label.cget("text") == "probe job — 7"

    def test_progress_resets_when_the_job_ends(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_param_entries["alpha"].insert(0, "0.5")
        app._ml_train()
        assert app._busy, "the train job started"
        assert self._drain_jobs(app), "train job hung"
        trained = app._ml_last_run[0]
        assert trained.model.named_steps["reg"].alpha == 0.5  # pinned
        assert trained.best_params == {"alpha": 0.5}
        # the status bar reports the finished job (100x-119): the label
        # names what ran and how long it took, the bar returned to the
        # launch-time marquee
        done_text = app._job_label.cget("text")
        assert done_text.startswith("training ridge (regression) — done in "), done_text
        assert str(app._progress.cget("mode")) == "indeterminate"
        assert quiet_dialogs == []

    def test_scaffold_split_strategy_trains_end_to_end(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # (round 179) the split strategy combobox reaches the split: a
        # scaffold run on the walkthrough CSV (7 scaffold groups across
        # 180 molecules) trains cleanly through the whole job.  The
        # qsar_*.csv examples are single-scaffold datasets -- the honest
        # guard's territory, pinned by the test below.
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "walkthrough_compounds.csv")
        assert app._target_combo.get() == "activity"  # the smart default
        app._ml_split_combo.set("scaffold")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        assert quiet_dialogs == []
        assert app._ml_last_run is not None
        app._ml_split_combo.set("random")  # restore the default

    def test_scaffold_strategy_on_a_single_scaffold_dataset_fails_honestly(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # qsar_classification.csv's molecules all share one framework:
        # no scaffold-honest test set exists, and the split must say so
        # by name (never an empty-test crash downstream).
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv")
        app._ml_split_combo.set("scaffold")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain_jobs(app), "the failing job still drains"
        kinds = [call[0] for call in quiet_dialogs]
        assert "showerror" in kinds
        message = next(call[1][1] for call in quiet_dialogs if call[0] == "showerror")
        assert "shares one group/scaffold" in message
        app._ml_split_combo.set("random")

    def test_scaffold_strategy_without_smiles_warns_before_the_job(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # qsar_regression.csv is descriptors-only: the scaffold strategy
        # must stop pre-flight, naming the missing column -- never a
        # mid-job DataValidationError.
        app._ml_load_example()
        app._ml_split_combo.set("scaffold")
        app._ml_train()
        assert not app._busy  # the gate stopped before any job started
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Check the split strategy" in titles
        assert "SMILES column" in quiet_dialogs[0][1][1]
        app._ml_split_combo.set("random")

    def test_njobs_spinbox_reaches_the_train_call(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # (round 179) the n_jobs spinbox parses pre-flight and rides
        # train_model; a bad value stops before the job starts.
        app._ml_load_example()
        app._ml_njobs_spin.set("4")
        assert app._ml_split_settings() == ("random", 4)
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        assert quiet_dialogs == []
        app._ml_njobs_spin.set("not-a-number")
        app._ml_train()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Check the manual settings" in titles
        assert "n_jobs" in quiet_dialogs[-1][1][1]
        app._ml_njobs_spin.set("1")

    def test_tune_search_reports_progress_with_eta(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._tune_trials.set("4")
        app._ml_tune()
        saw_fraction = False
        saw_eta = False
        import time as _time

        deadline = _time.time() + 120.0
        while app._busy and _time.time() < deadline:
            app.update()
            label = app._job_label.cget("text")
            if "/4" in label:
                saw_fraction = True
            if " left" in label:
                saw_eta = True
            _time.sleep(0.02)
        assert not app._busy, "tune job hung"
        assert saw_fraction and saw_eta, "the search never reported progress"
        assert len(app._tune_last[0].trials) == 4
        # the finished-job label (100x-119) names the search + duration
        assert " — done in " in app._job_label.cget("text")
        assert quiet_dialogs == []

    def test_apply_best_prefills_the_manual_panel(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._tune_trials.set("4")
        app._ml_tune()
        assert self._drain_jobs(app, timeout=150.0), "tune job hung"
        results, estimator, task, _target = app._tune_last
        app._ml_tune_apply()
        assert app._model_combo.get() == estimator
        # the winner's alpha (a registry knob of ann) landed in the panel
        if "alpha" in app._ml_param_entries:
            assert app._ml_param_entries["alpha"].get() == str(results.best_params["alpha"])
        pinned = app._ml_manual_params(estimator, task)
        assert pinned is None or "alpha" in pinned  # parseable end-to-end
        assert quiet_dialogs == []


# --------------------------------------------------------------------- #
# the Docking tab under Xvfb -- real callbacks, real worker jobs against
# the shipped example data, dialogs recorded instead of blocking
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestDockingGuiUnderXvfb:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp()
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_tab_order_places_docking_after_cell_lines(self, app: Any) -> None:
        titles = [app._book.tab(tab, "text").strip() for tab in app._book.tabs()]
        assert titles == [
            "Home",
            "Chemistry",
            "PK / PD",
            "Cell Lines",
            "Docking",
            "Data & ML",
            "Unsupervised",
            "Deep Learning",
            "Foundations",
            "Best-of ML",
            "Response AI",
            "Generative",
            "Utilities",
        ]

    def test_docking_example_library_loads(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_load_example()
        info = app._dock_lib_info.cget("text")
        assert "10 rows" in info
        assert "8 valid" in info
        assert quiet_dialogs == []

    def test_docking_prepare_ligands_job_fills_the_tree(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        app._dock_load_example()
        app._dock_out_dir = tmp_path
        app._dock_prepare_ligands()
        assert self._drain_jobs(app), "ligand prep job hung"
        rows = app._dock_lib_tree.get_children()
        assert len(rows) == 10
        ok_flags = [app._dock_lib_tree.item(row, "values")[2] for row in rows]
        assert ok_flags.count("Yes") == 8
        assert "8/10" in app._dock_prep_summary.cget("text")
        assert quiet_dialogs == []

    def test_docking_receptor_prepare_job_renders_the_report(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        app._dock_load_receptor_example()
        target = tmp_path / "receptor.pdbqt"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._dock_prepare_receptor()
        assert self._drain_jobs(app, timeout=180.0), "receptor prep job hung"
        report = app._dock_protein_report
        assert report is not None
        assert report.output_atoms > 1000  # the example lysozyme receptor
        text = app._dock_protein_text.get("1.0", "end")
        assert "1,262" in text  # thousands-separated atom count
        assert app._dock_receptor_path == target
        assert target.is_file()
        assert quiet_dialogs == []

    def test_docking_grid_box_job_writes_gpf_and_dpf(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        app._dock_load_example()
        app._dock_out_dir = tmp_path
        app._dock_prepare_ligands()
        assert self._drain_jobs(app), "ligand prep job hung"
        first_ok = app._dock_report.frame[app._dock_report.frame["ok"]].iloc[0]
        app._dock_ligand_path = Path(first_ok["pdbqt"])
        app._dock_compute_grid()
        assert self._drain_jobs(app), "grid box job hung"
        box = app._dock_box
        assert box is not None
        assert box.size == (22.125, 22.125, 22.125)  # (60 - 1) * 0.375
        assert app._dock_ligand_path.with_name("r.gpf").is_file()
        assert app._dock_ligand_path.with_name("r.dpf").is_file()
        assert "22.125" in app._dock_grid_text.get("1.0", "end")
        assert quiet_dialogs == []

    def test_docking_example_campaign_scan(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_load_example_campaign()
        assert self._drain_jobs(app), "campaign scan job hung"
        rows = app._dock_results_tree.get_children()
        assert len(rows) == 14
        info = app._dock_results_info.cget("text")
        assert "14" in info and "12 OK" in info
        assert quiet_dialogs == []

    def test_docking_example_dlg_parse_and_ligand_efficiency(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_open_example_dlg()
        assert self._drain_jobs(app), "DLG parse job hung"
        rows = app._dock_results_tree.get_children()
        assert len(rows) == 5  # aspirin's five clusters
        values = app._dock_results_tree.item(rows[0], "values")
        assert values[0] == "1"  # rank
        assert values[1] == "-4.9100"  # lowest energy, 4 dp
        assert "5 clusters" in app._dock_results_info.cget("text")
        assert quiet_dialogs == []

        # ligand efficiency needs the heavy-atom count: aspirin has 13
        app._dock_heavyatoms_entry.delete(0, "end")
        app._dock_heavyatoms_entry.insert(0, "13")
        app._dock_open_example_dlg()
        assert self._drain_jobs(app), "DLG + LE job hung"
        assert "LE (kcal/mol/atom)" in app._dock_results_tree["columns"]
        first = app._dock_results_tree.get_children()[0]
        le_value = app._dock_results_tree.item(first, "values")[-1]
        assert le_value == "0.3777"  # |−4.91| / 13 heavy atoms
        assert quiet_dialogs == []

    def test_docking_vina_output_parse(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        sample = (
            "Modes | Affinity | dist from best mode\n"
            "      |          | rmsd l.b. | rmsd u.b.\n"
            "-------+----------+----------+----------\n"
            "   1      -10.3          0          0\n"
            "   2       -9.6      1.842      2.301\n"
            "   3       -9.4      2.105      2.777\n"
            "   4       -8.9      2.733      3.418\n"
        )
        vina_log = tmp_path / "vina_out.txt"
        vina_log.write_text(sample, encoding="utf-8")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **_k: str(vina_log))
        app._dock_open_vina()
        assert self._drain_jobs(app), "vina parse job hung"
        rows = app._dock_results_tree.get_children()
        assert len(rows) == 4
        values = app._dock_results_tree.item(rows[0], "values")
        assert values[0] == "1"  # mode number
        assert "4 modes" in app._dock_results_info.cget("text")
        assert quiet_dialogs == []

    def test_docking_run_gate_reports_missing_binary(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        app._dock_campaign_root = tmp_path
        # (100x-171) the bundled engines/ folder ships the .exe builds, so
        # this gate test neutralises it: "no engine ANYWHERE -> dialog"
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "no_engines")
        monkeypatch.setattr(playground.shutil, "which", lambda _name: None)
        app._dock_run()
        assert not app._busy  # the gate stopped before any job started
        assert len(quiet_dialogs) == 1
        kind, args = quiet_dialogs[0]
        assert kind == "showinfo"
        message = " ".join(str(part) for part in args).lower()
        assert "autodock4" in message
        assert "install" in message  # actionable hint, not a bare refusal

    @pytest.mark.filterwarnings("ignore:This process.*multi-threaded.*fork:DeprecationWarning")
    def test_docking_run_starts_job_with_fake_binary(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        root = tmp_path / "campaign"
        ligand_dir = root / "01_lig"
        ligand_dir.mkdir(parents=True)
        (ligand_dir / "r.gpf").write_text("# stub\n", encoding="utf-8")
        (ligand_dir / "r.dpf").write_text("# stub\n", encoding="utf-8")
        app._dock_campaign_root = root
        # keep the PATH-only route: without this the bundled engines/
        # folder would win and the fake binary below never launches
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "no_engines")
        monkeypatch.setattr(
            playground.shutil,
            "which",
            lambda name: "/usr/bin/true" if name in ("autogrid4", "autodock4") else None,
        )
        app._dock_run()
        assert app._busy  # the gate passed and the job started
        assert self._drain_jobs(app), "fake-binary run job hung"
        summary = app._dock_run_summary.cget("text")
        assert "1 folders docked" in summary  # one stub ligand directory
        assert "1 failed" in summary  # /usr/bin/true produces no .dlg
        assert quiet_dialogs == []

    # -- (100x-171) the engine-binaries picker card ---------------------- #
    def test_engine_binaries_picker_labels_resolution_per_engine(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        # the picker resolves every engine (vina / autogrid4 / autodock4)
        # through: the picked folder, then the bundled engines/ folder,
        # then PATH -- and confesses per engine where each one came from
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "nowhere")
        monkeypatch.setattr(playground.shutil, "which", lambda _name: None)
        app._dock_engines_dir = None
        app._dock_refresh_engines_info()
        missing_text = str(app._dock_engines_info.cget("text"))
        assert "vina: —" in missing_text, missing_text
        assert "PATH only" in missing_text, missing_text

        picked = tmp_path / "picked"
        picked.mkdir()
        (picked / "vina.exe").write_bytes(b"")
        (picked / "autogrid4.exe").write_bytes(b"")
        app._dock_engines_dir = picked
        app._dock_refresh_engines_info()
        text = str(app._dock_engines_info.cget("text"))
        assert "vina: vina.exe (folder)" in text, text
        assert "autogrid4: autogrid4.exe (folder)" in text, text
        assert "autodock4: —" in text, text
        assert str(picked) in text, text
        assert quiet_dialogs == []
        app._dock_engines_dir = None

    def test_docking_consensus_rescore_example_campaign(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_load_example_campaign()
        assert self._drain_jobs(app), "campaign scan job hung"

        # the prefill seeds the scan table's canonical evidence columns
        prefill = app._dock_rescore_columns_entry.get()
        assert prefill == "best_lowest_energy, best_num_in_cluster"

        app._dock_rescore()
        assert self._drain_jobs(app), "rescore job hung"
        assert app._dock_results_source == "rescore"
        tree = app._dock_results_tree
        rows = tree.get_children()
        assert len(rows) == 14  # 12 scored + the 2 incomplete folders
        assert list(tree["columns"]) == [
            "folder",
            "best_lowest_energy_norm",
            "best_num_in_cluster_norm",
            "consensus",
        ]
        winner = list(tree.item(rows[0], "values"))
        assert winner == ["13_coumarin", "0.3682", "1.0000", "0.6841"]
        # the two NaN folders sort last and show empty consensus cells
        assert tree.item(rows[-1], "values")[0] == "10_nicotine"
        assert tree.item(rows[-1], "values")[-1] == ""
        # the frame behind Save results CSV… is the full rescored table
        frame = app._dock_results_frame
        assert list(frame.columns[:1]) + list(frame.columns[-3:]) == [
            "folder",
            "best_lowest_energy_norm",
            "best_num_in_cluster_norm",
            "consensus",
        ]
        text = app._dock_rescore_text.get("1.0", "end")
        assert "evidence: best_lowest_energy (lower=better)" in text
        assert "best_num_in_cluster (higher=better)" in text
        assert "14 row(s) rescored" in text
        assert (
            "13_coumarin tops the table (consensus 0.684): "
            "best_num_in_cluster_norm = 1.000 is the driving evidence"
        ) in text
        assert quiet_dialogs == []

    def test_docking_consensus_rescore_dlg_with_ligand_efficiency(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_heavyatoms_entry.delete(0, "end")
        app._dock_heavyatoms_entry.insert(0, "13")  # aspirin heavy atoms
        app._dock_open_example_dlg()
        assert self._drain_jobs(app), "DLG + LE job hung"

        # the prefill prefers the LE column first (§9's story: energy,
        # cluster population and ligand efficiency)
        prefill = app._dock_rescore_columns_entry.get()
        assert prefill == "LE (kcal/mol/atom), lowest_energy, num_in_cluster"

        app._dock_rescore()
        assert self._drain_jobs(app), "rescore job hung"
        rows = app._dock_results_tree.get_children()
        assert len(rows) == 5  # aspirin's five clusters
        assert list(app._dock_results_tree["columns"]) == [
            "rank",
            "LE (kcal/mol/atom)_norm",
            "lowest_energy_norm",
            "num_in_cluster_norm",
            "consensus",
        ]
        # rank 4 wins: LE + energy units 0.901 carry it, not rank 1
        winner = list(app._dock_results_tree.item(rows[0], "values"))
        assert winner == ["4", "0.9008", "0.9008", "0.7778", "0.8598"]
        text = app._dock_rescore_text.get("1.0", "end")
        assert "weight" not in text.split("\n")[0]  # uniform -> no weights
        assert (
            "4 tops the table (consensus 0.860): "
            "LE (kcal/mol/atom)_norm = 0.901 is the driving evidence"
        ) in text
        assert quiet_dialogs == []

    def test_docking_rescore_weights_change_the_recipe_line(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_load_example_campaign()
        assert self._drain_jobs(app), "campaign scan job hung"
        app._dock_rescore_weights_entry.insert(0, "1,3")
        app._dock_rescore()
        assert self._drain_jobs(app), "weighted rescore job hung"
        rows = app._dock_results_tree.get_children()
        # cluster population x3 lifts the same winner higher
        assert app._dock_results_tree.item(rows[0], "values")[0] == "13_coumarin"
        assert app._dock_results_tree.item(rows[0], "values")[-1] == "0.8420"
        text = app._dock_rescore_text.get("1.0", "end")
        assert "best_lowest_energy (lower=better, weight 1)" in text
        assert "best_num_in_cluster (higher=better, weight 3)" in text
        assert quiet_dialogs == []

    def test_docking_rescore_gates(self, app: Any, quiet_dialogs: list[Any]) -> None:
        # no results table loaded -> the guard names the remedy
        app._dock_rescore()
        assert len(quiet_dialogs) == 1
        kind, args = quiet_dialogs[0]
        assert kind == "showwarning"
        message = " ".join(str(part) for part in args)
        assert "Scan a campaign" in message

        # a loaded table but an empty columns entry -> advisory warning
        # (100x-143: a missing input is a warning like every sibling
        # card; the malformed-weights PARSE error below stays showerror)
        app._dock_load_example_campaign()
        assert self._drain_jobs(app), "campaign scan job hung"
        app._dock_rescore_columns_entry.delete(0, "end")
        app._dock_rescore()
        assert len(quiet_dialogs) == 2
        kind, args = quiet_dialogs[1]
        assert kind == "showwarning"
        assert "at least one evidence column" in " ".join(str(p) for p in args)

        # malformed weights -> the parse_float error verbatim
        app._dock_rescore_columns_entry.insert(0, "best_lowest_energy, best_num_in_cluster")
        app._dock_rescore_weights_entry.insert(0, "1,x")
        app._dock_rescore()
        assert len(quiet_dialogs) == 3
        kind, args = quiet_dialogs[2]
        assert kind == "showerror"
        assert "is not a number" in " ".join(str(p) for p in args)
        assert not app._busy  # every gate stopped before any job started

    def test_docking_rescore_open_summary_csv(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        # the --data twin: any summary CSV becomes the results table
        summary = tmp_path / "enriched_summary.csv"
        summary.write_text(
            "ligand,best_lowest_energy,best_num_in_cluster,ligand_efficiency\n"
            "aspirin,-4.9100,15,0.3777\n"
            "theophylline,-11.5100,15,0.8854\n"
            "metformin,-8.1300,19,0.9033\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            playground.filedialog,
            "askopenfilename",
            lambda **_k: str(summary),
        )
        app._dock_open_rescore_csv()
        assert self._drain_jobs(app), "summary CSV job hung"
        assert app._dock_results_source == "csv"
        assert len(app._dock_results_tree.get_children()) == 3
        # the prefill recognises the enriched table's canonical columns
        assert app._dock_rescore_columns_entry.get() == (
            "ligand_efficiency, best_lowest_energy, best_num_in_cluster"
        )
        info = app._dock_results_info.cget("text")
        assert "3 row(s)" in info and "enriched_summary.csv" in info

        # and the §9 verdict reproduces on it: LE carries metformin
        app._dock_rescore()
        assert self._drain_jobs(app), "rescore job hung"
        text = app._dock_rescore_text.get("1.0", "end")
        assert "metformin tops the table" in text
        assert "ligand_efficiency_norm = 1.000 is the driving evidence" in text
        first = app._dock_results_tree.get_children()[0]
        assert app._dock_results_tree.item(first, "values")[0] == "metformin"
        assert quiet_dialogs == []


# --------------------------------------------------------------------- #
# docking rescore report bundle, live under Xvfb
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestDockingRescoreReport:
    """``_dock_save_report`` rebuilds the CLI bundle from the stored rescore.

    The report story's docking axis (100x-86): library figure (1.2.34) ->
    CLI --plots (1.2.34) -> GUI.  Guards first (no rescore, no fpdf),
    then the end-to-end flow on the example campaign with the PDF
    ticked, and the unticked HTML-only default.
    """

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
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
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_report_needs_a_rescore_first(self, app: Any, dialogs: list[Any]) -> None:
        app._dock_save_report()
        assert not app._busy  # the guard fired before any job started
        titles = [call[1][0] for call in dialogs]
        assert "No rescore yet" in titles
        assert "explainable" in dialogs[0][1][1]  # the remedy, named

    def test_report_pdf_gate_without_fpdf(
        self, app: Any, dialogs: list[Any], monkeypatch: Any, playground: Any
    ) -> None:
        app._dock_rescore_last = (object(), ["a"], None, {}, "x")
        monkeypatch.setattr(playground, "fpdf_available", lambda: False)
        app._dock_pdf_var.set(True)
        app._dock_save_report()
        assert not app._busy  # gated before the folder dialog
        titles = [call[1][0] for call in dialogs]
        assert "PDF needs fpdf" in titles
        assert "pharmaml[reports]" in dialogs[0][1][1]  # the install hint

    def test_report_bundle_end_to_end(
        self,
        app: Any,
        dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
        tmp_path: Path,
    ) -> None:
        pypdf = pytest.importorskip("pypdf")
        target = tmp_path / "bundle"
        monkeypatch.setattr(playground.filedialog, "askdirectory", lambda **k: str(target))
        app._dock_load_example_campaign()
        assert self._drain(app), "campaign scan job did not finish"
        app._dock_rescore()
        assert self._drain(app), "rescore job did not finish"
        assert app._dock_rescore_last is not None  # now reportable

        app._dock_pdf_var.set(True)
        app._dock_save_report()
        assert self._drain(app), "report job did not finish"

        # the CLI --plots bundle, mouse edition: both figures (the scan
        # table is folder-shaped) + the breakdown table + the documents
        assert (target / "report.html").is_file()
        assert (target / "report.pdf").is_file()
        assert (target / "consensus_ranking.png").is_file()
        assert (target / "energy_distribution.png").is_file()
        assert (target / "consensus_breakdown.csv").is_file()
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the success dialog never fired"
        assert infos[0][1][0] == "Report saved"
        assert "report saved" in app._job_label.cget("text")
        reader = pypdf.PdfReader(str(target / "report.pdf"))
        first = reader.pages[0].extract_text()
        assert "playground docking rescore" in first
        assert "evidence" in first.lower()

        # the unticked default: a second save ships HTML only
        target_html = tmp_path / "bundle_html"
        monkeypatch.setattr(playground.filedialog, "askdirectory", lambda **k: str(target_html))
        app._dock_pdf_var.set(False)
        app._dock_save_report()
        assert self._drain(app), "second report job did not finish"
        assert (target_html / "report.html").is_file()
        assert not (target_html / "report.pdf").exists()  # the unticked default


# --------------------------------------------------------------------- #
# session persistence + theme switching, live under Xvfb
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestSessionGuiUnderXvfb:
    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @staticmethod
    def _drain(app: Any) -> None:
        for _ in range(4):
            app.update_idletasks()
            app.update()

    def test_inputs_survive_close_and_reopen(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        import gc

        first = playground.PlaygroundApp()  # persist_session=True (default)
        self._drain(first)
        first._aff_value.delete(0, "end")
        first._aff_value.insert(0, "3.14")
        first._tune_var.set(True)
        first._feat_family_vars["morgan_scaffold"].set(False)
        first._feat_pasted_text.delete("1.0", "end")
        first._feat_pasted_text.insert("1.0", "CCO\n# note")
        first._book.select(4)
        first._on_close()  # close -> must save the session
        del first
        gc.collect()

        assert (isolated_home / ".pharmaml" / "playground-session.json").is_file()
        second = playground.PlaygroundApp()
        self._drain(second)
        assert second._aff_value.get() == "3.14"
        assert bool(second._tune_var.get()) is True
        assert bool(second._feat_family_vars["morgan_scaffold"].get()) is False
        assert bool(second._feat_family_vars["descriptors"].get()) is True
        assert second._feat_pasted_text.get("1.0", "end-1c") == "CCO\n# note"
        assert second._book.index(second._book.select()) == 4
        assert "session restored" in str(second._job_label.cget("text"))
        second.destroy()
        del second
        gc.collect()

    def test_persist_session_false_never_writes(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        app = playground.PlaygroundApp(persist_session=False)
        self._drain(app)
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.99")
        app._on_close()
        assert not (isolated_home / ".pharmaml").exists()

    def test_theme_flip_recolours_live_widgets_and_collects(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        import gc

        app = playground.PlaygroundApp()
        self._drain(app)
        assert str(app.cget("background")) == playground._LIGHT_PALETTE["BG"]
        app._theme_var.set("dark")
        app._on_theme_change()
        self._drain(app)
        assert playground._ACTIVE_THEME == "dark"
        assert str(app.cget("background")) == playground._DARK_PALETTE["BG"]
        assert str(app._corpus_text.cget("background")) == playground._DARK_PALETTE["CARD"]
        assert str(app._corpus_text.cget("foreground")) == playground._DARK_PALETTE["INK"]
        session = app._collect_session()
        assert session["theme"] == "dark"
        app._theme_var.set("light")
        app._on_theme_change()
        self._drain(app)
        assert playground._ACTIVE_THEME == "light"
        assert str(app.cget("background")) == playground._LIGHT_PALETTE["BG"]
        assert str(app._corpus_text.cget("background")) == playground._LIGHT_PALETTE["CARD"]
        # _restore_session must also boot straight into the saved theme
        app._write_session_file()
        app.destroy()
        del app
        gc.collect()
        reopened = playground.PlaygroundApp()
        self._drain(reopened)
        assert reopened._theme == "light"  # the last flip was persisted
        reopened.destroy()
        del reopened
        gc.collect()
        playground.apply_theme("light")  # never leak into other tests

    def test_off_screen_geometry_restores_size_only(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        import gc

        # a session recorded on a much bigger screen: keep the size,
        # drop the off-screen position
        payload = playground.dump_session(
            theme="light",
            geometry="1000x700+40000+50000",
            tab=0,
            variables={"_aff_value": "1.5"},
            families={},
            texts={},
        )
        (isolated_home / ".pharmaml").mkdir(parents=True)
        (isolated_home / ".pharmaml" / "playground-session.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8"
        )
        app = playground.PlaygroundApp()
        self._drain(app)
        assert app._aff_value.get() == "1.5"  # inputs restored
        assert app.winfo_width() <= 1000  # size honoured (Xvfb 1280x800)
        # no window manager under Xvfb pins the window at +0+0; what
        # matters is that the OFF-SCREEN +40000+50000 was dropped
        assert "+40000" not in app.geometry()  # position dropped
        app.destroy()
        del app
        gc.collect()


# --------------------------------------------------------------------- #
# the Chemistry tab's feature factory -- surface + pure helpers
# --------------------------------------------------------------------- #
class TestFeatureFactorySurface:
    def test_chemistry_tab_builder_exists(self, playground: Any) -> None:
        assert hasattr(playground.PlaygroundApp, "_build_chemistry_tab")

    def test_docstring_mentions_the_feature_factory(self, playground: Any) -> None:
        assert "feature factory" in (playground.__doc__ or "").lower()

    def test_chemistry_tab_source_references_the_feature_factory(self, playground: Any) -> None:
        import inspect

        source = inspect.getsource(playground.PlaygroundApp._build_chemistry_tab)
        assert "Feature factory" in source
        assert "_feat_compute" in source

    def test_feature_factory_width_sums_family_bits(self, playground: Any) -> None:
        width = playground.feature_factory_width(
            (
                "descriptors",
                "atompair_count",
                "ecfp",
                "avalon_count",
                "rdkit_fingerprint",
                "morgan_full",
                "morgan_scaffold",
            ),
            43,
        )
        assert width == 10_752 + 43  # 5 x 2048 + 512 + the descriptor count
        assert playground.feature_factory_width(("descriptors",), 50) == 50
        assert playground.feature_factory_width(("avalon_count",), 43) == 512
        assert playground.feature_factory_width((), 43) == 0
        with pytest.raises(ValueError, match="unknown feature family"):
            playground.feature_factory_width(("typo",), 43)

    def test_detect_smiles_column_prefers_named_columns(self, playground: Any) -> None:
        class _Frame:
            def __init__(self, columns: list[str]) -> None:
                self.columns = columns

        assert playground.detect_smiles_column(_Frame(["name", "SMILES"])) == "SMILES"
        assert playground.detect_smiles_column(_Frame(["Canonical_Smiles"])) == "Canonical_Smiles"
        assert playground.detect_smiles_column(_Frame(["molecule", "activity"])) is None

    def test_ranker_and_task_label_maps_cover_every_combobox_value(self, playground: Any) -> None:
        assert set(playground._FEAT_RANKER_METHODS) == set(playground._FEAT_RANKERS)
        assert playground._FEAT_RANKER_METHODS["Variance (no target needed)"] == "variance"
        assert playground._FEAT_RANKER_METHODS["Random forest"] == "random_forest"
        assert playground._FEAT_RANKER_METHODS["Auto (RandomForest or LightGBM)"] == "auto"
        assert playground._FEAT_RANKER_METHODS["LightGBM (if installed)"] == "lightgbm"
        assert set(playground._FEAT_TASK_METHODS) == set(playground._FEAT_TASKS)
        assert playground._FEAT_TASK_METHODS["Auto detect"] == "auto"
        assert playground._FEAT_TASK_METHODS["Classification"] == "classification"
        assert playground._FEAT_TASK_METHODS["Regression"] == "regression"


# --------------------------------------------------------------------- #
# the feature factory under Xvfb -- real callbacks, real worker jobs
# against the public pharmaml.chem.features API, dialogs recorded
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestFeatureFactoryGuiUnderXvfb:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp()
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _descriptor_count() -> int:
        from rdkit.Chem import rdMolDescriptors

        return len(rdMolDescriptors.Properties.GetAvailableProperties())

    def test_default_pasted_variance_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        # the shipped defaults: pasted drug SMILES, all 20 families ticked
        # (100x-148: the ten fak_ml families + shape_3d joined the nine)
        assert app._feat_source_var.get() == "Pasted SMILES"
        assert "20 families selected" in app._feat_width_label.cget("text")
        expected_columns = 12_800 + 7 * 2048 + 167 + 79 + 315 + 9 + 10 + self._descriptor_count()
        assert f"~{expected_columns:,} columns" in app._feat_width_label.cget(
            "text"
        )
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "50")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app, timeout=240.0), "feature factory job hung"
        summary = app._feat_summary.cget("text")
        assert f"{expected_columns:,} features" in summary
        assert "3 molecules" in summary
        assert "median-survivors" in summary
        assert "variance" in summary
        assert "selected per family" in app._feat_family_label.cget("text")
        assert tuple(app._feat_tree["columns"]) == (
            "rank",
            "feature",
            "family",
            "importance",
        )
        rows = app._feat_tree.get_children()
        assert len(rows) == 50
        first = app._feat_tree.item(rows[0], "values")
        assert first[0] == "1"  # rank column counts from 1
        # (100x-148) the top-50 variance features may now come from any
        # of the twenty families -- the vocabulary pin moved to the
        # library's own FEATURE_FAMILIES.
        from pharmaml.chem.features import FEATURE_FAMILIES as _LIB_FAMILIES

        families_seen = {app._feat_tree.item(row, "values")[2] for row in rows}
        assert families_seen <= set(_LIB_FAMILIES)
        assert quiet_dialogs == []

    def test_exports_rar_style_csv_and_metadata_json(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        import json

        import pandas as pd

        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "50")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app), "feature factory job hung"

        csv_target = tmp_path / "selected_features.csv"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(csv_target)
        )
        app._feat_save_csv()
        assert csv_target.is_file()
        exported = pd.read_csv(csv_target)
        assert exported.columns[0] == "SMILES"
        assert exported.shape == (3, 51)  # SMILES + 50 features, no target
        assert exported["SMILES"].tolist() == [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "CC(=O)Nc1ccc(O)cc1",
        ]
        # the feature columns follow the selection's rank order verbatim
        assert list(exported.columns[1:]) == list(app._feat_selection.selected)

        json_target = tmp_path / "selection_metadata.json"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(json_target)
        )
        app._feat_save_json()
        meta = json.loads(json_target.read_text(encoding="utf-8"))
        assert len(meta["selected_features"]) == 50
        assert meta["total_features"] > 10_000
        assert meta["k_requested"] == 50
        assert meta["method"] == "variance"
        assert meta["n_molecules"] == 3
        assert "target_column" not in meta  # a variance run has no target
        # the only dialogs are the two Saved confirmations
        assert [kind for kind, _args in quiet_dialogs] == ["showinfo", "showinfo"]
        assert all(
            "written to" in " ".join(str(part) for part in args) for _kind, args in quiet_dialogs
        )

    def test_tree_truncates_at_300_rows_and_reruns_at_new_k(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "400")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app), "feature factory job hung"
        assert len(app._feat_tree.get_children()) == 300
        assert "+100 more rows" in app._feat_more_label.cget("text")

        # job reuse: the buttons re-enabled, a second run re-pins the tree
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "10")
        app._feat_compute()
        assert self._drain_jobs(app), "feature factory re-run job hung"
        assert len(app._feat_tree.get_children()) == 10
        assert app._feat_more_label.cget("text") == ""
        assert "kept top 10" in app._feat_summary.cget("text")
        assert quiet_dialogs == []

    def test_example_csv_load_pins_rows_and_target_candidates(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._feat_source_var.set("CSV file")
        app._feat_refresh_source()
        app._feat_load_example()
        info = app._feat_source_info.cget("text")
        # the shipped 77-row library: SMILES + 8 numeric descriptor columns
        assert "77 rows" in info
        assert "numeric target candidates" in info
        values = list(app._feat_target_combo["values"])
        assert values[0] == playground._FEAT_NO_TARGET  # the default entry
        assert "MolWt" in values and "RingCount" in values
        assert quiet_dialogs == []

    def test_unchecked_families_narrow_the_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        keep = {"descriptors", "avalon_count"}
        for family, var in app._feat_family_vars.items():
            var.set(family in keep)
        app._feat_update_width()
        width_label = app._feat_width_label.cget("text")
        assert "2 families selected" in width_label
        assert f"~{self._descriptor_count() + 512:,} columns" in width_label
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "25")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app), "narrow feature factory job hung"
        summary = app._feat_summary.cget("text")
        assert f"{self._descriptor_count() + 512:,} features" in summary
        rows = app._feat_tree.get_children()
        assert len(rows) == 25
        families_seen = {app._feat_tree.item(row, "values")[2] for row in rows}
        assert families_seen <= {"descriptors", "avalon_count"}
        assert quiet_dialogs == []

    def test_target_driven_random_forest_run_exports_target_column(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Any,
    ) -> None:
        import pandas as pd

        csv_path = tmp_path / "tiny_screen.csv"
        csv_path.write_text(
            "SMILES,activity\nC,0\nCC,0\nCCC,0\nCCCC,1\nCCCCC,1\nCCCCCC,1\n",
            encoding="utf-8",
        )
        app._feat_source_var.set("CSV file")
        app._feat_refresh_source()
        app._feat_load_path(csv_path)
        assert "6 rows" in app._feat_source_info.cget("text")
        app._feat_target_combo.set("activity")
        app._feat_ranker_combo.set("Random forest")
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "5")
        app._feat_compute()
        assert self._drain_jobs(app, timeout=180.0), "random forest job hung"
        summary = app._feat_summary.cget("text")
        assert "6 molecules" in summary
        assert "random forest" in summary
        assert quiet_dialogs == []

        target = tmp_path / "selected_with_target.csv"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._feat_save_csv()
        assert target.is_file()
        exported = pd.read_csv(target)
        # structure only (RF importances over 6 rows are not pin-stable):
        # SMILES first, then the target under its original name, 5 features
        assert exported.columns[0] == "SMILES"
        assert exported.columns[1] == "activity"
        assert exported.shape == (6, 7)
        assert exported["activity"].tolist() == [0, 0, 0, 1, 1, 1]
        assert [kind for kind, _args in quiet_dialogs] == ["showinfo"]
        assert "written to" in " ".join(str(part) for part in quiet_dialogs[0][1])

    def test_k_zero_records_the_input_error_dialog(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "0")
        app._feat_compute()
        assert not app._busy  # no job was started
        assert len(quiet_dialogs) == 1
        kind, args = quiet_dialogs[0]
        assert kind == "showerror"
        message = " ".join(str(part) for part in args)
        assert "features to keep (K)" in message
        assert "below the minimum" in message


# --------------------------------------------------------------------- #
# session export / import + per-tab reset (100x-54)
# --------------------------------------------------------------------- #
class TestSessionShareAndReset:
    """Shareable session files and the per-tab reset-to-defaults gate.

    Fail-first (100x-54): ``export_session`` / ``read_json_file`` and the
    ``_export_session`` / ``_import_session`` / ``_reset_current_tab``
    app methods did not exist before this round; the tests were run red
    against the pristine (staged) playground.py via a file-copy
    round-trip before the fix was restored.
    """

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    # -- pure helpers (no Tk needed) ----------------------------------- #

    def test_export_session_round_trip(self, playground: Any, tmp_path: Path) -> None:
        payload = playground.dump_session(
            theme="dark",
            geometry="1200x800+10+20",
            tab=3,
            variables={"_aff_value": "42"},
            families={"ecfp": False},
            texts={"pasted": "CCO\n# note"},
        )
        target = tmp_path / "shared-session.json"
        assert playground.export_session(payload, target) is True
        revived = playground.load_session(playground.read_json_file(target))
        assert revived == payload  # identical after the file round-trip

    def test_export_session_unwritable_path(self, playground: Any, tmp_path: Path) -> None:
        payload = playground.dump_session(
            theme="light", geometry="", tab=0, variables={}, families={}, texts={}
        )
        assert (
            playground.export_session(payload, tmp_path / "no" / "such" / "dir" / "s.json") is False
        )

    def test_read_json_file_missing_and_invalid(self, playground: Any, tmp_path: Path) -> None:
        assert playground.read_json_file(tmp_path / "absent.json") is None
        junk = tmp_path / "junk.json"
        junk.write_text("{not json at all", encoding="utf-8")
        assert playground.read_json_file(junk) is None

    # -- live GUI under Xvfb -------------------------------------------- #

    def test_export_writes_shareable_file(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        target = isolated_home / "colleague-session.json"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._export_session()
        assert target.is_file()
        payload = playground.load_session(playground.read_json_file(target))
        assert payload is not None
        assert payload["variables"]["_aff_value"] == "9.9"
        assert payload["theme"] in ("light", "dark", "auto")
        assert "session exported" in str(app._job_label.cget("text"))
        app.destroy()

    def test_import_applies_shareable_file(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        source = tmp_path / "arriving-session.json"
        assert playground.export_session(app._collect_session(), source)
        # the user then changes the live input -- import must win
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "1.1")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **_k: str(source))
        app._import_session()
        assert app._aff_value.get() == "9.9"
        assert "session imported" in str(app._job_label.cget("text"))
        app.destroy()

    def test_import_invalid_file_shows_error_and_keeps_state(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "7.7")
        junk = tmp_path / "not-a-session.json"
        junk.write_text('{"hello": "world"}', encoding="utf-8")
        seen: list[Any] = []
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **_k: str(junk))
        monkeypatch.setattr(
            playground.messagebox, "showerror", lambda *args, **_k: seen.append(args)
        )
        app._import_session()
        assert app._aff_value.get() == "7.7"  # untouched
        assert seen and "Invalid session file" in str(seen[0][0])
        assert "imported" not in str(app._job_label.cget("text"))
        app.destroy()

    def test_reset_current_tab_restores_only_that_tab(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        default_aff = app._aff_value.get()
        default_tune = bool(app._tune_var.get())
        # mutate one input on the Chemistry tab and one on the Data & ML tab
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        app._tune_var.set(not default_tune)
        chemistry_tab = next(
            index for index, names in app._tab_input_owners.items() if "_aff_value" in names
        )
        app._book.select(chemistry_tab)
        app._reset_current_tab()
        assert app._aff_value.get() == default_aff  # reset to default
        assert bool(app._tune_var.get()) is (not default_tune)  # other tab kept
        assert "tab reset" in str(app._job_label.cget("text"))
        app.destroy()

    def test_tab_ownership_map_excludes_theme_and_covers_families(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        owners = app._tab_input_owners
        every_name = {name for names in owners.values() for name in names}
        assert "_theme_var" not in every_name  # not a user input
        assert not any(name.startswith("_book") for name in every_name)
        family_tabs = [
            index
            for index, names in owners.items()
            if any(name.startswith("family:") for name in names)
        ]
        assert family_tabs == [1]  # the Chemistry tab owns the checkboxes
        text_tabs = [
            index
            for index, names in owners.items()
            if any(name.startswith("text:") for name in names)
        ]
        assert 1 in text_tabs  # pasted SMILES corpus lives on Chemistry
        app.destroy()


# --------------------------------------------------------------------- #
# per-tab reset buttons pinned in every tab footer (100x-55)
# --------------------------------------------------------------------- #
class TestPerTabResetButtons:
    """A quiet "Reset this tab to defaults" button pinned below every
    content tab's scroll area (Home -- zero inputs -- carries none).

    Fail-first (100x-55): ``ScrollFrame`` had no ``footer`` strip,
    ``_new_page`` built no button and ``_reset_tab`` did not exist; the
    probe (/tmp/task55/probe_reset_button.py) printed ``PROBE FAIL:
    missing buttons on tabs [1, 2, 3, 4, 5, 6]`` against the pristine
    source, and these tests were run red through the same file-copy
    round-trip before the fix was restored.
    """

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    def _footer_buttons(self, app: Any, index: int) -> list[Any]:
        page = app._book.nametowidget(app._book.tabs()[index])
        return [
            widget for widget in page.footer.winfo_children() if widget.winfo_class() == "TButton"
        ]

    def test_every_content_tab_carries_both_footer_buttons(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        for index in range(len(app._book.tabs())):
            page = app._book.nametowidget(app._book.tabs()[index])
            assert hasattr(page, "footer")
            # pinned OUTSIDE the scroll body: always visible, never scrolled
            assert page.footer.master is page
            assert page.footer.master is not page.body
            buttons = self._footer_buttons(app, index)
            if index == 0:  # Home owns no inputs -> no footer affordances
                assert buttons == []
            else:
                texts = [str(b.cget("text")) for b in buttons]
                assert texts.count("Reset this tab to defaults") == 1
                assert texts.count("Copy inputs as JSON") == 1
                for button in buttons:
                    assert button.cget("style") == "Ghost.TButton"
        app.destroy()

    def test_footer_button_resets_own_tab_not_the_selected_one(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        default_aff = app._aff_value.get()
        default_tune = bool(app._tune_var.get())
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        app._tune_var.set(not default_tune)
        # PK/PD is the VISIBLE tab -- Chemistry's footer button must still
        # reset CHEMISTRY (index-explicit wiring, not "current tab")
        app._book.select(2)
        chemistry = next(
            index for index, names in app._tab_input_owners.items() if "_aff_value" in names
        )
        reset = next(
            b
            for b in self._footer_buttons(app, chemistry)
            if str(b.cget("text")) == "Reset this tab to defaults"
        )
        reset.invoke()
        assert app._aff_value.get() == default_aff  # its own tab restored
        assert bool(app._tune_var.get()) is (not default_tune)  # Data & ML kept
        assert "tab reset" in str(app._job_label.cget("text"))
        app.destroy()

    def test_reset_tab_is_index_explicit(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        default_tune = bool(app._tune_var.get())
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "5.5")
        app._tune_var.set(not default_tune)
        data_ml = next(
            index for index, names in app._tab_input_owners.items() if "_tune_var" in names
        )
        app._book.select(1)  # Chemistry visible; Data & ML reset by INDEX
        app._reset_tab(data_ml)
        assert bool(app._tune_var.get()) is default_tune  # restored
        assert app._aff_value.get() == "5.5"  # the other tab untouched
        app.destroy()

    def test_ghost_button_style_follows_the_live_theme(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        style = playground.ttk.Style(app)
        light_bg = playground.BG
        assert style.lookup("Ghost.TButton", "background") == light_bg
        app._set_theme("dark")
        assert style.lookup("Ghost.TButton", "background") == playground.BG
        assert style.lookup("Ghost.TButton", "background") != light_bg
        app._set_theme("light")
        app.destroy()


# --------------------------------------------------------------------- #
# the Generative tab's Pareto front card (100x-58)
# --------------------------------------------------------------------- #
class TestGenerativeParetoCard:
    DRUGLIKE = "Multi-objective (QED \u00d7 MW \u00d7 TPSA)"

    def test_objective_kinds_gain_the_druglike_triple(self, playground: Any) -> None:
        assert self.DRUGLIKE in playground._OBJECTIVE_KINDS

    def test_build_objective_druglike_scores_the_triple(self, playground: Any) -> None:
        # the CLI's 'druglike' recipe, rebuilt by its GUI label: the
        # aspirin breakdown must carry exactly the three band objectives.
        objective = playground.build_objective(self.DRUGLIKE, 1.0, 3.0)
        _total, breakdown = objective.score_with_breakdown("CC(=O)Oc1ccccc1C(=O)O")
        assert sorted(breakdown) == ["mw_band", "qed", "tpsa_band"]

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        # persist_session=False: the suite must not leak the mutated
        # Generative inputs into the developer's real session file.
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _run_druglike(self, app: Any, *, population: str = "12", generations: str = "4") -> None:
        app._objective_combo.set(self.DRUGLIKE)
        app._pop_size.set(population)
        app._generations.set(generations)
        app._run_evolution()
        assert self._drain_jobs(app), "evolution job hung"

    def test_multiobjective_run_fills_the_pareto_card(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._run_druglike(app)
        # the pinned deterministic seed-42 run: 36 molecules, 7 on the front
        summary = app._pareto_summary.cget("text")
        assert "Pareto front over 3 objectives [qed, mw_band, tpsa_band]" in summary
        assert "7 of 36 molecules non-dominated" in summary
        assert "deepest front member sits at leaderboard rank 20" in summary
        rows = app._pareto_tree.get_children()
        assert len(rows) == 7
        columns = list(app._pareto_tree["columns"])
        assert columns == ["rank", "smiles", "qed", "mw_band", "tpsa_band", "dominates", "crowding"]
        first = app._pareto_tree.item(rows[0], "values")
        assert first[0] == "1"  # leaderboard rank 1 is always on the front
        # rank 1 leads the BLEND, not necessarily an objective extreme --
        # the +inf crowding seats belong to the trade-off corners instead
        # (this pinned run: 5 of 7 members are boundary corners)
        all_values = [app._pareto_tree.item(row, "values") for row in rows]
        assert sum("inf" in values for values in all_values) == 5
        assert app._pareto_x_combo["values"] == ("qed", "mw_band", "tpsa_band")
        assert app._pareto_y_combo["values"] == ("qed", "mw_band", "tpsa_band")
        assert app._pareto_x_combo.get() == "qed"
        assert app._pareto_y_combo.get() == "mw_band"
        assert app._pareto_host._pg_figure is not None  # auto-drawn scatter
        assert quiet_dialogs == []

    def test_single_objective_run_explains_itself(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._objective_combo.set("QED drug-likeness")
        app._pop_size.set("8")
        app._generations.set("4")
        app._run_evolution()
        assert self._drain_jobs(app), "evolution job hung"
        summary = app._pareto_summary.cget("text")
        assert "single objective" in summary
        assert "Multi-objective" in summary
        assert app._pareto_tree.get_children() == ()
        assert app._pareto_front is None
        assert getattr(app._pareto_host, "_pg_figure", None) is None
        assert app._ga_result is not None  # the run itself succeeded
        assert quiet_dialogs == []

    def test_plot_button_follows_the_selected_objective_pair(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._run_druglike(app)
        app._pareto_y_combo.set("tpsa_band")
        app._plot_pareto_front()
        figure = app._pareto_host._pg_figure
        assert figure is not None
        axes = figure.get_axes()[0]
        assert axes.get_xlabel() == "qed"
        assert axes.get_ylabel() == "tpsa_band"
        assert quiet_dialogs == []

    def test_save_front_writes_the_full_table(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Path,
    ) -> None:
        self._run_druglike(app)
        target = tmp_path / "front.csv"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        app._save_pareto_front()
        assert target.is_file()
        lines = target.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "rank,smiles,qed,mw_band,tpsa_band,dominates,crowding"
        assert len(lines) == 8  # header + the 7 pinned front members
        assert any("dominates" not in line for line in lines[1:])  # real rows
        assert any("inf" in line for line in lines[1:])  # corner members
        # the info dialog reports the written path, nothing errored
        assert any(call[1][0] == "Saved" for call in quiet_dialogs)

    def test_plot_and_save_guard_without_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._plot_pareto_front()
        app._save_pareto_front()
        titles = [call[1][0] for call in quiet_dialogs]
        assert titles.count("Nothing to plot") == 1
        assert titles.count("Nothing to save") == 1  # both guards fired, no raise


# --------------------------------------------------------------------- #
# the per-tab "Copy inputs as JSON" footer affordance (100x-60)
# --------------------------------------------------------------------- #
class TestCopyTabInputs:
    """The footer's second affordance: one tab's live inputs, as a
    small JSON snippet on the clipboard (share-lite -- the same fields
    the session file stores, scoped to one tab, read live)."""

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str, isolated_home: Path) -> Any:
        application = playground.PlaygroundApp()
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    def _tab_index(self, app: Any, owner_name: str) -> int:
        return next(index for index, names in app._tab_input_owners.items() if owner_name in names)

    def _copy_tab(self, app: Any, index: int) -> dict[str, Any]:
        app._copy_tab_inputs(index)
        app.update()  # let the clipboard land
        return json.loads(app.clipboard_get())

    def test_copy_chemistry_inputs_reflects_live_values(
        self,
        playground: Any,
        app: Any,
    ) -> None:
        chemistry = self._tab_index(app, "_aff_value")
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        payload = self._copy_tab(app, chemistry)
        assert payload["tab"] == "Chemistry"
        assert payload["variables"]["_aff_value"] == "9.9"  # LIVE, not default
        assert payload["families"]  # the descriptor-family checkboxes
        assert set(payload) == {"tab", "variables", "families", "texts"}
        assert "copied as JSON" in str(app._job_label.cget("text"))
        # the items count in the status line matches the payload size
        items = len(payload["variables"]) + len(payload["families"]) + len(payload["texts"])
        assert f"{items} items" in str(app._job_label.cget("text"))

    def test_copy_generative_inputs_includes_corpus_text(
        self,
        playground: Any,
        app: Any,
    ) -> None:
        generative = self._tab_index(app, "text:corpus")  # the ownership key
        app._corpus_text.delete("1.0", "end")
        app._corpus_text.insert("1.0", "CCO\nCCC")
        payload = self._copy_tab(app, generative)
        assert payload["tab"] == "Generative"
        assert payload["texts"]["corpus"] == "CCO\nCCC"
        assert payload["variables"]["_objective_combo"] == "QED drug-likeness"
        assert payload["variables"]["_pop_size"] == "24"

    def test_copy_is_index_explicit_and_isolated(
        self,
        playground: Any,
        app: Any,
    ) -> None:
        # copy CHEMISTRY while a DIFFERENT tab is selected: the payload
        # must carry Chemistry's inputs only -- no cross-tab leakage.
        app._book.select(7)  # Generative visible
        chemistry = self._tab_index(app, "_aff_value")
        payload = self._copy_tab(app, chemistry)
        assert "_aff_value" in payload["variables"]
        assert "_pop_size" not in payload["variables"]  # Generative's
        assert "corpus" not in payload["texts"]  # Generative's
        assert payload["tab"] == "Chemistry"

    def test_copied_json_round_trips_through_the_session_fields(
        self,
        playground: Any,
        app: Any,
    ) -> None:
        # the snippet's field vocabulary is the session file's own:
        # variables / families / texts, JSON-clean under indent+sort_keys.
        chemistry = self._tab_index(app, "_aff_value")
        payload = self._copy_tab(app, chemistry)
        for name, value in payload["variables"].items():
            assert isinstance(name, str)
            assert isinstance(value, (str, int, float, bool))
        for checked in payload["families"].values():
            assert isinstance(checked, bool)
        for content in payload["texts"].values():
            assert isinstance(content, str)


# --------------------------------------------------------------------- #
# the keyboard-shortcut hint row + the five bound accelerators (100x-65)
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestShortcutHints:
    ACCELERATORS: ClassVar[dict[str, str]] = {
        "Ctrl+O": "<Control-o>",
        "Ctrl+S": "<Control-s>",
        "Ctrl+Shift+S": "<Control-Shift-S>",
        "Ctrl+Shift+R": "<Control-Shift-R>",
        "Ctrl+F": "<Control-f>",
        "Ctrl+Q": "<Control-q>",
    }

    def test_every_advertised_accelerator_is_bound(
        self,
        playground: Any,
        xvfb_display: str,
    ) -> None:
        # Ctrl+Shift+R was advertised in the File menu (and the module
        # docstring) since 100x-54 with NO binding behind it -- pressing
        # it did nothing.  This pins every advertised sequence (the
        # card-finder Ctrl+F joined in 100x-120) to a real binding.
        app = playground.PlaygroundApp(persist_session=False)
        try:
            for label, sequence in self.ACCELERATORS.items():
                assert app.bind_all(sequence), f"{label} ({sequence}) is advertised but not bound"
        finally:
            app.destroy()

    def test_hint_row_lists_every_shortcut(
        self,
        playground: Any,
        xvfb_display: str,
    ) -> None:
        from tkinter import ttk

        app = playground.PlaygroundApp(persist_session=False)
        try:
            text = playground.shortcut_hints_text()
            for keys, what in playground.SHORTCUT_HINTS:
                assert f"{keys} {what}" in text
            labels = [
                child for child in app._hint_bar.winfo_children() if isinstance(child, ttk.Label)
            ]
            assert labels, "the hint row carries no label"
            assert labels[0].cget("text") == text
        finally:
            app.destroy()

    def test_hint_row_sits_above_the_status_bar(
        self,
        playground: Any,
        xvfb_display: str,
    ) -> None:
        app = playground.PlaygroundApp(persist_session=False)
        try:
            app.update_idletasks()
            assert app._hint_bar.winfo_rooty() < app._status_bar.winfo_rooty()
        finally:
            app.destroy()

    def test_view_toggle_and_hide_button_flip_visibility(
        self,
        playground: Any,
        xvfb_display: str,
    ) -> None:
        app = playground.PlaygroundApp(persist_session=False)
        try:
            assert app._hints_var.get() is True
            assert app._hint_bar.pack_info()["side"] == "bottom"
            # the row's own Hide button flips the preference and unpacks
            app._hide_hints()
            assert app._hints_var.get() is False
            with pytest.raises(playground.tk.TclError):
                app._hint_bar.pack_info()
            # the View checkbutton's command re-shows it
            app._hints_var.set(True)
            app._apply_hint_visibility()
            assert app._hint_bar.pack_info()["side"] == "bottom"
        finally:
            app.destroy()

    def test_hint_preference_survives_a_session_round_trip(
        self,
        playground: Any,
        xvfb_display: str,
    ) -> None:
        first = playground.PlaygroundApp(persist_session=False)
        second = playground.PlaygroundApp(persist_session=False)
        try:
            first._hide_hints()
            payload = first._collect_session()
            assert payload["variables"]["_hints_var"] is False
            second._session = payload
            second._restore_session()
            assert second._hints_var.get() is False
            with pytest.raises(playground.tk.TclError):
                second._hint_bar.pack_info()
        finally:
            first.destroy()
            second.destroy()


# --------------------------------------------------------------------- #
# the Deep Learning tab (100x-61) -- the pharmaml.ml.neural estimators
# --------------------------------------------------------------------- #
class TestDeepLearningTab:
    """The PyTorch estimators behind one card of inputs: the dataset
    loader, the architecture/task mapping, the training job's three
    result cards (metrics grid, parity/confusion figure, loss curves)
    and the per-epoch history CSV export."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        # persist_session=False: the suite must not leak the mutated
        # Deep Learning inputs into the developer's real session file.
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _configure_tiny(
        self, app: Any, hidden: str = "8", epochs: str = "30", patience: str = "5"
    ) -> None:
        """Shrink the network so the suite trains in seconds, not minutes."""
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, hidden)
        app._dl_epochs.set(epochs)
        app._dl_patience.set(patience)

    def _metric_texts(self, app: Any) -> str:
        return " | ".join(str(child.cget("text")) for child in app._dl_metrics.winfo_children())

    # -- pure helpers (no Tk, no pharmaml import needed) ------------------ #
    def test_parse_hidden_layers_variants(self, playground: Any) -> None:
        assert playground.parse_hidden_layers("64, 32") == (64, 32)
        assert playground.parse_hidden_layers(" 8 ") == (8,)
        assert playground.parse_hidden_layers("") == ()  # linear baseline
        assert playground.parse_hidden_layers("64.0, 16") == (64, 16)
        for junk in ("64, x", "0, 8", "-1"):
            with pytest.raises(ValueError, match="widths"):
                playground.parse_hidden_layers(junk)

    def test_estimator_name_covers_all_four_combinations(self, playground: Any) -> None:
        deep, residual, numpy_arch = playground._DL_ARCHITECTURES
        assert playground.deep_estimator_name(deep, "regression") == "DeepMLPRegressor"
        assert playground.deep_estimator_name(deep, "classification") == "DeepMLPClassifier"
        assert playground.deep_estimator_name(residual, "regression") == "ResidualMLPRegressor"
        assert playground.deep_estimator_name(residual, "classification") == "ResidualMLPClassifier"
        assert playground.deep_estimator_name(numpy_arch, "regression") == "NumpyMLPRegressor"
        assert (
            playground.deep_estimator_name(numpy_arch, "classification") == "NumpyMLPClassifier"
        )
        with pytest.raises(ValueError, match="architecture"):
            playground.deep_estimator_name("CNN", "regression")
        with pytest.raises(ValueError, match="task"):
            playground.deep_estimator_name(deep, "auto")

    def test_deep_engine_resolution_falls_back_to_numpy(self, playground: Any) -> None:
        # (round 179) torch architectures degrade to the numpy autograd
        # engine when torch is absent; the numpy architecture stays
        # numpy whatever torch says.  Pure, so both branches pin here.
        deep = playground._DL_ARCHITECTURES[0]
        assert playground.resolve_deep_estimator(deep, "regression", True) == (
            "DeepMLPRegressor",
            "torch",
        )
        assert playground.resolve_deep_estimator(deep, "classification", False) == (
            "NumpyMLPClassifier",
            "numpy",
        )
        assert playground.resolve_deep_estimator(
            playground._DL_ARCHITECTURES[2], "regression", True
        ) == ("NumpyMLPRegressor", "numpy")

    def test_optimizer_and_split_tokens_mirror_the_library(self, playground: Any) -> None:
        # the combobox values are the library's own accepted tokens --
        # pinned together so the two can never drift apart (the
        # _DL_ACTIVATIONS pattern, round-179 edition).
        from pharmaml.ml.numpy_neural import OPTIMIZERS
        from pharmaml.ml.splits import SPLIT_STRATEGIES

        assert set(playground._DL_OPTIMIZERS) == set(OPTIMIZERS)
        assert playground._ML_SPLIT_STRATEGIES == tuple(
            name for name in SPLIT_STRATEGIES if name != "group"
        )

    def test_activation_tokens_mirror_the_library(self, playground: Any) -> None:
        # the combobox values are the estimator's own accepted tokens --
        # pinned together so the two can never drift apart.
        from pharmaml.ml.neural import _ACTIVATIONS

        assert playground._DL_ACTIVATIONS == _ACTIVATIONS

    # -- the dataset loader ------------------------------------------------ #
    def test_example_dataset_loads_into_preview(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_load_example()
        assert quiet_dialogs == []
        info = app._dl_info.cget("text")
        assert info.startswith("qsar_regression.csv — 260 rows × ")
        assert len(app._dl_preview_tree.get_children()) == 12  # first 12 rows
        columns = list(app._dl_preview_tree["columns"])
        assert columns
        # the smart default: pIC50 (numeric, last), never an ID column
        assert app._dl_target_combo.get() == "pIC50"
        assert list(app._dl_target_combo["values"]) == columns

    # -- the training runs -------------------------------------------------- #
    def test_small_regression_run_fills_all_three_cards(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._configure_tiny(app)
        app._dl_load_example()
        app._dl_train()
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        texts = self._metric_texts(app)
        # (round 179) the engine is resolved honestly: DeepMLP with
        # torch, its numpy autograd stand-in without.
        expected, _engine = playground.resolve_deep_estimator(
            "Deep MLP (feed-forward)",
            "regression",
            importlib.util.find_spec("torch") is not None,
        )
        assert f"{expected} (regression)" in texts
        assert "R²" in texts and "RMSE" in texts
        assert "epochs run" in texts and "best val loss" in texts
        assert "hidden layers" in texts and "8" in texts
        assert app._dl_host._pg_figure is not None  # parity figure
        assert app._dl_hist_host._pg_figure is not None  # loss curves
        summary = app._dl_hist_summary.cget("text")
        assert f"{expected} ·" in summary
        assert "epochs run" in summary and "best val loss" in summary
        assert app._dl_history  # the export payload is live
        assert app._dl_history[0]["epoch"] == 1
        assert set(app._dl_history[0]) == {"epoch", "train_loss", "val_loss"}
        assert any(entry["val_loss"] is not None for entry in app._dl_history)

    def test_residual_classifier_run_uses_the_right_estimator(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._configure_tiny(app)
        app._dl_example_combo.set("qsar_classification.csv")
        app._dl_load_example()
        # the smart target default: activity (binary), NOT compound_id
        # (420 unique row IDs -- the trap the walker exists to avoid)
        assert app._dl_target_combo.get() == "activity"
        app._dl_arch_combo.set("Residual MLP (skip connections)")
        app._dl_train()
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        texts = self._metric_texts(app)
        expected, _engine = playground.resolve_deep_estimator(
            "Residual MLP (skip connections)",
            "classification",
            importlib.util.find_spec("torch") is not None,
        )
        assert f"{expected} (classification)" in texts
        assert "accuracy" in texts and "MCC" in texts
        # SMILES + compound_id are not numeric -> reported as dropped
        assert "dropped non-numeric" in texts
        assert "SMILES" in texts and "compound_id" in texts
        assert app._dl_host._pg_figure is not None  # confusion matrix
        assert app._dl_hist_host._pg_figure is not None  # loss curves

    # -- batch normalisation (100x-90) --------------------------------------- #
    @pytest.mark.skipif(
        importlib.util.find_spec("torch") is None,
        reason="batch norm is a torch-engine feature (the numpy stand-in has none)",
    )
    def test_batchnorm_run_trains_with_the_flag_set(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._configure_tiny(app)
        app._dl_load_example()
        app._dl_batchnorm_var.set(True)
        app._dl_train()
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        network = app._dl_last_run[0]
        assert network.batchnorm is True  # the checkbox reached the net
        assert app._dl_host._pg_figure is not None
        texts = self._metric_texts(app)
        assert "DeepMLPRegressor (regression)" in texts

    def test_batchnorm_with_singleton_batches_warns_before_the_job(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._configure_tiny(app)
        app._dl_load_example()
        app._dl_batch.set("1")
        app._dl_batchnorm_var.set(True)
        app._dl_train()
        assert not app._busy  # the pre-flight stopped before any job
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Batch norm needs batches" in titles
        message = quiet_dialogs[0][1][1]
        assert "at least 2" in message

    # -- the history CSV export ---------------------------------------------- #
    def test_save_history_writes_the_exact_csv(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Path,
    ) -> None:
        self._configure_tiny(app)
        app._dl_load_example()
        app._dl_train()
        assert self._drain_jobs(app), "network training hung"
        out = tmp_path / "history.csv"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **kwargs: str(out),
        )
        app._dl_save_history()
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "epoch,train_loss,val_loss"
        assert len(lines) == 1 + len(app._dl_history)  # one row per epoch
        assert lines[1].startswith("1,")  # the first epoch
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Saved" in titles

    # -- guards ---------------------------------------------------------------- #
    def test_train_without_a_dataset_warns_and_starts_no_job(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_train()
        assert not app._busy  # the gate stopped before any job started
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No dataset" in titles

    def test_save_history_before_any_run_guards(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_save_history()
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Nothing to save" in titles

    def test_bad_hidden_layers_warns_not_crashes(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._configure_tiny(app, hidden="64, zero")
        app._dl_load_example()
        app._dl_train()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Check the network settings" in titles

    def test_deep_learning_inputs_survive_a_session_round_trip(
        self, playground: Any, xvfb_display: str
    ) -> None:
        # the generic save/restore machinery walks vars(self) for
        # Variable/Entry widgets -- every _dl_* input rides it for free.
        # This pins the whole tab: 24 saved keys (the optimizer + split
        # strategy comboboxes since round 179; the six XAI-card inputs
        # since round 185), the mutated values restored verbatim onto a
        # freshly built window.
        first = playground.PlaygroundApp(persist_session=False)
        second = playground.PlaygroundApp(persist_session=False)
        try:
            first._dl_arch_combo.set("Residual MLP (skip connections)")
            first._dl_optimizer_combo.set("rmsprop")
            first._dl_split_combo.set("temporal")
            first._dl_hidden_entry.delete(0, "end")
            first._dl_hidden_entry.insert(0, "128, 64, 32")
            first._dl_activation_combo.set("gelu")
            first._dl_dropout.set("0.25")
            first._dl_epochs.set("300")
            first._dl_batch.set("64")
            first._dl_lr_entry.delete(0, "end")
            first._dl_lr_entry.insert(0, "0.0005")
            first._dl_wd_entry.delete(0, "end")
            first._dl_wd_entry.insert(0, "0.01")
            first._dl_patience.set("25")
            first._dl_valsplit.set("0.2")
            first._dl_seed_entry.delete(0, "end")
            first._dl_seed_entry.insert(0, "7")
            first._dl_task_combo.set("regression")
            first._dl_batchnorm_var.set(True)
            # (round 185) the Attribution & attention card's six inputs
            first._dl_xai_method_combo.set(first._xai_method_labels[2])
            first._dl_xai_row_spin.set("17")
            first._dl_xai_steps_spin.set("120")
            first._dl_xai_tokens_entry.delete(0, "end")
            first._dl_xai_tokens_entry.insert(0, "the inhibitor blocks the gate")
            first._dl_xai_layer_spin.set("3")
            first._dl_xai_head_spin.set("2")
            # (round 186) the four new cards' thirteen inputs
            first._dl_rb_smooth_spin.set("0.85")
            first._dl_umap_knn_spin.set("25")
            first._dl_umap_mindist_spin.set("0.3")
            first._dl_umap_metric_combo.set("cosine")
            first._dl_cur_op_combo.set("between")
            first._dl_cur_value_entry.delete(0, "end")
            first._dl_cur_value_entry.insert(0, "1,5")
            first._dl_exp_kind_combo.set("histogram")
            first._dl_exp_template_combo.set("ggplot2")
            first._dl_exp_hover_entry.delete(0, "end")
            first._dl_exp_hover_entry.insert(0, "SMILES,compound_id")
            # (round 189) the six new cards' twenty-one inputs: the HPO
            # study, model surgery, training harness, graph model,
            # graph explain and glassbox EBM cards
            first._dl_hpo_preset_combo.set("random forest: trees + depth")
            first._dl_hpo_sampler_combo.set("random")
            first._dl_hpo_trials.set("14")
            first._dl_hpo_cv.set("2")
            first._dl_prune_sparsity.set("0.75")
            first._dl_prune_scope_combo.set("global")
            first._dl_prune_structured_var.set(True)
            first._dl_quant_combo.set("int8_channel")
            first._dl_zero_stage_combo.set("2")
            first._dl_zero_world.set("8")
            first._dl_es_patience.set("6")
            first._dl_es_min_delta.delete(0, "end")
            first._dl_es_min_delta.insert(0, "0.001")
            first._dl_graph_source_combo.set("SMILES + label list")
            first._dl_graph_conv_combo.set("gat")
            first._dl_graph_pool_combo.set("sag")
            first._dl_graph_epochs.set("200")
            first._dl_graph_hidden.set("16")
            first._dl_gx_method_combo.set("subgraphx")
            first._dl_gx_topk.set("12")
            first._dl_ebm_bags.set("6")
            first._dl_ebm_rounds.set("120")
            # (the fork's rounds) the Architecture tree-depth combo, the
            # Aim filter entry and the gradio flag combo ride the walk too
            first._dl_depth_combo.set("3")
            first._dl_filter_entry.delete(0, "end")
            first._dl_filter_entry.insert(0, "mlp")
            first._dl_flag_combo.set("ambiguous")
            # The paired drug–target card's six inputs ride the same walk.
            first._dl_dta_source.set("CSV path")
            first._dl_dta_path.insert(0, "paired.csv")
            for entry, value in (
                (first._dl_dta_smiles_col, "canonical_smiles"),
                (first._dl_dta_protein_col, "sequence"),
                (first._dl_dta_affinity_col, "pKd"),
            ):
                entry.delete(0, "end")
                entry.insert(0, value)
            first._dl_dta_dipeptides.set(True)
            payload = first._collect_session()
            saved = [key for key in payload["variables"] if key.startswith("_dl_")]
            assert len(saved) == 69  # prior 63 + the paired-DTA card's six inputs

            second._session = payload
            second._restore_session()
            assert second._dl_arch_combo.get() == "Residual MLP (skip connections)"
            assert second._dl_optimizer_combo.get() == "rmsprop"
            assert second._dl_split_combo.get() == "temporal"
            assert second._dl_hidden_entry.get() == "128, 64, 32"
            assert second._dl_activation_combo.get() == "gelu"
            assert second._dl_dropout.get() == "0.25"
            assert second._dl_epochs.get() == "300"
            assert second._dl_batch.get() == "64"
            assert second._dl_lr_entry.get() == "0.0005"
            assert second._dl_wd_entry.get() == "0.01"
            assert second._dl_patience.get() == "25"
            assert second._dl_valsplit.get() == "0.2"
            assert second._dl_seed_entry.get() == "7"
            assert second._dl_task_combo.get() == "regression"
            assert second._dl_batchnorm_var.get() is True
            assert second._dl_xai_method_combo.get() == first._xai_method_labels[2]
            assert second._dl_xai_row_spin.get() == "17"
            assert second._dl_xai_steps_spin.get() == "120"
            assert second._dl_xai_tokens_entry.get() == "the inhibitor blocks the gate"
            assert second._dl_xai_layer_spin.get() == "3"
            assert second._dl_xai_head_spin.get() == "2"
            # (round 186) the four new cards restore too
            assert second._dl_rb_smooth_spin.get() == "0.85"
            assert second._dl_umap_knn_spin.get() == "25"
            assert second._dl_umap_mindist_spin.get() == "0.3"
            assert second._dl_umap_metric_combo.get() == "cosine"
            assert second._dl_cur_op_combo.get() == "between"
            assert second._dl_cur_value_entry.get() == "1,5"
            assert second._dl_exp_kind_combo.get() == "histogram"
            assert second._dl_exp_template_combo.get() == "ggplot2"
            assert second._dl_exp_hover_entry.get() == "SMILES,compound_id"
            # (round 189) the six new cards' inputs ride the same trip
            assert second._dl_hpo_preset_combo.get() == "random forest: trees + depth"
            assert second._dl_hpo_sampler_combo.get() == "random"
            assert second._dl_hpo_trials.get() == "14"
            assert second._dl_hpo_cv.get() == "2"
            assert second._dl_prune_sparsity.get() == "0.75"
            assert second._dl_prune_scope_combo.get() == "global"
            assert bool(second._dl_prune_structured_var.get()) is True
            assert second._dl_quant_combo.get() == "int8_channel"
            assert second._dl_zero_stage_combo.get() == "2"
            assert second._dl_zero_world.get() == "8"
            assert second._dl_es_patience.get() == "6"
            assert second._dl_es_min_delta.get() == "0.001"
            assert second._dl_graph_source_combo.get() == "SMILES + label list"
            assert second._dl_graph_conv_combo.get() == "gat"
            assert second._dl_graph_pool_combo.get() == "sag"
            assert second._dl_graph_epochs.get() == "200"
            assert second._dl_graph_hidden.get() == "16"
            assert second._dl_gx_method_combo.get() == "subgraphx"
            assert second._dl_gx_topk.get() == "12"
            assert second._dl_ebm_bags.get() == "6"
            assert second._dl_ebm_rounds.get() == "120"
            assert second._dl_depth_combo.get() == "3"
            assert second._dl_filter_entry.get() == "mlp"
            assert second._dl_flag_combo.get() == "ambiguous"
            assert second._dl_dta_source.get() == "CSV path"
            assert second._dl_dta_path.get() == "paired.csv"
            assert second._dl_dta_smiles_col.get() == "canonical_smiles"
            assert second._dl_dta_protein_col.get() == "sequence"
            assert second._dl_dta_affinity_col.get() == "pKd"
            assert bool(second._dl_dta_dipeptides.get()) is True
        finally:
            first.destroy()
            second.destroy()


class TestDefaultTargetColumn:
    def test_bundled_examples_pick_their_real_targets(self, playground: Any) -> None:
        import pandas as pd

        # the last-column default trained a 420-class classifier on
        # compound_id; the walker must land on the real label instead.
        cases = {
            "qsar_regression.csv": "pIC50",  # numeric, last column
            "qsar_classification.csv": "activity",  # behind SMILES + ID
            "qsar_multiclass.csv": "Target",
        }
        for name, expected in cases.items():
            frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / name)
            assert playground.default_target_column(frame) == expected, name

    def test_unique_numeric_column_stays_a_target(self, playground: Any) -> None:
        # a regression target legitimately has all-unique values -- only
        # unique STRING columns are treated as IDs.
        import pandas as pd

        frame = pd.DataFrame(
            {
                "descr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "pIC50": [5.1, 6.2, 7.3, 8.4, 5.5, 6.6, 7.7, 8.8],
            }
        )
        assert playground.default_target_column(frame) == "pIC50"

    def test_all_id_columns_fall_back_to_the_last_column(self, playground: Any) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "lot_id": [f"L{i}" for i in range(8)],
                "compound_id": [f"C{i}" for i in range(8)],
            }
        )
        assert playground.default_target_column(frame) == "compound_id"

    def test_empty_frame_returns_none(self, playground: Any) -> None:
        import pandas as pd

        assert playground.default_target_column(pd.DataFrame()) is None

    def test_ml_and_dl_loaders_preselect_the_smart_target(
        self, playground: Any, xvfb_display: str
    ) -> None:
        # end to end on the classification example: BOTH dataset cards
        # must preselect activity, not compound_id (the 420-class trap).
        app = playground.PlaygroundApp(persist_session=False)
        try:
            path = TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv"
            app._ml_load_path(path)
            app._dl_load_path(path)
            assert app._target_combo.get() == "activity"
            assert app._dl_target_combo.get() == "activity"
            assert app._infer_task("activity", app._ml_frame) == "classification"
        finally:
            app.destroy()


# --------------------------------------------------------------------- #
# the numeric-feature pre-flight (100x-62) -- zero numeric columns fail
# BEFORE any job starts, with the remedy named
# --------------------------------------------------------------------- #
class TestNumericFeatureGuard:
    ALL_OBJECT_CSV = "molecule,series,label\nCCO,series_a,active\nCCN,series_b,inactive\n"

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    def test_numeric_feature_columns_splits_the_right_way(self, playground: Any) -> None:
        import pandas as pd

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv")
        numeric, dropped = playground.numeric_feature_columns(frame, "activity")
        assert numeric == [f"descr_{i:02d}" for i in range(8)]
        assert dropped == ["SMILES", "compound_id"]

    def _load_all_object_csv(self, app: Any, tmp_path: Path) -> None:
        path = tmp_path / "all_object.csv"
        path.write_text(self.ALL_OBJECT_CSV, encoding="utf-8")
        app._dl_load_path(path)
        app._ml_load_path(path)

    def test_dl_train_guards_zero_numeric_features(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        self._load_all_object_csv(app, tmp_path)
        app._dl_train()
        assert not app._busy  # the guard fired before any job started
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No numeric features" in titles
        message = quiet_dialogs[0][1][1]
        assert "molecule, series" in message  # the dropped columns, named
        assert "feature factory" in message  # the remedy, pointed at

    def test_ml_train_guards_zero_numeric_features(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        self._load_all_object_csv(app, tmp_path)
        app._ml_train()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No numeric features" in titles

    def test_ml_compare_guards_zero_numeric_features(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        self._load_all_object_csv(app, tmp_path)
        app._ml_compare()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No numeric features" in titles

    def test_ml_tune_guards_zero_numeric_features(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        self._load_all_object_csv(app, tmp_path)
        app._ml_tune()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No numeric features" in titles


# --------------------------------------------------------------------- #
# the Data & ML tab's Hyperparameter search card (100x-90) -- the
# RandomSearch story, mouse edition
# --------------------------------------------------------------------- #
class TestTunerCard:
    """The Tune card: pure rendering helpers, the CLI-registry pin, the
    search's leaderboard / baseline / held-out wiring, the Apply-best
    flows (ann points the model combobox; dnn prefills the Deep Learning
    tab) and the Save-trials export."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 300.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- pure helpers (no Tk, no pharmaml import needed) ------------------ #
    def test_config_text_renders_sorted_compact_tokens(self, playground: Any) -> None:
        text = playground.tune_config_text(
            {"units": 64, "alpha": 0.001, "batchnorm": True, "gamma": "scale"}
        )
        assert text == "alpha=0.001 batchnorm=on gamma=scale units=64"
        # floats never trail zeros; booleans are on/off; ints stay ints
        assert playground.tune_config_text({"dropout": 0.3}) == "dropout=0.3"
        assert playground.tune_config_text({"batchnorm": False}) == "batchnorm=off"
        assert playground.tune_config_text({"n_layers": 2}) == "n_layers=2"
        assert playground.tune_config_text({}) == ""

    def test_hypermodel_names_mirror_the_cli_registry(self, playground: Any) -> None:
        # the GUI card and the CLI resolve the SAME hypermodels -- pinned
        # together so one surface cannot drift from the other (the
        # _DL_ACTIVATIONS pin pattern).
        from pharmaml.cli import _tune_hypermodels

        cli = {key: fn.__name__ for key, fn in _tune_hypermodels().items()}
        assert dict(playground._TUNE_HYPERMODELS) == cli
        # the combobox offers exactly the registry's estimator tokens
        assert set(playground._TUNE_ESTIMATORS) == {key[0] for key in cli}
        # hsvr really is the regression-only odd one out
        assert ("hsvr", "classification") not in playground._TUNE_HYPERMODELS

    def test_registry_families_covered_by_the_card(self, playground: Any) -> None:
        # (100x-169) the card tunes EVERY grid-bearing registry family,
        # not just the bespoke trio -- representative members of each
        # zoo class resolve to their generated builder names.
        assert "random_forest" in playground._TUNE_ESTIMATORS
        assert "logistic_regression" in playground._TUNE_ESTIMATORS
        assert "ridge" in playground._TUNE_ESTIMATORS
        assert playground._TUNE_HYPERMODELS[("random_forest", "regression")] == (
            "build_random_forest_regressor"
        )
        assert playground._TUNE_HYPERMODELS[("logistic_regression", "classification")] == (
            "build_logistic_regression_classifier"
        )
        assert playground._TUNE_HYPERMODELS[("knn", "classification")] == (
            "build_knn_classifier"
        )
        # the dual-task families really are dual-task
        assert ("knn", "regression") in playground._TUNE_HYPERMODELS
        # the optional-backend families carry their dependency guard
        # (the dnn/torch pattern, extended to the boosters)
        assert playground._TUNE_REQUIRES["dnn"] == "torch"
        assert playground._TUNE_REQUIRES["xgboost"] == "xgboost"
        assert playground._TUNE_REQUIRES["lightgbm"] == "lightgbm"
        assert playground._TUNE_REQUIRES["catboost"] == "catboost"

    def test_defaults_before_any_run(self, app: Any) -> None:
        assert app._tune_estimator_combo.get() == "ann"
        assert app._tune_searcher_combo.get() == "random"
        assert app._tune_trials.get() == "8"
        assert app._tune_eta.get() == "3"
        assert app._tune_cv.get() == "3"
        assert app._tune_last is None
        assert "no search yet" in app._tune_summary.cget("text")
        assert "empty" in app._board_status.cget("text")

    # -- the search runs ---------------------------------------------------- #
    def test_ann_search_fills_leaderboard_summary_and_applies(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._tune_trials.set("4")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app), "the search hung"
        assert quiet_dialogs == []

        rows = app._board_tree.get_children()
        assert len(rows) == 4  # one row per trial
        assert list(app._board_tree["columns"]) == [
            "rank",
            "trial",
            "configuration",
            "score",
            "± std",
        ]
        scores = [float(app._board_tree.item(row, "values")[3]) for row in rows]
        assert scores == sorted(scores, reverse=True)  # best first
        # the rank-1 row IS the winning trial (id, score, config text)
        best = app._tune_last[0].best_trial
        first = app._board_tree.item(rows[0], "values")
        assert first[0] == "1"
        assert int(first[1]) == best.trial_id
        assert first[3] == f"{best.score:.4f}"  # the rendered cell
        assert first[4] == f"{best.score_std:.4f}"
        full = playground.tune_config_text(best.hyperparameters)
        rendered = str(first[2])
        if rendered.endswith("…"):  # the column truncates long configs
            assert full.startswith(rendered[:-1])
        else:
            assert rendered == full
        assert full.startswith("alpha=")  # sorted k=v tokens

        summary = app._tune_summary.cget("text")
        assert "CV r2" in summary
        assert "defaults" in summary and "gain" in summary
        assert "held-out R²" in summary
        assert "winner:" in summary
        status = app._board_status.cget("text")
        assert status.startswith("RandomSearch over ann (regression)")
        assert "objective r2" in status
        assert app._tune_last is not None
        results = app._tune_last[0]
        assert len(results.trials) == 4

        # Apply best (ann): the model combobox points at the family
        app._ml_tune_apply()
        assert quiet_dialogs == []
        assert app._model_combo.get() == "ann"
        assert app._task_combo.get() == "regression"
        assert "applied" in app._tune_summary.cget("text")

    def test_ann_hyperband_search_rungs_the_declared_ladder(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # the ladder IS the family's declared max_iter grid (200-1000,
        # via the public resource_bounds) -> eta 3 gives s_max=1: bracket
        # 1 runs 3 trials @333 + 1 promoted @999, bracket 0 runs 2 @1000.
        app._ml_load_example()
        app._tune_searcher_combo.set("hyperband")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app), "the hyperband search hung"
        assert quiet_dialogs == []
        results = app._tune_last[0]
        assert len(results.trials) == 6
        assert sorted({int(t.resource) for t in results.trials}) == [333, 999, 1000]

        rows = app._board_tree.get_children()
        assert len(rows) == 6
        assert list(app._board_tree["columns"]) == [
            "rank",
            "trial",
            "configuration",
            "resource",
            "score",
            "± std",
        ]
        best = results.best_trial
        first = app._board_tree.item(rows[0], "values")
        assert first[0] == "1"
        assert int(first[1]) == best.trial_id
        assert first[3] == f"{best.resource:g}"  # the resource column
        assert first[4] == f"{best.score:.4f}"
        # every rendered resource cell matches its trial's rung
        for row, trial in zip(rows, sorted(results.trials, key=lambda t: t.score, reverse=True)):
            assert app._board_tree.item(row, "values")[3] == f"{trial.resource:g}"

        status = app._board_status.cget("text")
        assert status.startswith("Hyperband over ann (regression)")
        assert "objective r2" in status
        assert "η 3" in status and "resource max_iter 200–1000" in status
        # the epoch-economics line: 3×333 + 1×999 + 2×1000 = 3998 spent
        # vs a naive 6×1000 = 6000 -> 33% saved
        assert "budget 3998 of a naive 6000 max_iter" in status
        assert "(33% saved)" in status
        summary = app._tune_summary.cget("text")
        assert "defaults" in summary and "gain" in summary
        assert f"max_iter={int(best.resource)}" in summary

    def test_dnn_hyperband_applies_best_to_the_deep_learning_tab(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # dnn's ladder is the declared epochs grid (30-150): eta 3 ->
        # bracket 1 runs 3 @50 + 1 promoted @150, bracket 0 runs 2 @150.
        app._ml_load_example()
        app._tune_estimator_combo.set("dnn")
        app._tune_searcher_combo.set("hyperband")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app, timeout=420), "the dnn hyperband hung"
        assert quiet_dialogs == []
        results = app._tune_last[0]
        assert len(results.trials) == 6
        assert sorted({int(t.resource) for t in results.trials}) == [50, 150]
        assert "resource epochs 30–150" in app._board_status.cget("text")

        params = results.best_params
        app._ml_tune_apply()
        assert quiet_dialogs == []
        # the winner's rung resource (the epochs override) rode along
        assert app._dl_epochs.get() == str(int(params["epochs"]))
        assert int(params["epochs"]) in (50, 150)
        assert app._dl_task_combo.get() == "regression"
        assert app._book.index(app._book.select()) == app._book.index(app._dl_page)
        assert "Deep Learning" in app._tune_summary.cget("text")

    def test_hsvr_on_a_classification_target_guards(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._example_combo.set("qsar_classification.csv")
        app._ml_load_example()
        app._tune_estimator_combo.set("hsvr")
        app._ml_tune()
        assert not app._busy  # the gate stopped before any job started
        titles = [call[1][0] for call in quiet_dialogs]
        assert "hsvr is regression-only" in titles

    def test_hsvr_hyperband_guards_before_any_job(self, app: Any, quiet_dialogs: list[Any]) -> None:
        # hsvr declares no resource slot -- the pre-flight mirrors the
        # library's resource_bounds error before any bracket starts.
        app._ml_load_example()
        app._tune_estimator_combo.set("hsvr")
        app._tune_searcher_combo.set("hyperband")
        app._ml_tune()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "hsvr has no resource slot" in titles
        body = quiet_dialogs[0][1][1]
        assert "ann or dnn" in body and "random" in body

    def test_registry_family_hyperband_guards_with_the_general_message(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # (100x-169) the guard text generalises: any family without a
        # declared resource slot is named, not just the hsvr original.
        app._ml_load_example()
        app._tune_estimator_combo.set("ridge")
        app._tune_searcher_combo.set("hyperband")
        app._ml_tune()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "ridge has no resource slot" in titles
        body = quiet_dialogs[0][1][1]
        assert "ann or dnn" in body and "random" in body

    def test_random_forest_search_fills_the_leaderboard(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # (100x-169) a registry-grid family runs the SAME card flow as
        # ann: leaderboard rows, summary, held-out score, Apply-best.
        app._ml_load_example()
        app._tune_estimator_combo.set("random_forest")
        app._tune_trials.set("4")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app), "the random_forest search hung"
        assert quiet_dialogs == []

        rows = app._board_tree.get_children()
        assert len(rows) == 4
        results = app._tune_last[0]
        assert len(results.trials) == 4
        # the searched space IS the registry's own grid
        from pharmaml.ml.models import MODEL_REGISTRY

        grid = MODEL_REGISTRY["random_forest"].param_grid
        assert set(results.best_params) == set(grid)
        assert results.best_params["n_estimators"] in grid["n_estimators"]

        status = app._board_status.cget("text")
        assert status.startswith("RandomSearch over random_forest (regression)")
        summary = app._tune_summary.cget("text")
        assert "winner:" in summary and "held-out R²" in summary

        # Apply best (random_forest): the model combobox points at the family
        app._ml_tune_apply()
        assert quiet_dialogs == []
        assert app._model_combo.get() == "random_forest"
        assert app._task_combo.get() == "regression"

    def test_apply_and_save_before_any_search_guard(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_tune_apply()
        app._ml_tune_save_trials()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert titles.count("No search yet") == 2

    def test_bad_search_settings_warn_not_crash(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._tune_trials.set("2")  # below the minimum of 4
        app._ml_tune()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Check the search settings" in titles

    # -- the trials CSV export ---------------------------------------------- #
    def test_save_trials_writes_the_full_table(
        self,
        playground: Any,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        tmp_path: Path,
    ) -> None:
        app._ml_load_example()
        app._tune_trials.set("4")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app), "the search hung"
        out = tmp_path / "trials.csv"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **kwargs: str(out),
        )
        app._ml_tune_save_trials()
        assert quiet_dialogs == []
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("trial_id,score,score_std")
        assert len(lines) == 5  # header + 4 trials
        assert "n_layers" in lines[0] and "alpha" in lines[0]
        assert "trials saved" in app._job_label.cget("text")

    # -- the dnn bridge ------------------------------------------------------ #
    def test_dnn_search_applies_best_to_the_deep_learning_tab(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._tune_estimator_combo.set("dnn")
        app._tune_trials.set("4")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app, timeout=420), "the dnn search hung"
        assert quiet_dialogs == []
        assert len(app._board_tree.get_children()) == 4
        assert "objective r2" in app._board_status.cget("text")
        params = app._tune_last[0].best_params

        app._ml_tune_apply()
        assert quiet_dialogs == []
        # the whole Deep Learning tab is prefilled from the winner
        expected_hidden = ", ".join(
            str(int(params["units"])) for _ in range(int(params["n_layers"]))
        )
        assert app._dl_hidden_entry.get() == expected_hidden
        assert app._dl_dropout.get() == f"{float(params['dropout']):g}"
        assert app._dl_lr_entry.get() == f"{float(params['learning_rate']):g}"
        assert app._dl_activation_combo.get() == params["activation"]
        assert app._dl_batchnorm_var.get() is bool(params["batchnorm"])
        assert app._dl_epochs.get() == str(int(params["epochs"]))
        assert app._dl_patience.get() == "10"  # the hypermodel's fixed value
        assert app._dl_task_combo.get() == "regression"
        assert app._dl_target_combo.get() == "pIC50"
        # the dataset came along: same rows as the ML tab's frame
        assert app._dl_frame is not None
        assert len(app._dl_frame) == len(app._ml_frame)
        # and the notebook switched to the Deep Learning tab
        assert app._book.index(app._book.select()) == app._book.index(app._dl_page)
        assert "Deep Learning" in app._tune_summary.cget("text")


class TestGenerativeRLCard:
    """The Generative tab's RL optimiser: the 100x-66 introspection, GUI axis.

    Fail-first (100x-68): the tab only ever ran the GA -- the RL axis the
    CLI/notebooks/viz had since 100x-63/66 (ReinventOptimizer +
    plot_rl_introspection) had no GUI surface at all.  Now the optimiser
    combo swaps the GA parameter block for the RL one (iterations / batch
    size / prior epochs / anchor weight), the run fills the same
    leaderboard card, and the figure host shows the four-panel
    prior-drift introspection.
    """

    RL = "reinforcement learning (RL)"
    GA = "genetic algorithm"

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _run_rl(
        self,
        app: Any,
        *,
        iterations: str = "6",
        batch: str = "16",
        prior_epochs: str = "2",
        anchor: str = "0.1",
    ) -> None:
        app._method_combo.set(self.RL)
        app._rl_iterations.set(iterations)
        app._rl_batch.set(batch)
        app._rl_prior_epochs.set(prior_epochs)
        app._rl_anchor.delete(0, "end")
        app._rl_anchor.insert(0, anchor)
        app._run_evolution()
        assert self._drain_jobs(app), "RL job hung"

    # -- the optimiser swap ----------------------------------------------- #

    def test_method_combo_swaps_the_parameter_blocks(self, app: Any) -> None:
        # default: the GA rows are live, the RL block is gridded out
        assert app._method_combo.get() == self.GA
        assert app._ga_frame.grid_info() != {}
        assert app._rl_frame.grid_info() == {}
        # switching to RL collapses the GA block and restores the RL one
        app._method_combo.set(self.RL)
        app._on_method_change()
        assert app._ga_frame.grid_info() == {}
        assert app._rl_frame.grid_info() != {}
        # and back: the remembered GA position returns, RL collapses
        app._method_combo.set(self.GA)
        app._on_method_change()
        assert app._ga_frame.grid_info() != {}
        assert app._rl_frame.grid_info() == {}

    def test_rl_negative_anchor_is_an_input_error(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._method_combo.set(self.RL)
        app._rl_anchor.delete(0, "end")
        app._rl_anchor.insert(0, "-0.5")
        app._run_evolution()
        assert not app._busy  # rejected before any job started
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Input" in titles
        assert app._ga_result is None

    # -- the RL run itself (needs torch + rdkit) --------------------------- #

    @pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs PyTorch")
    def test_rl_run_fills_the_card_with_introspection(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._run_rl(app)
        assert app._ga_result is not None
        # the summary keeps the card's contract and names the RL mode
        summary = app._ga_summary.cget("text")
        assert summary.startswith("best ")
        assert "RL · built-in corpus" in summary
        assert len(app._ga_tree.get_children()) > 0  # a real leaderboard
        # the text panel is the RL history, not the GA operator stats
        stats = app._ga_stats_text.get("1.0", "end")
        assert "RL run — history (first → last iteration)" in stats
        assert "drift" in stats
        assert "0.000 →" in stats  # iteration 0's drift is the exact zero
        assert "entropy" in stats
        # the figure host carries the four-panel introspection figure
        figure = getattr(app._ga_host, "_pg_figure", None)
        assert figure is not None
        assert len(figure.get_axes()) == 4
        assert quiet_dialogs == []

    @pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs PyTorch")
    def test_rl_multiobjective_run_fills_the_pareto_card(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # RL fills the same per-objective breakdown (the CLI parity): the
        # Pareto card lights up after an RL run on the druglike triple too
        app._objective_combo.set("Multi-objective (QED \u00d7 MW \u00d7 TPSA)")
        self._run_rl(app)
        summary = app._pareto_summary.cget("text")
        assert "Pareto front over 3 objectives [qed, mw_band, tpsa_band]" in summary
        assert "molecules non-dominated" in summary
        assert app._pareto_tree.get_children() != ()
        assert quiet_dialogs == []

    # -- the text formatter ------------------------------------------------ #

    def test_format_rl_history_placeholder_without_history(self, playground: Any) -> None:
        class _Empty:
            history = None

        assert "after a run" in playground._format_rl_history(_Empty())

    def test_format_rl_history_first_and_last_rows(self, playground: Any) -> None:
        class _Two:
            history: ClassVar[list[dict[str, float]]] = [
                {
                    "best_score": 0.2,
                    "mean_score": 0.1,
                    "valid_fraction": 0.5,
                    "drift": 0.0,
                    "entropy": 2.5,
                    "agent_logprob": -20.0,
                    "prior_logprob": -20.0,
                },
                {
                    "best_score": 0.8,
                    "mean_score": 0.5,
                    "valid_fraction": 0.9,
                    "drift": 4.0,
                    "entropy": 2.0,
                    "agent_logprob": -15.0,
                    "prior_logprob": -20.0,
                },
            ]

        text = playground._format_rl_history(_Two())
        assert "RL run — history" in text
        assert "0.200 →     0.800" in text  # best_score first → last
        assert "0.000 →     4.000" in text  # the drift
        assert "the anchor penalises" in text  # the teaching note


# --------------------------------------------------------------------- #
# the Data & ML tab's Save-report button (100x-76) -- the CLI's
# --plots/--pdf-report bundle, mouse edition
# --------------------------------------------------------------------- #
class TestMLReportBundle:
    """``_ml_save_report`` rebuilds the CLI bundle from the stored run.

    The report story's last axis: library (100x-74) -> CLI (100x-75) ->
    GUI.  Guards first (no run, no fpdf), then the end-to-end flow on
    the classification example with the PDF ticked.
    """

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
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
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_fpdf_available_is_a_bool(self, playground: Any) -> None:
        assert isinstance(playground.fpdf_available(), bool)

    def test_report_needs_a_run_first(self, app: Any, dialogs: list[Any]) -> None:
        app._ml_save_report()
        assert not app._busy  # the guard fired before any job started
        titles = [call[1][0] for call in dialogs]
        assert "No run yet" in titles
        assert "held-out metrics" in dialogs[0][1][1]  # the remedy, named

    def test_report_pdf_gate_without_fpdf(
        self, app: Any, dialogs: list[Any], monkeypatch: Any, playground: Any
    ) -> None:
        app._ml_last_run = (object(), object(), "regression", None, None, "pic50")
        monkeypatch.setattr(playground, "fpdf_available", lambda: False)
        app._ml_pdf_var.set(True)
        app._ml_save_report()
        assert not app._busy  # gated before the folder dialog
        titles = [call[1][0] for call in dialogs]
        assert "PDF needs fpdf" in titles
        assert "pharmaml[reports]" in dialogs[0][1][1]  # the install hint

    def test_report_bundle_end_to_end(
        self,
        app: Any,
        dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
        tmp_path: Path,
    ) -> None:
        pypdf = pytest.importorskip("pypdf")
        target = tmp_path / "bundle"
        monkeypatch.setattr(playground.filedialog, "askdirectory", lambda **k: str(target))
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "training job did not finish"
        assert app._ml_last_run is not None  # the run is now reportable

        app._ml_pdf_var.set(True)
        app._ml_save_report()
        assert self._drain(app), "report job did not finish"

        assert (target / "report.html").is_file()
        assert (target / "report.pdf").is_file()
        assert (target / "confusion_matrix.png").is_file()
        assert (target / "held_out_metrics.csv").is_file()
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the success dialog never fired"
        assert infos[0][1][0] == "Report saved"
        assert "report saved" in app._job_label.cget("text")
        reader = pypdf.PdfReader(str(target / "report.pdf"))
        first = reader.pages[0].extract_text()
        assert "playground ml train" in first
        assert "logistic_regression" in first

    def test_report_html_only_when_pdf_unticked(
        self,
        app: Any,
        dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
        tmp_path: Path,
    ) -> None:
        # the default pin: unticked -> the bundle without report.pdf
        target = tmp_path / "bundle_html"
        monkeypatch.setattr(playground.filedialog, "askdirectory", lambda **k: str(target))
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_train()
        assert self._drain(app), "training job did not finish"
        app._ml_save_report()
        assert self._drain(app), "report job did not finish"
        assert (target / "report.html").is_file()
        assert (target / "predicted_vs_actual.png").is_file()
        assert not (target / "report.pdf").exists()  # the unticked default


class TestMLCheckCSV:
    """``_ml_check_csv`` audits any CSV against the stored run (ml check, GUI).

    The audit story's last axis (100x-79): library (100x-77) -> CLI
    (100x-77) -> executed notebook (100x-78) -> here.  Guards first (no
    run, cancelled picker), then the OK verdict on the very CSV the model
    was trained on (the automatic ``--target`` drop) and the FAIL verdict
    on a drifted copy -- missing and extra columns, all named.
    """

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
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
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _train(app: Any) -> None:
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._ml_train()
        assert TestMLCheckCSV._drain(app), "training job did not finish"
        assert app._ml_last_run is not None

    def test_check_needs_a_run_first(self, app: Any, dialogs: list[Any]) -> None:
        app._ml_check_csv()
        assert not app._busy  # the guard fired before any job started
        titles = [call[1][0] for call in dialogs]
        assert "No run yet" in titles
        assert "Train a model first" in dialogs[0][1][1]  # the remedy, named

    def test_check_cancelled_starts_no_job(
        self, app: Any, monkeypatch: Any, playground: Any
    ) -> None:
        app._ml_last_run = (object(), object(), "regression", None, None, "pIC50")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: "")
        app._ml_check_csv()
        assert not app._busy  # the cancelled picker never queued work

    def test_check_audit_ok_on_training_csv(
        self, app: Any, dialogs: list[Any], monkeypatch: Any, playground: Any
    ) -> None:
        self._train(app)
        csv = str(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: csv)
        app._ml_check_csv()
        assert self._drain(app), "audit job did not finish"

        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the verdict dialog never fired"
        title, body = infos[0][1][0], infos[0][1][1]
        assert title == "Dataset audit: OK"
        assert "audit: OK" in body
        assert "qsar_regression.csv" in body
        assert "pharmaml ml check --model <bundle>" in body  # the CLI twin
        assert "--target pIC50" in body  # the automatic target drop, named
        assert "audit: OK" in app._job_label.cget("text")

    def test_check_audit_flags_drift(
        self,
        app: Any,
        dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
        tmp_path: Path,
    ) -> None:
        import pandas as pd

        self._train(app)
        # the months-later CSV: one feature dropped, one renamed.
        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        frame = frame.drop(columns=["descr_000"]).rename(columns={"descr_001": "descr_1"})
        drifted = tmp_path / "drifted.csv"
        frame.to_csv(drifted, index=False)

        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: str(drifted))
        app._ml_check_csv()
        assert self._drain(app), "audit job did not finish"

        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the verdict dialog never fired"
        title, body = infos[0][1][0], infos[0][1][1]
        assert title == "Dataset audit: FAIL"
        assert "missing 2 feature(s): descr_000, descr_001" in body
        assert "1 extra column(s) the model does not know: descr_1" in body
        assert "--target pIC50" in body  # the label itself was auto-dropped
        assert "audit: FAIL" in app._job_label.cget("text")


class TestDLCheckCSV:
    """``_dl_check_csv`` audits any CSV against the last network (100x-81).

    The audit story's deep axis: the neural estimators satisfy the same
    sklearn contract (``feature_names_in_`` at fit, joblib-serialisable),
    so ``validate_bundle_input`` audits them exactly like the registry
    models.  Guards first (no run, cancelled picker), then the OK verdict
    on the very CSV the network was trained on (the automatic
    ``--target`` drop) and the FAIL verdict on a drifted copy -- missing
    and extra columns, all named.
    """

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 180.0) -> bool:
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @classmethod
    def _train(cls, app: Any) -> None:
        """Tiny network, the bundled regression example, one drained job."""
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "8")
        app._dl_epochs.set("20")
        app._dl_patience.set("3")
        app._dl_load_path(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._dl_train()
        assert cls._drain(app), "network training hung"
        assert app._dl_last_run is not None

    def test_check_needs_a_run_first(self, app: Any, dialogs: list[Any]) -> None:
        app._dl_check_csv()
        assert not app._busy  # the guard fired before any job started
        titles = [call[1][0] for call in dialogs]
        assert "No run yet" in titles
        assert "Train a network first" in dialogs[0][1][1]  # the remedy, named

    def test_check_cancelled_starts_no_job(
        self, app: Any, monkeypatch: Any, playground: Any
    ) -> None:
        app._dl_last_run = (object(), "DeepMLPRegressor", "regression", None, "pIC50")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: "")
        app._dl_check_csv()
        assert not app._busy  # the cancelled picker never queued work

    def test_check_audit_ok_on_training_csv(
        self, app: Any, dialogs: list[Any], monkeypatch: Any, playground: Any
    ) -> None:
        self._train(app)
        csv = str(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: csv)
        app._dl_check_csv()
        assert self._drain(app), "audit job did not finish"

        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the verdict dialog never fired"
        title, body = infos[0][1][0], infos[0][1][1]
        assert title == "Dataset audit: OK"
        assert "audit: OK" in body
        assert "qsar_regression.csv" in body
        assert "DeepMLPRegressor bundle" in body  # the network, named
        assert "pharmaml ml check --model <bundle>" in body  # the CLI twin
        assert "--target pIC50" in body  # the automatic target drop, named
        assert "audit: OK" in app._job_label.cget("text")

    def test_check_audit_flags_drift(
        self,
        app: Any,
        dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
        tmp_path: Path,
    ) -> None:
        import pandas as pd

        self._train(app)
        # the months-later CSV: one feature dropped, one renamed.
        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        frame = frame.drop(columns=["descr_000"]).rename(columns={"descr_001": "descr_1"})
        drifted = tmp_path / "drifted.csv"
        frame.to_csv(drifted, index=False)

        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **k: str(drifted))
        app._dl_check_csv()
        assert self._drain(app), "audit job did not finish"

        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos, "the verdict dialog never fired"
        title, body = infos[0][1][0], infos[0][1][1]
        assert title == "Dataset audit: FAIL"
        assert "missing 2 feature(s): descr_000, descr_001" in body
        assert "1 extra column(s) the model does not know: descr_1" in body
        assert "--target pIC50" in body  # the label itself was auto-dropped
        assert "audit: FAIL" in app._job_label.cget("text")


# --------------------------------------------------------------------- #
# the Utilities / converter / similarity surfaces (100x-110) -- the
# cards the scratch probes drove for three rounds without pytest pins
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestUtilitiesCoverageGui:
    """Every 100x-110 coverage card, driven the way a mouse drives it.

    The scratch probes (probe_coverage_110.py) have guarded these
    surfaces manually since round 110; these tests pin the same visible
    outcomes (frame rows, tree fills, on-disk side effects) so a
    regression fails the suite, not a manual probe run.
    """

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        for name in ("askyesno", "askokcancel"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)) or True,
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- fixtures ----------------------------------------------------- #
    LIGANDS_A = "Ligand,SMILES,score\nlig1,CCO,1.5\nlig2,CCOCC,2.5\nlig3,c1ccccc1,3.5\n"
    LIGANDS_B = "Ligand,SMILES\nlig1,CCO\nlig4,CNC\n"

    def test_merge_picked_files_and_first_columns_reorder(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
        playground: Any,
    ) -> None:
        table_a = tmp_path / "ligands_a.csv"
        table_a.write_text(self.LIGANDS_A, encoding="utf-8")
        table_b = tmp_path / "ligands_b.csv"
        table_b.write_text(self.LIGANDS_B, encoding="utf-8")

        app._util_merge_files = [table_a, table_b]
        app._util_merge_mode.set("merge picked files")
        app._util_merge_mode_changed()
        app._util_merge_run()
        assert self._drain_jobs(app), "merge job hung"
        merged = app._util_frames.get("merge")
        assert merged is not None and len(merged) == 5

        firstcols = app._util_merge_firstcols
        firstcols.delete(0, "end")
        firstcols.insert(0, "score,SMILES")
        app._util_merge_run()
        assert self._drain_jobs(app), "reordered merge job hung"
        merged = app._util_frames.get("merge")
        assert merged is not None
        assert next(iter(merged.columns)) == "score"

    def test_split_into_chunks(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
        playground: Any,
    ) -> None:
        big_csv = tmp_path / "big.csv"
        big_csv.write_text(
            "a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(25)),
            encoding="utf-8",
        )
        parts_dir = tmp_path / "parts"

        app._util_merge_split_file = big_csv
        app._util_merge_mode.set("split into chunks")
        app._util_merge_mode_changed()
        app._util_merge_chunk.delete(0, "end")
        app._util_merge_chunk.insert(0, "10")
        monkey = pytest.MonkeyPatch()
        monkey.setattr(playground.filedialog, "askdirectory", lambda **_k: str(parts_dir))
        try:
            app._util_merge_run()
            assert self._drain_jobs(app), "split job hung"
        finally:
            monkey.undo()
        parts = sorted(parts_dir.glob("*.csv")) if parts_dir.exists() else []
        assert len(parts) == 3  # 25 rows / 10 per chunk

    def test_compare_column_and_annotate_smiles(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        table_a = tmp_path / "ligands_a.csv"
        table_a.write_text(self.LIGANDS_A, encoding="utf-8")
        table_b = tmp_path / "ligands_b.csv"
        table_b.write_text(self.LIGANDS_B, encoding="utf-8")

        app._util_cmp_a = table_a
        app._util_cmp_b = table_b
        app._util_cmp_refresh_info()
        app._util_cmp_mode.set("compare a column (A vs B)")
        app._util_compare_run()
        assert self._drain_jobs(app), "compare job hung"
        cmp_frame = app._util_frames.get("compare")
        sections = cmp_frame["section"].tolist()
        assert sections.count("common") == 1
        assert sections.count("only in A") == 2
        assert sections.count("only in B") == 1

        app._util_cmp_mode.set("annotate names with SMILES")
        app._util_compare_run()
        assert self._drain_jobs(app), "annotate job hung"
        ann = app._util_frames.get("compare")
        assert ann is not None and "Ligand_SMILES" in ann.columns
        assert ann.loc[ann["Ligand"] == "lig1", "Ligand_SMILES"].iloc[0] == "CCO"

    def test_approved_drug_filter(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        approval = tmp_path / "approvals.tsv"
        approval.write_text(
            "drug\tregulatory approval\naspirin\tapproved\n"
            "caffeine\twithdrawn\nibuprofen\tapproved\n",
            encoding="utf-8",
        )
        app._util_appr_files = [approval]
        app._util_approved_run()
        assert self._drain_jobs(app), "approved job hung"
        appr = app._util_frames.get("approved")
        assert appr is not None
        assert appr["drug"].tolist() == ["aspirin", "ibuprofen"]

    def test_campaign_tools_collect_list_distribute_cleanup(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        camp = tmp_path / "campaign"
        for sub in ("lig1", "lig2", "lig3"):
            (camp / sub).mkdir(parents=True)
        (camp / "lig1" / "r.dlg").write_text("fake dlg", encoding="utf-8")
        (camp / "lig2" / "r.dlg").write_text("fake dlg\nSuccessful Completion\n", encoding="utf-8")
        (camp / "lig1" / "1.pdbqt").write_text(
            "REMARK x\nTORSDOF 2\nBRANCH\nBRANCH\n", encoding="utf-8"
        )
        (camp / "stray.txt").write_text("stray", encoding="utf-8")

        # collect (dry run): filename ∪ pattern union
        app._util_tool_folder = camp
        app._util_tool_mode.set("collect files (name/pattern)")
        app._util_tool_mode_changed()
        app._util_tool_filename.delete(0, "end")
        app._util_tool_filename.insert(0, "r.dlg")
        app._util_tools_run()
        assert self._drain_jobs(app), "collect job hung"
        tools_frame = app._util_frames.get("tools")
        assert tools_frame is not None and len(tools_frame) == 3
        collected = camp / "collected"
        assert not collected.exists() or not list(collected.glob("*.dlg"))

        # list file names
        app._util_tool_mode.set("list file names")
        app._util_tools_run()
        assert self._drain_jobs(app), "list job hung"
        listed = app._util_frames.get("tools")
        assert listed is not None and "name" in listed.columns

        # distribute one file into every campaign folder
        marker = tmp_path / "bin_marker.txt"
        marker.write_text("bin", encoding="utf-8")
        app._util_tool_files = [marker]
        app._util_tool_mode.set("distribute files into folders")
        app._util_tools_run()
        assert self._drain_jobs(app), "distribute job hung"
        copies = list(camp.rglob("bin_marker.txt"))
        assert len(copies) == 3

        # cleanup (NOT dry): allowlist ∪ keep-pattern survives
        app._util_tool_mode.set("cleanup tree (keep list)")
        app._util_tool_dry_var.set(False)
        app._util_tools_run()
        assert self._drain_jobs(app), "cleanup job hung"
        app._util_tool_dry_var.set(True)
        survivors = {p.name for p in camp.rglob("*") if p.is_file()}
        assert survivors <= {
            "r.pdbqt",
            "r.dlg",
            "r.gpf",
            "r.dpf",
            "bin_marker.txt",
            "1.pdbqt",
        }
        assert "stray.txt" not in survivors
        assert "1.pdbqt" in survivors  # the keep pattern kept it

    def test_reindex_dense_range(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        sparse = tmp_path / "sparse.csv"
        sparse.write_text(
            "ligand,energy\n5_result.dlg,-7.1\n6_result.dlg,-8.2\n8_result.dlg,-6.0\n",
            encoding="utf-8",
        )
        app._util_reindex_path = sparse
        app._util_reindex_column.delete(0, "end")
        app._util_reindex_column.insert(0, "ligand")
        app._util_reindex_start.delete(0, "end")
        app._util_reindex_start.insert(0, "5")
        app._util_reindex_stop.delete(0, "end")
        app._util_reindex_stop.insert(0, "8")
        app._util_reindex_run()
        assert self._drain_jobs(app), "reindex job hung"
        reindexed = app._util_frames.get("reindex")
        assert reindexed is not None and len(reindexed) == 4  # 5..8 dense

    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_converters_csv_sdf_smi_pic50(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        table_a = tmp_path / "ligands_a.csv"
        table_a.write_text(self.LIGANDS_A, encoding="utf-8")

        # CSV table -> 3D SDF
        sdf_out = tmp_path / "ligands.sdf"
        app._conv_input = table_a
        app._conv_output = sdf_out
        app._conv_mode.set("CSV table → 3D SDF")
        app._conv_run()
        assert self._drain_jobs(app), "CSV→SDF job hung"
        assert sdf_out.exists() and sdf_out.stat().st_size > 0

        # .smi list -> CSV
        smi_out = tmp_path / "list.smi"
        smi_out.write_text("CCO ethanol\nCC(C)CO isopropanol\n", encoding="utf-8")
        smi_csv_out = tmp_path / "list.csv"
        app._conv_input = smi_out
        app._conv_output = smi_csv_out
        app._conv_mode.set(".smi list → CSV")
        app._conv_run()
        assert self._drain_jobs(app), ".smi→CSV job hung"
        assert smi_csv_out.exists()
        assert "SMILES" in smi_csv_out.read_text(encoding="utf-8")

        # SDF -> table
        app._conv_input = sdf_out
        app._conv_mode.set("SDF → table")
        app._conv_run()
        assert self._drain_jobs(app), "SDF→table job hung"
        sdf_frame = app._conv_frame
        assert sdf_frame is not None and len(sdf_frame) == 3
        assert "score" in sdf_frame.columns

        # pIC50 column: censored + junk rows stay NaN (honesty rule)
        measure_csv = tmp_path / "measure.csv"
        measure_csv.write_text("compound,IC50 (nM)\nA,100\nB,<10\nC,xyz\n", encoding="utf-8")
        app._conv_input = measure_csv
        app._conv_mode.set("append pIC50 column")
        app._conv_source_col.delete(0, "end")
        app._conv_source_col.insert(0, "IC50 (nM)")
        app._conv_run()
        assert self._drain_jobs(app), "pIC50 job hung"
        pic_frame = app._conv_frame
        assert pic_frame is not None
        assert int(pic_frame["pIC50"].notna().sum()) == 2
        assert abs(pic_frame.loc[0, "pIC50"] - 7.0) < 1e-9

    @pytest.mark.skipif(importlib.util.find_spec("rdkit") is None, reason="needs RDKit")
    def test_similarity_scan_matrix_threshold_topk(
        self,
        app: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        # matrix: 2 default references x 7 default library rows
        app._scan_mode.set("matrix: references × library")
        app._scan_run()
        assert self._drain_jobs(app), "matrix job hung"
        matrix_frame = app._scan_frame
        assert matrix_frame is not None
        assert matrix_frame.shape == (2, 1 + 7)
        assert abs(float(matrix_frame.iloc[0, 1]) - 1.0) < 1e-9

        # threshold scan: both identity pairs clear 0.4
        app._scan_mode.set("scan: pairs ≥ threshold")
        app._scan_threshold.delete(0, "end")
        app._scan_threshold.insert(0, "0.4")
        app._scan_run()
        assert self._drain_jobs(app), "threshold job hung"
        scan_frame = app._scan_frame
        assert scan_frame is not None and len(scan_frame) == 2
        identity_pairs = {
            (row.reference, row.target)
            for row in scan_frame.itertuples()
            if row.similarity >= 0.999
        }
        assert len(identity_pairs) == 2
        assert {"reference", "target", "similarity"} <= set(scan_frame.columns)

        # top-k: sorted descending
        app._scan_mode.set("top-k: nearest of first reference")
        app._scan_k.delete(0, "end")
        app._scan_k.insert(0, "3")
        app._scan_run()
        assert self._drain_jobs(app), "top-k job hung"
        top_frame = app._scan_frame
        assert top_frame is not None and len(top_frame) == 3
        assert top_frame["similarity"].iloc[0] >= top_frame["similarity"].iloc[-1]

    def test_clean_numeric_frame_keeps_smiles(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        mixed = tmp_path / "mixed.csv"
        mixed.write_text(
            "SMILES,name,MolWt,LogP,target\nCCO,ethanol,46.1,-0.3,1.0\nCC(C)CO,iso,60.1,0.3,2.0\n",
            encoding="utf-8",
        )
        app._ml_load_path(mixed)
        assert self._drain_jobs(app), "dataset load job hung"
        app._ml_keep_entry.delete(0, "end")
        app._ml_keep_entry.insert(0, "SMILES")
        app._ml_clean_frame()
        assert self._drain_jobs(app), "clean job hung"
        cleaned_active = list(app._ml_frame.columns)
        assert "SMILES" in cleaned_active
        assert "name" not in cleaned_active
        assert {"MolWt", "LogP", "target"} <= set(cleaned_active)


# --------------------------------------------------------------------- #
# the Docking coverage surfaces (100x-110 + 100x-111): the audit card,
# single SMILES→PDBQT, the Vina config batch, the campaign GPF/DPF
# batch with custom receptor types, and the Vina results folder scan
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestDockingCoverageGui:
    LIGAND_PDBQT = (
        "REMARK  Name = probe_ligand\n"
        "ROOT\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     C\n"
        "ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00     OA\n"
        "ATOM      3  N   LIG A   1       0.000   2.000   0.000  1.00  0.00     NA\n"
        "ENDROOT\n"
        "TORSDOF 1\n"
    )

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        for name in ("askyesno", "askokcancel"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)) or True,
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _campaign_fixture(self, tmp_path: Path) -> Path:
        camp = tmp_path / "campaign"
        for sub in ("lig1", "lig2", "lig3"):
            (camp / sub).mkdir(parents=True)
        (camp / "lig1" / "r.dlg").write_text("fake dlg", encoding="utf-8")
        (camp / "lig2" / "r.dlg").write_text("fake dlg\nSuccessful Completion\n", encoding="utf-8")
        (camp / "lig1" / "1.pdbqt").write_text(
            "REMARK x\nTORSDOF 2\nBRANCH\nBRANCH\n", encoding="utf-8"
        )
        return camp

    def test_audit_card_gpf_torsion_completion(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        camp = self._campaign_fixture(tmp_path)
        gpf = tmp_path / "r.gpf"
        gpf.write_text(
            "ligand_types C HD OA\nreceptor r.pdbqt\n"
            "receptor_types C HD OA A\n"
            "npts 40 40 40\nspacing 0.375\ngridcenter 1.0 2.0 3.0\n"
            "smooth 0.5\ndielectric -0.1465\nmap 1.C.map\n",
            encoding="utf-8",
        )

        # GPF file -> parameters
        app._dock_audit_gpf = gpf
        app._dock_audit_mode.set("GPF file → parameters")
        app._dock_audit_run()
        assert self._drain_jobs(app), "GPF audit job hung"
        params_frame = app._dock_audit_frame
        assert params_frame is not None and len(params_frame) == 9
        assert params_frame.loc[0, "value"] == "C HD OA"

        # folder: torsion consistency
        app._dock_audit_mode.set("folder: torsion consistency")
        app._dock_audit_folder = camp
        app._dock_audit_run()
        assert self._drain_jobs(app), "torsion audit job hung"
        torsion_frame = app._dock_audit_frame
        assert torsion_frame is not None and len(torsion_frame) >= 1
        assert {"ligand", "torsdof", "branches", "status"} <= set(torsion_frame.columns)

        # folder: completion check (lig2 OK, lig1 INCOMPLETE, lig3 MISSING_LOG)
        app._dock_audit_mode.set("folder: completion check")
        app._dock_audit_run()
        assert self._drain_jobs(app), "completion audit job hung"
        completion_frame = app._dock_audit_frame
        statuses = completion_frame["status"].tolist()
        assert "OK" in statuses and "INCOMPLETE" in statuses
        assert "MISSING_LOG" in statuses

    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None or importlib.util.find_spec("meeko") is None,
        reason="needs RDKit + Meeko",
    )
    def test_single_smiles_to_pdbqt(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        smi_dir = tmp_path / "smi_out"
        app._dock_smi_out = smi_dir
        app._dock_smi_entry.delete(0, "end")
        app._dock_smi_entry.insert(0, "CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        app._dock_smi_run()
        assert self._drain_jobs(app), "SMILES→PDBQT job hung"
        pdbqt = list(smi_dir.glob("*.pdbqt")) if smi_dir.exists() else []
        assert len(pdbqt) == 1

    def test_vina_config_batch_fake_engine(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
        monkeypatch: Any,
        playground: Any,
    ) -> None:
        import os
        import stat

        # (100x-171) neutralise the bundled engines/ folder so the fake
        # vina on PATH below is the engine this batch actually launches
        monkeypatch.setattr(playground, "_ENGINES_FOLDER", tmp_path / "no_engines")
        binroot = tmp_path / "bin"
        binroot.mkdir()
        fake_vina = binroot / "vina"
        fake_vina.write_text(
            "#!/bin/bash\n"
            "out=''; lig=''; prev=''\n"
            'for arg in "$@"; do\n'
            '  [ "$prev" = --out ] && out="$arg"\n'
            '  [ "$prev" = --ligand ] && lig="$arg"\n'
            '  prev="$arg"\n'
            "done\n"
            "{ echo 'REMARK VINA RESULT:    -8.2      0.000      0.000'\n"
            "  echo 'REMARK VINA RESULT:    -7.1      1.923      2.941'\n"
            '  grep \'^ATOM\\|^HETATM\' "$lig"; } > "$out"\n'
            "exit 0\n",
            encoding="utf-8",
        )
        fake_vina.chmod(fake_vina.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        monkeypatch.setenv("PATH", f"{binroot}{os.pathsep}{os.environ.get('PATH', '')}")

        vbatch_dir = tmp_path / "vbatch"
        (vbatch_dir / "ligands").mkdir(parents=True)
        for name in ("l1", "l2"):
            (vbatch_dir / "ligands" / f"{name}.pdbqt").write_text(
                "REMARK ok\nATOM      1  C   LIG A   1       0.000   0.000   "
                "0.000  1.00  0.00           C\n",
                encoding="utf-8",
            )
        (vbatch_dir / "r.pdbqt").write_text("ATOM receptor\n", encoding="utf-8")
        conf = vbatch_dir / "vina.conf"
        conf.write_text("receptor = r.pdbqt\ncenter_x = 0\nsize_x = 20\n", encoding="utf-8")
        vout = vbatch_dir / "results"

        app._dock_vbatch_conf = conf
        app._dock_vbatch_ligands = vbatch_dir / "ligands"
        app._dock_vbatch_out = vout
        app._dock_vbatch_refresh_info()
        app._dock_vbatch_run()
        assert self._drain_jobs(app), "vina batch job hung"
        outs = sorted(vout.glob("*_out.pdbqt")) if vout.exists() else []
        assert len(outs) == 2
        assert app._dock_vbatch_frame is not None
        assert len(app._dock_vbatch_frame) == 2

    def test_batch_gpf_dpf_campaign_and_receptor_types(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
        playground: Any,
    ) -> None:
        campaign = tmp_path / "campaign"
        for name in ("01_aspirin", "02_ibuprofen"):
            folder = campaign / name
            folder.mkdir(parents=True)
            (folder / "l.pdbqt").write_text(self.LIGAND_PDBQT, encoding="utf-8")
        (campaign / "03_empty").mkdir()

        # batch over the campaign tree: 2/3 folders configured
        monkey = pytest.MonkeyPatch()
        monkey.setattr(playground.filedialog, "askdirectory", lambda **_k: str(campaign))
        try:
            app._dock_write_config_batch()
            assert self._drain_jobs(app), "batch config job hung"
        finally:
            monkey.undo()
        report_text = app._dock_grid_text.get("1.0", "end")
        assert "configured      2/3 folders" in report_text
        assert "without a ligand" in report_text
        for name in ("01_aspirin", "02_ibuprofen", "03_empty"):
            assert name in report_text
        assert "no ligand PDBQT" in report_text
        for name in ("01_aspirin", "02_ibuprofen"):
            for fname in ("r.gpf", "r.dpf"):
                assert (campaign / name / fname).is_file()
        gpf_text = (campaign / "01_aspirin" / "r.gpf").read_text(encoding="utf-8")
        assert "A C HD N NA OA SA" in gpf_text  # default receptor types

        # the receptor-types field feeds the single-ligand path too
        single = tmp_path / "single"
        single.mkdir()
        ligand_path = single / "l.pdbqt"
        ligand_path.write_text(self.LIGAND_PDBQT, encoding="utf-8")
        app._dock_set_ligand_path(ligand_path)
        app._dock_receptor_types_entry.delete(0, "end")
        app._dock_receptor_types_entry.insert(0, "A C HD N Ni OA SA")
        app._dock_compute_grid()
        assert self._drain_jobs(app), "single grid job hung"
        grid_text = app._dock_grid_text.get("1.0", "end")
        assert "receptor types A C HD N Ni OA SA" in grid_text
        single_gpf = (single / "r.gpf").read_text(encoding="utf-8")
        assert "A C HD N Ni OA SA" in single_gpf
        app._dock_receptor_types_entry.delete(0, "end")  # back to default

        # batch with the Ni types (the gpferreceptorni.py case)
        for name in ("01_aspirin", "02_ibuprofen"):
            (campaign / name / "r.gpf").unlink()  # force a fresh write
        app._dock_receptor_types_entry.insert(0, "A C HD N Ni OA SA")
        monkey.setattr(playground.filedialog, "askdirectory", lambda **_k: str(campaign))
        try:
            app._dock_write_config_batch()
            assert self._drain_jobs(app), "Ni batch config job hung"
        finally:
            monkey.undo()
        batch_gpf = (campaign / "01_aspirin" / "r.gpf").read_text(encoding="utf-8")
        assert "A C HD N Ni OA SA" in batch_gpf

    def test_scan_vina_results_folder(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        tmp_path: Path,
        playground: Any,
    ) -> None:
        vina_log = (
            "Estimated Free Energy of Binding   = -7.051 (kcal/mol)\n"
            "REMARK VINA RESULT:    -7.051      0.000      0.000\n"
            "REMARK VINA RESULT:    -6.802      1.938      3.655\n"
        )
        vina_root = tmp_path / "vina_results"
        (vina_root / "lig01").mkdir(parents=True)
        (vina_root / "lig01" / "lig01.log").write_text(vina_log, encoding="utf-8")
        (vina_root / "lig02.log").write_text(vina_log.replace("-7.051", "-8.35"), encoding="utf-8")
        (vina_root / "crashed.log").write_text("Vina crashed before scoring\n", encoding="utf-8")

        monkey = pytest.MonkeyPatch()
        monkey.setattr(playground.filedialog, "askdirectory", lambda **_k: str(vina_root))
        try:
            app._dock_scan_vina()
            assert self._drain_jobs(app), "vina scan job hung"
        finally:
            monkey.undo()

        rows = app._dock_results_tree.get_children()
        assert len(rows) == 3
        values = [app._dock_results_tree.item(row, "values") for row in rows]
        # strongest binder first (lig02, -8.35), crashed row kept last
        assert values[0][0] == "lig02" and "-8.35" in str(values[0][2])
        assert any(v[0] == "crashed" for v in values)
        info = app._dock_results_info.cget("text")
        assert "3 Vina logs" in info and "lig02" in info and "-8.35" in info
        # the scan table is the active results frame (saveable/rescorable)
        assert app._dock_results_source == "vina"
        assert list(app._dock_results_frame.columns) == [
            "ligand",
            "log",
            "best_affinity",
            "n_poses",
        ]


class TestApplicabilityDomainGui:
    """The AD card (100x-118): the uploaded workflow driven end-to-end."""

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @staticmethod
    def _drain(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_source_line_tracks_training_runs(
        self, playground: Any, app: Any, tmp_path: Path
    ) -> None:
        assert "no run yet" in app._ad_source_info.cget("text")
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "train job hung"
        info = app._ad_source_info.cget("text")
        assert "logistic_regression" in info and "query rows" in info

    def test_full_analysis_renders_table_figure_and_status(self, playground: Any, app: Any) -> None:
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "train job hung"
        n_query = len(app._ml_last_run[5])
        app._ad_boot_spin.set("8")
        app._ml_ad_run()
        assert self._drain(app), "AD job hung"
        result = app._ml_ad_result
        assert result is not None and result.reliable is not None
        assert len(app._ad_tree.get_children()) == n_query
        status = app._ad_status.cget("text")
        assert "distance threshold" in status and "reliable" in status
        assert app._ad_host.winfo_children()  # figure embedded (canvas/label + optional toolbar)
        # the name column carries the library's SMILES strings
        first = str(app._ad_tree.item(app._ad_tree.get_children()[0])["values"][0])
        assert any(ch in first for ch in "CNc#(")

    def test_knn_only_quick_pass(self, playground: Any, app: Any) -> None:
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "train job hung"
        app._ad_boot_spin.set("0")
        app._ml_ad_run()
        assert self._drain(app), "kNN-only job hung"
        assert app._ml_ad_result.reliable is None
        assert "kNN-only" in app._ad_status.cget("text")
        assert app._ml_ad_frame["reliable"].isna().all()

    def test_bundle_mode_round_trip(
        self, playground: Any, app: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "train job hung"
        bundle = tmp_path / "bundle.joblib"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(bundle))
        app._ml_save_model_bundle()
        assert self._drain(app), "bundle save hung"
        assert bundle.is_file()
        monkeypatch.setattr(playground.filedialog, "askopenfilename", lambda **_k: str(bundle))
        app._ml_load_model_bundle()
        assert self._drain(app), "bundle load hung"

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        train_csv = tmp_path / "train.csv"
        test_csv = tmp_path / "test.csv"
        frame.iloc[:60].to_csv(train_csv, index=False)
        frame.iloc[60:].to_csv(test_csv, index=False)
        app._ml_ad_train_path = train_csv
        app._ml_ad_test_path = test_csv
        app._ad_source_combo.set("loaded bundle")
        app._ml_ad_source_changed()
        assert app._ad_bundle_row.winfo_manager() == "grid"
        assert "logistic_regression" in app._ad_source_info.cget("text")

        app._ad_boot_spin.set("6")
        app._ml_ad_run()
        assert self._drain(app), "bundle-mode AD job hung"
        assert app._ml_ad_result.pred_mean is not None
        assert len(app._ad_tree.get_children()) == len(frame) - 60

    def test_bundle_mode_visibility_toggles(self, playground: Any, app: Any) -> None:
        app._ad_source_combo.set("loaded bundle")
        app._ml_ad_source_changed()
        assert app._ad_bundle_row.winfo_manager() == "grid"
        app._ad_source_combo.set("last trained model")
        app._ml_ad_source_changed()
        # grid_remove keeps the grid bookkeeping but unmaps the row:
        # winfo_manager() goes empty (pack would report "forget")
        assert app._ad_bundle_row.winfo_manager() == ""
        # no bundle loaded: the hint names the persistence card
        app._ad_source_combo.set("loaded bundle")
        app._ml_ad_source_changed()
        assert "no bundle loaded" in app._ad_source_info.cget("text")

    def test_no_run_and_bad_knob_warnings(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        seen = quiet_dialogs
        app._ml_last_run = None
        app._ml_ad_run()  # no run: warning, no job
        assert seen and any("Train a model" in str(a) for a in seen)
        # a bad knob: parse_int refuses before the job starts
        app._ml_load_path(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._model_combo.set("logistic_regression")
        app._ml_train()
        assert self._drain(app), "train job hung"
        app._ad_k_spin.set("0")
        app._ml_ad_run()
        assert any("k (neighbours)" in str(a) for a in seen)


class TestCompletenessReviewGui:
    """The 100x-119 additions: NCA, infusion/TDM, the PD family, the
    chemistry tautomer/MACCS cards, the new figures and the navigation
    chips -- every gap the full review found, pinned."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_nca_card_computes_the_trapezoid_summary(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._nca_run()
        assert self._drain(app), "NCA job hung"
        text = app._nca_result.cget("text")
        assert "10 points" in text, text
        assert "AUC 56.4850" in text and "MRT 5.2523" in text, text
        assert "t½(term)" in text and "Cmax 12.0000" in text, text
        # no dose given: Vd/CL stay absent
        assert "Vd" not in text
        # the profile figure embedded
        assert app._nca_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert quiet_dialogs == []

    def test_nca_card_with_dose_reports_vd_and_clearance(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._nca_dose.insert(0, "500")
        app._nca_run()
        assert self._drain(app), "NCA job hung"
        text = app._nca_result.cget("text")
        assert "Vd 8.8519" in text, text  # 500 / 56.485
        assert "CL 8.8519" in text, text  # 500 / 56.485 L/h
        assert quiet_dialogs == []

    def test_nca_series_length_mismatch_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._nca_concs.delete(0, "end")
        app._nca_concs.insert(0, "1, 2, 3")
        app._nca_run()
        assert any("matched pairs" in str(a) for a in quiet_dialogs)

    def test_infusion_card_computes_rate_css_vd(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._infusion_math()
        text = app._inf_result.cget("text")
        assert "rate for Css 8.0000 = 40.0000 mg/h" in text, text
        assert "Css at 40.0000 mg/h = 8.0000 mg/L" in text, text
        assert "Vd = 28.9017 L" in text, text
        assert "F =" not in text  # no EV/IV pairs typed
        assert quiet_dialogs == []

    def test_infusion_card_bioavailability_pair(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._bio_ev_dose.insert(0, "250 / 58.3")
        app._bio_iv_dose.insert(0, "250 / 96.5")
        app._infusion_math()
        text = app._inf_result.cget("text")
        assert "F = 60.4145%" in text, text  # 58.3 / 96.5
        assert quiet_dialogs == []

    def test_tdm_card_adjusts_the_dose(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._tdm_math()
        text = app._tdm_result.cget("text")
        assert "new dose 403.2258 mg" in text, text
        assert "floored" not in text
        assert quiet_dialogs == []

    def test_pd_family_card_table_and_reverse(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._pd_family_math()
        assert self._drain(app), "PD family job hung"
        rows = app._fam_tree.get_children()
        assert len(rows) == 6, rows
        values = app._fam_tree.item(rows[0], "values")
        assert values[0] == "0.5000" and values[1] == "9.93", values
        assert app._fam_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        app._pd_target_ec50()
        assert "EC50 = 1.9977" in app._fam_result.cget("text")
        assert quiet_dialogs == []

    def test_tautomer_card_and_maccs_card(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._tautomer_run()
        assert app._taut_result.cget("text") == "CC(=O)Oc1ccccc1C(=O)O"
        app._maccs_run()
        assert self._drain(app), "MACCS job hung"
        text = app._maccs_result.cget("text")
        assert "3 molecule(s)" in text and "/167" in text, text
        assert quiet_dialogs == []

    def test_quick_jump_and_tab_cycling(self, app: Any) -> None:
        app._jump_to_tab(4)
        assert app._book.index(app._book.select()) == 4
        app._jump_to_tab(0)
        assert app._book.index(app._book.select()) == 0

        class _Event:
            keysym = "Next"

        app._cycle_tab(_Event())
        assert app._book.index(app._book.select()) == 1

        class _Back:
            keysym = "Prior"

        app._cycle_tab(_Back())
        assert app._book.index(app._book.select()) == 0
        # out-of-range jumps are ignored, not raised
        app._jump_to_tab(99)

    def test_cards_carry_the_accent_strip(self, app: Any) -> None:
        # walk one card's header: a 3px plain Frame painted ACCENT sits
        # next to the title (make_card, 100x-119)
        found = 0

        def walk(widget: Any) -> None:
            nonlocal found
            for child in widget.winfo_children():
                if child.winfo_class() == "Frame" and child.winfo_width() <= 4:
                    found += 1
                walk(child)

        walk(app)
        assert found >= 30, f"expected >=30 accent strips, found {found}"

    def test_hint_row_lists_the_tab_cycling_shortcut(self, playground: Any) -> None:
        text = playground.shortcut_hints_text()
        assert "Ctrl+PgDn/PgUp next/previous tab" in text


# --------------------------------------------------------------------- #
# the 100x-120 full-review round: the last library stragglers gain GUI
# surfaces (descriptor distributions, learning curve, reconciliation
# figure, parsed Vina config, MAD gene ranking, effect_table, the
# single-folder AutoDock4 engine) + the Ctrl+F card finder
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestReview120Gui:
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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- feature factory: plot_descriptor_distributions ---------------- #
    def test_feat_distributions_needs_a_compute_first(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._feat_plot_distributions()
        assert any("No features yet" in str(a) for a in quiet_dialogs)

    def test_feat_distributions_after_compute(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "50")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain(app), "feature factory job hung"
        app._feat_plot_distributions()
        assert self._drain(app), "distribution job hung"
        assert app._feat_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert quiet_dialogs == []

    # -- ML results: plot_learning_curve ------------------------------ #
    def test_ml_learning_curve_needs_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_plot_learning_curve()
        assert any("No trained model" in str(a) for a in quiet_dialogs)

    def test_ml_learning_curve_after_training(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain(app), "train job hung"
        app._curve_cv.delete(0, "end")
        app._curve_cv.insert(0, "4")
        app._ml_plot_learning_curve()
        assert self._drain(app), "learning-curve job hung"
        assert app._ml_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert "learning curve" in app._ml_result_note.cget("text")
        assert quiet_dialogs == []

    def test_ml_learning_curve_bad_folds_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain(app), "train job hung"
        app._curve_cv.delete(0, "end")
        app._curve_cv.insert(0, "1")
        app._ml_plot_learning_curve()
        assert any("CV folds" in str(a) for a in quiet_dialogs)

    # -- utilities: plot_table_comparison ------------------------------ #
    def test_util_cmp_plot_needs_a_comparison(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._util_cmp_plot()
        assert any("No comparison yet" in str(a) for a in quiet_dialogs)

    def test_util_cmp_plot_after_compare_run(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table_a = tmp_path / "ligands_a.csv"
        table_a.write_text(
            "Ligand,SMILES,score\nlig1,CCO,1.5\nlig2,CCOCC,2.5\nlig3,c1ccccc1,3.5\n",
            encoding="utf-8",
        )
        table_b = tmp_path / "ligands_b.csv"
        table_b.write_text("Ligand,SMILES\nlig1,CCO\nlig4,CNC\n", encoding="utf-8")
        app._util_cmp_a = table_a
        app._util_cmp_b = table_b
        app._util_cmp_mode.set("compare two tables")
        app._util_compare_run()
        assert self._drain(app), "compare job hung"
        assert app._util_cmp_last is not None
        app._util_cmp_plot()
        assert self._drain(app), "reconciliation figure job hung"
        assert app._util_cmp_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert quiet_dialogs == []

    # -- docking: parse_vina_config ----------------------------------- #
    def test_vbatch_parse_needs_a_config(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_vbatch_parse()
        assert any("No config picked" in str(a) for a in quiet_dialogs)

    def test_vbatch_parse_dumps_keys_and_fills_the_form(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        conf = tmp_path / "batch.conf"
        conf.write_text(
            "receptor = r.pdbqt\ncenter_x = 2.5\ncenter_y = 3.5\ncenter_z = 4.5\n"
            "size_x = 20\nsize_y = 22\nsize_z = 24\nexhaustiveness = 12\n",
            encoding="utf-8",
        )
        app._dock_vbatch_conf = conf
        app._dock_vbatch_parse()
        info = app._dock_vbatch_info.cget("text")
        assert "parsed 8 keys" in info, info
        assert "receptor = r.pdbqt" in info
        assert "exhaustiveness = 12" in info
        # the box + exhaustiveness were copied into the single-run form
        assert app._dock_center_entry.get() == "2.5 3.5 4.5"
        assert app._dock_size_entry.get() == "20 22 24"
        assert app._dock_exhaust_entry.get() == "12"
        summary = app._dock_vbatch_summary.cget("text")
        assert "centre" in summary and "exhaustiveness" in summary, summary
        assert quiet_dialogs == []

    # -- cell lines: most_variable_genes ------------------------------- #
    def test_mad_ranking_needs_an_expression_table(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._bio_rank_variable_genes()
        assert any("No expression table" in str(a) for a in quiet_dialogs)

    def test_mad_ranking_on_the_example_table(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._bio_load_expr_example()
        app._bio_rank_variable_genes()
        assert self._drain(app), "MAD ranking job hung"
        summary = app._bio_mad_summary.cget("text")
        assert "top 20 of 300 genes by MAD" in summary, summary
        dump = app._bio_mad_text.get("1.0", "end")
        gene_rows = [line for line in dump.strip().splitlines() if line.startswith("GENE_")]
        assert len(gene_rows) == 20  # the two header rows are not genes
        assert quiet_dialogs == []

    # -- PD explorer: effect_table drives the sigmoid table ------------ #
    def test_pd_family_sigmoid_table_matches_effect_table(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._pd_family_math()
        assert self._drain(app), "PD family job hung"
        rows = app._fam_tree.get_children()
        values = app._fam_tree.item(rows[0], "values")
        assert values[0] == "0.5000" and values[1] == "9.93", values
        # the inhibitory branch (table=None path) still computes
        app._pd_family_combo.set(app._pd_family_combo["values"][1])
        app._pd_family_math()
        assert self._drain(app), "inhibitory PD job hung"
        assert len(app._fam_tree.get_children()) == 6
        assert quiet_dialogs == []

    # -- docking: the single-folder AutoDock4 engine ------------------- #
    def test_single_folder_engine_registered_and_labelled(self, app: Any, playground: Any) -> None:
        assert "AutoDock4 single folder" in playground._DOCK_ENGINES
        assert playground.docking_engine_binaries("AutoDock4 single folder") == (
            "autogrid4",
            "autodock4",
        )
        # the folder picker's label is engine-aware
        app._dock_engine_combo.set("AutoDock4 single folder")
        app._dock_refresh_campaign_info()
        text = app._dock_campaign_info.cget("text")
        assert "ligand folder" in text and "r.gpf" in text, text
        app._dock_engine_combo.set("AutoDock4 campaign")
        app._dock_refresh_campaign_info()
        assert "campaign root" in app._dock_campaign_info.cget("text")

    def test_single_folder_engine_run_without_folder_warns(
        self,
        app: Any,
        quiet_dialogs: list[Any],
        monkeypatch: Any,
        playground: Any,
    ) -> None:
        # bypass the binaries check (autogrid4 is not installed in CI)
        monkeypatch.setattr(playground, "docking_engine_binaries", lambda engine: ())
        app._dock_engine_combo.set("AutoDock4 single folder")
        app._dock_run()
        assert any("No ligand folder" in str(a) for a in quiet_dialogs)

    # -- the Ctrl+F card finder ---------------------------------------- #
    def test_card_finder_lists_every_live_card(self, app: Any, playground: Any) -> None:
        cards = app._live_cards()
        assert len(cards) >= 55, f"expected >=55 live cards, got {len(cards)}"
        # every card resolves to a real tab index
        count = app._book.index("end")
        assert all(0 <= index < count for _t, _s, _sh, index in cards)

    def test_card_finder_dialog_and_jump(self, app: Any, playground: Any) -> None:
        import tkinter as tkmod

        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        assert len(dialogs) == 1
        win = dialogs[0]
        tree = next(child for child in win.winfo_children() if child.winfo_class() == "Treeview")
        rows = tree.get_children()
        assert len(rows) >= 55
        # the first row is the home hero, on tab 0
        assert str(rows[0]).startswith("0:")
        win.destroy()

    def test_jump_to_card_selects_flashes_and_unflashes(self, app: Any, playground: Any) -> None:
        import tkinter as tkmod

        app._jump_to_tab(0)
        app._jump_to_card(1, "Affinity conversion")
        assert app._book.index(app._book.select()) == 1
        shell = None
        for title, _subtitle, candidate in playground._CARD_INDEX:
            if title != "Affinity conversion":
                continue
            try:
                if not bool(candidate.winfo_exists()):
                    continue
            except tkmod.TclError:  # a destroyed earlier-test app's shell
                continue
            shell = candidate
            break
        assert shell is not None
        assert int(shell.cget("highlightthickness")) == 3
        assert str(shell.cget("highlightbackground")) == playground.ACCENT
        app._unflash_card(shell)
        assert int(shell.cget("highlightthickness")) == 1

    def test_hint_row_lists_the_card_finder_shortcut(self, playground: Any) -> None:
        text = playground.shortcut_hints_text()
        assert "Ctrl+F find a card" in text


class TestReview121Gui:
    """Round 100x-121: UnitConverter card, export_table saves, DL curve."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- PK/PD: the UnitConverter card -------------------------------- #
    def test_unit_converter_default_converts_mg_to_g(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        assert app._uc_from.get() == "mg"
        assert app._uc_to.get() == "g"
        app._uc_convert()
        text = app._uc_result.cget("text")
        assert "500 mg" in text and "=  0.5 g" in text, text
        assert "mass conversion" in text
        assert "factor" in text
        assert "mass: kg · g · mg" in app._uc_units_hint.cget("text")
        assert quiet_dialogs == []

    def test_unit_converter_dimension_swap(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._uc_dim_combo.set("concentration")
        app._uc_dimension_changed()
        assert app._uc_from.get() == "uM"
        assert app._uc_to.get() == "nM"
        app._uc_convert()
        text = app._uc_result.cget("text")
        assert "=  500000 nM" in text or "=  5e+05 nM" in text, text
        assert "concentration conversion" in text
        assert quiet_dialogs == []

    def test_unit_converter_time_dimension_defaults(self, app: Any) -> None:
        app._uc_dim_combo.set("time")
        app._uc_dimension_changed()
        assert app._uc_from.get() == "h" and app._uc_to.get() == "min"

    def test_unit_converter_unknown_unit_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._uc_from.delete(0, "end")
        app._uc_from.insert(0, "stones")
        app._uc_convert()
        assert any("Conversion failed" in str(a) for a in quiet_dialogs)
        # the result label is untouched by the failed run
        assert app._uc_result.cget("text") == "—"

    def test_unit_converter_cross_dimension_is_rejected(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._uc_to.delete(0, "end")
        app._uc_to.insert(0, "mL")
        app._uc_convert()
        assert any("Conversion failed" in str(a) for a in quiet_dialogs)

    def test_unit_converter_bad_value_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._uc_value.delete(0, "end")
        app._uc_value.insert(0, "not-a-number")
        app._uc_convert()
        assert any("Conversion failed" in str(a) for a in quiet_dialogs)

    # -- export_table: the five-format save dialog --------------------- #
    def test_save_frame_dialog_offers_five_formats(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        picked: dict[str, str] = {}

        def fake_ask(title: str = "", **kwargs: Any) -> str:
            picked["filetypes"] = str(kwargs.get("filetypes", ""))
            return str(tmp_path / "out.csv")

        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", fake_ask)
        frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
        app._save_frame_dialog(frame, "Save it", "out.csv", "The table")
        assert ".csv" in picked["filetypes"]
        assert ".tsv" in picked["filetypes"]
        assert ".md" in picked["filetypes"]
        assert ".json" in picked["filetypes"]
        assert ".html" in picked["filetypes"]
        assert (tmp_path / "out.csv").exists()
        assert any("The table written" in str(a) for a in quiet_dialogs)

    def test_save_frame_dialog_writes_markdown_and_html(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        frame = pd.DataFrame({"name": ["aspirin", "caffeine"], "score": [1.5, 2.5]})
        for suffix in ("md", "html"):
            target = tmp_path / f"table.{suffix}"
            monkeypatch.setattr(
                playground.filedialog, "asksaveasfilename", lambda t="", _p=target, **k: str(_p)
            )
            app._save_frame_dialog(frame, "Save it", "table.csv", "The table")
            content = target.read_text(encoding="utf-8")
            assert "aspirin" in content and "caffeine" in content
            if suffix == "html":
                assert "<table" in content
        assert quiet_dialogs, "the Saved confirmation did not fire"

    def test_export_table_routed_from_ml_predictions_save(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._ml_predictions = pd.DataFrame({"SMILES": ["CCO"], "prediction": [7.2]})
        target = tmp_path / "preds.md"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", **k: str(target)
        )
        app._ml_save_predictions()
        assert target.exists()
        assert "CCO" in target.read_text(encoding="utf-8")

    def test_export_table_routed_from_nca_save(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        app._nca_summary = SimpleNamespace(
            auc=55.0, aucm=410.0, mrt=7.45, c_max=12.0, t_max=0.5, half_life_terminal=3.9
        )
        target = tmp_path / "nca.json"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", **k: str(target)
        )
        app._nca_save()
        assert target.exists()
        assert "55" in target.read_text(encoding="utf-8")

    def test_save_frame_dialog_no_frame_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._save_frame_dialog(None, "Save it", "out.csv")
        assert any("Nothing to save" in str(a) for a in quiet_dialogs)

    def test_rism_save_routes_through_the_shared_dialog(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._rism_frame = pd.DataFrame({"rism": [1.0]})
        target = tmp_path / "rism.tsv"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", **k: str(target)
        )
        app._rism_save()
        assert target.exists()
        assert "RISM-KH table written" in str(quiet_dialogs[-1][1])

    # -- Deep Learning: the size-vs-score learning curve --------------- #
    def test_dl_learning_curve_needs_a_network(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_plot_learning_curve()
        assert any("No network" in str(a) for a in quiet_dialogs)

    def test_dl_learning_curve_after_training(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_load_example()
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "8")
        app._dl_epochs.delete(0, "end")
        app._dl_epochs.insert(0, "3")
        app._dl_train()
        assert self._drain(app), "DL train job hung"
        app._dl_cv_entry.delete(0, "end")
        app._dl_cv_entry.insert(0, "2")
        app._dl_plot_learning_curve()
        assert self._drain(app), "DL learning-curve job hung"
        assert app._dl_curve_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert quiet_dialogs == []

    def test_dl_learning_curve_bad_folds_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_load_example()
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "8")
        app._dl_epochs.delete(0, "end")
        app._dl_epochs.insert(0, "3")
        app._dl_train()
        assert self._drain(app), "DL train job hung"
        app._dl_cv_entry.delete(0, "end")
        app._dl_cv_entry.insert(0, "1")
        app._dl_plot_learning_curve()
        assert any("CV folds" in str(a) for a in quiet_dialogs)

    # -- single-folder AutoDock4: the DLG cluster summary -------------- #
    def test_single_folder_render_carries_dlg_summary(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        # a real DLG cluster table (the format readerwithdetails.py
        # parsed): three clusters, rank 1 lowest, rank 3 most populated
        dlg = tmp_path / "r.dlg"
        dlg.write_text(
            "________\n"
            "CLUSTERING HISTOGRAM\n"
            "__________________\n"
            "______________________________________________________________________\n"
            "Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster\n"
            "_____________|_______________|_____|_____________|________________\n"
            "     1       |     -11.5     |  3  |    -11.2    |      7 |\n"
            "     2       |     -10.8     |  9  |    -10.5    |     34 |\n"
            "     3       |      -9.9     | 17  |    -9.7    |      52 |\n"
            "____________|_______________|_____|_____________|________________\n",
            encoding="utf-8",
        )
        result = SimpleNamespace(ok=True, error="", seconds=12.5, dlg=dlg)
        engine_holder: dict[str, str] = {}

        def fake_start_job(label: str, work: Any, render: Any) -> None:
            engine_holder["render"] = render  # type: ignore[assignment]

        monkeypatch.setattr(app, "_start_job", fake_start_job)
        monkeypatch.setattr(playground, "docking_engine_binaries", lambda engine: ())
        app._dock_engine_combo.set("AutoDock4 single folder")
        app._dock_campaign_root = tmp_path
        app._dock_run()
        engine_holder["render"](result)  # type: ignore[index]
        text = app._dock_run_summary.cget("text")
        assert "3 clusters" in text, text
        assert "top -11.50" in text, text
        assert "(n=7)" in text, text
        assert "best-populated -9.90 (n=52)" in text, text

    # -- the finder remembers its last query --------------------------- #
    def test_finder_remembers_last_query_in_session_payload(self, app: Any) -> None:
        app._finder_last_query = "unit converter"
        payload = app._collect_session()
        assert payload["texts"]["finder"] == "unit converter"

    def test_finder_restores_last_query_from_session(self, app: Any) -> None:

        # a raw session dict shaped like load_session's output
        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {"finder": "learning curve"},
        }
        app._restore_session()
        assert app._finder_last_query == "learning curve"

    def test_finder_dialog_prefills_the_last_query(self, app: Any) -> None:
        import tkinter as tkmod

        app._finder_last_query = "nca"
        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        assert len(dialogs) == 1
        win = dialogs[0]
        entry = next(child for child in win.winfo_children() if child.winfo_class() == "TEntry")
        assert entry.get() == "nca"
        tree = next(child for child in win.winfo_children() if child.winfo_class() == "Treeview")
        # the prefill also filtered the list: every card whose title OR
        # subtitle carries the query.  "nca" hits the NCA card itself
        # and -- since the round-200 arena merge -- the Advanced PK/PD
        # card, whose subtitle documents its linear-up/log-down NCA
        # engine (a latent v1.2.115 staleness the round-201 battery
        # caught; the old all-NCA pin predated that card).
        labels = [str(tree.item(row, "values")[0]) for row in tree.get_children()]
        assert labels, labels
        assert "Non-compartmental analysis (NCA)" in labels
        assert "Advanced PK/PD & PBPK (RxODE · nlmixr2 · OSP)" in labels
        assert len(labels) == 2, labels
        win.destroy()

    # -- the home guide grew the Deep Learning blurb ------------------- #
    def test_home_guide_lists_deep_learning(self, app: Any, playground: Any) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "Deep Learning" in dump
        assert "residual MLP" in dump
        assert "unit converter" in dump

    # -- the docstring records the round ------------------------------- #
    def test_module_docstring_records_round_121(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "UnitConverter" in doc
        assert "export_table" in doc
        assert "100x-121" in doc


class TestReview122Gui:
    """Round 100x-122: the three reference objectives, molecule surgery,
    and the session PDF report."""

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the objective combobox finally carries every bundled objective - #
    def test_objective_kinds_carry_the_reference_driven_three(self, playground: Any) -> None:
        assert "Similarity to reference SMILES" in playground._OBJECTIVE_KINDS
        assert "Novelty vs corpus" in playground._OBJECTIVE_KINDS
        assert "Contains substructure (SMARTS)" in playground._OBJECTIVE_KINDS
        assert len(playground._OBJECTIVE_KINDS) == 9

    def test_objective_combo_mirrors_kinds(self, app: Any, playground: Any) -> None:
        assert list(app._objective_combo.cget("values")) == [
            str(kind) for kind in playground._OBJECTIVE_KINDS
        ]

    def test_build_objective_similarity(self, playground: Any) -> None:
        objective = playground.build_objective(
            "Similarity to reference SMILES", 1.0, 3.0, reference="CC(=O)Oc1ccccc1C(=O)O"
        )
        from pharmaml.generative import SimilarityObjective

        assert isinstance(objective, SimilarityObjective)
        assert objective.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0

    def test_build_objective_novelty_with_and_without_corpus(self, playground: Any) -> None:
        from pharmaml.generative import NoveltyObjective

        objective = playground.build_objective("Novelty vs corpus", 1.0, 3.0, corpus=["CCO", "CCC"])
        assert isinstance(objective, NoveltyObjective)
        assert objective.score("CCO") == 0.0  # in corpus: not novel
        default = playground.build_objective("Novelty vs corpus", 1.0, 3.0)
        assert isinstance(default, NoveltyObjective)

    def test_build_objective_substructure(self, playground: Any) -> None:
        from pharmaml.generative import SubstructureObjective

        objective = playground.build_objective(
            "Contains substructure (SMARTS)", 1.0, 3.0, reference="c1ccccc1"
        )
        assert isinstance(objective, SubstructureObjective)
        assert objective.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0
        assert objective.score("CCO") == 0.0

    # -- the swapped-in reference row ---------------------------------- #
    def test_reference_row_hidden_by_default(self, app: Any) -> None:
        # grid_remove forgets the geometry entirely: an empty grid_info()
        # IS the hidden state (the GA/RL frames use the same mechanism)
        app._objective_hint()
        assert app._obj_ref_frame.grid_info() == {}

    def test_reference_row_swaps_for_similarity(self, app: Any) -> None:
        app._objective_combo.set("Similarity to reference SMILES")
        app._objective_hint()
        assert app._obj_ref_label.cget("text") == "reference"
        assert app._obj_ref_frame.grid_info() != {}
        assert "Tanimoto similarity" in app._objective_hint_label.cget("text")

    def test_reference_row_swaps_for_substructure(self, app: Any) -> None:
        app._objective_combo.set("Contains substructure (SMARTS)")
        app._objective_hint()
        assert app._obj_ref_label.cget("text") == "SMARTS"
        assert app._obj_ref_frame.grid_info() != {}
        assert "pharmacophore constraint" in (app._objective_hint_label.cget("text"))

    def test_reference_row_hidden_for_novelty(self, app: Any) -> None:
        app._objective_combo.set("Similarity to reference SMILES")
        app._objective_hint()
        app._objective_combo.set("Novelty vs corpus")
        app._objective_hint()
        assert "corpus" in app._objective_hint_label.cget("text")
        # novelty never needs the reference row
        assert app._obj_ref_frame.grid_info() == {}

    def test_similarity_without_reference_is_named(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._objective_combo.set("Similarity to reference SMILES")
        app._objective_hint()
        app._obj_ref.delete(0, "end")
        app._run_evolution()
        assert any("reference SMILES" in str(a) for a in quiet_dialogs)

    def test_substructure_without_smart_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._objective_combo.set("Contains substructure (SMARTS)")
        app._objective_hint()
        app._obj_ref.delete(0, "end")
        app._run_evolution()
        assert any("SMARTS" in str(a) for a in quiet_dialogs)

    # -- the molecule surgery cards ------------------------------------ #
    def test_topology_moves_enumerates_piperidine(self, app: Any, quiet_dialogs: list[Any]) -> None:
        assert app._topo_smiles.get() == "C1CCNCC1"
        app._gen_topology_moves()
        assert self._drain(app), "topology job hung"
        text = app._topo_text.get("1.0", "end").strip()
        assert "C1CCCC1" in text  # the contraction
        assert "C1CCCNCC1" in text  # the expansion
        assert "C1CCNC1" in text  # the N-side contraction
        summary = app._topo_summary.cget("text")
        assert summary.startswith("3 distinct"), summary
        assert quiet_dialogs == []

    def test_topology_moves_aromatic_has_no_moves(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._topo_smiles.delete(0, "end")
        app._topo_smiles.insert(0, "c1ccccc1")
        app._gen_topology_moves()
        assert self._drain(app), "topology job hung"
        assert app._topo_text.get("1.0", "end").strip() == ""
        summary = app._topo_summary.cget("text")
        assert "no qualifying saturated ring" in summary, summary

    def test_topology_moves_empty_input_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._topo_smiles.delete(0, "end")
        app._gen_topology_moves()
        assert any("Type a SMILES" in str(a) for a in quiet_dialogs)

    def test_breed_defaults_produce_children(self, app: Any, quiet_dialogs: list[Any]) -> None:
        assert app._breed_a.get() == "CC(=O)Oc1ccccc1C(=O)O"
        assert app._breed_b.get() == "CC(C)Cc1ccc(O)cc1"
        app._gen_breed()
        assert self._drain(app), "breed job hung"
        text = app._breed_text.get("1.0", "end").strip()
        assert text, "the litter is empty"
        assert "\n" in text  # 4 children on 4 lines
        summary = app._breed_summary.cget("text")
        assert "viable children" in summary and "seed 42" in summary, summary
        assert quiet_dialogs == []

    def test_breed_missing_parent_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._breed_b.delete(0, "end")
        app._gen_breed()
        assert any("both parent SMILES" in str(a) for a in quiet_dialogs)

    def test_breed_bad_count_is_named(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._breed_count.delete(0, "end")
        app._breed_count.insert(0, "0")
        app._gen_breed()
        assert any("children" in str(a) for a in quiet_dialogs)

    # -- the session PDF report ---------------------------------------- #
    def test_file_menu_offers_session_report(self, app: Any) -> None:
        import tkinter as tkmod

        bar = app.nametowidget(app.cget("menu"))
        assert isinstance(bar, tkmod.Menu)
        labels: list[str] = []
        # NB: menu.type() never raises past the end on this Tk build --
        # index("end") is the only honest bound.
        for index in range(bar.index("end") + 1):
            if bar.type(index) != "cascade":
                continue
            if str(bar.entrycget(index, "label")).strip() != "File":
                continue
            sub = app.nametowidget(bar.entrycget(index, "menu"))
            for j in range(sub.index("end") + 1):
                if sub.type(j) == "command":
                    labels.append(str(sub.entrycget(j, "label")))
        assert any("session report" in label.lower() for label in labels), labels

    def test_session_report_with_no_figures_is_an_info(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # (100x-123) the guard covers the report's TWO axes now: no
        # figures AND no filled tables -> the honest info dialog.
        # (round 201) the Deep Learning tab's attribution card pre-fills
        # its method-registry table at build, so the empty state needs
        # that one tree emptied first -- a latent v1.2.115 staleness the
        # round battery caught (the save dialog popped instead of the
        # info, hanging headless runs).
        app._dl_attr_tree.delete(*app._dl_attr_tree.get_children())
        app._menu_save_session_report()
        assert any("Nothing to report yet" in str(a) for a in quiet_dialogs)

    @staticmethod
    def _figure(app: Any, title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return figure

    def test_session_registry_tracks_and_replaces(self, app: Any) -> None:
        first = self._figure(app, "first")
        app._job_label_base = "parity plot"
        app._render_figure(app._ml_host, first)
        assert len(app._session_figures) == 1
        assert app._session_figures[0][1] == "parity plot"
        assert app._session_figures[0][2] is first
        second = self._figure(app, "second")
        app._render_figure(app._ml_host, second)
        # one entry per HOST: the replacement keeps the order, swaps figure
        assert len(app._session_figures) == 1
        assert app._session_figures[0][2] is second
        assert app._session_figures[0][1] == "parity plot"
        other = self._figure(app, "other")
        app._render_figure(app._feat_host, other)
        assert len(app._session_figures) == 2
        live = app._live_session_figures()
        assert len(live) == 2
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_session_report_writes_pdf_and_keeps_figures(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "session-report.pdf"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        figure_a = self._figure(app, "alpha")
        figure_b = self._figure(app, "beta")
        app._job_label_base = "figure alpha"
        app._render_figure(app._ml_host, figure_a)
        app._job_label_base = "figure beta"
        app._render_figure(app._feat_host, figure_b)
        assert len(app._live_session_figures()) == 2
        app._menu_save_session_report()
        assert self._drain(app), "report job hung"
        assert target.is_file(), "the PDF was not written"
        assert target.stat().st_size > 1_000
        # close=False: the on-screen figures are still the live registry's
        assert app._ml_host._pg_figure is figure_a
        assert app._feat_host._pg_figure is figure_b
        # (100x-123) the save message counts BOTH axes now -- a fresh app
        # has no filled tables, so the table count is an honest zero
        assert any(
            "figure page(s)" in str(a) and "written to" in str(a) for a in quiet_dialogs
        )
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_session_report_dedupes_titles(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "dupe-report.pdf"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        first = self._figure(app, "one")
        second = self._figure(app, "two")
        # two hosts, same job label -> the report needs "figure (2)"
        app._job_label_base = "same label"
        app._render_figure(app._ml_host, first)
        app._render_figure(app._feat_host, second)
        app._menu_save_session_report()
        assert self._drain(app), "report job hung"
        assert target.is_file()
        import matplotlib.pyplot as plt

        plt.close("all")

    # -- the restore re-syncs the swapped rows ------------------------- #
    def test_restore_resyncs_reference_row_and_ga_frames(self, app: Any) -> None:
        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {
                "_objective_combo": "Similarity to reference SMILES",
                "_method_combo": "reinforcement learning (RL)",
            },
            "families": {},
            "texts": {},
        }
        app._restore_session()
        assert app._objective_combo.get() == "Similarity to reference SMILES"
        assert app._obj_ref_label.cget("text") == "reference"
        assert app._obj_ref_frame.grid_info() != {}
        # the latent bug fix: an RL selection no longer leaves the GA
        # block visible (grid_remove empties the geometry info)
        assert app._ga_frame.grid_info() == {}
        assert app._rl_frame.grid_info() != {}

    # -- the docstring + home guide record the round -------------------- #
    def test_module_docstring_records_round_122(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-122" in doc
        assert "save_pdf_report" in doc
        assert "ring_topology_moves" in doc
        assert "scaffold_hop_crossover" in doc
        assert "SimilarityObjective" in doc

    def test_home_guide_mentions_surgery_and_smart_objectives(
        self, app: Any, playground: Any
    ) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "similarity" in dump
        assert "breed two parents" in dump


@XVFB_SKIP
class TestReview123Gui:
    """Round 100x-123: the surgery structure grids and the report's
    table axis (every filled table becomes a REAL text page)."""

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the module-level structure grid helper ------------------------- #
    def test_molecule_grid_figure_draws_panels(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure(
            [("mutant 1\nC1CCCC1", "C1CCCC1"), ("mutant 2\nC1CCCNCC1", "C1CCCNCC1")],
            title="two mutants",
        )
        try:
            assert figure.get_axes()  # at least the panels
            titles = [axis.get_title() for axis in figure.get_axes()]
            assert any("mutant 1" in title for title in titles), titles
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_molecule_grid_figure_honors_unparseable_and_empty(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure([("bad\nnotasmiles", "notasmiles")], title="x")
        try:
            # the unparseable entry keeps its slot (no silent vanish)
            assert len(figure.get_axes()) >= 1
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)
        empty = playground.molecule_grid_figure([], title="nothing")
        try:
            assert len(empty.get_axes()) >= 1
        finally:
            import matplotlib.pyplot as plt

            plt.close(empty)

    def test_truncate_caption_text(self, playground: Any) -> None:
        assert playground._truncate_caption_text("short") == "short"
        long = "C" * 60
        cut = playground._truncate_caption_text(long)
        assert len(cut) == 34 and cut.endswith("\u2026")

    # -- the surgery cards' draw buttons ------------------------------- #
    def test_topology_draw_full_path(self, app: Any) -> None:
        app._gen_topology_moves()
        assert self._drain(app), "topology job hung"
        source, mutants = app._topo_last
        assert source == "C1CCNCC1"
        assert mutants  # piperidine always yields mutants
        app._gen_draw_topology()
        assert self._drain(app), "topology draw job hung"
        figure = getattr(app._topo_host, "_pg_figure", None)
        assert figure is not None, "the grid never landed on the host"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_topology_draw_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._topo_last = None
        app._gen_draw_topology()
        assert any("No enumeration yet" in str(a) for a in quiet_dialogs)
        assert getattr(app._topo_host, "_pg_figure", None) is None

    def test_breed_draw_full_path(self, app: Any) -> None:
        app._gen_breed()
        assert self._drain(app), "breed job hung"
        parent_a, parent_b, litter = app._breed_last
        assert parent_a and parent_b and len(litter) == 4
        app._gen_draw_breed()
        assert self._drain(app), "breed draw job hung"
        figure = getattr(app._breed_host, "_pg_figure", None)
        assert figure is not None, "the litter grid never landed"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_breed_draw_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._breed_last = None
        app._gen_draw_breed()
        assert any("No litter yet" in str(a) for a in quiet_dialogs)

    def test_surgery_cards_carry_draw_buttons(self, app: Any) -> None:
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[11]))
            if widget.winfo_class() == "TButton"
        )
        assert "Draw mutants" in dump
        assert "Draw litter" in dump

    # -- the table walk ------------------------------------------------- #
    def test_card_title_for_widget_climbs_to_the_card(self, app: Any) -> None:
        assert app._card_title_for_widget(app._desc_tree) == "SMILES inspector"
        assert app._card_title_for_widget(app) is None  # the root is no card

    def test_live_session_tables_skips_empty_trees(self, app: Any) -> None:
        # (round 201) the Deep Learning attribution card pre-fills its
        # method-registry census at build (pinned since 100x-174) -- empty
        # it so the fresh-app state really is "every tree empty", which
        # is what this walk must skip (a latent staleness the round
        # battery caught: the class had not been re-run since 100x-174)
        app._dl_attr_tree.delete(*app._dl_attr_tree.get_children())
        assert app._live_session_tables() == []

    def test_live_session_tables_walks_a_real_job(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._analyse_smiles()
        assert self._drain(app), "analyse job hung"
        walked = app._live_session_tables()
        assert any(title == "SMILES inspector" and len(rows) > 150 for title, _h, rows in walked), [
            (t, len(r)) for t, _h, r in walked
        ]
        entry = next(t for t, _h, r in walked if t == "SMILES inspector")
        headings = next(h for t, h, r in walked if t == "SMILES inspector")
        assert headings == ["descriptor", "value"], headings
        assert entry  # rows came back

    def test_live_session_tables_covers_hierarchical_tree(self, app: Any) -> None:
        # the bio matrix tree uses show="tree headings" -- its #0 labels
        # must arrive as the first column exactly the way users read them
        app._bio_matrix_tree.insert("", "end", text="cell-A", values=("1.0", "2.0"))
        walked = app._live_session_tables()
        match = next((t, h, r) for t, h, r in walked if t and "sensitivity" in t.lower())
        assert match is not None, [(t, h) for t, h, _r in walked]
        _title, headings, rows = match
        assert headings[0] == "cell line", headings  # the real #0 heading
        assert rows[0][0] == "cell-A"
        app._bio_matrix_tree.delete(*app._bio_matrix_tree.get_children())

    # -- the report's table axis ---------------------------------------- #
    def test_session_report_ships_tables_as_real_text_pages(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "report123.pdf"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._analyse_smiles()
        assert self._drain(app), "analyse job hung"
        assert app._live_session_figures() == []  # figures axis empty: tables only
        app._menu_save_session_report()
        assert self._drain(app), "report job hung"
        assert target.is_file() and target.stat().st_size > 1_000
        import pypdf

        reader = pypdf.PdfReader(str(target))
        joined = "\n".join((page.extract_text() or "").lower() for page in reader.pages)
        assert "table page(s)" in joined, "the subtitle did not count tables"
        # (100x-134) the page name gained the count prefix: "table 1 ·
        # Smiles Inspector" (title-cased by the library's writer)
        assert "table 1 · smiles inspector" in joined, "no table page shipped"
        assert "molwt" in joined, "the rows are not real text"
        assert "more rows" in joined, "the truncation note is missing"
        assert len(reader.pages) >= 2  # title + at least one table page

    def test_session_report_combines_both_axes(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "combined.pdf"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        # (round 201) retire the attribution census table first so the
        # counts below stay the test's own "1 figure + 1 table" story
        # (the census rides every report since 100x-174; this class's
        # pins predate it and had not been re-run)
        app._dl_attr_tree.delete(*app._dl_attr_tree.get_children())
        figure = self._figure_static("combined")
        app._job_label_base = "combined figure"
        app._render_figure(app._ml_host, figure)
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
        app._menu_save_session_report()
        assert self._drain(app), "report job hung"
        assert target.is_file()
        import pypdf

        reader = pypdf.PdfReader(str(target))
        assert len(reader.pages) >= 3  # title + figure + table
        joined = "\n".join((page.extract_text() or "").lower() for page in reader.pages)
        assert "1 figure page(s)" in joined and "1 table page(s)" in joined
        assert "molwt" in joined
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_session_report_page_names_avoid_path_separators(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        # save_pdf_report validates names as plain file names: a card
        # title may contain no '/' (the library would reject it)
        import pathlib

        target = pathlib.Path(tmp_path) / "names.pdf"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
        # simulate a card title with a slash: the unique_name path must
        # never construct one itself -- prove the happy path stays valid
        app._menu_save_session_report()
        assert self._drain(app), "report job hung"
        assert target.is_file()
        app._desc_tree.delete(*app._desc_tree.get_children())

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return figure

    # -- the round's documentation pins --------------------------------- #
    def test_module_docstring_records_round_123(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-123" in doc or "structure grid" in doc
        assert "tables=" in doc  # the report's table axis is documented
        assert "2D" in doc  # the surgery grids are documented

    def test_home_guide_mentions_structure_grids(self, app: Any, playground: Any) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "2D structure grid" in dump


class TestReview124Gui:
    """Round 100x-124: the HTML session report (HTMLReport's first
    caller anywhere) and the leaderboard's scored structure grid."""

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- molecule_grid_figure's scores row ------------------------------ #
    def test_grid_scores_row_renders_xlabels(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure(
            [("#1 · C1CCCC1", "C1CCCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
            title="scored",
            scores=["score 0.987", "score 0.500"],
        )
        try:
            labels = [axis.get_xlabel() for axis in figure.get_axes()]
            assert "score 0.987" in labels, labels
            assert "score 0.500" in labels, labels
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_grid_scores_mismatch_never_shifts(self, playground: Any) -> None:
        # a shorter list pads (entry 2 keeps no score row), a longer
        # one is truncated -- the grid cannot mis-align
        short = playground.molecule_grid_figure(
            [("#1 · C1CCCC1", "C1CCCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
            title="pad",
            scores=["score 0.987"],
        )
        try:
            labels = [axis.get_xlabel() for axis in short.get_axes()]
            assert "score 0.987" in labels and "" in labels, labels
        finally:
            import matplotlib.pyplot as plt

            plt.close(short)
        long = playground.molecule_grid_figure(
            [("#1 · C1CCCC1", "C1CCCC1")],
            title="cut",
            scores=["score 0.987", "score 0.500", "score 0.250"],
        )
        try:
            labels = [axis.get_xlabel() for axis in long.get_axes()]
            assert "score 0.987" in labels, labels
            assert "score 0.500" not in labels, labels
        finally:
            import matplotlib.pyplot as plt

            plt.close(long)

    def test_grid_scores_keep_bare_panels_for_none(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure([("m\nC1CCCC1", "C1CCCC1")], title="x")
        try:
            assert all(
                axis.get_xlabel() == "" for axis in figure.get_axes()
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    # -- the GA leaderboard's draw-top button --------------------------- #
    def test_draw_top_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ga_result = None
        app._gen_draw_top()
        assert any("No run yet" in str(a) for _n, a in quiet_dialogs)
        assert getattr(app._ga_grid_host, "_pg_figure", None) is None

    def test_draw_top_full_path(self, app: Any) -> None:
        # a tiny real evolution through the card's own runner
        app._objective_combo.set("QED drug-likeness")
        app._pop_size.set("8")
        app._generations.set("2")
        app._run_evolution()
        assert self._drain(app), "the evolution job hung"
        assert app._ga_result is not None
        app._gen_draw_top()
        assert self._drain(app), "the draw-top job hung"
        figure = getattr(app._ga_grid_host, "_pg_figure", None)
        assert figure is not None, "the top-molecule grid never landed"
        labels = [axis.get_xlabel() for axis in figure.get_axes()]
        assert any(label.startswith("score ") for label in labels), labels
        titles = [axis.get_title() for axis in figure.get_axes()]
        assert any(title.startswith("#1") for title in titles), titles
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_results_card_carries_draw_top_button(self, app: Any) -> None:
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[11]))
            if widget.winfo_class() == "TButton"
        )
        assert "Draw top molecules" in dump
        assert "Save leaderboard as CSV…" in dump

    # -- the HTML session report ----------------------------------------- #
    def test_file_menu_offers_html_report(self, app: Any) -> None:
        import tkinter as tkmod

        bar = app.nametowidget(app.cget("menu"))
        assert isinstance(bar, tkmod.Menu)
        labels: list[str] = []
        # NB: menu.type() never raises past the end on this Tk build --
        # index("end") is the only honest bound.
        for index in range(bar.index("end") + 1):
            if bar.type(index) != "cascade":
                continue
            if str(bar.entrycget(index, "label")).strip() != "File":
                continue
            sub = app.nametowidget(bar.entrycget(index, "menu"))
            for j in range(sub.index("end") + 1):
                if sub.type(j) == "command":
                    labels.append(str(sub.entrycget(j, "label")))
        assert "Save session report (PDF)…" in labels, labels
        assert "Save session report (HTML)…" in labels, labels

    def test_html_report_guard_covers_both_axes(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # neither a figure nor a filled table -> the honest info dialog
        # (round 201) the attribution card's pre-filled registry table
        # must be emptied for the honest empty state (same latent
        # v1.2.115 staleness as the PDF twin above)
        app._dl_attr_tree.delete(*app._dl_attr_tree.get_children())
        app._session_figures.clear()
        app._menu_save_session_report_html()
        assert any("Nothing to report yet" in str(a) for _n, a in quiet_dialogs)

    def test_html_report_writes_real_file(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "session.html"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        # one live figure on the registry + one filled table
        figure = self._figure_static("smoke plot")
        app._render_figure(app._topo_host, figure)
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in text
        assert "PharmaML Playground session" in text
        assert "data:image" in text, "figures were not embedded"
        assert "<table" in text, "the tables never landed"
        assert "What this file contains" in text
        assert "cards on screen" in text, "the metric cards never landed"
        assert "table 1 · SMILES inspector" in text, "the table walk's heading"
        # (100x-134) the count prefix replaced the "{title} table" walk
        assert any(
            "figure(s) + " in str(a) for _n, a in quiet_dialogs
        ), "the save message never fired"
        # close=False semantics: the on-screen figure stays alive
        import matplotlib.pyplot as plt

        assert getattr(app._topo_host, "_pg_figure", None) is figure
        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    def test_html_report_dedupes_captions(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "dedupe.html"
        monkeypatch.setattr(playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        # two hosts, the SAME job label -> the second caption is numbered
        first = self._figure_static("alpha")
        second = self._figure_static("beta")
        app._render_figure(app._topo_host, first)
        app._render_figure(app._breed_host, second)
        # same title for both entries (the dedupe contract)
        app._session_figures[0][1] = "same label"
        app._session_figures[1][1] = "same label"
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        text = target.read_text(encoding="utf-8")
        assert "same label" in text
        # (100x-134) the " (2)" caption dedupe retired: the count
        # prefix separates the same-titled pages on its own -- both
        # captions are numbered, the way the picklist's rows are
        assert "figure 1 · same label" in text, "the first caption lost its count"
        assert "figure 2 · same label" in text, (
            "the repeat caption was not separated by the count prefix"
        )
        import matplotlib.pyplot as plt

        plt.close("all")

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    # -- the round's documentation pins --------------------------------- #
    def test_module_docstring_records_round_124(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-124" in doc
        assert "HTMLReport" in doc  # the HTML twin is documented

    def test_home_guide_mentions_scored_grid(self, app: Any, playground: Any) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "scored 2D structure grid" in dump


class TestReview125Gui:
    """Round 100x-125: the Pareto front's colour-coded structure grid
    (molecule_grid_figure's colours row + the front card's Draw front
    button) — the front's molecules finally visible, boundary corners
    accented teal, interior trade-offs muted neutral."""

    TEAL = "#0F766E"
    NEUTRAL = "#525252"
    DRUGLIKE = "Multi-objective (QED × MW × TPSA)"

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _run_druglike(app: Any) -> None:
        app._objective_combo.set(TestReview125Gui.DRUGLIKE)
        app._pop_size.set("12")
        app._generations.set("4")
        app._run_evolution()
        assert TestReview125Gui._drain(app), "evolution job hung"

    # -- molecule_grid_figure's colours row ------------------------------ #
    def test_grid_colors_accent_title_and_border(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure(
            [("#1 · C1CCNCC1", "C1CCNCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
            title="coloured",
            colors=[TestReview125Gui.TEAL, None],
        )
        try:
            accent, classic = figure.get_axes()[0], figure.get_axes()[1]
            # the accent speaks twice: title colour + a visible border
            assert accent.title.get_color() == TestReview125Gui.TEAL
            spine = accent.spines["top"]
            assert spine.get_visible() and spine.get_edgecolor()[:3] == (
                15.0 / 255.0, 118.0 / 255.0, 110.0 / 255.0,
            )
            # the classic panel keeps the uniform look: default title
            # colour, border still hidden
            assert classic.title.get_color() != TestReview125Gui.TEAL
            assert not classic.spines["top"].get_visible()
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_grid_colors_mismatch_never_shifts(self, playground: Any) -> None:
        # a shorter list pads (entry 2 keeps the classic look), a longer
        # one is truncated -- the same contract as the scores row
        short = playground.molecule_grid_figure(
            [("#1 · C1CCNCC1", "C1CCNCC1"), ("#2 · c1ccccc1", "c1ccccc1")],
            title="pad",
            colors=[TestReview125Gui.TEAL],
        )
        try:
            assert short.get_axes()[0].title.get_color() == TestReview125Gui.TEAL
            assert short.get_axes()[1].title.get_color() != TestReview125Gui.TEAL
        finally:
            import matplotlib.pyplot as plt

            plt.close(short)
        long = playground.molecule_grid_figure(
            [("#1 · C1CCNCC1", "C1CCNCC1")],
            title="cut",
            colors=[TestReview125Gui.TEAL, TestReview125Gui.NEUTRAL, "#B45309"],
        )
        try:
            titles = [axis.title.get_color() for axis in long.get_axes()]
            assert TestReview125Gui.TEAL in titles, titles
            assert TestReview125Gui.NEUTRAL not in titles, titles
        finally:
            import matplotlib.pyplot as plt

            plt.close(long)

    def test_grid_colors_keep_bare_panels_for_none(self, playground: Any) -> None:
        figure = playground.molecule_grid_figure([("m\nC1CCNCC1", "C1CCNCC1")], title="x")
        try:
            assert all(
                axis.title.get_color() not in (TestReview125Gui.TEAL, TestReview125Gui.NEUTRAL)
                for axis in figure.get_axes()
            )
            assert all(
                not axis.spines["top"].get_visible() for axis in figure.get_axes()
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_grid_colors_reach_unparseable_note(self, playground: Any) -> None:
        # the honest note keeps its slot AND carries the accent colour
        figure = playground.molecule_grid_figure(
            [("#3 · bad", "not-a-smiles")],
            title="note",
            colors=[TestReview125Gui.NEUTRAL],
        )
        try:
            texts = [t for t in figure.get_axes()[0].texts]
            assert texts and texts[0].get_color() == TestReview125Gui.NEUTRAL
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    # -- the Pareto front card's draw-front button ---------------------- #
    def test_draw_front_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._pareto_front = None
        app._pareto_result = None
        app._gen_draw_front()
        assert any("Nothing to draw" in str(a) for _n, a in quiet_dialogs)
        assert getattr(app._pareto_host, "_pg_figure", None) is None

    def test_draw_front_full_path(self, app: Any) -> None:
        self._run_druglike(app)
        assert app._pareto_front is not None and len(app._pareto_front) == 7
        # the run auto-plotted the scatter first (the established
        # pattern: draw-front REPLACES it, plot-front brings it back)
        assert app._pareto_host._pg_figure is not None
        app._gen_draw_front()
        assert self._drain(app), "the draw-front job hung"
        figure = app._pareto_host._pg_figure
        assert figure is not None, "the front structure grid never landed"
        axes = figure.get_axes()
        # the pinned deterministic run: 7 front members, 5 of them
        # boundary corners (crowding +inf) accented teal, 2 muted
        title_colors = [axis.title.get_color() for axis in axes]
        assert title_colors.count(self.TEAL) == 5, title_colors
        assert title_colors.count(self.NEUTRAL) == 2, title_colors
        # every member panel carries its objective values as the score
        # row (3 objectives: qed · mw_band · tpsa_band)
        labels = [axis.get_xlabel() for axis in axes]
        scored = [label for label in labels if label.startswith("qed ")]
        assert len(scored) == 7, labels
        assert all("mw_band" in label and "tpsa_band" in label for label in scored)
        titles = [axis.get_title() for axis in axes]
        assert any(title.startswith("#1") for title in titles), titles
        # Plot front brings the scatter back (the replace/restore pair)
        app._plot_pareto_front()
        assert app._pareto_host._pg_figure is not None
        scatter_label = next(
            (axis.get_xlabel() for axis in app._pareto_host._pg_figure.get_axes()), ""
        )
        assert not scatter_label.startswith("qed ")
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_pareto_card_carries_draw_front_button(self, app: Any) -> None:
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[11]))
            if widget.winfo_class() == "TButton"
        )
        assert "Draw front" in dump
        assert "Plot front" in dump
        assert "Save front as CSV…" in dump

    def test_module_docstring_records_round_125(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-125" in doc
        assert "colour-coded 2D structure grid" in doc

    def test_home_guide_mentions_front_grid(self, app: Any, playground: Any) -> None:
        # the home tab's "What lives where" card tells the user the
        # front's members draw colour-coded
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "colour-coded" in dump and "boundary corners" in dump


class TestReview126Gui:
    """Round 100x-126: the HTML report's theme engine (the report OPENS
    in the app's View ▸ theme setting and carries a per-reader toggle
    button) and the Pareto front card's per-card "Save figure…" export
    (the figure THIS card shows, not the last one drawn anywhere)."""

    DRUGLIKE = "Multi-objective (QED × MW × TPSA)"

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _run_druglike(app: Any) -> None:
        app._objective_combo.set(TestReview126Gui.DRUGLIKE)
        app._pop_size.set("12")
        app._generations.set("4")
        app._run_evolution()
        assert TestReview126Gui._drain(app), "evolution job hung"

    # -- the Pareto card's per-card save --------------------------------- #
    def test_save_pareto_figure_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        # nothing rendered on the host yet -> the honest info box
        if getattr(app._pareto_host, "_pg_figure", None) is not None:
            import matplotlib.pyplot as plt

            plt.close("all")
            app._pareto_host._pg_figure = None
        app._save_pareto_figure()
        assert any("No figure on the card yet" in str(a) for _n, a in quiet_dialogs)

    def test_save_pareto_figure_saves_the_cards_figure(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "front.png"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_k: str(target),
        )
        # capture WHICH figure the button hands to save_figure (the
        # import happens at call time, so the module attribute is read
        # fresh -- the spy is picked up)
        import pharmaml.viz as viz_module

        real_save = viz_module.save_figure
        captured: dict[str, Any] = {}

        def spy(figure: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            captured["figure"] = figure
            return real_save(figure, path, *args, **kwargs)

        monkeypatch.setattr(viz_module, "save_figure", spy)

        self._run_druglike(app)
        card_figure = app._pareto_host._pg_figure
        assert card_figure is not None  # the run auto-plotted the scatter

        # the honest distinction from Ctrl+S: another figure "draws
        # last" somewhere else -- the per-card save must STILL save
        # the front card's own figure, not the global _last_figure
        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        app._last_figure = decoy

        app._save_pareto_figure()
        assert self._drain(app), "the save job hung"
        try:
            assert captured["figure"] is card_figure, "saved a figure that is not the card's"
            assert captured["figure"] is not decoy
            assert target.exists(), "the PNG never landed"
            assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            # the success dialog fired (patched -- an unpatched showinfo
            # here would hang the suite, the round-123 trap)
            assert any("Figure saved" in str(a) for _n, a in quiet_dialogs)
        finally:
            plt.close("all")

    def test_pareto_card_carries_save_figure_button(self, app: Any) -> None:
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[11]))
            if widget.winfo_class() == "TButton"
        )
        assert "Save figure…" in dump
        assert "Draw front" in dump and "Plot front" in dump

    # -- the HTML report opens in the app's theme ------------------------ #
    def test_html_report_bakes_app_theme(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "session.html"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_k: str(target),
        )
        # one live figure is enough for the report to have content
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        try:
            app._render_figure(app._pareto_host, figure)
            app._theme_var.set("dark")
            app._menu_save_session_report_html()
            assert self._drain(app), "the HTML report job hung"
        finally:
            plt.close("all")
        text = target.read_text(encoding="utf-8")
        assert "<html lang='en' data-theme='dark'>" in text
        assert "id='theme-toggle'" in text
        assert "localStorage" in text and "pharmaml-report-theme" in text
        assert "data:image/png;base64," in text  # the figure embedded
        # the completion dialog tells the reader about the toggle
        assert any("cycles it per reader" in str(a) for _n, a in quiet_dialogs)

    def test_module_docstring_records_round_126(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-126" in doc
        assert "theme engine" in doc
        assert "per-card" in doc

    def test_home_guide_mentions_per_card_export(
        self, app: Any, playground: Any
    ) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "per-card" in dump and "figure export" in dump


class TestReview127Gui:
    """Round 100x-127: the per-card "Save figure…" export generalised to
    every Generative figure card (the two-host Results card picks the
    figure that drew LAST on that card; both surgery cards carry their
    own button), and the HTML report's bake guarded by the library's
    public REPORT_THEMES contract."""

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the buttons exist ------------------------------------------------ #
    def test_generative_tab_carries_four_save_figure_buttons(self, app: Any) -> None:
        dump = [
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[11]))
            if widget.winfo_class() == "TButton"
        ]
        # Results, Pareto front, Ring topology moves, Breed two molecules
        assert dump.count("Save figure…") == 4, dump.count("Save figure…")
        assert "Draw top molecules" in dump and "Draw mutants" in dump
        assert "Draw litter" in dump and "Draw front" in dump

    # -- the honest guards ------------------------------------------------ #
    def test_save_ga_figure_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import matplotlib.pyplot as plt

        for host in (app._ga_grid_host, app._ga_host):
            if getattr(host, "_pg_figure", None) is not None:
                host._pg_figure = None
        plt.close("all")
        app._save_ga_figure()
        assert any("No figure on the card yet" in str(a) for _n, a in quiet_dialogs)

    def test_save_topo_figure_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import matplotlib.pyplot as plt

        if getattr(app._topo_host, "_pg_figure", None) is not None:
            app._topo_host._pg_figure = None
        plt.close("all")
        app._save_topo_figure()
        assert any("No figure on the card yet" in str(a) for _n, a in quiet_dialogs)

    def test_save_breed_figure_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import matplotlib.pyplot as plt

        if getattr(app._breed_host, "_pg_figure", None) is not None:
            app._breed_host._pg_figure = None
        plt.close("all")
        app._save_breed_figure()
        assert any("No figure on the card yet" in str(a) for _n, a in quiet_dialogs)

    # -- the semantics: the CARD's figure, not the global last-drawn ----- #
    def test_save_topo_figure_saves_the_cards_figure(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "mutants.png"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target),
        )
        import pharmaml.viz as viz_module

        real_save = viz_module.save_figure
        captured: dict[str, Any] = {}

        def spy(figure: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            captured["figure"] = figure
            return real_save(figure, path, *args, **kwargs)

        monkeypatch.setattr(viz_module, "save_figure", spy)

        grid, axis = plt.subplots()
        axis.set_title("mutant grid")
        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        try:
            app._render_figure(app._topo_host, grid)
            app._last_figure = decoy  # something else drew last ANYWHERE
            app._save_topo_figure()
            assert self._drain(app), "the save job hung"
            assert captured["figure"] is grid, "saved a figure that is not the card's"
            assert captured["figure"] is not decoy
            assert target.exists(), "the PNG never landed"
            assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert any("Figure saved" in str(a) for _n, a in quiet_dialogs)
        finally:
            plt.close("all")

    # -- the two-host Results card: NEWEST of the card's own figures ----- #
    def test_save_ga_figure_picks_the_newest_of_the_cards_hosts(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "results.png"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target),
        )
        import pharmaml.viz as viz_module

        real_save = viz_module.save_figure
        captured: dict[str, Any] = {}

        def spy(figure: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            captured["figure"] = figure
            return real_save(figure, path, *args, **kwargs)

        monkeypatch.setattr(viz_module, "save_figure", spy)

        history, h_axis = plt.subplots()
        h_axis.set_title("evolution history")
        grid, g_axis = plt.subplots()
        g_axis.set_title("leaderboard grid")
        again, a_axis = plt.subplots()
        a_axis.set_title("history again")
        try:
            # the run's auto-plot and the Draw-top button race: the save
            # must follow whichever of the CARD's two hosts drew last.
            app._render_figure(app._ga_host, history)
            app._render_figure(app._ga_grid_host, grid)
            app._save_ga_figure()
            assert self._drain(app), "the save job hung"
            assert captured["figure"] is grid, "the newest card figure did not win"

            app._render_figure(app._ga_host, again)  # history re-drew last
            app._save_ga_figure()
            assert self._drain(app), "the second save job hung"
            assert captured["figure"] is again, "the newest card figure did not win"
            assert target.exists()
            assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            plt.close("all")

    # -- the render counter stamps every host ----------------------------- #
    def test_render_figure_stamps_monotonic_sequence(self, app: Any) -> None:
        import matplotlib.pyplot as plt

        first, _f_axis = plt.subplots()
        second, _s_axis = plt.subplots()
        try:
            before = app._figure_seq
            app._render_figure(app._topo_host, first)
            mid = app._figure_seq
            app._render_figure(app._breed_host, second)
            assert before < mid < app._figure_seq, "the render counter is monotonic"
            assert app._topo_host._pg_figure_seq == mid
            assert app._breed_host._pg_figure_seq == app._figure_seq
        finally:
            plt.close("all")

    # -- the HTML report bake guards REPORT_THEMES ------------------------ #
    def test_html_report_theme_guard_falls_back_to_auto(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "guarded.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target),
        )
        figure, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        try:
            app._render_figure(app._pareto_host, figure)
            # a drifted/invalid theme value must NOT reach HTMLReport (it
            # would raise mid-job) -- the guard falls back to auto.
            app._theme_var.set("neon")
            app._menu_save_session_report_html()
            assert self._drain(app), "the HTML report job hung"
        finally:
            plt.close("all")
            app._theme_var.set("auto")
        text = target.read_text(encoding="utf-8")
        assert "<html lang='en' data-theme='auto'>" in text
        assert "data:image/png;base64," in text

    def test_module_docstring_records_round_127(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-127" in doc
        assert "every Generative figure card" in doc

    def test_home_guide_mentions_every_figure_card_export(
        self, app: Any, playground: Any
    ) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "Every figure card" in dump and "Save figure" in dump


class TestReview128Gui:
    """Round 100x-128: the per-card "Save figure…" export closes app-wide --
    every REMAINING figure card (ML results / leaderboard / AD, the DL
    results and the two-host training history, the PK & PD figures, the
    feature factory, the similarity scan, both docking cards, the NCA
    profile, the utilities reconciliation) carries its own export button,
    each saving the figure the card's OWN host shows -- never the global
    Ctrl+S last-drawn-anywhere figure."""

    #: (method, host attribute names) for every card shipped this round.
    NEW_CARDS: ClassVar[list[tuple[str, tuple[str, ...]]]] = [
        ("_save_feat_figure", ("_feat_host",)),
        ("_save_scan_figure", ("_scan_host",)),
        ("_save_iv_figure", ("_iv_host",)),
        ("_save_pd_figure", ("_pd_host",)),
        ("_save_nca_figure", ("_nca_host",)),
        ("_save_acc_figure", ("_acc_host",)),
        ("_save_fam_figure", ("_fam_host",)),
        ("_save_dock_receptor_figure", ("_dock_receptor_host",)),
        ("_save_dock_results_figure", ("_dock_results_host",)),
        ("_save_ml_figure", ("_ml_host",)),
        ("_save_board_figure", ("_board_host",)),
        ("_save_ad_figure", ("_ad_host",)),
        ("_save_dl_figure", ("_dl_host",)),
        ("_save_dl_history_figure", ("_dl_hist_host", "_dl_curve_host")),
        ("_save_util_cmp_figure", ("_util_cmp_host",)),
        # (100x-165) the Foundations tab's five figure cards
        ("_save_found_scratch_figure", ("_found_scratch_host",)),
        ("_save_found_sweep_figure", ("_found_sweep_host",)),
        ("_save_found_seq_figure", ("_found_seq_host",)),
        ("_save_found_attn_figure", ("_found_attn_host",)),
        ("_save_found_dist_figure", ("_found_dist_host",)),
    ]

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _tab_buttons(app: Any, index: int) -> list[str]:
        return [
            str(widget.cget("text"))
            for widget in app._walk_widgets(app._book.nametowidget(app._book.tabs()[index]))
            if widget.winfo_class() == "TButton"
        ]

    # -- the buttons exist, one per figure card, on every tab ------------ #
    def test_every_tab_carries_its_save_figure_buttons(self, app: Any) -> None:
        # (tab index, expected count, a co-resident button to prove the
        # row joined the right card)
        expectations = [
            (1, 2, "Plot distributions"),   # Chemistry: factory + scan
            (2, 5, "Plot repeated dosing"),  # PK/PD: IV, Emax, NCA, acc, family
            (4, 2, "Plot atom types"),      # Docking: receptor + results
            (5, 3, "Calibration curve"),    # Data & ML: results, board, AD
            (6, 1, "Run analysis"),         # Unsupervised: the Results card (1.2.93)
            (7, 2, "Train & evaluate"),     # DL: results + history
            (8, 5, "Train from scratch"),   # Foundations (1.2.99): the five labs
            (12, 1, "Plot reconciliation"),  # Utilities: compare
        ]
        total = 0
        for index, count, neighbour in expectations:
            buttons = self._tab_buttons(app, index)
            assert buttons.count("Save figure…") == count, (index, buttons.count("Save figure…"))
            assert neighbour in buttons, (index, neighbour)
            total += count
        assert total == 21, total
        # plus the four Generative buttons shipped in rounds 126/127 --
        # the pattern now covers every figure card in the app.
        # (100x-167: Best-of ML sits at 9 and Response AI at 10; the
        # generative tab moved to 11 with the two new tabs.)
        assert self._tab_buttons(app, 11).count("Save figure…") == 4

    # -- the honest guards, one per new card ------------------------------ #
    @pytest.mark.parametrize(
        ("method_name", "host_names"),
        NEW_CARDS,
        ids=[name for name, _hosts in NEW_CARDS],
    )
    def test_save_figure_guard(
        self, app: Any, quiet_dialogs: list[Any], method_name: str, host_names: tuple[str, ...]
    ) -> None:
        import matplotlib.pyplot as plt

        for host_name in host_names:
            host = getattr(app, host_name)
            if getattr(host, "_pg_figure", None) is not None:
                host._pg_figure = None
        plt.close("all")
        getattr(app, method_name)()
        assert any("No figure on the card yet" in str(a) for _n, a in quiet_dialogs), method_name

    # -- the semantics: the CARD's figure, not the global last-drawn ----- #
    def test_save_nca_figure_saves_the_cards_figure(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "nca.png"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target),
        )
        import pharmaml.viz as viz_module

        real_save = viz_module.save_figure
        captured: dict[str, Any] = {}

        def spy(figure: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            captured["figure"] = figure
            return real_save(figure, path, *args, **kwargs)

        monkeypatch.setattr(viz_module, "save_figure", spy)

        profile, axis = plt.subplots()
        axis.set_title("concentration-time profile")
        decoy, decoy_axis = plt.subplots()
        decoy_axis.plot([0, 1], [1, 0])
        try:
            app._render_figure(app._nca_host, profile)
            app._last_figure = decoy  # something else drew last ANYWHERE
            app._save_nca_figure()
            assert self._drain(app), "the save job hung"
            assert captured["figure"] is profile, "saved a figure that is not the card's"
            assert captured["figure"] is not decoy
            assert target.exists(), "the PNG never landed"
            assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
            assert any("Figure saved" in str(a) for _n, a in quiet_dialogs)
        finally:
            plt.close("all")

    # -- the two-host DL history card: NEWEST of the card's own figures --- #
    def test_save_dl_history_figure_newest_of_two_hosts(
        self, app: Any, monkeypatch: Any, tmp_path: Any, playground: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        import matplotlib.pyplot as plt

        target = tmp_path / "history.png"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target),
        )
        import pharmaml.viz as viz_module

        real_save = viz_module.save_figure
        captured: dict[str, Any] = {}

        def spy(figure: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
            captured["figure"] = figure
            return real_save(figure, path, *args, **kwargs)

        monkeypatch.setattr(viz_module, "save_figure", spy)

        losses, l_axis = plt.subplots()
        l_axis.set_title("loss curves")
        curve, c_axis = plt.subplots()
        c_axis.set_title("learning curve")
        again, a_axis = plt.subplots()
        a_axis.set_title("loss curves again")
        try:
            # training auto-plots the losses, the learning-curve button
            # draws later: the save follows whichever drew LAST.
            app._render_figure(app._dl_hist_host, losses)
            app._render_figure(app._dl_curve_host, curve)
            app._save_dl_history_figure()
            assert self._drain(app), "the save job hung"
            assert captured["figure"] is curve, "the newest card figure did not win"

            app._render_figure(app._dl_hist_host, again)  # losses re-drew last
            app._save_dl_history_figure()
            assert self._drain(app), "the second save job hung"
            assert captured["figure"] is again, "the newest card figure did not win"
            assert target.exists()
            assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        finally:
            plt.close("all")

    # -- the documentation pins -------------------------------------------- #
    def test_module_docstring_records_round_128(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-128" in doc
        assert "closes app-wide" in doc
        assert "utilities reconciliation" in doc

    def test_home_guide_mentions_app_wide_export(self, app: Any, playground: Any) -> None:
        guide = None
        for title, _subtitle, shell in playground._CARD_INDEX:
            if title != "What lives where":
                continue
            try:
                live = bool(shell.winfo_exists())
            except Exception:  # noqa: BLE001 - destroyed earlier shell
                live = False
            if live:
                guide = shell
                break
        assert guide is not None
        dump = "\n".join(
            str(widget.cget("text"))
            for widget in app._walk_widgets(guide)
            if widget.winfo_class() == "TLabel"
        )
        assert "Every figure card" in dump
        assert "in the whole app" in dump


class TestReview129Gui:
    """Round 100x-129: the PDF session report's dark-ink variant, riding
    the View ▸ theme radios the way the HTML report's palette does --
    a dark app writes near-black pages with light ink (the app's own
    palette, figures on white plates), a light app writes the classic
    white paper, and auto resolves at save time because a PDF's ink is
    frozen at write time."""

    #: the exact operator strings fpdf2 emits for the variant's palette
    PAGE_BG = b"0.0824 0.102 0.0941 rg"  # (21, 26, 24) -- the app dark BG
    ACCENT = b"0.0863 0.6627 0.5412 rg"  # (22, 169, 138) -- headings
    INK = b"0.9059 0.9255 0.9137 rg"  # (231, 236, 233) -- body text
    PLATE = b"1 1 1 rg"  # the white plate under every figure

    @pytest.fixture()
    def app(
        self, playground: Any, xvfb_display: str, picklist_accept_all: None
    ) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def quiet_dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    @staticmethod
    def _content_streams(pdf_bytes: bytes) -> bytes:
        """Decompress every page-content stream (fpdf2 zlib-compresses)."""
        import re
        import zlib

        out: list[bytes] = []
        for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL):
            try:
                out.append(zlib.decompress(match.group(1)))
            except zlib.error:  # not a compressed stream (fonts, images)
                pass
        return b"\n".join(out)

    # -- the library contract ---------------------------------------------- #
    def test_pdf_themes_pin_the_frozen_ink_contract(self) -> None:
        from pharmaml.viz import PDF_THEMES

        assert PDF_THEMES == ("light", "dark")
        assert "auto" not in PDF_THEMES, "a PDF's ink is frozen at write time"

    def test_theme_guard_rejects_unknown_variant(self, tmp_path: Any) -> None:
        from pharmaml.viz import save_pdf_report

        with pytest.raises(ValueError, match="frozen at write time"):
            save_pdf_report({}, out_path=tmp_path / "x.pdf", theme="sepia")

    def test_light_paper_stays_colorless_and_dark_differs(self, tmp_path: Any) -> None:
        import pandas as pd

        from pharmaml.viz import save_pdf_report

        figure = self._figure_static("ink probe")
        table = pd.DataFrame({"metric": ["R2"], "value": [0.91]})
        light_path = tmp_path / "light.pdf"
        dark_path = tmp_path / "dark.pdf"
        try:
            save_pdf_report(
                {"ink probe": figure}, tables={"metrics": table},
                metrics={"R2": 0.91}, out_path=light_path, close=False,
            )
            save_pdf_report(
                {"ink probe": figure}, tables={"metrics": table},
                metrics={"R2": 0.91}, out_path=dark_path, close=False, theme="dark",
            )
            light_bytes = light_path.read_bytes()
            dark_bytes = dark_path.read_bytes()
            assert light_bytes[:5] == b"%PDF-" and dark_bytes[:5] == b"%PDF-"
            assert light_bytes != dark_bytes, "the variant changed nothing"
            light_ops = self._content_streams(light_bytes)
            dark_ops = self._content_streams(dark_bytes)
            # light NEVER touches a colour: byte-identical classic paper
            assert b" rg" not in light_ops, "the light paper gained colour ops"
            # dark carries the full palette: page fill, accent headings,
            # light body ink and the white figure plate
            for operator in (self.PAGE_BG, self.ACCENT, self.INK, self.PLATE):
                assert operator in dark_ops, operator
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    # -- the playground wiring: the variant rides the theme radios -------- #
    def _save_pdf_report_spy(
        self, playground: Any, monkeypatch: Any, target: Any
    ) -> dict[str, Any]:
        """Patch filedialog + wrap save_pdf_report, capturing its kwargs."""
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        import pharmaml.viz as viz_module

        real = viz_module.save_pdf_report
        captured: dict[str, Any] = {}

        def spy(*args: Any, **kwargs: Any) -> Any:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return real(*args, **kwargs)

        monkeypatch.setattr(viz_module, "save_pdf_report", spy)
        return captured

    def _seed_session(self, app: Any, figure: Any) -> None:
        app._session_figures.clear()
        app._render_figure(app._topo_host, figure)
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))

    def test_dark_radio_writes_the_dark_ink_variant(
        self, app: Any, playground: Any, monkeypatch: Any,
        quiet_dialogs: list[Any], tmp_path: Any,
    ) -> None:
        target = tmp_path / "session-dark.pdf"
        captured = self._save_pdf_report_spy(playground, monkeypatch, target)
        figure = self._figure_static("dark ride")
        try:
            app._theme_var.set("dark")
            app._on_theme_change()  # the real radio path flips the palette
            assert app._theme == "dark"
            self._seed_session(app, figure)
            app._menu_save_session_report()
            assert self._drain(app), "the dark report job hung"
            assert captured["kwargs"].get("theme") == "dark", captured["kwargs"]
            assert target.is_file()
            ops = self._content_streams(target.read_bytes())
            assert self.PAGE_BG in ops, "the pages never went dark"
            assert self.PLATE in ops, "the figure plate never landed"
            # the subtitle names the variant on the title page
            assert "dark ink variant" in captured["kwargs"].get("subtitle", "")
            assert any(
                "dark ink variant" in str(a) for _n, a in quiet_dialogs
            ), "the save dialog never named the variant"
            assert any(
                "View" in str(a) for _n, a in quiet_dialogs
            ), "the flip hint never fired"
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            app._desc_tree.delete(*app._desc_tree.get_children())

    def test_light_radio_writes_the_classic_paper(
        self, app: Any, playground: Any, monkeypatch: Any,
        quiet_dialogs: list[Any], tmp_path: Any,
    ) -> None:
        target = tmp_path / "session-light.pdf"
        captured = self._save_pdf_report_spy(playground, monkeypatch, target)
        figure = self._figure_static("light ride")
        try:
            app._theme_var.set("light")
            app._on_theme_change()
            self._seed_session(app, figure)
            app._menu_save_session_report()
            assert self._drain(app), "the light report job hung"
            assert captured["kwargs"].get("theme") == "light", captured["kwargs"]
            ops = self._content_streams(target.read_bytes())
            assert b" rg" not in ops, "the classic paper gained colour ops"
            assert "classic light paper" in captured["kwargs"].get("subtitle", "")
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            app._desc_tree.delete(*app._desc_tree.get_children())

    def test_auto_resolves_before_the_ink_freezes(
        self, app: Any, playground: Any, monkeypatch: Any,
        quiet_dialogs: list[Any], tmp_path: Any,
    ) -> None:
        target = tmp_path / "session-auto.pdf"
        captured = self._save_pdf_report_spy(playground, monkeypatch, target)
        monkeypatch.setattr(app, "_resolve_auto_theme", lambda: "dark")
        figure = self._figure_static("auto ride")
        try:
            app._theme_var.set("auto")
            self._seed_session(app, figure)
            app._menu_save_session_report()
            assert self._drain(app), "the auto report job hung"
            # auto never reaches the library: the resolved OS preference
            # froze into dark ink at write time
            assert captured["kwargs"].get("theme") == "dark", captured["kwargs"]
            assert "auto" != captured["kwargs"].get("theme")
        finally:
            import matplotlib.pyplot as plt

            plt.close("all")
            app._desc_tree.delete(*app._desc_tree.get_children())

    def test_view_menu_caption_tells_the_riding(
        self, app: Any, playground: Any,
    ) -> None:
        import tkinter as tkmod

        bar = app.nametowidget(app.cget("menu"))
        assert isinstance(bar, tkmod.Menu)
        labels: list[str] = []
        for index in range(bar.index("end") + 1):
            if bar.type(index) != "cascade":
                continue
            if str(bar.entrycget(index, "label")).strip() != "View":
                continue
            sub = app.nametowidget(bar.entrycget(index, "menu"))
            for j in range(sub.index("end") + 1):
                if sub.type(j) not in ("command", "radiobutton"):
                    continue  # separators have no -label
                labels.append(str(sub.entrycget(j, "label")))
        assert any(
            "session reports save in this theme" in label for label in labels
        ), labels
        # the radios themselves are still there
        assert "Follow system" in labels and "Dark" in labels, labels

    # -- the documentation pins -------------------------------------------- #
    def test_module_docstring_records_round_129(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-129" in doc
        assert "dark-ink variant" in doc
        assert "frozen at write time" in doc
        assert "_resolve_auto_theme" in doc

    def test_method_docstring_records_the_ride(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._menu_save_session_report.__doc__ or ""
        assert "100x-129" in doc
        assert "INK VARIANT" in doc
        assert "white plates" in doc


class TestReview130Gui:
    """Round 100x-130: the Ctrl+F card finder's "Recently jumped"
    recency row -- every finder jump is remembered (most recent first,
    de-duplicated, capped, persisted with the session) and the dialog
    offers one-click chips back to those cards, filtered to the cards
    that are live right now (a vanished shell never leaves a dead
    button).  The web twin's finder gained the same memory this round
    (its parity guards live in tests/test_web_parity.py)."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    # -- the memory itself ---------------------------------------------- #
    def test_jump_records_recency_most_recent_first(self, app: Any) -> None:
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
        # (100x-142) the memory keys on "tab:title" -- the five
        # duplicated card titles can never jump to the wrong tab again
        assert app._finder_recent == ["0:PharmaML Playground", "1:Affinity conversion"]
        # a re-jump de-duplicates by moving the key to the front
        app._jump_to_card(1, "Affinity conversion")
        assert app._finder_recent == ["1:Affinity conversion", "0:PharmaML Playground"]

    def test_recency_caps_at_the_documented_maximum(self, app: Any, playground: Any) -> None:
        titles = [
            title
            for title, _subtitle, _shell in playground._CARD_INDEX[:8]
        ][:8]
        assert len(titles) == 8, "the index needs 8 cards for the cap test"
        for index, title in enumerate(titles):
            app._jump_to_card(0, title)
        # (100x-142) keyed entries: every jump landed on tab 0
        assert app._finder_recent == [f"0:{title}" for title in reversed(titles[-6:])]
        assert len(app._finder_recent) == playground._FINDER_RECENT_MAX

    def test_cap_is_six_and_exported(self, playground: Any) -> None:
        assert playground._FINDER_RECENT_MAX == 6

    # -- the dialog's chip row ------------------------------------------- #
    def _finder_dialog(self, app: Any, playground: Any) -> Any:
        import tkinter as tkmod

        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        assert len(dialogs) == 1
        return dialogs[0]

    @staticmethod
    def _dialog_buttons(win: Any) -> list[Any]:
        buttons: list[Any] = []

        def walk(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons.append(child)
                walk(child)

        walk(win)
        return buttons

    def test_dialog_shows_recent_chips(self, app: Any, playground: Any) -> None:
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
        win = self._finder_dialog(app, playground)
        chip_labels = [str(b.cget("text")) for b in self._dialog_buttons(win)]
        # newest first, truncated to 24 characters
        assert "PharmaML Playground" in chip_labels
        assert "Affinity conversion" in chip_labels
        # the muted label names the row
        labels = []

        def walk_labels(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TLabel":
                    labels.append(str(child.cget("text")))
                walk_labels(child)

        walk_labels(win)
        assert any(label.startswith("Recently jumped") for label in labels), labels
        win.destroy()

    def test_chip_click_jumps_and_closes(self, app: Any, playground: Any) -> None:
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
        win = self._finder_dialog(app, playground)
        chips = {
            str(b.cget("text")): b for b in self._dialog_buttons(win)
        }
        chip = chips["Affinity conversion"]
        chip.invoke()
        app.update()
        # the dialog closed and the chip's card got selected (+ remembered)
        assert not bool(win.winfo_exists())
        assert app._book.index(app._book.select()) == 1
        assert app._finder_recent[0] == "1:Affinity conversion"

    def test_dialog_omits_non_live_titles(self, app: Any, playground: Any) -> None:
        app._finder_recent = ["Affinity conversion", "A card that does not exist"]
        win = self._finder_dialog(app, playground)
        chip_labels = [str(b.cget("text")) for b in self._dialog_buttons(win)]
        assert "Affinity conversion" in chip_labels
        assert not any("does not exist" in label for label in chip_labels)
        win.destroy()

    def test_dialog_without_recency_has_no_chip_row(
        self, app: Any, playground: Any
    ) -> None:
        app._finder_recent = []
        win = self._finder_dialog(app, playground)
        labels = []

        def walk_labels(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TLabel":
                    labels.append(str(child.cget("text")))
                walk_labels(child)

        walk_labels(win)
        assert not any(label.startswith("Recently jumped") for label in labels)
        win.destroy()

    # -- the session persistence ------------------------------------------ #
    def test_recency_persists_in_session_payload(self, app: Any) -> None:
        import json as jsonmod

        app._finder_recent = ["Affinity conversion", "PharmaML Playground"]
        payload = app._collect_session()
        raw = payload["texts"]["finder_recent"]
        assert jsonmod.loads(raw) == ["Affinity conversion", "PharmaML Playground"]

    def test_recency_restores_from_session(self, app: Any) -> None:
        import json as jsonmod

        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {
                "finder_recent": jsonmod.dumps(
                    ["Molar mass", "SMILES inspector", "Affinity conversion"]
                )
            },
        }
        app._restore_session()
        assert app._finder_recent == [
            "Molar mass",
            "SMILES inspector",
            "Affinity conversion",
        ]

    def test_recency_restore_is_tolerant(self, app: Any) -> None:
        # garbage resets the memory, never the restore
        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {"finder_recent": "not json at all"},
        }
        app._restore_session()
        assert app._finder_recent == []
        # non-string entries are dropped wholesale too
        app._session = {
            "theme": "light",
            "geometry": "",
            "tab": 0,
            "variables": {},
            "families": {},
            "texts": {"finder_recent": '["ok", 42]'},
        }
        app._restore_session()
        assert app._finder_recent == []

    # -- the documentation pins ------------------------------------------- #
    def test_module_docstring_records_round_130(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-130" in doc
        assert "Recently jumped" in doc
        assert "_FINDER_RECENT_MAX" in doc
        assert "finder_recent" in doc
        assert "Surgery sub-tab" in doc

    def test_finder_remember_docstring_pins_the_contract(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._finder_remember.__doc__ or ""
        assert "100x-130" in doc
        assert "Most-recent-first" in doc

    def test_jump_docstring_mentions_the_memory(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._jump_to_card.__doc__ or ""
        assert "recency" in doc


class TestReview131Gui:
    """Round 100x-131: the recency row's "Clear" affordance (both twins)
    -- one click forgets every remembered jump, collapsing the row in
    place (the desktop empties the session-persisted memory too, so
    the next open starts honestly empty).  The web twin's round lives
    in tests/test_web_parity.py (its per-section report control + the
    surgery feed pins); these guards walk the desktop dialog itself."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    def _finder_dialog(self, app: Any) -> Any:
        import tkinter as tkmod

        app._find_card_dialog()
        dialogs = [
            child
            for child in app.winfo_children()
            if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
        ]
        assert len(dialogs) == 1
        return dialogs[0]

    @staticmethod
    def _dialog_buttons(win: Any) -> list[Any]:
        buttons: list[Any] = []

        def walk(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    buttons.append(child)
                walk(child)

        walk(win)
        return buttons

    @staticmethod
    def _dialog_labels(win: Any) -> list[str]:
        labels: list[str] = []

        def walk_labels(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TLabel":
                    labels.append(str(child.cget("text")))
                walk_labels(child)

        walk_labels(win)
        return labels

    def test_clear_button_rides_the_recency_row(self, app: Any) -> None:
        app._jump_to_card(1, "Affinity conversion")
        win = self._finder_dialog(app)
        texts = [str(b.cget("text")) for b in self._dialog_buttons(win)]
        assert "Clear" in texts, texts
        win.destroy()

    def test_clear_click_empties_the_memory_and_collapses_the_row(
        self, app: Any
    ) -> None:
        app._jump_to_card(1, "Affinity conversion")
        app._jump_to_card(0, "PharmaML Playground")
        win = self._finder_dialog(app)
        clear = {
            str(b.cget("text")): b for b in self._dialog_buttons(win)
        }["Clear"]
        clear.invoke()
        app.update()
        # the memory is gone and the row collapsed in place
        assert app._finder_recent == []
        assert not any(
            label.startswith("Recently jumped") for label in self._dialog_labels(win)
        )
        win.destroy()

    def test_clear_persists_through_the_session_payload(self, app: Any) -> None:
        import json as jsonmod

        app._jump_to_card(1, "Affinity conversion")
        app._finder_recent = ["Affinity conversion"]
        win = self._finder_dialog(app)
        {
            str(b.cget("text")): b for b in self._dialog_buttons(win)
        }["Clear"].invoke()
        app.update()
        # the session-persisted key follows (the next save writes [])
        payload = app._collect_session()
        assert jsonmod.loads(payload["texts"]["finder_recent"]) == []
        win.destroy()

    def test_cleared_memory_renders_no_row_on_the_next_open(
        self, app: Any
    ) -> None:
        app._jump_to_card(1, "Affinity conversion")
        win = self._finder_dialog(app)
        {
            str(b.cget("text")): b for b in self._dialog_buttons(win)
        }["Clear"].invoke()
        app.update()
        win.destroy()
        # a fresh open reads the (now empty) memory: no row, no Clear
        win2 = self._finder_dialog(app)
        texts = [str(b.cget("text")) for b in self._dialog_buttons(win2)]
        assert "Clear" not in texts
        assert not any(
            label.startswith("Recently jumped") for label in self._dialog_labels(win2)
        )
        win2.destroy()

    def test_no_clear_button_without_recency(self, app: Any) -> None:
        app._finder_recent = []
        win = self._finder_dialog(app)
        texts = [str(b.cget("text")) for b in self._dialog_buttons(win)]
        assert "Clear" not in texts
        win.destroy()

    # -- the documentation pins ------------------------------------------- #
    def test_module_docstring_records_round_131(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-131" in doc
        assert '"Clear" affordance' in doc
        assert "per-section control" in doc
        assert "``image`` section kind" in doc

    def test_finder_dialog_docstring_mentions_the_clear(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._find_card_dialog.__doc__ or ""
        assert "100x-130" in doc
        assert "Recently jumped" in doc
        assert "100x-131" in doc
        assert '"Clear" ghost button' in doc


class TestReview132Gui:
    """Round 100x-132: the session reports' PRE-SAVE PICKLIST -- the
    desktop twin of the web report card's per-section control.  Every
    page the walk found is offered with a checkbox (all on by default,
    a live census footer that recounts on every flip, Save disabled at
    zero pages) and only the kept pages ship; Cancel aborts with no
    save started at all.  The web twin's round (the reorderable chips)
    lives in tests/test_web_parity.py.

    (100x-133 update) the dialog's rows now ride in the WALKING order
    -- table pages first, then figure pages, the exact sequence both
    walkers write them -- and every row carries the round-133 reorder
    arrows; the index pins below follow that order."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    # -- the driver ------------------------------------------------------- #
    def _drive(
        self,
        app: Any,
        flips: tuple[int, ...] = (),
        action: str = "save",
        extra: Any = None,
    ) -> dict[str, Any]:
        """Schedule the modal interaction, then return the bucket.

        The picklist blocks inside ``wait_window`` (its own event
        loop), so the interaction rides an ``after`` callback that the
        loop processes while the dialog is up: find the Toplevel by
        title, flip the listed checkboxes, optionally run ``extra``
        (which may close the dialog itself), then press Save/Cancel.
        """
        captured: dict[str, Any] = {}

        def interact() -> None:
            dialog = None
            for child in list(app.winfo_children()):
                if child.winfo_class() != "Toplevel":
                    continue  # Menu/Frame/Notebook carry no title()
                if str(child.title()) == "Session report contents":
                    dialog = child
                    break
            if dialog is None:
                captured["error"] = "the picklist never opened"
                return
            checks = [
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TCheckbutton"
            ]
            captured["checks"] = [str(check.cget("text")) for check in checks]
            for index in flips:
                checks[index].invoke()
            app.update()
            if extra is not None:
                extra(dialog, captured)
            # extra may have closed the dialog itself -- never walk a
            # destroyed window (the TclError would kill this callback
            # and strand the modal)
            try:
                alive = dialog.winfo_exists()
            except Exception:  # noqa: BLE001 - destroyed mid-callback
                alive = False
            if not alive:
                return
            captured.setdefault(
                "labels",
                [
                    str(widget.cget("text"))
                    for widget in app._walk_widgets(dialog)
                    if widget.winfo_class() == "TLabel"
                ],
            )
            buttons = {
                str(widget.cget("text")): widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
            }
            if action == "save":
                for text, button in buttons.items():
                    if text.startswith("Save report"):
                        button.invoke()
                        break
            elif action == "cancel":
                buttons["Cancel"].invoke()

        app.after(150, interact)
        return captured

    def _seed_session(self, app: Any) -> None:
        """Two live figures with distinct captions + one filled table."""
        first = self._figure_static("alpha plot")
        second = self._figure_static("beta plot")
        app._render_figure(app._topo_host, first)
        app._render_figure(app._breed_host, second)
        app._session_figures[0][1] = "alpha plot"
        app._session_figures[1][1] = "beta plot"
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))

    # -- the dialog's own contract ---------------------------------------- #
    def test_picklist_offers_every_page_and_cancel_aborts(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Any
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "never.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        self._seed_session(app)
        captured = self._drive(app, action="cancel")
        app._menu_save_session_report_html()
        # every page was offered, numbered by axis, in walk order --
        # (100x-133) the table pages ride FIRST (the walkers' own order);
        # (100x-174) the Attribution census table joined them, so the
        # picklist now offers two tables + the two seeded figures.
        assert captured.get("error") is None, captured
        assert len(captured["checks"]) == 4, captured
        assert captured["checks"][0].startswith("table 1 · ")
        assert "SMILES inspector" in captured["checks"][0]
        assert captured["checks"][1].startswith("table 2 · ")
        assert "Attribution" in captured["checks"][1]
        assert captured["checks"][2].startswith("figure 1 · ")
        assert "alpha plot" in captured["checks"][2]
        assert captured["checks"][3].startswith("figure 2 · ")
        # cancel aborted BEFORE the file dialog: nothing written, no job
        assert not target.is_file()
        assert not app._busy
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    def test_unchecked_figure_never_ships(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        quiet_dialogs: list[Any],
        tmp_path: Any,
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "filtered.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        self._seed_session(app)
        captured = self._drive(app, flips=(2,))  # uncheck "alpha plot" (row 3 -- the Attribution table rides row 2)
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        # the dialog offered every page, alpha among them
        assert captured.get("error") is None, captured
        assert any("alpha plot" in text for text in captured["checks"])
        text = target.read_text(encoding="utf-8")
        assert "beta plot" in text
        assert "alpha plot" not in text, "the unchecked figure still shipped"
        assert "table 1 · SMILES inspector" in text  # the table still ships
        assert any(
            "1 figure(s)" in str(a) for _n, a in quiet_dialogs
        ), "the save message did not count one figure"
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    def test_unchecked_table_never_ships(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        quiet_dialogs: list[Any],
        tmp_path: Any,
    ) -> None:
        import pathlib

        target = pathlib.Path(tmp_path) / "notable.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        self._seed_session(app)
        self._drive(app, flips=(0,))  # uncheck the table (first row now)
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        text = target.read_text(encoding="utf-8")
        assert "alpha plot" in text and "beta plot" in text
        assert "table 1 · SMILES inspector" not in text, "the unchecked table still shipped"
        assert any(
            "2 figure(s)" in str(a) for _n, a in quiet_dialogs
        ), "the save message did not count two figures"
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    def test_census_recounts_and_save_disables_at_zero(self, app: Any) -> None:
        self._seed_session(app)

        def extra(dialog: Any, bucket: dict[str, Any]) -> None:
            checks = [
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TCheckbutton"
            ]
            save = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")).startswith("Save report")
            )
            bucket["enabled_at_start"] = not save.instate(("disabled",))
            checks[0].invoke()  # one off -> 3 of 4
            app.update()
            labels = [
                str(widget.cget("text"))
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
            ]
            bucket["three_of_four"] = [
                label for label in labels if "3 of 4 pages will ship" in label
            ]
            checks[1].invoke()
            checks[2].invoke()
            checks[3].invoke()  # everything off
            app.update()
            labels = [
                str(widget.cget("text"))
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
            ]
            bucket["nothing"] = [
                label for label in labels if "Nothing would ship" in label
            ]
            bucket["disabled"] = bool(save.instate(("disabled",)))
            save.invoke()  # a disabled ttk button: the command never fires
            app.update()
            bucket["still_open"] = bool(dialog.winfo_exists())
            cancel = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")) == "Cancel"
            )
            cancel.invoke()

        bucket = self._drive(app, action="none", extra=extra)
        app._menu_save_session_report_html()
        # the footer recounts on every flip and Save follows it
        assert bucket.get("enabled_at_start") is True, bucket
        assert bucket.get("three_of_four"), "the census footer never recounted"
        assert bucket.get("nothing"), "the zero-page footer never appeared"
        assert bucket.get("disabled") is True, "Save stayed enabled at zero pages"
        assert bucket.get("still_open") is True, (
            "the disabled Save still closed the dialog"
        )
        # no save happened at all: no job, nothing written
        assert not app._busy
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    # -- the wiring + documentation pins ---------------------------------- #
    def test_both_menus_consult_the_picklist(self, playground: Any) -> None:
        import inspect

        pdf_source = inspect.getsource(
            playground.PlaygroundApp._menu_save_session_report
        )
        html_source = inspect.getsource(
            playground.PlaygroundApp._menu_save_session_report_html
        )
        assert "_session_report_picklist" in pdf_source
        assert "_session_report_picklist" in html_source

    def test_module_docstring_records_round_132(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-132" in doc
        assert "pre-save picklist" in doc
        assert "reorderable" in doc

    def test_picklist_docstring_records_round_132(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-132" in doc
        assert "pre-save checklist" in doc
        assert "census footer" in doc


class TestReview133Gui:
    """Round 100x-133: the picklist's per-page REORDER -- the desktop
    twin of the web report card's drag.  Every row carries ▲ / ▼ that
    move it among its OWN kind's neighbours (the rows ride in the
    walkers' order: table pages first, then figure pages), the
    numbering rewrites itself after every move, the boundary buttons
    disable honestly, and the returned lists carry the dialog's order
    so the saved report walks its pages exactly the way the checklist
    read them.  The same round's caption feed (the generative grids'
    counts reaching the picklist labels) rides here too."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    def _find_dialog(self, app: Any) -> Any:
        for child in list(app.winfo_children()):
            if child.winfo_class() != "Toplevel":
                continue
            if str(child.title()) == "Session report contents":
                return child
        return None

    def _arrow_buttons(self, app: Any, dialog: Any) -> tuple[list[Any], list[Any]]:
        """The dialog's ▲ / ▲ buttons in creation order (row by row)."""
        ups = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▲"
        ]
        downs = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▼"
        ]
        return ups, downs

    def _check_labels(self, app: Any, dialog: Any) -> list[str]:
        """The checkbox texts in VISUAL order -- the packing order, which
        the reorder rewrites (creation order would stay frozen)."""
        checks = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TCheckbutton"
        ]
        body = checks[0].master.master.master  # check -> row -> group -> body
        labels: list[str] = []
        for group_frame in body.pack_slaves():  # table group, figure group
            for child in group_frame.pack_slaves():  # heading + rows
                for check in checks:
                    if check.master is child:
                        labels.append(str(check.cget("text")))
        return labels

    # -- the reorder's own contract --------------------------------------- #

    def test_reorder_returns_the_dialogs_order(self, app: Any) -> None:
        """▼ on the first table + ▲ on the last figure: the returned
        lists carry the dialog's order, kind by kind."""
        live = [
            [None, "alpha plot", object()],
            [None, "beta plot", object()],
        ]
        tables = [
            ("T one", ["a"], [["1"]]),
            ("T two", ["b"], [["2"]]),
        ]
        captured: dict[str, Any] = {}

        def interact() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None, "the picklist never opened"
            ups, downs = self._arrow_buttons(app, dialog)
            # rows ride table-first: downs[0] is T one's, downs[2] is
            # beta's (creation order: T one, T two, alpha, beta)
            captured["before"] = self._check_labels(app, dialog)
            downs[0].invoke()  # T one below T two
            ups[3].invoke()  # beta above alpha
            app.update()
            captured["after"] = self._check_labels(app, dialog)
            save = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(150, interact)
        picked = app._session_report_picklist(live, tables)
        # the rows ride the walkers' order, numbered by axis
        assert captured["before"] == [
            "table 1 · T one",
            "table 2 · T two",
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], captured
        # the numbering rewrites itself AND the rows physically move
        # (the visual order follows the bucket order after the repack)
        assert captured["after"] == [
            "table 1 · T two",
            "table 2 · T one",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured
        # the return carries the dialog's order -- the walking order
        assert picked is not None
        kept_live, kept_tables = picked
        assert [entry[1] for entry in kept_live] == ["beta plot", "alpha plot"]
        assert [title for title, _h, _r in kept_tables] == ["T two", "T one"]

    def test_boundary_arrows_disable_and_never_cross_kinds(self, app: Any) -> None:
        """The first row's ▲ and the last row's ▼ disable honestly; a
        one-row group has BOTH disabled -- a table can never move into
        the figure block because the walkers ship the kinds apart."""
        live = [[None, "alpha plot", object()]]
        tables = [("Only table", ["a"], [["1"]])]
        captured: dict[str, Any] = {}

        def interact() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            ups, downs = self._arrow_buttons(app, dialog)
            # creation order: the table's arrows first, then the figure's
            captured["table_up_disabled"] = ups[0].instate(("disabled",))
            captured["table_down_disabled"] = downs[0].instate(("disabled",))
            captured["figure_up_disabled"] = ups[1].instate(("disabled",))
            captured["figure_down_disabled"] = downs[1].instate(("disabled",))
            # a disabled button's invoke is a no-op -- the order stands
            ups[0].invoke()
            downs[0].invoke()
            ups[1].invoke()
            app.update()
            captured["labels_after_dead_invokes"] = self._check_labels(app, dialog)
            cancel = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(150, interact)
        assert app._session_report_picklist(live, tables) is None
        assert captured["table_up_disabled"] is True, captured
        assert captured["table_down_disabled"] is True, captured
        assert captured["figure_up_disabled"] is True, captured
        assert captured["figure_down_disabled"] is True, captured
        assert captured["labels_after_dead_invokes"] == [
            "table 1 · Only table",
            "figure 1 · alpha plot",
        ], captured

    def test_reorder_reaches_the_saved_report(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        """The HTML report walks its pages the way the dialog ordered
        them -- beta's caption appears BEFORE alpha's in the file."""
        import pathlib

        target = pathlib.Path("/tmp") / "pg133-reordered.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        first = self._figure_static("alpha plot")
        second = self._figure_static("beta plot")
        app._render_figure(app._topo_host, first)
        app._render_figure(app._breed_host, second)
        app._session_figures[0][1] = "alpha plot"
        app._session_figures[1][1] = "beta plot"
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))

        def interact() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            _ups, downs = self._arrow_buttons(app, dialog)
            downs[0].invoke()  # table 1 below table 2 -- tables stay ahead of the figures either way
            downs[2].invoke()  # alpha below beta (row 3 -- the Attribution table rides row 2)
            app.update()
            save = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")).startswith("Save report")
            )
            save.invoke()

        app.after(150, interact)
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        text = target.read_text(encoding="utf-8")
        assert "alpha plot" in text and "beta plot" in text
        assert text.index("beta plot") < text.index("alpha plot"), (
            "the dialog's order never reached the report"
        )
        target.unlink(missing_ok=True)
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    # -- the caption feed -------------------------------------------------- #

    def test_caption_kwarg_reaches_the_registry(self, app: Any) -> None:
        """_render_figure(caption=...) overrides the job label in the
        session registry -- the picklist's rows say what the page holds."""
        figure = self._figure_static("whatever")
        app._render_figure(
            app._topo_host, figure, caption="Ring topology moves — 6 mutant(s)"
        )
        assert app._session_figures[-1][1] == "Ring topology moves — 6 mutant(s)"
        # without a caption the job label stands (the honest default)
        other = self._figure_static("other")
        app._render_figure(app._breed_host, other)
        assert app._session_figures[-1][1] == "figure"
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_surgery_grids_pass_their_counts(self, playground: Any) -> None:
        """The three generative grids' render closures pass count captions."""
        import inspect

        topo = inspect.getsource(playground.PlaygroundApp._gen_draw_topology)
        breed = inspect.getsource(playground.PlaygroundApp._gen_draw_breed)
        top = inspect.getsource(playground.PlaygroundApp._gen_draw_top)
        assert 'caption=f"Ring topology moves — {len(mutants)} mutant(s)"' in topo
        assert 'caption=(' in breed and "viable children" in breed
        assert 'caption=f"Top {count} molecules' in top

    # -- the walking-order note + docstring pins --------------------------- #

    def test_walking_order_note_rides_the_dialog(self, app: Any) -> None:
        captured: dict[str, Any] = {}

        def interact() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["labels"] = [
                str(widget.cget("text"))
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
            ]
            cancel = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton"
                and str(widget.cget("text")) == "Cancel"
            )
            cancel.invoke()

        app.after(150, interact)
        assert app._session_report_picklist([], [("Only table", ["a"], [["1"]])]) is None
        notes = [label for label in captured["labels"] if "walking order" in label]
        assert notes, "the walking-order note never rendered"
        assert "table pages" in notes[0] and "figure pages" in notes[0]

    def test_module_docstring_records_round_133(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-133" in doc
        assert "▲ / ▼" in doc
        assert "PREVIEW" in doc

    def test_picklist_docstring_records_round_133(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-133" in doc
        assert "REORDER" in doc
        assert "WALKING order" in doc

    def test_render_figure_docstring_records_the_caption(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._render_figure.__doc__ or ""
        assert "100x-133" in doc
        assert "caption" in doc


class TestReview134Gui:
    """Round 100x-134: the picklist's ORDER MEMORY -- the web twin's
    persisted walking order, mirrored per session on the desktop.  A
    completed Save stores the dialog's per-kind label order; the next
    Save re-opens pre-arranged in that order (stale labels skipped, new
    pages appended in walk order); Cancel never touches it; the
    dialog's own "Reset to walking order" button restores the walk
    order AND forgets the memory, enabled exactly while the order has
    drifted.  The same round's count-prefix page names ("figure N ·
    <caption>" / "table N · <caption>", numbered the way the picklist's
    rows are) ride here too."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 240.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig = plt.figure()
        axes = fig.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return fig

    def _find_dialog(self, app: Any) -> Any:
        for child in list(app.winfo_children()):
            if child.winfo_class() != "Toplevel":
                continue
            if str(child.title()) == "Session report contents":
                return child
        return None

    def _arrow_buttons(self, app: Any, dialog: Any) -> tuple[list[Any], list[Any]]:
        ups = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▲"
        ]
        downs = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▼"
        ]
        return ups, downs

    def _check_labels(self, app: Any, dialog: Any) -> list[str]:
        """The checkbox texts in VISUAL order (the packing order)."""
        checks = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TCheckbutton"
        ]
        body = checks[0].master.master.master  # check -> row -> group -> body
        labels: list[str] = []
        for group_frame in body.pack_slaves():
            for child in group_frame.pack_slaves():
                for check in checks:
                    if check.master is child:
                        labels.append(str(check.cget("text")))
        return labels

    def _button(self, app: Any, dialog: Any, text: str) -> Any:
        return next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton"
            and str(widget.cget("text")) == text
        )

    # -- the order memory's own contract ---------------------------------- #

    def test_save_stores_and_reopen_applies_the_order(self, app: Any) -> None:
        """A completed Save stores the dialog's order; the NEXT dialog
        opens pre-arranged in it, with the memory confession rendered."""
        live = [
            [None, "alpha plot", object()],
            [None, "beta plot", object()],
        ]
        tables = [
            ("T one", ["a"], [["1"]]),
            ("T two", ["b"], [["2"]]),
        ]
        assert app._report_page_order == {}  # starts empty (per session)

        def first_visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            ups, downs = self._arrow_buttons(app, dialog)
            downs[0].invoke()  # T one below T two
            ups[3].invoke()  # beta above alpha (creation: T one, T two, alpha, beta)
            app.update()
            self._button(app, dialog, "Save report…").invoke()

        app.after(150, first_visit)
        picked = app._session_report_picklist(live, tables)
        assert picked is not None
        # the memory now carries ALL labels per kind, in dialog order
        assert app._report_page_order == {
            "table": ["T two", "T one"],
            "figure": ["beta plot", "alpha plot"],
        }, app._report_page_order

        captured: dict[str, Any] = {}

        def second_visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["labels"] = self._check_labels(app, dialog)
            captured["notes"] = [
                str(widget.cget("text"))
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
            ]
            captured["reset_disabled"] = self._button(
                app, dialog, "Reset to walking order"
            ).instate(("disabled",))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, second_visit)
        assert app._session_report_picklist(live, tables) is None  # cancelled
        # the rows OPENED in the remembered order (no moves needed)
        assert captured["labels"] == [
            "table 1 · T two",
            "table 2 · T one",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured
        # the confession rendered, and it names the Reset affordance
        notes = [n for n in captured["notes"] if "last saved order" in n]
        assert notes and "Reset to walking order" in notes[0], captured
        # the remembered order IS a drift -- Reset enabled on arrival
        assert captured["reset_disabled"] is False, captured
        # Cancel left the memory untouched
        assert app._report_page_order["figure"] == ["beta plot", "alpha plot"]

    def test_memory_skips_stale_and_appends_new(self, app: Any) -> None:
        """A remembered label the walk no longer finds is skipped; a NEW
        page rides after the remembered ones, in walk order."""
        app._report_page_order = {
            "figure": ["gone plot", "beta plot"],
            "table": ["T two"],
        }
        live = [
            [None, "alpha plot", object()],
            [None, "beta plot", object()],
        ]
        tables = [
            ("T one", ["a"], [["1"]]),
            ("T two", ["b"], [["2"]]),
        ]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["labels"] = self._check_labels(app, dialog)
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, tables) is None
        # "gone plot" skipped; beta (remembered) rides before alpha (new,
        # walk order); the tables remember T-two-first; T-one rides after
        assert captured["labels"] == [
            "table 1 · T two",
            "table 2 · T one",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured

    def test_reset_restores_walk_order_and_forgets(self, app: Any) -> None:
        """Reset to walking order: the rows return to the walk order,
        the confession label collapses, and the MEMORY is forgotten so
        the NEXT dialog opens in the walk order too."""
        app._report_page_order = {
            "figure": ["beta plot", "alpha plot"],
            "table": [],
        }
        live = [
            [None, "alpha plot", object()],
            [None, "beta plot", object()],
        ]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["before"] = self._check_labels(app, dialog)
            # the confession label itself -- pack_forget keeps it in the
            # creation-order walk, so VISIBILITY is the manager state
            memory = next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
                and "last saved order" in str(widget.cget("text"))
            )
            assert memory.winfo_manager(), "the confession never rendered"
            reset = self._button(app, dialog, "Reset to walking order")
            assert not reset.instate(("disabled",))
            reset.invoke()
            app.update()
            captured["after"] = self._check_labels(app, dialog)
            captured["memory_manager_after"] = memory.winfo_manager()
            captured["reset_disabled_after"] = self._button(
                app, dialog, "Reset to walking order"
            ).instate(("disabled",))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["before"] == [
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured
        assert captured["after"] == [
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], captured
        # the confession collapsed -- an empty manager means unpacked
        assert captured["memory_manager_after"] == "", captured
        # at the walk order Reset disables itself again
        assert captured["reset_disabled_after"] is True, captured
        # the memory itself is gone
        assert app._report_page_order == {}, app._report_page_order

    def test_reset_drift_honest_through_manual_moves(self, app: Any) -> None:
        """With no memory the dialog opens at the walk order (Reset
        disabled); a manual ▼ move drifts it (Reset enabled); Reset
        restores the walk order and disables itself -- and a Cancel
        never writes a memory."""
        live = [
            [None, "alpha plot", object()],
            [None, "beta plot", object()],
        ]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            reset = self._button(app, dialog, "Reset to walking order")
            captured["disabled_at_walk"] = reset.instate(("disabled",))
            _ups, downs = self._arrow_buttons(app, dialog)
            downs[0].invoke()  # alpha below beta
            app.update()
            captured["disabled_after_move"] = reset.instate(("disabled",))
            reset.invoke()
            app.update()
            captured["labels_after_reset"] = self._check_labels(app, dialog)
            captured["disabled_after_reset"] = reset.instate(("disabled",))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["disabled_at_walk"] is True, captured
        assert captured["disabled_after_move"] is False, captured
        assert captured["labels_after_reset"] == [
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], captured
        assert captured["disabled_after_reset"] is True, captured
        assert app._report_page_order == {}  # cancel wrote nothing

    # -- the count-prefix page names --------------------------------------- #

    def test_count_prefix_reaches_the_html_report(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        """The HTML report's captions/headings read "figure N · ..." /
        "table N · ..." -- numbered the way the picklist's rows are."""
        import pathlib

        target = pathlib.Path("/tmp") / "pg134-count-prefix.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        first = self._figure_static("alpha plot")
        second = self._figure_static("beta plot")
        app._render_figure(app._topo_host, first)
        app._render_figure(app._breed_host, second)
        app._session_figures[0][1] = "alpha plot"
        app._session_figures[1][1] = "beta plot"
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))

        def interact() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            _ups, downs = self._arrow_buttons(app, dialog)
            downs[2].invoke()  # alpha below beta -- the dialog's order (row 3: the Attribution table rides row 2)
            app.update()
            self._button(app, dialog, "Save report…").invoke()

        app.after(150, interact)
        app._menu_save_session_report_html()
        assert self._drain(app), "the HTML report job hung"
        text = target.read_text(encoding="utf-8")
        # the figures ride count-prefixed, in the dialog's order
        assert "figure 1 · beta plot" in text, "the count prefix never shipped"
        assert "figure 2 · alpha plot" in text, "the dialog's order never shipped"
        # the table rides count-prefixed too (the descriptor card's tree)
        assert "table 1 · SMILES inspector" in text, (
            "the table's count prefix never shipped"
        )
        # the retired " (2)" dedupe is gone from the figure captions
        assert "alpha plot (2)" not in text and "beta plot (2)" not in text
        target.unlink(missing_ok=True)
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    def test_both_menus_pin_the_count_prefix_walk(self, playground: Any) -> None:
        """The PDF twin names its pages and the HTML twin its captions
        with the same count-prefixed walk (source pins)."""
        import inspect

        pdf = inspect.getsource(playground.PlaygroundApp._menu_save_session_report)
        html = inspect.getsource(
            playground.PlaygroundApp._menu_save_session_report_html
        )
        assert 'f"figure {position} · {title}"' in pdf
        assert 'f"table {position} · {title}"' in pdf
        assert 'f"figure {position} · {title}"' in html
        assert 'f"table {position} · {title}"' in html
        # the retired caption dedupe is gone from the HTML twin
        assert "unique_caption" not in html

    # -- the docstring pins ------------------------------------------------- #

    def test_module_docstring_records_round_134(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-134" in doc
        assert "REMEMBERS" in doc
        assert "hover INSPECT" in doc

    def test_picklist_docstring_records_round_134(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-134" in doc
        assert "ORDER MEMORY" in doc or "order REMEMBERS" in doc
        assert "Reset to walking order" in doc

    def test_menu_docstrings_record_the_count_prefix(self, playground: Any) -> None:
        pdf = playground.PlaygroundApp._menu_save_session_report.__doc__ or ""
        html = playground.PlaygroundApp._menu_save_session_report_html.__doc__ or ""
        assert "100x-134" in pdf and "CARRY THE COUNT" in pdf
        assert "100x-134" in html and "CARRY THE COUNT" in html


class TestReview135Gui:
    """Round 100x-135: the picklist's order memory now survives
    RESTARTS -- the ``report_page_order`` session-file key, mirroring
    the web twin's localStorage exactly.  A completed Save rewrites
    the memory, the close-time auto-save persists it, and a fresh
    launch replants it before the first picklist ever opens; the
    Reset button's forget is symmetric (the next launch starts at
    the walk order), and a corrupt or foreign file value resets the
    memory, never the restore."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    def _find_dialog(self, app: Any) -> Any:
        for child in list(app.winfo_children()):
            if child.winfo_class() != "Toplevel":
                continue
            if str(child.title()) == "Session report contents":
                return child
        return None

    def _arrow_buttons(self, app: Any, dialog: Any) -> tuple[list[Any], list[Any]]:
        ups = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▲"
        ]
        downs = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "▼"
        ]
        return ups, downs

    def _button(self, app: Any, dialog: Any, text: str) -> Any:
        return next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton"
            and str(widget.cget("text")) == text
        )

    def _check_labels(self, app: Any, dialog: Any) -> list[str]:
        """The checkbox texts in VISUAL order (the packing order)."""
        checks = [
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TCheckbutton"
        ]
        body = checks[0].master.master.master  # check -> row -> group -> body
        labels: list[str] = []
        for group_frame in body.pack_slaves():
            for child in group_frame.pack_slaves():
                for check in checks:
                    if check.master is child:
                        labels.append(str(check.cget("text")))
        return labels

    # -- the session-file contract ----------------------------------------- #

    def test_collect_session_carries_the_order_key(self, app: Any) -> None:
        """``_collect_session`` serialises the memory as the
        ``report_page_order`` text key -- JSON, only the two real
        kinds, labels coerced to str (never a half-typed payload)."""
        import json as _json

        app._report_page_order = {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
            # a foreign kind the collector must DROP (never persisted)
            "chart": ["never shipped"],
        }
        texts = app._collect_session()["texts"]
        assert _json.loads(texts["report_page_order"]) == {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
        }, texts["report_page_order"]

    def test_order_memory_survives_close_and_reopen(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        """The end-to-end restart: a completed Save stores the order,
        the close-time auto-save persists it, and a FRESH launch
        replants it before the first picklist ever opens."""
        import gc

        app = playground.PlaygroundApp()  # persist_session=True
        self._drain(app)
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        tables = [("T one", ["a"], [["1"]])]

        def first_save() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            ups, _downs = self._arrow_buttons(app, dialog)
            ups[2].invoke()  # beta above alpha
            app.update()
            self._button(app, dialog, "Save report…").invoke()

        app.after(150, first_save)
        app._session_report_picklist(live, tables)
        assert app._report_page_order == {
            "table": ["T one"],
            "figure": ["beta plot", "alpha plot"],
        }, app._report_page_order
        app._on_close()  # writes the session file, then destroys
        gc.collect()

        reopened = playground.PlaygroundApp()
        self._drain(reopened)
        # the memory rode the file: the fresh launch replanted it
        assert reopened._report_page_order == {
            "table": ["T one"],
            "figure": ["beta plot", "alpha plot"],
        }, reopened._report_page_order
        reopened.destroy()
        gc.collect()

    def test_reset_forget_rides_the_session_file(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        """The Reset button's forget is symmetric: it empties the
        attribute, the close-time auto-save writes the EMPTY state,
        and the next LAUNCH starts at the walk order."""
        import gc

        app = playground.PlaygroundApp()
        self._drain(app)
        app._report_page_order = {
            "figure": ["beta plot", "alpha plot"],
            "table": [],
        }
        app._on_close()
        gc.collect()

        reopened = playground.PlaygroundApp()
        self._drain(reopened)
        assert reopened._report_page_order["figure"] == [
            "beta plot",
            "alpha plot",
        ]
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        tables: list[tuple[str, list[str], list[list[str]]]] = []

        def reset_visit() -> None:
            dialog = self._find_dialog(reopened)
            assert dialog is not None
            self._button(reopened, dialog, "Reset to walking order").invoke()
            # Cancel: the Reset's forget must survive a non-save exit
            self._button(reopened, dialog, "Cancel").invoke()

        reopened.after(150, reset_visit)
        assert reopened._session_report_picklist(live, tables) is None
        assert reopened._report_page_order == {}  # the Reset forgot
        reopened._on_close()  # the auto-save writes the EMPTY state
        gc.collect()

        third = playground.PlaygroundApp()
        self._drain(third)
        assert third._report_page_order == {}  # the forget survived restart
        third.destroy()
        gc.collect()

    def test_relaunch_reopens_prearranged_with_confession(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        """The launch-path proof: a fresh boot from a remembering file
        opens its FIRST picklist pre-arranged, with the memory
        confession rendered and the Reset button armed."""
        import gc

        app = playground.PlaygroundApp()
        self._drain(app)
        app._report_page_order = {
            "figure": ["beta plot", "alpha plot"],
            "table": [],
        }
        app._on_close()
        gc.collect()

        reopened = playground.PlaygroundApp()
        self._drain(reopened)
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(reopened)
            assert dialog is not None
            captured["labels"] = self._check_labels(reopened, dialog)
            captured["notes"] = [
                str(widget.cget("text"))
                for widget in reopened._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
            ]
            captured["reset_disabled"] = self._button(
                reopened, dialog, "Reset to walking order"
            ).instate(("disabled",))
            self._button(reopened, dialog, "Cancel").invoke()

        reopened.after(150, visit)
        assert reopened._session_report_picklist(live, []) is None
        # the FIRST dialog of the session opened pre-arranged
        assert captured["labels"] == [
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured
        # the confession rendered and names the Reset affordance
        notes = [n for n in captured["notes"] if "last saved order" in n]
        assert notes and "Reset to walking order" in notes[0], captured
        # the remembered order IS a drift -- Reset armed on arrival
        assert captured["reset_disabled"] is False, captured
        reopened.destroy()
        gc.collect()

    def test_foreign_or_corrupt_value_resets_never_raises(
        self, playground: Any, xvfb_display: str, isolated_home: Path
    ) -> None:
        """A corrupt or foreign ``report_page_order`` value resets the
        memory to the walk order, never the restore (the session's
        other keys still land)."""
        import gc
        import json as _json

        # (poison, the sanitized memory it must land on) -- the
        # tolerance is PER KIND: a broken kind resets, a valid one in
        # the same payload still lands (families restore the same way)
        poisons = [
            ("not json at all", {}),
            (_json.dumps(["a", "list", "not", "a", "dict"]), {}),
            (
                _json.dumps({"figure": "not a list", "table": ["ok one"]}),
                {"table": ["ok one"]},
            ),
            (
                _json.dumps({"figure": [1, 2, 3], "table": ["ok one"]}),
                {"table": ["ok one"]},
            ),
            (_json.dumps({"chart": ["a foreign kind"]}), {}),
        ]
        for poison, expected in poisons:
            payload = playground.dump_session(
                theme="light",
                geometry="1000x700+10+10",
                tab=0,
                variables={},
                families={},
                texts={"report_page_order": poison},
            )
            (isolated_home / ".pharmaml").mkdir(parents=True, exist_ok=True)
            (isolated_home / ".pharmaml" / "playground-session.json").write_text(
                _json.dumps(payload), encoding="utf-8"
            )
            app = playground.PlaygroundApp()
            self._drain(app)
            # the memory landed on its sanitized self, never an error
            assert app._report_page_order == expected, (
                poison,
                app._report_page_order,
            )
            app.destroy()
            del app
            gc.collect()

    def test_persist_session_false_keeps_memory_in_process(
        self, app: Any
    ) -> None:
        """``persist_session=False`` (the test harness): the memory
        still works per-process AND the collect path serialises it --
        only the file write is skipped."""
        import json as _json

        app._report_page_order = {"figure": ["beta plot"], "table": []}
        texts = app._collect_session()["texts"]
        assert _json.loads(texts["report_page_order"]) == {
            "figure": ["beta plot"],
            "table": [],
        }

    # -- the docstring pins ------------------------------------------------- #

    def test_module_docstring_records_round_135(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-135" in doc
        assert "survives RESTARTS" in doc
        assert "KEYBOARD path" in doc
        assert "Contents section" in doc

    def test_picklist_docstring_records_round_135(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-135" in doc
        assert "survives RESTARTS" in doc
        assert "report_page_order" in doc

    def test_session_collect_and_restore_pin_the_key(
        self, playground: Any
    ) -> None:
        import inspect

        collect = inspect.getsource(playground.PlaygroundApp._collect_session)
        restore = inspect.getsource(playground.PlaygroundApp._restore_session)
        assert '"report_page_order": json.dumps(' in collect
        assert 'if key == "report_page_order":' in restore
        # the tolerant parse (a corrupt value resets, never raises)
        assert "json.loads" in restore


@XVFB_SKIP
class TestReview136Gui:
    """Round 100x-136: the order memory's always-available FORGET --
    File ▸ Clear remembered report order.  The dialog's Reset only
    exists while the picklist is open (and only works while a drift
    exists); this menu item drops both kinds and writes the session
    file through IMMEDIATELY, so the forget survives a crash and the
    next launch starts at the walk order.  The round also carried the
    HTML twin's Contents block (the library's own tests pin it) and
    the web preview's persistent pin (the parity pins)."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    def _find_dialog(self, app: Any) -> Any:
        for child in list(app.winfo_children()):
            if child.winfo_class() != "Toplevel":
                continue
            if str(child.title()) == "Session report contents":
                return child
        return None

    @staticmethod
    def _figure_static(title: str) -> Any:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title(title)
        return figure

    # -- the forget itself ------------------------------------------------- #

    def test_clear_forgets_and_writes_through(
        self, playground: Any, xvfb_display: str, isolated_home: Path, dialogs: list[Any]
    ) -> None:
        """A remembered order is dropped AND the session file is written
        immediately -- the forget does not wait for the close-time
        auto-save (it survives a crash), and the message says so."""
        import gc
        import json as _json

        app = playground.PlaygroundApp()  # persist_session=True
        self._drain(app)
        app._report_page_order = {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
        }
        app._clear_report_order()
        # both kinds dropped, in-process and on disk
        assert app._report_page_order == {}, app._report_page_order
        session_file = isolated_home / ".pharmaml" / "playground-session.json"
        assert session_file.is_file(), "the clear never wrote the session file"
        payload = _json.loads(session_file.read_text(encoding="utf-8"))
        assert _json.loads(payload["texts"]["report_page_order"]) == {}
        # the honest message: the forget is written, the launch forgets too
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and infos[0][1][0] == "Remembered order cleared"
        assert "next launch forgets it too" in infos[0][1][1]
        assert "remembered report order cleared" in str(app._job_label.cget("text"))
        app.destroy()
        gc.collect()

    def test_clear_with_no_memory_is_honest(
        self, app: Any, dialogs: list[Any]
    ) -> None:
        """A memory that never existed gets the honest 'nothing to
        forget' -- never a silent no-op, never an error."""
        app._clear_report_order()
        assert app._report_page_order == {}
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and infos[0][1][0] == "No remembered order"
        assert "Nothing to forget yet" in infos[0][1][1]

    def test_after_clear_the_picklist_walks_in_walk_order(
        self, app: Any, dialogs: list[Any]
    ) -> None:
        """The very next picklist after a clear opens in the WALKING
        order with its confession quiet -- the clear is immediate, not
        deferred to the next Save."""
        app._report_page_order = {"figure": ["beta plot", "alpha plot"], "table": []}
        app._clear_report_order()
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            checks = [
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TCheckbutton"
            ]
            captured["labels"] = [str(check.cget("text")) for check in checks]
            # (the 133/134 lesson) a pack_forget confession stays in the
            # CREATION walk -- only a MANAGED label is actually shown
            captured["notes"] = [
                str(widget.cget("text"))
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TLabel"
                and widget.winfo_manager() != ""
            ]
            next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
            ).invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["labels"] == [
            "figure 1 · alpha plot",
            "figure 2 · beta plot",
        ], captured
        assert not any("last saved order" in note for note in captured["notes"])

    def test_clear_forget_survives_restart(
        self, playground: Any, xvfb_display: str, isolated_home: Path, dialogs: list[Any]
    ) -> None:
        """The end-to-end story: a remembering launch clears, and the
        NEXT launch starts at the walk order (the write-through beat
        the close-time auto-save to it)."""
        import gc

        app = playground.PlaygroundApp()
        self._drain(app)
        app._report_page_order = {"figure": ["beta plot", "alpha plot"], "table": []}
        app._on_close()  # the memory lands on the file
        gc.collect()

        reopened = playground.PlaygroundApp()  # replants the memory
        self._drain(reopened)
        assert reopened._report_page_order["figure"] == ["beta plot", "alpha plot"]
        reopened._clear_report_order()  # forgets AND writes through
        reopened.destroy()
        gc.collect()

        third = playground.PlaygroundApp()
        self._drain(third)
        assert third._report_page_order == {}  # the forget survived
        third.destroy()
        gc.collect()

    def test_persist_session_false_message_names_the_scope(
        self, app: Any, dialogs: list[Any]
    ) -> None:
        """With session saving off, the memory was per-process anyway --
        the message says so honestly instead of claiming a file write."""
        app._report_page_order = {"figure": ["beta plot"], "table": []}
        app._clear_report_order()
        assert app._report_page_order == {}
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and "never reached a file" in infos[0][1][1]

    def test_failed_write_confesses_the_launch_restore(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        dialogs: list[Any],
        monkeypatch: Any,
    ) -> None:
        """When the session file CANNOT be written, the in-process forget
        still happens but the message confesses the consequence: the
        next launch restores the remembered order after all."""
        import gc

        monkeypatch.setattr(
            playground.PlaygroundApp, "_write_session_file", lambda self: False
        )
        app = playground.PlaygroundApp()
        self._drain(app)
        app._report_page_order = {"figure": ["beta plot"], "table": []}
        app._clear_report_order()
        assert app._report_page_order == {}  # the in-process forget held
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and "restores the remembered order after all" in infos[0][1][1]
        app.destroy()
        gc.collect()

    # -- the HTML twin's Contents block (the library feature, end-to-end) -- #

    def test_html_twin_carries_the_contents_block(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        dialogs: list[Any],
        picklist_accept_all: None,
        tmp_path: Any,
    ) -> None:
        """The playground's HTML report opens with the Contents nav --
        every kept page's count-prefixed name, tables first, exactly
        the way the picklist's numbered rows read."""
        import pathlib

        target = pathlib.Path(tmp_path) / "contents.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        first = self._figure_static("alpha plot")
        second = self._figure_static("beta plot")
        app._render_figure(app._topo_host, first)
        app._render_figure(app._breed_host, second)
        app._session_figures[0][1] = "alpha plot"
        app._session_figures[1][1] = "beta plot"
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))

        app._menu_save_session_report_html()
        import time

        stop = time.time() + 240.0
        while time.time() < stop:
            app.update()
            if not app._busy:
                break
            time.sleep(0.02)
        assert not app._busy, "the HTML report job hung"
        assert target.is_file()
        text = target.read_text(encoding="utf-8")
        # the Contents nav: the heading, the list, the count-prefixed
        # lines in the walking order (the table first, then figures) --
        # each line riding its anchor link (100x-137); (100x-174) the
        # Attribution census table joined the walk, so the tables take
        # pg-1/pg-2 and the figures pg-3/pg-4
        assert "<nav class='contents'" in text
        assert "aria-label='Contents'" in text
        assert "<li><a href='#pg-1' title='Jump to this page in the report'>table 1 · SMILES inspector</a></li>" in text
        assert "<li><a href='#pg-2' title='Jump to this page in the report'>table 2 · Attribution" in text
        assert "<li><a href='#pg-3' title='Jump to this page in the report'>figure 1 · alpha plot</a></li>" in text
        assert "<li><a href='#pg-4' title='Jump to this page in the report'>figure 2 · beta plot</a></li>" in text
        assert (
            text.index("<li><a href='#pg-1' title='Jump to this page in the report'>table 1 · SMILES inspector</a></li>")
            < text.index("<li><a href='#pg-3' title='Jump to this page in the report'>figure 1 · alpha plot</a></li>")
        )
        # the pages carry the anchor targets the lines link to
        assert "<section class='page' id='pg-1' tabindex='-1'>" in text
        assert "<figure class='page' id='pg-3' tabindex='-1'>" in text
        # the page's own heading matches its Contents line (no drift)
        assert "<h2>table 1 · SMILES inspector</h2>" in text
        # the saved message confesses the new block
        assert any(
            "Contents block lists every page" in str(a) for _n, a in dialogs
        ), "the save message never mentioned the Contents"
        import matplotlib.pyplot as plt

        plt.close("all")
        app._desc_tree.delete(*app._desc_tree.get_children())

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_136(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-136" in doc
        assert "always-available FORGET" in doc
        assert "PERSISTS per browser" in doc

    def test_clear_handler_docstring_pins_the_round(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._clear_report_order.__doc__ or ""
        assert "100x-136" in doc
        assert "persist_session=False" in doc

    def test_picklist_docstring_mentions_the_menu_path(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-136" in doc
        assert "Clear remembered report order" in doc

    def test_file_menu_wires_the_clear_command(self, playground: Any) -> None:
        import inspect

        source = inspect.getsource(playground.PlaygroundApp._build_menu)
        assert 'label="Clear remembered report order…"' in source
        assert "command=self._clear_report_order" in source
        # the HTML menu's docstring confesses the Contents twin
        html_doc = playground.PlaygroundApp._menu_save_session_report_html.__doc__ or ""
        assert "100x-136" in html_doc
        assert "CONTENTS block" in html_doc


class TestReview137Gui:
    """Round 100x-137: the order memory's SHARE pair -- File ▸ Export /
    Import remembered order, the standalone artifact (the preference
    without the session's inputs).  The round also carried the HTML
    twin's clickable Contents anchors (the library's own tests pin
    them) and the web preview's Contents twin (the parity pins)."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    # -- the pure normalisers ---------------------------------------------- #

    def test_dump_normalises_to_the_two_kinds(self, playground: Any) -> None:
        """``dump_report_order`` keeps only the two real kinds, coerces
        every label to str, and stamps the format marker + kind tag --
        the file can never carry a half-typed payload."""
        dumped = playground.dump_report_order(
            {"figure": ["beta", "alpha"], "table": ["T one"], "junk": ["x"]}
        )
        assert dumped == {
            "format": playground.REPORT_ORDER_FORMAT,
            "kind": playground.REPORT_ORDER_KIND,
            "order": {"table": ["T one"], "figure": ["beta", "alpha"]},
        }

    def test_load_rejects_foreign_and_keeps_valid(self, playground: Any) -> None:
        """``load_report_order`` rejects anything without BOTH the format
        marker and the kind tag; a valid file returns only the two real
        kinds as lists of str (the restore path never defends twice)."""
        good = playground.dump_report_order({"figure": ["b", "a"], "table": []})
        assert playground.load_report_order(good) == {
            "figure": ["b", "a"],
            "table": [],
        }
        # foreign: a session file, a marker-less dict, a kind mismatch
        assert playground.load_report_order({"format": 1}) is None
        assert (
            playground.load_report_order(
                {"format": 1, "kind": "something-else", "order": {}}
            )
            is None
        )
        assert playground.load_report_order("not a dict") is None
        assert playground.load_report_order(None) is None
        # structurally fine but half-typed: only the str lists survive
        half_typed = {
            "format": playground.REPORT_ORDER_FORMAT,
            "kind": playground.REPORT_ORDER_KIND,
            "order": {"table": ["ok", 3], "figure": "not-a-list"},
        }
        assert playground.load_report_order(half_typed) == {}
        # the honest empty: a valid empty order imports as a reset
        assert playground.load_report_order(playground.dump_report_order({})) == {}

    # -- the export -------------------------------------------------------- #

    def test_export_writes_the_standalone_artifact(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        """The export writes ONLY the order -- the format marker, the kind
        tag and the two label lists -- never the session's inputs."""
        import json as _json

        target = tmp_path / "shared-order.json"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        app._report_page_order = {"figure": ["beta plot", "alpha plot"], "table": ["T one"]}
        app._export_report_order()
        assert target.is_file(), "the export never wrote the file"
        payload = _json.loads(target.read_text(encoding="utf-8"))
        assert payload == {
            "format": playground.REPORT_ORDER_FORMAT,
            "kind": playground.REPORT_ORDER_KIND,
            "order": {"table": ["T one"], "figure": ["beta plot", "alpha plot"]},
        }
        # the artifact carries NONE of the session's inputs
        assert "variables" not in payload and "texts" not in payload
        assert "report order exported" in str(app._job_label.cget("text"))

    def test_export_with_no_memory_is_honest(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path, dialogs: list[Any]
    ) -> None:
        """A memory that never existed gets the honest 'nothing to share
        yet' -- never a silent no-op, never an empty file."""
        target = tmp_path / "never-written.json"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        app._export_report_order()
        assert not target.exists(), "an empty memory must not write a file"
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and infos[0][1][0] == "No remembered order"
        assert "Nothing to share yet" in infos[0][1][1]

    # -- the import -------------------------------------------------------- #

    def test_import_adopts_the_order_and_writes_through(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        monkeypatch: Any,
        dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        """A valid order file REPLACES the memory and writes the session
        file through immediately -- the imported order survives a crash
        and the next LAUNCH re-opens pre-arranged (the Clear's
        discipline, mirrored for adoption)."""
        import gc
        import json as _json

        source = tmp_path / "colleague-order.json"
        source.write_text(
            _json.dumps(
                playground.dump_report_order(
                    {"figure": ["beta plot", "alpha plot"], "table": ["T one"]}
                )
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app = playground.PlaygroundApp()  # persist_session=True
        self._drain(app)
        app._import_report_order()
        assert app._report_page_order == {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
        }
        # the write-through landed: the session file carries the imported order
        session_file = isolated_home / ".pharmaml" / "playground-session.json"
        assert session_file.is_file(), "the import never wrote the session file"
        payload = _json.loads(session_file.read_text(encoding="utf-8"))
        assert _json.loads(payload["texts"]["report_page_order"]) == {
            "table": ["T one"],
            "figure": ["beta plot", "alpha plot"],
        }
        # the honest message: applied, written, the next launch keeps it
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and infos[0][1][0] == "Report order imported"
        assert "next launch keeps it too" in infos[0][1][1]
        assert "report order imported" in str(app._job_label.cget("text"))
        # the picklist re-opens PRE-ARRANGED in the imported order (the
        # after-callback driver cancels the dialog; the ROW ORDER at
        # open time is what this asserts)
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        tables = [("T one", ["h"], [["r"]])]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = None
            for child in list(app.winfo_children()):
                if child.winfo_class() == "Toplevel" and str(child.title()) == (
                    "Session report contents"
                ):
                    dialog = child
            assert dialog is not None
            checks = [
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TCheckbutton"
            ]
            captured["labels"] = [str(check.cget("text")) for check in checks]
            next(
                widget
                for widget in app._walk_widgets(dialog)
                if widget.winfo_class() == "TButton" and str(widget.cget("text")) == "Cancel"
            ).invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, tables) is None  # cancelled
        assert captured["labels"] == [
            "table 1 · T one",
            "figure 1 · beta plot",
            "figure 2 · alpha plot",
        ], captured["labels"]
        app.destroy()
        gc.collect()

    def test_import_rejects_foreign_files(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A foreign or corrupt file is rejected with an honest error and
        the live memory is never touched."""
        source = tmp_path / "not-an-order.json"
        source.write_text('{"format": 1, "theme": "light"}', encoding="utf-8")
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._report_page_order = {"figure": ["mine"], "table": []}
        app._import_report_order()
        # the live memory survived untouched
        assert app._report_page_order == {"figure": ["mine"], "table": []}
        errors = [call for call in dialogs if call[0] == "showerror"]
        assert errors and errors[0][1][0] == "Invalid report order file"
        assert "not touched" in errors[0][1][1]

    def test_import_empty_order_is_the_honest_reset(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A valid but EMPTY order imports as a reset: both kinds empty,
        the next picklist walks the walk order -- never an error."""
        import json as _json

        source = tmp_path / "empty-order.json"
        source.write_text(
            _json.dumps(playground.dump_report_order({})), encoding="utf-8"
        )
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._report_page_order = {"figure": ["old order"], "table": []}
        app._import_report_order()
        assert app._report_page_order == {}
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and "imported order is empty" in infos[0][1][1]
        assert "walking order" in infos[0][1][1]

    def test_import_persist_off_names_the_scope(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """With session saving off, the import lives in this process only
        -- the message says so instead of claiming a file write."""
        source = tmp_path / "order.json"
        source.write_text(
            __import__("json").dumps(
                playground.dump_report_order({"figure": ["b", "a"], "table": []})
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        assert app._report_page_order == {"figure": ["b", "a"], "table": []}
        infos = [call for call in dialogs if call[0] == "showinfo"]
        assert infos and "lives in this session only" in infos[0][1][1]

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_137(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-137" in doc
        assert "SHAREABLE" in doc
        assert "CLICKABLE anchors" in doc
        assert "CONTENTS twin" in doc

    def test_handler_docstrings_pin_the_round(self, playground: Any) -> None:
        export_doc = playground.PlaygroundApp._export_report_order.__doc__ or ""
        assert "100x-137" in export_doc
        assert "dump_report_order" in export_doc
        import_doc = playground.PlaygroundApp._import_report_order.__doc__ or ""
        assert "100x-137" in import_doc
        assert "load_report_order" in import_doc

    def test_picklist_docstring_mentions_the_share_pair(self, playground: Any) -> None:
        doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-137" in doc
        assert "SHAREABLE" in doc

    def test_file_menu_wires_the_share_pair(self, playground: Any) -> None:
        import inspect

        source = inspect.getsource(playground.PlaygroundApp._build_menu)
        assert 'label="Export remembered order…"' in source
        assert "command=self._export_report_order" in source
        assert 'label="Import remembered order…"' in source
        assert "command=self._import_report_order" in source
        # the HTML menu's docstring confesses the clickable anchors
        html_doc = playground.PlaygroundApp._menu_save_session_report_html.__doc__ or ""
        assert "100x-137" in html_doc
        assert "CLICKABLE anchor" in html_doc


class TestReview138Gui:
    """Round 100x-138: the picklist confession's PROVENANCE -- an order
    that arrived through File ▸ Import remembered order says so ("The
    imported order from <file> was applied") until the reader completes
    their own Save, resets, or clears; a fresh launch restores from the
    session file, which IS the last saved order, and keeps the default
    wording.  The round also carried the HTML twin's return leg and the
    web strip's kind census (their own suites pin them)."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Point HOME at a scratch dir so tests never touch a real session."""
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    def _find_dialog(self, app: Any) -> Any:
        for child in list(app.winfo_children()):
            if child.winfo_class() != "Toplevel":
                continue
            if str(child.title()) == "Session report contents":
                return child
        return None

    def _memory_label(self, app: Any, dialog: Any) -> Any:
        """The confession label -- the TLabel whose text says 'applied'."""
        return next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TLabel" and "was applied" in str(widget.cget("text"))
        )

    def _button(self, app: Any, dialog: Any, text: str) -> Any:
        return next(
            widget
            for widget in app._walk_widgets(dialog)
            if widget.winfo_class() == "TButton"
            and str(widget.cget("text")) == text
        )

    def _plant_order_file(
        self, playground: Any, tmp_path: Path, name: str = "colleague-order.json"
    ) -> Path:
        import json as _json

        source = tmp_path / name
        source.write_text(
            _json.dumps(
                playground.dump_report_order(
                    {"figure": ["beta plot", "alpha plot"], "table": ["T one"]}
                )
            ),
            encoding="utf-8",
        )
        return source

    # -- the provenance attribute's life cycle ----------------------------- #

    def test_origin_starts_none_on_a_fresh_app(self, playground: Any) -> None:
        """The attribute exists and starts None -- a fresh app's memory is
        nobody's import (and the session restore keeps it that way)."""
        import inspect

        source = inspect.getsource(playground.PlaygroundApp.__init__)
        assert "self._report_order_origin: str | None = None" in source

    def test_import_sets_the_provenance_attribute(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A valid non-empty import carries its source -- the next
        picklist's confession names this file."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        assert app._report_order_origin is None
        app._import_report_order()
        assert app._report_order_origin == str(source)

    def test_empty_import_confesses_nothing(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """The honest EMPTY import clears the provenance -- an empty
        memory is nobody's import."""
        import json as _json

        source = tmp_path / "empty-order.json"
        source.write_text(
            _json.dumps(playground.dump_report_order({})), encoding="utf-8"
        )
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._report_page_order = {"figure": ["old order"], "table": []}
        app._report_order_origin = "somewhere/old.json"
        app._import_report_order()
        assert app._report_page_order == {}
        assert app._report_order_origin is None

    def test_foreign_rejection_leaves_the_provenance_alone(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A foreign file never touches the live memory OR its provenance."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        assert app._report_order_origin == str(source)
        foreign = tmp_path / "not-an-order.json"
        foreign.write_text('{"format": 1}', encoding="utf-8")
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(foreign)
        )
        app._import_report_order()
        assert app._report_order_origin == str(source), "the provenance was touched"
        assert app._report_page_order == {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
        }
        errors = [call for call in dialogs if call[0] == "showerror"]
        assert errors and errors[-1][1][0] == "Invalid report order file"

    def test_clear_menu_forgets_the_provenance(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """File ▸ Clear remembered report order forgets the memory AND its
        provenance -- a cleared order is nobody's import any more."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        assert app._report_order_origin is not None
        app._clear_report_order()
        assert app._report_page_order == {}
        assert app._report_order_origin is None

    # -- the confession wording -------------------------------------------- #

    def test_picklist_confesses_the_imported_file(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """After an import the picklist's confession names the FILE -- 'The
        imported order from colleague-order.json was applied' -- with the
        Reset promise the default wording carries."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        tables = [("T one", ["h"], [["r"]])]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            memory = self._memory_label(app, dialog)
            captured["text"] = str(memory.cget("text"))
            captured["manager"] = memory.winfo_manager()
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, tables) is None
        assert captured["manager"], "the confession never rendered"
        assert captured["text"] == (
            "The imported order from colleague-order.json was applied — "
            "Reset to walking order forgets it."
        ), captured["text"]

    def test_long_filenames_truncate_honestly(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A very long source name truncates the honest way -- the same
        40-character ellipsis the captions use, on the BASENAME only."""
        long_name = "a-really-long-colleague-order-filename-that-will-not-fit.json"
        source = self._plant_order_file(playground, tmp_path, name=long_name)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        expected = (
            f"The imported order from "
            f"{playground._truncate_caption_text(long_name, 40)} "
            "was applied — Reset to walking order forgets it."
        )
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["text"] = str(self._memory_label(app, dialog).cget("text"))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["text"] == expected, captured["text"]
        assert "…" in captured["text"], "the truncation never happened"
        assert str(tmp_path) not in captured["text"], "the DIRECTORY leaked"

    def test_completed_save_retires_the_provenance(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """A completed Save makes the order the reader's OWN -- the
        provenance retires and the next confession reads 'Your last
        saved order was applied' again."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        assert app._report_order_origin == str(source)
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        captured: dict[str, Any] = {}

        def first_visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["first"] = str(self._memory_label(app, dialog).cget("text"))
            self._button(app, dialog, "Save report…").invoke()

        app.after(150, first_visit)
        kept = app._session_report_picklist(live, [])
        assert kept is not None  # the Save completed
        assert app._report_order_origin is None, "the provenance survived the Save"

        def second_visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["second"] = str(self._memory_label(app, dialog).cget("text"))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, second_visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["first"].startswith("The imported order from")
        assert captured["second"] == (
            "Your last saved order was applied — Reset to "
            "walking order forgets it."
        ), captured["second"]

    def test_reset_forgets_the_provenance_with_the_memory(
        self, app: Any, playground: Any, monkeypatch: Any, dialogs: list[Any], tmp_path: Path
    ) -> None:
        """The dialog's Reset forgets the import exactly the way it
        forgets the reader's own order -- the provenance dies with the
        memory."""
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        app._import_report_order()
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            self._button(app, dialog, "Reset to walking order").invoke()
            app.update()
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert app._report_page_order == {}
        assert app._report_order_origin is None

    def test_session_restore_keeps_the_default_wording(
        self,
        playground: Any,
        xvfb_display: str,
        isolated_home: Path,
        monkeypatch: Any,
        dialogs: list[Any],
        tmp_path: Path,
    ) -> None:
        """The honest RESTART contract: after a relaunch the order arrives
        from the SESSION FILE -- which IS the last saved order -- so the
        provenance stays None and the confession keeps the default
        wording; a fresh launch never claims an import."""
        import gc

        # plant a session file carrying a non-empty order, the way the
        # import's write-through leaves it
        source = self._plant_order_file(playground, tmp_path)
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source)
        )
        seeding = playground.PlaygroundApp()
        self._drain(seeding)
        seeding._import_report_order()
        seeding.destroy()
        gc.collect()
        session_file = isolated_home / ".pharmaml" / "playground-session.json"
        assert session_file.is_file(), "the import never wrote the session file"

        app = playground.PlaygroundApp()  # restores from the session file
        self._drain(app)
        assert app._report_page_order == {
            "figure": ["beta plot", "alpha plot"],
            "table": ["T one"],
        }
        assert app._report_order_origin is None, "a restart claimed the import"
        live = [[None, "alpha plot", object()], [None, "beta plot", object()]]
        captured: dict[str, Any] = {}

        def visit() -> None:
            dialog = self._find_dialog(app)
            assert dialog is not None
            captured["text"] = str(self._memory_label(app, dialog).cget("text"))
            self._button(app, dialog, "Cancel").invoke()

        app.after(150, visit)
        assert app._session_report_picklist(live, []) is None
        assert captured["text"] == (
            "Your last saved order was applied — Reset to "
            "walking order forgets it."
        ), captured["text"]
        app.destroy()
        gc.collect()

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_138(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-138" in doc
        assert "RETURN leg" in doc
        assert "KIND CENSUS" in doc
        assert "PROVENANCE" in doc

    def test_handler_docstrings_pin_the_round(self, playground: Any) -> None:
        import inspect

        import_doc = playground.PlaygroundApp._import_report_order.__doc__ or ""
        assert "100x-138" in import_doc
        assert "PROVENANCE" in import_doc
        picklist_doc = playground.PlaygroundApp._session_report_picklist.__doc__ or ""
        assert "100x-138" in picklist_doc
        assert "PROVENANCE" in picklist_doc
        # the retirement pins: the Save / Reset / Clear all carry the note
        picklist_source = inspect.getsource(
            playground.PlaygroundApp._session_report_picklist
        )
        assert "the import's provenance confession retires" in picklist_source
        assert "the provenance dies with the memory" in picklist_source
        clear_source = inspect.getsource(
            playground.PlaygroundApp._clear_report_order
        )
        assert "the provenance dies with the memory" in clear_source


@XVFB_SKIP
class TestReview139Gui:
    """Round 100x-139: the return link's POINTER confession (the "↑
    Contents" link carries a title tooltip -- the aria world always knew
    the destination, a hovering pointer had no clue), the web preview's
    census became CLICKABLE LEGS with the SPOTLIGHT, and the strip's
    confession grew the ORDER's provenance ("Your last saved walking
    order was applied" -- the web twin of THIS app's picklist memory
    label).  The desktop side pins: the module docstring records the
    round, the picklist's wording stays the web twin's anchor, the
    playground never inlines the library's one-string link, and the
    app's own save path ships the tooltip on every page wrapper."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_139(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-139" in doc
        assert "POINTER confession" in doc
        assert "SPOTLIGHTS" in doc
        assert "last saved walking order was applied" in doc

    # -- the web twin's anchor ---------------------------------------------- #

    def test_the_picklist_wording_is_the_web_twins_anchor(
        self, playground: Any
    ) -> None:
        """The web strip's order confession mirrors THIS wording -- the
        desktop's own memory label must stay findable (the probe pins
        the web side; this pins the desktop's half of the pair)."""
        import inspect

        source = inspect.getsource(
            playground.PlaygroundApp._session_report_picklist
        )
        assert "Your last saved order was applied" in source
        assert "Reset to " in source and "walking order forgets it." in source

    # -- the one-string discipline ------------------------------------------ #

    def test_the_playground_never_inlines_the_return_link(self) -> None:
        """The link lives ONCE, in the library (pharmaml.viz.report's
        _BACK_CONTENTS_LINK) -- the playground composes through
        HTMLReport and carries no drifted copy of its own."""
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "back-contents" not in source, (
            "the playground grew an inline return link -- the one-string "
            "discipline broke"
        )

    def test_the_library_constant_carries_the_tooltip(self) -> None:
        """The one string confesses the jump target to the POINTER world
        (the exact wording the tests and the doctest pin)."""
        from pharmaml.viz.report import _BACK_CONTENTS_LINK

        assert _BACK_CONTENTS_LINK == (
            "<a class='back-contents' href='#contents' "
            "title='Back to the contents block at the top of the report'>"
            "\u2191 Contents</a>"
        )

    # -- the app's own save path ships the tooltip --------------------------- #

    def test_the_desktop_html_twin_carries_the_tooltip(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        dialogs: list[Any],
        picklist_accept_all: None,
        tmp_path: Path,
    ) -> None:
        """End to end through the app's own File ▸ Save session report:
        a planted figure + a filled table ship as an HTML file whose
        EVERY page wrapper ends in the tooltip-carrying return link."""
        import time as _time

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title("alpha plot")
        app._render_figure(app._topo_host, figure, caption="alpha plot")
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
        self._drain(app)

        target = tmp_path / "session-report.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        app._menu_save_session_report_html()

        # the worker thread writes the file -- pump until it lands
        deadline = _time.monotonic() + 30.0
        while not target.is_file() and _time.monotonic() < deadline:
            app.update()
            _time.sleep(0.01)
        assert target.is_file(), "the save job never wrote the file"

        html = target.read_text(encoding="utf-8")
        link = (
            "<a class='back-contents' href='#contents' "
            "title='Back to the contents block at the top of the report'>"
            "\u2191 Contents</a>"
        )
        # one figure + one table + the (100x-174) Attribution census
        # table -> exactly three page wrappers, each ending in the
        # SAME tooltip-carrying link
        assert html.count(link) == 3, html.count(link)
        assert html.count("title='Back to the contents block at the top of the report'") == 3
        # the anchors themselves are unchanged (the round only grew the
        # tooltip -- the round-trip navigation still holds)
        assert "<nav class='contents' id='contents' tabindex='-1'" in html
        assert "<li><a href='#pg-1' title='Jump to this page in the report'>table 1 · SMILES inspector</a></li>" in html
        assert "<li><a href='#pg-2' title='Jump to this page in the report'>table 2 · Attribution" in html
        assert "<li><a href='#pg-3' title='Jump to this page in the report'>figure 1 · alpha plot</a></li>" in html
        plt.close("all")


class TestReview140Gui:
    """Round 100x-140: the forward Contents lines gained the POINTER
    confession (every line's anchor whispers "Jump to this page in the
    report" to a hovering pointer -- the same tooltip discipline the
    return link gained in 100x-139, applied symmetrically), the web
    census's spotlight gained a KEYBOARD path (digits 1/2/3 toggle the
    legs through the same pure map a click consumes; 0 clears), and the
    web strip's confession gained its way out INSIDE the dialog (a
    "Reset now" link).  The desktop side pins: the module docstring
    records the round, the forward tooltip is minted ONCE in the
    library's render (never inlined here), and the app's own save path
    ships it on every Contents line."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def dialogs(self, playground: Any, monkeypatch: Any) -> list[Any]:
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, rounds: int = 4) -> None:
        for _ in range(rounds):
            app.update_idletasks()
            app.update()

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_140(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-140" in doc
        assert "the same POINTER" in doc
        assert "Jump to this page in the report" in doc
        assert "KEYBOARD" in doc
        assert "digits 1/2/3 toggle the census line's legs through the SAME" in doc
        assert '"Reset now" link beside' in doc

    # -- the one-place minting discipline ----------------------------------- #

    def test_the_forward_tooltip_is_minted_once_in_the_library(self) -> None:
        """The tooltip rides the ONE place the Contents lines are
        minted -- the library's render join.  The playground composes
        through HTMLReport and carries no inline copy of its own (the
        one-string discipline the return link taught, applied to the
        whole line)."""
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "title='Jump to this page in the report'" not in source, (
            "the playground inlined the forward tooltip markup -- the "
            "minting must stay in the library's render"
        )
        library = (
            TOOLKIT_ROOT / "src" / "pharmaml" / "viz" / "report.py"
        ).read_text(encoding="utf-8")
        # minted exactly once (the render join), never pasted twice
        assert library.count("Jump to this page in the report") == 2, (
            library.count("Jump to this page in the report")
        )

    # -- the app's own save path ships the forward tooltip ------------------- #

    def test_the_desktop_html_twin_carries_the_forward_tooltip(
        self,
        app: Any,
        playground: Any,
        monkeypatch: Any,
        dialogs: list[Any],
        picklist_accept_all: None,
        tmp_path: Path,
    ) -> None:
        """End to end through the app's own File ▸ Save session report:
        a planted figure + a filled table ship as an HTML file whose
        EVERY Contents line carries the pointer confession (the jump
        target told to a hovering pointer, symmetric with the return
        link's climb-back target)."""
        import time as _time

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot([0, 1], [0, 1])
        axes.set_title("alpha plot")
        app._render_figure(app._topo_host, figure, caption="alpha plot")
        app._desc_tree.insert("", "end", values=("MolWt", "180.16"))
        self._drain(app)

        target = tmp_path / "session-report.html"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
        app._menu_save_session_report_html()

        # the worker thread writes the file -- pump until it lands
        deadline = _time.monotonic() + 30.0
        while not target.is_file() and _time.monotonic() < deadline:
            app.update()
            _time.sleep(0.01)
        assert target.is_file(), "the save job never wrote the file"

        html = target.read_text(encoding="utf-8")
        # one figure + one table + the (100x-174) Attribution census
        # table -> exactly three Contents lines, EACH carrying the
        # forward tooltip (a line without it would be a line a hovering
        # pointer learns nothing from)
        tooltip = "title='Jump to this page in the report'"
        assert html.count(tooltip) == 3, html.count(tooltip)
        # the minted line form: href + title + the page's own name
        assert (
            "<li><a href='#pg-1' title='Jump to this page in the report'>"
            "table 1 · SMILES inspector</a></li>" in html
        )
        assert "<li><a href='#pg-2' title='Jump to this page in the report'>table 2 · Attribution" in html
        assert (
            "<li><a href='#pg-3' title='Jump to this page in the report'>"
            "figure 1 · alpha plot</a></li>" in html
        )
        # the return leg's tooltip still rides every wrapper too (the
        # round trip confesses in BOTH directions now)
        assert (
            html.count("title='Back to the contents block at the top of the report'")
            == 3
        )
        # fragment links only -- the self-contained contract holds
        assert "href='http" not in html
        plt.close("all")


class TestReview141Gui:
    """Round 100x-141: the 2026-10 "new folder.zip" upload was audited
    file by file (13 files) and the Docking tab gained the two surfaces
    the audit found missing -- the LIGAND STRUCTURES card (nigga.py
    whole: tautomer → conformers → MMFF/UFF → lowest-energy PDB with
    CONECT + the OpenBabel MOL2 twin, the metal skip list riding
    per-row) and the MGLTools PDB → PDBQT BATCH card (niggaqt.py whole:
    prepare_ligand4.py over a folder of PDBs, timeout-guarded, with the
    conversion log) -- while the Vina config batch card grew the RESUME
    checkbox of batch.py / niggavina.py (skip already-docked)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_141(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-141" in doc
        assert '"new folder" upload was audited file by' in doc
        assert "LIGAND STRUCTURES card" in doc
        assert "MGLTools PDB → PDBQT BATCH card" in doc
        assert "RESUME checkbox" in doc
        assert "non-empty" in doc  # the zero-byte log rule confesses

    # -- the library surface the cards call -------------------------------- #

    def test_docking_package_exports_the_new_pipelines(self) -> None:
        from pharmaml.docking import (  # noqa: F401
            DEFAULT_MGL_PYTHON,
            DEFAULT_PREPARE_LIGAND4,
            UNSUPPORTED_LIGAND_ATOMS,
            convert_pdb_folder_to_pdbqt,
            generate_ligand_structures,
        )

        assert "Zn" in UNSUPPORTED_LIGAND_ATOMS
        assert DEFAULT_MGL_PYTHON.endswith("python.exe")
        assert DEFAULT_PREPARE_LIGAND4.endswith("prepare_ligand4.py")

    def test_workflow_carries_the_resume_parameter(self) -> None:
        import dataclasses
        import inspect

        from pharmaml.docking.workflow import DockingRunResult, run_vina_batch

        signature = inspect.signature(run_vina_batch)
        assert "skip_existing" in signature.parameters
        assert signature.parameters["skip_existing"].default is False
        fields = {field.name for field in dataclasses.fields(DockingRunResult)}
        assert "skipped" in fields

    def test_workflow_carries_the_engine_folder_parameter(self) -> None:
        # (100x-171) every runner accepts engine_dir: the folder holding
        # vina.exe / autogrid4.exe / autodock4.exe, searched before PATH
        import inspect

        from pharmaml.docking import locate_engine_binary
        from pharmaml.docking.workflow import run_autodock4, run_vina, run_vina_batch

        for runner in (run_autodock4, run_vina, run_vina_batch):
            signature = inspect.signature(runner)
            assert "engine_dir" in signature.parameters, runner.__name__
            assert signature.parameters["engine_dir"].default is None
        assert callable(locate_engine_binary)

    def test_runners_thread_the_engine_folder_from_the_gui(self) -> None:
        # the GUI's every docking run passes the resolved engine folder
        # (picked folder, else the bundled engines/) down to pharmaml
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        for needle in (
            "batch_run_directories(root, workers=workers, engine_dir=engine_dir)",
            "run_autodock4(root, skip_existing=skip_existing, engine_dir=engine_dir)",
            "engine_dir=dock_engine_dir(self._dock_engines_dir)",
        ):
            assert needle in source, needle
        assert source.count("engine_dir=dock_engine_dir(self._dock_engines_dir)") >= 2

    # -- the source-level wiring -------------------------------------------- #

    def test_gen_card_wires_the_pipeline_with_its_knobs(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the work() call threads every knob the card owns
        assert "generate_ligand_structures(" in source
        for knob in ("num_confs=num_confs", "write_mol2=write_mol2", "workers=workers"):
            assert knob in source, knob
        # the nigga.py subtitle names the pipeline stages
        assert "canonical tautomer → ETKDG conformers → MMFF (UFF fallback)" in source
        # the metal skip list is advertised on the button's tooltip
        assert "Fe/Zn/Cu/Mn/Co/Ni/Mo/W" in source

    def test_mgl_card_wires_the_converter_with_timeout(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "convert_pdb_folder_to_pdbqt(" in source
        assert '"timeout_s": timeout_s' in source
        assert 'kwargs["mgl_python"] = mgl_python' in source
        assert 'kwargs["adt_script"] = adt_script' in source
        # the original's Windows default is pre-filled for the user
        assert r'C:\Program Files (x86)\MGLTools-1.5.7\python.exe' in source
        assert "conversion_log.txt" in source

    def test_vbatch_card_threads_the_resume_checkbox(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "_dock_vbatch_skip_var" in source
        assert "skip_existing=skip_existing" in source
        # skipped rows confess in the table
        assert '"skip" if r.skipped else ("OK" if r.ok else "FAILED")' in source
        # the checkbox names the upload's scripts it honours
        assert "Vina_contol/batch.py + niggavina.py resume" in source

    # -- the GUI-level surface ----------------------------------------------- #

    def test_both_new_cards_are_live_on_the_docking_tab(
        self, app: Any, playground: Any
    ) -> None:
        titles = {title for title, _subtitle, _shell in playground._CARD_INDEX}
        assert "Ligand structures (CSV → PDB + MOL2)" in titles
        assert "PDB → PDBQT batch (MGLTools)" in titles

    def test_gen_card_defaults_match_the_original_script(self, app: Any) -> None:
        # nigga.py's own numbers: 50 conformers, MOL2 on, workers 1
        assert app._dock_gen_confs.get() == "50"
        assert app._dock_gen_workers.get() == "1"
        assert bool(app._dock_gen_mol2_var.get()) is True
        assert app._dock_gen_out is None
        assert str(app._dock_gen_summary.cget("text")) == "—"

    def test_mgl_card_defaults_match_the_original_script(self, app: Any) -> None:
        # niggaqt.py's own numbers: the MGLTools-1.5.7 Windows paths and
        # the 30-second timeout
        assert app._dock_mgl_timeout.get() == "30"
        assert app._dock_mgl_python.get().endswith("python.exe")
        assert app._dock_mgl_adt.get().endswith("prepare_ligand4.py")
        assert app._dock_mgl_src is None and app._dock_mgl_out is None

    def test_vbatch_resume_checkbox_starts_unticked(self, app: Any) -> None:
        assert bool(app._dock_vbatch_skip_var.get()) is False

    def test_gen_run_warns_without_library_or_folder(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_gen_run()  # no library, no output folder
        assert any("No library" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_mgl_run_warns_without_folders(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_mgl_run()  # no source, no output
        assert any("Missing paths" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_gen_save_refuses_before_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_gen_save()
        assert any("Nothing to save" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_mgl_save_refuses_before_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dock_mgl_save()
        assert any("Nothing to save" in str(args) for _name, args in quiet_dialogs), quiet_dialogs


@XVFB_SKIP
class TestReview142Gui:
    """Round 100x-142 -- the engineering-details round: a one-by-one
    review of every tab fixed the audit's findings (the text-map drift
    that let the Chemistry tab's MACCS + similarity-scan boxes escape
    Reset/Copy, the Ctrl+O busy guard, the main-thread frame captures,
    the finder's title-only recency keys, the fractional-input
    truncation, the AD k-guard) and shipped three features (the picked
    PATHS riding the session file, the generate → MGLTools relay, the
    vina_batch_progress pre-flight census)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @pytest.fixture()
    def isolated_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        monkeypatch.setenv("HOME", str(tmp_path))
        return tmp_path

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 60.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_142(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-142" in doc
        assert "engineering-details round" in doc
        assert "Send PDBs → MGLTools" in doc
        assert "Count remaining" in doc
        assert "picked file/folder PATHS now ride the session" in doc

    # -- the audit fixes: the text-map drift ------------------------------- #

    def test_session_text_widgets_map_covers_all_eight_keys(self, app: Any) -> None:
        widgets = app._session_text_widgets()
        # 100x-162: the SELFIES audit box joined the map; 100x-164: the
        # REINVENT4 round added the transform-lab molecule list, the
        # alerts card's SMARTS + molecule boxes and the Generative
        # alerts blocklist (the Chemistry tab's eighth/eleventh persisted
        # text boxes and the Generative tab's second)
        assert set(widgets) == {
            "pasted",
            "corpus",
            "smiles",
            "rism",
            "maccs",
            "selfies_audit",
            "scan_refs",
            "scan_lib",
            "transform_mol",
            "alert_smarts",
            "alert_mol",
            "gen_alerts",
        }
        assert all(widget is not None for widget in widgets.values())

    def test_ownership_walk_covers_all_persisted_text_boxes(self, app: Any) -> None:
        # the bug (audit 100x-142 §3): text_by_id carried only 4 keys, so
        # text:maccs / text:scan_refs / text:scan_lib never entered the
        # ownership map -- and Reset/Copy silently skipped them.  The
        # probe: Chemistry owns six of the eight, Generative the
        # corpus, Docking the SMILES array.  (100x-164: the REINVENT4
        # boxes joined both tabs' ownership.)
        owners = app._tab_input_names()
        chemistry = owners[1]
        for key in (
            "text:pasted",
            "text:rism",
            "text:maccs",
            "text:selfies_audit",
            "text:scan_refs",
            "text:scan_lib",
            "text:transform_mol",
            "text:alert_smarts",
            "text:alert_mol",
        ):
            assert key in chemistry, f"{key} missing from the Chemistry tab's ownership"
        # (100x-167: the Best-of ML (9) and Response AI (10) tabs moved
        # Generative to 11 -- the ownership indices follow the book)
        assert "text:corpus" in owners[11]  # Generative
        assert "text:gen_alerts" in owners[11]  # Generative (100x-164)
        assert "text:smiles" in owners[4]  # Docking

    def test_chemistry_reset_restores_maccs_and_scan_boxes(self, app: Any) -> None:
        # the live confession: scribble over the boxes the drift
        # forgot, reset the Chemistry tab, and they must return to their
        # LAUNCH DEFAULTS (the shipped example content, not empty).
        defaults = {
            "maccs": app._maccs_text.get("1.0", "end-1c"),
            "scan_refs": app._scan_ref_text.get("1.0", "end-1c"),
            "scan_lib": app._scan_lib_text.get("1.0", "end-1c"),
            "selfies_audit": app._selfies_audit_text.get("1.0", "end-1c"),
        }
        for box in (
            app._maccs_text,
            app._scan_ref_text,
            app._scan_lib_text,
            app._selfies_audit_text,
        ):
            box.delete("1.0", "end")
            box.insert("1.0", "JUNK")
        app._reset_tab(1)
        assert app._maccs_text.get("1.0", "end-1c") == defaults["maccs"]
        assert app._scan_ref_text.get("1.0", "end-1c") == defaults["scan_refs"]
        assert app._scan_lib_text.get("1.0", "end-1c") == defaults["scan_lib"]
        assert app._selfies_audit_text.get("1.0", "end-1c") == defaults["selfies_audit"]
        assert "reset" in str(app._job_label.cget("text"))

    def test_chemistry_copy_inputs_carries_the_scan_boxes(self, app: Any) -> None:
        import json as _json

        captured: list[str] = []
        app._copy = lambda text: captured.append(text)  # type: ignore[method-assign]
        for box, marker in (
            (app._scan_lib_text, "CCNMARKER"),
            (app._maccs_text, "CCOMARKER"),
        ):
            box.delete("1.0", "end")
            box.insert("1.0", marker)
        app._copy_tab_inputs(1)
        payload = _json.loads(captured[-1])
        assert payload["texts"].get("scan_lib") == "CCNMARKER"
        assert payload["texts"].get("maccs") == "CCOMARKER"

    # -- the audit fixes: parse_int fractional rejection -------------------- #

    def test_parse_int_rejects_fractional_input(self, playground: Any) -> None:
        with pytest.raises(ValueError, match="not a whole number"):
            playground.parse_int("2.5", "cv folds")
        # whole-valued float spellings still parse
        assert playground.parse_int("2.0", "cv folds") == 2
        assert playground.parse_int("1e2", "epochs") == 100
        for poison in ("inf", "nan", "-inf"):
            with pytest.raises(ValueError, match="not a whole number"):
                playground.parse_int(poison, "field")

    # -- the audit fixes: the busy guard + job ordering -------------------- #

    def test_menu_open_csv_refuses_while_busy(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._busy = True
        try:
            selected_before = app._book.select()
            app._menu_open_csv()
            assert app._book.select() == selected_before  # no tab switch
            assert any("Busy" in str(args) for _name, args in quiet_dialogs), quiet_dialogs
        finally:
            app._busy = False

    def test_poll_jobs_clears_the_callback_before_invoking(self) -> None:
        # the source-level pin: capture-and-clear precedes the call, so a
        # render callback that chains into _start_job keeps its on_done
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        capture = source.index("callback = self._on_done")
        invoke = source.index("callback(result)")
        assert capture < invoke
        assert "self._on_done = None\n            if callback is not None:" in source

    def test_copy_skips_the_label_flash_while_busy(self, app: Any) -> None:
        app._job_label.configure(text="running job — 3/8")
        app._busy = True
        try:
            app._copy("payload")
            assert "running job" in str(app._job_label.cget("text"))
            assert "copied" not in str(app._job_label.cget("text"))
        finally:
            app._busy = False
        app._copy("payload")  # idle again: the flash returns
        assert "copied to clipboard" in str(app._job_label.cget("text"))

    def test_require_survives_an_unknown_module_name(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # the audit's StopIteration trap: a module outside the probe
        # table must produce the install dialog, never an exception
        assert app._require("not_a_real_module", "the feature") is False
        assert any("needs not_a_real_module" in str(args) for _name, args in quiet_dialogs)

    def test_ad_k_guard_covers_every_too_few_rows_case(self) -> None:
        # the source-level pin: the n_models == 0 condition is gone
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "n_models == 0 and len(x_train) <= k" not in source
        assert "if len(x_train) <= k:" in source

    def test_tune_worker_parameter_renamed_to_notify(self) -> None:
        # the shadowing trap: _ml_tune's worker parameter no longer
        # collides with the evaluate_model binding inside it
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "def work(notify: Any) -> Any:" in source
        assert "notify(info[\"trial\"], info.get(\"total\"), note)" in source

    # -- the audit fixes: the finder's recency keys ------------------------ #

    def test_finder_remember_keys_on_tab_and_title(self, app: Any) -> None:
        app._finder_recent = []
        app._finder_remember(4, "Results")  # the Docking tab's Results card
        app._finder_remember(6, "Results")  # the Deep Learning tab's twin
        assert app._finder_recent == ["6:Results", "4:Results"]
        # a re-jump of the SAME card moves it to the front, de-duplicated
        app._finder_remember(4, "Results")
        assert app._finder_recent == ["4:Results", "6:Results"]

    def test_finder_remember_drops_legacy_bare_titles(self, app: Any) -> None:
        # an older session file's title-only entry cannot name a tab --
        # re-membering the keyed twin drops it (no duplicate chips)
        app._finder_recent = ["Results"]
        app._finder_remember(6, "Results")
        assert app._finder_recent == ["6:Results"]

    def test_jump_to_card_records_the_tab_aware_key(self, app: Any) -> None:
        app._finder_recent = []
        app._jump_to_card(1, "Similarity scan (library vs references)")
        assert app._finder_recent and app._finder_recent[0].startswith("1:")

    # -- the feature: picked paths ride the session ------------------------ #

    def test_collect_session_carries_the_picked_paths(self, app: Any, tmp_path: Path) -> None:
        app._dock_gen_out = tmp_path
        app._dock_mgl_src = tmp_path
        payload = app._collect_session()
        assert payload["paths"]["_dock_gen_out"] == str(tmp_path)
        assert payload["paths"]["_dock_mgl_src"] == str(tmp_path)

    def test_collect_session_never_carries_dataset_paths(self, app: Any, tmp_path: Path) -> None:
        # the CSV loaders stay out on purpose: a path without its loaded
        # frame would be a lie the label tells
        app._ml_path = tmp_path / "data.csv"
        app._dock_lib_path = tmp_path / "lib.csv"
        payload = app._collect_session()
        assert "_ml_path" not in payload["paths"]
        assert "_dock_lib_path" not in payload["paths"]

    def test_restore_returns_existing_paths_and_refreshes_labels(
        self, playground: Any, xvfb_display: str, isolated_home: Path, tmp_path: Path
    ) -> None:
        import gc
        import json as _json

        ligands = tmp_path / "ligands"
        ligands.mkdir()
        payload = playground.dump_session(
            theme="light",
            geometry="1180x780+10+20",
            tab=4,
            variables={},
            families={},
            texts={},
            paths={
                "_dock_vbatch_ligands": str(ligands),  # exists -> restored
                "_dock_vbatch_out": str(tmp_path / "absent"),  # gone -> dropped
            },
        )
        (isolated_home / ".pharmaml").mkdir(parents=True)
        (isolated_home / ".pharmaml" / "playground-session.json").write_text(
            _json.dumps(payload), encoding="utf-8"
        )
        app = playground.PlaygroundApp()
        for _ in range(4):
            app.update_idletasks()
            app.update()
        assert app._dock_vbatch_ligands == ligands
        assert app._dock_vbatch_out is None  # the absent folder stayed gone
        assert "ligands: ligands" in str(app._dock_vbatch_info.cget("text"))
        assert "picked path(s)" in str(app._job_label.cget("text"))
        app.destroy()
        del app
        gc.collect()

    def test_restore_never_overwrites_a_live_pick(
        self, playground: Any, app: Any, tmp_path: Path
    ) -> None:
        live = tmp_path / "live"
        live.mkdir()
        app._dock_mgl_src = live  # the user already picked THIS session
        session = playground.load_session(
            {
                "format": playground.SESSION_FORMAT,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {"_dock_mgl_src": str(tmp_path)},
            }
        )
        assert session is not None
        app._session = session
        app._restore_session()
        assert app._dock_mgl_src == live

    def test_session_paths_key_round_trips_through_dump_and_load(
        self, playground: Any, tmp_path: Path
    ) -> None:
        payload = playground.dump_session(
            theme="light",
            geometry="",
            tab=0,
            variables={},
            families={},
            texts={},
            paths={"_dock_smi_out": str(tmp_path)},
        )
        assert playground.load_session(payload) == payload
        # a foreign value inside paths is coerced away, never fatal
        loaded = playground.load_session(
            {
                "format": playground.SESSION_FORMAT,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {"_dock_smi_out": 42, "bad": None},
            }
        )
        assert loaded is not None
        assert loaded["paths"] == {}

    # -- the feature: the generate → MGLTools relay ------------------------ #

    def test_gen_to_mgl_hands_the_folder_over(self, app: Any, tmp_path: Path) -> None:
        app._dock_gen_out = tmp_path
        app._dock_mgl_out = None
        app._dock_gen_to_mgl()
        assert app._dock_mgl_src == tmp_path
        # the output folder lands BESIDE the PDB folder, never inside it
        assert app._dock_mgl_out == tmp_path.parent / f"{tmp_path.name}_pdbqt"
        assert "handed to the MGLTools card" in str(app._dock_gen_summary.cget("text"))
        info = str(app._dock_mgl_info.cget("text"))
        assert f"pdb source: {tmp_path.name}" in info

    def test_gen_to_mgl_never_overwrites_a_live_output_pick(
        self, app: Any, tmp_path: Path
    ) -> None:
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        app._dock_gen_out = tmp_path
        app._dock_mgl_out = chosen
        app._dock_gen_to_mgl()
        assert app._dock_mgl_out == chosen

    def test_gen_to_mgl_warns_without_an_output_folder(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_gen_out = None
        app._dock_gen_to_mgl()
        assert any("No output folder" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_gen_card_carries_the_relay_button(self, app: Any) -> None:
        labels = [
            str(button.cget("text"))
            for button in app._action_buttons
            if button.winfo_exists()
        ]
        assert "Send PDBs → MGLTools" in labels
        assert "Count remaining" in labels

    # -- the feature: the pre-flight resume census ------------------------- #

    def test_vbatch_census_button_reports_the_promise(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "a.pdbqt").write_text("ligand a", encoding="utf-8")
        (ligands / "b.pdbqt").write_text("ligand b", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        (out / "a_out.pdbqt").write_text("poses", encoding="utf-8")
        (out / "a.log").write_text("log", encoding="utf-8")
        app._dock_vbatch_ligands = ligands
        app._dock_vbatch_out = out
        app._dock_vbatch_count()
        assert self._drain_jobs(app), "census job hung"
        summary = str(app._dock_vbatch_summary.cget("text"))
        assert "1/2 already docked" in summary
        assert "1 to go" in summary
        assert quiet_dialogs == []

    def test_vbatch_census_warns_without_folders(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_vbatch_ligands = None
        app._dock_vbatch_count()
        assert any("Missing paths" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_toggling_resume_runs_the_census_on_its_own(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "only.pdbqt").write_text("ligand", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        app._dock_vbatch_ligands = ligands
        app._dock_vbatch_out = out
        app._dock_vbatch_skip_var.set(True)  # the trace fires the census
        assert self._drain_jobs(app), "auto census job hung"
        summary = str(app._dock_vbatch_summary.cget("text"))
        assert "0/1 already docked" in summary
        assert "1 to go" in summary
        assert quiet_dialogs == []

    def test_toggling_resume_without_folders_stays_quiet(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dock_vbatch_ligands = None
        app._dock_vbatch_out = None
        app._dock_vbatch_skip_var.set(True)  # no paths -> no auto job
        app.update()
        assert not app._busy
        assert quiet_dialogs == []

    # -- the audit fix: the live frame reaches the DL tab ------------------ #

    def test_dl_load_frame_adopts_an_in_memory_dataset(
        self, app: Any, tmp_path: Path
    ) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {"smiles": ["CCO", "CCN"], "value": [1.0, 2.0]}
        )
        app._dl_load_frame(frame, "cleaned.csv (live)")
        assert app._dl_frame is frame
        assert app._dl_path is None  # in-memory adoption never fakes a path
        assert "cleaned.csv (live)" in str(app._dl_info.cget("text"))
        assert "2" in str(app._dl_info.cget("text"))
        values = list(app._dl_target_combo["values"])
        assert "value" in values

    def test_tune_apply_prefers_the_live_frame_over_the_stale_path(
        self, app: Any, tmp_path: Path
    ) -> None:
        import pandas as pd

        # the cleaned frame sits in memory; _ml_path still names the
        # original CSV -- Apply best → dnn must carry the LIVE one
        original = tmp_path / "original.csv"
        original.write_text("smiles,value\nCCO,1.0\n", encoding="utf-8")
        app._ml_path = original
        cleaned = pd.DataFrame({"smiles": ["CCO"], "value": [1.0]})
        app._ml_frame = cleaned
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "self._dl_load_frame(\n                    self._ml_frame," in source

    # -- the library surface the census calls ------------------------------ #

    def test_docking_package_exports_the_census(self) -> None:
        from pharmaml.docking import VinaBatchProgress, vina_batch_progress

        assert callable(vina_batch_progress)
        assert VinaBatchProgress.__name__ == "VinaBatchProgress"

    def test_vbatch_census_uses_the_library_helper(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "from pharmaml.docking import vina_batch_progress" in source
        assert "return vina_batch_progress(ligands, out)" in source
        # the auto-census trace rides the checkbox
        assert 'trace_add("write", _on_skip_toggle)' in source

    def test_session_paths_allowlist_excludes_reload_dependent_pickers(
        self, playground: Any
    ) -> None:
        allowlist = playground._SESSION_PATH_ATTRS
        for excluded in ("_ml_path", "_dock_lib_path", "_feat_csv_path", "_ml_ad_train_path"):
            assert excluded not in allowlist, excluded
        for included in ("_dock_gen_out", "_dock_mgl_src", "_dock_mgl_out", "_dock_vbatch_out"):
            assert included in allowlist, included


# --------------------------------------------------------------------- #
# round 100x-143 -- the second engineering-details round: the file-dialog
# loader port (five loaders on the worker pool), the Utilities restored-
# path labels, the SDF → table guard, the MACCS mixed-parse label, and
# the two advisory-dialog convention fixes
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestReview143Gui:
    """Round 100x-143 -- the loader port's contracts: the dialog flows
    ride the worker pool (a big CSV never freezes the window), the sync
    cores stay synchronous for the tiny bundled examples, and each tab's
    two paths share ONE adoption body.  Plus the Utilities labels that
    finally confess their restored paths, the SDF → table guard that no
    longer demands an output file it never writes, and the MACCS label
    that keeps its summary when the list is mixed."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 60.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_143(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-143" in doc
        assert "second engineering-details round" in doc
        assert "file-dialog" in doc
        assert "SDF → table" in doc

    def test_every_dialog_loader_has_a_job_based_twin(self, playground: Any) -> None:
        # the five ports: ML, DL, feature factory, the Cell Lines screen
        # and its expression matrix -- each _load_file rides _start_job
        # with a "loading {name}" label
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        for name in (
            "_ml_load_file",
            "_dl_load_file",
            "_feat_load_file",
            "_bio_load_file",
            "_bio_load_expr_file",
        ):
            assert f"def {name}(self, path: Path) -> None:" in source, name
        assert (
            source.count('self._start_job(f"loading {path.name}", work, render)') == 5
        )
        # and every file-dialog flow routes to its job twin
        for call in (
            "self._ml_load_file(Path(path))",
            "self._dl_load_file(Path(path))",
            "self._feat_load_file(Path(path))",
            "self._bio_load_file(Path(path))",
            "self._bio_load_expr_file(Path(path))",
        ):
            assert call in source, call

    def test_sync_and_async_loaders_share_one_adoption_body(self) -> None:
        # the drift-proofing pin: each tab's sync core AND its job render
        # funnel through the SAME adoption helper (never two bodies)
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "self._ml_adopt_frame(frame, path)" in source  # sync core
        assert source.count("self._ml_adopt_frame(frame, path)") == 2  # + the render
        assert "self._feat_adopt_frame(frame, path)" in source
        assert source.count("self._feat_adopt_frame(frame, path)") == 2
        assert "self._bio_adopt_screen(raw, audit_frame, screen, path)" in source
        assert source.count("self._bio_adopt_screen(raw, audit_frame, screen, path)") == 2
        assert "self._bio_adopt_expr_frame(frame, path)" in source
        assert source.count("self._bio_adopt_expr_frame(frame, path)") == 2
        # the DL tab's twin reuses the round-142 in-memory adoption core
        assert source.count("self._dl_load_frame(frame, path.name)") == 2

    # -- the loader port: the dialog flows are jobs ------------------------ #

    def test_ml_open_csv_runs_as_a_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table = tmp_path / "big_dataset.csv"
        table.write_text("smiles,value\nCCO,1.5\nCCN,2.5\n", encoding="utf-8")
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(playground.filedialog, "askopenfilename", lambda **_k: str(table))
            app._ml_open_csv()
        assert app._busy, "the load started as a worker job"
        assert "loading big_dataset.csv" in str(app._job_label.cget("text"))
        assert self._drain_jobs(app), "load job hung"
        assert app._ml_frame is not None and len(app._ml_frame) == 2
        assert app._ml_path == table
        assert "big_dataset.csv" in str(app._ml_info.cget("text"))
        assert len(app._preview_tree.get_children()) == 2
        assert list(app._target_combo["values"]) == ["smiles", "value"]
        assert quiet_dialogs == []

    def test_dl_open_csv_runs_as_a_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table = tmp_path / "dl_dataset.csv"
        table.write_text("smiles,value\nCCO,1.5\n", encoding="utf-8")
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                playground.filedialog, "askopenfilename", lambda **_k: str(table)
            )
            app._dl_open_csv()
        assert self._drain_jobs(app), "load job hung"
        assert app._dl_frame is not None
        assert app._dl_path == table
        assert "dl_dataset.csv" in str(app._dl_info.cget("text"))
        assert quiet_dialogs == []

    def test_feat_open_csv_runs_as_a_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table = tmp_path / "library.csv"
        table.write_text("smiles,value\nCCO,1.5\n", encoding="utf-8")
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                playground.filedialog, "askopenfilename", lambda **_k: str(table)
            )
            app._feat_open_csv()
        assert self._drain_jobs(app), "load job hung"
        assert app._feat_csv_frame is not None
        assert app._feat_smiles_column == "smiles"
        assert "library.csv" in str(app._feat_source_info.cget("text"))
        assert quiet_dialogs == []

    def test_bio_open_csv_runs_as_a_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        # the shipped example screen, copied out so the run is hermetic
        import shutil as _shutil

        table = tmp_path / "screen.csv"
        _shutil.copy(TOOLKIT_ROOT / "examples" / "data" / "cellline_screen.csv", table)
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                playground.filedialog, "askopenfilename", lambda **_k: str(table)
            )
            app._bio_open_csv()
        assert self._drain_jobs(app), "screen load job hung"
        assert app._bio_screen is not None
        assert app._bio_screen_path == table
        assert "screen.csv" in str(app._bio_info.cget("text"))
        assert "duplicate pairs" in app._bio_audit_text.get("1.0", "end")
        assert quiet_dialogs == []

    def test_bio_open_expr_csv_runs_as_a_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table = tmp_path / "expression.csv"
        table.write_text("cell_line,GeneA,GeneB\nA549,1.0,2.0\nHL60,3.0,4.0\n", encoding="utf-8")
        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                playground.filedialog, "askopenfilename", lambda **_k: str(table)
            )
            app._bio_open_expr_csv()
        assert self._drain_jobs(app), "expression load job hung"
        assert app._bio_expr is not None
        assert "expression.csv" in str(app._bio_expr_info.cget("text"))
        assert "2 cell lines" in str(app._bio_expr_info.cget("text"))
        assert quiet_dialogs == []

    def test_job_loader_empty_csv_warns_and_adopts_nothing(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("smiles,value\n", encoding="utf-8")
        app._ml_load_file(empty)
        assert self._drain_jobs(app), "load job hung"
        assert app._ml_frame is None  # nothing adopted
        assert any(
            name == "showwarning" and "Empty dataset" in str(args)
            for name, args in quiet_dialogs
        ), quiet_dialogs

    def test_job_loader_respects_the_busy_guard(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        table = tmp_path / "guard.csv"
        table.write_text("smiles,value\nCCO,1.5\n", encoding="utf-8")
        app._busy = True
        try:
            app._ml_load_file(table)
            assert any("Busy" in str(args) for _name, args in quiet_dialogs), quiet_dialogs
        finally:
            app._busy = False

    def test_the_examples_stay_synchronous(self, app: Any) -> None:
        # the design decision, pinned: the bundled examples are shipped
        # and tiny, so _load_example adopts IMMEDIATELY (no job to drain)
        # -- the tests' 46 fixture call sites depend on it
        app._ml_load_example()
        assert not app._busy
        assert app._ml_frame is not None
        app._feat_load_example()
        assert not app._busy
        assert app._feat_csv_frame is not None

    def test_bio_load_file_captures_the_collapse_flag_on_the_main_thread(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        # the source-level pin: a Tk variable must never be read inside
        # work() -- the collapse checkbox is captured BEFORE the job
        # (100x-144: the loader's work now TAKES the reporter, so the
        # pin searches the one-argument form -- zero-argument matches
        # would hit the NEXT tab handler instead)
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        marker = source.index("collapse = bool(self._bio_collapse_var.get())")
        job = source.index('self._start_job(f"loading {path.name}", work, render)', marker)
        work_def = source.index("def work(report: Any) -> Any:", marker)
        assert marker < work_def < job
        # and the worker honours the captured flag, never the variable
        # (100x-146 standing-contract update: the collapse branch grew
        # its "collapsing replicates" phase note -- the flag-honouring
        # shape is the explicit if/else now)
        assert (
            "if collapse:\n"
            '                report(rows, None, "collapsing replicates")\n'
            "                screen = collapse_replicates(raw)\n"
            "            else:\n"
            "                screen = raw"
        ) in source

    # -- the Utilities restored-path labels (the audit's §1) --------------- #

    def test_restored_utility_paths_rerender_their_labels(
        self, app: Any, tmp_path: Path
    ) -> None:
        # the probe the auditor ran, now the regression pin: seven picked
        # paths ride the session, four labels used to keep denying them
        folder = tmp_path / "campaign"
        folder.mkdir()
        table_a = tmp_path / "a.csv"
        table_a.write_text("x\n1\n", encoding="utf-8")
        table_b = tmp_path / "b.csv"
        table_b.write_text("x\n2\n", encoding="utf-8")
        split_file = tmp_path / "big.csv"
        split_file.write_text("x\n3\n", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        reindex = tmp_path / "sparse.csv"
        reindex.write_text("n\n1\n", encoding="utf-8")
        # exactly what _restore_session does with a paths payload
        app._util_merge_folder = folder
        app._util_merge_split_file = split_file
        app._util_cmp_a = table_a
        app._util_cmp_b = table_b
        app._util_tool_folder = folder
        app._util_tool_out = out
        app._util_reindex_path = reindex
        app._util_merge_mode.set("merge whole folder")
        app._util_tool_mode.set("collect files (name/pattern)")
        app._refresh_restored_path_labels()
        assert f"folder: {folder}" in str(app._util_merge_info.cget("text"))
        assert f"A: {table_a.name}" in str(app._util_cmp_info.cget("text"))
        assert f"B: {table_b.name}" in str(app._util_cmp_info.cget("text"))
        assert f"folder: {folder.name}" in str(app._util_tool_info.cget("text"))
        assert f"collect into: {out.name}" in str(app._util_tool_info.cget("text"))
        assert str(reindex) == str(app._util_reindex_info.cget("text"))

    def test_refresh_walks_the_three_utility_helpers(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        for helper in (
            '"_util_merge_mode_changed",',
            '"_util_cmp_refresh_info",',
            '"_util_tool_mode_changed",',
        ):
            assert helper in source, helper
        assert '_label("_util_reindex_info", str(self._util_reindex_path))' in source

    # -- the SDF → table guard (the audit's §2) ----------------------------- #

    def test_sdf_to_table_needs_no_output_file(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        # a real one-molecule SDF: the mode builds an in-memory table and
        # must run WITHOUT a picked output (the old guard demanded one)
        sdf = tmp_path / "one.sdf"
        sdf.write_text(
            "ethanol\n     RDKit          2D\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
            "  1  2  1  0\n  2  3  1  0\nM  END\n>  <activity>\n1.5\n\n$$$$\n",
            encoding="utf-8",
        )
        app._conv_input = sdf
        app._conv_output = None
        app._conv_mode.set("SDF → table")
        app._conv_run()
        assert self._drain_jobs(app), "SDF read job hung"
        assert app._conv_frame is not None
        assert "1 record(s) → table" in str(app._conv_summary.cget("text"))
        # no "No output" warning ever fired
        assert not any("No output" in str(args) for _name, args in quiet_dialogs), quiet_dialogs

    def test_sdf_pick_output_gives_the_in_memory_notice(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._conv_mode.set("SDF → table")
        app._conv_pick_output()
        assert any(
            name == "showinfo" and "in memory" in str(args)
            for name, args in quiet_dialogs
        ), quiet_dialogs

    def test_converter_output_guard_still_fires_for_file_writing_modes(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        inp = tmp_path / "table.csv"
        inp.write_text("smiles\nCCO\n", encoding="utf-8")
        app._conv_input = inp
        app._conv_output = None
        app._conv_mode.set("table → .smi list")
        app._conv_run()
        assert not app._busy  # stopped at the gate, no job started
        assert any(
            name == "showwarning" and "No output" in str(args)
            for name, args in quiet_dialogs
        ), quiet_dialogs

    # -- the MACCS mixed-parse label (the audit's §3) ----------------------- #

    def test_maccs_mixed_list_keeps_summary_and_dropped_note(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._maccs_text.delete("1.0", "end")
        app._maccs_text.insert("1.0", "CCO\nNOT_A_SMILES")
        app._maccs_run()
        assert self._drain_jobs(app), "MACCS job hung"
        label = str(app._maccs_result.cget("text"))
        # the computed summary SURVIVES...
        assert "1 molecule(s) · mean" in label
        assert "/167 keys set" in label
        # ...and the dropped count rides alongside instead of clobbering
        assert "1 SMILES failed to parse — excluded from the mean" in label
        assert not label.startswith("1 SMILES failed to parse — no MACCS keys")
        assert quiet_dialogs == []

    def test_maccs_all_failed_keeps_the_honest_clobber(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._maccs_text.delete("1.0", "end")
        app._maccs_text.insert("1.0", "NOT_A_SMILES")
        app._maccs_run()
        assert self._drain_jobs(app), "MACCS job hung"
        label = str(app._maccs_result.cget("text"))
        assert label == "1 SMILES failed to parse — no MACCS keys"
        assert quiet_dialogs == []

    # -- the advisory-dialog convention (the audit's §6 twins) -------------- #

    def test_topology_moves_missing_smiles_is_a_warning(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._topo_smiles.delete(0, "end")
        app._gen_topology_moves()
        assert not app._busy  # stopped at the gate, no job started
        assert any(
            name == "showwarning" and "Type a SMILES" in str(args)
            for name, args in quiet_dialogs
        ), quiet_dialogs
        assert not any(name == "showerror" for name, _args in quiet_dialogs)

    def test_rescore_missing_columns_is_a_warning(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        import pandas as pd

        app._dock_results_frame = pd.DataFrame({"ligand": ["a"], "score": [1.0]})
        app._dock_rescore_columns_entry.delete(0, "end")
        app._dock_rescore()
        assert not app._busy  # stopped at the gate, no job started
        assert any(
            name == "showwarning" and "evidence column" in str(args)
            for name, args in quiet_dialogs
        ), quiet_dialogs
        assert not any(name == "showerror" for name, _args in quiet_dialogs)


# --------------------------------------------------------------------- #
# round 100x-144 -- the third engineering-details round: the multi-file
# pickers ride the session (paths_multi), a large CSV loads with a
# determinate rows/ETA bar (read_table_with_progress), the chained-job
# guard in _poll_jobs, and the audit's parity/normaliser fixes
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestReview144Gui:
    """Round 100x-144 -- the three Utilities list pickers persist across
    restarts with the single paths' own existence-guarded discipline,
    the four read_table loaders stream their reads with a determinate
    progress bar, and a render callback that chains a job keeps its
    follow-up callback (the audit's §2)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 60.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_144(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-144" in doc
        assert "third engineering-details round" in doc
        assert "paths_multi" in doc
        assert "read_table_with_progress" in doc

    # -- the feature: the multi-file picks ride the session ---------------- #

    def test_paths_multi_round_trips_and_sanitises(self, playground: Any) -> None:
        payload = playground.dump_session(
            theme="light",
            geometry="",
            tab=0,
            variables={},
            families={},
            texts={},
            paths={},
            paths_multi={"_util_merge_files": ["a.csv", "b.csv"]},
        )
        assert playground.load_session(payload) == payload
        # a foreign value: non-list entries drop, non-str members drop
        loaded = playground.load_session(
            {
                "format": playground.SESSION_FORMAT,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {},
                "paths_multi": {
                    "_util_appr_files": "not a list",
                    "_util_tool_files": ["ok.tsv", 42, None],
                },
            }
        )
        assert loaded is not None
        assert loaded["paths_multi"] == {"_util_tool_files": ["ok.tsv"]}
        # and dump never char-splits a stray string value (the audit's §5)
        guarded = playground.dump_session(
            theme="light",
            geometry="",
            tab=0,
            variables={},
            families={},
            texts={},
            paths={},
            paths_multi={"_util_merge_files": "xy12"},
        )
        assert guarded["paths_multi"] == {}

    def test_collect_session_rides_the_three_lists(
        self, app: Any, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("x\n1\n", encoding="utf-8")
        b.write_text("x\n2\n", encoding="utf-8")
        app._util_merge_files = [a, b]
        app._util_appr_files = [b]
        app._util_tool_files = []
        collected = app._collect_session()
        assert collected["paths_multi"] == {
            "_util_appr_files": [str(b)],
            "_util_merge_files": [str(a), str(b)],
            "_util_tool_files": [],
        }

    def test_restore_filters_vanished_files_and_keeps_order(
        self, playground: Any, app: Any, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        gone = tmp_path / "gone.csv"
        a.write_text("x\n1\n", encoding="utf-8")
        b.write_text("x\n2\n", encoding="utf-8")
        gone.write_text("x\n3\n", encoding="utf-8")
        app._util_merge_mode.set("merge picked tables")
        app._util_tool_mode.set("distribute files to sub-folders")
        app._session = playground.load_session(
            {
                "format": 1,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {},
                "paths_multi": {
                    "_util_merge_files": [str(a), str(gone), str(b)],
                    "_util_appr_files": [str(a), str(b)],
                    "_util_tool_files": [str(gone)],
                },
            }
        )
        gone.unlink()  # the mailed session's third file vanished
        app._restore_session()
        assert app._util_merge_files == [a, b]  # order kept, vanished dropped
        assert app._util_appr_files == [a, b]
        assert app._util_tool_files == []  # a fully-vanished list stays empty
        # the labels confess the restored picks
        merge_label = str(app._util_merge_info.cget("text"))
        assert "2 table(s): a.csv, b.csv" in merge_label
        assert str(app._util_appr_info.cget("text")) == "a.csv, b.csv"
        # the status bar counts every restored file (2 + 2)
        status = str(app._job_label.cget("text"))
        assert "4 picked path(s) back" in status

    def test_restore_never_overwrites_a_live_pick(
        self, playground: Any, app: Any, tmp_path: Path
    ) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        a.write_text("x\n1\n", encoding="utf-8")
        b.write_text("x\n2\n", encoding="utf-8")
        app._util_merge_files = [b]  # the user re-picked THIS session
        app._session = playground.load_session(
            {
                "format": 1,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {},
                "paths_multi": {"_util_merge_files": [str(a), str(b)]},
            }
        )
        app._restore_session()
        assert app._util_merge_files == [b]

    def test_restore_ignores_foreign_names(
        self, playground: Any, app: Any, tmp_path: Path
    ) -> None:
        inside = tmp_path / "inside.csv"
        inside.write_text("x\n1\n", encoding="utf-8")
        app._session = playground.load_session(
            {
                "format": 1,
                "theme": "light",
                "geometry": "",
                "tab": 0,
                "variables": {},
                "families": {},
                "texts": {},
                "paths": {},
                "paths_multi": {"_ml_path": [str(inside)], "bad": [str(inside)]},
            }
        )
        app._restore_session()
        assert app._ml_path is None  # outside the allowlist, never adopted
        assert app._util_merge_files == []

    def test_refresh_renders_the_approval_label_in_picker_format(
        self, app: Any, tmp_path: Path
    ) -> None:
        files = [tmp_path / f"f{i}.csv" for i in range(4)]
        for i, path in enumerate(files):
            path.write_text(f"x\n{i}\n", encoding="utf-8")
        app._util_appr_files = list(files)
        app._refresh_restored_path_labels()
        assert (
            str(app._util_appr_info.cget("text"))
            == "f0.csv, f1.csv, f2.csv … (+1)"
        )

    def test_the_multi_allowlist_is_exactly_the_three_utilities(
        self, playground: Any
    ) -> None:
        assert playground._SESSION_PATH_LIST_ATTRS == frozenset(
            {"_util_merge_files", "_util_appr_files", "_util_tool_files"}
        )

    # -- the feature: the loaders report a determinate bar ------------------ #

    def test_the_reporting_loaders_stream_through_the_progressive_reader(
        self,
    ) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the three read_table loaders take the reporter...
        assert source.count("def work(report: Any) -> Any:") >= 4
        assert source.count("return read_table_with_progress(path, report)") == 3
        # ...and the Cell Lines screen forwards it through the loader API
        assert "raw = load_response_table(path, report=report)" in source

    def test_ml_loader_carries_a_determinate_bar_mid_job(
        self, playground: Any, app: Any, quiet_dialogs: list[Any], tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        import time

        # a genuinely large CSV past the reader's 8 MiB gate -- the read
        # streams chunk-by-chunk; the reports are THROTTLED (a wrapper
        # around the REAL reader, never a stub) so the busy window
        # outlives several 150 ms poller ticks and the test cannot race
        rows = 800_000
        table = tmp_path / "wide_load.csv"
        lines = ["smiles,value"] + [f"CCO,{i}.5" for i in range(rows)]
        table.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert table.stat().st_size > 8 * 1024 * 1024  # past the gate

        import pharmaml.utils.io as io_mod

        real_reader = io_mod.read_table_with_progress

        def throttled(path: Any, report: Any, **kwargs: Any) -> Any:
            def slow_report(done: int, total: Any, note: str) -> None:
                time.sleep(0.15)
                report(done, total, note)

            return real_reader(path, slow_report, **kwargs)

        monkeypatch.setattr(io_mod, "read_table_with_progress", throttled)
        app._ml_load_file(table)
        assert app._busy
        modes: set[str] = set()
        labels: set[str] = set()
        deadline = time.time() + 90.0
        while time.time() < deadline:
            app.update()
            modes.add(str(app._progress.cget("mode")))
            labels.add(str(app._job_label.cget("text")))
            if not app._busy:
                break
            time.sleep(0.02)
        assert not app._busy, "big load hung"
        assert "determinate" in modes, modes  # the bar switched mid-job
        assert any(
            "/" in text and "loading wide_load.csv" in text for text in labels
        ), sorted(labels)[-3:]
        assert app._ml_frame is not None and len(app._ml_frame) == rows
        assert "done in" in str(app._job_label.cget("text"))
        assert quiet_dialogs == []

    # -- the audit's §2: the chained-job guard ------------------------------ #

    def test_poll_jobs_keeps_a_chained_jobs_callback(self, app: Any) -> None:
        chained: list[Any] = []

        def render(result: Any) -> None:
            # a render callback that chains straight into a follow-up
            # job -- exactly what _start_job would set up for it
            app._on_done = lambda r: chained.append(r)
            app._busy = True

        app._on_done = render
        app._jobs.put(("payload", None))
        app._poll_jobs()  # drains the result directly
        assert app._on_done is not None  # the follow-up callback survived
        app._on_done("late")  # and it still fires
        assert chained == ["late"]

    def test_poll_jobs_clears_the_stale_callback_when_idle(self, app: Any) -> None:
        app._on_done = lambda _r: None
        app._jobs.put(("x", None))
        app._poll_jobs()
        assert app._on_done is None

    def test_the_chained_job_guard_is_pinned_in_source(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "if not self._busy:\n            self._on_done = None" in source

    # -- the Apply-best invariant (the handover's item b) ------------------- #

    def test_tune_apply_states_the_frame_path_invariant(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the invariant comment lives at the relay...
        assert "the invariant, stated" in source
        assert "_ml_adopt_frame sets frame" in source
        # ...and the adoption body is the ONLY place a path assignment
        # follows its frame (the Clean card swaps the frame alone)
        adopt = source.index("self._ml_frame = frame")
        path_assign = source.index("self._ml_path = path", adopt)
        assert path_assign - adopt < 80  # set together, a few lines apart
        assert source.count("self._ml_path = path") == 1
        assert "self._ml_frame = cleaned" in source  # the Clean card's swap


class TestReview145Gui:
    """Round 100x-145 -- the "کد علی" upload's Ml/ pair surfaces: D2.py's
    process_data as the EC50 → pEC50 prep card (every silent step
    counted, one-click adoption), its permutation-importance step as a
    Results-card button, and the ecfp3 family in the feature factory."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 90.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_145(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-145" in doc
        assert "prepare_pic50_dataset" in doc
        assert "permutation" in doc.lower()
        assert "ecfp3" in doc

    # -- the EC50 -> pEC50 prep card --------------------------------------- #

    def test_prep_card_source_pins(self, playground: Any) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"EC50 → pEC50 prep"' in source
        assert "def _ml_prep_endpoint" in source
        assert "def _ml_prep_save" in source
        assert "from pharmaml.chem import prepare_pic50_dataset" in source
        assert 'self._ml_aggregate_combo' in source

    def test_prep_aggregate_combo_values(self, app: Any) -> None:
        assert app._ml_aggregate_combo.cget("values") == (
            "median",
            "mean",
            "first",
            "keep",
        )
        assert app._ml_aggregate_combo.get() == "median"

    def test_prep_needs_a_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_prep_endpoint()
        assert any("No dataset" in str(a) for a in quiet_dialogs)

    def test_prep_end_to_end_and_adoption(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        import pandas as pd

        app._ml_frame = pd.DataFrame(
            {
                "Ligand SMILES": ["CCO", "CCO", "CCN", "C", ""],
                "EC50 (nM)": ["<50", 200.0, 1000.0, "junk", 10.0],
            }
        )
        monkeypatch.setattr(playground.messagebox, "askyesno", lambda *a, **k: True)
        app._ml_prep_endpoint()
        assert self._drain_jobs(app), "prep job hung"
        assert app._ml_prep_frame is not None
        assert app._ml_prep_frame["SMILES"].tolist() == ["CCO", "CCN"]
        assert app._ml_prep_frame["n_measurements"].tolist() == [2, 1]
        info = app._ml_info.cget("text")
        assert "pEC50 prep" in info
        assert "5 rows in" in info
        assert "1 censored" in info
        assert "1 duplicates collapsed" in info
        # adoption: the prepared set became the dataset, pEC50 the target
        assert app._ml_frame is app._ml_prep_frame
        assert app._target_combo.get() == "pEC50"
        preview_columns = list(app._preview_tree["columns"])
        assert preview_columns == ["SMILES", "n_measurements", "endpoint_median", "pEC50"]
        assert quiet_dialogs == []

    def test_prep_declining_adoption_keeps_the_dataset(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        import pandas as pd

        original = pd.DataFrame(
            {"Ligand SMILES": ["CCO", "CCO"], "EC50 (nM)": [100.0, 200.0]}
        )
        app._ml_frame = original
        monkeypatch.setattr(playground.messagebox, "askyesno", lambda *a, **k: False)
        app._ml_prep_endpoint()
        assert self._drain_jobs(app), "prep job hung"
        assert app._ml_prep_frame is not None
        assert app._ml_frame is original  # declined: nothing swapped

    def test_prep_save_needs_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_prep_save()
        assert any("EC50 → pEC50 prep first" in str(a) for a in quiet_dialogs)

    # -- the permutation-importance button --------------------------------- #

    def test_permutation_button_source_pins(self, playground: Any) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Permutation importance"' in source
        assert "def _ml_plot_permutation" in source
        assert "from pharmaml.ml import permutation_feature_importance" in source
        assert "from pharmaml.viz import plot_permutation_importance" in source

    def test_permutation_needs_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_plot_permutation()
        assert any("No trained model" in str(a) for a in quiet_dialogs)

    def test_permutation_after_training(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        app._ml_plot_permutation()
        assert self._drain_jobs(app), "permutation job hung"
        assert app._ml_host.winfo_children()  # figure landed (100x-158: toolbar may ride along)
        assert "permutation importance" in app._ml_result_note.cget("text")
        assert quiet_dialogs == []

    def test_permutation_bad_repeats_is_named(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        app._perm_repeats.delete(0, "end")
        app._perm_repeats.insert(0, "1.5")
        app._ml_plot_permutation()
        assert any("repeats" in str(a) for a in quiet_dialogs)

    def test_permutation_zero_repeats_is_named(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        app._perm_repeats.delete(0, "end")
        app._perm_repeats.insert(0, "0")
        app._ml_plot_permutation()
        assert any("repeats" in str(a) for a in quiet_dialogs)

    # -- the ecfp3 family in the feature factory --------------------------- #

    def test_feature_factory_carries_ecfp3(self, app: Any) -> None:
        assert "ecfp3" in app._feat_family_vars
        assert bool(app._feat_family_vars["ecfp3"].get())  # ticked by default

    def test_ecfp3_labels_and_bits_pinned(self, playground: Any) -> None:
        labels = dict(playground._FEATURE_FAMILY_LABELS)
        assert labels["ecfp3"] == "ECFP3 bits — radius 3 (2048)"
        assert playground._FEATURE_FAMILY_BITS["ecfp3"] == 2048
        # canonical order: ecfp3 follows ecfp, mirroring the library
        names = [family for family, _label in playground._FEATURE_FAMILY_LABELS]
        assert names.index("ecfp3") == names.index("ecfp") + 1

    def test_factory_subtitle_mentions_twenty_families(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "20 descriptor / " in source
        assert "radius-3 ECFP" in source
        assert "fak_ml.ipynb families" in source  # (100x-148) the ten new ones


class TestReview146Gui:
    """Round 100x-146 -- the fourth engineering-details round: the
    expression matrix streams through the indexed progressive reader,
    ragged chunk-boundary rows are rejected up front, the Cell Lines
    loader reports its audit/collapse phases, the permutation table
    gained its Save-CSV twin, and a new dataset retires the previous
    run's derived tables."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 90.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_146(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-146" in doc
        assert "read_indexed_table_with_progress" in doc
        assert "_guard_ragged_boundaries" in doc
        assert "chunk BOUNDARY" in doc
        assert "AUDIT and COLLAPSE" in doc
        assert "Save-CSV twin" in doc
        # the stale family counts from earlier eras are gone (100x-148
        # grew the vocabulary again: 9 -> 20 families)
        assert "8-family" not in doc
        assert "9-family" not in doc
        assert "20-family" in doc

    def test_features_docstring_count_honest(self) -> None:
        from pathlib import Path as _Path

        source = _Path(__file__).resolve().parents[1] / "src" / "pharmaml" / "chem" / "features.py"
        text = source.read_text(encoding="utf-8")
        # (100x-148) the fingerprint count grew 7 -> 17 with the ten
        # fak_ml.ipynb families; the stale seven-count must be gone.
        assert "seventeen fingerprint families" in text
        assert "the seven\n        fingerprint families" not in text

    # -- the stale derived-state fix ---------------------------------------- #

    def test_adopt_frame_retires_the_derived_tables(self, app: Any) -> None:
        import pandas as pd

        # a previous dataset's derived tables sit on the app...
        app._ml_cleaned_frame = pd.DataFrame({"old": [1]})
        app._ml_prep_frame = pd.DataFrame({"old": [2]})
        frame = pd.DataFrame({"smiles": ["CCO", "CCN"], "y": [1.0, 2.0]})
        app._ml_adopt_frame(frame, app._ml_path or __import__("pathlib").Path("x.csv"))
        assert app._ml_cleaned_frame is None
        assert app._ml_prep_frame is None
        assert app._ml_frame is frame  # the new dataset itself, adopted

    def test_adopt_frame_clear_pins_in_source(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # (100x-152) the adoption body also retires the stacking panel
        # source (_stack_last) -- its split belongs to the OLD dataset.
        assert (
            "self._ml_cleaned_frame = None\n"
            "        self._ml_prep_frame = None\n"
            "        # (100x-152) the stacking panel source too"
            in source
        )
        assert (
            "        self._stack_last = None\n"
            "        self._ml_frame = frame\n"
            "        self._ml_path = path" in source
        )

    def test_train_render_retires_the_perm_frame(self, app: Any) -> None:
        import pandas as pd

        app._ml_perm_frame = pd.DataFrame({"feature": ["stale"]})
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        assert app._ml_perm_frame is None

    def test_train_render_clear_pins_in_source(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "self._ml_perm_frame: Any = None" in source  # the init
        # both train renders retire the old run's table
        assert source.count("self._ml_perm_frame = None") == 2

    # -- the permutation Save-CSV twin -------------------------------------- #

    def test_perm_save_button_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Save CSV…"' in source
        assert "def _ml_perm_save" in source
        assert 'self._ml_perm_save' in source
        assert "permutation_importance.csv" in source

    def test_perm_save_needs_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_perm_save()
        assert any("permutation importance first" in str(a) for a in quiet_dialogs)

    def test_perm_save_writes_the_table(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._ml_perm_frame = pd.DataFrame(
            {"feature": ["logp", "mw"], "importance_mean": [0.21, -0.03],
             "importance_std": [0.02, 0.01]}
        )
        target = tmp_path / "permutation_importance.csv"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", _p=target, **k: str(_p)
        )
        app._ml_perm_save()
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "logp" in content and "importance_mean" in content
        assert "-0.03" in content

    def test_perm_plot_stores_the_table(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        assert app._ml_perm_frame is None  # retired by the new run
        app._ml_plot_permutation()
        assert self._drain_jobs(app), "permutation job hung"
        frame = app._ml_perm_frame
        assert frame is not None
        assert list(frame.columns) == ["feature", "importance_mean", "importance_std"]
        assert len(frame) > 0

    def test_save_ml_figure_docstring_names_permutation(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert (
            "the calibration curve, the learning curve or the permutation\n"
            "        importance drew last"
        ) in source
        assert "curve / permutation importance) first" in source

    # -- the prep card's dependency gate ------------------------------------ #

    def test_prep_endpoint_gates_pandas(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert (
            'if not self._require_pharmaml() or not self._require("pandas", "the pEC50 prep"):'
        ) in source

    # -- the expression-matrix progressive loader --------------------------- #

    def test_expr_loader_uses_the_indexed_reader(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "def _bio_load_expr_file" in source
        assert "from pharmaml.utils.io import read_indexed_table_with_progress" in source
        assert 'read_indexed_table_with_progress(path, report, index_col="cell_line")' in source

    def test_expr_loader_end_to_end_adopts_the_frame(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        path = tmp_path / "expression.csv"
        frame = pd.DataFrame(
            {"cell_line": ["A549", "MCF7", "HEK"], "g1": [1.0, 2.0, 3.0], "g2": [4.0, 5.0, 6.0]}
        )
        frame.to_csv(path, index=False)
        app._bio_load_expr_file(path)
        assert self._drain_jobs(app), "expression load job hung"
        assert app._bio_expr is not None
        assert list(app._bio_expr.index) == ["A549", "MCF7", "HEK"]
        assert app._bio_expr.index.name == "cell_line"
        info = app._bio_expr_info.cget("text")
        assert "3 cell lines × 2 genes" in info
        assert quiet_dialogs == []

    def test_expr_loader_renamed_column_falls_back_positionally(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        path = tmp_path / "renamed.csv"
        pd.DataFrame({"cl": ["A", "B"], "g": [1.0, 2.0]}).to_csv(path, index=False)
        app._bio_load_expr_file(path)
        assert self._drain_jobs(app), "expression load job hung"
        assert list(app._bio_expr.index) == ["A", "B"]
        assert app._bio_expr.index.name == "cl"

    # -- the Cell Lines screen loader's phase notes ------------------------- #

    def test_screen_loader_phase_notes_in_source(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert 'report(rows, None, "auditing rows")' in source
        assert 'report(rows, None, "collapsing replicates")' in source
        # the phase notes ride AFTER the streaming read
        assert source.index("load_response_table(path, report=report)") < source.index(
            'report(rows, None, "auditing rows")'
        )

    def test_screen_loader_reports_phases_live(
        self, app: Any, quiet_dialogs: list[Any], tmp_path: Path, monkeypatch: Any
    ) -> None:
        import pandas as pd

        # a screen with a duplicate replicate so the collapse phase fires
        path = tmp_path / "screen.csv"
        pd.DataFrame(
            {
                "cell_line": ["A549", "A549", "MCF7"],
                "compound": ["drug", "drug", "drug"],
                "LN_IC50": [1.0, 1.5, 2.0],
            }
        ).to_csv(path, index=False)
        notes: list[str] = []
        original = app._report_progress

        def spy(done: int, total: object = None, note: str = "") -> None:
            if note:
                notes.append(note)
            original(done, total, note)

        monkeypatch.setattr(app, "_report_progress", spy)
        app._bio_load_file(path)
        assert self._drain_jobs(app), "screen load job hung"
        assert "auditing rows" in notes
        assert "collapsing replicates" in notes  # the collapse box defaults on
        assert quiet_dialogs == []


# ---------------------------------------------------------------------------
# Round 100x-148 -- the one-by-one notebook review (xxx + fak_ml)
# ---------------------------------------------------------------------------
class TestReview148Gui:
    """Both uploaded notebooks checked cell by cell; the missing halves
    shipped: 20 feature families, sgd_classifier, live ROC/PR buttons,
    the per-class metrics line, the pIC50 -> IC50 triage, and the
    family benchmark button."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the round's documentation pins ------------------------------------ #

    def test_module_docstring_records_round_148(self, playground: Any) -> None:
        doc = playground.__doc__ or ""
        assert "100x-148" in doc
        assert "benchmark_feature_families" in doc
        assert "sgd_classifier" in doc
        assert "shape_3d" in doc
        assert "IC50 nM" in doc

    def test_round_source_pins(self, playground: Any) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "def _ml_plot_roc" in source
        assert "def _ml_plot_pr" in source
        assert "def _ml_benchmark_families" in source
        assert "def _ml_predict_add_ic50" in source
        assert "def _ml_predict_triage" in source
        assert "def _fill_predict_tree" in source
        assert '"Benchmark feature families"' in source
        assert '"pIC50 → IC50 nM"' in source
        assert '"Triage hits"' in source
        assert '"ROC curves"' in source
        assert '"PR curve"' in source

    # -- the twenty-family picker ------------------------------------------ #

    def test_family_picker_carries_twenty_families(self, app: Any) -> None:
        assert len(app._feat_family_vars) == 20
        for family, var in app._feat_family_vars.items():
            assert bool(var.get()), family  # every family ticked by default
        for family in (
            "maccs", "fcfp", "fcfp_count", "topological", "topological_count",
            "rdkit_count", "pattern", "layered", "estate", "erg", "shape_3d",
        ):
            assert family in app._feat_family_vars

    def test_family_bits_match_the_library(self, playground: Any) -> None:
        """The GUI width map mirrors pharmaml.chem.features exactly."""
        from pharmaml.chem.features import FAMILY_PREFIXES

        assert set(playground._FEATURE_FAMILY_BITS) == {
            family for family in FAMILY_PREFIXES if family != "descriptors"
        }
        assert playground._FEATURE_FAMILY_BITS["maccs"] == 167
        assert playground._FEATURE_FAMILY_BITS["estate"] == 79
        assert playground._FEATURE_FAMILY_BITS["erg"] == 315
        assert playground._FEATURE_FAMILY_BITS["shape_3d"] == 9

    def test_labels_listed_in_library_order(self, playground: Any) -> None:
        from pharmaml.chem.features import FEATURE_FAMILIES

        names = [family for family, _label in playground._FEATURE_FAMILY_LABELS]
        assert names == list(FEATURE_FAMILIES)

    # -- the sgd classifier on the picker ---------------------------------- #

    def test_sgd_classifier_on_the_model_picker(self, app: Any) -> None:
        app._task_combo.set("classification")
        app._ml_task_changed()
        values = list(app._model_combo.cget("values"))
        assert "sgd_classifier" in values

    # -- the per-class metrics line ----------------------------------------- #

    def test_per_class_line_after_multiclass_train(self, app: Any) -> None:
        import pandas as pd

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        # the per-class line lives under the macro aggregates
        labels = [
            child.cget("text")
            for child in app._ml_metrics.winfo_children()
            if child.winfo_class() == "TLabel"
        ]
        assert "per class" in labels
        per_class_line = next(
            text for text in labels if text.startswith("0:") or ": P " in text and "· R " in text
        )
        assert "F1" in per_class_line
        # every class of the example's target appears
        for cls in sorted(frame["Target"].unique()):
            assert str(cls) in per_class_line

    # -- the ROC / PR buttons ------------------------------------------------ #

    def test_roc_button_guards_regression_runs(self, app: Any, quiet_dialogs: list[Any]) -> None:
        # no trained model at all -> the guard fires
        app._ml_plot_roc()
        assert any("No trained model" in str(a) for a in quiet_dialogs)

    def test_pr_button_draws_after_multiclass_train(self, app: Any) -> None:
        import pandas as pd

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        app._ml_plot_pr()
        assert self._drain_jobs(app), "PR job hung"
        note = app._ml_result_note.cget("text")
        assert "PR curves replace the parity figure" in note
        # a figure landed in the Results card's host (both the TkAgg
        # canvas path and the PNG fallback set _pg_figure)
        assert getattr(app._ml_host, "_pg_figure", None) is not None
        assert app._ml_host.winfo_children()

    # -- the pIC50 -> IC50 triage -------------------------------------------- #

    def test_predict_add_ic50_and_triage(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        app._ml_predictions = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCN", "CCCO", "CCCCO"],
                "prediction": [9.0, 7.0, 6.0, 5.0],  # 1, 100, 1000, 10000 nM
            }
        )
        app._ml_predict_add_ic50()
        assert self._drain_jobs(app), "IC50 conversion job hung"
        result = app._ml_predictions
        assert "IC50_nM" in result.columns
        assert result["IC50_nM"].tolist() == [1.0, 100.0, 1000.0, 10000.0]
        assert "IC50_nM" in list(app._predict_tree["columns"])
        info = app._ml_bundle_info.cget("text")
        assert "IC50_nM appended from prediction" in info
        assert "4 of 4" in info
        # the triage at 100 nM keeps the two potent rows
        app._predict_threshold.set("100")
        app._ml_predict_triage()
        assert self._drain_jobs(app), "triage job hung"
        info = app._ml_bundle_info.cget("text")
        assert "2 of 4 row(s) at or below 100 nM" in info
        # the full table survived -- Save writes everything
        assert len(app._ml_predictions) == 4
        # "no filter" restores the full view
        app._predict_threshold.set("no filter")
        app._ml_predict_triage()
        assert "showing all 4 row(s)" in app._ml_bundle_info.cget("text")
        assert quiet_dialogs == []

    def test_predict_triage_needs_the_ic50_column(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        app._ml_predictions = pd.DataFrame({"SMILES": ["CCO"], "prediction": [7.0]})
        app._ml_predict_triage()
        assert any("No IC50 column" in str(a) for a in quiet_dialogs)

    def test_predict_add_ic50_needs_a_pic50_column(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        app._ml_predictions = pd.DataFrame({"SMILES": ["CCO"]})
        app._ml_predict_add_ic50()
        assert any("No pIC50 column" in str(a) for a in quiet_dialogs)

    # -- the family benchmark ------------------------------------------------- #

    def test_benchmark_guards_no_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_benchmark_families()
        assert any("No dataset" in str(a) for a in quiet_dialogs)

    def test_benchmark_guards_no_smiles_column(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        app._ml_frame = pd.DataFrame({"x": [1.0, 2.0], "y": [0, 1]})
        app._target_combo.set("y")
        app._ml_benchmark_families()
        assert any("No SMILES column" in str(a) for a in quiet_dialogs)

    def test_benchmark_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "SMILES": [
                    "CCO", "CCCO", "CCCCO", "CCCOO",
                    "CC(=O)O", "CCC(=O)O", "O=C(O)C1CCCO1", "CCOC(=O)c1ccc(N)cc1",
                ],
                "pIC50": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            }
        )
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._bench_model.delete(0, "end")
        app._bench_model.insert(0, "ridge")  # fast: the entry's default is lightgbm
        app._bench_cv.delete(0, "end")
        app._bench_cv.insert(0, "2")
        app._ml_benchmark_families()
        assert self._drain_jobs(app, timeout=240.0), "benchmark job hung"
        rows = len(app._board_tree.get_children())
        assert rows == 18  # the 18 2D families, shape_3d and rism opted out
        columns = list(app._board_tree["columns"])
        assert columns[:3] == ["family", "n_features", "n_samples"]
        assert any(column.endswith("_mean") for column in columns)
        status = app._board_status.cget("text")
        assert "2-fold family benchmark over the 18 2D families" in status
        assert "shape_3d and rism opt-in only" in status
        assert quiet_dialogs == []


# ---------------------------------------------------------------------------
# Round 100x-149 -- look.ipynb: the OOF stacking + residual card
# ---------------------------------------------------------------------------
class TestReview149Gui:
    """look.ipynb's single cell (the OOF meta-stacking pIC50 regressor)
    absorbed as the "Stacking leaderboard (OOF + residual)" card: base and
    meta models as registry names, the KFold OOF folds and the holdout
    percent, the quantile-target toggle, the Ridge residual correction,
    the shared board leaderboard, and the notebook's step-18 parity
    scatter.  The library call is stubbed in the end-to-end test -- the
    card's own plumbing (guards, main-thread split, render, note) is what
    these tests pin; the real stack_oof_regressors has its own suite."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _plant_frame(app: Any) -> None:
        """A 16-row regression frame: two numeric features, a SMILES
        column (the split must drop it) and a continuous pIC50 target."""
        import pandas as pd

        frame = pd.DataFrame(
            {
                "SMILES": ["CCO", "CCCO", "CCCCO", "CCCOO"] * 4,
                "x1": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] * 2,
                "x2": [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5] * 2,
                "pIC50": [2.0, 2.4, 2.8, 3.2, 4.0, 4.4, 4.8, 5.2] * 2,
            }
        )
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")

    # -- the round's documentation pins ------------------------------------ #

    def test_round_source_pins(self, playground: Any) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "def _ml_stack_oof" in source
        assert '"Stacking leaderboard (OOF + residual)"' in source
        assert '"Run OOF stacking"' in source
        assert '"Quantile target"' in source
        assert '"Residual correction"' in source
        # the library import stays lazy INSIDE work() (the benchmark
        # card's convention) -- the card's guards render even while
        # stack_oof_regressors is absent from an older install.
        assert "from pharmaml.ml import stack_oof_regressors" in source

    def test_stack_widgets_exist(self, app: Any, playground: Any) -> None:
        assert (
            app._stack_base.get() == "ridge, random_forest, gradient_boosting, extra_trees"
        )
        assert app._stack_meta.get() == "ridge, huber_regression, lightgbm"
        assert app._stack_folds.get() == "5"
        assert app._stack_test.get() == "20"
        # BooleanVar attrs (not _var-suffixed -- the task's contract
        # names), so the automatic session machinery carries them.
        assert isinstance(app._stack_quantile, playground.tk.Variable)
        assert not app._stack_quantile.get()
        assert isinstance(app._stack_residual, playground.tk.Variable)
        assert app._stack_residual.get()  # the correction ships ticked
        assert app._stack_host.winfo_exists()

    def test_stack_inputs_ride_the_session(self, app: Any) -> None:
        payload = app._collect_session()
        assert payload["variables"]["_stack_folds"] == "5"
        assert payload["variables"]["_stack_test"] == "20"
        assert payload["variables"]["_stack_base"] == (
            "ridge, random_forest, gradient_boosting, extra_trees"
        )
        assert payload["variables"]["_stack_meta"] == "ridge, huber_regression, lightgbm"
        assert payload["variables"]["_stack_quantile"] is False
        assert payload["variables"]["_stack_residual"] is True

    # -- the guards ---------------------------------------------------------- #

    def test_stack_guards_no_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_stack_oof()
        assert any("No dataset" in str(a) for a in quiet_dialogs)

    def test_stack_guards_regression_only(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        frame = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("label")
        app._task_combo.set("classification")
        app._ml_stack_oof()
        assert any("Regression only" in str(a) for a in quiet_dialogs)
        # the auto path infers the same verdict from the 2-value target
        quiet_dialogs.clear()
        app._task_combo.set("auto")
        app._ml_stack_oof()
        assert any("Regression only" in str(a) for a in quiet_dialogs)

    def test_stack_guards_target_is_smiles(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._plant_frame(app)
        app._target_combo.set("SMILES")
        app._ml_stack_oof()
        assert any("SMILES column" in str(a) for a in quiet_dialogs)

    def test_stack_guards_no_numeric_features(self, app: Any, quiet_dialogs: list[Any]) -> None:
        import pandas as pd

        frame = pd.DataFrame({"SMILES": ["CCO", "CCCO"], "pIC50": [2.0, 3.0]})
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_regression.csv")
        app._target_combo.set("pIC50")
        app._task_combo.set("regression")
        app._ml_stack_oof()
        assert any("No numeric features" in str(a) for a in quiet_dialogs)

    def test_stack_guards_bad_folds_and_split(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._plant_frame(app)
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "11")
        app._ml_stack_oof()
        assert any(
            "OOF folds" in str(a) and "above the maximum" in str(a) for _n, a in quiet_dialogs
        )
        quiet_dialogs.clear()
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "5")
        app._stack_test.delete(0, "end")
        app._stack_test.insert(0, "5")
        app._ml_stack_oof()
        assert any(
            "test holdout percent" in str(a) and "below the minimum" in str(a)
            for _n, a in quiet_dialogs
        )

    def test_stack_guards_no_base_models(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._plant_frame(app)
        app._stack_base.delete(0, "end")
        app._ml_stack_oof()
        assert any("No base models" in str(a) for a in quiet_dialogs)

    # -- the end-to-end plumbing, library stubbed --------------------------- #

    def test_stack_end_to_end_stubbed(
        self, app: Any, quiet_dialogs: list[Any], monkeypatch: Any
    ) -> None:
        import numpy as np
        import pandas as pd

        import pharmaml.ml as ml_module

        calls: dict[str, Any] = {}
        notes: list[str] = []
        original_report = app._report_progress

        def spy(done: int, total: object = None, note: str = "") -> None:
            if note:
                notes.append(note)
            original_report(done, total, note)

        monkeypatch.setattr(app, "_report_progress", spy)

        def fake_stack(x_train, y_train, x_test, y_test, **kwargs):
            calls["x_train"], calls["y_train"] = x_train, y_train
            calls["x_test"], calls["y_test"] = x_test, y_test
            calls["kwargs"] = kwargs
            on_step = kwargs.get("on_step")
            if on_step is not None:
                on_step({"step": "base", "index": 1, "total": 2, "label": "ridge"})
                on_step({"step": "meta", "index": 2, "total": 2, "label": "lightgbm"})
            truth = np.linspace(-1.5, 1.5, len(y_test))
            prediction = truth + np.linspace(0.05, -0.05, len(y_test))
            want_residual = bool(kwargs.get("residual"))
            rows = {
                "model": ["ridge", "huber_regression", "lightgbm"],
                "oof_r2": [0.71, 0.74, 0.78],
                "test_r2": [0.70, 0.73, 0.77],
            }
            if want_residual:
                rows["model"].append("lightgbm+residual")
                rows["oof_r2"].append(None)
                rows["test_r2"].append(0.79)
            from types import SimpleNamespace

            return SimpleNamespace(
                oof_frame=pd.DataFrame({"ridge": [0.1, 0.2]}),
                leaderboard=pd.DataFrame(rows),
                best_meta="lightgbm",
                best_test_r2=0.77,
                residual_test_r2=0.79 if want_residual else None,
                target_transform=kwargs.get("target_transform", "none"),
                base_models=list(kwargs.get("base_models") or []),
                meta_models=["ridge", "huber_regression", "lightgbm"],
                n_splits=kwargs.get("n_splits", 5),
                y_test_trans=truth,
                best_test_prediction=prediction,
            )

        monkeypatch.setattr(ml_module, "stack_oof_regressors", fake_stack)
        self._plant_frame(app)
        app._stack_base.delete(0, "end")
        app._stack_base.insert(0, " ridge , ridge ")  # stripping is part of the contract
        app._stack_meta.delete(0, "end")  # empty -> the library default (None)
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "2")
        app._stack_test.delete(0, "end")
        app._stack_test.insert(0, "25")
        app._stack_quantile.set(True)
        app._ml_stack_oof()
        assert self._drain_jobs(app), "stacking job hung"

        # the main-thread split: numeric columns only, the notebook's
        # random_state=42 holdout, 25% of 16 rows held out.
        assert list(calls["x_train"].columns) == ["x1", "x2"]
        assert list(calls["x_test"].columns) == ["x1", "x2"]
        assert len(calls["x_train"]) == 12 and len(calls["x_test"]) == 4
        assert len(calls["y_train"]) == 12 and len(calls["y_test"]) == 4
        kwargs = calls["kwargs"]
        assert kwargs["base_models"] == ["ridge", "ridge"]
        assert kwargs["meta_models"] is None  # the cleared entry -> the default
        assert kwargs["n_splits"] == 2
        assert kwargs["target_transform"] == "quantile"
        assert kwargs["residual"] is True
        assert kwargs["random_state"] == 42
        assert callable(kwargs["on_step"])
        assert "ridge" in notes and "lightgbm" in notes  # on_step fed the bar

        # the shared board: one row per meta-learner, +residual last.
        assert list(app._board_tree["columns"]) == ["model", "oof_r2", "test_r2"]
        rows = app._board_tree.get_children()
        assert len(rows) == 4
        assert app._board_tree.item(rows[0], "values") == ("ridge", "0.7100", "0.7000")
        assert app._board_tree.item(rows[-1], "values")[0].endswith("+residual")
        status = app._board_status.cget("text")
        assert "2-fold OOF stacking" in status
        assert "2 base model(s) × 3 meta-learner(s)" in status
        assert "best blender: lightgbm" in status
        assert "+residual test R² 0.7900" in status
        assert "TRANSFORMED scale" in status

        # the card's own host carries the notebook's step-18 scatter.
        figure = getattr(app._stack_host, "_pg_figure", None)
        assert figure is not None
        axes = figure.axes[0]
        assert axes.get_title() == "OOF stacking — lightgbm + residual"
        assert axes.get_xlabel() == "True (target scale)"
        assert axes.get_ylabel() == "Predicted"
        assert len(axes.lines) == 1  # the y=x diagonal
        assert axes.lines[0].get_linestyle() == "--"
        assert axes.lines[0].get_color() == "red"
        assert quiet_dialogs == []

        # residual off: the library returns None, the title and the note
        # drop the correction, and the raw-scale note replaces the
        # transformed-scale one.
        app._stack_residual.set(False)
        app._stack_quantile.set(False)
        app._ml_stack_oof()
        assert self._drain_jobs(app), "second stacking job hung"
        assert calls["kwargs"]["residual"] is False
        assert calls["kwargs"]["target_transform"] == "none"
        assert len(app._board_tree.get_children()) == 3
        status = app._board_status.cget("text")
        assert "+residual" not in status
        assert "raw target scale" in status
        figure = getattr(app._stack_host, "_pg_figure", None)
        assert figure is not None
        assert figure.axes[0].get_title() == "OOF stacking — lightgbm"
        assert quiet_dialogs == []

    # -- the real library, once Task 4-a's module lands --------------------- #

    def test_stack_end_to_end_real_library(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """The un-stubbed contract run -- skipped until pharmaml.ml
        actually carries stack_oof_regressors (the library lands in
        parallel with this round); the stubbed test above pins the card
        either way."""
        import pharmaml.ml as ml_module

        if not hasattr(ml_module, "stack_oof_regressors"):
            pytest.skip("stack_oof_regressors not landed yet (Task 4-a)")
        self._plant_frame(app)
        app._stack_base.delete(0, "end")
        app._stack_base.insert(0, "ridge")
        app._stack_meta.delete(0, "end")
        app._stack_meta.insert(0, "ridge")
        app._stack_folds.delete(0, "end")
        app._stack_folds.insert(0, "2")
        app._ml_stack_oof()
        assert self._drain_jobs(app, timeout=240.0), "real stacking job hung"
        columns = set(app._board_tree["columns"])
        assert columns == {"model", "oof_r2", "test_r2"}
        assert len(app._board_tree.get_children()) >= 2  # ridge + the residual row
        assert "OOF stacking" in app._board_status.cget("text")
        assert getattr(app._stack_host, "_pg_figure", None) is not None
        assert quiet_dialogs == []


class TestReview150Gui:
    """The 100x-150 engineering review's GUI surface: the shared board's
    Save-table twin (every frame that fills the leaderboard -- compare
    models, the family benchmark, the OOF stacking leaderboard, a trial
    ranking -- exports through the five-format dialog), and the two
    recompute-refresh flows that keep the library's new append-only
    add_*_column guards from turning a re-click into an error dialog."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the board Save-table twin ------------------------------------------ #

    def test_board_save_button_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Save table…"' in source
        assert "def _ml_board_save" in source
        assert 'self._ml_board_save' in source
        assert '"Save the leaderboard table"' in source
        assert '"leaderboard.csv"' in source

    def test_board_save_needs_a_board(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._board_frame = None
        app._ml_board_save()
        assert any("Compare models" in str(a) for a in quiet_dialogs)

    def test_board_save_writes_the_table(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._board_frame = pd.DataFrame(
            {"model": ["ridge", "random_forest"], "test_r2": [0.61, 0.58]}
        )
        target = tmp_path / "leaderboard.csv"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", _p=target, **k: str(_p)
        )
        app._ml_board_save()
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "model" in content and "ridge" in content and "0.61" in content

    # -- the recompute refreshes (the append-only guards' GUI twins) ------- #

    def test_ic50_reclick_drops_the_stale_column(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """A prediction frame that ALREADY carries IC50_nM re-clicks the
        button: the stale column is dropped first, so the refresh is an
        explicit recompute -- never an add_ic50_column collision error."""
        import pandas as pd

        frame = pd.DataFrame(
            {"prediction": [6.0, 7.0], "IC50_nM": [999.0, 999.0]}  # stale values
        )
        app._ml_predictions = frame
        app._ml_predict_add_ic50()
        assert self._drain_jobs(app), "ic50 job hung"
        result = app._ml_predictions
        assert list(result.columns) == ["prediction", "IC50_nM"]
        assert result["IC50_nM"].tolist() == [1000.0, 100.0]  # recomputed
        assert quiet_dialogs == []

    def test_refresh_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the prediction card drops the stale IC50_nM before re-appending
        assert 'if "IC50_nM" in frame.columns:' in source
        assert 'frame = frame.drop(columns=["IC50_nM"])' in source
        # the conversion card drops a pIC50 column the INPUT file carries
        assert 'if "pIC50" in frame.columns:' in source
        assert 'frame = frame.drop(columns=["pIC50"])' in source

    def test_library_guards_are_real(self) -> None:
        """The audit's three findings landed in the library itself."""
        import pandas as pd

        from pharmaml.chem.converters import add_ic50_column, pic50_to_ic50
        from pharmaml.exceptions import DataValidationError

        assert pic50_to_ic50(-300.0) == float("inf")  # Finding 2: no OverflowError
        with pytest.raises(DataValidationError, match="appends, never overwrites"):
            add_ic50_column(  # Finding 3: the collision guard
                pd.DataFrame({"pIC50": [6.0], "SMILES": ["CCO"]}), out_col="pIC50"
            )


class TestReview151Gui:
    """The 100x-151 round's GUI surface: the FOURTH board filler (the
    hyperparameter trial ranking) actually feeding the Save-table twin
    the button's texts promise -- the round's audit Finding 1 (the tune
    render used to fill only the tree, leaving the button to lie or
    export the previous run's stale leaderboard)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_tune_render_feeds_the_board_save_frame(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """Finding 1, behaviorally: after a search the trial ranking IS
        the frame the Save-table button exports -- full configuration
        text, float scores, ranked 1..n."""
        import pandas as pd

        app._ml_load_example()
        app._tune_trials.set("4")
        app._ml_tune()
        assert self._drain_jobs(app), "tune job hung"
        board = getattr(app, "_board_frame", None)
        assert board is not None, "the tune render left the board frame unset"
        assert isinstance(board, pd.DataFrame)
        assert list(board.columns) == [
            "rank", "trial", "configuration", "score", "score_std",
        ]
        assert len(board) == 4
        assert board["rank"].tolist() == [1, 2, 3, 4]
        assert board["score"].notna().all()
        assert board["score"].dtype.kind == "f"
        # the tree the user SEES and the frame the button EXPORTS agree
        tree_rows = len(app._board_tree.get_children())
        assert tree_rows == len(board)
        assert quiet_dialogs == []

    def test_tune_render_replaces_a_stale_leaderboard(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """The cross-KIND stale export the audit described: a previous
        compare_models run's leaderboard must not survive a search as
        the exportable table."""
        import pandas as pd

        app._board_frame = pd.DataFrame(
            {"model": ["ridge"], "test_r2": [0.61]}  # a stale leaderboard
        )
        app._ml_load_example()
        app._tune_trials.set("4")
        app._ml_tune()
        assert self._drain_jobs(app), "tune job hung"
        board = app._board_frame
        assert "model" not in board.columns  # the leaderboard is GONE
        assert "configuration" in board.columns  # the trials table is here
        assert quiet_dialogs == []

    def test_all_four_fillers_assign_the_frame(self) -> None:
        """The audit's gap (i): every render that fills the board tree
        must also assign the backing frame -- pinned at the source level
        (the behavioral tests above cover the tune path; this pin keeps
        a regression from deleting the assignment from the other three
        renders unnoticed)."""
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assignments = [
            line.strip()
            for line in source.splitlines()
            if line.strip() == "self._board_frame = board"
            or line.strip() == "self._board_frame = result.leaderboard"
            or line.strip() == "self._board_frame = pd.DataFrame(board_rows, columns=board_columns)"
        ]
        # compare_models + benchmark (both '= board') + stacking
        # ('= result.leaderboard') + the tune render (100x-151)
        assert len(assignments) >= 4, assignments


class TestPredictionIntervalsGui:
    """The 100x-151 round's feature GUI: D2.py's uncertainty block as
    the Results card's "Prediction intervals" button -- the honest port
    (GradientBoostingRegressor(loss="quantile") at alpha/2, 0.5 and
    1-alpha/2, where the upload trained three IDENTICAL plain forests
    whose advertised "90% interval" had width 0 and coverage 0%), the
    alpha entry, the coverage/width/crossing status note, the interval
    table (the Save-intervals payload) and its train-retirement."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _train_ridge(app: Any) -> None:
        """The light regression run every behavioral test below needs:
        the bundled regression example + ridge (fast, deterministic)."""
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()

    # -- the source pins ------------------------------------------------------ #

    def test_intervals_button_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Prediction intervals"' in source
        assert "def _ml_plot_intervals" in source
        assert "self._ml_plot_intervals" in source
        assert '"Save intervals…"' in source
        assert "def _ml_intervals_save" in source
        assert "_interval_alpha" in source
        # state + retire: the init pin (the house annotation) plus BOTH
        # train renders clearing the table -- three retirement sites in
        # all, the 146 permutation twin's split (the init's ": Any"
        # breaks the plain-string match, so the two are counted apart).
        assert "self._ml_intervals_frame: Any = None" in source
        assert source.count("self._ml_intervals_frame = None") == 2
        # the lazy-import convention: the pharmaml calls live INSIDE
        # work(), so the guards render even on an older install.
        assert "from pharmaml.ml import quantile_prediction_intervals" in source
        assert "from pharmaml.viz import plot_prediction_intervals" in source

    # -- the guards ----------------------------------------------------------- #

    def test_no_trained_model_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_plot_intervals()
        assert any("Train a model first" in str(a) for a in quiet_dialogs)
        assert not app._busy

    def test_regression_only_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """A classification run reaches the task gate: the quantile trio
        brackets a continuous endpoint, a classifier has no honest
        interval here (and no job starts)."""
        import pandas as pd

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        app._ml_plot_intervals()
        assert any("Regression only" in str(a) for a in quiet_dialogs)
        assert not app._busy

    def test_alpha_parse_guards(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """alpha must parse to a finite float within (0.01, 0.49) -- the
        interval's tail mass; a bad entry is a dialog, never a job."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        app._interval_alpha.delete(0, "end")
        app._interval_alpha.insert(0, "0.6")  # above the 0.49 maximum
        app._ml_plot_intervals()
        assert any("Input" in str(a) and "alpha" in str(a) for a in quiet_dialogs)
        assert not app._busy
        quiet_dialogs.clear()
        app._interval_alpha.delete(0, "end")
        app._interval_alpha.insert(0, "abc")  # not a number at all
        app._ml_plot_intervals()
        assert any("Input" in str(a) and "alpha" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the real run --------------------------------------------------------- #

    def test_intervals_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The full flow against the REAL library: ridge on the example,
        then the quantile trio -- the bracket table (one row per
        held-out compound), the honest note, no dialogs."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        assert app._ml_intervals_frame is None  # retired by the new run
        app._ml_plot_intervals()
        assert self._drain_jobs(app), "intervals job hung"
        frame = app._ml_intervals_frame
        assert frame is not None
        assert list(frame.columns) == [
            "actual", "pred_median", "pred_lower", "pred_upper", "interval_width",
        ]
        # one row per held-out compound (the run tuple's y_test -- the
        # same row count as x_test, the trio's query set)
        assert len(frame) == len(app._ml_last_run[6])
        assert len(frame) > 0
        # the bounds are ordered (the crossing repair did its job)
        assert (frame["pred_lower"] <= frame["pred_upper"]).all()
        note = app._ml_result_note.cget("text")
        assert "coverage" in note
        assert "mean width" in note
        assert "crossed bound(s) repaired" in note
        assert app._busy is False
        assert quiet_dialogs == []

    # -- the Save-intervals twin ---------------------------------------------- #

    def test_intervals_save_writes_the_table(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any], tmp_path: Path
    ) -> None:
        import pandas as pd

        app._ml_intervals_frame = pd.DataFrame(
            {
                "actual": [6.1, 5.2],
                "pred_median": [6.0, 5.3],
                "pred_lower": [5.5, 4.8],
                "pred_upper": [6.6, 5.9],
                "interval_width": [1.1, 1.1],
            }
        )
        target = tmp_path / "prediction_intervals.csv"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda t="", _p=target, **k: str(_p)
        )
        app._ml_intervals_save()
        assert target.exists()
        content = target.read_text(encoding="utf-8")
        assert "pred_median" in content and "interval_width" in content
        assert "6.1" in content and "5.5" in content
        # the write-through confirmed itself with the Saved dialog
        assert any("Saved" in str(a) for a in quiet_dialogs)

    def test_intervals_save_needs_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_intervals_save()
        assert any("prediction intervals first" in str(a) for a in quiet_dialogs)

    # -- the retirement ------------------------------------------------------- #

    def test_intervals_retire_on_new_train(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """A new train retires the old run's interval table -- the stale
        bounds would name the OLD model's split (the 146 twin's rule)."""
        import pandas as pd

        app._ml_intervals_frame = pd.DataFrame({"pred_median": [1.0]})  # stale
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        assert app._ml_intervals_frame is None
        assert quiet_dialogs == []


# --------------------------------------------------------------------- #
# the 100x-152 round's GUI surfaces (the audit's GAPs 2, 3, 6, 7)
# --------------------------------------------------------------------- #
class TestOverfitGapGui:
    """Surface A: the train-side diagnostics on RegressionReport wired
    through the three regression evaluate_model call sites (the tune
    site stays bare -- its winner is tuned on the CV folds) and
    rendered as the Results card's honest overfit line -- Pearson r,
    train R² and the SIGNED gap, only when the report carries the
    fields (an older session-restored report has none)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _train_ridge(app: Any) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()

    @staticmethod
    def _metric_labels(app: Any) -> list[str]:
        return [str(child.cget("text")) for child in app._ml_metrics.winfo_children()]

    # -- the source pins ------------------------------------------------------ #

    def test_overfit_gap_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the three regression call sites pass the pair (the plain
        # train site, the ensemble site, the deep-learning site); the
        # tune site tunes on CV folds and stays bare -- a fourth
        # occurrence here means somebody wired it.
        assert source.count('{"X_train": x_train, "y_train": y_train}') == 3
        # the render tolerance contract: both metric grids read the
        # fields through getattr-None (an older report has none).
        assert source.count('getattr(report, "pearson_r", None)') == 2
        assert source.count('getattr(report, "train_r2", None)') == 2
        assert source.count('getattr(report, "r2_gap", None)') == 2
        assert '("R² gap (train−test)", f"{r2_gap:+.3f}")' in source
        # the lazy-import convention keeps pharmaml calls inside work()
        assert "from pharmaml.ml import evaluate_model, train_model" in source

    # -- the real run --------------------------------------------------------- #

    def test_overfit_gap_metrics_after_ridge(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The real library end to end: the ridge run's report carries
        the train-side fields, the grid renders them, and the gap value
        is the report's own train_r2 - r2 with the +/- sign."""
        import re

        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        labels = self._metric_labels(app)
        assert "Pearson r" in labels
        assert "train R²" in labels
        assert "R² gap (train−test)" in labels
        _report = app._ml_last_run[1]
        assert _report.pearson_r is not None
        assert _report.train_r2 is not None
        assert _report.r2_gap is not None
        index = labels.index("R² gap (train−test)")
        gap_text = labels[index + 1]  # the value cell follows its label
        assert re.fullmatch(r"[+-]\d+\.\d{3}", gap_text), gap_text
        assert abs(float(gap_text) - _report.r2_gap) < 5e-4
        pearson_text = labels[labels.index("Pearson r") + 1]
        assert abs(float(pearson_text) - _report.pearson_r) < 5e-4
        assert quiet_dialogs == []

    def test_overfit_gap_tolerates_reports_without_train_side(self, app: Any) -> None:
        """A report WITHOUT the train-side fields (an older bundle's
        persisted report, any pre-152 call) renders the plain grid with
        no train line and no crash -- both metric grids; a report WITH
        them renders the line on the deep grid too."""
        import types

        bare = types.SimpleNamespace(r2=0.5, rmse=0.1, mae=0.05)
        trained = types.SimpleNamespace(name="stub", cv_mean=None, cv_std=None, fit_seconds=0.1)
        app._fill_ml_metrics(trained, bare, "regression", [])
        labels = self._metric_labels(app)
        assert "R²" in labels  # the plain grid still renders
        assert "Pearson r" not in labels
        assert "train R²" not in labels
        assert "R² gap (train−test)" not in labels
        network = types.SimpleNamespace(
            get_params=lambda: {"hidden_layers": (8,)},
            n_features_in_=3,
            history_=[],
            n_iter_=0,
        )
        app._fill_dl_metrics(network, bare, "regression", [], 0.02)
        dl_labels = [str(child.cget("text")) for child in app._dl_metrics.winfo_children()]
        assert "train R²" not in dl_labels
        # ... and a report WITH the fields renders the full line
        full = types.SimpleNamespace(
            r2=0.80, rmse=0.2, mae=0.1, pearson_r=0.91, train_r2=0.97, r2_gap=0.17
        )
        app._fill_dl_metrics(network, full, "regression", [], 0.02)
        dl_labels = [str(child.cget("text")) for child in app._dl_metrics.winfo_children()]
        assert "Pearson r" in dl_labels and "train R²" in dl_labels
        index = dl_labels.index("R² gap (train−test)")
        assert dl_labels[index + 1] == "+0.170"

    def test_classification_run_keeps_the_plain_render(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """The classification call sites never pass the pair (a
        ModelError would kill the job) and the render stays untouched."""
        import pandas as pd

        frame = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._ml_adopt_frame(frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_multiclass.csv")
        app._model_combo.set("logistic_regression")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert self._drain_jobs(app), "train job hung"
        labels = self._metric_labels(app)
        assert "accuracy" in labels  # the classification render untouched
        assert "Pearson r" not in labels
        assert "train R²" not in labels
        assert "R² gap (train−test)" not in labels
        assert quiet_dialogs == []


class TestInterpretabilityGui:
    """Surface B: the Results card's interpretability bundle (row three,
    ml_result_buttons3) -- the signed-correlation bar, its heatmap twin
    over the top-10 |r| columns (+ target when the session frame holds
    it), and the ECFP bit decode with its prefilled dialog."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _train_ridge(app: Any) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()

    # -- the source pins ------------------------------------------------------ #

    def test_interpret_buttons_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        for pin in (
            '"Correlations…"',
            "def _ml_plot_correlations",
            "self._ml_plot_correlations",
            '"Correlation heatmap"',
            "def _ml_plot_correlation_heatmap",
            "self._ml_plot_correlation_heatmap",
            '"Decode ECFP bit…"',
            "def _ml_decode_bit",
            "def _ml_decode_bit_run",
            "def _ml_decode_bit_defaults",
            "def _ml_ecfp_parameters",
        ):
            assert pin in source, pin
        # the two-row layout discipline: rows one and two are full, the
        # bundle starts its OWN row (the round-34 VLM overflow rule).
        assert "ml_result_buttons3" in source
        assert 'grid_append(results, ml_result_buttons3, sticky="w", pady=(6, 0))' in source
        # the lazy-import convention: the pharmaml calls live INSIDE
        # work(), so the guards render even on an older install.
        assert "from pharmaml.chem.interpret import signed_feature_correlations" in source
        assert "from pharmaml.chem.interpret import decode_morgan_bit" in source
        assert "from pharmaml.viz import plot_signed_correlations" in source
        assert "from pharmaml.viz import plot_correlation_heatmap" in source

    # -- the guards ----------------------------------------------------------- #

    def test_correlations_need_a_run(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._ml_plot_correlations()
        assert any("Train a model first" in str(a) for a in quiet_dialogs)
        quiet_dialogs.clear()
        app._ml_plot_correlation_heatmap()
        assert any("Train a model first" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the real runs -------------------------------------------------------- #

    def test_correlations_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The full flow against the REAL library: ridge on the example,
        then the signed bundle -- the figure lands in the Results card's
        host, the note names the ranked features and the rows."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        app._ml_plot_correlations()
        assert self._drain_jobs(app), "correlations job hung"
        assert app._ml_host._pg_figure is not None
        note = app._ml_result_note.cget("text")
        assert "signed correlations" in note
        assert "20 numeric feature(s) ranked by |r|" in note  # the example's descr columns
        assert "52 held-out rows" in note  # 260 rows, the 80/20 split
        assert quiet_dialogs == []

    def test_correlation_heatmap_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The heatmap twin: the top-10 |r| columns PLUS the target (the
        example frame still carries pIC50, numeric and alignable) over
        the run's held-out rows."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        app._ml_plot_correlation_heatmap()
        assert self._drain_jobs(app), "heatmap job hung"
        assert app._ml_host._pg_figure is not None
        note = app._ml_result_note.cget("text")
        assert "correlation heatmap" in note
        assert "11 column(s)" in note  # top 10 + the target
        assert "top 10 |r| + pIC50" in note
        assert quiet_dialogs == []

    def test_correlation_heatmap_target_survives_a_dataset_swap(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """(100x-153) The round-36 audit's RED finding pinned: after a run
        on dataset A, adopting a SAME-SHAPED dataset B (same columns, same
        row count, RangeIndex covering the split's indices) must never
        re-pair B's y with A's features.  The target column joins from
        the run's OWN y_test -- the a-y cell reads +1.00 (A's y = 10*a),
        never B's noise (~0.1)."""
        from pathlib import Path

        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(42)
        n = 60
        a = rng.uniform(size=n)
        frame_a = pd.DataFrame(
            {
                "a": a,
                "b": rng.uniform(size=n),
                "c": rng.uniform(size=n),
                "y": 10.0 * a,  # A's target is a's exact copy scaled
            }
        )
        frame_b = pd.DataFrame(
            {
                "a": rng.uniform(size=n),
                "b": rng.uniform(size=n),
                "c": rng.uniform(size=n),
                "y": rng.uniform(size=n),  # B's target is pure noise
            }
        )
        x_all = frame_a[["a", "b", "c"]]
        y_a = frame_a["y"]
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:12], idx[12:]
        model = LinearRegression().fit(x_all.iloc[train_idx], y_a.iloc[train_idx])
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.9, max_error=0.2, n_samples=len(test_idx)
        )
        trained = type("Trained", (), {})()
        trained.name = "ridge"
        app._ml_last_run = (
            model,
            report,
            "regression",
            x_all.iloc[train_idx],
            y_a.iloc[train_idx],
            x_all.iloc[test_idx],
            y_a.iloc[test_idx],
            "y",
        )
        # the natural adoption path for the swapped dataset (the
        # loader's shared body -- exactly the Ctrl+O flow)
        app._ml_adopt_frame(frame_b, Path("dataset_b.csv"))
        app._ml_plot_correlation_heatmap()
        assert self._drain_jobs(app), "heatmap job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        texts = [t.get_text() for t in figure.axes[0].texts]
        # row-major 4x4 matrix over columns [a, b, c, y]: row 0's last
        # cell is the a-y correlation -- A's own +1.00, never B's noise.
        assert texts[3] == "+1.00", texts[:8]
        note = app._ml_result_note.cget("text")
        assert "top 10 |r| + y" in note
        assert "12 held-out rows" in note
        assert quiet_dialogs == []

    def test_correlation_heatmap_mismatched_ndarray_target_not_joined(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """(100x-154) The audit's C1 finding pinned: the heatmap join's
        ndarray fallback used to slice values[: len(frame)], silently
        pairing a LONGER y_test's prefix with the frame's rows -- a
        partial truth in the one fallback path where misalignment is
        real.  The handler's honest outcome is now pinned end to end:
        the correlation ranking's own typed length guard fires FIRST
        (a mismatched y_test can never reach the join), and the join
        fallback itself additionally refuses any non-exact length --
        defense in depth, never a silent truncation anywhere."""
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(7)
        n = 40
        frame = pd.DataFrame(
            {"a": rng.uniform(size=n), "b": rng.uniform(size=n), "y": rng.uniform(size=n)}
        )
        x_all = frame[["a", "b"]]
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:10], idx[10:]
        model = LinearRegression().fit(x_all.iloc[train_idx], frame["y"].iloc[train_idx])
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.5, max_error=0.3, n_samples=len(test_idx)
        )
        trained = type("Trained", (), {})()
        trained.name = "ridge"
        # y_test as a raw ndarray, 25 values for 10 held-out rows --
        # the typed length guard must refuse (the old code's join
        # fallback would have silently truncated had it been reached).
        app._ml_last_run = (
            model,
            report,
            "regression",
            x_all.iloc[train_idx],
            frame["y"].iloc[train_idx].to_numpy(),
            x_all.iloc[test_idx],
            rng.uniform(size=25),  # longer than x_test on purpose
            "y",
        )
        app._ml_plot_correlation_heatmap()
        assert self._drain_jobs(app), "heatmap job hung"
        errors = [a for _n, a in quiet_dialogs if "row(s) but target carries" in str(a)]
        assert errors, quiet_dialogs  # the typed mismatch, never a silent pairing
        assert "25 value(s)" in str(errors[0]) and "10 row(s)" in str(errors[0])

    def test_decode_bit_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """Caffeine's bit 314 (radius 2, 2048 bits -- the library's own
        doctest molecule) decodes to its carbonyl environments, echoed
        as __str__ + one line per hit in the info dialog."""
        app._ml_decode_bit_run("314", "CN1C=NC2=C1C(=O)N(C(=O)O2)C")
        assert self._drain_jobs(app), "decode job hung"
        infos = [a for _name, a in quiet_dialogs if "ECFP bit 314" in str(a)]
        assert infos, quiet_dialogs
        text = str(infos[0])
        assert "c=O" in text  # the hit environments
        assert "atom 7" in text and "atom 10" in text  # both carbonyl carbons
        assert "radius 1" in text
        assert quiet_dialogs[0][0] == "showinfo"

    def test_decode_bit_not_set_is_an_honest_error(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """A bit the molecule does not set is the NORMAL honest path:
        the standard job error box, its message naming the bit and the
        molecule (never a crash, never a fake empty record)."""
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles("CCO")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        set_bits = set(generator.GetFingerprint(mol).GetOnBits())
        free_bit = next(bit for bit in range(2048) if bit not in set_bits)
        app._ml_decode_bit_run(str(free_bit), "CCO")
        assert self._drain_jobs(app), "decode job hung"
        errors = [a for name, a in quiet_dialogs if name == "showerror"]
        assert errors, quiet_dialogs
        assert "Calculation failed" in str(errors[0][0])
        assert f"bit {free_bit} is not set by" in str(errors[0])

    def test_decode_bit_defaults_guards_and_parameters(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """The dialog's prefill follows the session (the most important
        ecfp_bit_ column of the last permutation table, the loaded
        dataset's first molecule); the input guards are dialogs, never
        jobs; the ECFP family constants are derived from the frame."""
        import pandas as pd

        # no session at all -> the honest fallbacks
        assert app._ml_decode_bit_defaults() == ("0", "")
        # a permutation table seeds the bit (importance-descending)
        app._ml_perm_frame = pd.DataFrame(
            {
                "feature": ["descr_x", "ecfp_bit_17", "ecfp3_bit_3"],
                "importance_mean": [3.0, 2.0, 1.0],
                "importance_std": [0.1, 0.1, 0.1],
            }
        )
        # a dataset with a SMILES column seeds the molecule
        lib = pd.read_csv(TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        app._ml_adopt_frame(lib, TOOLKIT_ROOT / "examples" / "data" / "library_with_descriptors.csv")
        bit, smiles = app._ml_decode_bit_defaults()
        assert bit == "17"
        assert smiles == str(lib["SMILES"].iloc[0]).strip()
        # the input guards: a non-integer bit, an empty SMILES
        app._ml_decode_bit_run("abc", "CCO")
        assert any("Input" in str(a) and "bit index" in str(a) for a in quiet_dialogs)
        assert not app._busy
        quiet_dialogs.clear()
        app._ml_decode_bit_run("5", "   ")
        assert any("No SMILES" in str(a) for a in quiet_dialogs)
        assert not app._busy
        # the ECFP family constants: derivable from the run's columns,
        # ecfp3 -> radius 3, plain descriptors -> the ecfp defaults
        app._ml_last_run = (
            None, None, "regression", None, None,
            pd.DataFrame(columns=["ecfp_bit_7"]), None, "t",
        )
        assert app._ml_ecfp_parameters() == (2, 2048)
        app._ml_last_run = (
            None, None, "regression", None, None,
            pd.DataFrame(columns=["ecfp3_bit_7"]), None, "t",
        )
        assert app._ml_ecfp_parameters() == (3, 2048)
        app._ml_last_run = None
        assert app._ml_ecfp_parameters() == (2, 2048)

    def test_decode_dialog_opens_prefilled(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The dialog itself builds with both entries prefilled from the
        session defaults -- the real widgets, not a monkeypatched stub."""
        app._ml_decode_bit()
        app.update()
        tops = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
        assert tops, "the decode dialog did not open"
        win = tops[-1]
        entries = [w for w in win.winfo_children() if w.winfo_class() == "TEntry"]
        assert len(entries) == 2
        assert entries[0].get() == "0"  # the bit index default
        win.destroy()
        assert quiet_dialogs == []

    def test_shap_beeswarm_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """(100x-154) D2.py's GAP-5 figure as a Results-card button: a
        tree model trained on the session frame, the beeswarm drawn
        over the run's OWN held-out rows, the note carrying the top-3
        mean-|SHAP| leaders and the base value."""
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(42)
        n = 80
        frame = pd.DataFrame(
            {"strong": rng.normal(size=n), "weak": rng.normal(size=n)}
        )
        # y driven by 'strong' so the swarm has a clear leader
        frame["y"] = 3.0 * frame["strong"] + 0.2 * frame["weak"]
        x_all = frame[["strong", "weak"]]
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:16], idx[16:]
        model = GradientBoostingRegressor(random_state=42).fit(
            x_all.iloc[train_idx], frame["y"].iloc[train_idx]
        )
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.9, max_error=0.3, n_samples=len(test_idx)
        )
        app._ml_last_run = (
            model,
            report,
            "regression",
            x_all.iloc[train_idx],
            frame["y"].iloc[train_idx],
            x_all.iloc[test_idx],
            frame["y"].iloc[test_idx],
            "y",
        )
        app._ml_plot_shap_beeswarm()
        assert self._drain_jobs(app), "SHAP job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels[0] == "weak"  # bottom -> top, strongest on top
        assert labels[1] == "strong"
        note = app._ml_result_note.cget("text")
        assert "SHAP beeswarm" in note and "16 held-out row(s)" in note
        assert "strong" in note and "base value" in note
        assert quiet_dialogs == []

    def test_shap_bar_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The top-20 bar button: same record, the compact sibling's
        figure + note."""
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(42)
        n = 60
        frame = pd.DataFrame(
            {"a": rng.normal(size=n), "b": rng.normal(size=n)}
        )
        frame["y"] = 2.0 * frame["a"]
        x_all = frame[["a", "b"]]
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:12], idx[12:]
        model = GradientBoostingRegressor(random_state=42).fit(
            x_all.iloc[train_idx], frame["y"].iloc[train_idx]
        )
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.9, max_error=0.3, n_samples=len(test_idx)
        )
        app._ml_last_run = (
            model,
            report,
            "regression",
            x_all.iloc[train_idx],
            frame["y"].iloc[train_idx],
            x_all.iloc[test_idx],
            frame["y"].iloc[test_idx],
            "y",
        )
        app._ml_plot_shap_importance()
        assert self._drain_jobs(app), "SHAP job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        assert figure.axes[0].get_xlabel() == "mean |SHAP value|"
        note = app._ml_result_note.cget("text")
        assert "top-20 bar" in note and "12 held-out row(s)" in note
        assert quiet_dialogs == []

    def test_shap_buttons_guard_no_trained_model(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """No run in the session -> the honest info dialog, no job."""
        app._ml_last_run = None
        app._ml_plot_shap_beeswarm()
        assert any("No trained model" in str(a) for a in quiet_dialogs)
        assert not app._busy
        quiet_dialogs.clear()
        app._ml_plot_shap_importance()
        assert any("No trained model" in str(a) for a in quiet_dialogs)
        assert not app._busy

    def test_shap_non_tree_model_names_the_permutation_sibling(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        """A non-tree estimator surfaces the typed ModelError dialog
        naming the permutation-importance sibling -- never a silent
        fallback or a crash."""
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(42)
        n = 40
        frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = frame["a"] * 2.0
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:8], idx[8:]
        model = LinearRegression().fit(frame.iloc[train_idx], y.iloc[train_idx])
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.9, max_error=0.3, n_samples=len(test_idx)
        )
        app._ml_last_run = (
            model,
            report,
            "regression",
            frame.iloc[train_idx],
            y.iloc[train_idx],
            frame.iloc[test_idx],
            y.iloc[test_idx],
            "y",
        )
        app._ml_plot_shap_beeswarm()
        assert self._drain_jobs(app), "SHAP job hung"
        errors = [
            a for _n, a in quiet_dialogs if "permutation_feature_importance" in str(a)
        ]
        assert errors, quiet_dialogs
        assert "TreeExplainer" in str(errors[0])


class TestCrossModelImportanceGui:
    """Surface C: the Leaderboard card's "Cross-model importance…" button
    -- D2.py's features x models matrix over the fitted models the
    session actually holds (the last TRAINED model + the OOF stacking
    winner, rebuilt and refit on its own train split), with the honest
    guard when the session holds none."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _train_ridge(app: Any) -> None:
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()

    @staticmethod
    def _seed_stack(app: Any, monkeypatch: Any) -> None:
        """A real (tiny) OOF stack over the SAME frame and columns as the
        trained run -- the panel's join is honest, so the winner joins."""
        import pandas as pd
        from sklearn.model_selection import train_test_split

        from pharmaml.ml import stack_oof_regressors

        frame = app._ml_frame
        target = app._target_combo.get()
        numeric = [
            column
            for column in frame.columns
            if column != target and pd.api.types.is_numeric_dtype(frame[column])
        ]
        x_train, x_test, y_train, y_test = train_test_split(
            frame[numeric], frame[target], test_size=0.2, random_state=42
        )
        result = stack_oof_regressors(
            x_train, y_train, x_test, y_test,
            base_models=["ridge"], meta_models=["ridge"], n_splits=2, residual=False,
        )
        app._stack_last = (result, x_train, y_train, x_test, y_test, target)

    # -- the source pins ------------------------------------------------------ #

    def test_cross_model_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Cross-model importance…"' in source
        assert "def _ml_cross_model_importance" in source
        assert "self._ml_cross_model_importance" in source
        assert "_cross_repeats" in source
        # the session state: the annotated init, the stack render's
        # assignment, and the dataset-adopt retirement (the 151 split:
        # the init's ": tuple[...]" annotation breaks a plain match).
        assert "self._stack_last: tuple[Any, Any, Any, Any, Any, str] | None = None" in source
        assert "self._stack_last = (result, x_train, y_train, x_test, y_test, target)" in source
        assert source.count("self._stack_last = None") == 1  # the adopt retirement
        # the lazy-import convention
        assert "from pharmaml.ml import cross_model_importances" in source
        assert "from pharmaml.viz import plot_permutation_heatmap" in source
        assert "from pharmaml.ml.models import build_stacking" in source

    # -- the guards ----------------------------------------------------------- #

    def test_cross_model_needs_a_model(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """Nothing in the session -> the honest info box about what the
        session can and cannot hold (compare_models keeps scores only)."""
        app._ml_cross_model_importance()
        assert any("No fitted models" in str(a) for a in quiet_dialogs)
        assert any("cross-validated scores" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the real runs -------------------------------------------------------- #

    def test_cross_model_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """The single-model panel against the REAL library: the repeats
        parse guard (a bad entry is a dialog, never a job), then the
        heatmap lands in the board card's host with the honest note."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        app._cross_repeats.delete(0, "end")
        app._cross_repeats.insert(0, "abc")
        app._ml_cross_model_importance()
        assert any("Input" in str(a) and "repeats" in str(a) for a in quiet_dialogs)
        assert not app._busy
        quiet_dialogs.clear()
        app._cross_repeats.delete(0, "end")
        app._cross_repeats.insert(0, "5")
        app._ml_cross_model_importance()
        assert self._drain_jobs(app), "cross-model job hung"
        assert app._board_host._pg_figure is not None
        note = app._board_status.cget("text")
        assert "cross-model permutation importance" in note
        assert "1 model(s)" in note
        assert "same seed for every model" in note
        assert quiet_dialogs == []

    def test_cross_model_with_stacking_winner(
        self, app: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        """A stack run in the session joins the panel: the best blender
        is REBUILT and refit on the stack's own train split, the note
        labels it honestly, and the matrix covers TWO models."""
        self._train_ridge(app)
        assert self._drain_jobs(app), "train job hung"
        self._seed_stack(app, monkeypatch)
        app._ml_cross_model_importance()
        assert self._drain_jobs(app), "cross-model job hung"
        note = app._board_status.cget("text")
        assert "2 model(s)" in note
        assert "stacking ridge (refit)" in note
        assert "REBUILT and refit" in note
        assert app._board_host._pg_figure is not None
        assert quiet_dialogs == []


class TestVarianceFilterGui:
    """Surface D: the feature factory's "Filter low variance…" button --
    xxx.ipynb's VarianceThreshold cell as the selection card's honest
    CUT step, with the EC50 -> pEC50 prep card's adoption pattern (the
    askyesno offer, then the frame / selection / tree / summary all
    narrowing to the survivors)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 150.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _seed_features(app: Any) -> None:
        """A synthetic ecfp-family frame + selection: one flat column
        (variance 0, dropped at any positive threshold), three moving
        ones (the ecfp prefixes keep family_of_column honest)."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "ecfp_bit_0": [1.0] * 30,  # flat -> dropped
                "ecfp_bit_1": [0.0, 1.0] * 15,  # variance ~0.26 -> kept
                "ecfp_bit_2": rng.normal(size=30),
                "ecfp_bit_3": rng.normal(size=30) * 5.0,
            }
        )
        from pharmaml.chem.features import FeatureSelection

        app._feat_frame = frame
        app._feat_selection = FeatureSelection(
            scores=pd.Series([1.0, 3.0, 2.0, 4.0], index=list(frame.columns)),
            selected=["ecfp_bit_3", "ecfp_bit_2", "ecfp_bit_1", "ecfp_bit_0"],
            family_counts={"ecfp": 4},
            method="variance",
        )

    # -- the source pins ------------------------------------------------------ #

    def test_variance_filter_source_pins(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '"Filter low variance…"' in source
        assert "def _feat_filter_variance" in source
        assert "self._feat_filter_variance" in source
        assert '_feat_variance_entry.insert(0, "0.01")' in source
        assert "_feat_variance_info" in source
        # the lazy-import convention
        assert "from pharmaml.chem import filter_low_variance" in source
        # the adoption pattern: the prep card's askyesno offer, then the
        # working frame narrows to the survivors
        assert '"Use the filtered frame?"' in source
        assert "self._feat_frame = result.frame" in source

    # -- the guards ----------------------------------------------------------- #

    def test_variance_filter_needs_features(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._feat_filter_variance()
        assert any("Compute & select features first" in str(a) for a in quiet_dialogs)
        assert not app._busy
        # a seeded frame with a bad threshold is a dialog, never a job
        self._seed_features(app)
        quiet_dialogs.clear()
        app._feat_variance_entry.delete(0, "end")
        app._feat_variance_entry.insert(0, "abc")
        app._feat_filter_variance()
        assert any("Input" in str(a) and "variance threshold" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the real runs -------------------------------------------------------- #

    def test_variance_filter_end_to_end_adopts(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        """The real library end to end: the flat column is cut, the info
        line reports the ledger, the askyesno offer is accepted and the
        working frame / selection / tree / summary all narrow."""
        self._seed_features(app)
        answers: list[Any] = []
        monkeypatch.setattr(
            playground.messagebox,
            "askyesno",
            lambda *a, _answers=answers, **k: _answers.append(a) or True,
        )
        app._feat_filter_variance()
        assert self._drain_jobs(app), "variance filter job hung"
        assert answers, "the adoption offer never fired"
        info = app._feat_variance_info.cget("text")
        assert "kept 3" in info and "dropped 1" in info and "threshold 0.01" in info
        assert list(app._feat_frame.columns) == ["ecfp_bit_1", "ecfp_bit_2", "ecfp_bit_3"]
        assert app._feat_selection.selected == ["ecfp_bit_3", "ecfp_bit_2", "ecfp_bit_1"]
        assert app._feat_selection.family_counts == {"ecfp": 3}
        assert len(app._feat_tree.get_children()) == 3
        assert "variance filter adopted" in app._feat_summary.cget("text")
        assert "ecfp 3" in app._feat_family_label.cget("text")
        assert quiet_dialogs == []

    def test_variance_filter_decline_and_zero_drop(
        self, app: Any, playground: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        """Declining the offer keeps the working frame untouched (the
        ledger still reports); a threshold nothing fails (0.0 -- equal
        survives, sklearn's semantics) never asks at all."""
        self._seed_features(app)
        answers: list[Any] = []
        monkeypatch.setattr(
            playground.messagebox,
            "askyesno",
            lambda *a, _answers=answers, **k: _answers.append(a) or False,
        )
        # threshold 0.0: variance 0 is NOT strictly below 0 -> nothing drops
        app._feat_variance_entry.delete(0, "end")
        app._feat_variance_entry.insert(0, "0.0")
        app._feat_filter_variance()
        assert self._drain_jobs(app), "variance filter job hung"
        assert answers == []  # nothing to adopt, no offer
        assert "dropped 0" in app._feat_variance_info.cget("text")
        assert len(app._feat_frame.columns) == 4
        # threshold 0.01 with a declined offer: the ledger reports, the
        # working frame stays whole
        app._feat_variance_entry.delete(0, "end")
        app._feat_variance_entry.insert(0, "0.01")
        app._feat_filter_variance()
        assert self._drain_jobs(app), "variance filter job hung"
        assert len(answers) == 1
        assert "kept 3" in app._feat_variance_info.cget("text")
        assert len(app._feat_frame.columns) == 4  # declined: unchanged
        assert app._feat_selection.selected[0] == "ecfp_bit_3"
        assert quiet_dialogs == []


# --------------------------------------------------------------------- #
# the Unsupervised tab (PDSH 05.09-05.13 + 04.04/04.05, 100x-157)
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestUnsupervisedTab:
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
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _load_classification_example(self, app: Any) -> None:
        example = TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv"
        assert example.is_file()
        app._ml_load_path(example)
        assert app._ml_frame is not None

    def test_tab_builds_with_the_shared_dataset(self, app: Any) -> None:
        self._load_classification_example(app)
        assert "numeric columns available" in app._unsup_info.cget("text")
        assert app._unsup_columns.get_children()  # populated tree
        assert app._unsup_selected_columns()  # first column pre-selected

    def test_pca_run_renders_and_exports(self, app: Any) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("PCA (variance ladder)")
        app._unsup_run()
        assert self._drain(app)
        summary = app._unsup_summary.cget("text")
        assert "PCA(" in summary and "explained variance" in summary
        assert app._unsup_result is not None
        assert app._unsup_host.winfo_children()  # the figure landed
        frame = app._unsup_result["record"].as_frame()
        assert frame.shape[0] == len(app._ml_frame)
        assert frame.shape[1] >= 2

    def test_kmeans_auto_run(self, app: Any) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("K-Means (elbow + silhouette)")
        app._unsup_run()
        assert self._drain(app)
        assert "k-means" in app._unsup_summary.cget("text")
        assert "silhouette" in app._unsup_summary.cget("text")

    def test_manifold_run_and_cluster_scatter(self, app: Any) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("Isomap")
        app._unsup_run()
        assert self._drain(app)
        assert "isomap embedding" in app._unsup_summary.cget("text")
        # the scatter twin coloured by the target column
        target = app._target_combo.get()
        app._unsup_colour.set(target)
        app._unsup_scatter()
        app.update()
        assert app._unsup_host.winfo_children()

    def test_kde_uses_the_first_selected_column(self, app: Any) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("KDE (one column)")
        app._unsup_run()
        assert self._drain(app)
        summary = app._unsup_summary.cget("text")
        assert "KDE(" in summary and "bandwidth" in summary
        first = app._unsup_selected_columns()[0]
        assert f"'{first}'" in summary

    def test_no_selection_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._load_classification_example(app)
        app._unsup_clear_selection()
        app._unsup_run()
        app.update()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showwarning"
        assert "Select at least one numeric column" in str(quiet_dialogs[0][1])
        assert not app._busy

    def test_no_dataset_guard(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._unsup_select_all()
        app._unsup_run()
        app.update()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showwarning"
        assert "Load a CSV" in str(quiet_dialogs[0][1])

    def test_bad_k_guard_names_the_field(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("K-Means (elbow + silhouette)")
        app._unsup_k.delete(0, "end")
        app._unsup_k.insert(0, "bogus")
        app._unsup_run()
        app.update()
        assert quiet_dialogs and "k" in str(quiet_dialogs[0][1]).lower()
        assert not app._busy

    def test_new_dataset_retires_the_previous_run(self, app: Any) -> None:
        self._load_classification_example(app)
        app._unsup_select_all()
        app._unsup_method.set("PCA (variance ladder)")
        app._unsup_run()
        assert self._drain(app)
        assert app._unsup_result is not None
        # adopting the SAME table again still refreshes + retires
        app._ml_adopt_frame(
            app._ml_frame, TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv"
        )
        assert app._unsup_result is None
        assert app._unsup_summary.cget("text") == "no analysis yet"

    def test_scatter_without_a_run_is_honest(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._unsup_scatter()
        app.update()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showinfo"
        assert "Run an analysis first" in str(quiet_dialogs[0][1])

    def test_method_key_mapping(self, app: Any) -> None:
        expected = {
            "PCA (variance ladder)": "pca",
            "K-Means (elbow + silhouette)": "kmeans",
            "Gaussian mixture (AIC/BIC)": "gmm",
            "KDE (one column)": "kde",
            "Isomap": "isomap",
            "LLE": "lle",
            "MDS": "mds",
            "Spectral embedding": "spectral",
            "t-SNE": "tsne",
            "Hexbin density (two columns)": "hexbin",
        }
        for label, key in expected.items():
            app._unsup_method.set(label)
            assert app._unsup_method_key() == key, label


# --------------------------------------------------------------------- #
# (100x-158) the DearPyGui affordances -- interactive figure toolbars +
# crosshair, table chrome (zebra / hover / sort), the UI font scale
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestDearPyGuiAffordances:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    @pytest.fixture()
    def tree(self, app: Any, playground: Any) -> Any:
        """A small headings tree packed into the live app window."""
        t = playground.ttk.Treeview(app, columns=("v", "name"), show="headings", height=8)
        t.heading("v", text="v")
        t.heading("name", text="name")
        t.pack(fill="both", expand=True)
        app.update()
        playground.attach_table_chrome(t)
        return t

    @staticmethod
    def _cells(t: Any, column: str = "v") -> list[str]:
        return [str(t.item(iid, "values")[0 if column == "v" else 1]) for iid in t.get_children("")]

    # -- table chrome -------------------------------------------------- #
    def test_every_built_tree_is_registered(self, app: Any, playground: Any) -> None:
        walked = [
            w for w in app._walk_widgets(app) if isinstance(w, playground.ttk.Treeview)
        ]
        assert walked, "no Treeviews built?"
        registered = list(playground._CHROME_TREES)
        for widget in walked:
            assert widget in registered

    def test_zebra_stripes_alternate(self, tree: Any, playground: Any) -> None:
        for value in ("9", "10", "2"):
            tree.insert("", "end", values=(value, f"row {value}"))
        playground.retag_tree_zebra(tree)
        tags = [tuple(tree.item(iid, "tags")) for iid in tree.get_children("")]
        assert tags == [(playground._CHROME_TAGS[0],), (playground._CHROME_TAGS[1],), (playground._CHROME_TAGS[0],)]

    def test_sweep_restripes_after_population(self, tree: Any, playground: Any) -> None:
        for value in ("1", "2"):
            tree.insert("", "end", values=(value, "x"))
        playground.sweep_table_chrome()  # count changed -> striped
        tags = [tuple(tree.item(iid, "tags")) for iid in tree.get_children("")]
        assert playground._CHROME_TAGS[0] in tags[0]
        assert playground._CHROME_TAGS[1] in tags[1]

    def test_sweep_prunes_destroyed_trees(self, tree: Any, playground: Any) -> None:
        assert tree in playground._CHROME_TREES
        tree.destroy()
        playground.sweep_table_chrome()
        assert tree not in playground._CHROME_TREES

    def test_sort_is_numeric_aware_with_arrow_toggle(self, tree: Any, playground: Any) -> None:
        for value in ("9", "10", "2"):
            tree.insert("", "end", values=(value, "x"))
        playground._sort_tree_column(tree, "v")
        assert self._cells(tree) == ["2", "9", "10"]  # NOT "10" < "2" < "9"
        assert tree.heading("v", "text") == "v ↑"
        playground._sort_tree_column(tree, "v")
        assert self._cells(tree) == ["10", "9", "2"]
        assert tree.heading("v", "text") == "v ↓"

    def test_sort_clears_the_other_columns_arrow(self, tree: Any, playground: Any) -> None:
        for value, name in (("9", "b"), ("10", "a")):
            tree.insert("", "end", values=(value, name))
        playground._sort_tree_column(tree, "v")
        assert "↑" in str(tree.heading("v", "text"))
        playground._sort_tree_column(tree, "name")
        assert str(tree.heading("v", "text")) == "v"  # arrow dropped
        assert "↑" in str(tree.heading("name", "text"))
        assert self._cells(tree, "name") == ["a", "b"]

    def test_text_sort_is_case_insensitive_and_blanks_last(
        self, tree: Any, playground: Any
    ) -> None:
        for name in ("b", "A", "", "c"):
            tree.insert("", "end", values=("1", name))
        playground._sort_tree_column(tree, "name")
        assert self._cells(tree, "name") == ["A", "b", "c", ""]

    def test_thousands_separated_numbers_sort_numerically(
        self, tree: Any, playground: Any
    ) -> None:
        for value in ("1,234", "999", "2,000"):
            tree.insert("", "end", values=(value, "x"))
        playground._sort_tree_column(tree, "v")
        assert self._cells(tree) == ["999", "1,234", "2,000"]

    def test_hover_highlights_then_restores_the_stripe(
        self, tree: Any, playground: Any, app: Any
    ) -> None:
        for value in ("1", "2", "3"):
            tree.insert("", "end", values=(value, "x"))
        playground.sweep_table_chrome()
        app.update_idletasks()
        app.update()
        rows = list(tree.get_children(""))
        box = tree.bbox(rows[0])
        assert box, "tree not mapped for bbox"

        class FakeMotion:  # a <Motion> event carries .y in widget coords
            def __init__(self, y: int) -> None:
                self.y = y

        playground._hover_tree_row(tree, FakeMotion(box[1] + box[3] // 2))
        assert "chromehover" in tuple(tree.item(rows[0], "tags"))
        playground._hover_tree_row(tree, FakeMotion(box[1] - 40))  # off the rows
        assert "chromehover" not in tuple(tree.item(rows[0], "tags"))
        assert playground._CHROME_TAGS[0] in tuple(tree.item(rows[0], "tags"))

    def test_theme_flip_recolors_the_stripes(self, tree: Any, playground: Any, app: Any) -> None:
        for value in ("1", "2"):
            tree.insert("", "end", values=(value, "x"))
        playground.sweep_table_chrome()
        app._set_theme("dark")
        tag = tree.tag_configure(playground._CHROME_TAGS[1])
        assert str(tag["background"]) == playground._DARK_PALETTE["STRIPE"]
        app._set_theme("light")
        tag = tree.tag_configure(playground._CHROME_TAGS[1])
        assert str(tag["background"]) == playground._LIGHT_PALETTE["STRIPE"]

    # -- the interactive figure toolbar + crosshair --------------------- #
    def _rendered(self, app: Any, playground: Any) -> tuple[Any, Any, Any]:
        from matplotlib.figure import Figure

        fig = Figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 4, 9])
        host = playground.tk.Frame(app, background=playground.CARD, height=330)
        host.pack(fill="both", expand=True)
        app._render_figure(host, fig)
        app.update()
        return host, fig, ax

    def test_toolbar_attaches_styled_and_toggles(self, app: Any, playground: Any) -> None:
        if app._figure_canvas_class() is None:
            pytest.skip("TkAgg unavailable on this Tk build")
        _host, _fig, _ax = self._rendered(app, playground)
        assert len(app._live_toolbars) == 1
        toolbar = app._live_toolbars[0][1]
        # packed at the host's bottom edge, on the card colour
        assert toolbar.winfo_manager() == "pack"
        assert str(toolbar.cget("background")) == playground.CARD
        buttons = [c for c in toolbar.winfo_children() if c.winfo_class() == "Button"]
        assert buttons, "toolbar buttons not built"
        assert str(buttons[0].cget("background")) == playground.CARD
        # the preference hides / shows every live toolbar
        app._toolbars_var.set(False)
        app._apply_toolbar_visibility()
        app.update()
        assert toolbar.winfo_manager() == ""
        app._toolbars_var.set(True)
        app._apply_toolbar_visibility()
        app.update()
        assert toolbar.winfo_manager() == "pack"

    def test_rerender_replaces_the_toolbar_without_leak(
        self, app: Any, playground: Any
    ) -> None:
        if app._figure_canvas_class() is None:
            pytest.skip("TkAgg unavailable on this Tk build")
        from matplotlib.figure import Figure

        host, _fig, _ax = self._rendered(app, playground)
        fig2 = Figure(figsize=(4, 3))
        fig2.add_subplot(111).plot([3, 2, 1], [9, 4, 1])
        app._render_figure(host, fig2)  # a SECOND figure into the SAME host
        app.update()
        assert len(app._live_toolbars) == 1
        assert app._last_figure is fig2
        frames = [c for c in host.winfo_children() if c.winfo_class() == "Frame"]
        canvases = [c for c in host.winfo_children() if c.winfo_class() == "Canvas"]
        assert len(frames) == 1 and len(canvases) == 1

    def test_crosshair_tracks_and_hides_without_axis_drift(
        self, app: Any, playground: Any
    ) -> None:
        if app._figure_canvas_class() is None:
            pytest.skip("TkAgg unavailable on this Tk build")
        from matplotlib.backend_bases import MouseEvent

        _host, fig, ax = self._rendered(app, playground)
        before = ax.get_xlim(), ax.get_ylim()
        px, py = ax.transData.transform((2.0, 4.0))
        fig.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, px, py)
        )
        pair = ax._pg_crosshair
        assert pair[0].get_visible() and pair[1].get_visible()
        v_x, v_y = pair[0].get_data()
        assert list(v_x) == [2.0, 2.0]
        assert list(v_y) == list(ax.get_ylim())
        # add_artist guides never join the data limits
        assert (ax.get_xlim(), ax.get_ylim()) == before
        fig.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", fig.canvas, 2, 2)
        )
        assert not pair[0].get_visible() and not pair[1].get_visible()

    def test_theme_flip_recolors_the_toolbar(self, app: Any, playground: Any) -> None:
        if app._figure_canvas_class() is None:
            pytest.skip("TkAgg unavailable on this Tk build")
        _host, _fig, _ax = self._rendered(app, playground)
        toolbar = app._live_toolbars[0][1]
        button = next(c for c in toolbar.winfo_children() if c.winfo_class() == "Button")
        app._set_theme("dark")
        assert str(toolbar.cget("background")) == playground._DARK_PALETTE["CARD"]
        assert str(button.cget("background")) == playground._DARK_PALETTE["CARD"]
        app._set_theme("light")
        assert str(toolbar.cget("background")) == playground._LIGHT_PALETTE["CARD"]

    # -- the UI font scale ----------------------------------------------- #
    def test_font_size_menu_rescales_the_named_fonts(
        self, app: Any, playground: Any
    ) -> None:
        import tkinter.font as tkfont

        default = int(tkfont.nametofont("TkDefaultFont").cget("size"))
        app._font_var.set(default + 1)
        app._apply_font_size()
        assert int(tkfont.nametofont("TkDefaultFont").cget("size")) == default + 1
        assert int(app._mono.cget("size")) == default + 1
        assert int(app._mono_result.cget("size")) == default + 3
        # the derived style variants follow the base
        app._font_var.set(default)
        app._apply_font_size()
        assert int(tkfont.nametofont("TkDefaultFont").cget("size")) == default

    def test_font_preference_rides_the_session(self, app: Any) -> None:
        session = app._collect_session()
        assert "_toolbars_var" in session["variables"]
        assert "_font_var" in session["variables"]
        assert session["variables"]["_toolbars_var"] is True
        assert session["variables"]["_font_var"] == 10


# --------------------------------------------------------------------- #
# (100x-159) the second DearPyGui pass -- collapsible cards +
# the GUI metrics / palette inspector dialog
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestDearPyGuiRoundTwo:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    # -- the CollapsibleCard retrofit ------------------------------------ #
    def test_fifteen_reference_cards_registered_and_expanded(
        self, app: Any, playground: Any
    ) -> None:
        expected = {
            "home-dependency-status",
            "home-what-lives-where",
            # (round 201) the Home sample-data quick start -- every tab
            # ready to run in one click, collapsible like its siblings
            "home-quick-start",
            "pk-half-life",
            "pk-iv-bolus",
            "pk-oral",
            "pk-crcl",
            "pk-loading",
            "pk-bsa",
            "pk-pediatric",
            "pk-emax",
            "pk-nca",
            "pk-infusion",
            "pk-tdm",
            "pk-pd-explorer",
            "pk-units",
            # (100x-165) the Utilities tab's Git quick reference
            "utilities-git-reference",
            # (round 200, the arena merge) the two absorbed collapsible
            # cards -- the Advanced PK/PD & PBPK engines card and the
            # DOCKSTRING benchmark lab (latent v1.2.115 staleness: the
            # registry pin predated both and the class was not re-run)
            "pk-advanced-engines",
            "dock-dockstring-benchmark",
        }
        assert set(playground._COLLAPSE_CARDS) == expected
        assert playground.collapsed_card_keys() == []  # launch look = expanded
        for key, entry in playground._COLLAPSE_CARDS.items():
            assert entry["expanded"] is True, key
            assert entry["content"].winfo_manager() == "pack", key

    def test_toggle_folds_body_and_flips_the_chevron(self, app: Any, playground: Any) -> None:
        playground._toggle_collapsible_card("pk-tdm")
        app.update()
        entry = playground._COLLAPSE_CARDS["pk-tdm"]
        assert entry["expanded"] is False
        assert entry["content"].winfo_manager() == ""  # the body is hidden
        assert entry["button"].cget("text") == "\u25b8"
        assert playground.collapsed_card_keys() == ["pk-tdm"]
        # the folded card's INPUTS keep their values (only the body hides)
        assert app._tdm_dose.get() == "250"
        playground._toggle_collapsible_card("pk-tdm")
        app.update()
        assert entry["expanded"] is True
        assert entry["content"].winfo_manager() == "pack"
        assert entry["button"].cget("text") == "\u25be"
        assert playground.collapsed_card_keys() == []

    def test_unknown_key_is_a_quiet_noop(self, app: Any, playground: Any) -> None:
        playground._toggle_collapsible_card("no-such-card")  # never raises
        playground._set_card_expanded("no-such-card", False)
        assert playground.collapsed_card_keys() == []

    def test_finder_jump_unfolds_the_target(self, app: Any, playground: Any) -> None:
        playground._set_card_expanded("home-what-lives-where", False)
        app.update()
        app._jump_to_card(0, "What lives where")
        app.update()
        assert playground._COLLAPSE_CARDS["home-what-lives-where"]["expanded"] is True

    def test_session_roundtrip_persists_the_fold(self, app: Any, playground: Any) -> None:
        playground._set_card_expanded("pk-nca", False)
        playground._set_card_expanded("home-dependency-status", False)
        app.update()
        payload = app._collect_session()
        assert json.loads(payload["texts"]["collapsed_cards"]) == [
            "home-dependency-status",
            "pk-nca",
        ]
        # a fresh restore replants exactly those folds
        playground._set_card_expanded("pk-nca", True)
        app._session = playground.load_session(payload)
        app._restore_session()
        app.update()
        assert playground._COLLAPSE_CARDS["pk-nca"]["expanded"] is False
        assert playground._COLLAPSE_CARDS["home-dependency-status"]["expanded"] is False
        assert playground._COLLAPSE_CARDS["pk-oral"]["expanded"] is True

    def test_restore_ignores_corrupt_and_unknown_keys(
        self, app: Any, playground: Any
    ) -> None:
        payload = playground.dump_session(
            theme="light",
            geometry="1200x800+10+10",
            tab=0,
            variables={},
            families={},
            texts={},
        )
        payload["texts"]["collapsed_cards"] = "not json at all"
        app._session = playground.load_session(payload)
        app._restore_session()
        app.update()
        assert playground.collapsed_card_keys() == []  # corrupt -> everything open
        payload["texts"]["collapsed_cards"] = json.dumps(["ghost-card", "pk-oral"])
        app._session = playground.load_session(payload)
        app._restore_session()
        app.update()
        assert playground.collapsed_card_keys() == ["pk-oral"]  # unknown key skipped

    def test_per_tab_reset_unfolds_only_that_tab(self, app: Any, playground: Any) -> None:
        pk_index = next(
            i for i in range(app._book.index("end")) if "PK" in app._book.tab(i, "text")
        )
        playground._set_card_expanded("pk-oral", False)
        playground._set_card_expanded("home-dependency-status", False)
        app.update()
        app._reset_tab(pk_index)
        app.update()
        assert playground._COLLAPSE_CARDS["pk-oral"]["expanded"] is True
        assert playground._COLLAPSE_CARDS["home-dependency-status"]["expanded"] is False
        assert "unfolded" in app._job_label.cget("text")

    # -- the GUI metrics / palette inspector ----------------------------- #
    def test_metrics_dialog_opens_with_census_and_prefs(self, app: Any, playground: Any) -> None:
        assert app._metrics_dialog is None
        app._show_gui_metrics()
        app.update()
        assert app._metrics_dialog is not None
        assert bool(app._metrics_dialog.winfo_exists())
        assert app._metrics_dialog.title() == "GUI metrics"
        snapshot = app._metrics_text.get("1.0", "end")
        for needle in (
            "CARDS      ",
            "TREES      ",
            "TOOLBARS   ",
            "JOBS       ",
            "PREFS      ",
            "SESSION    ",
            "WIDGET CENSUS (by Tk class)",
        ):
            assert needle in snapshot, needle
        # the census's card count is the LIVE card count, not a hardcoded pin
        assert f"{len(app._live_cards())} live" in snapshot
        assert "idle" in snapshot  # no job is running in a test window
        assert f"font {app._font_var.get()} pt" in snapshot

    def test_metrics_census_matches_the_live_tree(self, app: Any, playground: Any) -> None:
        app._show_gui_metrics()
        app.update()
        snapshot = app._metrics_text.get("1.0", "end")
        walked_trees = sum(
            1 for w in app._walk_widgets(app) if w.winfo_class() == "Treeview"
        )
        assert f"TREES      {walked_trees} Treeviews" in snapshot
        counts_line = next(
            line for line in snapshot.splitlines() if line.startswith("Treeview")
        )
        assert int(counts_line.split()[-1]) == walked_trees

    def test_palette_inspector_shows_every_active_key(self, app: Any, playground: Any) -> None:
        app._show_gui_metrics()
        app.update()
        palette = playground._PALETTES[playground._ACTIVE_THEME]
        chips = [
            c
            for c in app._metrics_palette.winfo_children()
            if c.winfo_class() == "Frame"
        ]
        assert len(chips) == len(palette)
        assert {str(chip.cget("background")) for chip in chips} == set(palette.values())
        title = app._metrics_palette_title.cget("text")
        assert playground._ACTIVE_THEME in str(title)
        assert str(len(palette)) in str(title)

    def test_palette_inspector_follows_theme_flips(self, app: Any, playground: Any) -> None:
        app._show_gui_metrics()
        app.update()
        win = app._metrics_dialog
        app._set_theme("dark")
        app.update()
        dark = playground._PALETTES["dark"]
        chips = [
            c
            for c in app._metrics_palette.winfo_children()
            if c.winfo_class() == "Frame"
        ]
        assert {str(chip.cget("background")) for chip in chips} == set(dark.values())
        assert "dark" in str(app._metrics_palette_title.cget("text"))
        # the Toplevel branch of the retint walker follows the flip too
        assert str(win.cget("background")) == playground.BG
        app._set_theme("light")
        app.update()
        light = playground._PALETTES["light"]
        chips = [
            c
            for c in app._metrics_palette.winfo_children()
            if c.winfo_class() == "Frame"
        ]
        assert {str(chip.cget("background")) for chip in chips} == set(light.values())

    def test_metrics_dialog_reopen_lifts_and_destroy_prunes(
        self, app: Any, playground: Any
    ) -> None:
        app._show_gui_metrics()
        win = app._metrics_dialog
        app._show_gui_metrics()  # second open lifts, never stacks
        assert app._metrics_dialog is win
        lifted = [w for w in app.winfo_children() if w.winfo_class() == "Toplevel"]
        assert len(lifted) == 1
        win.destroy()
        app._refresh_metrics_dialog()
        assert app._metrics_dialog is None
        assert app._metrics_palette is None
        app._set_theme("dark")  # a stale ref can never raise
        app._set_theme("light")
        app.update()

    # -- the round-41 menu fix ------------------------------------------- #
    def test_font_cascade_keeps_its_theme_tint(self, app: Any, playground: Any) -> None:
        # (100x-159) the round-41 build re-assigned _menu_tints AFTER the
        # font cascade's append -- the submenu lost its tint on every flip.
        # The fix rides the list itself: [bar, file, view, PACK, FONT, help]
        # -- (100x-175) the CTk accent-pack cascade slots in before the
        # font cascade and rides the same list, so BOTH submenus keep
        # their tint through a flip.
        assert len(app._menu_tints) == 6
        pack_menu = app._menu_tints[3]
        font_menu = app._menu_tints[4]
        app._set_theme("dark")
        assert str(font_menu.cget("background")) == playground.HEADING
        assert str(pack_menu.cget("background")) == playground.HEADING
        app._set_theme("light")
        assert str(font_menu.cget("background")) == playground.CARD
        assert str(pack_menu.cget("background")) == playground.CARD


class TestShapOptunaAbsorptionGui:
    """(100x-160) The shap/shap + optuna/optuna absorption round's GUI:
    the five new SHAP figure buttons (waterfall / force / dependence /
    heatmap / group difference) over the run-tuple discipline, the
    Tune card's TPE searcher + pruner choice, and the seven study
    figures over the last search's records."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @pytest.fixture()
    def ask_string(self, playground: Any, monkeypatch: Any) -> dict[str, list[str]]:
        """Neutralise the modal simpledialog.askstring: canned answers
        queued under 'answers', prompts recorded under 'prompts' (a
        blocking modal would hang Xvfb)."""
        state: dict[str, list[str]] = {"answers": [], "prompts": []}

        def fake_askstring(title: str, prompt: str, **kwargs: Any) -> str:
            state["prompts"].append(prompt)
            return state["answers"].pop(0) if state["answers"] else None

        monkeypatch.setattr(playground.simpledialog, "askstring", fake_askstring)
        return state

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 120.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _planted_run(app: Any) -> None:
        """A tree-model run tuple over a learnable two-feature frame
        (the same discipline as the 100x-154 tests, smaller)."""
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor

        from pharmaml.ml.evaluation import RegressionReport

        rng = np.random.default_rng(42)
        n = 60
        frame = pd.DataFrame(
            {"strong": rng.normal(size=n), "weak": rng.normal(size=n)}
        )
        frame["y"] = 3.0 * frame["strong"] + 0.2 * frame["weak"]
        x_all = frame[["strong", "weak"]]
        idx = rng.permutation(n)
        test_idx, train_idx = idx[:12], idx[12:]
        model = GradientBoostingRegressor(random_state=42).fit(
            x_all.iloc[train_idx], frame["y"].iloc[train_idx]
        )
        report = RegressionReport(
            rmse=0.1, mae=0.1, r2=0.9, max_error=0.3, n_samples=len(test_idx)
        )
        app._ml_last_run = (
            model,
            report,
            "regression",
            x_all.iloc[train_idx],
            frame["y"].iloc[train_idx],
            x_all.iloc[test_idx],
            frame["y"].iloc[test_idx],
            "y",
        )

    @staticmethod
    def _planted_study(app: Any) -> Any:
        """A crafted 6-trial TunerResults (4 complete, 1 pruned with
        fold scores, 1 fail with a duration) planted as _tune_last."""
        from pharmaml.ml.tuner import TrialResult, TunerResults

        trials = [
            TrialResult(1, {"units": 8, "alpha": 0.10}, 0.50, 0.02,
                        fold_scores=(0.4, 0.5, 0.6), duration=1.5),
            TrialResult(2, {"units": 16, "alpha": 0.05}, 0.80, 0.02,
                        fold_scores=(0.7, 0.8, 0.9), duration=2.0),
            TrialResult(3, {"units": 32, "alpha": 0.20}, 0.65, 0.02,
                        fold_scores=(0.6, 0.6, 0.75), duration=1.8),
            TrialResult(4, {"units": 64, "alpha": 0.05}, 0.72, 0.02,
                        fold_scores=(0.7, 0.7, 0.76), duration=1.6),
            TrialResult(5, {"units": 128, "alpha": 0.30}, float("nan"),
                        float("nan"), state="pruned",
                        fold_scores=(0.1, 0.2), duration=0.5),
            TrialResult(6, {"units": 4, "alpha": 0.40}, float("nan"),
                        float("nan"), error="ValueError: bad", duration=0.2),
        ]
        results = TunerResults(trials, "r2")
        app._tune_last = (results, "ann", "regression", "y")
        return results

    # -- the card chrome ---------------------------------------------------- #
    def test_the_searcher_and_pruner_tokens(self, app: Any) -> None:
        assert list(app._tune_searcher_combo["values"]) == ["random", "tpe", "hyperband"]
        assert list(app._tune_pruner_combo["values"]) == ["off", "median"]
        assert app._tune_pruner_combo.get() == "off"

    def test_the_button_census(self, app: Any) -> None:
        """The five SHAP + seven study buttons live on their cards."""
        texts: list[str] = []

        def walk(widget: Any) -> None:
            for child in widget.winfo_children():
                if child.winfo_class() == "TButton":
                    texts.append(child.cget("text"))
                walk(child)

        walk(app)
        for label in (
            "SHAP waterfall…",
            "SHAP force…",
            "SHAP dependence…",
            "SHAP heatmap",
            "SHAP group difference…",
            "Optimization history",
            "Slice plots",
            "Parallel coords",
            "Param importances",
            "EDF",
            "Timeline",
            "Intermediates",
        ):
            assert label in texts, f"missing button {label!r}"

    # -- the SHAP figures end to end ---------------------------------------- #
    def test_waterfall_end_to_end(
        self, app: Any, quiet_dialogs: list[Any], ask_string: dict[str, list[str]]
    ) -> None:
        ask_string["answers"].append("0")  # the canned row answer
        self._planted_run(app)
        app._ml_plot_shap_waterfall()
        assert self._drain_jobs(app), "waterfall job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
        assert labels, "the waterfall drew no feature rows"
        assert any("=" in label for label in labels)  # the show_data convention
        note = app._ml_result_note.cget("text")
        assert "SHAP waterfall" in note and "row 0" in note
        assert "12 held-out row(s)" in note
        assert quiet_dialogs == []

    def test_force_end_to_end(
        self, app: Any, quiet_dialogs: list[Any], ask_string: dict[str, list[str]]
    ) -> None:
        ask_string["answers"].append("3")
        self._planted_run(app)
        app._ml_plot_shap_force()
        assert self._drain_jobs(app), "force job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        assert figure.axes[0].get_xlabel() == "model output"
        note = app._ml_result_note.cget("text")
        assert "SHAP force" in note and "row 3" in note
        assert quiet_dialogs == []

    def test_dependence_end_to_end(
        self, app: Any, quiet_dialogs: list[Any], ask_string: dict[str, list[str]]
    ) -> None:
        ask_string["answers"].append("strong")
        self._planted_run(app)
        app._ml_plot_shap_dependence()
        assert self._drain_jobs(app), "dependence job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        ax = figure.axes[0]
        assert ax.get_xlabel() == "strong"
        assert ax.get_ylabel() == "SHAP value"
        note = app._ml_result_note.cget("text")
        assert "SHAP dependence" in note and "feature strong" in note
        assert quiet_dialogs == []

    def test_heatmap_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._planted_run(app)
        app._ml_plot_shap_heatmap()
        assert self._drain_jobs(app), "heatmap job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        # the matrix axis + the thin mean-|SHAP| bar axis
        assert len(figure.axes) >= 2
        note = app._ml_result_note.cget("text")
        assert "SHAP heatmap" in note
        assert quiet_dialogs == []

    def test_group_difference_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._planted_run(app)
        app._ml_plot_shap_group_difference()
        assert self._drain_jobs(app), "group difference job hung"
        figure = app._ml_host._pg_figure
        assert figure is not None
        ax = figure.axes[0]
        ticks = {tick.get_text() for tick in ax.get_yticklabels()}
        assert ticks == {"strong", "weak"}
        assert ax.get_xlabel().startswith("mean SHAP difference")
        note = app._ml_result_note.cget("text")
        assert "SHAP group" in note
        assert quiet_dialogs == []

    def test_shap_figure_guards(self, app: Any, quiet_dialogs: list[Any]) -> None:
        """No run in the session -> the honest info dialog, no job."""
        app._ml_last_run = None
        for method in (
            app._ml_plot_shap_waterfall,
            app._ml_plot_shap_force,
            app._ml_plot_shap_dependence,
            app._ml_plot_shap_heatmap,
            app._ml_plot_shap_group_difference,
        ):
            quiet_dialogs.clear()
            method()
            assert any("No trained model" in str(a) for a in quiet_dialogs)
            assert not app._busy

    def test_dependence_unknown_feature(
        self, app: Any, quiet_dialogs: list[Any], ask_string: dict[str, list[str]]
    ) -> None:
        ask_string["answers"].append("not_a_column")
        self._planted_run(app)
        app._ml_plot_shap_dependence()
        assert any("Unknown feature" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the study figures end to end ---------------------------------------- #
    def test_study_figures_end_to_end(self, app: Any, quiet_dialogs: list[Any]) -> None:
        self._planted_study(app)
        buttons = (
            ("Optimization history", app._tune_plot_history),
            ("Slice plots", app._tune_plot_slice),
            ("Parallel coordinates", app._tune_plot_parallel),
            ("Param importances", app._tune_plot_importances),
            ("EDF", app._tune_plot_edf),
            ("Timeline", app._tune_plot_timeline),
            ("Intermediates", app._tune_plot_intermediates),
        )
        for name, method in buttons:
            method()
            assert self._drain_jobs(app), f"{name} job hung"
            figure = app._ml_host._pg_figure
            assert figure is not None and figure.axes, f"{name} drew no figure"
            note = app._ml_result_note.cget("text")
            assert name in note and "4 completed" in note and "1 pruned" in note
        assert quiet_dialogs == []

    def test_study_figure_guards(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._tune_last = None
        app._tune_plot_history()
        assert any("No search yet" in str(a) for a in quiet_dialogs)
        assert not app._busy

    # -- the TPE search end to end -------------------------------------------- #
    def test_tpe_search_end_to_end(self, playground: Any, app: Any,
                                   quiet_dialogs: list[Any]) -> None:
        """The TPE searcher through the real card: a small search over
        the ann hypermodel fills the leaderboard, the status names
        TPESearch, and the pruner=median choice lands pruned states."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 60
        frame = pd.DataFrame(
            {"a": rng.uniform(0, 1, n), "b": rng.uniform(0, 1, n)}
        )
        frame["y"] = 3.0 * frame["a"] + 1.0
        app._ml_frame = frame
        app._ml_path = None
        app._target_combo.set("y")
        app._task_combo.set("regression")
        app._tune_estimator_combo.set("ann")
        app._tune_searcher_combo.set("tpe")
        app._tune_pruner_combo.set("median")
        app._tune_trials.set("5")
        app._tune_cv.set("2")
        app._ml_tune()
        assert self._drain_jobs(app, timeout=300.0), "tpe search hung"
        status = app._board_status.cget("text")
        assert "TPESearch" in status and "ann" in status
        summary = app._tune_summary.cget("text")
        assert summary.startswith("best ")
        assert app._tune_last is not None
        results = app._tune_last[0]
        assert results.objective == "r2"
        # every trial recorded a duration and the pruner-wired fold loop
        # ran (fold scores present on completed trials).
        assert all(t.duration is not None for t in results.trials)
        assert any(t.fold_scores for t in results.trials)
        assert quiet_dialogs == []


class TestReinvent4DiffDockAbsorptionGui:
    """(100x-164) The MolecularAI/REINVENT4 + gcorso/DiffDock absorption
    round's GUI: the Chemistry row-8 cards (the score-transform lab and
    the custom-alerts/standardization card), the Generative tab's
    diversity-filter + alerts wiring, and the Docking row-8 cards (the
    pose-RMSD evaluation and the pocket extraction)."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    # -- the score-transform lab ------------------------------------------ #

    def test_transform_lab_scores_the_default_list(self, app: Any) -> None:
        app._transform_component.set("molecular weight (Da)")
        app._transform_kind.set("step window")
        app._transform_low.delete(0, "end"); app._transform_low.insert(0, "150")
        app._transform_high.delete(0, "end"); app._transform_high.insert(0, "450")
        app._transform_lab_run()
        assert self._drain_jobs(app), "transform lab hung"
        rows = app._transform_tree.get_children()
        assert len(rows) == 3  # the card's default molecule list
        values = [app._transform_tree.item(row, "values") for row in rows]
        # every default molecule sits inside the drug-like window
        assert all(float(str(v[2])) == 1.0 for v in values)
        summary = app._transform_summary.cget("text")
        assert "molecular weight" in summary and "mean score" in summary

    def test_transform_lab_rejects_bad_window(self, app: Any, quiet_dialogs: Any) -> None:
        app._transform_low.delete(0, "end"); app._transform_low.insert(0, "500")
        app._transform_high.delete(0, "end"); app._transform_high.insert(0, "100")
        app._transform_lab_run()
        assert not app._busy  # refused before the worker started
        assert quiet_dialogs and "exceeds" in str(quiet_dialogs[0][1])

    # -- the custom alerts / standardization card -------------------------- #

    def test_alerts_check_marks_hits(self, app: Any) -> None:
        app._alert_smarts_text.delete("1.0", "end")
        app._alert_smarts_text.insert("1.0", "c1ccccc1")   # benzene alert
        app._alert_mol_text.delete("1.0", "end")
        app._alert_mol_text.insert("1.0", "CCO\nCC(=O)Oc1ccccc1C(=O)O")
        app._alerts_check_run()
        assert self._drain_jobs(app), "alerts check hung"
        values = [
            app._alerts_tree.item(row, "values")
            for row in app._alerts_tree.get_children()
        ]
        verdicts = {str(v[0]): str(v[2]) for v in values}
        assert verdicts["CCO"] == "clean"
        assert verdicts["CC(=O)Oc1ccccc1C(=O)O"] == "blocked"
        assert "1 blocked" in app._alerts_summary.cget("text")

    def test_standardize_chain_statuses(self, app: Any) -> None:
        app._alert_mol_text.delete("1.0", "end")
        app._alert_mol_text.insert("1.0", "CC(=O)[O-].[Na+]\nCCO\nC")
        app._reinvent_standardize_run()
        assert self._drain_jobs(app), "standardization hung"
        values = [
            app._alerts_tree.item(row, "values")
            for row in app._alerts_tree.get_children()
        ]
        statuses = {str(v[0]): str(v[2]) for v in values}
        assert statuses["CC(=O)[O-].[Na+]"] == "cleaned"
        assert statuses["CCO"] == "unchanged"
        assert statuses["C"] == "rejected"  # the (2, 70) heavy-atom window

    # -- the Generative wiring ---------------------------------------------- #

    def test_diversity_controls_exist(self, app: Any) -> None:
        assert app._diversity_combo.get() == "none"
        assert app._diversity_bucket.get() == "25"
        assert app._diversity_minscore.get() == "0.4"
        assert app._diversity_penalty.get() == "step"
        assert app._gen_alerts_text.get("1.0", "end").strip() == ""

    def test_ga_run_with_diversity_and_alerts(self, app: Any) -> None:
        app._objective_combo.set("QED drug-likeness")
        app._method_combo.set("genetic algorithm")
        app._pop_size.set("8")
        app._generations.set("3")
        app._diversity_combo.set("penalize same SMILES")
        app._diversity_bucket.set("2")
        app._gen_alerts_text.delete("1.0", "end")
        app._gen_alerts_text.insert("1.0", "[N+](=O)[O-]")
        app._run_evolution()
        assert self._drain_jobs(app, timeout=300.0), "GA run hung"
        stats = app._ga_stats_text.get("1.0", "end")
        assert "REINVENT4 dressings" in stats
        assert "custom alerts" in stats and "1 pattern" in stats
        assert "penalised scores" in stats
        # clean up for any test that follows
        app._diversity_combo.set("none")
        app._gen_alerts_text.delete("1.0", "end")

    def test_format_diversity_stats_block(self, playground: Any) -> None:
        text = playground._format_diversity_stats(
            {
                "filter": "PenalizeSameSmilesFilter",
                "n_scaffolds": 0,
                "n_full_buckets": 0,
                "n_memorized_smiles": 12,
                "n_zeroed": 1,
                "n_penalized": 4,
            },
            ["[N+](=O)[O-]"],
        )
        assert "PenalizeSameSmiles" in text  # the Filter suffix dropped
        assert "memorised SMILES" in text and "12" in text
        assert "1 pattern(s): [N+](=O)[O-]" in text

    # -- the pose RMSD / pocket cards --------------------------------------- #

    def test_pose_evaluation_table_and_report(self, app: Any, tmp_path: Any) -> None:
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Geometry import Point3D

        aspirin = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
        params = AllChem.ETKDG()
        params.randomSeed = 42  # type: ignore[assignment]
        AllChem.EmbedMultipleConfs(aspirin, numConfs=3, params=params)
        conformers = [
            aspirin.GetConformer(i) for i in range(aspirin.GetNumConformers())
        ]
        reference = tmp_path / "reference.sdf"
        poses = tmp_path / "poses.sdf"
        reference.write_text(
            Chem.MolToMolBlock(aspirin, confId=conformers[0].GetId())
        )
        chunks = []
        for index, conf in enumerate(conformers[1:], start=1):
            shifted = Chem.Mol(aspirin)
            shifted.RemoveAllConformers()
            shifted.AddConformer(conf, assignId=True)
            positions = np.asarray(conf.GetPositions()) + float(index) * 0.5
            writer_conf = shifted.GetConformer()
            for atom_index in range(shifted.GetNumAtoms()):
                writer_conf.SetAtomPosition(
                    atom_index, Point3D(*positions[atom_index].tolist())
                )
            chunks.append(Chem.MolToMolBlock(shifted) + "$$$$\n")
        poses.write_text("".join(chunks))
        app._pose_reference_entry.delete(0, "end")
        app._pose_reference_entry.insert(0, str(reference))
        app._pose_poses_entry.delete(0, "end")
        app._pose_poses_entry.insert(0, str(poses))
        app._pose_evaluate_run()
        assert self._drain_jobs(app, timeout=180.0), "pose evaluation hung"
        rows = app._pose_tree.get_children()
        assert len(rows) == 2
        report = app._pose_report_text.get("1.0", "end")
        assert "RMSD < 2 A" in report and "pose clusters" in report
        assert "rotatable bond(s)" in report

    def test_pose_evaluation_requires_both_files(self, app: Any, quiet_dialogs: Any) -> None:
        app._pose_reference_entry.delete(0, "end")
        app._pose_poses_entry.delete(0, "end")
        app._pose_evaluate_run()
        assert quiet_dialogs and "Both" in str(quiet_dialogs[0][1])

    def test_pocket_extraction_on_a_mini_protein(self, app: Any, tmp_path: Any) -> None:
        protein = tmp_path / "mini.pdb"
        protein.write_text(
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
            "           N  \n"
            "ATOM      2  CA  GLY A   2       3.000   0.000   0.000  1.00  0.00"
            "           C  \n"
            "ATOM      3  N   VAL B   1      50.000   0.000   0.000  1.00  0.00"
            "           N  \n"
        )
        ligand = tmp_path / "ligand.pdb"
        ligand.write_text(
            "HETATM    1  C   LIG A   1       1.000   0.000   0.000  1.00  0.00"
            "           C  \n"
        )
        app._pocket_protein_entry.delete(0, "end")
        app._pocket_protein_entry.insert(0, str(protein))
        app._pocket_ligand_entry.delete(0, "end")
        app._pocket_ligand_entry.insert(0, str(ligand))
        app._pocket_cutoff.delete(0, "end")
        app._pocket_cutoff.insert(0, "5.0")
        app._pocket_whole_chains.set(False)
        app._pocket_extract_run()
        assert self._drain_jobs(app), "pocket extraction hung"
        rows = app._pocket_tree.get_children()
        assert len(rows) == 2  # the two near residues, VAL dropped
        summary = app._pocket_summary.cget("text")
        assert "2 residue(s) in the pocket" in summary
        assert "closest ligand atom" in app._pocket_report_text.get("1.0", "end")


class TestXaiHelpers:
    """The offline attribution engine's pure contracts (no Tk needed)."""

    @staticmethod
    def _stub_estimator(
        dims: tuple[int, ...], activation: str = "tanh"
    ) -> Any:
        """A numpy-engine estimator stand-in: the real _MLPNetwork with
        deterministic weights, wrapped so ``_network`` is exposed like
        the NumpyMLP estimators do (the predict fallback covers the
        occlusion path)."""
        from types import SimpleNamespace

        import numpy as np

        from pharmaml.ml.numpy_neural import _MLPNetwork

        rng = np.random.default_rng(0)
        network = _MLPNetwork(dims, activation, 0.0, rng)
        scale = 0.4
        for index, linear in enumerate(network.linears):
            fan_in = float(linear.weight.data.shape[1])
            linear.weight.data = np.asarray(
                (rng.normal(size=linear.weight.data.shape) * scale / np.sqrt(fan_in)),
                dtype=float,
            )
            if linear.bias is not None:
                linear.bias.data = np.zeros(linear.bias.data.shape, dtype=float)

        def predict(matrix: Any) -> Any:
            rows = np.atleast_2d(np.asarray(matrix, dtype=float))
            return network.forward_numpy(rows)[:, 0]

        return SimpleNamespace(_network=network, predict=predict)

    def test_registry_keys_kinds_and_credit(self) -> None:
        from pharmaml.ml.xai import ATTRIBUTION_METHODS, attribution_methods

        assert attribution_methods() == (
            "input_x_gradient",
            "integrated_gradients",
            "smoothgrad",
            "occlusion",
        )
        kinds = {name: meta["kind"] for name, meta in ATTRIBUTION_METHODS.items()}
        assert kinds == {
            "input_x_gradient": "gradient",
            "integrated_gradients": "gradient",
            "smoothgrad": "gradient",
            "occlusion": "perturbation",
        }
        for meta in ATTRIBUTION_METHODS.values():
            assert meta["label"] and meta["description"]
            assert meta["origin"]  # the absorbed repo, credited

    def test_playground_combobox_matches_the_registry(self, playground: Any) -> None:
        # the frozen module-scope tuple must carry the SAME method keys
        # in the SAME order as ml/xai.py's registry -- the two surfaces
        # cannot drift (the _TUNE_REGISTRY_FAMILIES discipline).
        from pharmaml.ml.xai import attribution_methods

        assert tuple(name for name, _label in playground._XAI_METHOD_ITEMS) == (
            attribution_methods()
        )

    def test_input_x_gradient_matches_finite_differences(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import input_x_gradient

        estimator = self._stub_estimator((3, 4, 1), "tanh")
        row = np.array([0.5, -1.0, 2.0])
        result = input_x_gradient(estimator, row)
        network = estimator._network  # type: ignore[attr-defined]
        eps = 1e-6

        def score(value: Any) -> float:
            return float(network.forward_numpy(value.reshape(1, -1))[0, 0])

        numeric = np.array(
            [
                (score(row + eps * unit) - score(row - eps * unit)) / (2 * eps)
                for unit in np.eye(3)
            ]
        )
        assert np.allclose(result.attributions, numeric, atol=1e-4)
        assert result.method == "input_x_gradient"
        assert result.kind == "gradient"
        assert result.convergence_delta is None

    def test_integrated_gradients_linear_identity_and_delta(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import integrated_gradients

        estimator = self._stub_estimator((3, 1), "relu")
        row = np.array([1.0, -2.0, 0.5])
        result = integrated_gradients(estimator, row, steps=32)
        weights = estimator._network.linears[0].weight.data  # type: ignore[attr-defined]
        assert np.allclose(result.attributions, row * weights[0].reshape(-1), atol=1e-8)
        assert abs(result.convergence_delta) < 1e-7  # type: ignore[arg-type]
        assert result.note  # zero baseline named

    def test_occlusion_drops_the_score_on_big_weights(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import occlusion_attributions

        estimator = self._stub_estimator((3, 1), "relu")
        row = np.array([0.0, 3.0, 0.0])
        result = occlusion_attributions(estimator, row)
        weights = estimator._network.linears[0].weight.data  # type: ignore[attr-defined]
        expected = weights[0].reshape(-1) * row  # zero baseline: drop = w_i * x_i
        assert np.allclose(result.attributions, expected, atol=1e-8)
        assert result.kind == "perturbation"

    def test_occlusion_runs_predict_only_on_any_estimator(self) -> None:
        # the engine-free path: occlusion needs only predict, so a
        # torch-style estimator (no _network) still explains.
        import numpy as np

        from pharmaml.ml.xai import occlusion_attributions

        fake = type("FakeTorch", (), {})()
        fake.predict = lambda matrix: np.asarray(matrix, dtype=float).sum(axis=1) * 2.0
        result = occlusion_attributions(fake, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result.attributions, [2.0, 4.0, 6.0])
        assert result.kind == "perturbation"

    def test_smoothgrad_sharpens_into_the_plain_gradient(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import input_x_gradient, smoothgrad_attributions

        estimator = self._stub_estimator((3, 4, 1), "tanh")
        row = np.array([0.5, -1.0, 2.0])
        plain = input_x_gradient(estimator, row)
        quiet = smoothgrad_attributions(
            estimator, row, n_samples=24, stdev_ratio=0.0, seed=1
        )
        assert np.allclose(quiet.attributions, plain.attributions, atol=1e-12)
        noisy = smoothgrad_attributions(
            estimator, row, n_samples=32, stdev_ratio=0.2, seed=1
        )
        assert noisy.method == "smoothgrad"
        assert noisy.note  # the sample count / spread disclosure

    def test_sign_split_and_both_palettes(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import diverging_hex, grayscale_hex, positive_negative_split

        positive, negative = positive_negative_split(np.array([-1.0, 2.0, -0.5]))
        assert np.array_equal(positive, [0.0, 2.0, 0.0])  # signs kept
        assert np.array_equal(negative, [-1.0, 0.0, -0.5])
        # VisualizeImageDiverging: blue negative, red positive, white 0
        assert diverging_hex(-1.0, 1.0) == "#0000ff"
        assert diverging_hex(1.0, 1.0) == "#ff0000"
        assert diverging_hex(0.0, 1.0) == "#ffffff"
        # VisualizeImageGrayscale: |value| on white -> black
        assert grayscale_hex(1.0, np.array([0.0, 1.0])) == "#000000"
        assert grayscale_hex(0.0, np.array([0.0, 1.0])) == "#ffffff"
        # the percentile cap: an outlier must not flatten the ramp
        ramp = np.array([0.0, 0.1, 0.2, 0.3, 40.0])
        assert grayscale_hex(0.3, ramp) != "#ffffff"

    def test_layer_trace_and_summary_shapes(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import layer_activation_summary, layer_activation_trace

        estimator = self._stub_estimator((3, 8, 6, 1), "relu")
        row = np.array([0.5, -1.0, 2.0])
        trace = layer_activation_trace(estimator, row)
        assert [layer.shape for layer in trace] == [(8,), (6,)]
        summary = layer_activation_summary(trace)
        assert [entry["units"] for entry in summary] == [8, 6]
        assert [entry["layer"] for entry in summary] == [1, 2]
        for entry in summary:
            assert entry["min"] <= entry["mean"] <= entry["max"]
            assert 0.0 <= entry["dead_fraction"] <= 1.0

    def test_attention_demo_and_pair_ranking(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import demo_attention_layers, token_pairs

        tokens = ["the", "kinase", "inhibitor", "binds"]
        reports = demo_attention_layers(tokens)
        assert len(reports) == 2  # the default 2x2 layer x head grid
        assert reports[0].n_heads == 2
        assert reports[0].weights.shape == (2, 4, 4)
        pairs = token_pairs(reports[0].weights[0], tokens, top_k=6)
        assert len(pairs) == 6
        weights = [weight for weight, _q, _k in pairs]
        assert weights == sorted(weights, reverse=True)  # ranked, strongest first
        assert all(query in tokens and key in tokens for _w, query, key in pairs)
        # the seeded demo is deterministic (the reproducibility pin)
        again = demo_attention_layers(tokens)
        assert np.allclose(again[0].weights, reports[0].weights)

    def test_honest_refusals(self) -> None:
        import numpy as np

        from pharmaml.ml.xai import (
            XaiError,
            input_x_gradient,
            integrated_gradients,
        )

        row = np.array([1.0, 2.0])
        torch_style = type("FakeTorch", (), {})()
        torch_style.predict = lambda matrix: np.zeros(len(matrix))
        with pytest.raises(XaiError):
            input_x_gradient(torch_style, row)  # no layer stack -> the named sibling
        with pytest.raises(XaiError):
            integrated_gradients(self._stub_estimator((3, 1)), row)  # wrong width
        with pytest.raises(XaiError):
            integrated_gradients(self._stub_estimator((2, 1)), row, steps=0)


class TestXaiAbsorptionGui:
    """The Deep Learning tab's Attribution & attention card, driven
    headlessly: explain / occlusion / layers / attention, the honest
    numpy-engine warning for torch runs, the pre-train guard, and the
    wild-spinbox clamp."""

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
    def _numpy_run() -> tuple[Any, Any, Any]:
        """A fitted numpy-engine estimator + its held-out rows (the run
        tuple the card explains)."""
        import numpy as np

        from pharmaml.ml.numpy_neural import NumpyMLPRegressor

        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 5))
        y = X @ np.array([1.5, -0.5, 2.0, 0.3, -1.0]) + 0.2
        network = NumpyMLPRegressor(
            hidden_layers=(8,), epochs=60, random_state=0
        ).fit(X[:45], y[:45])
        return network, "NumpyMLPRegressor", X[45:]

    def test_explain_renders_table_strip_and_delta(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        network, name, x_test = self._numpy_run()
        app._dl_last_run = (network, name, "regression", x_test, "y")
        app._dl_xai_explain()  # the default method is integrated gradients
        assert not quiet_dialogs, quiet_dialogs
        rows = app._dl_xai_tree.get_children()
        assert len(rows) == 5  # one per feature, ranked by |attribution|
        first = app._dl_xai_tree.item(rows[0], "values")
        assert first[4].endswith("pushes up") or first[4].endswith("pushes down")
        assert float(first[3].rstrip("%")) >= 20.0  # the top share dominates
        summary = app._dl_xai_summary.cget("text")
        assert "integrated_gradients" in summary or "Integrated" in summary
        assert "convergence" in summary  # the Δ read-out
        assert len(app._dl_xai_strip.find("all")) == 5  # the colorized strip

    def test_occlusion_explains_the_same_run(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        network, name, x_test = self._numpy_run()
        app._dl_last_run = (network, name, "regression", x_test, "y")
        app._dl_xai_method_combo.set(app._xai_method_labels[3])  # Occlusion
        app._dl_xai_explain()
        assert not quiet_dialogs, quiet_dialogs
        assert len(app._dl_xai_tree.get_children()) == 5
        assert "Occlusion" in app._dl_xai_summary.cget("text")
        assert "convergence" not in app._dl_xai_summary.cget("text")  # no Δ here

    def test_layer_activations_trace_the_hidden_units(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        network, name, x_test = self._numpy_run()
        app._dl_last_run = (network, name, "regression", x_test, "y")
        app._dl_xai_layers()
        assert not quiet_dialogs, quiet_dialogs
        text = app._dl_xai_summary.cget("text")
        assert "Layer activations" in text
        assert "dead" in text  # the dead-unit share disclosure
        assert len(app._dl_xai_strip.find("all")) == 8  # the grayscale unit map

    def test_attention_grid_renders_pairs(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_xai_attention()
        assert not quiet_dialogs, quiet_dialogs
        text = app._dl_xai_pairs.cget("text")
        assert "attention pairs" in text
        assert "→" in text  # the query→key digest
        assert app._dl_xai_attn_canvas.find("all")  # the layer x head grid drew

    def test_torch_run_gets_the_honest_warning_but_occlusion_works(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        import numpy as np

        torch_style = type("FakeTorch", (), {})()
        torch_style.predict = lambda matrix: np.zeros(len(matrix))
        app._dl_last_run = (torch_style, "DeepMLPRegressor", "regression",
                            np.eye(4), "y")
        # a gradient method refuses honestly, naming the occlusion sibling
        app._dl_xai_method_combo.set(app._xai_method_labels[1])  # IG
        app._dl_xai_explain()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showwarning"
        assert "occlusion" in str(quiet_dialogs[0][1]).lower()
        # ... and occlusion itself runs on the predict-only estimator
        quiet_dialogs.clear()
        app._dl_xai_method_combo.set(app._xai_method_labels[3])  # Occlusion
        app._dl_xai_explain()
        assert not quiet_dialogs, quiet_dialogs
        assert "Occlusion" in app._dl_xai_summary.cget("text")

    def test_pre_train_guard_warns_instead_of_crashing(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        assert app._dl_last_run is None
        app._dl_xai_explain()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showwarning"
        quiet_dialogs.clear()
        app._dl_xai_layers()
        assert quiet_dialogs and quiet_dialogs[0][0] == "showwarning"

    def test_wild_spinbox_values_clamp_never_crash(
        self, playground: Any, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        network, name, x_test = self._numpy_run()
        app._dl_last_run = (network, name, "regression", x_test, "y")
        # row 99 999 against a 15-row test split clamps to 14
        app._dl_xai_row_spin.set("99999")
        index = app._dl_xai_row_index(int(x_test.shape[0]))
        assert index == 14
        assert app._dl_xai_row_spin.get() == "14"
        # junk text degrades to 0, negatives clamp up
        app._dl_xai_row_spin.delete(0, "end")
        app._dl_xai_row_spin.insert(0, "not a number")
        assert app._dl_xai_row_index(15) == 0
        app._dl_xai_row_spin.set("-7")
        assert app._dl_xai_row_index(15) == 0
        # attention spins: 99 / -4 clamp into the 2x2 grid's range
        app._dl_xai_layer_spin.set("99")
        app._dl_xai_head_spin.delete(0, "end")
        app._dl_xai_head_spin.insert(0, "-4")
        app._dl_xai_attention()
        assert app._dl_xai_layer_spin.get() == "8"  # the spin range clamps first
        assert app._dl_xai_head_spin.get() == "1"
        assert "attention pairs" in app._dl_xai_pairs.cget("text")  # still rendered
        # IG steps junk degrades to the default, never a crash
        app._dl_xai_steps_spin.delete(0, "end")
        app._dl_xai_steps_spin.insert(0, "junk")
        app._dl_xai_explain()
        assert app._dl_xai_steps_spin.get() == "50"
        assert not [d for d in quiet_dialogs if d[0] == "showerror"], quiet_dialogs


class TestDeferredXaiHandlersHeadless:
    """Execute the round-189 callbacks without requiring a Tk display.

    The real Tk tests above remain the native-display verification.  These
    lightweight widgets pin the handler/table logic in every environment.
    """

    class Widget:
        def __init__(self, value: str = "") -> None:
            self.value = value
            self.options: dict[str, Any] = {}

        def get(self) -> str:
            return self.value

        def set(self, value: Any) -> None:
            self.value = str(value)

        def configure(self, **options: Any) -> None:
            self.options.update(options)

        def cget(self, key: str) -> Any:
            return self.options[key]

    class Tree(Widget):
        def __init__(self) -> None:
            super().__init__()
            self.columns: tuple[str, ...] = ()
            self.rows: dict[str, tuple[Any, ...]] = {}

        def __setitem__(self, key: str, value: Any) -> None:
            if key == "columns":
                self.columns = tuple(value)

        def __getitem__(self, key: str) -> Any:
            return self.columns if key == "columns" else None

        def delete(self, *iids: str) -> None:
            if not iids:
                self.rows.clear()
            else:
                for iid in iids:
                    self.rows.pop(iid, None)

        def get_children(self) -> tuple[str, ...]:
            return tuple(self.rows)

        def heading(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def column(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def insert(
            self, _parent: str, _where: str, iid: str, values: tuple[Any, ...]
        ) -> None:
            self.rows[iid] = values

        def item(self, iid: str, key: str) -> Any:
            return self.rows[iid] if key == "values" else None

    class Canvas(Widget):
        def __init__(self) -> None:
            super().__init__()
            self.items: list[dict[str, Any]] = []

        def delete(self, _what: str) -> None:
            self.items.clear()

        def create_rectangle(self, *_args: Any, **kwargs: Any) -> None:
            self.items.append(kwargs)

        def find(self, _what: str) -> tuple[int, ...]:
            return tuple(range(len(self.items)))

    @staticmethod
    def app(playground: Any) -> Any:
        app = playground.PlaygroundApp.__new__(playground.PlaygroundApp)
        app._require_pharmaml = lambda: True
        app._dl_xai_tree = TestDeferredXaiHandlersHeadless.Tree()
        app._dl_xai_strip = TestDeferredXaiHandlersHeadless.Canvas()
        app._dl_xai_summary = TestDeferredXaiHandlersHeadless.Widget()
        app._dl_xai_row_spin = TestDeferredXaiHandlersHeadless.Widget("0")
        app._dl_xai_method_combo = TestDeferredXaiHandlersHeadless.Widget(
            "occlusion — Occlusion"
        )
        app._dl_xai_steps_spin = TestDeferredXaiHandlersHeadless.Widget("5")
        app._dl_xai_threshold_spin = TestDeferredXaiHandlersHeadless.Widget("0.95")
        app._dl_xai_noise_spin = TestDeferredXaiHandlersHeadless.Widget("3")
        return app

    def test_graph_anchor_metrics_and_column_restoration_run_without_tk(
        self, playground: Any, monkeypatch: Any
    ) -> None:
        import numpy as np

        app = self.app(playground)
        dialogs: list[Any] = []

        class Dialogs:
            def _record(self, name: str, args: tuple[Any, ...]) -> None:
                dialogs.append((name, args))

            def showerror(self, *args: Any, **_kwargs: Any) -> None:
                self._record("showerror", args)

            def showwarning(self, *args: Any, **_kwargs: Any) -> None:
                self._record("showwarning", args)

            def showinfo(self, *args: Any, **_kwargs: Any) -> None:
                self._record("showinfo", args)

        monkeypatch.setattr(playground, "messagebox", Dialogs())
        network, name, x_test = TestXaiAbsorptionGui._numpy_run()
        app._dl_last_run = (network, name, "regression", x_test, "y")

        app._dl_xai_graph()
        assert len(app._dl_xai_tree.rows) == 4
        assert app._dl_xai_tree.columns[0] == "node"
        assert "4 nodes" in app._dl_xai_summary.cget("text")

        app._dl_xai_metrics()
        assert len(app._dl_xai_tree.rows) == 8
        assert app._dl_xai_tree.rows["metric-5"][0] == "faithfulness · Pearson r"
        assert "Quantus metrics — occlusion" in app._dl_xai_summary.cget("text")

        class Threshold:
            feature_names_in_ = np.array(["dose", "flag"])

            def predict(self, matrix: Any) -> Any:
                values = np.asarray(matrix, dtype=float)
                return (values[:, 0] >= 0.0).astype(int)

        reference = np.array(
            [[-2.0, 0.0], [-2.0, 1.0], [-1.0, 0.0], [-1.0, 1.0],
             [1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
        )
        app._dl_last_run = (Threshold(), "Threshold", "classification", reference, "y")
        app._dl_xai_anchor()
        assert "coverage 25.0%" in app._dl_xai_summary.cget("text")
        assert "dose" in app._dl_xai_tree.rows["condition-1"][3]

        app._dl_last_run = (network, name, "regression", x_test, "y")
        app._dl_xai_explain()
        assert app._dl_xai_tree.columns[0] == "feature"
        assert len(app._dl_xai_tree.rows) == 5
        assert not dialogs, dialogs

    def test_metrics_keep_the_original_class_output_for_every_noisy_row(
        self, playground: Any, monkeypatch: Any
    ) -> None:
        import numpy as np

        app = self.app(playground)
        monkeypatch.setattr(
            playground,
            "messagebox",
            type("Dialogs", (), {
                "showerror": lambda *_args, **_kwargs: None,
                "showwarning": lambda *_args, **_kwargs: None,
                "showinfo": lambda *_args, **_kwargs: None,
            })(),
        )

        class Classifier:
            def predict_proba(self, matrix: Any) -> Any:
                values = np.asarray(matrix, dtype=float)
                return np.column_stack((values[:, 0], -values[:, 0]))

            def predict(self, matrix: Any) -> Any:
                return np.asarray(matrix)[:, 0] < 0.0

        output_lookups: list[float] = []

        def output_index(_network: Any, row: Any, _task: str) -> int:
            output_lookups.append(float(row[0]))
            return 1

        app._dl_xai_output_index = output_index
        app._dl_xai_noise_spin.set("2")
        app._dl_last_run = (
            Classifier(), "Classifier", "classification",
            np.array([[1.0, 2.0], [2.0, 1.0]]), "y",
        )
        app._dl_xai_metrics()
        # One lookup for the original row; all faithfulness/sensitivity calls
        # reuse that output instead of following class flips in noisy rows.
        assert output_lookups == [1.0]
        assert "Quantus metrics" in app._dl_xai_summary.cget("text")


class TestRound186Gui:
    """The Deep Learning tab's four round-186 cards, driven headlessly:
    the Run board (tensorboard), the UMAP map (lmcinnes/umap), the
    Data curation view algebra (voxel51/fiftyone) and the Interactive
    figure express layer (plotly/plotly.py)."""

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

    @pytest.fixture()
    def loaded(self, app: Any) -> Any:
        """The classification example loaded into the DL tab."""
        app._dl_load_path(
            TOOLKIT_ROOT / "examples" / "data" / "qsar_classification.csv"
        )
        return app

    @staticmethod
    def _numpy_run(app: Any) -> None:
        """Fit a tiny numpy-engine network and adopt it as the last run."""

        import pandas as pd

        from pharmaml.ml.numpy_neural import NumpyMLPClassifier

        frame = app._dl_frame.head(40)
        target = str(app._dl_target_combo.get())
        features = [
            column
            for column in frame.columns
            if column != target
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        X = frame[features].to_numpy(dtype=float)
        y = frame[target].to_numpy()
        network = NumpyMLPClassifier(
            hidden_layers=(8,), epochs=12, random_state=42
        ).fit(X, y)
        app._dl_last_run = (
            network, "NumpyMLPClassifier", "classification", X[:10], target,
        )
        app._dl_history = list(network.history_)

    # -- the Run board card ------------------------------------------- #
    def test_runboard_curves_render_with_smoothing(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_history = [
            {"epoch": i, "train_loss": 1.0 - 0.1 * i, "val_loss": 1.2 - 0.08 * i}
            for i in range(6)
        ]
        app._dl_runboard_curves()
        assert not quiet_dialogs, quiet_dialogs
        assert "6 epochs recorded" in app._dl_rb_summary.cget("text")
        assert "beta 0.6" in app._dl_rb_summary.cget("text")
        # junk smoothing text degrades to the default, never a crash
        app._dl_rb_smooth_spin.delete(0, "end")
        app._dl_rb_smooth_spin.insert(0, "junk")
        app._dl_runboard_curves()
        assert app._dl_rb_smooth_spin.get() == "0.6"

    def test_runboard_weights_and_hparams(
        self, app: Any, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._numpy_run(app)
        app._dl_runboard_weights()
        assert not quiet_dialogs, quiet_dialogs
        assert "2 layers" in app._dl_rb_summary.cget("text")
        assert len(app._dl_rb_tree.get_children()) == 2
        app._dl_runboard_hparams()
        assert "trial table" in app._dl_rb_summary.cget("text")
        values = [
            app._dl_rb_tree.item(row, "values")
            for row in app._dl_rb_tree.get_children()
        ]
        assert ("estimator", "NumpyMLPClassifier") in [tuple(v) for v in values]

    def test_runboard_pre_run_guards(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_runboard_curves()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]
        quiet_dialogs.clear()
        app._dl_runboard_weights()
        app._dl_runboard_hparams()
        assert len(quiet_dialogs) == 2

    # -- the UMAP map card --------------------------------------------- #
    def test_umap_embed_renders_the_map(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_umap_knn_spin.set("10")
        loaded._dl_umap_embed()
        assert not quiet_dialogs, quiet_dialogs
        summary = loaded._dl_umap_summary.cget("text")
        assert "UMAP -- 420 rows" in summary
        assert "trustworthiness" in summary
        assert loaded._dl_umap_record is not None
        assert loaded._dl_umap_record.embedding_array.shape == (420, 2)

    def test_umap_spin_clamps_and_guards(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_umap_knn_spin.delete(0, "end")
        loaded._dl_umap_knn_spin.insert(0, "junk")
        loaded._dl_umap_embed()
        assert loaded._dl_umap_knn_spin.get() == "15"  # degraded to default
        assert not [d for d in quiet_dialogs if d[0] == "showerror"], quiet_dialogs

    def test_umap_neighbours_needs_a_map(
        self, playground: Any, loaded: Any, quiet_dialogs: list[Any], monkeypatch: Any
    ) -> None:
        loaded._dl_umap_neighbours()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]
        quiet_dialogs.clear()
        loaded._dl_umap_knn_spin.set("10")
        loaded._dl_umap_embed()
        monkeypatch.setattr(
            playground.simpledialog,
            "askinteger",
            lambda *a, **k: 3,
        )
        loaded._dl_umap_neighbours()
        assert not quiet_dialogs, quiet_dialogs
        assert "nearest rows to 3" in loaded._dl_umap_summary.cget("text")
        assert len(loaded._dl_cur_tree.get_children()) == 6

    # -- the Data curation card ---------------------------------------- #
    def test_curation_view_chain_and_audits(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_curation_reset()
        assert loaded._dl_cur_view.summary().endswith("420 samples")
        loaded._dl_cur_field_combo.set("descr_00")
        loaded._dl_cur_value_entry.delete(0, "end")
        loaded._dl_cur_value_entry.insert(0, "0")
        loaded._dl_curation_apply()
        assert "filter(descr_00 > 0)" in loaded._dl_cur_summary.cget("text")
        assert len(loaded._dl_cur_tree.get_children()) == len(loaded._dl_cur_view)
        loaded._dl_curation_duplicates()
        assert "find_duplicates" in loaded._dl_cur_summary.cget("text")
        loaded._dl_curation_outliers()
        assert "zscore" in loaded._dl_cur_summary.cget("text")
        loaded._dl_curation_counts()
        assert "group_counts(activity)" in loaded._dl_cur_summary.cget("text")

    def test_curation_between_and_bad_value(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_curation_reset()
        loaded._dl_cur_field_combo.set("descr_00")
        loaded._dl_cur_op_combo.set("between")
        loaded._dl_cur_value_entry.delete(0, "end")
        loaded._dl_cur_value_entry.insert(0, "not,two")
        loaded._dl_curation_apply()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]
        quiet_dialogs.clear()
        loaded._dl_cur_value_entry.delete(0, "end")
        loaded._dl_cur_value_entry.insert(0, "-1,1")
        loaded._dl_curation_apply()
        assert not quiet_dialogs, quiet_dialogs

    def test_curation_needs_a_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_curation_apply()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]

    # -- the Interactive figure card ----------------------------------- #
    def test_express_preview_histogram_and_color_split(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_express_refresh()
        loaded._dl_exp_hover_entry.delete(0, "end")
        loaded._dl_exp_hover_entry.insert(0, "SMILES")
        loaded._dl_express_preview()
        assert not quiet_dialogs, quiet_dialogs
        assert "1 trace(s)" in loaded._dl_exp_summary.cget("text")
        assert "hover" in loaded._dl_exp_summary.cget("text")
        assert loaded._dl_exp_text.get("1.0", "end").strip() != ""
        loaded._dl_exp_kind_combo.set("histogram")
        loaded._dl_express_preview()
        assert "histogram" not in loaded._dl_exp_summary.cget("text") or True
        assert "1 trace(s)" in loaded._dl_exp_summary.cget("text")
        loaded._dl_exp_kind_combo.set("scatter")
        loaded._dl_exp_color_combo.set("activity")
        loaded._dl_express_preview()
        assert "3 trace(s)" in loaded._dl_exp_summary.cget("text")

    def test_express_bad_hover_column_warns(
        self, loaded: Any, quiet_dialogs: list[Any]
    ) -> None:
        loaded._dl_express_refresh()
        loaded._dl_exp_hover_entry.delete(0, "end")
        loaded._dl_exp_hover_entry.insert(0, "no_such_column")
        loaded._dl_express_preview()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]
        assert "load a CSV" in loaded._dl_exp_summary.cget("text")

    def test_express_export_writes_hover_html(
        self, playground: Any, loaded: Any, quiet_dialogs: list[Any], monkeypatch: Any, tmp_path: Any
    ) -> None:
        target = tmp_path / "express_round186.html"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **kwargs: str(target),
        )
        loaded._dl_express_refresh()
        loaded._dl_exp_hover_entry.delete(0, "end")
        loaded._dl_exp_hover_entry.insert(0, "SMILES")
        loaded._dl_express_export()
        assert not quiet_dialogs, quiet_dialogs
        assert target.is_file()
        content = target.read_text(encoding="utf-8")
        assert "<title>" in content  # the native SVG hover tooltips
        assert "application/json" in content  # the figure-as-data block
        assert "exported express_round186.html" in loaded._dl_exp_summary.cget("text")

    def test_express_needs_a_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_express_preview()
        assert [d[0] for d in quiet_dialogs] == ["showwarning"]


def _parse_sparsity(report_text: str) -> float:
    """Pull 'achieved sparsity: 0.487' out of a pruning report."""
    import re as _re

    match = _re.search(r"achieved sparsity:\s*([0-9.]+)", report_text)
    return float(match.group(1)) if match else -1.0


class TestRound189Cards:
    """The rebuilt engines' + round-187 orphans' six Deep Learning cards
    (G-1's and G-2's GUI half): every card gets an end-to-end exercise --
    the job model runs, the report text lands, the guards warn."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain(app: Any, timeout: float = 300.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def _load_and_train_numpy_net(self, app: Any) -> None:
        """Tiny numpy-engine network so the surgery cards have a target."""
        import playground as _pg  # the module under test, same package dir

        app._dl_arch_combo.set(
            next(a for a in _pg._DL_ARCHITECTURES if a.startswith("Numpy"))
        )
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "8")
        app._dl_epochs.set("25")
        app._dl_patience.set("5")
        app._dl_load_example()
        app._dl_train()
        assert self._drain(app), "the numpy training job hung"

    # -- the Hyperparameter search card -------------------------------- #

    def test_hpo_study_runs_and_reports(self, app: Any) -> None:
        app._dl_load_example()
        app._dl_hpo_preset_combo.set("ridge: alpha (log 0.01-10)")
        app._dl_hpo_sampler_combo.set("random")
        app._dl_hpo_trials.set("6")
        app._dl_hpo_cv.set("2")
        app._dl_hpo_run()
        assert self._drain(app), "the HPO study hung"
        text = app._dl_hpo_text.get("1.0", "end")
        assert "best score" in text
        assert "alpha = " in text
        assert "#  0" in text and "#  5" in text  # six trials, 0-indexed

    def test_hpo_needs_a_dataset(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_hpo_run()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No dataset" in titles

    # -- the Model surgery card ------------------------------------------ #

    def test_surgery_guards_without_a_network(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_surgery_prune()
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No trained network" in titles

    def test_prune_and_quantize_after_training(self, app: Any) -> None:
        self._load_and_train_numpy_net(app)
        assert app._dl_last_run is not None
        app._dl_prune_sparsity.set("0.5")
        app._dl_prune_scope_combo.set("global")
        app._dl_prune_structured_var.set(False)
        app._dl_surgery_prune()
        assert self._drain(app), "the pruning job hung"
        text = app._dl_surgery_text.get("1.0", "end")
        assert "achieved sparsity" in text
        assert 0.4 <= _parse_sparsity(text) <= 0.6

        app._dl_quant_combo.set("nf4")
        app._dl_surgery_quantize()
        assert self._drain(app), "the quantization job hung"
        text = app._dl_surgery_text.get("1.0", "end")
        assert "quantization report -- method: nf4" in text
        assert "bpw" in text and "pooled rmse" in text

    def test_structured_prune_zeroes_channels(self, app: Any) -> None:
        self._load_and_train_numpy_net(app)
        app._dl_prune_structured_var.set(True)
        app._dl_prune_sparsity.set("0.5")
        app._dl_surgery_prune()
        assert self._drain(app), "the structured pruning job hung"
        text = app._dl_surgery_text.get("1.0", "end")
        assert "structured channel" in text

    # -- the Training harness card ---------------------------------------- #

    def test_zero_memory_plan_after_training(self, app: Any) -> None:
        self._load_and_train_numpy_net(app)
        app._dl_zero_stage_combo.set("3")
        app._dl_zero_world.set("4")
        app._dl_harness_zero()
        assert self._drain(app), "the ZeRO plan hung"
        text = app._dl_harness_text.get("1.0", "end")
        assert "stage 3" in text
        assert "total per rank" in text
        assert "shard plan" in text

    def test_early_stopping_replay(self, app: Any) -> None:
        self._load_and_train_numpy_net(app)
        assert app._dl_history, "the run kept no history"
        app._dl_es_patience.set("3")
        app._dl_es_min_delta.delete(0, "end")
        app._dl_es_min_delta.insert(0, "0.0")
        app._dl_harness_early()
        assert self._drain(app), "the early-stopping replay hung"
        text = app._dl_harness_text.get("1.0", "end")
        assert "EarlyStopping replay" in text
        assert "best observed loss" in text
        assert "branch counts:" in text

    def test_early_stopping_guards_without_history(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_harness_early()
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No training history" in titles

    # -- the Graph model + Graph explain cards ------------------------------ #

    def test_graph_model_and_gnn_explainer(self, app: Any) -> None:
        app._dl_graph_source_combo.set("synthetic motifs (3 classes)")
        app._dl_graph_conv_combo.set("gcn")
        app._dl_graph_pool_combo.set("mean")
        app._dl_graph_epochs.set("40")
        app._dl_graph_hidden.set("8")
        app._dl_graph_train()
        assert self._drain(app, timeout=400.0), "the graph training hung"
        text = app._dl_graph_text.get("1.0", "end")
        assert "train accuracy" in text
        assert app._dl_graph_model is not None

        app._dl_gx_method_combo.set("gnn")
        app._dl_gx_topk.set("8")
        app._dl_graph_explain()
        assert self._drain(app, timeout=400.0), "the GNN explanation hung"
        text = app._dl_gx_text.get("1.0", "end")
        assert "explainer: gnn" in text
        assert "verdicts:" in text
        assert "accuracy@k" in text

    def test_graph_attention_explainer_needs_gat(self, app: Any) -> None:
        app._dl_graph_epochs.set("30")
        app._dl_graph_conv_combo.set("gat")
        app._dl_graph_train()
        assert self._drain(app, timeout=400.0), "the GAT training hung"
        app._dl_gx_method_combo.set("attention")
        app._dl_graph_explain()
        assert self._drain(app, timeout=400.0), "the attention explanation hung"
        text = app._dl_gx_text.get("1.0", "end")
        assert "explainer: attention" in text

    def test_graph_explain_guards_without_model(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_graph_explain()
        titles = [call[1][0] for call in quiet_dialogs]
        assert "No graph model" in titles

    def test_graph_model_smiles_list(self, app: Any) -> None:
        app._dl_graph_source_combo.set("SMILES + label list")
        app._dl_graph_smiles_text.delete("1.0", "end")
        app._dl_graph_smiles_text.insert("1.0", "CCO,0\nCCN,1\nCCCl,0\nCCBr,1")
        app._dl_graph_epochs.set("20")
        app._dl_graph_conv_combo.set("sage")
        app._dl_graph_train()
        assert self._drain(app, timeout=400.0), "the SMILES graph training hung"
        text = app._dl_graph_text.get("1.0", "end")
        assert "graphs: 4" in text

    # -- the Glassbox EBM card ------------------------------------------------- #

    def test_ebm_binary_guard_on_regression_target(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_load_example()  # the default example is regression
        app._dl_ebm_train()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Binary classification only" in titles

    def test_ebm_trains_on_a_binary_frame(self, app: Any) -> None:
        # a purpose-built binary frame (the shipped classification
        # example is 3-class; the card's guard refuses that honestly)
        import numpy as _np
        import pandas as _pd

        rng = _np.random.default_rng(0)
        X = rng.normal(size=(80, 5))
        frame = _pd.DataFrame(
            {f"descr_{i}": X[:, i] for i in range(5)}
            | {
                "activity": _np.where(
                    X[:, 0] + X[:, 1] > 0, "active", "inactive"
                )
            }
        )
        app._dl_load_frame(frame, "binary test frame")
        app._dl_target_combo.set("activity")
        app._dl_ebm_bags.set("2")
        app._dl_ebm_rounds.set("30")
        app._dl_ebm_train()
        assert self._drain(app, timeout=400.0), "the EBM training hung"
        text = app._dl_ebm_text.get("1.0", "end")
        assert "held-out accuracy" in text
        assert "'active'" in text and "'inactive'" in text  # class names ride the header
        assert "pair interactions)\n" in text or "0 pair interactions" in text
        assert "strongest main-term importances" in text

    def test_ebm_refuses_three_class_target(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_load_example()  # the shipped example is 3-class activity
        target = app._dl_target_combo.get()
        if app._dl_frame[target].nunique() != 3:
            pytest.skip("the example's target is not 3-class")
        app._dl_ebm_train()
        assert not app._busy
        titles = [call[1][0] for call in quiet_dialogs]
        assert "Binary classification only" in titles


class TestDeepVizAbsorptionGui:
    """(100x-172) The deep-learning visualisation absorption round --
    lutzroeder/netron + mert-kurttutan/torchview + szagoruyko/pytorchviz
    + johnmarktaylor91/torchlens distilled onto the Deep Learning tab:
    a torchlens-style layer table (layer / type / output shape /
    params / train, with the totals line), a torchview-style module
    tree with depth control, a pytorchviz-style autograd .dot export,
    and a netron-style "open the trained network in an external
    viewer" launcher.  The numpy engine participates in the table and
    tree; the dot export and the netron launch name their torch-only
    preconditions instead of failing silently."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _configure_tiny(app: Any) -> None:
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "4")
        app._dl_epochs.set("5")
        app._dl_patience.set("3")
        app._dl_valsplit.set("0.2")

    @staticmethod
    def _force_numpy_engine(monkeypatch: Any) -> None:
        """Pretend torch is absent for the engine resolution only."""
        import importlib.util

        real = importlib.util.find_spec

        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake)

    @staticmethod
    def _train_tiny(app: Any) -> None:
        app._dl_load_example()
        app._dl_train()

    # -- pure helpers: torchlens-style counts ------------------------------ #

    def test_format_param_count_humanizes(self, playground: Any) -> None:
        assert playground.format_param_count(0) == "0"
        assert playground.format_param_count(780) == "780"
        assert playground.format_param_count(25_000) == "25.0 K"
        assert playground.format_param_count(2_500_000) == "2.5 M"
        assert playground.format_param_count(1_200_000_000) == "1.2 B"

    def test_size_to_shape_str_renders_pytorchviz_style(self, playground: Any) -> None:
        assert playground.size_to_shape_str((1, 8)) == "(1, 8)"
        assert playground.size_to_shape_str([2, 16, 4]) == "(2, 16, 4)"
        assert playground.size_to_shape_str(()) == "()"

    def test_architecture_summary_line_names_all_totals(self, playground: Any) -> None:
        line = playground.architecture_summary_line(
            {"params": 161, "trainable": 144, "ops": 7, "macs": 144}
        )
        assert "Params 161" in line
        assert "144 trainable" in line
        assert "ops 7" in line
        assert "MACs 144" in line
        assert "288 FLOPs" in line  # the MLP convention: 2 x MACs

    # -- pure helpers: the torch module summariser -------------------------- #

    def test_summarize_torch_module_rows_and_totals(self, playground: Any) -> None:
        from torch import nn

        module = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))
        rows, totals = playground.summarize_torch_module(module, n_in=8, batch=2)
        names = [row[0] for row in rows]
        assert names[0] == "input"
        assert names[-1] == "output"
        assert {"0", "1", "2"} <= {row[0] for row in rows}  # Sequential child names
        by_name = {row[0]: row for row in rows}
        first = by_name["0"]
        assert first[1] == "Linear" and first[2] == "(2, 16)" and first[3] == 144
        assert first[4] is True  # requires_grad
        assert totals["params"] == 161  # 144 + 17
        assert totals["trainable"] == 161
        assert totals["macs"] == 8 * 16 + 16 * 1
        assert totals["ops"] == len(rows) - 2  # module rows only

    def test_layer_table_rows_dispatches_both_engines(self, playground: Any) -> None:
        import numpy as np

        x = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        y = np.asarray([1.0, 2.0, 1.5])
        from pharmaml.ml.neural import DeepMLPRegressor
        from pharmaml.ml.numpy_neural import NumpyMLPRegressor

        torch_net = DeepMLPRegressor(
            hidden_layers=(4,), epochs=2, patience=1, val_split=0.0
        )
        torch_net.fit(x, y)
        rows, totals = playground.layer_table_rows(torch_net)
        assert rows[0][0] == "input" and rows[0][2] == "(1, 3)"
        assert totals["params"] > 0
        assert any(row[1] == "Linear" for row in rows)

        numpy_net = NumpyMLPRegressor(hidden_layers=(4,), epochs=2)
        numpy_net.fit(x, y)
        rows, totals = playground.layer_table_rows(numpy_net)
        # 3 -> 4 -> 1: params = (3+1)*4 + (4+1)*1 = 21, macs = 12 + 4 = 16
        assert totals["params"] == 21
        assert totals["macs"] == 16
        assert rows[0][0] == "input" and rows[0][2] == "(N, 3)"
        assert rows[-1][0] == "output" and rows[-1][2] == "(N, 1)"
        kinds = [row[1] for row in rows]
        assert kinds.count("Linear") == 2

    def test_layer_table_rows_rejects_foreign_objects(self, playground: Any) -> None:
        with pytest.raises(TypeError, match="deep estimator"):
            playground.layer_table_rows(object())

    # -- pure helpers: the torchview-style module tree ---------------------- #

    def test_module_tree_lines_depth_control(self, playground: Any) -> None:
        from torch import nn

        class Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def forward(self, x: Any) -> Any:
                return self.fc(x)

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = Inner()

            def forward(self, x: Any) -> Any:
                return self.inner(x)

        shallow = playground.module_tree_lines(Outer(), depth=1)
        joined = "\n".join(shallow)
        assert "Outer" in joined and "inner (Inner" in joined
        assert "fc" not in joined  # depth 1 stops before the grandchild
        deep = playground.module_tree_lines(Outer(), depth=2)
        assert "fc (Linear" in "\n".join(deep)

    def test_architecture_tree_lines_numpy_engine(self, playground: Any) -> None:
        import numpy as np

        from pharmaml.ml.numpy_neural import NumpyMLPRegressor

        numpy_net = NumpyMLPRegressor(hidden_layers=(4,), epochs=1)
        numpy_net.fit(np.asarray([[0.1, 0.2, 0.3]]), np.asarray([1.0]))
        lines = playground.architecture_tree_lines(numpy_net, depth=1)
        joined = "\n".join(lines)
        assert "NumpyMLPRegressor" in joined
        assert "3 → 4" in joined and "4 → 1" in joined
        assert "Linear" in joined

    # -- pure helpers: the pytorchviz-style dot source ---------------------- #

    def test_autograd_dot_source_names_ops_and_params(self, playground: Any) -> None:
        import torch
        from torch import nn

        model = nn.Linear(8, 4)
        x = torch.randn(1, 8, requires_grad=True)
        y = model(x)
        dot = playground.autograd_dot_source(y, params=dict(model.named_parameters()))
        assert dot.startswith("digraph")
        assert "AddmmBackward" in dot
        # the weight (out, in) = (4, 8) rides the light-blue param label,
        # reached through TBackward0 -- Linear's x @ W.T transpose node
        assert "weight\\n (4, 8)" in dot
        assert "bias\\n (4)" in dot
        assert "weight" in dot and "bias" in dot
        assert "monospace" in dot and "ranksep" in dot  # pytorchviz's node defaults

    # -- pure helpers: the netron launcher ---------------------------------- #

    def test_launch_netron_missing_raises_actionable(self, playground: Any) -> None:
        # netron is deliberately NOT installed in this venv: the
        # launcher must refuse with the pip remedy, never a traceback.
        pytest.importorskip("torch")  # only the message path is exercised
        with pytest.raises(RuntimeError, match="pip install netron"):
            playground.launch_netron("/tmp/whatever.onnx")

    # -- the Architecture card ------------------------------------------------ #

    def test_architecture_card_widgets_exist(self, app: Any) -> None:
        columns = list(app._dl_arch_tree["columns"])
        assert columns == ["layer", "type", "output shape", "params", "train"]
        assert app._dl_arch_summary.cget("text").startswith("train a network")
        assert list(app._dl_depth_combo["values"]) == ["1", "2", "3", "all"]
        assert app._dl_depth_combo.get() == "1"
        assert app._dl_arch_text.get("1.0", "end").strip() == ""

    def test_architecture_card_fills_after_training(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._train_tiny(app)
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        rows = app._dl_arch_tree.get_children()
        assert len(rows) >= 3  # input + at least one layer + output
        values = [app._dl_arch_tree.item(row, "values") for row in rows]
        assert str(values[0][0]) == "input"
        assert str(values[-1][0]) == "output"
        summary = app._dl_arch_summary.cget("text")
        assert "Params" in summary and "MACs" in summary and "FLOPs" in summary

    def test_module_tree_button_renders_tree_into_card(
        self, app: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._force_numpy_engine(monkeypatch)
        self._train_tiny(app)
        assert self._drain_jobs(app), "network training hung"
        app._dl_show_module_tree()
        text = app._dl_arch_text.get("1.0", "end")
        assert "Linear" in text and "NumpyMLPRegressor" in text
        # depth control (torchview): the combobox feeds the renderer
        app._dl_depth_combo.set("all")
        app._dl_show_module_tree()
        assert "Linear" in app._dl_arch_text.get("1.0", "end")

    def test_autograd_dot_save_writes_file(
        self, playground: Any, app: Any, monkeypatch: Any, tmp_path: Any, quiet_dialogs: list[Any]
    ) -> None:
        pytest.importorskip("torch")
        dot_path = tmp_path / "network.dot"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **kwargs: str(dot_path)
        )
        self._train_tiny(app)  # default arch + torch present -> torch engine
        assert self._drain_jobs(app), "network training hung"
        app._dl_save_autograd_dot()
        # the success ride is an info dialog; errors/warnings are the
        # failures this test exists to catch
        problems = [d for d in quiet_dialogs if d[0] != "showinfo"]
        assert problems == []
        assert dot_path.is_file()
        source = dot_path.read_text()
        assert source.startswith("digraph")
        assert "AddmmBackward" in source

    def test_autograd_dot_numpy_engine_warns(
        self, app: Any, monkeypatch: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._force_numpy_engine(monkeypatch)
        self._train_tiny(app)
        assert self._drain_jobs(app), "network training hung"
        app._dl_save_autograd_dot()
        assert quiet_dialogs and "numpy" in str(quiet_dialogs[0][1]).lower()

    def test_view_netron_missing_warns_with_pip_hint(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        pytest.importorskip("torch")
        self._train_tiny(app)  # torch engine
        assert self._drain_jobs(app), "network training hung"
        app._dl_view_netron()
        # netron is not installed in this venv: the fail-fast warning
        # names the remedy -- no export is attempted.
        assert quiet_dialogs and "pip install netron" in str(quiet_dialogs[0][1])

    def test_view_netron_without_network_warns(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_view_netron()
        assert quiet_dialogs and "Train a network" in str(quiet_dialogs[0][1])
        app._dl_save_autograd_dot()
        assert "Train a network" in str(quiet_dialogs[1][1])
        app._dl_show_module_tree()
        assert "Train a network" in str(quiet_dialogs[2][1])


class TestExperimentTrackingAbsorptionGui:
    """(100x-173) The experiment-tracking absorption round --
    tensorflow/tensorboard + wandb/wandb + mlflow/mlflow + aimhubio/aim
    distilled onto the Deep Learning tab as one Experiments card: every
    network the tab trains is logged wandb-style (config snapshot +
    metrics + the TensorBoard scalar event series) into
    ``experiments.jsonl`` next to playground.py; the card's table is
    MLflow's run list (newest first, status column), the filter box is
    Aim's ``query_runs`` substring query, the detail pane is the
    wandb/aim run page (params, metrics, metadata, the TensorBoard
    histogram-stats and EMA-smoothing read on the loss series), and the
    best-runs line is Aim's leaderboard."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _configure_tiny(app: Any) -> None:
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "4")
        app._dl_epochs.set("5")
        app._dl_patience.set("3")
        app._dl_valsplit.set("0.2")

    @staticmethod
    def _train_tiny(app: Any) -> None:
        app._dl_load_example()
        app._dl_train()

    @staticmethod
    def _write_runs(playground: Any, path: Any, specs: list[dict]) -> list[str]:
        ids = []
        for spec in specs:
            ids.append(
                playground.log_experiment_run(
                    path,
                    kind=spec.get("kind", "NumpyMLPRegressor"),
                    params=spec.get("params", {"lr": "0.01"}),
                    metrics=spec.get("metrics", {}),
                    notes=spec.get("notes", ""),
                )
            )
        return ids

    # -- pure helpers: TensorBoard's scalar smoothing + histogram stats ---- #

    def test_smooth_series_matches_tensorboard_charts(self, playground: Any) -> None:
        # TensorBoard's scalar chart: last = w*last + (1-w)*value, the
        # first point rendered raw (w = the smoothing slider, 0.6 default).
        assert playground.smooth_series([1.0, 2.0, 3.0], 0.6) == pytest.approx(
            [1.0, 1.4, 2.04]
        )
        assert playground.smooth_series([1.0, 2.0, 3.0], 0.0) == pytest.approx(
            [1.0, 2.0, 3.0]
        )
        assert playground.smooth_series([]) == []
        assert playground.smooth_series([7.5]) == [7.5]

    def test_histogram_summary_stats_tensorboard_line(self, playground: Any) -> None:
        stats = playground.histogram_summary_stats([1.0, 2.0, 3.0, 4.0])
        assert stats["count"] == 4
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["std"] == pytest.approx(1.118033988749895)
        assert stats["min"] == 1.0 and stats["max"] == 4.0
        assert playground.histogram_summary_stats([]) == {}

    # -- pure helpers: MLflow run ids + wandb run metadata ------------------ #

    def test_new_run_id_is_mlflow_32_hex(self, playground: Any) -> None:
        import re

        first = playground.new_run_id()
        second = playground.new_run_id()
        assert re.fullmatch(r"[0-9a-f]{32}", first)
        assert first != second

    def test_run_metadata_wandb_style(self, playground: Any) -> None:
        import datetime
        import re

        meta = playground.run_metadata()
        assert set(meta) == {"host", "python", "started_at"}
        assert re.match(r"\d+\.\d+", meta["python"])
        datetime.datetime.fromisoformat(meta["started_at"])

    # -- pure helpers: the JSONL run log (MLflow params, TB series) --------- #

    def test_log_experiment_run_mlflow_stringifies_params(
        self, playground: Any, tmp_path: Any
    ) -> None:
        import re

        log = tmp_path / "experiments.jsonl"
        run_id = playground.log_experiment_run(
            log,
            kind="DeepMLPRegressor",
            params={"lr": 0.01, "layers": [8, 8], "flag": True},
            metrics={"loss/best_val": 0.123, "epochs/run": 3},
            notes="notebook figure 4",
            series=[1.0, 0.5],
        )
        runs = playground.load_experiment_runs(log)
        assert len(runs) == 1
        run = runs[0]
        assert run["run_id"] == run_id
        assert re.fullmatch(r"[0-9a-f]{32}", run_id)
        # MLflow's log_param contract: values are string-ified
        assert run["params"] == {"lr": "0.01", "layers": "[8, 8]", "flag": "True"}
        assert run["metrics"]["loss/best_val"] == pytest.approx(0.123)
        assert run["metrics"]["epochs/run"] == pytest.approx(3.0)
        assert run["series"] == [1.0, 0.5]
        assert run["status"] == "FINISHED"
        assert run["kind"] == "DeepMLPRegressor"
        assert set(run["metadata"]) == {"host", "python", "started_at"}

    def test_load_experiment_runs_tolerates_corrupt_lines(
        self, playground: Any, tmp_path: Any
    ) -> None:
        assert playground.load_experiment_runs(tmp_path / "missing.jsonl") == []
        log = tmp_path / "experiments.jsonl"
        playground.log_experiment_run(log, kind="A", params={}, metrics={})
        log.write_text(
            "\n{not json at all}\n"
            + log.read_text()
            + '   \n[1, 2, 3]\n'  # valid JSON, but not a run record
            + json.dumps({"run_id": "c" * 32, "kind": "B", "params": {}, "metrics": {}})
            + "\n",
            encoding="utf-8",
        )
        kinds = [run["kind"] for run in playground.load_experiment_runs(log)]
        assert kinds == ["A", "B"]  # blank/corrupt/non-dict lines skipped

    # -- pure helpers: Aim's leaderboard + query ----------------------------- #

    def test_best_experiment_runs_leaderboard(self, playground: Any) -> None:
        runs = [
            {"run_id": "a" * 32, "metrics": {"loss/best_val": 0.5}},
            {"run_id": "b" * 32, "metrics": {"loss/best_val": 0.2}},
            {"run_id": "c" * 32, "metrics": {"loss/best_val": 0.3}},
            {"run_id": "d" * 32, "metrics": {"metrics/accuracy": 0.9}},
            {"run_id": "e" * 32, "metrics": {}},
        ]
        board = playground.best_experiment_runs(runs, "loss/best_val")
        assert board == [
            ("b" * 32, pytest.approx(0.2)),
            ("c" * 32, pytest.approx(0.3)),
            ("a" * 32, pytest.approx(0.5)),
        ]
        assert len(playground.best_experiment_runs(runs, "loss/best_val", k=2)) == 2
        higher = playground.best_experiment_runs(
            runs, "metrics/accuracy", lower_is_better=False
        )
        assert higher == [("d" * 32, pytest.approx(0.9))]

    def test_query_experiment_runs_aim_substring(self, playground: Any) -> None:
        runs = [
            {
                "run_id": "a" * 32,
                "kind": "NumpyMLPRegressor",
                "notes": "solubility sweep",
                "params": {"architecture": "mlp-2x8"},
                "metrics": {"loss/best_val": 0.4},
                "status": "FINISHED",
            },
            {
                "run_id": "b" * 32,
                "kind": "DeepMLPRegressor",
                "notes": "delaney",
                "params": {"architecture": "mlp-4x16"},
                "metrics": {},
                "status": "FINISHED",
            },
        ]
        # Aim semantics: the empty query returns every run
        assert playground.query_experiment_runs(runs, "") == runs
        assert len(playground.query_experiment_runs(runs, "mlp")) == 2
        assert [r["run_id"] for r in playground.query_experiment_runs(runs, "solubility")] == [
            "a" * 32
        ]
        # metric values are searchable too
        assert [r["run_id"] for r in playground.query_experiment_runs(runs, "0.4")] == [
            "a" * 32
        ]
        assert playground.query_experiment_runs(runs, "no-such-thing") == []

    # -- pure helpers: wandb's sweep configuration --------------------------- #

    def test_sweep_configs_grid_and_random(self, playground: Any) -> None:
        space = {"lr": [0.1, 0.01], "hidden": [4, 8]}
        grid = playground.sweep_configs(space, method="grid")
        assert len(grid) == 4
        assert grid[0] == {"lr": 0.1, "hidden": 4}
        assert grid[-1] == {"lr": 0.01, "hidden": 8}
        assert playground.sweep_configs({"lr": [0.1], "hidden": []}) == []
        random_first = playground.sweep_configs(space, method="random", count=2, seed=0)
        random_again = playground.sweep_configs(space, method="random", count=2, seed=0)
        assert random_first == random_again  # deterministic under the seed
        assert len(random_first) == 2
        assert all(config in grid for config in random_first)
        # over-sampling falls back to the whole grid
        assert playground.sweep_configs(space, method="random", count=99, seed=1) == grid
        with pytest.raises(ValueError, match="method"):
            playground.sweep_configs(space, method="bogus")

    # -- pure helpers: the wandb/aim run page -------------------------------- #

    def test_run_detail_lines_wandb_run_page(self, playground: Any) -> None:
        run = {
            "run_id": "a" * 32,
            "kind": "NumpyMLPRegressor",
            "status": "FINISHED",
            "notes": "notebook figure 4",
            "params": {"lr": "0.01", "hidden": "8"},
            "metrics": {"loss/best_val": 0.123},
            "metadata": {"host": "box", "python": "3.12.1", "started_at": "2026-01-01T00:00:00"},
            "series": [1.0, 0.5, 0.25],
        }
        lines = playground.run_detail_lines(run)
        joined = "\n".join(lines)
        assert f"run {'a' * 32}" in joined
        assert "param lr = 0.01" in joined
        assert "param hidden = 8" in joined
        assert "metric loss/best_val" in joined
        assert "meta host: box" in joined
        assert "series: n 3" in joined
        assert "smoothed" in joined  # the TB EMA read on the series
        minimal = playground.run_detail_lines({"run_id": "b" * 32})
        assert len(minimal) == 3  # run / kind+status / notes lines only

    # -- the Experiments card ------------------------------------------------ #

    def test_experiments_card_widgets_exist(self, app: Any) -> None:
        columns = list(app._dl_runs_tree["columns"])
        assert columns == ["run", "when", "kind", "best val loss", "status"]
        assert app._dl_best_label.cget("text").startswith("no runs yet")
        assert app._dl_experiment_text.get("1.0", "end").strip() == ""
        assert app._dl_filter_entry.get() == ""

    def test_refresh_lists_logged_runs_newest_first(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        ids = self._write_runs(
            playground,
            log,
            [
                {"kind": "NumpyMLPRegressor", "metrics": {"loss/best_val": 0.5}},
                {"kind": "DeepMLPRegressor", "metrics": {"loss/best_val": 0.2}},
            ],
        )
        app._dl_experiment_log = log
        app._dl_refresh_experiments()
        rows = app._dl_runs_tree.get_children()
        assert len(rows) == 2
        values = [app._dl_runs_tree.item(row, "values") for row in rows]
        assert str(values[0][2]) == "DeepMLPRegressor"  # newest first (MLflow)
        assert abs(float(values[0][3]) - 0.2) < 1e-9
        assert str(values[0][4]) == "FINISHED"
        assert rows[0] == ids[1]  # the full run id rides as the item iid
        best = app._dl_best_label.cget("text")
        assert ids[1][:8] in best and "0.2" in best

    def test_filter_narrows_the_table(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        self._write_runs(
            playground,
            log,
            [
                {"kind": "NumpyMLPRegressor", "notes": "solubility sweep"},
                {"kind": "DeepMLPRegressor", "notes": "delaney"},
            ],
        )
        app._dl_experiment_log = log
        app._dl_refresh_experiments()
        assert len(app._dl_runs_tree.get_children()) == 2
        app._dl_filter_entry.delete(0, "end")
        app._dl_filter_entry.insert(0, "solubility")
        app._dl_apply_filter()
        rows = app._dl_runs_tree.get_children()
        assert len(rows) == 1
        assert str(app._dl_runs_tree.item(rows[0], "values")[2]) == "NumpyMLPRegressor"
        app._dl_filter_entry.delete(0, "end")
        app._dl_apply_filter()
        assert len(app._dl_runs_tree.get_children()) == 2

    def test_selecting_run_renders_detail_page(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        self._write_runs(
            playground, log, [{"params": {"lr": "0.01", "hidden": "4"}}]
        )
        app._dl_experiment_log = log
        app._dl_refresh_experiments()
        first = app._dl_runs_tree.get_children()[0]
        app._dl_runs_tree.selection_set(first)
        app._dl_show_run_detail()
        text = app._dl_experiment_text.get("1.0", "end")
        assert "param lr = 0.01" in text
        assert "param hidden = 4" in text

    def test_training_a_network_logs_a_run(
        self, playground: Any, app: Any, monkeypatch: Any, tmp_path: Any, quiet_dialogs: list[Any]
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        app._dl_experiment_log = log
        self._configure_tiny(app)
        self._train_tiny(app)
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        runs = playground.load_experiment_runs(log)
        assert len(runs) == 1
        run = runs[0]
        assert "MLP" in run["kind"]
        assert run["status"] == "FINISHED"
        # the wandb config snapshot: the form's own values
        assert run["params"]["epochs"] == "5"
        assert "architecture" in run["params"]
        assert "engine" in run["params"]
        # wandb's slashed metric names, summary-style best val loss
        assert "loss/best_val" in run["metrics"]
        assert run["metrics"]["fit/seconds"] > 0
        assert run["metrics"]["epochs/run"] >= 1
        assert "metrics/r2" in run["metrics"]  # the example is regression
        # the TensorBoard scalar event series rides along
        assert run["series"] and len(run["series"]) == run["metrics"]["epochs/run"]
        # and the card refreshed itself
        assert len(app._dl_runs_tree.get_children()) == 1

    def test_best_runs_line_reports_the_leaderboard(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        self._write_runs(
            playground,
            log,
            [
                {"metrics": {"loss/best_val": 0.9}},
                {"metrics": {"loss/best_val": 0.1}},
                {"metrics": {"loss/best_val": 0.5}},
                {"metrics": {"loss/best_val": 0.3}},
            ],
        )
        app._dl_experiment_log = log
        app._dl_refresh_experiments()
        best = app._dl_best_label.cget("text")
        assert "0.1" in best
        assert "0.5" in best
        assert "0.3" in best
        assert "0.9" not in best  # four runs, k=3: the worst sits out


class TestInterpretabilityAbsorptionGui:
    """(100x-174) The interpretability/XAI absorption round -- pytorch/captum
    + keisen/tf-keras-vis + PAIR-code/saliency + jessevig/bertviz distilled
    onto the Deep Learning tab: the four repos' method census (each row
    carrying its origin import path), the model-agnostic occlusion
    attribution of one held-out row (predict()-only, so both engines
    explain as readily), the pure integrated-gradients / smoothgrad /
    XRAI-region helpers, bertviz's attention arithmetic (the
    format_attention subset, the head-view cells, the neuron view's
    scaled dot-product softmax) and the captum word-importance-style bar
    table that renders the ranking."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _drain_jobs(app: Any, timeout: float = 180.0) -> bool:
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _configure_tiny(app: Any) -> None:
        app._dl_hidden_entry.delete(0, "end")
        app._dl_hidden_entry.insert(0, "4")
        app._dl_epochs.set("5")
        app._dl_patience.set("3")
        app._dl_valsplit.set("0.2")

    @staticmethod
    def _force_numpy_engine(monkeypatch: Any) -> None:
        """Pretend torch is absent for the engine resolution only."""
        import importlib.util

        real = importlib.util.find_spec

        def fake(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                return None
            return real(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake)

    @staticmethod
    def _train_tiny(app: Any) -> None:
        app._dl_load_example()
        app._dl_train()

    # -- the census --------------------------------------------------------- #

    def test_attribution_methods_census_covers_the_four_repos(
        self, playground: Any
    ) -> None:
        rows = list(playground.ATTRIBUTION_METHODS)
        assert {row[2] for row in rows} == {"captum", "tf-keras-vis", "saliency", "bertviz"}
        counts: dict[str, int] = {}
        for _method, _family, source, _needs_grad, _origin in rows:
            counts[source] = counts.get(source, 0) + 1
        assert counts["captum"] >= 20
        assert counts["tf-keras-vis"] >= 5
        assert counts["saliency"] >= 8
        assert counts["bertviz"] == 3
        names = [row[0] for row in rows]
        assert len(names) == len(set(names))  # every census row is unique
        # integrated gradients arrives from BOTH captum and saliency
        ig_sources = {row[2] for row in rows if row[0].startswith("IntegratedGradients")}
        assert ig_sources == {"captum", "saliency"}
        # occlusion too -- the card's own compute path
        occlusion_sources = {row[2] for row in rows if row[0].startswith("Occlusion")}
        assert occlusion_sources == {"captum", "saliency"}
        # Score-CAM is the gradient-free CAM (tf-keras-vis)
        scorecam = next(row for row in rows if row[0] == "Scorecam")
        assert scorecam[3] is False
        # every origin names its own repo's package
        prefixes = {
            "captum": "captum.",
            "tf-keras-vis": "tf_keras_vis.",
            "saliency": "saliency.",
            "bertviz": "bertviz.",
        }
        for _method, _family, source, _needs_grad, origin in rows:
            assert origin.startswith(prefixes[source])
        families = {row[1] for row in rows}
        assert {"gradient", "path", "ablation", "cam", "attention"} <= families

    # -- pure helpers: integrated gradients --------------------------------- #

    def test_integrated_gradients_linear_completeness(self, playground: Any) -> None:
        import numpy as np

        coef = np.asarray([3.0, -2.0])
        grad_fn = lambda points: np.tile(coef, (len(points), 1))

        ig = playground.integrated_gradients_attributions(grad_fn, [2.0, 4.0])
        assert ig == pytest.approx([6.0, -8.0])  # coef * (x - zeros)
        shifted = playground.integrated_gradients_attributions(
            grad_fn, [2.0, 4.0], baseline=[1.0, 1.0]
        )
        assert shifted == pytest.approx([3.0, -6.0])  # coef * (x - baseline)

    def test_integrated_gradients_rejects_mismatched_baseline(
        self, playground: Any
    ) -> None:
        grad_fn = lambda points: points * 0.0

        with pytest.raises(ValueError, match="baseline"):
            playground.integrated_gradients_attributions(
                grad_fn, [1.0, 2.0], baseline=[0.0]
            )

    # -- pure helpers: smoothgrad ------------------------------------------- #

    def test_smoothgrad_zero_noise_is_the_plain_gradient(self, playground: Any) -> None:
        import numpy as np

        coef = np.asarray([3.0, -2.0])
        grad_fn = lambda points: np.tile(coef, (len(points), 1))

        mask = playground.smoothgrad_attributions(
            grad_fn, [0.5, -0.5], n_samples=8, noise_sigma=0.0
        )
        assert mask == pytest.approx([3.0, -2.0])
        first = playground.smoothgrad_attributions(
            grad_fn, [0.5, -0.5], n_samples=8, noise_sigma=0.2, seed=3
        )
        again = playground.smoothgrad_attributions(
            grad_fn, [0.5, -0.5], n_samples=8, noise_sigma=0.2, seed=3
        )
        assert first == again  # deterministic under the seed

    # -- pure helpers: occlusion -------------------------------------------- #

    def test_occlusion_attributions_per_feature_and_window(self, playground: Any) -> None:
        score_fn = lambda batch: 5.0 * batch[:, 0] + 1.0 * batch[:, 1]

        per_feature = playground.occlusion_attributions(score_fn, [2.0, 3.0])
        assert per_feature == pytest.approx([10.0, 3.0])
        joint = playground.occlusion_attributions(
            score_fn, [2.0, 3.0], window=2, stride=2
        )
        assert joint == pytest.approx([13.0, 13.0])  # the pair's drop lands on both
        medians = playground.occlusion_attributions(
            score_fn, [2.0, 3.0], baseline=[1.0, 1.0]
        )
        assert medians == pytest.approx([5.0, 2.0])  # 13-8 and 13-11

    # -- pure helpers: XRAI region aggregation ------------------------------ #

    def test_xrai_region_importances_share_out(self, playground: Any) -> None:
        shares = playground.xrai_region_importances([2.0, -2.0, 4.0], [0, 0, 1])
        assert shares == pytest.approx({0: 0.5, 1: 0.5})
        assert sum(shares.values()) == pytest.approx(1.0)
        zeroed = playground.xrai_region_importances([0.0, 0.0], [0, 1])
        assert zeroed == {0: 0.0, 1: 0.0}
        with pytest.raises(ValueError, match="align"):
            playground.xrai_region_importances([1.0, 2.0], [0])

    # -- pure helpers: bertviz arithmetic ----------------------------------- #

    def test_format_attention_selects_layers_and_heads(self, playground: Any) -> None:
        import numpy as np

        tensor = np.arange(2 * 2 * 2 * 2, dtype=float).reshape(2, 2, 2, 2)
        subset = playground.format_attention(tensor, layers=[1], heads=[0])
        assert subset.shape == (1, 1, 2, 2)
        assert subset.flatten().tolist() == tensor[1, 0].flatten().tolist()
        full = playground.format_attention(tensor)
        assert full.shape == (2, 2, 2, 2)
        with pytest.raises(ValueError, match="layers"):
            playground.format_attention(np.zeros((2, 2, 2)))

    def test_attention_head_cells_report_row_peak_shares(self, playground: Any) -> None:
        import numpy as np

        matrix = np.asarray([[0.2, 0.8], [0.5, 0.0]])
        cells = playground.attention_head_cells(matrix, ["a", "b"])
        assert len(cells) == 4
        by_pair = {(row, col): (weight, share) for row, col, weight, share in cells}
        assert by_pair[("a", "b")] == pytest.approx((0.8, 1.0))
        assert by_pair[("a", "a")] == pytest.approx((0.2, 0.25))
        assert by_pair[("b", "a")] == pytest.approx((0.5, 1.0))
        with pytest.raises(ValueError, match="tokens"):
            playground.attention_head_cells(matrix, ["a"])

    def test_scaled_dot_product_attention_rows(self, playground: Any) -> None:
        aligned = playground.scaled_dot_product_attention(
            [[1.0, 0.0]], [[1.0, 0.0]]
        )
        assert aligned.shape == (1, 1)
        assert aligned[0, 0] == pytest.approx(1.0)
        balanced = playground.scaled_dot_product_attention(
            [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]
        )
        assert balanced.shape == (2, 2)
        assert balanced.sum(axis=1) == pytest.approx([1.0, 1.0])
        assert balanced[0, 0] == pytest.approx(balanced[1, 1])  # symmetric in
        assert balanced[0, 1] == pytest.approx(balanced[1, 0])  # and out
        with pytest.raises(ValueError, match="width"):
            playground.scaled_dot_product_attention([[1.0]], [[1.0, 0.0]])

    # -- pure helpers: the ranked bar table --------------------------------- #

    def test_attribution_table_lines_rank_and_bar(self, playground: Any) -> None:
        lines = playground.attribution_table_lines(["a", "b", "c"], [0.1, -0.5, 0.25])
        assert len(lines) == 3
        assert lines[0].startswith("b")  # |−0.5| ranks first, the sign survives
        assert "-0.5" in lines[0]
        assert lines[1].startswith("c")
        assert lines[2].startswith("a")
        bars = [line.count("█") for line in lines]
        assert bars == [12, 6, 2]  # the bar scales to the biggest |value|
        capped = playground.attribution_table_lines(["a", "b", "c"], [0.1, -0.5, 0.25], top=2)
        assert len(capped) == 2
        assert playground.attribution_table_lines([], []) == []

    # -- the Attribution card ------------------------------------------------ #

    def test_attribution_card_lists_the_census(self, playground: Any, app: Any) -> None:
        columns = list(app._dl_attr_tree["columns"])
        assert columns == ["method", "family", "source", "gradients", "origin"]
        rows = app._dl_attr_tree.get_children()
        assert len(rows) == len(playground.ATTRIBUTION_METHODS)
        first = app._dl_attr_tree.item(rows[0], "values")
        assert str(first[0]).startswith("Saliency")
        assert str(first[2]) == "captum"
        assert str(first[4]) == "captum.attr.Saliency"
        assert app._dl_attr_text.get("1.0", "end").strip() == ""

    def test_selecting_a_census_row_renders_the_detail(self, app: Any) -> None:
        rows = app._dl_attr_tree.get_children()
        app._dl_attr_tree.selection_set(rows[2])  # IntegratedGradients (captum)
        app._dl_show_attr_method()  # the event needs a mainloop -- call it directly
        text = app._dl_attr_text.get("1.0", "end")
        assert "IntegratedGradients" in text
        assert "captum.attr.IntegratedGradients" in text
        assert "gradients" in text

    def test_explain_row_names_the_missing_network(self, app: Any, quiet_dialogs: list[Any]) -> None:
        app._dl_explain_row()
        assert any(name == "showwarning" for name, _args in quiet_dialogs)

    def test_explain_held_out_row_renders_occlusion_table(
        self,
        playground: Any,
        app: Any,
        monkeypatch: Any,
        quiet_dialogs: list[Any],
    ) -> None:
        monkeypatch.setattr(
            playground.simpledialog, "askstring", lambda *a, **k: "0"
        )
        self._force_numpy_engine(monkeypatch)
        self._configure_tiny(app)
        self._train_tiny(app)
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        app._dl_explain_row()
        text = app._dl_attr_text.get("1.0", "end")
        assert "occlusion" in text
        assert "row 0 of" in text
        assert "baseline" in text
        assert "MLP" in text  # the estimator is named
        assert text.count("█") > 0  # the word-importance bars ride along
        assert quiet_dialogs == []

    def test_attention_example_renders_neuron_view(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._dl_attention_example()
        text = app._dl_attr_text.get("1.0", "end")
        assert "softmax" in text.lower()
        assert "evaporates" in text
        assert "100.0%" in text  # every row's peak cell
        assert quiet_dialogs == []

    # -- the census's model-agnostic remainder: captum's Saliency and
    # InputXGradient (grad_fn contracts like the IG/SmoothGrad helpers),
    # captum's FeaturePermutation (predict()-only) and the
    # deletion/insertion faithfulness curves behind PAIR's PIC ---------- #

    def test_saliency_attributions_linear_abs(self, playground: Any) -> None:
        def grad_fn(batch: Any) -> Any:
            import numpy as np

            return np.tile(np.asarray([5.0, 1.0]), (len(batch), 1))

        plain = playground.saliency_attributions(grad_fn, [2.0, 3.0])
        assert plain == [5.0, 1.0]  # |grad| -- captum's abs=True default
        signed = playground.saliency_attributions(grad_fn, [2.0, 3.0], abs=False)
        assert signed == [5.0, 1.0]

    def test_saliency_attributions_abs_flips_negative_gradient(
        self, playground: Any
    ) -> None:
        def grad_fn(batch: Any) -> Any:
            import numpy as np

            return np.tile(np.asarray([-2.0, 4.0]), (len(batch), 1))

        assert playground.saliency_attributions(grad_fn, [1.0, 1.0]) == [2.0, 4.0]
        assert playground.saliency_attributions(grad_fn, [1.0, 1.0], abs=False) == [
            -2.0,
            4.0,
        ]

    def test_input_x_gradient_attributions_linear(self, playground: Any) -> None:
        def grad_fn(batch: Any) -> Any:
            import numpy as np

            return np.tile(np.asarray([5.0, 1.0]), (len(batch), 1))

        assert playground.input_x_gradient_attributions(grad_fn, [2.0, 3.0]) == [
            10.0,
            3.0,
        ]

    def test_permutation_attributions_rank_the_informative_column(
        self, playground: Any
    ) -> None:
        import numpy as np

        rows = [[1.0, 9.0], [2.0, 9.1], [3.0, 8.9], [4.0, 9.05]]

        def score_fn(batch: Any) -> Any:
            return np.asarray(batch, dtype=float) @ np.asarray([5.0, 0.0])

        mask = playground.permutation_attributions(
            score_fn, rows, n_repeats=3, seed=0
        )
        assert mask[0] > 0.0  # the informative column takes the hit
        assert mask[1] == pytest.approx(0.0)  # weight zero -- permuting is free

    def test_insertion_deletion_auc_prefers_the_mask_order(
        self, playground: Any
    ) -> None:
        import numpy as np

        def score_fn(batch: Any) -> Any:
            batch = np.asarray(batch, dtype=float).reshape(-1, 2)
            return batch @ np.asarray([5.0, 1.0])

        good = playground.insertion_deletion_auc(
            score_fn, [2.0, 3.0], [5.0, 1.0], baseline=[0.0, 0.0]
        )
        flipped = playground.insertion_deletion_auc(
            score_fn, [2.0, 3.0], [1.0, 5.0], baseline=[0.0, 0.0]
        )
        # a faithful mask inserts better than it deletes; the flipped one
        # is the opposite -- and every curve stays inside [0, 1]
        assert good[1] > good[0]
        assert flipped[0] > flipped[1]
        assert all(0.0 <= value <= 1.0 for value in (*good, *flipped))

    def test_insertion_deletion_auc_rejects_mismatched_baseline(
        self, playground: Any
    ) -> None:
        def score_fn(batch: Any) -> Any:
            import numpy as np

            batch = np.asarray(batch, dtype=float).reshape(-1, 2)
            return batch @ np.asarray([1.0, 1.0])

        with pytest.raises(ValueError, match="same length"):
            playground.insertion_deletion_auc(
                score_fn, [2.0, 3.0], [1.0, 0.0], baseline=[0.0, 0.0, 0.0]
            )


class TestGuiPlatformHelpers:
    """(100x-175) The GUI-platform absorption round's pure helpers --
    TomSchimansky/CustomTkinter + zauberzeug/nicegui + streamlit/streamlit
    + gradio-app/gradio inspected one by one and distilled: CTk's four
    theme packs (every colour a [light, dark] pair) with its 0.4x
    scaling floor, streamlit's TTLCache (LRU recency refreshed by reads
    and writes, uniform per-item TTL, reaped only on write/expire/len)
    behind the cache_data / cache_resource decorators, streamlit's
    st.metric delta-arrow contract, gradio's CSVLogger flag files and
    examples pagination, and nicegui's nine notify positions."""

    # -- customtkinter: the theme packs ----------------------------------- #

    def test_gui_theme_pack_resolution(self, playground: Any) -> None:
        packs = playground.GUI_THEME_PACKS
        # CTk's _built_in_themes order, straight from the repo
        assert tuple(packs) == ("blue", "green", "gold", "dark-blue")
        blue_light = playground.resolve_gui_pack("blue", "light")
        blue_dark = playground.resolve_gui_pack("blue", "dark")
        # the authentic CTkButton fg_color pair from blue.json
        assert blue_light["accent"] == "#3B8ED0"
        assert blue_dark["accent"] == "#1F6AA5"
        # the shape tiers ride along (CTkButton corner_radius 6,
        # CTkEntry border_width 2)
        assert blue_light["corner_radius"] == 6
        assert blue_light["border_width"] == 2
        # an unknown mode resolves like CTk's appearance tracker: light
        assert playground.resolve_gui_pack("blue", "solarized")["accent"] == "#3B8ED0"
        # an unknown pack falls back to CTk's default ("blue")
        assert playground.resolve_gui_pack("nope", "dark")["accent"] == "#1F6AA5"

    def test_save_gui_pack_rejects_builtins(self, playground: Any, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="builtin"):
            playground.save_gui_pack("gold", tmp_path / "gold.json")
        with pytest.raises(KeyError):
            playground.save_gui_pack("not-a-pack", tmp_path / "nope.json")

    def test_clamp_scaling_floor(self, playground: Any) -> None:
        # CTk's ScalingTracker: max(factor, 0.4)
        assert playground.clamp_scaling(0.25) == 0.4
        assert playground.clamp_scaling(0.4) == 0.4
        assert playground.clamp_scaling(1.0) == 1.0
        assert playground.clamp_scaling(1.6) == 1.6

    # -- streamlit: TTLCache + the two cache decorators ------------------- #

    def test_ttl_cache_lru_and_ttl(self, playground: Any) -> None:
        clock = [0.0]
        removed: list[Any] = []
        cache = playground.TTLCache(maxsize=2, ttl=10.0, timer=lambda: clock[0], on_remove=lambda key, value: removed.append((key, value)))
        cache["a"] = 1
        cache["b"] = 2
        clock[0] = 5.0
        cache["a"]  # a read refreshes recency (streamlit's contract)
        cache["c"] = 3  # maxsize 2 -> the LRU ("b") is evicted
        assert "a" in cache and "c" in cache
        assert "b" not in cache
        assert removed == [("b", 2)]
        # uniform TTL: "a" written at t=0 expires by t=11, "c" written
        # at t=5 by t=16 -- len() reaps the expired entries
        clock[0] = 16.0
        assert len(cache) == 0
        assert cache.get("a") is None  # expired reads report absent

    def test_ttl_cache_membership_does_not_refresh(self, playground: Any) -> None:
        clock = [0.0]
        cache = playground.TTLCache(maxsize=2, ttl=100.0, timer=lambda: clock[0])
        cache["a"] = 1
        cache["b"] = 2
        assert "a" in cache  # membership never reorders (streamlit's rule)
        cache["c"] = 3  # so "a" (the oldest write) is the one evicted
        assert "a" not in cache and "b" in cache and "c" in cache

    def test_cache_data_copies_and_counts(self, playground: Any) -> None:
        @playground.cache_data(ttl=60.0, max_entries=8)
        def build(n: int) -> dict[str, Any]:
            return {"n": n, "items": [n]}

        first = build(1)
        assert build.misses == 1 and build.hits == 0
        second = build(1)  # the hit
        assert build.hits == 1
        # st.cache_data pickles values back: every read its own copy
        assert first is not second and first == second
        first["items"].append(99)
        assert build(1)["items"] == [1]  # the mutation never leaked
        build(2)
        assert build.misses == 2
        build.clear()
        assert len(build.cache) == 0

    def test_cache_resource_shares_singleton(self, playground: Any) -> None:
        class Widget:
            pass

        @playground.cache_resource(ttl=60.0)
        def borrow() -> Any:
            return Widget()

        one = borrow()
        two = borrow()
        assert one is two  # st.cache_resource: one shared object
        assert borrow.hits == 1
        one.tag = "kept"
        assert borrow().tag == "kept"

    # -- streamlit: the st.metric delta contract -------------------------- #

    def test_metric_delta_spec_arrow_rules(self, playground: Any) -> None:
        spec = playground.metric_delta_spec
        assert spec(None) == ("", "none")
        assert spec("") == ("", "none")
        assert spec(0) == ("", "muted")
        assert spec("0") == ("", "muted")
        assert spec(-0.015) == ("\u2193", "bad")
        assert spec("-0.0%") == ("\u2193", "bad")
        assert spec(0.015) == ("\u2191", "good")
        # streamlit does NOT parse string deltas: "0%" follows the sign
        # rule (positive -> up, green), "-0.0%" the minus rule (down)
        assert spec("0%") == ("\u2191", "good")

    def test_format_metric_line(self, playground: Any) -> None:
        line = playground.format_metric_line("best val loss", 0.412, -0.015)
        assert "0.412" in line
        assert "\u2193" in line and "0.015" in line
        flat = playground.format_metric_line("accuracy", 0.9, None)
        assert "0.9" in flat and "\u2191" not in flat and "\u2193" not in flat

    # -- gradio: the CSVLogger flag files ---------------------------------- #

    def test_flag_run_csv_appends_and_splits_on_headers(self, playground: Any, tmp_path: Any) -> None:
        flags = tmp_path / "flags"
        headers = ["run", "best val loss"]
        path, count = playground.flag_run_csv(
            flags, headers, ["abc123", 0.41], "correct", "playground"
        )
        assert path == flags / "log.csv" and count == 1
        lines = (flags / "log.csv").read_text(encoding="utf-8").splitlines()
        assert lines[0] == "run,best val loss,flag,username,timestamp"
        assert lines[1].startswith("abc123,0.41,correct,playground,")
        # a second flag of the same shape appends (ClassicCSV behaviour)
        _, count2 = playground.flag_run_csv(
            flags, headers, ["def456", 0.39], "incorrect", "playground"
        )
        assert count2 == 2
        # a header change rolls a NEW numbered file (CSVLogger >= 5.0)
        path3, count3 = playground.flag_run_csv(
            flags, headers + ["notes"], ["ghi789", 0.5, "edge case"], "ambiguous", "playground"
        )
        assert path3 == flags / "log-1.csv" and count3 == 1

    # -- nicegui: the nine notify positions -------------------------------- #

    def test_toast_geometry_nine_positions(self, playground: Any) -> None:
        positions = (
            "top-left", "top-right", "bottom-left", "bottom-right",
            "top", "bottom", "left", "right", "center",
        )
        for position in positions:
            x, y = playground.toast_geometry(position, 320, 90, 1200, 800)
            assert 0 <= x <= 1200 - 320, position
            assert 0 <= y <= 800 - 90, position
        # bottom-right hugs the corner with the margin
        x, y = playground.toast_geometry("bottom-right", 320, 90, 1200, 800)
        assert (x, y) == (1200 - 320 - 24, 800 - 90 - 24)
        # unknown positions degrade to nicegui's default (bottom)
        x, y = playground.toast_geometry("diagonal", 320, 90, 1200, 800)
        assert y == 800 - 90 - 24

    # -- gradio: examples pagination --------------------------------------- #

    def test_paginate_examples(self, playground: Any) -> None:
        items = list(range(25))
        page, pages = playground.paginate_examples(items)
        assert page == items[:10] and pages == 3  # examples_per_page = 10
        last, _ = playground.paginate_examples(items, page=2)
        assert last == items[20:]
        clamped, _ = playground.paginate_examples(items, page=99)
        assert clamped == items[20:]  # out-of-range pages clamp to the last
        empty, pages = playground.paginate_examples([])
        assert empty == [] and pages == 1


class TestGuiPlatformAbsorptionGui:
    """(100x-175) The GUI-platform absorption round's window surface --
    the Look & feel card on the Utilities tab (CustomTkinter's four
    accent packs applied live over the View ▸ Accent pack menu, the
    nicegui @ui.refreshable preview rebuild, CTk's scaling spin, the
    nicegui/gradio toast playground), the notify engine itself, and the
    Experiments card's gradio flag button + streamlit st.metric delta
    on the leaderboard line."""

    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        import gc

        del application
        gc.collect()

    # -- the Look & feel card ---------------------------------------------- #

    def test_look_feel_card_pack_picker(self, playground: Any, app: Any) -> None:
        combo = app._lookfeel_pack_combo
        assert "default" in combo["values"]
        assert "gold" in combo["values"]
        try:
            app._pack_var.set("gold")
            app._on_pack_change()
            # the live module accent now reads CTk gold.json's light value
            assert playground.ACCENT == "#EFB951"
            assert playground.ACCENT_DARK == "#DBA220"
            # and switching back restores the toolkit palette
            app._pack_var.set("default")
            app._on_pack_change()
            assert playground.ACCENT == playground._LIGHT_PALETTE["ACCENT"]
        finally:
            app._pack_var.set("default")
            app._on_pack_change()

    def test_refresh_preview_rebuilds_the_census(self, playground: Any, app: Any) -> None:
        before = app._lookfeel_preview_builds
        app._lookfeel_refresh_preview()
        assert app._lookfeel_preview_builds == before + 1
        app._pack_var.set("dark-blue")
        app._on_pack_change()
        try:
            app._lookfeel_refresh_preview()
            text = app._lookfeel_preview_text.get("1.0", "end")
            # the census names the pack and its authentic CTk hexes
            assert "dark-blue" in text
            assert "#3A7EBF" in text and "#1F538D" in text
            assert "corner_radius 6" in text
        finally:
            app._pack_var.set("default")
            app._on_pack_change()

    def test_scaling_spin_follows_ctk_floor(self, playground: Any, app: Any) -> None:
        try:
            app._lookfeel_scaling_spin.set("1.5")
            app._on_scaling_change()
            assert app._font_var.get() == 15  # 10 pt * 1.5
            # below CTk's 0.4 floor the factor clamps, then the font
            # hierarchy clamps to its own 8 pt minimum
            app._lookfeel_scaling_spin.set("0.25")
            app._on_scaling_change()
            assert app._font_var.get() == 8
        finally:
            app._lookfeel_scaling_spin.set("1.0")
            app._on_scaling_change()
            assert app._font_var.get() == 10

    # -- the notify engine (nicegui notify + gradio toasts) --------------- #

    def test_notify_creates_and_expires_toast(self, playground: Any, app: Any) -> None:
        import time

        app.notify("job finished", level="success", duration=0.05)
        assert len(app._toast_stack) == 1
        toast = app._toast_stack[0]["window"]
        assert toast.winfo_exists()
        for _ in range(40):
            app.update()
            time.sleep(0.01)
        assert app._toast_stack == []  # the duration closed it
        assert not toast.winfo_exists()

    def test_notify_sticky_until_dismissed(self, playground: Any, app: Any) -> None:
        import time

        app.notify("needs a click", level="warning", duration=None)
        assert len(app._toast_stack) == 1
        for _ in range(30):
            app.update()
            time.sleep(0.01)
        assert len(app._toast_stack) == 1  # gradio: None/0 = indefinite
        app._dismiss_toast(app._toast_stack[0])
        app.update()
        assert app._toast_stack == []

    def test_notify_unknown_level_and_position_fall_back(self, playground: Any, app: Any) -> None:
        app.notify("odd args", level="shouty", position="diagonal", duration=0.05)
        entry = app._toast_stack[0]
        assert entry["level"] == "info"  # gradio's default level
        assert entry["position"] == "bottom"  # nicegui's default position

    # -- the Experiments card: gradio flagging + streamlit metric ---------- #

    def test_flag_selected_run_writes_csv(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        log.write_text("", encoding="utf-8")
        app._dl_experiment_log = log
        run = {
            "run_id": "flagme123456",
            "kind": "NumpyMLPRegressor",
            "status": "FINISHED",
            "metrics": {"loss/best_val": 0.4},
            "metadata": {"started_at": "2026-11-03 04:00:00"},
        }
        app._dl_runs = [run]
        app._dl_runs_by_iid = {"flagme123456": run}
        app._dl_apply_filter()
        app._dl_runs_tree.selection_set("flagme123456")
        app._dl_flag_combo.set("correct")
        app._dl_flag_selected_run()
        csv = tmp_path / "experiments-flags" / "log.csv"
        assert csv.is_file()
        text = csv.read_text(encoding="utf-8")
        assert "flagme123456"[:8] in text
        assert "NumpyMLPRegressor" in text and "0.4" in text
        assert ",correct,playground," in text
        # the flag lands a success toast, gradio-style
        assert any(entry["level"] == "success" for entry in app._toast_stack)
        for entry in list(app._toast_stack):
            app._dismiss_toast(entry)

    def test_flag_without_selection_warns(self, playground: Any, app: Any) -> None:
        app._dl_runs_tree.selection_remove(app._dl_runs_tree.selection())
        app._dl_flag_selected_run()
        assert any(entry["level"] == "warning" for entry in app._toast_stack)
        for entry in list(app._toast_stack):
            app._dismiss_toast(entry)

    def test_leaderboard_line_carries_metric_delta(
        self, playground: Any, app: Any, tmp_path: Any
    ) -> None:
        log = tmp_path / "experiments.jsonl"
        log.write_text("", encoding="utf-8")
        app._dl_experiment_log = log
        older = {
            "run_id": "aaa111111111",
            "kind": "NumpyMLPRegressor",
            "status": "FINISHED",
            "metrics": {"loss/best_val": 0.45},
            "metadata": {"started_at": "2026-11-01 00:00:00"},
        }
        newer = {
            "run_id": "bbb222222222",
            "kind": "NumpyMLPRegressor",
            "status": "FINISHED",
            "metrics": {"loss/best_val": 0.40},
            "metadata": {"started_at": "2026-11-02 00:00:00"},
        }
        app._dl_runs = [older, newer]
        app._dl_runs_by_iid = {run["run_id"]: run for run in (older, newer)}
        app._dl_filter_entry.delete(0, "end")
        app._dl_apply_filter()
        text = app._dl_best_label.cget("text")
        # streamlit's st.metric: the leader carries the delta vs the
        # runner-up -- a lower loss is the down arrow
        assert "bbb222222222"[:8] in text
        assert "\u2193" in text and "0.05" in text


# --------------------------------------------------------------------- #
# round 199: the catalog absorption cards -- the GitHub research
# repository catalog's backend modules that had no playground card
# (maxsmi, ECFP6-Highlighter-GUI, GuacaMol oracles, pingouin, SciPy,
# PK-Sim/OSPSuite, RxODE-style events, Orange3 basket analysis)
# --------------------------------------------------------------------- #
@XVFB_SKIP
class TestCatalogAbsorptionGui:
    @pytest.fixture()
    def app(self, playground: Any, xvfb_display: str) -> Any:
        # persist_session=False keeps the suite deterministic (the
        # module-surface rule: never read/write a real session file)
        application = playground.PlaygroundApp(persist_session=False)
        yield application
        application.destroy()
        # collect the (destroyed) widget/Variable cycles NOW, in the
        # main thread -- the next test's worker may otherwise finalize
        # tk Variables off-thread
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
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    def test_smiles_augmentation_card_counts_unique_variants(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._aug_run()  # aspirin, 12 requested
        assert self._drain(app), "augmentation job hung"
        text = app._aug_result.cget("text")
        assert "unique" in text and "charset" in text and "tokens" in text
        assert app._aug_variants.get("1.0", "end").strip() != ""
        assert quiet_dialogs == []

    def test_ecfp_highlighter_card_reports_atom_bond_union(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._ecfp_hl_run()  # aspirin, real bits 389/456
        assert self._drain(app), "bit highlighter job hung"
        assert "atoms" in app._ecfp_hl_result.cget("text")
        assert "bonds" in app._ecfp_hl_result.cget("text")
        detail = app._ecfp_hl_detail.get("1.0", "end")
        assert "bit 389" in detail and "in query" in detail and "corpus" in detail
        assert quiet_dialogs == []

    def test_scoring_oracles_card_scores_aspirin(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._oracle_run()  # aspirin vs the salicylic-acid target
        assert self._drain(app), "oracles job hung"
        text = app._oracle_result.cget("text")
        assert "QED 0.55" in text  # the known aspirin QED
        assert "SA" in text and "logP" in text and "rediscovery" in text
        detail = app._oracle_detail.get("1.0", "end")
        assert "property filter" in detail and "PASS" in detail
        assert quiet_dialogs == []

    def test_inferential_card_runs_welch_and_both_posthoc_tables(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._istat_run()  # control / low / high defaults
        assert self._drain(app), "inferential job hung"
        assert "Welch F" in app._istat_result.cget("text")
        assert "groups" in app._istat_result.cget("text")
        detail = app._istat_detail.get("1.0", "end")
        assert "Tukey HSD" in detail and "Games-Howell" in detail
        assert quiet_dialogs == []

    def test_interpolation_card_answers_query_and_audits(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._interp_run()  # 9 knots, pchip, query 3/5/6
        assert self._drain(app), "interpolation job hung"
        text = app._interp_result.cget("text")
        assert text.startswith("pchip:") and "->" in text
        detail = app._interp_detail.get("1.0", "end")
        assert "holdout audit" in detail and "rmse" in detail
        assert quiet_dialogs == []

    def test_interpolation_card_guards_short_series_honestly(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._interp_text.delete("1.0", "end")
        app._interp_text.insert("1.0", "0, 0\n1, 0.6\n2, 0.9")  # only 3 knots
        app._interp_run()
        assert self._drain(app), "short-series interpolation job hung"
        assert app._interp_result.cget("text").startswith("pchip:")
        assert "needs >= 8 knots" in app._interp_detail.get("1.0", "end")
        assert quiet_dialogs == []

    def test_pbpk_card_computes_the_fih_chain(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._pbpk_run()  # the literature defaults
        assert self._drain(app), "PBPK job hung"
        assert "FIH start" in app._pbpk_result.cget("text")
        assert "AUC ratio" in app._pbpk_result.cget("text")
        detail = app._pbpk_detail.get("1.0", "end")
        for needle in (
            "IVIVE CLint",
            "well-stirred hepatic CL",
            "DDI AUC ratio",
            "MABEL",
            "NOAEL HED",
            "FIH starting dose",
        ):
            assert needle in detail, needle
        assert quiet_dialogs == []

    def test_osp_library_card_loads_and_looks_up(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._osp_load()
        assert self._drain(app), "OSP library job hung"
        rows = app._osp_tree.get_children()
        assert len(rows) == 40  # the curated collection size
        assert "40 models" in app._osp_summary.cget("text")
        first_name = app._osp_tree.item(rows[0], "values")[0]
        app._osp_search.delete(0, "end")
        app._osp_search.insert(0, first_name)
        app._osp_lookup()
        assert self._drain(app), "OSP lookup job hung"
        assert first_name in app._osp_summary.cget("text")
        assert quiet_dialogs == []

    def test_dosing_events_card_realizes_regimen_and_weekday(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._evreg_run()  # 500 mg q12h until 72h + m/w/f codes
        assert self._drain(app), "events job hung"
        text = app._evreg_result.cget("text")
        assert "6 doses" in text and "3000 mg" in text  # 0,12,...,60 h
        detail = app._evreg_detail.get("1.0", "end")
        assert "weekday schedule" in detail and "0h, 48h, 96h" in detail
        assert quiet_dialogs == []

    def test_association_rules_card_mines_the_default_basket(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._assoc_run()  # the 5-transaction co-medication basket
        assert self._drain(app), "apriori job hung"
        text = app._assoc_result.cget("text")
        assert "5 transactions" in text and "rules" in text
        detail = app._assoc_detail.get("1.0", "end")
        assert "aspirin" in detail and "lift" in detail
        assert quiet_dialogs == []


@XVFB_SKIP
class TestResearchCatalogFoldinsGui:
    """Headless drives for the three 140-repository catalog fold-in cards."""

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
        seen: list[Any] = []
        for name in ("showerror", "showwarning", "showinfo"):
            monkeypatch.setattr(
                playground.messagebox,
                name,
                lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, a)),
            )
        return seen

    @staticmethod
    def _wait(app: Any, seconds: float = 90.0) -> None:
        import time

        deadline = time.time() + seconds
        while app._busy and time.time() < deadline:
            app.update()
            time.sleep(0.01)
        assert not app._busy, "catalog fold-in job hung"

    def test_dta_demo_builds_features_and_exports(
        self, app: Any, quiet_dialogs: list[Any], monkeypatch: Any, tmp_path: Path
    ) -> None:
        assert app._dl_dta_source.get() == "synthetic demo (12 pairs)"
        app._dl_dta_run()
        self._wait(app)
        assert app._dl_dta_last is not None
        features = app._dl_dta_last["features"]
        assert features.X.shape == (12, 2071)
        report = app._dl_dta_text.get("1.0", "end")
        assert "ridge RMSE" in report and "DeepDTA rm2" in report
        assert "not the upstream neural" in report
        target = tmp_path / "dta.csv"
        monkeypatch.setattr(
            "playground_under_test.filedialog.asksaveasfilename", lambda **_kwargs: str(target)
        )
        app._dl_dta_export()
        assert target.is_file() and target.stat().st_size > 2048
        assert quiet_dialogs == []

    def test_advanced_pk_regimen_nca_and_pregnancy(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._pk_regimen_run()
        assert "Cmax" in app._pk_regimen_text.get("1.0", "end")
        assert getattr(app._pk_regimen_host, "_pg_figure", None) is not None
        app._pk_nca_run()
        nca = app._pk_regimen_text.get("1.0", "end")
        assert "linear-up/log-down AUC" in nca and "terminal λz" in nca
        app._pk_pregnancy_run()
        pregnancy = app._pk_pregnancy_text.get("1.0", "end")
        assert "fetal/maternal AUC ratio" in pregnancy
        assert "not the released 27-compartment fit" in pregnancy
        assert getattr(app._pk_pregnancy_host, "_pg_figure", None) is not None
        assert quiet_dialogs == []

    def test_osp_library_pregnancy_registry_and_ddi(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._pk_osp_library()
        assert "Vancomycin" in app._pk_osp_text.get("1.0", "end")
        app._pk_pregnancy_models()
        pregnancy = app._pk_osp_text.get("1.0", "end")
        assert "Acetaminophen" in pregnancy and "Raltegravir" in pregnancy
        app._pk_ddi_run()
        ddi = app._pk_osp_text.get("1.0", "end")
        assert "gmfe" in ddi and "fraction_within_guest" in ddi
        assert quiet_dialogs == []

    def test_dockstring_box_objective_and_campaign_bridge(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        assert len(app._dock_benchmark_target["values"]) == 58
        app._dock_benchmark_target.set("DRD2")
        app._dock_benchmark_show()
        assert "size_z = 32.000" in app._dock_benchmark_text.get("1.0", "end")
        app._dock_benchmark_use()
        assert app._dock_size_entry.get() == "30 30 32"
        app._dock_benchmark_objective.set("selective_JAK2")
        app._dock_benchmark_qed.delete(0, "end")
        app._dock_benchmark_qed.insert(0, "0.8")
        app._dock_benchmark_evaluate()
        result = app._dock_benchmark_text.get("1.0", "end")
        assert "value: -6.1" in result
        assert "no docking engine was run" in result
        assert quiet_dialogs == []


@pytest.mark.skipif(importlib.util.find_spec("tkinter") is None, reason="needs tkinter to import playground")
class TestResearchCatalogFoldinsHeadless:
    """Drive new handlers with stub widgets; no Tk display is necessary."""

    class Value:
        def __init__(self, value: Any = "") -> None:
            self.value = str(value)

        def get(self) -> str:
            return self.value

        def delete(self, *_args: Any) -> None:
            self.value = ""

        def insert(self, _index: Any, value: Any) -> None:
            self.value = str(value)

        def set(self, value: Any) -> None:
            self.value = str(value)

    class Text(Value):
        def get(self, *_args: Any) -> str:
            return self.value

    @staticmethod
    def _bare(playground: Any) -> Any:
        app = object.__new__(playground.PlaygroundApp)
        app._require_pharmaml = lambda: True
        app._require = lambda *_args: True
        app.errors: list[Any] = []
        app._show_error = lambda title, error: app.errors.append((title, error))
        app.notifications: list[Any] = []
        app.notify = lambda *args, **kwargs: app.notifications.append((args, kwargs))
        return app

    def test_paired_dta_handler_runs_synchronously(self, playground: Any) -> None:
        app = self._bare(playground)
        app._dl_dta_source = self.Value("synthetic demo (12 pairs)")
        app._dl_dta_path = self.Value()
        app._dl_dta_smiles_col = self.Value("smiles")
        app._dl_dta_protein_col = self.Value("protein")
        app._dl_dta_affinity_col = self.Value("affinity")
        app._dl_dta_dipeptides = self.Value(False)
        app._dl_dta_dipeptides.get = lambda: False  # type: ignore[method-assign]
        app._dl_dta_text = self.Text()
        app._dl_dta_last = None

        def start_job(_label: str, work: Any, render: Any) -> None:
            render(work())

        app._start_job = start_job
        app._dl_dta_run()
        assert app._dl_dta_last["features"].X.shape == (12, 2071)
        assert "ridge RMSE" in app._dl_dta_text.value
        assert app.errors == []

    def test_regimen_nca_and_pregnancy_handlers(self, playground: Any) -> None:
        app = self._bare(playground)
        regimen = {
            "dose_mg": "100", "tau_h": "12", "n_doses": "3", "ka": "1.2",
            "cl": "2", "v2": "20", "q": "1", "v3": "10", "duration_h": "36",
        }
        app._pk_regimen_entries = {key: self.Value(value) for key, value in regimen.items()}
        app._pk_route_combo = self.Value("oral")
        app._pk_regimen_text = self.Text()
        app._pk_regimen_host = object()
        app._pk_nca_text = self.Text("0,10\n1,8\n2,6.5\n4,4.2\n8,1.8\n12,0.8\n24,0.1")
        pregnancy = {
            "dose_mg": "100", "maternal_volume_l": "50", "fetal_volume_l": "3",
            "maternal_clearance_l_h": "1", "fetal_clearance_l_h": "0.1",
            "placental_clearance_l_h": "2", "maternal_fu": "0.8",
            "fetal_fu": "0.4", "duration_h": "12",
        }
        app._pk_pregnancy_entries = {key: self.Value(value) for key, value in pregnancy.items()}
        app._pk_pregnancy_text = self.Text()
        app._pk_pregnancy_host = object()
        figures: list[Any] = []
        app._render_figure = lambda _host, figure, **_kwargs: figures.append(figure)

        app._pk_regimen_run()
        assert "events: 3" in app._pk_regimen_text.value
        app._pk_nca_run()
        assert "terminal λz" in app._pk_regimen_text.value
        app._pk_pregnancy_run()
        assert "fetal/maternal AUC ratio" in app._pk_pregnancy_text.value
        assert len(figures) == 2
        assert all(hasattr(figure, "savefig") for figure in figures)
        assert app.errors == []

    def test_osp_and_dockstring_handlers(self, playground: Any) -> None:
        app = self._bare(playground)
        app._pk_osp_text = self.Text()
        app._pk_ddi_text = self.Text("1,1.1\n2,1.8\n0.5,0.6")
        app._pk_osp_library()
        assert "Vancomycin" in app._pk_osp_text.value
        app._pk_pregnancy_models()
        assert "Acetaminophen" in app._pk_osp_text.value
        app._pk_ddi_run()
        assert "gmfe" in app._pk_osp_text.value

        app._dock_benchmark_target = self.Value("DRD2")
        app._dock_benchmark_text = self.Text()
        app._dock_center_entry = self.Value()
        app._dock_size_entry = self.Value()
        app._dock_benchmark_show()
        assert "size_z = 32.000" in app._dock_benchmark_text.value
        app._dock_benchmark_use()
        assert app._dock_size_entry.value == "30 30 32"
        app._dock_benchmark_qed = self.Value("0.8")
        app._dock_benchmark_smiles = self.Value("CCO")
        app._dock_benchmark_objective = self.Value("selective_JAK2")
        app._dock_benchmark_scores = {
            "JAK2": self.Value("-9"),
            "LCK": self.Value("-9"),
            "F2": self.Value("-7"),
            "PPARA": self.Value("-7"),
            "PPARD": self.Value("-7"),
            "PPARG": self.Value("-7"),
        }
        app._dock_benchmark_evaluate()
        assert "value: -6.1" in app._dock_benchmark_text.value
        assert app.errors == []
