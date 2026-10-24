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
        assert modules == {"numpy", "pandas", "sklearn", "scipy", "matplotlib", "rdkit", "torch"}


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
        assert set(status) == {
            "numpy",
            "pandas",
            "sklearn",
            "scipy",
            "matplotlib",
            "rdkit",
            "torch",
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
            "Deep Learning",
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

        # a loaded table but an empty columns entry -> input error
        app._dock_load_example_campaign()
        assert self._drain_jobs(app), "campaign scan job hung"
        app._dock_rescore_columns_entry.delete(0, "end")
        app._dock_rescore()
        assert len(quiet_dialogs) == 2
        kind, args = quiet_dialogs[1]
        assert kind == "showerror"
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
        # the shipped defaults: pasted drug SMILES, all 8 families ticked
        assert app._feat_source_var.get() == "Pasted SMILES"
        assert "8 families selected" in app._feat_width_label.cget("text")
        assert f"~{10_752 + 10 + self._descriptor_count():,} columns" in app._feat_width_label.cget(
            "text"
        )
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "50")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app), "feature factory job hung"
        summary = app._feat_summary.cget("text")
        assert f"{10_752 + 10 + self._descriptor_count():,} features" in summary
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
        families_seen = {app._feat_tree.item(row, "values")[2] for row in rows}
        assert families_seen <= {
            "descriptors",
            "atompair_count",
            "ecfp",
            "avalon_count",
            "rdkit_fingerprint",
            "morgan_full",
            "morgan_scaffold",
            "rism",  # the 8th family (1.2.49)
        }
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
        deep, residual = playground._DL_ARCHITECTURES
        assert playground.deep_estimator_name(deep, "regression") == "DeepMLPRegressor"
        assert playground.deep_estimator_name(deep, "classification") == "DeepMLPClassifier"
        assert playground.deep_estimator_name(residual, "regression") == "ResidualMLPRegressor"
        assert playground.deep_estimator_name(residual, "classification") == "ResidualMLPClassifier"
        with pytest.raises(ValueError, match="architecture"):
            playground.deep_estimator_name("CNN", "regression")
        with pytest.raises(ValueError, match="task"):
            playground.deep_estimator_name(deep, "auto")

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
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        self._configure_tiny(app)
        app._dl_load_example()
        app._dl_train()
        assert self._drain_jobs(app), "network training hung"
        assert quiet_dialogs == []
        texts = self._metric_texts(app)
        assert "DeepMLPRegressor (regression)" in texts
        assert "R²" in texts and "RMSE" in texts
        assert "epochs run" in texts and "best val loss" in texts
        assert "hidden layers" in texts and "8" in texts
        assert app._dl_host._pg_figure is not None  # parity figure
        assert app._dl_hist_host._pg_figure is not None  # loss curves
        summary = app._dl_hist_summary.cget("text")
        assert "DeepMLPRegressor ·" in summary
        assert "epochs run" in summary and "best val loss" in summary
        assert app._dl_history  # the export payload is live
        assert app._dl_history[0]["epoch"] == 1
        assert set(app._dl_history[0]) == {"epoch", "train_loss", "val_loss"}
        assert any(entry["val_loss"] is not None for entry in app._dl_history)

    def test_residual_classifier_run_uses_the_right_estimator(
        self, app: Any, quiet_dialogs: list[Any]
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
        assert "ResidualMLPClassifier (classification)" in texts
        assert "accuracy" in texts and "MCC" in texts
        # SMILES + compound_id are not numeric -> reported as dropped
        assert "dropped non-numeric" in texts
        assert "SMILES" in texts and "compound_id" in texts
        assert app._dl_host._pg_figure is not None  # confusion matrix
        assert app._dl_hist_host._pg_figure is not None  # loss curves

    # -- batch normalisation (100x-90) --------------------------------------- #
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
        # This pins the whole tab: 15 saved keys, 13 mutated values
        # restored verbatim onto a freshly built window.
        first = playground.PlaygroundApp(persist_session=False)
        second = playground.PlaygroundApp(persist_session=False)
        try:
            first._dl_arch_combo.set("Residual MLP (skip connections)")
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
            payload = first._collect_session()
            saved = [key for key in payload["variables"] if key.startswith("_dl_")]
            assert len(saved) == 16  # every input on the tab (CV folds since 121)

            second._session = payload
            second._restore_session()
            assert second._dl_arch_combo.get() == "Residual MLP (skip connections)"
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
        finally:
            first.destroy()
            second.destroy()


# --------------------------------------------------------------------- #
# the smart target-column default (100x-61) -- never train on row IDs
# --------------------------------------------------------------------- #
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
    ) -> None:
        import os
        import stat

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
        assert len(app._ad_host.winfo_children()) == 1  # figure embedded
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
        assert len(app._nca_host.winfo_children()) == 1
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
        assert len(app._fam_host.winfo_children()) == 1
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
        assert len(app._feat_host.winfo_children()) == 1
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
        assert len(app._ml_host.winfo_children()) == 1
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
        assert len(app._util_cmp_host.winfo_children()) == 1
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
        assert len(app._dl_curve_host.winfo_children()) == 1
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
        # the prefill also filtered the list: NCA cards only
        labels = [str(tree.item(row, "values")[0]) for row in tree.get_children()]
        assert labels and all("NCA" in label for label in labels), labels
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
        app._menu_save_session_report()
        assert any("No figures yet" in str(a) for a in quiet_dialogs)

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
        assert any("figure page(s) written" in str(a) for a in quiet_dialogs)
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
