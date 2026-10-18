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
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
            "probe_dependencies", "load_pharmaml", "parse_float", "parse_int",
            "parse_smiles_lines", "format_number", "build_objective",
            "toolkit_version", "PlaygroundApp", "main", "self_test",
            "print_banner", "apply_theme", "session_path", "dump_session",
            "load_session", "parse_geometry",
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
        for raw_line in (TOOLKIT_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        ).splitlines():
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
        assert modules == {
            "numpy", "pandas", "sklearn", "scipy", "matplotlib", "rdkit", "torch"
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
    def test_probe_dependencies_returns_booleans_for_every_module(
        self, playground: Any
    ) -> None:
        status = playground.probe_dependencies()
        assert set(status) == {
            "numpy", "pandas", "sklearn", "scipy", "matplotlib", "rdkit", "torch"
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
            "autogrid4", "autodock4",
        )
        assert playground.docking_engine_binaries("Vina single ligand") == ("vina",)


# --------------------------------------------------------------------- #
# session persistence + themes (pure helpers -- no Tk required)
# --------------------------------------------------------------------- #
class TestSessionHelpers:
    def test_session_path_lives_under_home_dot_pharmaml(
        self, playground: Any
    ) -> None:
        path = playground.session_path()
        assert path.name == "playground-session.json"
        assert path.parent == Path.home() / ".pharmaml"

    def test_parse_geometry_accepts_only_full_specs(self, playground: Any) -> None:
        assert playground.parse_geometry("1180x780+100+50") == (1180, 780, 100, 50)
        assert playground.parse_geometry(" 900x600-20+0 ") == (900, 600, -20, 0)
        for broken in ("", "1180x780", "+100+50", "1180x780+100", "junk",
                       "0x0+0+0extra", "1180xx780+1+1"):
            assert playground.parse_geometry(broken) is None, broken

    def test_dump_session_normalises_every_field(self, playground: Any) -> None:
        payload = playground.dump_session(
            theme="neon", geometry=1180, tab=True,
            variables={"_a": "x", 5: "y"},
            families={"descriptors": 1, "ecfp": None},
            texts={"pasted": "CCO", "corpus": 42},
        )
        assert payload["format"] == playground.SESSION_FORMAT
        assert payload["theme"] == "light"          # unknown theme -> light
        assert payload["geometry"] == "1180"          # coerced to str
        assert payload["tab"] == 1                    # bool -> int
        assert payload["variables"] == {"_a": "x", "5": "y"}
        assert payload["families"] == {"descriptors": True, "ecfp": False}
        assert payload["texts"] == {"pasted": "CCO", "corpus": "42"}

    def test_load_session_rejects_malformed_payloads(self, playground: Any) -> None:
        for raw in (None, [], "session", {}, {"format": 0},
                    {"format": "1"}, {"format": 2, "theme": "dark"}):
            assert playground.load_session(raw) is None, raw

    def test_load_session_sanitises_half_formed_payloads(
        self, playground: Any
    ) -> None:
        loaded = playground.load_session({
            "format": playground.SESSION_FORMAT,
            "theme": "solarized", "geometry": None, "tab": -4,
            "variables": "not a dict", "families": {"descriptors": "yes"},
            "texts": {"pasted": "ok", "corpus": 7},
        })
        assert loaded == {
            "format": playground.SESSION_FORMAT,
            "theme": "light", "geometry": "", "tab": 0,
            "variables": {}, "families": {}, "texts": {"pasted": "ok"},
        }

    def test_dump_load_round_trip_is_lossless(self, playground: Any) -> None:
        payload = playground.dump_session(
            theme="dark", geometry="1180x780+10+20", tab=4,
            variables={"_aff_value": "3.14", "_tune_var": True,
                       "_feat_k_entry": "777"},
            families={"descriptors": True, "morgan_scaffold": False},
            texts={"pasted": "CCO\nCCN", "corpus": ""},
        )
        assert playground.load_session(payload) == payload

    def test_apply_theme_rebinds_and_restores_module_colours(
        self, playground: Any
    ) -> None:
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

    def test_format_operator_stats_renders_levels_and_probabilities(
        self, playground: Any
    ) -> None:
        text = playground._format_operator_stats({
            "operator_stats": {
                "level1": {
                    "clone": {"draws": 24, "successes": 20, "final_probability": 0.893},
                },
                "mutation": {
                    "topology": {"draws": 200, "successes": 180, "final_probability": 0.896},
                },
            }
        })
        assert "[level1]" in text and "[mutation]" in text
        assert "clone" in text and "p 0.893" in text
        assert "topology" in text and "p 0.896" in text

    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None, reason="needs RDKit"
    )
    def test_build_objective_qed_has_score(self, playground: Any) -> None:
        objective = playground.build_objective("QED drug-likeness", 1.0, 3.0)
        assert callable(objective.score)

    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None, reason="needs RDKit"
    )
    def test_build_objective_logp_window_scores_inside_band(self, playground: Any) -> None:
        objective = playground.build_objective("LogP window", 1.0, 3.0)
        aspirin = "CC(=O)Oc1ccccc1C(=O)O"  # LogP ~ 1.31 -- inside the band
        assert objective.score(aspirin) > 0.0

    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None, reason="needs RDKit"
    )
    def test_build_objective_blend_is_multiobjective(self, playground: Any) -> None:
        from pharmaml.generative import MultiObjective

        objective = playground.build_objective("QED + LogP blend", 1.0, 3.0)
        assert isinstance(objective, MultiObjective)


# --------------------------------------------------------------------- #
# the cell-line pipeline the Cell Lines tab drives (same pharmaml calls,
# no Tk -- pins the shipped example data's flagship facts)
# --------------------------------------------------------------------- #
class TestCellLinePipeline:
    @pytest.mark.skipif(
        importlib.util.find_spec("scipy") is None, reason="needs SciPy"
    )
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

    @pytest.mark.skipif(
        importlib.util.find_spec("scipy") is None, reason="needs SciPy"
    )
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
        expression = pd.read_csv(
            data / "cellline_expression.csv", index_col="cell_line"
        )
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
    @pytest.mark.skipif(
        importlib.util.find_spec("rdkit") is None, reason="needs RDKit"
    )
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
        gpf = write_gpf(ligand, receptor_file="r.pdbqt", box=box,
                        output_path=ligand.parent / "r.gpf")
        dpf = write_dpf(ligand, receptor_file="r.pdbqt",
                        output_path=ligand.parent / "r.dpf",
                        ga_run=100, ga_pop_size=150)
        assert gpf.is_file() and dpf.is_file()

    @pytest.mark.skipif(
        importlib.util.find_spec("pandas") is None, reason="needs pandas"
    )
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
            [{"rank": c.rank, "lowest_energy": c.lowest_energy,
              "mean_energy": c.mean_energy, "num_in_cluster": c.num_in_cluster,
              "n_heavy": 13} for c in clusters]
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
            check=False, capture_output=True, text=True, timeout=120,
            cwd=TOOLKIT_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert "pharmaml" in result.stdout
        assert "numpy" in result.stdout and "RDKit" in result.stdout

    def test_version_prints_both_versions(self) -> None:
        result = subprocess.run(
            [sys.executable, str(PLAYGROUND_PATH), "--version"],
            check=False, capture_output=True, text=True, timeout=60,
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
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            check=False, capture_output=True, text=True, timeout=180,
            cwd=TOOLKIT_ROOT, env={**os.environ, "DISPLAY": xvfb_display},
        )
        assert result.returncode == 0, result.stderr
        assert "no errors" in result.stdout

    def test_example_corpus_loader_fills_the_text_widget(
        self, playground: Any, app: Any
    ) -> None:
        app._load_example_corpus()
        lines = [line for line in app._corpus_text.get("1.0", "end").splitlines()
                 if line.strip()]
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
                playground.messagebox, name,
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

    def test_cellline_matrix_job_fills_the_tree(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        app._bio_load_example()
        app._bio_build_matrix()
        assert self._drain_jobs(app), "matrix job hung"
        rows = app._bio_matrix_tree.get_children()
        assert len(rows) == 23  # 23 cell lines in the example screen
        assert "23 cell lines × 8 compounds" in app._bio_matrix_info.cget("text")
        assert app._bio_matrix is not None and app._bio_matrix.shape == (23, 8)
        assert quiet_dialogs == []

    def test_cellline_sensitive_resistant_calls(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
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
                playground.messagebox, name,
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
            "Home", "Chemistry", "PK / PD", "Cell Lines", "Docking",
            "Data & ML", "Generative",
        ]

    def test_docking_example_library_loads(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
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
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
    ) -> None:
        app._dock_load_receptor_example()
        target = tmp_path / "receptor.pdbqt"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
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

    def test_docking_example_campaign_scan(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
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
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
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
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(vina_log)
        )
        app._dock_open_vina()
        assert self._drain_jobs(app), "vina parse job hung"
        rows = app._dock_results_tree.get_children()
        assert len(rows) == 4
        values = app._dock_results_tree.item(rows[0], "values")
        assert values[0] == "1"  # mode number
        assert "4 modes" in app._dock_results_info.cget("text")
        assert quiet_dialogs == []

    def test_docking_run_gate_reports_missing_binary(
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
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

    @pytest.mark.filterwarnings(
        "ignore:This process.*multi-threaded.*fork:DeprecationWarning"
    )
    def test_docking_run_starts_job_with_fake_binary(
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
    ) -> None:
        root = tmp_path / "campaign"
        ligand_dir = root / "01_lig"
        ligand_dir.mkdir(parents=True)
        (ligand_dir / "r.gpf").write_text("# stub\n", encoding="utf-8")
        (ligand_dir / "r.dpf").write_text("# stub\n", encoding="utf-8")
        app._dock_campaign_root = root
        monkeypatch.setattr(
            playground.shutil, "which",
            lambda name: "/usr/bin/true" if name in ("autogrid4", "autodock4") else None,
        )
        app._dock_run()
        assert app._busy  # the gate passed and the job started
        assert self._drain_jobs(app), "fake-binary run job hung"
        summary = app._dock_run_summary.cget("text")
        assert "1 folders docked" in summary  # one stub ligand directory
        assert "1 failed" in summary  # /usr/bin/true produces no .dlg
        assert quiet_dialogs == []


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
            theme="light", geometry="1000x700+40000+50000", tab=0,
            variables={"_aff_value": "1.5"}, families={}, texts={},
        )
        (isolated_home / ".pharmaml").mkdir(parents=True)
        (isolated_home / ".pharmaml" / "playground-session.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8")
        app = playground.PlaygroundApp()
        self._drain(app)
        assert app._aff_value.get() == "1.5"      # inputs restored
        assert app.winfo_width() <= 1000           # size honoured (Xvfb 1280x800)
        # no window manager under Xvfb pins the window at +0+0; what
        # matters is that the OFF-SCREEN +40000+50000 was dropped
        assert "+40000" not in app.geometry()      # position dropped
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

    def test_chemistry_tab_source_references_the_feature_factory(
        self, playground: Any
    ) -> None:
        import inspect

        source = inspect.getsource(playground.PlaygroundApp._build_chemistry_tab)
        assert "Feature factory" in source
        assert "_feat_compute" in source

    def test_feature_factory_width_sums_family_bits(self, playground: Any) -> None:
        width = playground.feature_factory_width(
            (
                "descriptors", "atompair_count", "ecfp", "avalon_count",
                "rdkit_fingerprint", "morgan_full", "morgan_scaffold",
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
        assert (
            playground.detect_smiles_column(_Frame(["Canonical_Smiles"]))
            == "Canonical_Smiles"
        )
        assert playground.detect_smiles_column(_Frame(["molecule", "activity"])) is None

    def test_ranker_and_task_label_maps_cover_every_combobox_value(
        self, playground: Any
    ) -> None:
        assert set(playground._FEAT_RANKER_METHODS) == set(playground._FEAT_RANKERS)
        assert (
            playground._FEAT_RANKER_METHODS["Variance (no target needed)"]
            == "variance"
        )
        assert playground._FEAT_RANKER_METHODS["Random forest"] == "random_forest"
        assert (
            playground._FEAT_RANKER_METHODS["Auto (RandomForest or LightGBM)"]
            == "auto"
        )
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
                playground.messagebox, name,
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

    def test_default_pasted_variance_run(self, app: Any,
                                         quiet_dialogs: list[Any]) -> None:
        # the shipped defaults: pasted drug SMILES, all 7 families ticked
        assert app._feat_source_var.get() == "Pasted SMILES"
        assert "7 families selected" in app._feat_width_label.cget("text")
        assert (
            f"~{10_752 + self._descriptor_count():,} columns" in
            app._feat_width_label.cget("text")
        )
        app._feat_k_entry.delete(0, "end")
        app._feat_k_entry.insert(0, "50")
        app._feat_ranker_combo.set("Variance (no target needed)")
        app._feat_compute()
        assert self._drain_jobs(app), "feature factory job hung"
        summary = app._feat_summary.cget("text")
        assert f"{10_752 + self._descriptor_count():,} features" in summary
        assert "3 molecules" in summary
        assert "median-survivors" in summary
        assert "variance" in summary
        assert "selected per family" in app._feat_family_label.cget("text")
        assert tuple(app._feat_tree["columns"]) == (
            "rank", "feature", "family", "importance",
        )
        rows = app._feat_tree.get_children()
        assert len(rows) == 50
        first = app._feat_tree.item(rows[0], "values")
        assert first[0] == "1"  # rank column counts from 1
        families_seen = {app._feat_tree.item(row, "values")[2] for row in rows}
        assert families_seen <= {
            "descriptors", "atompair_count", "ecfp", "avalon_count",
            "rdkit_fingerprint", "morgan_full", "morgan_scaffold",
        }
        assert quiet_dialogs == []

    def test_exports_rar_style_csv_and_metadata_json(
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
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
            "written to" in " ".join(str(part) for part in args)
            for _kind, args in quiet_dialogs
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

    def test_unchecked_families_narrow_the_run(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
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
        self, playground: Any, app: Any, quiet_dialogs: list[Any],
        monkeypatch: Any, tmp_path: Any,
    ) -> None:
        import pandas as pd

        csv_path = tmp_path / "tiny_screen.csv"
        csv_path.write_text(
            "SMILES,activity\n"
            "C,0\nCC,0\nCCC,0\nCCCC,1\nCCCCC,1\nCCCCCC,1\n",
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
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target)
        )
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
            theme="dark", geometry="1200x800+10+20", tab=3,
            variables={"_aff_value": "42"}, families={"ecfp": False},
            texts={"pasted": "CCO\n# note"})
        target = tmp_path / "shared-session.json"
        assert playground.export_session(payload, target) is True
        revived = playground.load_session(playground.read_json_file(target))
        assert revived == payload  # identical after the file round-trip

    def test_export_session_unwritable_path(
        self, playground: Any, tmp_path: Path
    ) -> None:
        payload = playground.dump_session(
            theme="light", geometry="", tab=0, variables={}, families={},
            texts={})
        assert playground.export_session(
            payload, tmp_path / "no" / "such" / "dir" / "s.json") is False

    def test_read_json_file_missing_and_invalid(
        self, playground: Any, tmp_path: Path
    ) -> None:
        assert playground.read_json_file(tmp_path / "absent.json") is None
        junk = tmp_path / "junk.json"
        junk.write_text("{not json at all", encoding="utf-8")
        assert playground.read_json_file(junk) is None

    # -- live GUI under Xvfb -------------------------------------------- #

    def test_export_writes_shareable_file(
        self, playground: Any, xvfb_display: str, isolated_home: Path,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "9.9")
        target = isolated_home / "colleague-session.json"
        monkeypatch.setattr(
            playground.filedialog, "asksaveasfilename", lambda **_k: str(target))
        app._export_session()
        assert target.is_file()
        payload = playground.load_session(playground.read_json_file(target))
        assert payload is not None
        assert payload["variables"]["_aff_value"] == "9.9"
        assert payload["theme"] in ("light", "dark", "auto")
        assert "session exported" in str(app._job_label.cget("text"))
        app.destroy()

    def test_import_applies_shareable_file(
        self, playground: Any, xvfb_display: str, isolated_home: Path,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(source))
        app._import_session()
        assert app._aff_value.get() == "9.9"
        assert "session imported" in str(app._job_label.cget("text"))
        app.destroy()

    def test_import_invalid_file_shows_error_and_keeps_state(
        self, playground: Any, xvfb_display: str, isolated_home: Path,
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        app._aff_value.delete(0, "end")
        app._aff_value.insert(0, "7.7")
        junk = tmp_path / "not-a-session.json"
        junk.write_text('{"hello": "world"}', encoding="utf-8")
        seen: list[Any] = []
        monkeypatch.setattr(
            playground.filedialog, "askopenfilename", lambda **_k: str(junk))
        monkeypatch.setattr(
            playground.messagebox, "showerror",
            lambda *args, **_k: seen.append(args))
        app._import_session()
        assert app._aff_value.get() == "7.7"  # untouched
        assert seen and "Invalid session file" in str(seen[0][0])
        assert "imported" not in str(app._job_label.cget("text"))
        app.destroy()

    def test_reset_current_tab_restores_only_that_tab(
        self, playground: Any, xvfb_display: str, isolated_home: Path,
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
            index for index, names in app._tab_input_owners.items()
            if "_aff_value" in names)
        app._book.select(chemistry_tab)
        app._reset_current_tab()
        assert app._aff_value.get() == default_aff  # reset to default
        assert bool(app._tune_var.get()) is (not default_tune)  # other tab kept
        assert "tab reset" in str(app._job_label.cget("text"))
        app.destroy()

    def test_tab_ownership_map_excludes_theme_and_covers_families(
        self, playground: Any, xvfb_display: str, isolated_home: Path,
    ) -> None:
        app = playground.PlaygroundApp()
        app.update_idletasks()
        owners = app._tab_input_owners
        every_name = {name for names in owners.values() for name in names}
        assert "_theme_var" not in every_name  # not a user input
        assert not any(
            name.startswith("_book") for name in every_name)
        family_tabs = [
            index for index, names in owners.items()
            if any(name.startswith("family:") for name in names)]
        assert family_tabs == [1]  # the Chemistry tab owns the checkboxes
        text_tabs = [
            index for index, names in owners.items()
            if any(name.startswith("text:") for name in names)]
        assert 1 in text_tabs  # pasted SMILES corpus lives on Chemistry
        app.destroy()
