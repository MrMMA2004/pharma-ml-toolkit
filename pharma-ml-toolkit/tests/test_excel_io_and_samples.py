"""Round 201 -- the Excel I/O absorption and the sample-data-ready tabs.

The owner's directive: every playground tab must carry sample data
ready to run, and the playground must speak Excel (.xlsx) input AND
output beside CSV.  These tests pin both halves:

* the library arms -- ``export_table``'s sixth format ``xlsx``, the
  ``write_table`` workbook arm, and the indexed expression reader's
  Excel branch (label index + positional fallback);
* the playground wiring -- every dataset dialog offers the Excel
  patterns (``DATA_FILE_TYPES``), the shared save dialog writes
  workbooks, and the DTA / UMAP / canonical-SMILES exporters honour a
  ``.xlsx`` path;
* the per-tab sample-data story -- the NCA concentration-time series
  and the basket transactions arrive prefilled, the Unsupervised /
  Best-of-ML / Utilities one-click loaders land real bundled tables,
  the Home quick-start loads an example AND jumps to its tab, and the
  round-201 jump-chip fix resolves every chip by TITLE (the old index
  math silently sent three chips to wrong tabs).
"""

from __future__ import annotations

import contextlib
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
PLAYGROUND_PATH = TOOLKIT_ROOT / "playground.py"


def _load_playground() -> Any:
    """Import playground.py by path (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location("playground_excel_test", PLAYGROUND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["playground_excel_test"] = module
    return module


@pytest.fixture(scope="module")
def playground() -> Any:
    return _load_playground()


# --------------------------------------------------------------------- #
# the library arms
# --------------------------------------------------------------------- #
class TestExcelLibraryArms:
    def test_table_formats_pin_includes_xlsx(self) -> None:
        from pharmaml.viz import TABLE_FORMATS

        assert "xlsx" in TABLE_FORMATS
        assert TABLE_FORMATS == ("csv", "tsv", "md", "json", "html", "xlsx")

    def test_export_table_xlsx_round_trip(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table
        from pharmaml.viz import export_table

        frame = pd.DataFrame(
            {"drug": ["aspirin", "ibuprofen", "caffeine"], "pIC50": [3.19, 3.97, 4.08]}
        )
        target = tmp_path / "nested" / "table.xlsx"
        written = export_table(frame, target)
        assert written == target and target.is_file()
        back = read_table(target)
        assert list(back.columns) == ["drug", "pIC50"]
        assert len(back) == 3
        assert back["drug"].tolist() == frame["drug"].tolist()

    def test_export_table_unsupported_extension_still_raises(self, tmp_path: Path) -> None:
        from pharmaml.exceptions import DataValidationError
        from pharmaml.viz import export_table

        with pytest.raises(DataValidationError):
            export_table(pd.DataFrame({"a": [1]}), tmp_path / "table.parquet")

    def test_write_table_xlsx_round_trip(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_table, write_table

        frame = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"], "qed": [0.41, 0.60]})
        written = write_table(frame, tmp_path / "w.xlsx")
        assert written.is_file()
        back = read_table(written)
        assert list(back.columns) == ["smiles", "qed"] and len(back) == 2

    def test_indexed_reader_xlsx_label_index(self, tmp_path: Path) -> None:
        from pharmaml.utils.io import read_indexed_table_with_progress

        frame = pd.DataFrame(
            {"cell_line": ["A549", "MCF7", "NCI-H460"], "EGFR": [1.0, 2.0, 3.0]}
        )
        target = tmp_path / "expr.xlsx"
        frame.to_excel(target, index=False)
        got = read_indexed_table_with_progress(target, lambda *_a: None, index_col="cell_line")
        assert list(got.index) == ["A549", "MCF7", "NCI-H460"]
        assert list(got.columns) == ["EGFR"]

    def test_indexed_reader_xlsx_label_missing_fallback(self, tmp_path: Path) -> None:
        """No ``cell_line`` column: the FIRST column becomes the index."""
        from pharmaml.utils.io import read_indexed_table_with_progress

        frame = pd.DataFrame({"drug": ["a", "b"], "score": [1.0, 2.0]})
        target = tmp_path / "plain.xlsx"
        frame.to_excel(target, index=False)
        got = read_indexed_table_with_progress(target, lambda *_a: None, index_col="cell_line")
        assert list(got.index) == ["a", "b"]
        assert list(got.columns) == ["score"]

    def test_indexed_reader_csv_path_unchanged(self, tmp_path: Path) -> None:
        """The CSV twin keeps byte-for-byte behaviour (no regression)."""
        from pharmaml.utils.io import read_indexed_table_with_progress

        target = tmp_path / "expr.csv"
        frame = pd.DataFrame(
            {"cell_line": ["A549", "MCF7"], "EGFR": [1.0, 2.0], "KRAS": [3.0, 4.0]}
        )
        frame.to_csv(target, index=False)
        got = read_indexed_table_with_progress(target, lambda *_a: None, index_col="cell_line")
        assert list(got.index) == ["A549", "MCF7"]
        assert list(got.columns) == ["EGFR", "KRAS"]


# --------------------------------------------------------------------- #
# the playground wiring (module surface, no window)
# --------------------------------------------------------------------- #
class TestPlaygroundExcelWiring:
    def test_data_file_types_constant_offers_excel(self, playground: Any) -> None:
        patterns = playground.DATA_FILE_TYPES[0][1]
        assert "*.xlsx" in patterns and "*.xls" in patterns and "*.csv" in patterns

    def test_every_data_dialog_uses_the_shared_constant(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        # the 14 dataset pickers wired in round 201 (three of them via
        # identical "Open a dataset"/"Choose the dataset to audit" blocks)
        assert source.count("filetypes=DATA_FILE_TYPES") >= 14

    def test_save_dialog_offers_excel_workbook(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert '("Excel workbook", "*.xlsx")' in source

    def test_jump_chips_are_title_based_not_index_based(self) -> None:
        source = PLAYGROUND_PATH.read_text(encoding="utf-8")
        assert "_jump_to_tab_by_name(title)" in source


# --------------------------------------------------------------------- #
# the window itself -- under Xvfb when the sandbox can provide one
# --------------------------------------------------------------------- #
def _xvfb_available() -> bool:
    return shutil.which("Xvfb") is not None


XVFB_SKIP = pytest.mark.skipif(
    importlib.util.find_spec("tkinter") is None or not _xvfb_available(),
    reason="needs tkinter + Xvfb for a headless display",
)


@XVFB_SKIP
class TestExcelAndSampleDataGui:
    @pytest.fixture(scope="class")
    def xvfb_display(self) -> Any:
        """A class-scoped virtual display (its own :97 -- never the :98
        the main playground suite owns, so parallel modules collide on
        nothing)."""
        import os
        import socket

        display = ":97"
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
        assert ready, "Xvfb did not come up on :97 within 10 s"
        os.environ["DISPLAY"] = display
        try:
            yield display
        finally:
            process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)

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
        """Pump the Tk loop until the worker queue drains (probe pattern)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.update()
            if not app._busy:
                return True
            time.sleep(0.02)
        return False

    @staticmethod
    def _selected_tab_text(app: Any) -> str:
        return str(app._book.tab(app._book.select(), "text")).strip().lower()

    # -- Excel input through the real loaders ------------------------- #

    def test_excel_input_adopts_dataset(self, app: Any, tmp_path: Path) -> None:
        frame = pd.DataFrame(
            {
                "molwt": [180.16, 206.29, 194.19],
                "logp": [1.19, 3.97, 0.07],
                "pIC50": [3.19, 4.51, 3.85],
            }
        )
        target = tmp_path / "qsar.xlsx"
        frame.to_excel(target, index=False)
        app._ml_load_path(target)
        assert app._ml_frame is not None
        assert list(app._ml_frame.columns) == ["molwt", "logp", "pIC50"]
        assert len(app._ml_frame) == 3
        assert Path(app._ml_path).suffix == ".xlsx"

    def test_expression_excel_adopts_indexed_matrix(
        self, app: Any, tmp_path: Path, quiet_dialogs: list[Any]
    ) -> None:
        frame = pd.DataFrame(
            {"cell_line": ["A549", "MCF7"], "EGFR": [1.0, 2.0], "KRAS": [3.0, 4.0]}
        )
        target = tmp_path / "expr.xlsx"
        frame.to_excel(target, index=False)
        app._bio_load_expr_path(target)
        assert app._bio_expr is not None
        assert list(app._bio_expr.index) == ["A549", "MCF7"]
        assert not quiet_dialogs, quiet_dialogs

    # -- Excel output through the real exporters ---------------------- #

    def test_save_frame_dialog_writes_xlsx(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path,
        quiet_dialogs: list[Any]
    ) -> None:
        target = tmp_path / "saved.xlsx"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        frame = pd.DataFrame({"metric": ["AUC", "Cmax"], "value": [41.2, 3.9]})
        app._save_frame_dialog(frame, "Save the table", "table.csv")
        assert target.is_file()
        from pharmaml.utils.io import read_table

        back = read_table(target)
        assert back["metric"].tolist() == ["AUC", "Cmax"]

    def test_dta_export_writes_xlsx(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class _Features:
            @staticmethod
            def training_frame() -> pd.DataFrame:
                return pd.DataFrame({"smiles": ["CCO"], "affinity": [5.1]})

        app._dl_dta_last = {"features": _Features()}
        target = tmp_path / "dta.xlsx"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        app._dl_dta_export()
        assert target.is_file()

    def test_umap_save_writes_xlsx(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class _Record:
            @staticmethod
            def as_frame() -> pd.DataFrame:
                frame = pd.DataFrame({"x": [0.1, 0.2], "y": [0.3, 0.4]})
                frame.index = ["lib-1", "lib-2"]
                return frame

        app._dl_umap_record = _Record()
        target = tmp_path / "umap.xlsx"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        app._dl_umap_save()
        assert target.is_file()
        # the index (embedding labels) rides along on the workbook arm
        back = pd.read_excel(target, index_col=0)
        assert list(back.index) == ["lib-1", "lib-2"]

    def test_csvstd_export_writes_xlsx(
        self, app: Any, playground: Any, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class _Report:
            @staticmethod
            def to_frame() -> pd.DataFrame:
                return pd.DataFrame(
                    {"raw": ["CCO", "c1ccccc1"], "canonical": ["CCO", "c1ccccc1"]}
                )

        app._csvstd_report = _Report()
        target = tmp_path / "canonical.xlsx"
        monkeypatch.setattr(
            playground.filedialog,
            "asksaveasfilename",
            lambda **_kwargs: str(target),
        )
        app._csvstd_export()
        assert target.is_file()
        back = pd.read_excel(target)
        assert list(back.columns) == ["raw", "canonical"]

    # -- every dataset picker offers Excel ---------------------------- #

    def test_data_dialogs_offer_excel(
        self, app: Any, playground: Any, monkeypatch: Any
    ) -> None:
        captured: list[Any] = []

        def _record(**kwargs: Any) -> str:
            captured.append(kwargs.get("filetypes"))
            return ""  # cancel -- the handler must no-op safely

        monkeypatch.setattr(playground.filedialog, "askopenfilename", _record)
        for handler in (
            app._ml_open_csv,
            app._feat_open_csv,
            app._scan_pick_library,
            app._bio_open_csv,
            app._bio_open_expr_csv,
            app._dl_open_csv,
            app._found_open_csv,
            app._dl_dta_browse,
            app._ml_ad_pick_train_csv,
            app._ml_ad_pick_test_csv,
            app._util_merge_pick_split_file,
        ):
            handler()
        assert len(captured) == 11
        for filetypes in captured:
            assert filetypes == playground.DATA_FILE_TYPES
            patterns = filetypes[0][1]
            assert "*.xlsx" in patterns and "*.csv" in patterns

    # -- the sample-data-ready tabs ----------------------------------- #

    def test_nca_series_prefilled_matched(self, app: Any) -> None:
        times = [float(token) for token in app._nca_times.get().split(",")]
        concs = [float(token) for token in app._nca_concs.get().split(",")]
        assert len(times) == len(concs) and len(times) >= 6
        assert all(value >= 0.0 for value in concs)

    def test_assoc_transactions_prefilled(self, app: Any) -> None:
        text = app._assoc_text.get("1.0", "end").strip()
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) >= 4
        # at least three multi-item baskets (single-item transactions
        # like the lone "metformin" line are valid apriori input too)
        assert sum("," in line for line in lines) >= 3

    def test_unsupervised_example_loader(self, app: Any, quiet_dialogs: list[Any]) -> None:
        assert app._ml_frame is None
        app._unsup_load_example()
        assert app._ml_frame is not None and len(app._ml_frame) == 260
        children = app._unsup_columns.get_children()
        assert len(children) >= 5  # the numeric descriptor columns
        assert "no dataset loaded" not in app._unsup_info.cget("text")
        assert not quiet_dialogs, quiet_dialogs

    def test_bestof_example_loader_fills_pickers(
        self, app: Any, quiet_dialogs: list[Any]
    ) -> None:
        assert app._found_frame is None
        app._bestof_load_example()
        assert app._found_frame is not None
        values = list(app._bestof_column["values"])
        assert values, "the privacy column picker stayed empty"
        assert not quiet_dialogs, quiet_dialogs

    def test_utilities_example_loader_and_merge_run(self, app: Any) -> None:
        assert app._util_merge_files == []
        app._util_merge_load_example()
        assert len(app._util_merge_files) == 2
        assert app._util_merge_split_file is not None
        assert app._util_merge_mode.get() == "merge picked files"
        assert "2 table(s)" in app._util_merge_info.cget("text")
        app._util_merge_run()
        assert self._drain(app), "the merge job did not drain"
        merged = app._util_frames.get("merge")
        assert merged is not None and len(merged) == 79 + 180
        assert "rows" in app._util_merge_summary.cget("text")

    # -- the Home quick-start + the chip fix --------------------------- #

    def test_jump_by_title_resolves_every_chip(self, app: Any) -> None:
        for title in (
            "Home", "Chemistry", "PK / PD", "Cell Lines", "Docking", "Data & ML",
            "Unsupervised", "Deep Learning", "Generative", "Foundations",
            "Best-of ML", "Response AI", "Utilities",
        ):
            app._jump_to_tab_by_name(title)
            assert self._selected_tab_text(app) == title.lower(), title

    def test_quick_start_qsar_loads_and_jumps(self, app: Any) -> None:
        assert app._ml_frame is None
        app._quick_start("qsar")
        assert app._ml_frame is not None and len(app._ml_frame) == 260
        assert self._selected_tab_text(app) == "data & ml"

    def test_quick_start_stats_loads_and_jumps(self, app: Any) -> None:
        assert app._found_frame is None
        app._quick_start("stats")
        assert app._found_frame is not None
        assert self._selected_tab_text(app) == "foundations"
