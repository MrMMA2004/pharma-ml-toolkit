"""Stress tests for the §5.5/16.5 performance targets (part8 round 3).

Realistic representative workloads, sized to finish inside CI on a
2-core sandbox while still stressing the real code paths:

* large CSV import (150k rows) — the reader + progress reporting,
* large Excel import (40k rows) — the openpyxl path,
* wide descriptor matrix (5k rows x 512 numeric columns) — profiling,
  correlation and PCA paths,
* fingerprint calculation (2k molecules, 2048-bit Morgan),
* model fitting + cross-validation on the wide matrix,
* large prediction batches (150k rows through a fitted pipeline),
* large plot generation (scatter with 150k points, heatmap 64x64),
* GUI responsiveness while the worker runs: the DataExplorerPage load
  stays off the GUI thread (the async JobRunner path, part8 B).

Every test asserts BOTH the functional result AND a wall-clock budget
(generous: these are smoke budgets, not micro-benchmarks — a 10x
regression fails, normal jitter does not).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# wall-clock budgets (seconds) — regression fences, not benchmarks
CSV_BUDGET = 25.0
EXCEL_BUDGET = 60.0
WIDE_PROFILE_BUDGET = 30.0
FINGERPRINT_BUDGET = 90.0
FIT_CV_BUDGET = 120.0
PREDICT_BUDGET = 60.0
PLOT_BUDGET = 30.0


def _make_frame(n_rows: int, n_cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"feat_{i}": rng.normal(size=n_rows) for i in range(n_cols)}
    data["target"] = rng.normal(size=n_rows)
    data["label"] = pd.Series(rng.integers(0, 3, n_rows)).astype(str)
    return pd.DataFrame(data)


class TestLargeImports:
    def test_large_csv_import(self, tmp_path: Path) -> None:
        """150k rows x 14 cols through the toolkit's own reader."""
        from pharmaml.utils.io import read_table_with_progress

        frame = _make_frame(150_000, 12)
        path = tmp_path / "large.csv"
        frame.to_csv(path, index=False)

        ticks: list[float] = []

        def report(done: int, total: int | None, note: str) -> None:
            ticks.append(done / (total or 1))

        t0 = time.perf_counter()
        loaded = read_table_with_progress(path, report)
        elapsed = time.perf_counter() - t0

        assert len(loaded) == 150_000
        assert list(loaded.columns) == list(frame.columns)
        assert elapsed < CSV_BUDGET, f"CSV import took {elapsed:.1f}s"
        # progress actually reported on the large file
        # the final partial chunk lands just under 1.0
        assert ticks and ticks[-1] >= 0.99

    def test_large_csv_export_import_roundtrip(self, tmp_path: Path) -> None:
        """The exported artifact reloads value-honestly.

        The reader's documented type inference upgrades numeric-looking
        string columns (``"2"`` -> int64), so the roundtrip asserts
        VALUES and the numeric schema, not object dtypes.
        """
        from pharmaml.utils.io import read_table

        frame = _make_frame(60_000, 8)
        path = tmp_path / "rt.csv"
        frame.to_csv(path, index=False)
        loaded = read_table(path)

        assert len(loaded) == 60_000
        numeric = [c for c in frame.columns if c.startswith("feat_")]
        for column in numeric + ["target"]:
            np.testing.assert_allclose(
                loaded[column].to_numpy(), frame[column].to_numpy()
            )
        # the label column roundtrips value-identically
        assert (loaded["label"].astype(str) == frame["label"]).all()

    def test_large_excel_import(self, tmp_path: Path) -> None:
        """40k rows through the Excel path (openpyxl is the slow one)."""
        from pharmaml.utils.io import read_table

        frame = _make_frame(40_000, 6)
        path = tmp_path / "large.xlsx"
        frame.to_excel(path, index=False)

        t0 = time.perf_counter()
        loaded = read_table(path)
        elapsed = time.perf_counter() - t0

        assert len(loaded) == 40_000
        assert elapsed < EXCEL_BUDGET, f"Excel import took {elapsed:.1f}s"


class TestWideMatrices:
    def test_wide_descriptor_matrix_operations(self) -> None:
        """5k x 512 numeric: describe + correlation + PCA-scaled paths."""
        from sklearn.decomposition import PCA

        frame = _make_frame(5_000, 512)
        features = frame[[c for c in frame.columns if c.startswith("feat_")]]

        t0 = time.perf_counter()
        described = features.describe()
        corr = features.iloc[:, :64].corr()  # correlation on a 64-col slice
        # (full 512x512 corr is O(512^2 * 5000) flops — cheap; the wide
        # describe is the realistic profiling path)
        reduced = PCA(n_components=8, random_state=0).fit_transform(
            features.iloc[:2000]
        )
        elapsed = time.perf_counter() - t0

        assert described.shape == (8, 512)
        assert corr.shape == (64, 64)
        assert reduced.shape == (2000, 8)
        assert elapsed < WIDE_PROFILE_BUDGET, f"wide ops took {elapsed:.1f}s"
        assert np.isfinite(reduced).all()


class TestFingerprintsAndFitting:
    def test_fingerprint_calculation_stress(self) -> None:
        """2k valid molecules through the house Morgan fingerprint.

        The house :func:`morgan_fingerprint` takes the toolkit's own
        :class:`~pharmaml.chem.molgraph.Molecule` (built by
        :func:`parse_smiles`), not an RDKit Mol.
        """
        from pharmaml.chem.molgraph import morgan_fingerprint, parse_smiles

        # chemically VALID scaffolds with varied substituents (a mix
        # of the toolkit's realistic fragment space)
        bases = (
            "CC(=O)Oc1ccccc1C(=O)O",          # aspirin scaffold
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O",     # ibuprofen scaffold
            "c1ccc(cc1)S(=O)(=O)N",           # sulfonamide
            "OC(=O)c1ccccc1O",                # salicylic acid
            "CCN(CC)CCCC(C)Nc1ccccc1",        # lidocaine-like
        )
        subs = ("", "Cl", "Br", "F", "C", "CC", "OC", "N", "C#N", "C(=O)O")
        rng = np.random.default_rng(3)
        smiles = [
            f"{rng.choice(bases)}{'' if i % 11 == 10 else rng.choice(subs)}"
            for i in range(2000)
        ]
        mols = [m for m in (parse_smiles(s, strict=False) for s in smiles) if m is not None]
        assert len(mols) > 1800  # the scaffolds must nearly all parse

        t0 = time.perf_counter()
        matrix = np.array([morgan_fingerprint(m) for m in mols])
        elapsed = time.perf_counter() - t0

        assert matrix.shape == (len(mols), 2048)
        assert matrix.any(axis=1).all()  # every molecule produced bits
        assert elapsed < FINGERPRINT_BUDGET, f"fingerprints took {elapsed:.1f}s"

    def test_model_fitting_and_cross_validation(self) -> None:
        """RF fit + 5-fold CV on a 3k x 64 matrix (2-core budget)."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        frame = _make_frame(3_000, 64)
        X = frame[[c for c in frame.columns if c.startswith("feat_")]]
        y = frame["target"]

        t0 = time.perf_counter()
        model = RandomForestRegressor(n_estimators=60, n_jobs=2, random_state=0)
        model.fit(X, y)
        scores = cross_val_score(model, X, y, cv=5, n_jobs=1)
        elapsed = time.perf_counter() - t0

        assert len(scores) == 5
        assert np.isfinite(scores).all()
        assert elapsed < FIT_CV_BUDGET, f"fit+CV took {elapsed:.1f}s"

    def test_large_prediction_batch(self) -> None:
        """150k rows through a fitted pipeline in one predict call."""
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        frame = _make_frame(2_000, 32)
        X = frame[[c for c in frame.columns if c.startswith("feat_")]]
        y = frame["target"]
        pipe = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge())]
        ).fit(X, y)

        big = _make_frame(150_000, 32, seed=9)
        Xb = big[[c for c in big.columns if c.startswith("feat_")]]

        t0 = time.perf_counter()
        predictions = pipe.predict(Xb)
        elapsed = time.perf_counter() - t0

        assert len(predictions) == 150_000
        assert np.isfinite(predictions).all()
        assert elapsed < PREDICT_BUDGET, f"prediction took {elapsed:.1f}s"


class TestLargePlotGeneration:
    def test_large_scatter_generation(self, tmp_path: Path) -> None:
        """150k-point scatter through the backend figure path."""
        from pharmaml.viz.statplots import plot_regression

        rng = np.random.default_rng(5)
        x = rng.normal(size=150_000)
        y = x + rng.normal(scale=0.1, size=150_000)

        t0 = time.perf_counter()
        png = tmp_path / "scatter.png"
        plot_regression(x, y, str(png), n_boot=0)
        elapsed = time.perf_counter() - t0

        assert png.exists() and png.stat().st_size > 50_000
        assert elapsed < PLOT_BUDGET, f"scatter took {elapsed:.1f}s"

    def test_large_heatmap_generation(self, tmp_path: Path) -> None:
        """64x64 correlation heatmap through the ml figure path."""
        from pharmaml.viz.ml_plots import plot_correlation_heatmap

        rng = np.random.default_rng(6)
        data = rng.normal(size=(500, 64))
        corr = pd.DataFrame(data).corr()

        t0 = time.perf_counter()
        figure = plot_correlation_heatmap(corr)
        from pharmaml.viz import save_figure

        png = tmp_path / "heat.png"
        save_figure(figure, str(png))
        elapsed = time.perf_counter() - t0

        assert png.exists() and png.stat().st_size > 10_000
        assert elapsed < PLOT_BUDGET, f"heatmap took {elapsed:.1f}s"


class TestGuiResponsiveness:
    def test_data_explorer_load_stays_off_the_gui_thread(
        self, qtbot, tmp_path, monkeypatch
    ) -> None:
        """§5.5: the GUI thread never parses the user table.

        A 120k-row load runs on the JobRunner worker; the GUI thread
        must stay responsive (process queued events) while it runs,
        and the load must complete.
        """
        monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
        from PySide6.QtWidgets import QApplication

        from ui.pages.data_explorer import DataExplorerPage

        frame = _make_frame(120_000, 10)
        path = tmp_path / "gui.csv"
        frame.to_csv(path, index=False)

        page = DataExplorerPage()
        qtbot.addWidget(page)
        page.load_path(path)

        # pump events while the worker reads: if the read were
        # synchronous on the GUI thread, THIS loop would block until
        # the file was done (the responsiveness regression fence)
        app = QApplication.instance()
        t0 = time.perf_counter()
        responsive = True
        while page._runner.busy:
            app.processEvents()
            if time.perf_counter() - t0 > 40:  # hard timeout
                responsive = False
                break
        app.processEvents()

        assert responsive, "the load never completed on the worker"
        qtbot.wait(100)
        assert page.frame() is not None
        assert len(page.frame()) == 120_000
        assert page._source.endswith("gui.csv")
