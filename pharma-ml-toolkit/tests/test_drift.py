"""Distribution-shift detection: PSI + KS, and the ``ml drift`` command.

The unit half pins the statistics (PSI is 0 on identical data, large on a
mean shift; KS flags a shape shift PSI's fixed bins can miss).  The CLI
half drives ``ml train`` (which must record the sketch) and ``ml drift``
through both reference sources — the bundle's sketch and the raw training
table — the way a shell would.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.drift import (
    detect_drift,
    drift_from_metadata,
    feature_distributions_from_frame,
    ks_statistic,
    psi,
)
from pharmaml.ml.persistence import bundle_metadata, load_model, save_model

logging.disable(logging.CRITICAL)


@pytest.fixture()
def train_csv(tmp_path) -> str:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=["d0", "d1", "d2"])
    frame["pIC50"] = 6 + frame["d0"] - 0.5 * frame["d1"] + rng.normal(scale=0.2, size=200)
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return str(path)


class TestPsi:
    def test_identical_is_zero(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        assert psi(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_mean_shift_is_large(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(3, 1, 500)
        assert psi(ref, cur) > 0.2

    def test_constant_reference_is_finite(self):
        ref = np.ones(50)
        assert np.isfinite(psi(ref, np.ones(50)))
        assert np.isfinite(psi(ref, np.full(50, 9.0)))

    def test_rejects_empty(self):
        with pytest.raises(DataValidationError):
            psi([], [1.0])

    def test_rejects_non_finite(self):
        with pytest.raises(DataValidationError):
            psi([1.0, np.nan], [1.0, 2.0])

    def test_rejects_bad_bins(self):
        with pytest.raises(DataValidationError):
            psi([1.0, 2.0, 3.0], [1.0, 2.0], bins=1)


class TestKs:
    def test_flags_a_shift(self):
        rng = np.random.default_rng(0)
        stat, p = ks_statistic(rng.normal(0, 1, 400), rng.normal(2, 1, 400))
        assert stat > 0.5 and p < 1e-6

    def test_same_distribution_is_not_significant(self):
        rng = np.random.default_rng(0)
        _, p = ks_statistic(rng.normal(0, 1, 400), rng.normal(0, 1, 400))
        assert p > 0.01


class TestDetectDrift:
    def _frames(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(5, 2, 400)})
        same = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(5, 2, 400)})
        moved = pd.DataFrame({"a": rng.normal(3, 1, 400), "b": rng.normal(5, 2, 400)})
        return ref, same, moved

    def test_stable_batch_has_no_drift(self):
        ref, same, _ = self._frames()
        report = detect_drift(ref, same)
        assert report.n_drifted == 0
        assert report.source == "frame"
        assert report.drift_share < 0.5

    def test_shifted_feature_is_named_with_direction(self):
        ref, _, moved = self._frames()
        report = detect_drift(ref, moved)
        assert [f.name for f in report.drifted] == ["a"]
        assert report.worst.name == "a"
        assert report.worst.mean_direction == "up"
        assert report.worst.psi > 0.2
        assert report.worst.ks_pvalue < 0.01

    def test_to_frame_has_one_row_per_feature(self):
        ref, _, moved = self._frames()
        table = detect_drift(ref, moved).to_frame()
        assert list(table["feature"]) == ["a", "b"]
        assert {"psi", "ks_statistic", "ks_pvalue", "verdict"}.issubset(table.columns)

    def test_str_names_the_drifted_feature(self):
        ref, _, moved = self._frames()
        assert "a (PSI" in str(detect_drift(ref, moved))

    def test_str_reports_no_drift(self):
        ref, same, _ = self._frames()
        assert "no feature shifted" in str(detect_drift(ref, same))

    def test_skips_features_absent_from_current(self):
        ref, _, moved = self._frames()
        report = detect_drift(ref, moved[["a"]])
        assert [f.name for f in report.features] == ["a"]

    def test_restricts_to_requested_features(self):
        ref, _, moved = self._frames()
        report = detect_drift(ref, moved, features=["b"])
        assert [f.name for f in report.features] == ["b"]

    def test_with_ks_false_leaves_ks_empty(self):
        ref, _, moved = self._frames()
        report = detect_drift(ref, moved, with_ks=False)
        assert all(f.ks_statistic is None for f in report.features)

    def test_rejects_non_frame_reference(self):
        with pytest.raises(DataValidationError):
            detect_drift(np.zeros((3, 2)), pd.DataFrame({"a": [1.0]}))


class TestSketch:
    def test_records_numeric_features_only(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
        sketch = feature_distributions_from_frame(frame)
        assert set(sketch) == {"a"}
        assert len(sketch["a"]["edges"]) == 11  # 10 bins -> 11 edges

    def test_round_trips_through_metadata(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"a": rng.normal(0, 1, 300)})
        meta = bundle_metadata(
            "m", "regression",
            feature_distributions=feature_distributions_from_frame(ref),
        )
        moved = pd.DataFrame({"a": rng.normal(3, 1, 300)})
        report = drift_from_metadata(meta, moved)
        assert report is not None
        assert report.source == "bundle"
        assert [f.name for f in report.drifted] == ["a"]
        assert report.worst.ks_statistic is None  # sketch holds no raw values

    def test_metadata_none_without_sketch(self):
        assert drift_from_metadata(bundle_metadata("m", "regression"),
                                   pd.DataFrame({"a": [1.0]})) is None


def _train(csv, tmp_path, target=None):
    argv = ["ml", "train", "--data", csv, "--model", "ridge", "--no-tune",
            "--cv", "3", "--out", str(tmp_path / "art")]
    if target:
        argv += ["--target", target]
    assert main(argv) == 0
    return str(next((tmp_path / "artifacts").glob("*.joblib")))


class TestTrainRecordsSketch:
    def test_bundle_carries_the_distribution_sketch(self, train_csv, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        _, meta = load_model(bundle)
        assert meta["feature_distributions"]
        assert set(meta["feature_distributions"]) == {"d0", "d1", "d2"}


class TestDriftCli:
    def _batch(self, csv, tmp_path, shift=None):
        frame = pd.read_csv(csv).drop(columns=["pIC50"]).head(60)
        if shift is not None:
            frame = frame.copy()
            frame["d0"] = frame["d0"] + shift
        path = tmp_path / "batch.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_bundle_source_reports_shift(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        batch = self._batch(train_csv, tmp_path, shift=5.0)
        code = main(["ml", "drift", "--model", bundle, "--data", batch])
        assert code == 0
        out = capsys.readouterr().out
        assert "bundle's sketch" in out
        assert "d0" in out and "shifted" in out

    def test_train_data_source_runs_ks(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        batch = self._batch(train_csv, tmp_path)
        code = main(["ml", "drift", "--train-data", train_csv, "--data", batch,
                     "--out", str(tmp_path / "d.csv")])
        assert code == 0
        assert "training frame" in capsys.readouterr().out
        table = pd.read_csv(tmp_path / "d.csv")
        assert {"feature", "psi", "ks_statistic", "verdict"}.issubset(table.columns)

    def test_recorded_target_is_auto_dropped(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        code = main(["ml", "drift", "--model", bundle, "--data", train_csv])
        assert code == 0
        assert "dropped:  'pIC50'" in capsys.readouterr().out

    def test_requires_exactly_one_reference(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = main(["ml", "drift", "--data", train_csv])
        assert code == 1
        assert "exactly one of" in capsys.readouterr().err

    def test_legacy_bundle_without_sketch_errors(self, tmp_path, capsys):
        from sklearn.linear_model import Ridge

        model = Ridge().fit(np.zeros((5, 2)), np.zeros(5))
        bundle = save_model(model, tmp_path / "legacy.joblib", name="legacy",
                            task="regression", features=["a", "b"])
        batch = tmp_path / "b.csv"
        pd.DataFrame({"a": [1.0], "b": [2.0]}).to_csv(batch, index=False)
        code = main(["ml", "drift", "--model", str(bundle), "--data", str(batch)])
        assert code == 1
        assert "no distribution sketch" in capsys.readouterr().err


class TestPredictReportsDrift:
    def _batch(self, csv, tmp_path, shift):
        frame = pd.read_csv(csv).drop(columns=["pIC50"]).head(60).copy()
        frame["d0"] = frame["d0"] + shift
        path = tmp_path / "batch.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def test_predict_prints_the_drift_verdict(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        batch = self._batch(train_csv, tmp_path, shift=5.0)
        code = main(["ml", "predict", "--model", bundle, "--data", batch,
                     "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "drift:" in capsys.readouterr().out

    def test_no_drift_check_suppresses_it(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        batch = self._batch(train_csv, tmp_path, shift=5.0)
        code = main(["ml", "predict", "--model", bundle, "--data", batch,
                     "--no-drift-check", "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "drift:" not in capsys.readouterr().out

    def test_stable_batch_reports_no_shift(self, train_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(train_csv, tmp_path)
        frame = pd.read_csv(train_csv).drop(columns=["pIC50"])
        batch = tmp_path / "batch.csv"
        frame.to_csv(batch, index=False)
        code = main(["ml", "predict", "--model", bundle, "--data", str(batch),
                     "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "no feature shifted" in capsys.readouterr().out
