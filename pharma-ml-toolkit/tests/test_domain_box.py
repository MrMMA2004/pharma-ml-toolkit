"""Applicability-domain box: recording it in a bundle, checking a query.

``feature_ranges_from_frame`` records the training box at save time;
``ml predict`` flags rows outside it in an ``in_domain`` column; ``ml
card`` reports the box the bundle carries even without the training
frame.  These tests cover the helper, the box check, and both CLI paths.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.ml.applicability import DomainBox, check_domain_box
from pharmaml.ml.persistence import (
    bundle_metadata,
    feature_ranges_from_frame,
    load_model,
)

logging.disable(logging.CRITICAL)


@pytest.fixture()
def reg_csv(tmp_path) -> str:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(150, 3)), columns=["d0", "d1", "d2"])
    frame["pIC50"] = 6 + frame["d0"] - 0.5 * frame["d1"] + rng.normal(scale=0.2, size=150)
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _train(csv: str, tmp_path, model: str = "ridge", target: str | None = None) -> str:
    argv = ["ml", "train", "--data", csv, "--model", model, "--no-tune",
            "--cv", "3", "--out", str(tmp_path / "art")]
    if target is not None:
        argv += ["--target", target]
    assert main(argv) == 0
    return str(next((tmp_path / "artifacts").glob("*.joblib")))


class TestFeatureRanges:
    def test_records_only_numeric_columns(self):
        frame = pd.DataFrame({"a": [1.0, 4.0], "b": [10, 20], "c": ["x", "y"]})
        box = feature_ranges_from_frame(frame)
        assert set(box) == {"a", "b"}
        assert box["a"] == {"min": 1.0, "max": 4.0}
        assert box["b"] == {"min": 10.0, "max": 20.0}

    def test_respects_requested_feature_order(self):
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        box = feature_ranges_from_frame(frame, ["b", "a"])
        assert list(box) == ["b", "a"]

    def test_skips_requested_columns_absent_from_frame(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        box = feature_ranges_from_frame(frame, ["a", "ghost"])
        assert set(box) == {"a"}


class TestDomainBox:
    def test_from_metadata_round_trip(self):
        meta = bundle_metadata("m", "regression", feature_ranges={"a": {"min": 0.0, "max": 1.0}})
        box = DomainBox.from_metadata(meta)
        assert box is not None and box.n_features == 1 and box.ranges["a"]["max"] == 1.0

    def test_from_metadata_none_without_box(self):
        assert DomainBox.from_metadata(bundle_metadata("m", "regression")) is None
        assert DomainBox.from_metadata({"feature_ranges": {}}) is None

    def test_check_flags_below_and_above(self):
        box = DomainBox.from_metadata(
            bundle_metadata("m", "regression", feature_ranges={"a": {"min": 0.0, "max": 10.0}})
        )
        report = box.check(pd.DataFrame({"a": [-1.0, 5.0, 11.0]}))
        assert report.in_domain.tolist() == [False, True, False]
        assert report.below == {"a": 1} and report.above == {"a": 1}
        assert report.fraction_out == pytest.approx(2 / 3)

    def test_check_nan_is_out_of_domain(self):
        box = DomainBox.from_metadata(
            bundle_metadata("m", "regression", feature_ranges={"a": {"min": 0.0, "max": 1.0}})
        )
        report = box.check(pd.DataFrame({"a": [0.5, np.nan]}))
        assert report.in_domain.tolist() == [True, False]

    def test_check_ignores_features_absent_from_query(self):
        box = DomainBox.from_metadata(
            bundle_metadata(
                "m", "regression",
                feature_ranges={"a": {"min": 0.0, "max": 1.0}, "b": {"min": 0.0, "max": 1.0}},
            )
        )
        report = box.check(pd.DataFrame({"a": [0.5, 2.0]}))
        assert report.in_domain.tolist() == [True, False]

    def test_str_reports_clean_and_flagged(self):
        box = DomainBox.from_metadata(
            bundle_metadata("m", "regression", feature_ranges={"a": {"min": 0.0, "max": 1.0}})
        )
        assert "inside" in str(box.check(pd.DataFrame({"a": [0.5]})))
        flagged = str(box.check(pd.DataFrame({"a": [2.0]})))
        assert "outside the training box" in flagged and "a (" in flagged

    def test_check_domain_box_none_for_legacy_bundle(self):
        assert check_domain_box({}, pd.DataFrame({"a": [1.0]})) is None


class TestTrainRecordsBox:
    def test_trained_bundle_carries_feature_ranges(self, reg_csv, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, tmp_path)
        _, meta = load_model(bundle)
        assert meta["feature_ranges"], "train must record the box"
        assert set(meta["feature_ranges"]) == {"d0", "d1", "d2"}


class TestPredictFlagsDomain:
    def test_out_of_domain_rows_are_flagged(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(20)
        # push two rows far outside the training box
        query.iloc[0, 0] = 1e6
        query.iloc[1, 1] = -1e6
        query_path = tmp_path / "q.csv"
        query.to_csv(query_path, index=False)

        code = main(["ml", "predict", "--model", bundle, "--data", str(query_path),
                     "--out", str(tmp_path / "p.csv")])
        assert code == 0
        out = capsys.readouterr().out
        assert "outside the training box" in out
        scored = pd.read_csv(tmp_path / "p.csv")
        assert "in_domain" in scored.columns
        assert scored["in_domain"].sum() == 18

    def test_no_domain_check_suppresses_the_column(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(10)
        query.iloc[0, 0] = 1e6
        query_path = tmp_path / "q.csv"
        query.to_csv(query_path, index=False)

        code = main(["ml", "predict", "--model", bundle, "--data", str(query_path),
                     "--no-domain-check", "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "training box" not in capsys.readouterr().out
        scored = pd.read_csv(tmp_path / "p.csv")
        assert "in_domain" not in scored.columns

    def test_all_in_domain_prints_a_clean_verdict(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(15)
        query_path = tmp_path / "q.csv"
        query.to_csv(query_path, index=False)
        code = main(["ml", "predict", "--model", bundle, "--data", str(query_path),
                     "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "all 15 row(s) inside" in capsys.readouterr().out


class TestCardUsesBundleBox:
    def test_card_reports_the_box_without_training_data(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, tmp_path)
        code = main(["ml", "card", "--model", bundle, "--out", str(tmp_path / "card.md")])
        assert code == 0
        text = (tmp_path / "card.md").read_text(encoding="utf-8")
        box = text.split("## Applicability domain", 1)[1].split("## ", 1)[0]
        assert "Not computed" not in box
        assert "d0" in box
