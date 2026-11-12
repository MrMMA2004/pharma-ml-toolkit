"""CLI tests for ``ml predict`` and ``ml card``.

Both commands are exercised through :func:`pharmaml.cli.main` exactly the
way a shell would invoke them, after a real ``ml train`` produces the
bundle they consume — so the feature-contract alignment, the target
auto-drop, the conformal branch and the card output are all verified
end to end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main

logging.disable(logging.CRITICAL)


@pytest.fixture()
def reg_csv(tmp_path) -> str:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"d{i}" for i in range(4)])
    frame["pIC50"] = 5 + frame["d0"] * 2 - frame["d1"] + rng.normal(scale=0.3, size=120)
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] + frame["d1"] > 0).astype(int)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _train(csv: str, model: str, tmp_path: Path, target: str | None = None) -> str:
    argv = ["ml", "train", "--data", csv, "--model", model, "--no-tune",
            "--cv", "3", "--out", str(tmp_path / "art")]
    if target is not None:
        argv += ["--target", target]
    assert main(argv) == 0
    bundle = next((tmp_path / "artifacts").glob("*.joblib"))
    return str(bundle)


class TestMLPredict:
    def test_predicts_and_writes_default_output(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(30)
        query_path = tmp_path / "query.csv"
        query.to_csv(query_path, index=False)

        code = main(["ml", "predict", "--model", bundle, "--data", str(query_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "ridge (regression)" in out
        assert "rows:     30 x 4 feature(s)" in out
        written = tmp_path / "query_predictions.csv"
        assert written.is_file()
        scored = pd.read_csv(written)
        assert "prediction" in scored.columns and len(scored) == 30

    def test_target_scores_against_truth(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main([
            "ml", "predict", "--model", bundle, "--data", reg_csv,
            "--target", "pIC50", "--out", str(tmp_path / "scored.csv"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "metrics:" in out and "r2=" in out
        assert (tmp_path / "scored.csv").is_file()

    def test_recorded_target_is_auto_dropped(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic", tmp_path, target="Target")
        # the query still carries the label column, but no --target is passed
        code = main(["ml", "predict", "--model", bundle, "--data", clf_csv,
                     "--out", str(tmp_path / "p.csv")])
        assert code == 0
        assert "dropped:  'Target' (the recorded training target)" in capsys.readouterr().out

    def test_missing_feature_errors(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"d0": [0.1], "d1": [0.2]}).to_csv(bad, index=False)
        code = main(["ml", "predict", "--model", bundle, "--data", str(bad)])
        assert code == 1
        assert "missing from" in capsys.readouterr().err

    def test_unknown_target_errors(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "predict", "--model", bundle, "--data", reg_csv,
                     "--target", "nope"])
        assert code == 1
        assert "is not a column" in capsys.readouterr().err

    def test_conformal_regression_emits_intervals(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main([
            "ml", "predict", "--model", bundle, "--data", reg_csv,
            "--target", "pIC50", "--conformal", "0.1",
            "--out", str(tmp_path / "conf.csv"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "target coverage 90%" in out
        assert "coverage:" in out
        scored = pd.read_csv(tmp_path / "conf.csv")
        assert {"prediction", "lower", "upper"}.issubset(scored.columns)
        assert (scored["lower"] <= scored["upper"]).all()

    def test_conformal_classification_emits_sets(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(clf_csv, "logistic", tmp_path, target="Target")
        code = main([
            "ml", "predict", "--model", bundle, "--data", clf_csv,
            "--target", "Target", "--conformal", "0.2",
            "--out", str(tmp_path / "conf.csv"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "mean set size" in out
        scored = pd.read_csv(tmp_path / "conf.csv")
        assert {"prediction_set", "set_size"}.issubset(scored.columns)

    def test_conformal_without_labels_errors(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(10)
        query_path = tmp_path / "q.csv"
        query.to_csv(query_path, index=False)
        code = main(["ml", "predict", "--model", bundle, "--data", str(query_path),
                     "--conformal", "0.1"])
        assert code == 1
        assert "needs labelled rows" in capsys.readouterr().err

    def test_conformal_with_separate_calibration_table(
        self, reg_csv, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        query = pd.read_csv(reg_csv).drop(columns=["pIC50"]).head(25)
        query_path = tmp_path / "q.csv"
        query.to_csv(query_path, index=False)
        # --target names the calibration table's label, not the query's
        code = main([
            "ml", "predict", "--model", bundle, "--data", str(query_path),
            "--calibration-data", reg_csv, "--target", "pIC50",
            "--conformal", "0.1", "--out", str(tmp_path / "c.csv"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "scored:   25 row(s)" in out
        assert "separate calibration table" in out
        scored = pd.read_csv(tmp_path / "c.csv")
        assert {"prediction", "lower", "upper"}.issubset(scored.columns)
        assert len(scored) == 25


class TestMLCard:
    def test_card_defaults_beside_the_bundle(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "card", "--model", bundle])
        assert code == 0
        out = capsys.readouterr().out
        assert "card:     " in out and "pass --train-data" in out
        card_path = Path(bundle).with_name(Path(bundle).stem + "_card.md")
        assert card_path.is_file()
        assert "# Model card:" in card_path.read_text(encoding="utf-8")

    def test_card_with_training_data_records_the_domain(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main([
            "ml", "card", "--model", bundle, "--train-data", reg_csv,
            "--target", "pIC50", "--out", str(tmp_path / "card.md"),
        ])
        assert code == 0
        text = (tmp_path / "card.md").read_text(encoding="utf-8")
        assert "## Applicability domain" in text
        assert "Not computed" not in text

    def test_card_auto_drops_the_recorded_target(self, reg_csv, tmp_path, capsys, monkeypatch):
        # no --target: the bundle recorded pIC50, so it must not leak into
        # the applicability-domain box.
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "card", "--model", bundle, "--train-data", reg_csv,
                     "--out", str(tmp_path / "card.md")])
        assert code == 0
        text = (tmp_path / "card.md").read_text(encoding="utf-8")
        box = text.split("## Applicability domain", 1)[1].split("## ", 1)[0]
        assert "pIC50" not in box

    def test_card_json_output(self, reg_csv, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bundle = _train(reg_csv, "ridge", tmp_path)
        code = main(["ml", "card", "--model", bundle, "--out", str(tmp_path / "c.json")])
        assert code == 0
        payload = json.loads((tmp_path / "c.json").read_text(encoding="utf-8"))
        assert payload["task"] == "regression"

    def test_card_missing_bundle_errors(self, tmp_path, capsys):
        code = main(["ml", "card", "--model", "ghost.joblib"])
        assert code == 1
        assert "error:" in capsys.readouterr().err
