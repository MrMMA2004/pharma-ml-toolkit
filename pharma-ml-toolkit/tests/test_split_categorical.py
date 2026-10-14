"""Tests for the stratified table splitter and categorical encoding.

Covers the round's new public surface:

* :func:`pharmaml.data.split_table` / :class:`SplitResult` — the
  ``Split CSV file.py`` upgrade with stratification, validation carving
  and file writing.
* ``build_preprocessor(encode_categorical=True)`` — the one-hot
  ColumnTransformer branch.
* ``train_model`` / ``compare_models`` auto-detecting categorical
  columns and ``load_dataset(keep_categorical=True)``.
* ``resolve_model_name`` alias resolution.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pharmaml.data import load_dataset, split_table
from pharmaml.exceptions import DataValidationError, ModelError
from pharmaml.ml.models import MODEL_ALIASES, resolve_model_name
from pharmaml.ml.preprocessing import build_preprocessor, has_categorical_columns

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def balanced_frame() -> pd.DataFrame:
    """100-row table with a 60/40 two-class target."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    frame["Target"] = ["x"] * 60 + ["y"] * 40
    return frame


@pytest.fixture()
def mixed_frame() -> pd.DataFrame:
    """Numeric + categorical features with a binary target."""
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(80, 2)), columns=["d1", "d2"])
    frame["assay"] = rng.choice(["A", "B"], size=80)
    frame["Target"] = (frame["d1"] + (frame["assay"] == "B") * 0.5 > 0).astype(int)
    return frame


@pytest.fixture()
def mixed_csv(tmp_path, mixed_frame) -> str:
    path = tmp_path / "mixed.csv"
    mixed_frame.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# split_table
# ---------------------------------------------------------------------------
class TestSplitTable:
    def test_stratified_balance(self, balanced_frame):
        result = split_table(frame=balanced_frame, test_size=0.2, random_state=0)
        assert result.stratified is True
        assert result.target == "Target"
        # class proportions preserved on both sides
        for side in (result.train, result.test):
            share = (side["Target"] == "x").mean()
            assert share == pytest.approx(0.6, abs=0.05)
        # disjoint and complete (frames are index-reset, so compare rows)
        assert len(result.train) + len(result.test) == 100
        combined = pd.concat([result.train, result.test])
        assert not combined.duplicated().any()

    def test_three_way_split(self, balanced_frame):
        result = split_table(
            frame=balanced_frame, test_size=0.2, val_size=0.1, random_state=0
        )
        assert result.val is not None
        # val fraction is relative to the ORIGINAL rows: 100 -> 20 test,
        # 10 val, 70 train (the val split happens inside the 80 remaining).
        assert {len(result.train), len(result.val), len(result.test)} == {70, 10, 20}
        sides = result.sides()
        assert set(sides) == {"train", "val", "test"}

    def test_unstratified_fallback_on_rare_class(self, balanced_frame):
        frame = balanced_frame.copy()
        frame.loc[frame.index[-1], "Target"] = "rare"  # single-member class
        result = split_table(frame=frame, test_size=0.2, random_state=0)
        assert result.stratified is False
        assert len(result.train) + len(result.test) == len(frame)

    def test_continuous_target_falls_back(self):
        frame = pd.DataFrame({"x": range(50), "pIC50": np.linspace(4, 9, 50)})
        result = split_table(frame=frame, test_size=0.2, random_state=0)
        # every value unique -> stratification impossible -> random split
        assert result.stratified is False
        assert result.target == "pIC50"
        assert result._balance(result.train) is None  # >20 unique values

    def test_explicit_target_and_no_stratify(self, balanced_frame):
        result = split_table(frame=balanced_frame, target="Target", stratify=False)
        assert result.stratified is False
        assert result.target == "Target"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # neither path nor frame
            dict(path="x.csv", frame=pd.DataFrame()),  # both
            dict(frame=pd.DataFrame({"a": [1]}), test_size=1.5),
            dict(frame=pd.DataFrame({"a": [1]}), test_size=0.5, val_size=2.0),
            dict(frame=pd.DataFrame({"a": [1]}), test_size=0.5, val_size=-0.1),
        ],
    )
    def test_argument_errors(self, kwargs):
        with pytest.raises(DataValidationError):
            split_table(**kwargs)

    def test_missing_target_error(self):
        with pytest.raises(DataValidationError, match="target column"):
            split_table(frame=pd.DataFrame({"a": [1, 2, 3, 4]}), test_size=0.5)

    def test_too_few_rows(self):
        with pytest.raises(DataValidationError, match="too few"):
            split_table(frame=pd.DataFrame({"a": [1, 2, 3]}), test_size=0.5)

    def test_val_size_leaves_no_train(self, balanced_frame):
        with pytest.raises(DataValidationError, match="leaves no training rows"):
            split_table(frame=balanced_frame, test_size=0.9, val_size=0.85)

    def test_write_produces_files(self, tmp_path, balanced_frame):
        result = split_table(frame=balanced_frame, test_size=0.2, random_state=0)
        paths = result.write(tmp_path / "splits", prefix="demo")
        assert set(paths) == {"train", "test"}
        assert paths["train"].is_file()
        loaded = pd.read_csv(paths["train"])
        assert list(loaded.columns) == list(balanced_frame.columns)
        assert result.paths["train"] == paths["train"]

    def test_str_includes_audit_lines(self, balanced_frame):
        result = split_table(frame=balanced_frame, test_size=0.2, val_size=0.1)
        text = str(result)
        assert "stratified" in text
        assert "train" in text and "val" in text and "test" in text
        assert "classes" in text

    def test_path_input(self, tmp_path, balanced_frame):
        source = tmp_path / "src.csv"
        balanced_frame.to_csv(source, index=False)
        result = split_table(path=source, test_size=0.25)
        assert len(result.test) == 25


# ---------------------------------------------------------------------------
# categorical encoding
# ---------------------------------------------------------------------------
class TestHasCategoricalColumns:
    def test_dataframe_with_strings(self, mixed_frame):
        assert has_categorical_columns(mixed_frame[["d1", "d2", "assay"]]) is True

    def test_numeric_only(self, mixed_frame):
        assert has_categorical_columns(mixed_frame[["d1", "d2"]]) is False

    def test_ndarray(self, mixed_frame):
        assert has_categorical_columns(mixed_frame[["d1", "d2"]].to_numpy()) is False


class TestBuildPreprocessorCategorical:
    def test_onehot_expansion(self, mixed_frame):
        X = mixed_frame[["d1", "d2", "assay"]]
        prep = build_preprocessor(encode_categorical=True)
        prep.fit(X)
        transformed = prep.transform(X)
        # 2 numeric + 2 one-hot columns (assay A/B)
        assert transformed.shape[1] == 4
        names = list(prep.get_feature_names_out())
        assert "d1" in names and "assay_A" in names and "assay_B" in names

    def test_unknown_category_ignored(self, mixed_frame):
        X = mixed_frame[["d1", "d2", "assay"]]
        prep = build_preprocessor(encode_categorical=True)
        prep.fit(X)
        unseen = pd.DataFrame({"d1": [0.0], "d2": [0.0], "assay": ["Z"]})
        out = prep.transform(unseen)
        assert out.shape == (1, 4)  # no crash; 'Z' encodes as all-zero

    def test_missing_categorical_imputed(self, mixed_frame):
        X = mixed_frame[["d1", "d2", "assay"]].copy()
        X.loc[X.index[0], "assay"] = None
        prep = build_preprocessor(encode_categorical=True)
        out = prep.fit_transform(X)
        assert np.isfinite(out).all()

    def test_numeric_unaffected_without_flag(self, mixed_frame):
        X = mixed_frame[["d1", "d2", "assay"]]
        with pytest.raises(ValueError):
            # default preprocessor is numeric-only -> strings crash loudly
            # (the auto-detect in train_model is what protects users)
            build_preprocessor().fit(X)


class TestTrainModelAutoDetect:
    def test_end_to_end_classification(self, mixed_frame):
        from pharmaml.ml import evaluate_model, train_model
        from pharmaml.ml.preprocessing import make_train_test_split

        X, y = mixed_frame[["d1", "d2", "assay"]], mixed_frame["Target"]
        X_tr, X_te, y_tr, y_te = make_train_test_split(X, y, test_size=0.25)
        result = train_model("logistic", "classification", X_tr, y_tr, tune=False)
        report = evaluate_model(result.model, X_te, y_te, "classification")
        assert 0.0 <= report.accuracy <= 1.0

    def test_compare_models_with_categorical(self, mixed_frame):
        from pharmaml.ml import compare_models

        board = compare_models(
            mixed_frame[["d1", "d2", "assay"]],
            mixed_frame["Target"],
            "classification",
            models=["logistic", "rf"],
            cv=3,
        )
        assert len(board) == 2
        assert board["model"].tolist() == ["logistic_regression", "random_forest"] or set(
            board["model"]
        ) == {"logistic_regression", "random_forest"}

    def test_numpy_input_stays_numeric_path(self, mixed_frame):
        from pharmaml.ml import train_model

        X = mixed_frame[["d1", "d2"]].to_numpy()
        y = mixed_frame["Target"].to_numpy()
        result = train_model("rf", "classification", X, y, tune=False, preprocess=True)
        assert isinstance(result.model, Pipeline)


class TestLoadDatasetKeepCategorical:
    def test_keep(self, mixed_csv):
        ds = load_dataset(mixed_csv, keep_categorical=True)
        assert "assay" in ds.X.columns
        assert ds.categorical_columns == ["assay"]
        assert ds.dropped_columns == []
        summary = ds.summary()
        assert summary.categorical_columns == ["assay"]
        assert "categorical_columns" in summary.as_dict()
        assert "assay" in str(summary)

    def test_default_drops(self, mixed_csv):
        ds = load_dataset(mixed_csv)
        assert "assay" not in ds.X.columns
        assert ds.dropped_columns == ["assay"]
        assert ds.categorical_columns == []

    def test_end_to_end_through_training(self, mixed_csv):
        from pharmaml.ml import train_model

        ds = load_dataset(mixed_csv, keep_categorical=True)
        result = train_model("rf", ds.task, ds.X, ds.y, tune=False)
        names = list(result.model[:-1].get_feature_names_out())
        assert any(name.startswith("assay_") for name in names)


class TestResolveModelName:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("logistic", "logistic_regression"),
            ("logreg", "logistic_regression"),
            ("rf", "random_forest"),
            ("random_forest", "random_forest"),
            ("Random-Forest", "random_forest"),
            ("RandomForest", "random_forest"),
            ("xgb", "xgboost"),
            ("mlp", "ann"),
            ("SVM", "svm"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert resolve_model_name(raw) == expected

    def test_every_alias_maps_to_registry(self):
        for alias, target in MODEL_ALIASES.items():
            assert resolve_model_name(alias) == target

    def test_unknown_raises(self):
        with pytest.raises(ModelError, match="Unknown model"):
            resolve_model_name("gradient_boosting_supreme")


class TestFeatureImportanceNames:
    def test_expanded_names_used(self, mixed_frame):
        from pharmaml.viz.ml_plots import _extract_importances
        from pharmaml.ml import train_model

        X, y = mixed_frame[["d1", "d2", "assay"]], mixed_frame["Target"]
        result = train_model("rf", "classification", X, y, tune=False)
        names, values = _extract_importances(result.model, None)
        assert len(names) == len(values) == 4
        assert "assay_A" in names and "assay_B" in names
