"""Tests for :mod:`pharmaml.ml.curation` -- voxel51/fiftyone absorbed
offline (round 186): the Sample/SampleSet model, the immutable
view-stage algebra (filter / match / sort_by / limit / skip / shuffle /
take / select / exclude / exists / has_tag / select_fields / concat),
the brain methods (find_duplicates, find_outliers, uniqueness,
sort_by_similarity) and the ground-truth-vs-prediction evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.curation import (
    FILTER_OPS,
    OUTLIER_METHODS,
    SampleSet,
    evaluate_classification,
    find_duplicates,
    find_outliers,
    sort_by_similarity,
    uniqueness,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CCO", "CCO", "CCN", "CCC", "CCF", "CCCl"],
            "ic50": [1.2, 1.2, 3.4, 5.6, 100.0, 4.0],
            "mw": [46.0, 46.0, 45.0, 48.0, 50.0, 64.0],
            "cls": ["a", "a", "b", "b", "a", "b"],
            "pred": ["a", "b", "b", "b", "b", "a"],
        }
    )


@pytest.fixture()
def samples(frame: pd.DataFrame) -> SampleSet:
    return SampleSet.from_frame(frame)


class TestSampleSet:
    def test_from_frame_round_trip(self, samples: SampleSet, frame: pd.DataFrame) -> None:
        assert len(samples) == 6
        assert samples.fields == ["smiles", "ic50", "mw", "cls", "pred"]
        rebuilt = samples.to_frame()
        assert list(rebuilt.columns) == list(frame.columns)
        assert rebuilt["smiles"].tolist() == frame["smiles"].tolist()

    def test_sample_access_by_id(self, samples: SampleSet) -> None:
        assert samples.sample(2).fields["smiles"] == "CCN"
        with pytest.raises(DataValidationError):
            samples.sample(99)

    def test_empty_frame_refused(self) -> None:
        with pytest.raises(DataValidationError):
            SampleSet.from_frame(pd.DataFrame())

    def test_missing_values_survive_the_round_trip(self) -> None:
        holed = pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", None]})
        set_ = SampleSet.from_frame(holed)
        frame = set_.to_frame()
        assert pd.isna(frame.loc[1, "a"])
        assert pd.isna(frame.loc[2, "b"])


class TestViewStages:
    def test_filter_comparison_ops(self, samples: SampleSet) -> None:
        view = samples.view().filter("ic50", ">", 3.0)
        assert view.sample_ids() == [2, 3, 4, 5]
        assert samples.view().filter("ic50", "<=", 1.2).sample_ids() == [0, 1]
        assert samples.view().filter("ic50", "==", 4.0).sample_ids() == [5]
        assert samples.view().filter("ic50", "!=", 1.2).sample_ids() == [2, 3, 4, 5]

    def test_filter_string_equality_and_contains(self, samples: SampleSet) -> None:
        assert samples.view().filter("smiles", "==", "CCN").sample_ids() == [2]
        assert samples.view().filter("smiles", "contains", "CC").sample_ids() == [
            0, 1, 2, 3, 4, 5,
        ]
        assert samples.view().filter("smiles", "contains", "cN").sample_ids() == [2]

    def test_filter_in_and_between(self, samples: SampleSet) -> None:
        assert samples.view().filter("smiles", "in", ["CCO", "CCC"]).sample_ids() == [0, 1, 3]
        assert samples.view().filter("ic50", "between", (3.0, 5.6)).sample_ids() == [2, 3, 5]

    def test_filter_missing_values_never_match(self) -> None:
        holed = SampleSet.from_frame(
            pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"]})
        )
        assert holed.view().filter("a", ">", 0.5).sample_ids() == [0, 2]

    def test_filter_bad_op_refused(self, samples: SampleSet) -> None:
        with pytest.raises(DataValidationError):
            samples.view().filter("ic50", "~=", 3.0)

    def test_match_predicate(self, samples: SampleSet) -> None:
        view = samples.view().match(lambda s: s.id % 2 == 0)
        assert view.sample_ids() == [0, 2, 4]
        halogen = samples.view().match(
            lambda s: str(s.fields["smiles"]).endswith(("F", "Cl"))
        )
        assert halogen.sample_ids() == [4, 5]

    def test_sort_by(self, samples: SampleSet) -> None:
        assert samples.view().sort_by("ic50").sample_ids() == [0, 1, 2, 5, 3, 4]
        assert samples.view().sort_by("ic50", reverse=True).sample_ids()[0] == 4
        # string sort: CCC < CCCl < CCF < CCN < CCO
        assert samples.view().sort_by("smiles").sample_ids()[0] == 3

    def test_limit_and_skip(self, samples: SampleSet) -> None:
        assert samples.view().limit(2).sample_ids() == [0, 1]
        assert samples.view().skip(4).sample_ids() == [4, 5]
        assert samples.view().skip(2).limit(2).sample_ids() == [2, 3]
        with pytest.raises(DataValidationError):
            samples.view().limit(-1)

    def test_shuffle_and_take_are_seeded(self, samples: SampleSet) -> None:
        first = samples.view().shuffle(seed=42).sample_ids()
        second = samples.view().shuffle(seed=42).sample_ids()
        assert first == second
        assert sorted(first) == [0, 1, 2, 3, 4, 5]
        taken = samples.view().take(3, seed=42).sample_ids()
        assert len(taken) == 3 and len(set(taken)) == 3

    def test_select_and_exclude(self, samples: SampleSet) -> None:
        assert samples.view().select([3, 1]).sample_ids() == [3, 1]
        assert samples.view().exclude([0, 1, 4]).sample_ids() == [2, 3, 5]
        with pytest.raises(DataValidationError):
            samples.view().select([99])

    def test_exists_skips_missing(self) -> None:
        holed = SampleSet.from_frame(
            pd.DataFrame({"a": [1.0, None, 3.0], "b": ["x", "y", "z"]})
        )
        assert holed.view().exists("a").sample_ids() == [0, 2]
        assert holed.view().exists("b").sample_ids() == [0, 1, 2]
        assert holed.view().exists("nope").sample_ids() == []

    def test_select_fields_projection(self, samples: SampleSet) -> None:
        view = samples.view().select_fields("smiles", "ic50")
        assert list(view.to_frame().columns) == ["smiles", "ic50"]
        assert len(view) == 6  # projection does not drop samples

    def test_concat_dedupes_and_keeps_order(self, samples: SampleSet) -> None:
        left = samples.view().limit(3)
        right = samples.view().skip(1).limit(3)
        assert left.concat(right).sample_ids() == [0, 1, 2, 3]
        assert left.concat(samples.view().filter("ic50", ">", 99.0)).sample_ids() == [0, 1, 2, 4]
        with pytest.raises(DataValidationError):
            other = SampleSet.from_frame(
                pd.DataFrame({"a": [1.0], "b": ["x"]})
            )
            left.concat(other.view())

    def test_views_are_immutable_chains(self, samples: SampleSet) -> None:
        base = samples.view()
        filtered = base.filter("ic50", ">", 3.0)
        assert base.sample_ids() == [0, 1, 2, 3, 4, 5]  # untouched
        assert filtered.stages == ["filter(ic50 > 3.0)"]
        chained = filtered.sort_by("ic50").limit(2)
        assert chained.stages == ["filter(ic50 > 3.0)", "sort_by(ic50)", "limit(2)"]
        assert chained.summary() == "6 samples -> filter(ic50 > 3.0) -> sort_by(ic50) -> limit(2) -> 2 samples"

    def test_tag_and_untag_mutate_samples(self, samples: SampleSet) -> None:
        samples.view().filter("cls", "==", "a").tag("keep", "round-186")
        assert samples.sample(0).tags == {"keep", "round-186"}
        assert samples.sample(2).tags == set()
        assert samples.view().has_tag("keep").sample_ids() == [0, 1, 4]
        samples.view().has_tag("keep").untag("round-186")
        assert samples.sample(0).tags == {"keep"}

    def test_group_counts(self, samples: SampleSet) -> None:
        assert samples.view().group_counts("cls") == {"a": 3, "b": 3}
        assert samples.view().filter("ic50", ">", 3.5).group_counts("cls") == {
            "a": 1, "b": 2,
        }


class TestBrainMethods:
    def test_find_duplicates(self, samples: SampleSet) -> None:
        report = find_duplicates(samples, fields=["smiles"])
        assert report.n_input == 6
        assert report.n_duplicate_rows == 1
        assert report.groups == [[0, 1]]
        assert "1 duplicate" in report.summary()

    def test_find_duplicates_all_fields(self, samples: SampleSet) -> None:
        # rows 0/1 repeat smiles/ic50/mw but differ in pred -- a
        # full-row duplicate needs EVERY field equal, so none exist here
        report = find_duplicates(samples)
        assert report.groups == []
        cloned = SampleSet.from_frame(
            pd.concat([samples.to_frame(), samples.to_frame().head(2)])
        )
        assert find_duplicates(cloned).n_duplicate_rows == 2

    def test_find_duplicates_on_a_view(self, samples: SampleSet) -> None:
        view = samples.view().filter("ic50", ">", 2.0)
        assert find_duplicates(view, fields=["smiles"]).groups == []

    def test_find_outliers_zscore_and_iqr(self, samples: SampleSet) -> None:
        zscore = find_outliers(samples, fields=["ic50"], method="zscore", threshold=2.0)
        iqr = find_outliers(samples, fields=["ic50"], method="iqr", threshold=1.5)
        assert iqr.flagged == {"ic50": [4]}
        assert iqr.fences["ic50"][1] < 100.0
        assert "1 sample(s) flagged" in iqr.summary()
        # the lone 100 among 1.2..5.6 sits at z ~= 2.08: flagged at 2.0,
        # the outlier inflating its own std keeps it under 2.5
        assert zscore.flagged == {"ic50": [4]}
        relaxed = find_outliers(samples, fields=["ic50"], method="zscore", threshold=2.5)
        assert relaxed.flagged == {}
        assert find_outliers(samples, fields=["smiles"]).flagged == {}

    def test_find_outliers_bad_method(self, samples: SampleSet) -> None:
        with pytest.raises(DataValidationError):
            find_outliers(samples, method="mahalanobis")

    def test_uniqueness_scores(self) -> None:
        np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "x": [0.0, 0.1, 0.9, 1.0, 5.0],
                "y": [0.0, 0.1, 0.9, 1.0, 5.0],
            }
        )
        scores = uniqueness(SampleSet.from_frame(frame))
        assert scores[4] > scores[2]  # the far point is the most unique
        assert 0.0 <= min(scores.values())

    def test_uniqueness_skips_non_numeric(self, samples: SampleSet) -> None:
        scores = uniqueness(samples, fields=["ic50", "mw"])
        assert len(scores) == 6

    def test_sort_by_similarity(self) -> None:
        frame = pd.DataFrame(
            {"x": [1.0, 1.1, 0.9, -1.0], "y": [0.0, 0.1, -0.1, 0.0]}
        )
        hits = sort_by_similarity(SampleSet.from_frame(frame), 0, k=2)
        assert hits[0] == (0, pytest.approx(1.0))
        assert {sid for sid, _sim in hits[1:]} <= {1, 2}
        assert hits[1][1] > hits[2][1] if len(hits) > 2 else True

    def test_sort_by_similarity_guards(self, samples: SampleSet) -> None:
        with pytest.raises(DataValidationError):
            sort_by_similarity(samples, 99, k=2)


class TestEvaluation:
    def test_confusion_and_per_class(self, samples: SampleSet) -> None:
        report = evaluate_classification(samples, "cls", "pred")
        assert report.accuracy == pytest.approx(3.0 / 6.0)
        assert set(report.confusion.index) == {"a", "b"}
        assert report.confusion.loc["a", "a"] == 1
        assert report.confusion.loc["a", "b"] == 2
        assert report.per_class.loc["a", "support"] == 3
        assert report.mistakes == [1, 4, 5]

    def test_perfect_predictions(self) -> None:
        frame = pd.DataFrame({"t": ["x", "y"], "p": ["x", "y"]})
        report = evaluate_classification(SampleSet.from_frame(frame), "t", "p")
        assert report.accuracy == 1.0
        assert report.mistakes == []
        assert report.per_class.loc["x", "f1"] == pytest.approx(1.0)

    def test_empty_view_refused(self, samples: SampleSet) -> None:
        empty = samples.view().filter("ic50", ">", 10_000.0)
        with pytest.raises(DataValidationError):
            evaluate_classification(empty, "cls", "pred")


class TestConstants:
    def test_vocabulary_exports(self) -> None:
        assert FILTER_OPS == (
            "==", "!=", ">", ">=", "<", "<=", "in", "between", "contains",
        )
        assert OUTLIER_METHODS == ("zscore", "iqr")

    def test_view_is_iterable(self, samples: SampleSet) -> None:
        view = samples.view().limit(2)
        assert [sample.id for sample in view] == [0, 1]
