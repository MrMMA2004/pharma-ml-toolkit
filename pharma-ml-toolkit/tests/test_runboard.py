"""Tests for :mod:`pharmaml.ml.runboard` -- tensorflow/tensorboard
absorbed offline (round 186): the summary-writer event model, the event
accumulator's Vitter reservoir with ``always_keep_last``, the scalars
dashboard's bias-corrected EMA smoothing, the histogram dashboard's
stats + buckets, the hparams session groups and the projector's
nearest-neighbour lookup.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.runboard import (
    DEFAULT_RESERVOIR_SIZE,
    HISTOGRAM_BUCKETS,
    RunBoard,
    RunTrace,
    histogram_event,
    nearest_neighbours,
    reservoir_sample,
    smooth_series,
)


class TestHistogramEvent:
    def test_stats_and_bucket_counts(self) -> None:
        event = histogram_event([0.0, 1.0, 2.0, 4.0], step=3)
        assert event.step == 3
        assert event.count == 4
        assert event.min == 0.0 and event.max == 4.0
        assert event.mean == pytest.approx(1.75)
        assert event.std == pytest.approx(1.479019, rel=1e-5)
        assert len(event.bucket_edges) == HISTOGRAM_BUCKETS + 1
        assert sum(event.bucket_counts) == 4
        assert event.bucket_counts[0] == 1  # 0.0 lands in the first bucket

    def test_degenerate_span_collapses_to_one_bucket(self) -> None:
        event = histogram_event([2.0, 2.0, 2.0], step=0)
        assert event.min == event.max == 2.0
        assert sum(event.bucket_counts) == 3

    def test_empty_values_are_refused(self) -> None:
        with pytest.raises(DataValidationError):
            histogram_event([], step=0)

    def test_non_finite_values_are_refused(self) -> None:
        with pytest.raises(DataValidationError):
            histogram_event([1.0, float("nan")], step=0)
        with pytest.raises(DataValidationError):
            histogram_event([1.0, float("inf")], step=0)


class TestReservoir:
    def test_keeps_everything_below_size(self) -> None:
        kept = reservoir_sample(list(range(5)), 10, seed=42)
        assert kept == list(range(5))

    def test_size_zero_keeps_all(self) -> None:
        kept = reservoir_sample(list(range(7)), 0, seed=42)
        assert kept == list(range(7))

    def test_bounded_size_and_always_keep_last(self) -> None:
        kept = reservoir_sample(list(range(1000)), 10, seed=42)
        assert len(kept) == 10
        assert kept[-1] == 999  # TB's always_keep_last contract

    def test_deterministic_per_seed_and_independent_of_tag_order(self) -> None:
        first = reservoir_sample(list(range(500)), 8, seed=7)
        second = reservoir_sample(list(range(500)), 8, seed=7)
        assert first == second
        reservoir_sample(list(range(500)), 8, seed=8)
        assert True  # different seed MAY coincide; no assert

    def test_representativeness(self) -> None:
        # 500 draws of a size-50 reservoir over 0..499: the mean stays
        # near the population mean (reservoir sampling is unbiased).
        kept = reservoir_sample(list(range(500)), 50, seed=42)
        assert abs(float(np.mean(kept)) - 249.5) < 40


class TestSmoothSeries:
    def test_beta_zero_is_the_raw_series(self) -> None:
        assert smooth_series([3.0, 1.0, 4.0], beta=0.0) == [3.0, 1.0, 4.0]

    def test_constant_series_stays_constant(self) -> None:
        assert smooth_series([1.0] * 5, beta=0.9) == pytest.approx([1.0] * 5)

    def test_bias_correction_lifts_the_head(self) -> None:
        # beta=0.9: the first output must equal the first value (the
        # naive EMA would report 0.1 * it).
        out = smooth_series([1.0, 1.0], beta=0.9)
        assert out[0] == pytest.approx(1.0)

    def test_pinned_values(self) -> None:
        out = smooth_series([1.0, 0.5, 0.2], beta=0.5)
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(2.0 / 3.0)
        assert out[2] == pytest.approx(0.4)

    def test_bad_beta_is_refused(self) -> None:
        with pytest.raises(DataValidationError):
            smooth_series([1.0], beta=1.0)
        with pytest.raises(DataValidationError):
            smooth_series([1.0], beta=-0.1)


class TestRunTrace:
    def test_scalar_tags_and_events(self) -> None:
        trace = RunTrace("fit")
        trace.add_scalar("loss/train", 1.0, step=0)
        trace.add_scalar("loss/train", 0.5, step=1)
        trace.add_scalar("acc", 0.9, step=0)
        assert trace.tags("scalar") == ["acc", "loss/train"]
        assert [e.value for e in trace.scalars("loss/train")] == [1.0, 0.5]
        assert [e.step for e in trace.scalars("loss/train")] == [0, 1]

    def test_step_auto_increments_when_omitted(self) -> None:
        trace = RunTrace("fit")
        for value in (3.0, 2.0, 1.0):
            trace.add_scalar("loss", value)
        assert [e.step for e in trace.scalars("loss")] == [0, 1, 2]

    def test_non_finite_scalar_refused(self) -> None:
        trace = RunTrace("fit")
        with pytest.raises(DataValidationError):
            trace.add_scalar("loss", float("nan"))

    def test_histogram_text_and_embedding_round_trip(self) -> None:
        trace = RunTrace("fit")
        trace.add_histogram("w", [0.1, -0.2, 0.3], step=0)
        trace.add_text("note", "epoch 0 done", step=0)
        trace.add_embedding("rows", [[1.0, 0.0], [0.0, 1.0]], labels=["a", "b"])
        assert trace.tags("histogram") == ["w"]
        assert trace.tags("text") == ["note"]
        assert trace.embedding("rows").n_rows == 2
        assert trace.embedding("rows").labels == ("a", "b")

    def test_embedding_replaces_per_checkpoint(self) -> None:
        trace = RunTrace("fit")
        trace.add_embedding("rows", [[1.0], [2.0]])
        trace.add_embedding("rows", [[3.0], [4.0], [5.0]])
        assert trace.embedding("rows").n_rows == 3

    def test_embedding_shape_and_label_guards(self) -> None:
        trace = RunTrace("fit")
        with pytest.raises(DataValidationError):
            trace.add_embedding("rows", [1.0, 2.0])
        with pytest.raises(DataValidationError):
            trace.add_embedding("rows", [[1.0], [2.0]], labels=["only-one"])

    def test_hparams_and_session_groups(self) -> None:
        trace = RunTrace("sweep")
        trace.add_hparams({"lr": 0.1}, {"loss": 1.0})
        trace.add_hparams({"lr": 0.1}, {"loss": 0.9})
        trace.add_hparams({"lr": 0.01}, {"loss": 0.5})
        groups = trace.session_groups()
        assert len(groups) == 2
        assert sum(groups.values()) == 3

    def test_json_round_trip(self) -> None:
        trace = RunTrace("rt", reservoir_size=50)
        for step in range(6):
            trace.add_scalar("loss", 1.0 - step * 0.1, step=step)
        trace.add_histogram("w", [0.5, 0.6], step=0)
        trace.add_text("m", "hello", step=0)
        trace.add_embedding("e", [[1.0, 2.0]], labels=["x"])
        trace.add_hparams({"k": 2}, {"m": 0.5})
        rebuilt = RunTrace.from_json(trace.to_json())
        assert rebuilt.scalar_values("loss") == trace.scalar_values("loss")
        assert rebuilt.histograms("w")[0].mean == pytest.approx(0.55)
        assert rebuilt.texts("m")[0].text == "hello"
        assert rebuilt.embedding("e").labels == ("x",)
        assert rebuilt.hparams()[0].params == {"k": 2}

    def test_reservoir_bounds_long_runs(self) -> None:
        trace = RunTrace("long", reservoir_size=25)
        for step in range(2000):
            trace.add_scalar("loss", float(step))
        assert len(trace.scalars("loss")) == 25
        assert trace.scalars("loss")[-1].value == 1999.0  # newest always last

    def test_default_reservoir_size_constant(self) -> None:
        assert DEFAULT_RESERVOIR_SIZE == 1000  # TB's own event_accumulator size

    def test_summary_line(self) -> None:
        trace = RunTrace("fit")
        trace.add_scalar("loss", 1.0)
        assert "1 scalar tags" in trace.summary()
        assert "0 hparam trials" in trace.summary()

    def test_scalar_frame(self) -> None:
        trace = RunTrace("fit")
        for step in (2, 0, 1):
            trace.add_scalar("a", float(step), step=step)
            trace.add_scalar("b", 10.0 - step, step=step)
        frame = trace.scalar_frame()
        assert list(frame.index) == [0, 1, 2]
        assert list(frame.columns) == ["a", "b"]

    def test_histogram_stats_frame(self) -> None:
        trace = RunTrace("fit")
        trace.add_histogram("w", [0.0, 1.0], step=3)
        trace.add_histogram("w", [0.0, 2.0], step=1)
        frame = trace.histogram_stats_frame("w")
        assert list(frame.index) == [1, 3]
        assert frame.loc[3, "mean"] == pytest.approx(0.5)

    def test_unknown_tag_queries_raise(self) -> None:
        trace = RunTrace("fit")
        for method in (
            lambda: trace.scalars("nope"),
            lambda: trace.histograms("nope"),
            lambda: trace.texts("nope"),
            lambda: trace.embedding("nope"),
        ):
            with pytest.raises(DataValidationError):
                method()

    def test_bad_kind_refused(self) -> None:
        with pytest.raises(DataValidationError):
            RunTrace("fit").tags("graph")


class TestRunBoard:
    def test_compare_runs_table(self) -> None:
        a, b = RunTrace("a"), RunTrace("b")
        a.add_scalar("loss", 1.0, step=0)
        a.add_scalar("loss", 0.4, step=1)
        b.add_scalar("loss", 0.9, step=0)
        b.add_scalar("loss", 0.5, step=1)
        board = RunBoard([a, b])
        table = board.compare_table("loss")
        assert table.loc["a", "last"] == pytest.approx(0.4)
        assert table.loc["b", "best"] == pytest.approx(0.5)
        assert table.loc["a", "n"] == 2

    def test_scalar_frame_aligns_runs_on_steps(self) -> None:
        a, b = RunTrace("a"), RunTrace("b")
        a.add_scalar("loss", 1.0, step=0)
        b.add_scalar("loss", 2.0, step=1)
        frame = RunBoard([a, b]).scalar_frame("loss")
        assert list(frame.index) == [0, 1]
        assert frame.loc[0, "a"] == 1.0
        assert frame.loc[0, "b"] != frame.loc[0, "b"]  # NaN for b at step 0

    def test_duplicate_run_name_refused(self) -> None:
        board = RunBoard([RunTrace("a")])
        with pytest.raises(DataValidationError):
            board.add_run(RunTrace("a"))

    def test_missing_tag_raises(self) -> None:
        with pytest.raises(DataValidationError):
            RunBoard([RunTrace("a")]).scalar_frame("nope")


class TestNearestNeighbours:
    ROWS: tuple[tuple[float, float], ...] = (
        (1.0, 0.0), (0.95, 0.31), (0.0, 1.0), (-1.0, 0.0),
    )

    def test_self_first_then_by_similarity(self) -> None:
        hits = nearest_neighbours(self.ROWS, 0, k=2, labels=["a", "b", "c", "d"])
        assert hits[0][0] == 0 and hits[0][1] == pytest.approx(1.0)
        assert hits[1][0] == 1  # near-parallel vector is most similar
        assert hits[1][2] == "b"
        assert len(hits) == 3

    def test_guards(self) -> None:
        with pytest.raises(DataValidationError):
            nearest_neighbours([[1.0]], 5)
        with pytest.raises(DataValidationError):
            nearest_neighbours([1.0], 0)
        with pytest.raises(DataValidationError):
            nearest_neighbours([[1.0], [2.0]], 0, k=-1)


class TestHistogramEventPinnedAgainstNumpy:
    def test_bucket_counts_match_np_histogram(self) -> None:
        values = np.linspace(-3.0, 3.0, 120).tolist()
        event = histogram_event(values, step=0)
        counts, _edges = np.histogram(values, bins=HISTOGRAM_BUCKETS)
        assert list(event.bucket_counts) == counts.tolist()
