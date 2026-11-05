"""Regression tests for the GitHub registry's deferred XAI fold-ins.

The numerical pins are independent of the GUI and of the three source
projects: torchlens-style graph records, alibi-style empirical tabular
anchors, and Quantus-style complexity/faithfulness/robustness metrics.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pharmaml.ml.numpy_neural import NumpyMLPRegressor
from pharmaml.ml.xai import (
    XaiError,
    attribution_complexity,
    evaluate_attribution_metrics,
    faithfulness_estimate,
    max_sensitivity,
    model_activation_trace,
    occlusion_attributions,
    tabular_anchor,
)


class _ThresholdClassifier:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        return (values[:, 0] >= 0.0).astype(int)


class _LinearRegressor:
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float) @ self.weights


def _network() -> tuple[NumpyMLPRegressor, np.ndarray]:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 4))
    target = matrix @ np.array([2.0, -1.0, 0.5, 0.25])
    network = NumpyMLPRegressor(
        hidden_layers=(3, 2), epochs=5, random_state=0
    ).fit(matrix, target)
    return network, matrix


class TestActivationGraph:
    def test_every_operation_has_shape_parent_parameter_and_memory_evidence(self) -> None:
        network, matrix = _network()
        records = model_activation_trace(network, matrix[0])

        assert [record.name for record in records] == [
            "input",
            "linear_1",
            "activation_1",
            "linear_2",
            "activation_2",
            "linear_3",
        ]
        assert [record.operation for record in records] == [
            "input",
            "linear",
            "relu",
            "linear",
            "relu",
            "linear",
        ]
        assert records[1].parents == ("input",)
        assert records[2].parents == ("linear_1",)
        assert records[-1].parents == ("activation_2",)
        assert [record.parameters for record in records] == [0, 15, 0, 8, 0, 3]
        assert [record.output_shape for record in records] == [
            (4,),
            (3,),
            (3,),
            (2,),
            (2,),
            (1,),
        ]
        assert all(record.activation_nbytes > 0 for record in records)
        assert all(0.0 <= record.dead_fraction <= 1.0 for record in records)
        assert records[-1].minimum == pytest.approx(records[-1].maximum)

    def test_wrong_width_is_rejected(self) -> None:
        network, _matrix = _network()
        with pytest.raises(XaiError, match="expects 4"):
            model_activation_trace(network, [1.0, 2.0])


class TestTabularAnchor:
    def test_precision_coverage_and_rule_are_measured_on_reference_rows(self) -> None:
        reference = np.array(
            [
                [-2.0, 0.0],
                [-2.0, 1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        result = tabular_anchor(
            _ThresholdClassifier(),
            [1.0, 0.0],
            reference,
            feature_names=["dose", "flag"],
            threshold=0.95,
            max_anchor_size=2,
        )

        assert result.success
        assert result.target == 1
        assert result.precision == pytest.approx(1.0)
        assert result.coverage == pytest.approx(0.25)
        assert result.n_covered == 2
        assert result.n_samples == 8
        assert result.covered_indices == (4, 5)
        assert len(result.conditions) == 1
        assert result.conditions[0].feature_name == "dose"
        assert result.rule == result.names[0]
        assert "dose" in result.rule
        assert result.candidates_evaluated >= 3  # empty + both one-feature candidates

    def test_empty_anchor_is_allowed_when_the_whole_frame_is_precise(self) -> None:
        class Constant:
            def predict(self, matrix: np.ndarray) -> np.ndarray:
                return np.ones(len(matrix), dtype=int)

        result = tabular_anchor(Constant(), [0.0], [[0.0], [0.0]], threshold=0.9)
        assert result.success
        assert result.conditions == ()
        assert result.rule == "TRUE (empty anchor)"
        assert result.coverage == result.precision == 1.0

    def test_unreachable_threshold_returns_best_candidate_with_honest_failure(self) -> None:
        class Alternating:
            def predict(self, matrix: np.ndarray) -> np.ndarray:
                return np.arange(len(matrix)) % 2

        result = tabular_anchor(
            Alternating(), [0.0], [[0.0], [0.0]], threshold=0.99
        )
        assert not result.success
        assert result.precision == pytest.approx(0.5)
        assert result.coverage == pytest.approx(1.0)

    def test_validation_and_missing_predictor(self) -> None:
        with pytest.raises(XaiError, match="at least two rows"):
            tabular_anchor(_ThresholdClassifier(), [1.0, 0.0], [[1.0, 0.0]])
        with pytest.raises(XaiError, match="threshold"):
            tabular_anchor(
                _ThresholdClassifier(), [1.0, 0.0], [[1.0, 0.0]] * 2, threshold=0.0
            )
        with pytest.raises(XaiError, match="finite"):
            tabular_anchor(
                _ThresholdClassifier(), [1.0, 0.0], [[1.0, np.nan]] * 2
            )
        with pytest.raises(XaiError, match="predict"):
            tabular_anchor(object(), [1.0], [[1.0], [2.0]])


class TestQuantusMetrics:
    def test_complexity_matches_hand_computed_entropy_and_gini(self) -> None:
        result = attribution_complexity([1.0, -1.0, 0.0, 0.0])
        assert result.entropy == pytest.approx(math.log(2.0))
        assert result.normalized_entropy == pytest.approx(0.5)
        assert result.effective_units == pytest.approx(2.0)
        assert result.gini == pytest.approx(0.5)
        assert result.nonzero_fraction == pytest.approx(0.5)

        zero = attribution_complexity([0.0, 0.0])
        assert zero.entropy == zero.effective_units == zero.nonzero_fraction == 0.0

    def test_faithfulness_recovers_individual_linear_prediction_drops(self) -> None:
        weights = np.array([1.0, 2.0, -1.0])
        row = np.array([3.0, 1.0, 2.0])
        estimator = _LinearRegressor(weights)
        attributions = weights * row
        result = faithfulness_estimate(estimator, row, attributions)

        order = np.argsort(-attributions)
        assert result.attribution_sums == pytest.approx(attributions[order])
        assert result.prediction_drops == pytest.approx(attributions[order])
        assert result.score == pytest.approx(1.0)

        grouped = faithfulness_estimate(
            estimator, row, attributions, features_in_step=2
        )
        assert grouped.attribution_sums == pytest.approx([5.0, -2.0])
        assert grouped.prediction_drops == pytest.approx([5.0, -2.0])
        assert grouped.score == pytest.approx(1.0)

    def test_undefined_faithfulness_stays_nan_with_a_reason(self) -> None:
        estimator = _LinearRegressor(np.array([1.0, 1.0]))
        result = faithfulness_estimate(estimator, [1.0, 2.0], [1.0, 1.0])
        assert math.isnan(result.score)
        assert "undefined" in result.note

    def test_max_sensitivity_is_seeded_bounded_and_recomputable(self) -> None:
        row = np.array([1.0, 2.0, 3.0])

        def explain(values: np.ndarray) -> np.ndarray:
            return 2.0 * values

        first = max_sensitivity(row, explain, n_samples=8, noise=0.2, seed=9)
        second = max_sensitivity(row, explain, n_samples=8, noise=0.2, seed=9)
        assert first.samples == pytest.approx(second.samples)
        assert first.score == pytest.approx(first.samples.max())
        assert first.mean == pytest.approx(first.samples.mean())
        assert first.score >= first.mean >= 0.0

        unchanged = max_sensitivity(row, explain, n_samples=2, noise=0.0)
        assert unchanged.score == pytest.approx(0.0)
        with pytest.raises(XaiError, match="all-zero"):
            max_sensitivity(row, lambda _values: np.zeros(3))

    def test_combined_report_uses_the_same_attribution_method(self) -> None:
        network, matrix = _network()
        row = matrix[0]
        base = occlusion_attributions(network, row).attributions

        def explain(values: np.ndarray) -> np.ndarray:
            return occlusion_attributions(network, values).attributions

        report = evaluate_attribution_metrics(
            network, row, base, explain_fn=explain, n_samples=3, seed=2
        )
        assert report.complexity.entropy >= 0.0
        assert report.faithfulness.score == pytest.approx(1.0)
        assert report.sensitivity is not None
        assert report.sensitivity.samples.shape == (3,)
        assert np.all(np.isfinite(report.sensitivity.samples))
