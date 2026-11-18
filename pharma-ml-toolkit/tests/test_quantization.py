"""Tests for ``pharmaml.ml.quantization`` -- NF4 / int8 / LLM.int8() engines.

The lost round-188/189 engine rebuilt per ``scratch/round189_state_after_rollback.md``
§3: the 16-level NF4 grid with block-64 absmax scaling and double-quantized
scales, symmetric int8 absmax (per-tensor and per-row), the LLM.int8()
outlier-split matmul with the 6.0 threshold, and the report bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.ml.quantization import (
    NF4_BLOCK_SIZE,
    NF4_LEVELS,
    OUTLIER_THRESHOLD,
    Int8Quantized,
    NF4Quantized,
    QuantizationReport,
    dequantize_int8,
    llm_int8_matmul,
    quantization_report,
    quantize_int8,
    quantize_nf4,
    quantize_scales_int8,
)

# -- the NF4 grid -------------------------------------------------------------- #


def test_nf4_grid_has_sixteen_levels() -> None:
    assert len(NF4_LEVELS) == 16


def test_nf4_grid_spans_minus_one_to_one() -> None:
    assert NF4_LEVELS[0] == -1.0
    assert NF4_LEVELS[-1] == 1.0
    assert NF4_LEVELS[7] == 0.0  # the zero code sits at index 7


def test_nf4_grid_is_sorted() -> None:
    assert np.all(np.diff(NF4_LEVELS) > 0)


def test_nf4_grid_is_not_uniformly_spaced() -> None:
    # the normal-quantile fit: central gaps must differ from edge gaps
    gaps = np.diff(NF4_LEVELS)
    assert len(set(np.round(gaps, 4))) > 4


def test_nf4_grid_denser_near_zero_than_edges() -> None:
    gaps = np.diff(NF4_LEVELS)
    assert gaps[6] < gaps[0]  # the central spacing beats the edge spacing


def test_default_block_size_is_64() -> None:
    assert NF4_BLOCK_SIZE == 64


def test_outlier_threshold_is_the_papers_six() -> None:
    assert OUTLIER_THRESHOLD == 6.0


# -- NF4 quantize / dequantize --------------------------------------------------- #


def test_nf4_round_trip_shape_preserved() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(6, 11))  # 66 elements -> 2 blocks (padding)
    packed = quantize_nf4(w)
    assert packed.shape == (6, 11)
    assert packed.dequantize().shape == w.shape


def test_nf4_one_block_low_error() -> None:
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 16))  # exactly one 64 block
    restored = quantize_nf4(w).dequantize()
    rmse = float(np.sqrt(np.mean((w - restored) ** 2)))
    assert rmse < 0.12


def test_nf4_error_below_naive_uniform_4bit() -> None:
    # uniform 16-level on the same block must lose to the normal-fit grid
    rng = np.random.default_rng(2)
    w = rng.normal(size=(64,))
    restored = quantize_nf4(w).dequantize()
    nf4_rmse = float(np.sqrt(np.mean((w - restored) ** 2)))
    absmax = float(np.abs(w).max())
    uni = np.round(np.clip(w / absmax, -1, 1) * 7.5) / 7.5 * absmax
    uni_rmse = float(np.sqrt(np.mean((w - uni) ** 2)))
    assert nf4_rmse <= uni_rmse


def test_nf4_codes_are_uint8_within_grid() -> None:
    rng = np.random.default_rng(3)
    w = rng.normal(size=(5, 64))
    packed = quantize_nf4(w)
    assert packed.codes.dtype == np.uint8
    assert int(packed.codes.max()) <= 15
    assert int(packed.codes.min()) >= 0


def test_nf4_block_count() -> None:
    w = np.ones((8, 16))  # 128 elements / 64 -> 2 blocks
    packed = quantize_nf4(w)
    assert len(packed.block_scales) == 2
    assert len(packed.scale_codes) == 2


def test_nf4_padding_zero_entries_are_cut() -> None:
    w = np.arange(65, dtype=float).reshape(65, 1) / 65.0  # 65 -> pad 63
    packed = quantize_nf4(w)
    assert packed.dequantize().shape == (65, 1)


def test_nf4_double_quantized_scales_reconstruct() -> None:
    rng = np.random.default_rng(4)
    w = rng.normal(size=(4, 64)) * 3.0
    packed = quantize_nf4(w)
    scales = (
        packed.scale_offset
        + np.asarray(packed.scale_codes, dtype=np.float64) * packed.scale_step
    )
    assert np.max(np.abs(scales - packed.block_scales)) < 0.05


def test_nf4_zero_block_all_zero_codes_at_zero_level() -> None:
    w = np.zeros((2, 64))
    packed = quantize_nf4(w)
    restored = packed.dequantize()
    assert np.allclose(restored, 0.0)


def test_nf4_constant_block_small_error() -> None:
    w = np.full((2, 64), 0.5)
    restored = quantize_nf4(w).dequantize()
    assert np.max(np.abs(restored - 0.5)) < 0.12


def test_nf4_empty_weights() -> None:
    packed = quantize_nf4(np.array([]))
    assert packed.dequantize().size == 0
    assert packed.bits_per_weight() == 0.0


def test_nf4_bits_per_weight_amortizes_scales() -> None:
    w = np.ones((64, 64))  # 4096 weights, 64 blocks
    packed = quantize_nf4(w)
    bpw = packed.bits_per_weight()
    assert 4.0 < bpw < 4.2  # 4 + (8*64 + 32) / 4096 = 4.1328


def test_nf4_rejects_bad_block_size() -> None:
    with pytest.raises(ValueError):
        quantize_nf4(np.ones(64), block_size=0)


def test_nf4_dequantize_functional_twin() -> None:
    from pharmaml.ml.quantization import dequantize_nf4

    rng = np.random.default_rng(5)
    w = rng.normal(size=(3, 64))
    packed = quantize_nf4(w)
    assert np.array_equal(dequantize_nf4(packed), packed.dequantize())


def test_nf4quantized_is_a_dataclass_with_fields() -> None:
    packed = quantize_nf4(np.ones((2, 64)))
    assert isinstance(packed, NF4Quantized)
    assert packed.scale_step > 0


# -- int8 absmax -------------------------------------------------------------------- #


def test_int8_round_trip_per_tensor() -> None:
    rng = np.random.default_rng(6)
    w = rng.normal(size=(5, 7))
    q = quantize_int8(w)
    restored = q.dequantize()
    rel = float(np.sqrt(np.mean((w - restored) ** 2)) / np.sqrt(np.mean(w**2)))
    assert rel < 0.02  # 8 bits: ~1/2^8 relative error scale


def test_int8_scale_is_absmax_over_127() -> None:
    w = np.array([[0.5, -1.0], [0.25, 0.75]])
    q = quantize_int8(w)
    assert q.scale == pytest.approx(1.0 / 127)


def test_int8_codes_within_bounds() -> None:
    rng = np.random.default_rng(7)
    q = quantize_int8(rng.normal(size=(6, 6)) * 10)
    assert int(np.abs(q.codes).max()) <= 127


def test_int8_extreme_entry_maps_to_127() -> None:
    w = np.array([[3.0, 1.0]])
    q = quantize_int8(w)
    assert q.codes[0, 0] == 127


def test_int8_per_row_scales() -> None:
    w = np.array([[1.0, 0.1], [0.5, 0.4]])
    q = quantize_int8(w, axis=1)
    assert np.shape(q.scale) == (2,)
    assert q.scale[0] == pytest.approx(1.0 / 127)
    assert q.scale[1] == pytest.approx(0.5 / 127)
    assert q.axis == 1


def test_int8_per_row_round_trip() -> None:
    rng = np.random.default_rng(8)
    w = rng.normal(size=(8, 6)) * np.array([1.0, 0.01, 1.0, 1.0, 1.0, 1.0])
    restored = quantize_int8(w, axis=1).dequantize()
    # the tiny-magnitude row survives far better than per-tensor would
    rel_row = np.sqrt(np.mean((w[1] - restored[1]) ** 2)) / np.sqrt(
        np.mean(w[1] ** 2)
    )
    per_tensor = dequantize_int8(quantize_int8(w))
    rel_tensor = np.sqrt(np.mean((w[1] - per_tensor[1]) ** 2)) / np.sqrt(
        np.mean(w[1] ** 2)
    )
    assert rel_row <= rel_tensor


def test_int8_per_column_axis0() -> None:
    w = np.array([[1.0, 0.1], [0.9, 0.2]])
    q = quantize_int8(w, axis=0)
    assert np.shape(q.scale) == (2,)
    assert q.axis == 0
    restored = q.dequantize()
    assert restored.shape == w.shape


def test_int8_zero_field_is_zero() -> None:
    q = quantize_int8(np.ones((2, 2)))
    assert q.zero == 0.0


def test_int8_shape_preserved() -> None:
    q = quantize_int8(np.ones((3, 4, 5)))
    assert q.shape == (3, 4, 5)


def test_dequantize_int8_standalone() -> None:
    q = Int8Quantized(
        codes=np.array([[127, -127]], dtype=np.int8), scale=1.0 / 127,
        shape=(1, 2),
    )
    assert dequantize_int8(q).round(6).tolist() == [[1.0, -1.0]]


# -- double quantization helper --------------------------------------------------------- #


def test_scale_quantization_round_trip() -> None:
    scales = np.abs(np.random.default_rng(9).normal(size=64)) + 0.1
    offset, codes, step = quantize_scales_int8(scales)
    restored = offset + codes.astype(np.float64) * step
    assert np.max(np.abs(restored - scales)) < 0.01


def test_scale_quantization_empty() -> None:
    offset, codes, step = quantize_scales_int8(np.array([]))
    assert offset == 0.0 and len(codes) == 0 and step == 1.0


def test_scale_quantization_codes_are_int8() -> None:
    _, codes, _ = quantize_scales_int8(np.array([1.0, 2.0, 3.0]))
    assert codes.dtype == np.int8


# -- LLM.int8() --------------------------------------------------------------------------- #


def test_llm_int8_outlier_row_is_exact() -> None:
    a = np.array([[6.5, 0.1], [0.2, 0.3]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    c = llm_int8_matmul(a, b)
    assert c[0].tolist() == (a @ b)[0].tolist()  # fp path, bit-exact


def test_llm_int8_inlier_row_is_close() -> None:
    rng = np.random.default_rng(10)
    a = rng.normal(0.0, 0.5, size=(8, 6))
    b = rng.normal(0.0, 0.5, size=(6, 4))
    c = llm_int8_matmul(a, b)
    rel = np.linalg.norm(c - a @ b) / np.linalg.norm(a @ b)
    assert rel < 0.02


def test_llm_int8_threshold_routing() -> None:
    a = np.array([[0.1, 0.2], [6.0, 0.1]])
    b = np.eye(2)
    c = llm_int8_matmul(a, b, threshold=6.0)
    # 6.0 >= threshold -> row 1 routes to fp; row 0 int8
    assert c[1].tolist() == a[1].tolist()


def test_llm_int8_default_threshold_is_six() -> None:
    a = np.array([[5.9, 0.1], [6.1, 0.1]])
    c = llm_int8_matmul(a, np.eye(2))
    assert c[1].tolist() == a[1].tolist()


def test_llm_int8_all_inliers_all_outliers() -> None:
    b = np.eye(3)
    small = np.full((2, 3), 0.5)
    big = np.full((2, 3), 7.0)
    assert np.allclose(llm_int8_matmul(small, b), small, atol=0.01)
    assert llm_int8_matmul(big, b).tolist() == big.tolist()


def test_llm_int8_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        llm_int8_matmul(np.ones(4), np.ones((4, 4)))


def test_llm_int8_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        llm_int8_matmul(np.ones((2, 3)), np.ones((4, 5)))


def test_llm_int8_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError):
        llm_int8_matmul(np.ones((2, 2)), np.ones((2, 2)), threshold=0.0)


# -- the report ------------------------------------------------------------------------------ #


def test_report_int8_bits() -> None:
    rng = np.random.default_rng(11)
    report = quantization_report(rng.normal(size=(8, 8)), method="int8")
    assert report.bits_per_weight == 8.0
    assert report.n_weights == 64


def test_report_nf4_bits_near_four() -> None:
    rng = np.random.default_rng(12)
    report = quantization_report(rng.normal(size=(16, 64)), method="nf4")
    assert 4.0 <= report.bits_per_weight < 4.2
    assert report.rmse > 0


def test_report_relative_error_consistent() -> None:
    rng = np.random.default_rng(13)
    w = rng.normal(size=(32, 32))
    report = quantization_report(w, method="int8")
    rms = float(np.sqrt(np.mean(w**2)))
    assert report.relative_error == pytest.approx(report.rmse / rms)


def test_report_as_dict_fields() -> None:
    rng = np.random.default_rng(14)
    payload = quantization_report(rng.normal(size=(4, 4)), method="int8_channel").as_dict()
    assert set(payload) == {
        "method", "n_weights", "bits_per_weight", "rmse",
        "relative_error", "shape",
    }
    assert payload["method"] == "int8_channel"


def test_report_unknown_method() -> None:
    with pytest.raises(ValueError):
        quantization_report(np.ones((2, 2)), method="fp8")


def test_report_empty_weights() -> None:
    with pytest.raises(ValueError):
        quantization_report(np.array([]), method="int8")


def test_report_int8_channel_beats_per_tensor_on_row_scale_spread() -> None:
    rng = np.random.default_rng(15)
    w = rng.normal(size=(8, 8)) * np.array([10.0, 0.1, 1, 1, 1, 1, 1, 1])
    tensor = quantization_report(w, method="int8")
    channel = quantization_report(w, method="int8_channel")
    assert channel.rmse < tensor.rmse


def test_quantization_report_dataclass() -> None:
    assert QuantizationReport is not None
