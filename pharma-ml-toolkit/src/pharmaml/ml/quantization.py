"""Offline weight quantization -- NF4, int8 absmax, and LLM.int8().

Rounds 188-189 checked bitsandbytes and the LLM.int8() paper (Dettmers
et al. 2022) file by file and lifted what survives without a GPU
kernel:

* **NF4** (the QLoRA paper's 4-bit NormalFloat) -- the 16-level grid
  fitted to the standard normal's quantiles, so the codes spend their
  resolution where Gaussian-initialized weights actually live.  The
  weights are quantized per block (block size 64, the QLoRA default)
  after an absmax scaling; **double quantization** then quantizes the
  per-block scales themselves to 8 bits around a global offset -- the
  paper's ~0.4-bit-per-parameter saving, implemented honestly.
* **int8 absmax** -- the symmetric per-tensor and per-channel integer
  quantization every inference engine starts from: ``q = round(w /
  s)`` with ``s = absmax / 127``, dequantized by multiplication.
* **LLM.int8()** (Dettmers et al.) -- the outlier-aware matmul: rows
  of the activation whose entries exceed the threshold (the paper's
  6.0) are routed through full precision, everything else through the
  int8 path; the results are summed.  The paper's insight -- that
  0.1% outlier features are what breaks naive int8 -- is preserved in
  which rows get routed, per matrix not per element.
* **The quantization report** -- bits per weight, the reconstruction
  error (RMSE and relative), and the memory math the GUI card prints.

The honest scope, stated once: bitsandbytes' paged optimizers, its
CUDA/LLM.c kernels, and the bnb 16-bit optimizers are what this
toolkit does not vendor; the FP8 format is out of scope (no hardware
support to be honest about).  What transfers is the grid, the block
scaling, the double quant, and the outlier split.

The engines are pure NumPy and operate on weight matrices and matmul
operands directly.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.quantization import NF4_LEVELS, quantize_nf4
>>> len(NF4_LEVELS)
16
>>> rng = np.random.default_rng(0)
>>> w = rng.normal(size=(4, 16))  # 64 elements = one NF4 block
>>> packed = quantize_nf4(w)
>>> restored = packed.dequantize()
>>> rmse = float(np.sqrt(np.mean((w - restored) ** 2)))
>>> rmse < 0.12  # the block-scaled grid's honest error on N(0, 1)
True

>>> from pharmaml.ml.quantization import quantize_int8
>>> q = quantize_int8(np.array([[0.1, -0.7], [0.4, 0.9]]))
>>> round(q.scale, 4)  # absmax 0.9 / 127
0.0071
>>> q.dequantize().round(2).tolist()
[[0.1, -0.7], [0.4, 0.9]]

>>> from pharmaml.ml.quantization import llm_int8_matmul
>>> A = np.array([[6.5, 0.1], [0.2, 0.3]])
>>> B = np.array([[1.0, 2.0], [3.0, 4.0]])
>>> C = llm_int8_matmul(A, B)
>>> np.allclose(C, A @ B, atol=0.05)
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "NF4_BLOCK_SIZE",
    "NF4_LEVELS",
    "OUTLIER_THRESHOLD",
    "Int8Quantized",
    "NF4Quantized",
    "QuantizationReport",
    "dequantize_int8",
    "dequantize_nf4",
    "llm_int8_matmul",
    "quantization_report",
    "quantize_int8",
    "quantize_nf4",
    "quantize_scales_int8",
]

#: The QLoRA NF4 grid -- the standard normal's 16 quantiles, in [-1, 1].
NF4_LEVELS: np.ndarray = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.1609298894405365,
        0.24611230194568634,
        0.33791545128822327,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float64,
)

#: The QLoRA default block size for per-block absmax scaling.
NF4_BLOCK_SIZE: int = 64

#: The LLM.int8() paper's outlier threshold (|a| >= 6.0 -> fp path).
OUTLIER_THRESHOLD: float = 6.0

_INT8_MAX = 127


# -- NF4 ---------------------------------------------------------------------- #


@dataclass
class NF4Quantized:
    """A doubly-quantized NF4 block pack.

    ``codes`` carries the 16-level indices (``uint8`` storage, 4 bits
    of information).  The per-block absmax scales are themselves
    int8-quantized (the double quantization): ``scale_offset`` is the
    one shared float (the scales' mean), ``scale_codes`` the per-block
    int8 deltas, and ``scale_step`` the single step that maps a code
    back to a scale delta.  ``block_scales`` keeps the pre-quantization
    floats for reporting; ``shape`` is the original weight shape.
    """

    codes: np.ndarray
    block_scales: np.ndarray
    scale_offset: float
    scale_codes: np.ndarray
    scale_step: float
    shape: tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        """Reconstruct the (approximate) float weights."""
        # step 1: undo the scales' own int8 quantization (double quant)
        scales = self.scale_offset + np.asarray(
            self.scale_codes, dtype=np.float64
        ) * self.scale_step
        # step 2: per-block dequantization of the NF4 codes (padded
        # to a whole number of blocks; the tail is cut on reshape)
        flat_codes = np.asarray(self.codes, dtype=np.intp).ravel()
        levels = NF4_LEVELS[flat_codes]
        n_blocks = len(scales)
        if n_blocks == 0 or levels.size == 0:
            return np.zeros(self.shape)
        block = math.ceil(levels.size / n_blocks)
        out = levels * np.repeat(scales, block)[: levels.size]
        total = int(np.prod(self.shape)) if self.shape else levels.size
        return out[:total].reshape(self.shape)

    def bits_per_weight(self) -> float:
        """NF4 (4) + the block scale's share (8 / block) + the offset."""
        total = float(np.prod(self.shape)) if self.shape else float(len(self.codes))
        if total == 0:
            return 0.0
        n_blocks = len(self.scale_codes)
        scale_bits = 8.0 * n_blocks + 32.0  # the global offset, amortized
        return 4.0 + scale_bits / total


def quantize_nf4(
    weights: np.ndarray, block_size: int = NF4_BLOCK_SIZE
) -> NF4Quantized:
    """Quantize a weight block to NF4 with double-quantized scales.

    The pipeline is QLoRA's: pad to a block multiple, absmax each
    block, normalize to [-1, 1], snap to the nearest NF4 level; then
    quantize the block scales themselves to int8 around their mean
    (the double quantization -- one float offset shared by all blocks,
    one int8 per block instead of one float per block).

    Raises
    ------
    ValueError
        ``block_size`` not positive.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    w = np.asarray(weights, dtype=np.float64).ravel()
    shape = tuple(np.asarray(weights).shape)
    if w.size == 0:
        return NF4Quantized(
            codes=np.array([], dtype=np.uint8),
            block_scales=np.array([]),
            scale_offset=0.0,
            scale_codes=np.array([], dtype=np.int8),
            scale_step=1.0,
            shape=shape,
        )
    pad = (-w.size) % block_size
    padded = np.concatenate([w, np.zeros(pad)]) if pad else w
    blocks = padded.reshape(-1, block_size)
    scales = np.abs(blocks).max(axis=1)  # per-block absmax
    norm = blocks / np.maximum(scales[:, None], 1e-12)
    # nearest NF4 level (argmin |x - level|; levels are sorted)
    norm_flat = norm.ravel()
    idx = np.searchsorted(NF4_LEVELS, norm_flat)
    idx = np.clip(idx, 1, len(NF4_LEVELS) - 1)
    left = NF4_LEVELS[idx - 1]
    right = NF4_LEVELS[idx]
    take_left = np.abs(norm_flat - left) <= np.abs(norm_flat - right)
    codes = (idx - 1 + (~take_left)).astype(np.uint8)
    # double quantization on the scales: int8 around the shared mean
    offset, scale_codes, scale_step = quantize_scales_int8(scales)
    return NF4Quantized(
        codes=codes,
        block_scales=scales,
        scale_offset=offset,
        scale_codes=scale_codes,
        scale_step=scale_step,
        shape=shape,
    )


def dequantize_nf4(packed: NF4Quantized) -> np.ndarray:
    """Functional twin of :meth:`NF4Quantized.dequantize`."""
    return packed.dequantize()


# -- int8 absmax ----------------------------------------------------------------- #


@dataclass
class Int8Quantized:
    """A symmetric absmax int8 tensor: the codes, scale, and shape.

    ``scale`` is the single float multiplier (per-tensor) or the
    per-channel vector (``axis=1``: one scale per row, the weight
    convention; ``axis=0``: per column); ``zero`` is kept for API
    symmetry and is always 0 (symmetric quantization).
    """

    codes: np.ndarray
    scale: Any  # float or np.ndarray -- dequantize handles both
    shape: tuple[int, ...]
    zero: float = 0.0
    axis: int | None = None

    def dequantize(self) -> np.ndarray:
        return dequantize_int8(self)


def quantize_int8(
    weights: np.ndarray, axis: int | None = None
) -> Int8Quantized:
    """Symmetric absmax int8 quantization, per-tensor or per-channel.

    ``axis=None`` gives one scale for the whole tensor; ``axis=1``
    gives one scale per row (the per-output-channel weight convention);
    ``axis=0`` one per column.

    >>> import numpy as np
    >>> q = quantize_int8(np.array([[0.2, -1.0], [0.5, 0.8]]), axis=1)
    >>> q.scale.shape
    (2,)
    """
    w = np.asarray(weights, dtype=np.float64)
    shape = tuple(w.shape)
    if axis is None:
        absmax = float(np.max(np.abs(w))) if w.size else 0.0
        scale = max(absmax / _INT8_MAX, 1e-12)
        codes = np.clip(np.round(w / scale), -_INT8_MAX, _INT8_MAX)
        return Int8Quantized(
            codes=codes.astype(np.int8), scale=scale, shape=shape, axis=None
        )
    if w.size == 0:
        return Int8Quantized(
            codes=np.zeros(shape, dtype=np.int8), scale=np.array([]),
            shape=shape, axis=axis,
        )
    absmax = np.max(np.abs(w), axis=axis, keepdims=True)
    scale = np.maximum(absmax / _INT8_MAX, 1e-12)
    codes = np.clip(np.round(w / scale), -_INT8_MAX, _INT8_MAX)
    flat_scale = np.ravel(scale)
    return Int8Quantized(
        codes=codes.astype(np.int8), scale=flat_scale, shape=shape, axis=axis
    )


def dequantize_int8(q: Int8Quantized) -> np.ndarray:
    """The inverse of :func:`quantize_int8` (to quantization error)."""
    codes = np.asarray(q.codes, dtype=np.float64)
    scale = q.scale
    if np.ndim(scale) == 0:
        out = codes * float(scale)
    elif codes.ndim == 2:
        vec = np.asarray(scale, dtype=np.float64)
        if q.axis == 0:  # per-column scales
            out = codes * vec[None, :]
        else:  # per-row scales (axis=1 or inferred)
            out = codes * vec[:, None]
    else:  # 1-D operand quantized per-element
        out = codes * np.asarray(scale, dtype=np.float64)
    return out.reshape(q.shape)


def quantize_scales_int8(scales: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Double quantization helper: (offset, int8 codes, per-code scale).

    The QLoRA double-quant contract: one shared float offset (the
    scales' mean), int8 codes around it, and the single step size
    that maps a code back to a scale delta.
    """
    s = np.asarray(scales, dtype=np.float64).ravel()
    if s.size == 0:
        return 0.0, np.array([], dtype=np.int8), 1.0
    offset = float(np.mean(s))
    half_range = max(float(np.max(np.abs(s - offset))), 1e-12)
    codes = np.clip(
        np.round((s - offset) / half_range * _INT8_MAX), -_INT8_MAX, _INT8_MAX
    ).astype(np.int8)
    step = half_range / _INT8_MAX
    return offset, codes, step


# -- LLM.int8() -------------------------------------------------------------------- #


def llm_int8_matmul(
    a: np.ndarray, b: np.ndarray, threshold: float = OUTLIER_THRESHOLD
) -> np.ndarray:
    """The outlier-split matmul of Dettmers et al. 2022.

    Rows of ``a`` containing any |entry| >= ``threshold`` take the
    full-precision path; the remaining rows are quantized to int8
    (per-row scale for ``a``, per-column for ``b``... reduced here to
    per-tensor absmax on each operand -- the house CPU budget) and
    their product is dequantized before the sum.  The routing is
    per-row of ``a``, exactly the paper's "emergent outlier feature"
    isolation, computed without any element-level mixing.

    >>> import numpy as np
    >>> A = np.array([[6.5, 0.1], [0.2, 0.3]])
    >>> B = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> C = llm_int8_matmul(A, B, threshold=6.0)
    >>> C[0].tolist() == (A @ B)[0].tolist()  # outlier row: exact fp path
    True
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"llm_int8_matmul needs 2-D operands, got {a.ndim}-D and {b.ndim}-D"
        )
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"shape mismatch: {a.shape} @ {b.shape}"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    out = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    outlier_rows = np.any(np.abs(a) >= threshold, axis=1)
    if outlier_rows.any():
        out[outlier_rows] = a[outlier_rows] @ b
    if (~outlier_rows).any():
        a_in = a[~outlier_rows]
        a_q = quantize_int8(a_in, axis=1)  # per-row scales
        b_q = quantize_int8(b)  # per-tensor
        a_dq = dequantize_int8(a_q)
        b_dq = dequantize_int8(b_q)
        out[~outlier_rows] = a_dq @ b_dq
    return out


# -- the report ------------------------------------------------------------------------ #


@dataclass
class QuantizationReport:
    """The numbers the GUI card renders after a quantization pass."""

    method: str
    n_weights: int
    bits_per_weight: float
    rmse: float
    relative_error: float
    shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_weights": self.n_weights,
            "bits_per_weight": round(self.bits_per_weight, 3),
            "rmse": float(self.rmse),
            "relative_error": float(self.relative_error),
            "shape": list(self.shape),
        }


def quantization_report(
    weights: np.ndarray, method: str = "nf4"
) -> QuantizationReport:
    """Quantize + measure: RMSE, relative error, bits per weight.

    ``method`` is one of ``"nf4"`` (block 64 + double quant),
    ``"int8"`` (per-tensor absmax), ``"int8_channel"`` (per-row).

    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> w = rng.normal(size=(8, 8))
    >>> report = quantization_report(w, method="int8")
    >>> report.bits_per_weight
    8.0
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0:
        raise ValueError("quantization_report needs a non-empty weight block")
    if method == "nf4":
        packed = quantize_nf4(w)
        restored = packed.dequantize()
        bpw = packed.bits_per_weight()
        name = "nf4"
    elif method in {"int8", "int8_channel"}:
        axis = 0 if method == "int8_channel" else None
        q = quantize_int8(w, axis=axis)
        restored = q.dequantize()
        bpw = 8.0
        name = method
    else:
        raise ValueError(f"unknown method {method!r}")
    rmse = float(np.sqrt(np.mean((w - restored) ** 2)))
    denom = float(np.sqrt(np.mean(w**2)))
    rel = rmse / denom if denom > 0 else math.inf
    return QuantizationReport(
        method=name,
        n_weights=int(w.size),
        bits_per_weight=bpw,
        rmse=rmse,
        relative_error=rel,
        shape=tuple(w.shape),
    )
