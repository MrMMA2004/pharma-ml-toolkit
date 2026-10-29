"""The Transformer's mathematics: attention and positional encoding.

Chapter 15 of François Chollet's *Deep Learning with Python* notebooks
(``fchollet/deep-learning-with-python-notebooks``) builds the
Transformer brick by brick -- dot-product attention, the encoder block
with its residual + layer-norm sandwich, the decoder with its causal
mask, and positional information -- on top of an autodiff engine the
toolkit does not ship.  What needed no engine at all was the
*mathematics*, and that is what lands here: plain NumPy functions
whose formulas match the notebook's layers argument by argument, so a
numeric sequence can be pushed through the exact attention computation
and *seen*.

* :func:`sinusoidal_positions` -- the classic fixed positional
  encoding (``Attention Is All You Need``, which the chapter's
  ``PositionalEmbedding`` section cites next to its learned variant):
  ``sin(pos/10000^(2i/d))`` interleaved with the cosine twin.  The
  notebook's learned ``PositionalEmbedding`` (an embedding table over
  positions) is the honest stated difference.
* :func:`dot_product_attention` -- the scaled ``softmax(QKᵀ/√dₖ)V``
  with optional causal or padding masks, numerically stabilised by
  the max-subtraction softmax.  This is Keras'
  ``layers.MultiHeadAttention`` inner step exactly.
* :func:`multi_head_attention` -- the head-split/concat machinery over
  seeded Gaussian projections: ``n_heads`` independent subspaces, each
  attending separately, outputs recombined by ``W_O``.
* :func:`self_attention_report` -- push a matrix through the whole
  forward pass and read the weights: the per-head attention grids,
  each head's entropy (a head that attends uniformly to everything
  carries no information; a sharp head has found a pattern), and each
  position's focus (argmax of its attention row).

The honest scope, stated once: this is the analysis half of the
chapter -- the *trainable* Transformer (the encoder/decoder blocks
with their feed-forward halves and layer norms, backprop through all
of it, and the pretrained-LLM material of chapters 15-16) needs the
autodiff engines the toolkit deliberately refuses to vendor; the
sampling half of chapter 16 lands in
:meth:`pharmaml.generative.rnn.SmilesRnn.sample` (``top_k``), and the
sequence-model half of chapter 13 in
:mod:`pharmaml.ml.sequences`.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.attention import (
...     dot_product_attention, self_attention_report, sinusoidal_positions,
... )
>>> pe = sinusoidal_positions(seq_len=6, dim=8)
>>> pe.shape
(6, 8)
>>> round(float(np.linalg.norm(pe[0])), 4)      # every position's norm is √(dim/2)
2.0

>>> q = np.eye(3)
>>> k = np.eye(3)
>>> v = np.arange(9.0).reshape(3, 3)
>>> output, weights = dot_product_attention(q, k, v)
>>> weights.shape
(3, 3)
>>> round(float(weights.sum(axis=1)[0]), 6)     # rows are distributions
1.0

>>> rng = np.random.default_rng(42)
>>> matrix = rng.normal(size=(10, 8))
>>> report = self_attention_report(matrix, n_heads=4, causal=True, seed=42)
>>> report.weights.shape
(4, 10, 10)
>>> len(report.entropies)
4
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

#: Random seed threaded through the seeded projections (the pin).
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Chapter 15 -- positional information
# ---------------------------------------------------------------------------


def sinusoidal_positions(seq_len: int, dim: int) -> np.ndarray:
    """The classic fixed positional encoding, ``(seq_len, dim)``.

    ``PE[pos, 2i] = sin(pos / 10000^(2i/d))`` and
    ``PE[pos, 2i+1] = cos(pos / 10000^(2i/d))`` -- even positions carry
    the sine, odd ones the cosine (the interleaving every reference
    implementation, including Keras' own
    ``keras_nlp.layers.SinePositionEncoding``, uses).  ``dim`` odd is
    rejected rather than silently truncated: the pairing is the
    formula.

    Raises
    ------
    DataValidationError
        On non-positive sizes or an odd ``dim``.
    """
    if seq_len < 1:
        raise DataValidationError(f"seq_len must be >= 1, got {seq_len}")
    if dim < 2:
        raise DataValidationError(f"dim must be >= 2, got {dim}")
    if dim % 2 != 0:
        raise DataValidationError(f"dim must be even (the sin/cos pairing), got {dim}")
    positions = np.arange(seq_len, dtype=float)[:, None]
    angles = np.arange(0, dim, 2, dtype=float)[None, :]
    denominator = np.power(10000.0, angles / dim)
    encoding = np.zeros((seq_len, dim))
    encoding[:, 0::2] = np.sin(positions / denominator)
    encoding[:, 1::2] = np.cos(positions / denominator)
    return encoding


@dataclass(frozen=True)
class PositionalReport:
    """The positional encoding as an auditable object."""

    encoding: np.ndarray
    seq_len: int
    dim: int

    @property
    def norms(self) -> np.ndarray:
        """L2 norm of every position's code (constant for this scheme)."""
        return np.linalg.norm(self.encoding, axis=1)

    @property
    def adjacent_similarity(self) -> np.ndarray:
        """Cosine similarity between neighbouring positions.

        The slow decay down the sequence is the property the encoding
        exists for: nearby positions share most of their code, distant
        ones nearly none.
        """
        normalised = self.encoding / np.clip(np.linalg.norm(self.encoding, axis=1, keepdims=True), 1e-12, None)
        return np.sum(normalised[1:] * normalised[:-1], axis=1)


# ---------------------------------------------------------------------------
# Chapter 15 -- the attention step
# ---------------------------------------------------------------------------


def _softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax with max subtraction (the stable form)."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def _as_2d(name: str, matrix: np.ndarray, dim: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise DataValidationError(f"{name} must be 2-D, got shape {matrix.shape}")
    if matrix.shape[1] != dim:
        raise DataValidationError(
            f"{name} has feature width {matrix.shape[1]}, expected {dim}"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError(f"{name} contains NaN/inf")
    return matrix.astype(float, copy=False)


def dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention: ``softmax(QKᵀ/√dₖ)·V``.

    Parameters
    ----------
    query, key:
        ``(nq, dk)`` and ``(nk, dk)`` -- the score matrix is
        ``(nq, nk)``.
    value:
        ``(nk, dv)`` -- the attended mixture.
    mask:
        Optional ``(nq, nk)`` boolean array (or anything
        ``np.asarray`` turns boolean): ``True`` = ATTEND, ``False`` =
        blocked.  Masked scores drop to ``-1e9`` before the softmax,
        the standard additive-mask convention.  Pass
        ``~np.triu(np.ones((n, n), dtype=bool), k=1)`` for the causal
        mask the decoder's self-attention uses (chapter 15's
        ``use_causal_mask=True``).

    Returns
    -------
    (output, weights):
        ``(nq, dv)`` and ``(nq, nk)`` -- the weights ARE the
        attention, the thing the lab visualises.

    Raises
    ------
    DataValidationError
        On shape mismatches or NaN input.
    """
    q = np.asarray(query, dtype=float)
    k = np.asarray(key, dtype=float)
    v = np.asarray(value, dtype=float)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise DataValidationError("query/key/value must all be 2-D")
    if q.shape[1] != k.shape[1]:
        raise DataValidationError(
            f"query width {q.shape[1]} != key width {k.shape[1]} (the dot needs the same dk)"
        )
    if k.shape[0] != v.shape[0]:
        raise DataValidationError(
            f"key rows {k.shape[0]} != value rows {v.shape[0]} (attention mixes value rows)"
        )
    if not (np.isfinite(q).all() and np.isfinite(k).all() and np.isfinite(v).all()):
        raise DataValidationError("query/key/value contain NaN/inf")
    scores = q @ k.T / math.sqrt(k.shape[1])
    if mask is not None:
        mask_array = np.asarray(mask)
        if mask_array.shape != scores.shape:
            raise DataValidationError(
                f"mask shape {mask_array.shape} != score shape {scores.shape}"
            )
        scores = np.where(mask_array, scores, -1e9)
    weights = _softmax_rows(scores)
    return weights @ v, weights


def multi_head_attention(
    x: np.ndarray,
    *,
    n_heads: int,
    seed: int = RANDOM_STATE,
    causal: bool = False,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-head self-attention over one sequence, seeded projections.

    The input ``(n, d)`` is projected per head by seeded Gaussian
    ``W_Q``/``W_K``/``W_V`` (``d → d/heads`` each -- the notebook's
    ``key_dim = hidden_dim // num_heads``), every head runs
    :func:`dot_product_attention` over its own subspace, the heads are
    concatenated and mixed back through ``W_O`` (``d → d``).  The
    returned weights stack is ``(heads, n, n)``.

    This is the forward mathematics of Keras'
    ``layers.MultiHeadAttention(num_heads, key_dim)`` on a single
    sequence -- analysis, not training: the projections are random
    draws pinned by ``seed``, because *learning* them is precisely the
    part that needs the engine the toolkit does not vendor.

    Raises
    ------
    DataValidationError
        On a non-2-D input, ``d`` not divisible by ``n_heads``, bad
         head counts or NaN input.
    """
    if n_heads < 1 or n_heads > 64:
        raise DataValidationError(f"n_heads must be in [1, 64], got {n_heads}")
    matrix = np.asarray(x, dtype=float)
    if matrix.ndim != 2:
        raise DataValidationError(f"x must be 2-D (n, d), got shape {matrix.shape}")
    n, d = matrix.shape
    if n < 2:
        raise DataValidationError("the sequence needs at least two tokens")
    if d % n_heads != 0:
        raise DataValidationError(
            f"feature width {d} is not divisible by {n_heads} head(s)"
        )
    if not np.isfinite(matrix).all():
        raise DataValidationError("x contains NaN/inf")
    head_dim = d // n_heads
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(d)
    outputs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for _head in range(n_heads):
        W_Q = rng.normal(scale=scale, size=(d, head_dim))
        W_K = rng.normal(scale=scale, size=(d, head_dim))
        W_V = rng.normal(scale=scale, size=(d, head_dim))
        q = matrix @ W_Q
        k = matrix @ W_K
        v = matrix @ W_V
        if causal:
            head_mask = ~np.triu(np.ones((n, n), dtype=bool), k=1)
        else:
            head_mask = mask
        head_output, head_weights = dot_product_attention(q, k, v, mask=head_mask)
        outputs.append(head_output)
        weights.append(head_weights)
    W_O = rng.normal(scale=1.0 / math.sqrt(d), size=(d, d))
    concatenated = np.concatenate(outputs, axis=1)  # (n, heads * head_dim) == (n, d)
    return concatenated @ W_O, np.stack(weights, axis=0)


# ---------------------------------------------------------------------------
# The report object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionReport:
    """One sequence's pass through the attention machinery.

    ``weights`` is ``(heads, n, n)``; ``entropies`` (nats) one entry
    per head (``ln(n)`` = uniform attention, 0 = a single hard focus);
    ``focus`` is each position's argmax column per head, the "who
    attends to whom" digest.
    """

    weights: np.ndarray
    entropies: tuple[float, ...]
    focus: tuple[tuple[int, ...], ...]
    n_heads: int
    n_tokens: int
    causal: bool
    max_entropy: float

    def headline(self) -> str:
        sharp = [
            index
            for index, entropy in enumerate(self.entropies)
            if entropy < 0.75 * self.max_entropy
        ]
        if sharp:
            listing = ", ".join(f"head {index}" for index in sharp)
            return (
                f"{self.n_heads} head(s) over {self.n_tokens} token(s) · sharp: {listing} "
                f"(entropy < 75% of the uniform {self.max_entropy:.2f} nats) · "
                + ("causal mask" if self.causal else "full attention")
            )
        return (
            f"{self.n_heads} head(s) over {self.n_tokens} token(s) · every head "
            f"attends near-uniformly (entropy ≈ {self.max_entropy:.2f} nats) · "
            + ("causal mask" if self.causal else "full attention")
        )


def self_attention_report(
    matrix: np.ndarray,
    *,
    n_heads: int = 4,
    causal: bool = True,
    seed: int = RANDOM_STATE,
) -> AttentionReport:
    """Run one sequence through multi-head self-attention and read it.

    Standardises the matrix first (per-feature z-scores -- attention
    scores blow up on raw magnitudes, the same conditioning every
    chapter's model applies), runs :func:`multi_head_attention`, and
    reduces the weights to the report: entropies per head and each
    position's focus.

    Raises
    ------
    DataValidationError
        Via :func:`multi_head_attention` (bad shapes, ``d`` not
        divisible by the head count, NaN input).
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise DataValidationError(f"matrix must be 2-D (n, d), got shape {values.shape}")
    std = values.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    scaled = (values - values.mean(axis=0)) / std
    _output, weights = multi_head_attention(scaled, n_heads=n_heads, seed=seed, causal=causal)
    n_tokens = weights.shape[1]
    entropies = tuple(
        float(-np.sum(head * np.log(np.clip(head, 1e-12, None)), axis=-1).mean())
        for head in weights
    )
    focus = tuple(tuple(int(row.argmax()) for row in head) for head in weights)
    return AttentionReport(
        weights=weights,
        entropies=entropies,
        focus=focus,
        n_heads=n_heads,
        n_tokens=n_tokens,
        causal=causal,
        max_entropy=math.log(n_tokens),
    )


__all__ = [
    "RANDOM_STATE",
    "AttentionReport",
    "PositionalReport",
    "dot_product_attention",
    "multi_head_attention",
    "self_attention_report",
    "sinusoidal_positions",
]
