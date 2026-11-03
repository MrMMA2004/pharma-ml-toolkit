"""The offline run-trace board -- TensorBoard, absorbed.

Round 186 checked tensorflow/tensorboard file by file and lifted what
survives without a TensorFlow graph or a web server:

* **The summary writer protocol** -- ``add_scalar`` / ``add_histogram``
  / ``add_text`` / ``add_embedding`` / hparams trials, one tag at a
  time, event records kept per tag (the ``summary_v2`` API reduced to
  its data model).
* **The event accumulator's reservoir** -- Vitter reservoir sampling
  with TensorBoard's own twist (``always_keep_last``: the newest event
  is always the bucket's last entry, replaced in place when the draw
  misses), ported faithfully from
  ``tensorboard/backend/event_processing/reservoir.py`` including the
  ``randint(0, num_seen)`` draw and per-tag independent determinism.
* **The scalars dashboard's smoothing** -- the exponential moving
  average with leading bias correction (the honest statement of the
  slider's formula; TensorBoard's JS variant additionally collapses
  consecutive equal values, a nicety that changes nothing at float
  precision and is documented as skipped).
* **The histogram dashboard's two views** -- per-step summary stats
  (min/max/mean/std/count -- the overlay mode's curves) and fixed-width
  bucket counts (the offset mode's ridgeline data).
* **The hparams dashboard** -- trial rows of ``{hyperparameters}`` ×
  ``{metrics}``, plus a session-group count by identical params (the
  grouping the plugin calls session groups).
* **The embedding projector's nearest-neighbour lookup** -- cosine
  similarity against a labelled matrix, the "click a point, meet its
  neighbours" interaction without a browser canvas.  The projection
  itself (PCA/t-SNE/UMAP) is the toolkit's own manifold machinery.

The honest scope, stated once: TensorBoard's served dashboards, event
*files* on disk, the TensorFlow graph viewer, the profile/pr-curve/text
plugins and the image sprite grids are what this toolkit does not
vendor.  What transfers is the event model those dashboards consume and
the two algorithms that make a million-event run browsable in bounded
memory: the reservoir and the smoother.

Examples
--------
>>> from pharmaml.ml.runboard import RunTrace, smooth_series
>>> trace = RunTrace("fit-01")
>>> for epoch in range(3):
...     trace.add_scalar("loss/train", 1.0 - epoch * 0.3, step=epoch)
...     trace.add_scalar("loss/val", 1.1 - epoch * 0.25, step=epoch)
>>> trace.add_histogram("weights/layer0", [0.1, -0.2, 0.3], step=0)
>>> trace.add_hparams({"lr": 0.01, "layers": 2}, {"val_loss": 0.35})
>>> sorted(trace.tags("scalar"))
['loss/train', 'loss/val']
>>> [round(v, 4) for v in smooth_series([1.0, 0.5, 0.2], beta=0.5)]
[1.0, 0.6667, 0.4]
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

#: Tag kinds a trace records (the plugin family, data-side).
TRACE_KINDS = ("scalar", "histogram", "text", "embedding")

#: Histogram bucket count per step (TensorBoard uses 30 fixed-width
#: buckets when building proto histograms from raw values).
HISTOGRAM_BUCKETS = 30

#: The reservoir's replacement draw uses this many seen-items -- see
#: :class:`ReservoirBucket`; TensorBoard's own default size for loaded
#: tags is 1000 (``event_accumulator.DEFAULT_SIZE``).
DEFAULT_RESERVOIR_SIZE = 1000


# --------------------------------------------------------------------------- #
# Event records -- the summary_v2 payload types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScalarEvent:
    """One ``add_scalar(tag, value, step=...)`` record."""

    step: int
    value: float


@dataclass(frozen=True)
class HistogramEvent:
    """One ``add_histogram`` record: stats + fixed-width buckets.

    TensorBoard's histogram proto carries the summary stats AND the
    bucket counts; the offset view plots the buckets, the overlay view
    plots the stats.  Both survive here as plain data.
    """

    step: int
    count: int
    min: float
    max: float
    mean: float
    std: float
    bucket_edges: tuple[float, ...]
    bucket_counts: tuple[int, ...]


@dataclass(frozen=True)
class TextEvent:
    """One ``add_text(tag, text, step=...)`` record (the text plugin)."""

    step: int
    text: str


@dataclass(frozen=True)
class EmbeddingEvent:
    """One ``add_embedding(tag, matrix, labels)`` record.

    The projector plugin's payload: an (n, d) matrix plus one label per
    row.  The projection is computed on demand, not stored.
    """

    matrix: tuple[tuple[float, ...], ...]
    labels: tuple[str | None, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=float)

    @property
    def n_rows(self) -> int:
        return len(self.matrix)


@dataclass(frozen=True)
class HparamTrial:
    """One hparams trial row: ``{params}`` x ``{metrics}`` (the plugin's
    session-group row, collapsed to a single trial per group)."""

    params: dict[str, Any]
    metrics: dict[str, float]

    def session_key(self) -> str:
        """The plugin's session-group key: the sorted param tuple."""
        return json.dumps(self.params, sort_keys=True, default=str)


def histogram_event(values: Any, step: int) -> HistogramEvent:
    """Build one :class:`HistogramEvent` from raw values.

    The reduction mirrors TensorBoard's proto builder: min/max/mean/std
    plus ``HISTOGRAM_BUCKETS`` fixed-width buckets spanning
    ``[min, max]``.  Empty input is refused (an empty histogram has no
    honest summary).
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise DataValidationError(
            "histogram events need at least one value "
            f"(step {step} got {array.size})"
        )
    lo, hi = float(array.min()), float(array.max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise DataValidationError(
            f"histogram values must be finite at step {step} "
            f"(min {lo}, max {hi})"
        )
    if hi <= lo:
        # Degenerate span (all-equal values): one bucket holds everything.
        edges: tuple[float, ...] = (lo - 0.5, hi + 0.5)
        counts: tuple[int, ...] = (int(array.size),)
    else:
        edges = tuple(
            lo + (hi - lo) * i / HISTOGRAM_BUCKETS
            for i in range(HISTOGRAM_BUCKETS + 1)
        )
        idx = np.clip(
            ((array - lo) / (hi - lo) * HISTOGRAM_BUCKETS).astype(int),
            0,
            HISTOGRAM_BUCKETS - 1,
        )
        counts = tuple(int(c) for c in np.bincount(idx, minlength=HISTOGRAM_BUCKETS))
    return HistogramEvent(
        step=int(step),
        count=int(array.size),
        min=lo,
        max=hi,
        mean=float(array.mean()),
        std=float(array.std()),
        bucket_edges=tuple(float(e) for e in edges),
        bucket_counts=counts,
    )


# --------------------------------------------------------------------------- #
# The reservoir -- tensorboard/backend/event_processing/reservoir.py, ported
# --------------------------------------------------------------------------- #
class ReservoirBucket:
    """Vitter reservoir sampling with TensorBoard's ``always_keep_last``.

    Determinism contract (TB's own docstring, kept verbatim in spirit):
    the kept sequence for a tag depends only on (size, seed, that tag's
    item sequence) -- never on other tags' insertions.
    """

    def __init__(
        self, max_size: int, rng: random.Random, always_keep_last: bool = True
    ) -> None:
        if max_size < 0:
            raise DataValidationError(
                f"reservoir size must be >= 0, got {max_size}"
            )
        self.items: list[Any] = []
        self._max_size = int(max_size)
        self._num_items_seen = 0
        self._rng = rng
        self.always_keep_last = always_keep_last

    def add(self, item: Any) -> None:
        """Add one item; replace via the ``randint(0, num_seen)`` draw.

        TB's exact replacement rule: if the draw lands below the bucket
        size, that index is popped and the new item appended; otherwise
        (with ``always_keep_last``) the LAST item is replaced in place --
        the newest event is always the bucket's newest entry.
        """
        if self._max_size == 0 or len(self.items) < self._max_size:
            self.items.append(item)
        else:
            draw = self._rng.randint(0, self._num_items_seen)
            if draw < self._max_size:
                self.items.pop(draw)
                self.items.append(item)
            elif self.always_keep_last:
                self.items[-1] = item
        self._num_items_seen += 1

    @property
    def num_seen(self) -> int:
        return self._num_items_seen


def reservoir_sample(
    items: Sequence[Any], size: int, seed: int = 0
) -> list[Any]:
    """Reservoir-sample ``items`` down to ``size`` entries.

    The one-shot convenience over :class:`ReservoirBucket` (size 0 keeps
    everything -- TB's "unbounded" mode).

    >>> reservoir_sample(list(range(10)), 4, seed=42)   # doctest: +ELLIPSIS
    [...]
    >>> len(reservoir_sample(list(range(10)), 4, seed=42))
    4
    """
    bucket = ReservoirBucket(int(size), random.Random(int(seed)))
    for item in items:
        bucket.add(item)
    return list(bucket.items)


# --------------------------------------------------------------------------- #
# Smoothing -- the scalars dashboard's EMA slider
# --------------------------------------------------------------------------- #
def smooth_series(values: Sequence[float], beta: float = 0.6) -> list[float]:
    """Exponentially weighted smoothing with leading bias correction.

    ``acc_t = beta * acc_{t-1} + (1 - beta) * y_t`` rescaled by
    ``(1 - beta ** (t + 1))`` -- the standard bias-corrected EMA
    TensorBoard's slider computes (its JS additionally merges runs of
    equal consecutive values; at float precision that refinement only
    matters for exact repeats, and is honestly skipped here).

    >>> smooth_series([1.0, 1.0, 1.0], beta=0.0)
    [1.0, 1.0, 1.0]
    >>> smooth_series([0.0, 1.0], beta=0.9)[0] < 0.6   # bias-corrected head
    True
    """
    if not 0.0 <= beta < 1.0:
        raise DataValidationError(f"beta must be in [0, 1), got {beta}")
    out: list[float] = []
    acc = 0.0
    weight_sum = 0.0
    for index, raw in enumerate(values):
        value = float(raw)
        acc = beta * acc + (1.0 - beta) * value
        weight_sum = beta * weight_sum + (1.0 - beta)
        divisor = 1.0 - beta ** (index + 1)
        out.append(acc / divisor if divisor > 0 else value)
    return out


# --------------------------------------------------------------------------- #
# RunTrace -- the summary writer + event accumulator, in memory
# --------------------------------------------------------------------------- #
class RunTrace:
    """One run's event log: tag -> event list, per kind.

    ``add_*`` mirrors the writer API (``step`` auto-increments per tag
    when omitted -- the accumulator's step numbering); query methods
    mirror the accumulator (``tags`` / ``scalars`` / ...).  Every event
    list passes through a :class:`ReservoirBucket` at
    ``reservoir_size`` so a million-epoch run stays bounded, exactly the
    property TB's ``event_accumulator`` buys with its own reservoir.
    """

    def __init__(
        self,
        name: str = "run",
        *,
        reservoir_size: int = DEFAULT_RESERVOIR_SIZE,
        seed: int = 0,
    ) -> None:
        if not name:
            raise DataValidationError("run name must be non-empty")
        if reservoir_size < 0:
            raise DataValidationError(
                f"reservoir_size must be >= 0, got {reservoir_size}"
            )
        self.name = str(name)
        self._reservoir_size = int(reservoir_size)
        self._seed = int(seed)
        self._scalars: dict[str, ReservoirBucket] = {}
        self._histograms: dict[str, ReservoirBucket] = {}
        self._texts: dict[str, ReservoirBucket] = {}
        self._embeddings: dict[str, EmbeddingEvent] = {}
        self._hparams: list[HparamTrial] = []
        self._steps: dict[tuple[str, str], int] = {}

    # -- writer side ----------------------------------------------------- #
    def _bucket(self, store: dict[str, ReservoirBucket], tag: str) -> ReservoirBucket:
        if tag not in store:
            store[tag] = ReservoirBucket(
                self._reservoir_size, random.Random(self._seed)
            )
        return store[tag]

    def _next_step(self, kind: str, tag: str, step: int | None) -> int:
        if step is not None:
            self._steps[(kind, tag)] = int(step)
            return int(step)
        current = self._steps.get((kind, tag), -1)
        self._steps[(kind, tag)] = current + 1
        return current + 1

    def add_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Record one scalar (``add_scalar``).  ``step`` auto-increments."""
        number = float(value)
        if not math.isfinite(number):
            raise DataValidationError(f"scalar {tag!r} must be finite, got {value}")
        self._bucket(self._scalars, tag).add(
            ScalarEvent(step=self._next_step("scalar", tag, step), value=number)
        )

    def add_histogram(
        self, tag: str, values: Any, step: int | None = None
    ) -> None:
        """Record one histogram over raw values (``add_histogram``)."""
        event = histogram_event(values, self._next_step("histogram", tag, step))
        self._bucket(self._histograms, tag).add(event)

    def add_text(self, tag: str, text: str, step: int | None = None) -> None:
        """Record one text entry (``add_text``)."""
        self._bucket(self._texts, tag).add(
            TextEvent(step=self._next_step("text", tag, step), text=str(text))
        )

    def add_embedding(
        self,
        tag: str,
        matrix: Any,
        labels: Sequence[str | None] | None = None,
    ) -> None:
        """Record one embedding payload (``add_embedding``).

        The newest record replaces the tag's previous one -- the
        projector plugin's checkpoint semantics.
        """
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise DataValidationError(
                f"embedding {tag!r} needs a non-empty 2-D matrix, got shape {array.shape}"
            )
        if labels is None:
            row_labels: list[str | None] = [None] * array.shape[0]
        else:
            row_labels = [str(lbl) if lbl is not None else None for lbl in labels]
            if len(row_labels) != array.shape[0]:
                raise DataValidationError(
                    f"embedding {tag!r}: {len(row_labels)} labels for "
                    f"{array.shape[0]} rows"
                )
        self._embeddings[tag] = EmbeddingEvent(
            matrix=tuple(tuple(float(v) for v in row) for row in array),
            labels=tuple(row_labels),
        )

    def add_hparams(
        self, hparams: dict[str, Any], metrics: dict[str, float]
    ) -> None:
        """Record one hparams trial row (``hparams_hook`` + metrics)."""
        self._hparams.append(
            HparamTrial(
                params=dict(hparams),
                metrics={str(k): float(v) for k, v in metrics.items()},
            )
        )

    # -- accumulator side ------------------------------------------------ #
    def tags(self, kind: str) -> list[str]:
        """The tags recorded for ``kind`` (sorted)."""
        if kind not in TRACE_KINDS:
            raise DataValidationError(
                f"kind must be one of {TRACE_KINDS}, got {kind!r}"
            )
        if kind == "scalar":
            return sorted(self._scalars)
        if kind == "histogram":
            return sorted(self._histograms)
        if kind == "text":
            return sorted(self._texts)
        return sorted(self._embeddings)

    def scalars(self, tag: str) -> list[ScalarEvent]:
        if tag not in self._scalars:
            raise DataValidationError(f"no scalar tag {tag!r} in run {self.name!r}")
        return [e for e in self._scalars[tag].items if isinstance(e, ScalarEvent)]

    def scalar_values(self, tag: str) -> list[float]:
        return [event.value for event in self.scalars(tag)]

    def histograms(self, tag: str) -> list[HistogramEvent]:
        if tag not in self._histograms:
            raise DataValidationError(
                f"no histogram tag {tag!r} in run {self.name!r}"
            )
        return [
            e
            for e in self._histograms[tag].items
            if isinstance(e, HistogramEvent)
        ]

    def texts(self, tag: str) -> list[TextEvent]:
        if tag not in self._texts:
            raise DataValidationError(f"no text tag {tag!r} in run {self.name!r}")
        return [e for e in self._texts[tag].items if isinstance(e, TextEvent)]

    def embedding(self, tag: str) -> EmbeddingEvent:
        if tag not in self._embeddings:
            raise DataValidationError(
                f"no embedding tag {tag!r} in run {self.name!r}"
            )
        return self._embeddings[tag]

    def hparams(self) -> list[HparamTrial]:
        return list(self._hparams)

    def session_groups(self) -> dict[str, int]:
        """The plugin's grouping: trials counted by identical params."""
        groups: dict[str, int] = {}
        for trial in self._hparams:
            groups[trial.session_key()] = groups.get(trial.session_key(), 0) + 1
        return groups

    def scalar_frame(self, tags: Sequence[str] | None = None) -> pd.DataFrame:
        """A DataFrame indexed by step, one column per tag."""
        chosen = list(tags) if tags is not None else self.tags("scalar")
        if not chosen:
            return pd.DataFrame()
        series: dict[str, dict[int, float]] = {}
        for tag in chosen:
            series[tag] = {event.step: event.value for event in self.scalars(tag)}
        return pd.DataFrame(series).sort_index().rename_axis("step")

    def histogram_stats_frame(self, tag: str) -> pd.DataFrame:
        """The overlay view's table: one row per histogram step."""
        events = self.histograms(tag)
        if not events:
            raise DataValidationError(f"histogram tag {tag!r} has no events")
        return pd.DataFrame(
            [
                {
                    "step": event.step,
                    "count": event.count,
                    "min": event.min,
                    "max": event.max,
                    "mean": event.mean,
                    "std": event.std,
                }
                for event in events
            ]
        ).set_index("step").sort_index()

    def summary(self) -> str:
        """One honest line, the ``event_file_inspector`` reduced."""
        return (
            f"run {self.name!r}: "
            f"{len(self._scalars)} scalar tags, "
            f"{len(self._histograms)} histogram tags, "
            f"{len(self._texts)} text tags, "
            f"{len(self._embeddings)} embeddings, "
            f"{len(self._hparams)} hparam trials"
        )

    # -- serialisation --------------------------------------------------- #
    def to_json(self) -> str:
        """Deterministic JSON (sorted keys, no timestamps)."""
        payload = {
            "name": self.name,
            "reservoir_size": self._reservoir_size,
            "scalars": {
                tag: [
                    {"step": e.step, "value": e.value}
                    for e in (
                        item
                        for item in bucket.items
                        if isinstance(item, ScalarEvent)
                    )
                ]
                for tag, bucket in self._scalars.items()
            },
            "histograms": {
                tag: [
                    {
                        "step": e.step,
                        "count": e.count,
                        "min": e.min,
                        "max": e.max,
                        "mean": e.mean,
                        "std": e.std,
                        "bucket_edges": list(e.bucket_edges),
                        "bucket_counts": list(e.bucket_counts),
                    }
                    for e in (
                        item
                        for item in bucket.items
                        if isinstance(item, HistogramEvent)
                    )
                ]
                for tag, bucket in self._histograms.items()
            },
            "texts": {
                tag: [
                    {"step": e.step, "text": e.text}
                    for e in (
                        item for item in bucket.items if isinstance(item, TextEvent)
                    )
                ]
                for tag, bucket in self._texts.items()
            },
            "embeddings": {
                tag: {
                    "matrix": [list(row) for row in event.matrix],
                    "labels": list(event.labels),
                }
                for tag, event in self._embeddings.items()
            },
            "hparams": [
                {"params": t.params, "metrics": t.metrics}
                for t in self._hparams
            ],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> RunTrace:
        data = json.loads(text)
        trace = cls(str(data["name"]), reservoir_size=int(data["reservoir_size"]))
        for tag, events in data["scalars"].items():
            for event in events:
                trace.add_scalar(tag, event["value"], step=event["step"])
        for tag, events in data["histograms"].items():
            for event in events:
                trace._bucket(trace._histograms, tag).add(
                    HistogramEvent(
                        step=int(event["step"]),
                        count=int(event["count"]),
                        min=float(event["min"]),
                        max=float(event["max"]),
                        mean=float(event["mean"]),
                        std=float(event["std"]),
                        bucket_edges=tuple(float(x) for x in event["bucket_edges"]),
                        bucket_counts=tuple(int(x) for x in event["bucket_counts"]),
                    )
                )
        for tag, events in data["texts"].items():
            for event in events:
                trace.add_text(tag, event["text"], step=event["step"])
        for tag, payload in data["embeddings"].items():
            trace.add_embedding(
                tag, payload["matrix"], labels=payload["labels"]
            )
        for trial in data["hparams"]:
            trace.add_hparams(trial["params"], trial["metrics"])
        return trace


# --------------------------------------------------------------------------- #
# RunBoard -- the multiplexer: many runs, compared
# --------------------------------------------------------------------------- #
class RunBoard:
    """The event multiplexer's data side: several runs, one comparison.

    >>> board = RunBoard()
    >>> board.add_run(RunTrace("a"))
    >>> board.add_run(RunTrace("b"))
    >>> board.run_names()
    ['a', 'b']
    """

    def __init__(self, runs: Iterable[RunTrace] = ()) -> None:
        self._runs: dict[str, RunTrace] = {}
        for run in runs:
            self.add_run(run)

    def add_run(self, trace: RunTrace) -> None:
        if trace.name in self._runs:
            raise DataValidationError(
                f"run {trace.name!r} already on the board"
            )
        self._runs[trace.name] = trace

    def run_names(self) -> list[str]:
        return sorted(self._runs)

    def run(self, name: str) -> RunTrace:
        if name not in self._runs:
            raise DataValidationError(f"no run {name!r} on the board")
        return self._runs[name]

    def scalar_frame(self, tag: str) -> pd.DataFrame:
        """One column per run, aligned on the union of steps."""
        columns: dict[str, dict[int, float]] = {}
        for name in self.run_names():
            trace = self._runs[name]
            if tag in trace._scalars:
                columns[name] = {
                    event.step: event.value for event in trace.scalars(tag)
                }
        if not columns:
            raise DataValidationError(f"no run recorded scalar tag {tag!r}")
        return (
            pd.DataFrame(columns)
            .sort_index()
            .rename_axis("step")
        )

    def compare_table(self, tag: str) -> pd.DataFrame:
        """The compare-runs strip: last / best / mean per run."""
        frame = self.scalar_frame(tag)
        if frame.empty:  # pragma: no cover - guarded by scalar_frame
            return frame
        best_min = tag.startswith(("loss", "error", "mae", "rmse"))
        return pd.DataFrame(
            {
                "last": frame.iloc[-1],
                "best": frame.min() if best_min else frame.max(),
                "mean": frame.mean(),
                "n": frame.count(),
            }
        )


# --------------------------------------------------------------------------- #
# The projector's neighbour lookup (no browser needed)
# --------------------------------------------------------------------------- #
def nearest_neighbours(
    matrix: Any,
    row: int,
    *,
    k: int = 5,
    labels: Sequence[str | None] | None = None,
) -> list[tuple[int, float, str | None]]:
    """The projector's click interaction: one row's nearest neighbours.

    Cosine similarity over the matrix (the projector's default metric),
    returning ``(index, similarity, label)`` tuples sorted by descending
    similarity, the query row first, ``k`` neighbours after it.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        raise DataValidationError(
            f"nearest_neighbours needs a non-empty 2-D matrix, got {array.shape}"
        )
    row = int(row)
    if not 0 <= row < array.shape[0]:
        raise DataValidationError(
            f"row index {row} outside [0, {array.shape[0]})"
        )
    if k < 0:
        raise DataValidationError(f"k must be >= 0, got {k}")
    query = array[row].reshape(1, -1)
    norms = np.linalg.norm(array, axis=1) * np.linalg.norm(query, axis=1)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = (array @ query.reshape(-1)) / np.where(norms > 0, norms, np.nan)
    sims = np.nan_to_num(sims, nan=0.0)
    order = sorted(range(array.shape[0]), key=lambda i: (-sims[i], i))
    picked = [i for i in order if i != row][: int(k)]
    row_labels = list(labels) if labels is not None else None
    out: list[tuple[int, float, str | None]] = []
    for index in [row] + picked:
        label = row_labels[index] if row_labels is not None else None
        out.append((index, float(sims[index]), label))
    return out
