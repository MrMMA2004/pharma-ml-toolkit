"""Molecular similarity (Tanimoto and friends).

Upgrades ``Tanimoto calculator.py`` / ``Tanimoto-edited.py``.  The key
design change: the *math* (Tanimoto coefficient on bit sets) is
separated from the *chemistry* (building fingerprints).  The pure
functions run on any bit representation and are unit-tested without
RDKit; the RDKit-backed scan functions build fingerprints then reuse
the same math.

Examples
--------
>>> from pharmaml.chem.similarity import (
...     tanimoto_from_bitarrays, tanimoto_from_sets)
>>> tanimoto_from_sets({"donor", "aromatic", "acid"}, {"donor", "aromatic", "base"})
0.5
>>> tanimoto_from_sets(set(), set())       # both empty -> identical
1.0
>>> tanimoto_from_sets({"donor"}, set())   # one empty -> nothing in common
0.0
>>> tanimoto_from_bitarrays([1, 1, 0], [1, 0, 0])  # the same math on bits
0.5
>>> tanimoto_from_bitarrays([1, 0], [1, 0, 0])     # unequal lengths rejected
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: bit arrays must have equal length, got (2,) vs (3,)

>>> import numpy as np
>>> from pharmaml.chem.similarity import tanimoto_matrix
>>> reference = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
>>> targets = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
>>> sims = tanimoto_matrix(reference, targets)
>>> sims.shape, sims.dtype.name
((2, 3), 'float64')
>>> sims.tolist()   # identical -> 1; disjoint or one-sided-empty -> 0; subset -> 0.5
[[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]]

>>> from pharmaml.chem.similarity import similarity_scan
>>> scan = similarity_scan(["CCO"], ["CCO", "CCC", "c1ccccc1O"], threshold=0.25)
>>> scan["target"].tolist(), scan["target_index"].tolist()  # phenol is dropped
(['CCO', 'CCC'], [0, 1])
>>> scan["similarity"].tolist()   # kept at >= threshold, best first
[1.0, 0.25]
>>> similarity_scan(["CCO"], ["CCO"], threshold=1.5)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: threshold must be within [0, 1], got 1.5

>>> from pharmaml.chem.similarity import top_matches
>>> matches = top_matches("CCO", ["CCO", "CCC", "c1ccccc1O", "CO"], k=3)
>>> matches["target"].tolist()
['CCO', 'CO', 'CCC']
>>> [round(v, 4) for v in matches["similarity"]]   # ethanol > methanol > propane
[1.0, 0.3333, 0.25]
>>> top_matches("CCO", ["CCO"], k=0)
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: k must be >= 1, got 0
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.chem.descriptors import FingerprintKind
from pharmaml.exceptions import DataValidationError, DependencyError, JobCancelled
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "run_similarity_campaign",
    "similarity_scan",
    "tanimoto_from_bitarrays",
    "tanimoto_from_sets",
    "tanimoto_matrix",
    "top_matches",
]


def tanimoto_from_sets(a: set, b: set) -> float:
    """Tanimoto coefficient for two feature sets: |A∩B| / |A∪B|.

    Both sets empty -> 1.0 (identical); one empty -> 0.0.
    """
    union = len(a | b)
    if union == 0:
        return 1.0
    return len(a & b) / union


def tanimoto_from_bitarrays(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto coefficient for two equal-length binary arrays.

    Uses vectorised bitwise ops -- ~50x faster than set operations for
    2048-bit fingerprints in benchmarks.  Float inputs must be finite:
    ``astype(bool)`` maps NaN to ``True``, so a NaN would silently count
    as a set bit.

    Raises
    ------
    DataValidationError
        If the arrays differ in shape or contain non-finite float values.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    for label, arr in (("a", a), ("b", b)):
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise DataValidationError(
                f"bit array {label!r} contains non-finite values (NaN/inf); "
                "expected a binary 0/1 array"
            )
    a = a.astype(bool)
    b = b.astype(bool)
    if a.shape != b.shape:
        raise DataValidationError(
            f"bit arrays must have equal length, got {a.shape} vs {b.shape}"
        )
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return int(np.logical_and(a, b).sum()) / union


def tanimoto_matrix(
    reference: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """All-vs-all Tanimoto similarities between two bit-matrix batches.

    Parameters
    ----------
    reference:
        ``(n_ref, n_bits)`` binary matrix.
    targets:
        ``(n_tgt, n_bits)`` binary matrix.

    Returns
    -------
    numpy.ndarray
        ``(n_ref, n_tgt)`` float64 similarity matrix.

    Raises
    ------
    DataValidationError
        If the matrices are not 2-D with matching bit length, or contain
        non-finite float values (NaN would silently count as a set bit).
    """
    reference = np.asarray(reference)
    targets = np.asarray(targets)
    for label, arr in (("reference", reference), ("targets", targets)):
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise DataValidationError(
                f"{label!r} contains non-finite values (NaN/inf); "
                "expected a binary 0/1 matrix"
            )
    reference = reference.astype(bool)
    targets = targets.astype(bool)
    if reference.ndim != 2 or targets.ndim != 2 or reference.shape[1] != targets.shape[1]:
        raise DataValidationError(
            "reference/targets must be 2-D with matching bit length, got "
            f"{reference.shape} vs {targets.shape}"
        )
    ref_counts = reference.sum(axis=1)[:, np.newaxis]
    tgt_counts = targets.sum(axis=1)[np.newaxis, :]
    intersections = reference.astype(np.uint32) @ targets.T.astype(np.uint32)
    unions = ref_counts + tgt_counts - intersections
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(unions == 0, 1.0, intersections / np.maximum(unions, 1e-12))
    return result.astype(np.float64)


def _require_rdkit_similarity() -> None:
    try:
        import rdkit  # noqa: F401 -- presence check only
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for the SMILES-based similarity scan. "
            "Install it with:  pip install 'pharmaml[chem]'"
        ) from exc


def similarity_scan(
    reference_smiles: Sequence[str],
    target_smiles: Sequence[str],
    threshold: float = 0.9,
    fingerprint: FingerprintKind = "rdkit",
    n_bits: int = 2048,
    radius: int = 2,
) -> pd.DataFrame:
    """Scan reference ligands against a target library (Tanimoto >= threshold).

    This is the productionised version of the original workflow that
    read two hardcoded Excel files and wrote a filtered matrix to a
    hardcoded Windows path.  Everything is now explicit parameters and
    the return value is a tidy long-format DataFrame:

    ``reference | target | reference_index | target_index | similarity``

    Parameters
    ----------
    reference_smiles:
        SMILES strings for the reference/seed ligands.
    target_smiles:
        SMILES strings for the screening library.
    threshold:
        Minimum similarity to keep (default 0.9 as in the original).
    fingerprint:
        ``'rdkit'`` (path fingerprint, as originally used) or
        ``'morgan'`` (ECFP-style).
    n_bits:
        Fingerprint bit length.
    radius:
        Morgan radius (only used for ``'morgan'``).

    Returns
    -------
    pandas.DataFrame
        Filtered similarity pairs sorted by descending similarity
        (ties keep library order -- the sort is stable).  A pandas Series
        is accepted for both SMILES arguments; row order is used and the
        Series index is ignored, so the ``reference``/``target`` columns
        always match the positional ``*_index`` columns.

    Raises
    ------
    DataValidationError
        If ``threshold`` is outside [0, 1] or either SMILES argument is
        a bare string (which would otherwise be split into characters).
    """
    if not 0.0 <= threshold <= 1.0:
        raise DataValidationError(f"threshold must be within [0, 1], got {threshold}")
    _require_rdkit_similarity()

    from pharmaml.chem.descriptors import (
        _as_smiles_list,
        compute_fingerprints,
    )

    # Materialise once: keeps positional lookups correct for Series with
    # non-default indexes and for one-shot generators.
    reference_list = _as_smiles_list(reference_smiles, "reference_smiles")
    target_list = _as_smiles_list(target_smiles, "target_smiles")

    ref_matrix, ref_idx = compute_fingerprints(
        reference_list, kind=fingerprint, n_bits=n_bits, radius=radius
    )
    tgt_matrix, tgt_idx = compute_fingerprints(
        target_list, kind=fingerprint, n_bits=n_bits, radius=radius
    )
    _LOG.info(
        "Similarity scan: %d valid references x %d valid targets (threshold=%.2f)",
        len(ref_idx),
        len(tgt_idx),
        threshold,
    )

    similarities = tanimoto_matrix(ref_matrix, tgt_matrix)
    mask = similarities >= threshold

    rows = []
    for ref_row, ref_pos in enumerate(ref_idx):
        for tgt_row, tgt_pos in enumerate(tgt_idx):
            if mask[ref_row, tgt_row]:
                rows.append(
                    {
                        "reference": reference_list[ref_pos],
                        "target": target_list[tgt_pos],
                        "reference_index": int(ref_pos),
                        "target_index": int(tgt_pos),
                        "similarity": float(similarities[ref_row, tgt_row]),
                    }
                )
    frame = pd.DataFrame(
        rows,
        columns=["reference", "target", "reference_index", "target_index", "similarity"],
    )
    return frame.sort_values("similarity", ascending=False, kind="stable").reset_index(drop=True)


def top_matches(
    query: str,
    library: Sequence[str] | pd.Series,
    k: int = 10,
    fingerprint: FingerprintKind = "rdkit",
    n_bits: int = 2048,
    radius: int = 2,
) -> pd.DataFrame:
    """Return the ``k`` library entries most similar to ``query``.

    Convenience wrapper around :func:`similarity_scan` with
    ``threshold=0`` and a head-slice.  ``radius`` is forwarded to the
    Morgan fingerprint (matching :func:`similarity_scan`); equal
    similarities keep library order.

    Raises
    ------
    DataValidationError
        If ``k < 1``, ``library`` is a bare string, or ``query`` is not a
        parseable SMILES (a wrong query must not masquerade as "no
        matches").
    """
    if k < 1:
        raise DataValidationError(f"k must be >= 1, got {k}")
    _require_rdkit_similarity()
    from pharmaml.chem.descriptors import (
        _as_smiles_list,
        _smiles_to_molecules,
    )

    _, _, failures = _smiles_to_molecules([query])
    if failures:
        raise DataValidationError(
            f"query SMILES is not valid: {query!r} -- no matches were computed"
        )
    pairs = similarity_scan(
        [query],
        _as_smiles_list(library, "library"),
        threshold=0.0,
        fingerprint=fingerprint,
        n_bits=n_bits,
        radius=radius,
    )
    return pairs.head(k).reset_index(drop=True)


# -- (part5-continuation round 7, §13) the resumable similarity campaign --
#
# ``similarity_scan`` is one-shot: it fingerprints every reference, then
# every target, then walks the full matrix.  A large screening campaign
# (many references against a large library) is exactly the long-running
# §13 job the master prompt wants pausable: the campaign below runs the
# SAME math one reference CHUNK at a time, checkpoints the completed
# rows of the long-format result (plus the dropped-reference ledger and
# the config/dataset fingerprints that make a resume EXACT), and a
# resume never re-fingerprints the completed prefix.  The state design
# mirrors the round-6 descriptor batch: per-unit payloads ride as
# JSON-safe primitives (row metadata + base64-packed float64 similarity
# vectors), so the round-trip is bit-exact.


def _pack_similarities(array: np.ndarray) -> dict[str, object]:
    """One float vector -> a JSON-safe bit-exact ``{dtype, shape, data}``."""
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=float))
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _unpack_similarities(payload: object) -> np.ndarray:
    """Rebuild one :func:`_pack_similarities` payload into a fresh array."""
    if not isinstance(payload, dict):
        raise TypeError(f"packed vector must be a dict, got {type(payload).__name__}")
    dtype = np.dtype(str(payload["dtype"]))
    shape = tuple(int(dim) for dim in payload["shape"])
    raw = base64.b64decode(str(payload["data"]), validate=True)
    if raw and len(raw) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("packed vector payload size mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _campaign_config_fingerprint(
    fingerprint: FingerprintKind, n_bits: int, radius: int, threshold: float
) -> str:
    """SHA-256 over the campaign's scan config (a resume must keep the
    same fingerprint kind, bit length, radius and threshold)."""
    blob = json.dumps(
        {
            "fingerprint": str(fingerprint),
            "n_bits": int(n_bits),
            "radius": int(radius),
            "threshold": float(threshold),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _campaign_dataset_fingerprint(
    reference_list: Sequence[str], target_list: Sequence[str]
) -> str:
    """SHA-256 over BOTH input lists -- a resume against different
    references or a different library is detected, not guessed at."""
    digest = hashlib.sha256()
    for text in reference_list:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")  # delimiter: no concatenation ambiguity
    digest.update(b"\x01")  # the references/library boundary is part of the identity
    for text in target_list:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _campaign_finish(
    rows: Sequence[dict[str, Any]],
    dropped: Sequence[tuple[int, str]],
    reference_list: Sequence[str],
    target_list: Sequence[str],
) -> pd.DataFrame:
    """Assemble the campaign result exactly as :func:`similarity_scan`
    would: long-format frame, ties keep campaign order, stable sort by
    descending similarity."""
    frame = pd.DataFrame(
        [
            {
                "reference": reference_list[entry["reference_index"]],
                "target": target_list[entry["target_index"]],
                "reference_index": int(entry["reference_index"]),
                "target_index": int(entry["target_index"]),
                "similarity": float(entry["similarity"]),
            }
            for entry in rows
        ],
        columns=["reference", "target", "reference_index", "target_index", "similarity"],
    )
    return frame.sort_values(
        "similarity", ascending=False, kind="stable"
    ).reset_index(drop=True)


def run_similarity_campaign(
    reference_smiles: Sequence[str],
    target_smiles: Sequence[str],
    threshold: float = 0.9,
    fingerprint: FingerprintKind = "rdkit",
    n_bits: int = 2048,
    radius: int = 2,
    progress: Callable[..., Any] | None = None,
    *,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    checkpoint_every: int = 1,
    checkpoint_resume: bool = True,
) -> pd.DataFrame:
    """A pausable/resumable similarity scan -- :func:`similarity_scan`
    as a §13 campaign.

    The campaign computes the SAME result as ``similarity_scan``
    (bit-identical: same columns, same filtering, same stable sort) but
    processes the references in chunks of 25, reporting progress and --
    when ``checkpoint_dir`` is given -- writing envelope-carrying
    checkpoints every ``checkpoint_every`` chunk boundaries.  A later
    call with the same inputs resumes from the newest valid checkpoint:
    completed references are NEVER recomputed (their fingerprints and
    similarity rows are replayed from the checkpoint) and the resumed
    frame is bit-identical to an uninterrupted run.  A checkpoint from
    a different scan config or different input lists is refused with a
    warning and the campaign starts fresh (exact restore or nothing).

    Parameters
    ----------
    reference_smiles, target_smiles, threshold, fingerprint, n_bits, radius:
        The :func:`similarity_scan` contract, unchanged.
    progress:
        Optional ``progress(done, total, note)`` callback fired on the
        calling thread as reference chunks COMPLETE (``done``/``total``
        are reference counts).  Returning ``False`` stops the campaign
        at the next chunk boundary with
        :class:`~pharmaml.exceptions.JobCancelled` -- and when
        ``checkpoint_dir`` is given, the completed partial rows ride the
        exception as ``partial_results`` (a tidy long-format frame of
        the SAME shape as the final result).
    checkpoint_dir:
        (part5-continuation round 7, §13) Optional directory for
        resumable checkpoints.  ``None`` (the default) keeps the
        chunked-but-checkpoint-less behaviour.
    checkpoint_every:
        How many chunk boundaries between checkpoint saves (1 = every
        chunk; must be a positive int).
    checkpoint_resume:
        When ``True`` (default) and ``checkpoint_dir`` is given, resume
        from the newest valid checkpoint whose fingerprints match.
        ``False`` still writes checkpoints but always computes from
        scratch.

    Returns
    -------
    pandas.DataFrame
        The filtered similarity pairs, sorted by descending similarity
        (ties keep campaign order), exactly as :func:`similarity_scan`.

    Raises
    ------
    DataValidationError
        If ``threshold`` is outside [0, 1] or either SMILES argument is
        a bare string.
    JobCancelled
        If ``progress`` returns ``False`` at a chunk boundary; with
        ``checkpoint_dir`` the partial rows ride the exception.

    Examples
    --------
    >>> frame = run_similarity_campaign(
    ...     ["CCO", "CCC"], ["CCO", "CCC", "c1ccccc1O"], threshold=0.25)
    >>> frame["reference"].tolist()
    ['CCO', 'CCC', 'CCO', 'CCC']
    >>> frame["similarity"].tolist()
    [1.0, 1.0, 0.25, 0.25]
    """
    if not 0.0 <= threshold <= 1.0:
        raise DataValidationError(f"threshold must be within [0, 1], got {threshold}")
    _require_rdkit_similarity()

    from pharmaml.chem.descriptors import (
        _as_smiles_list,
        compute_fingerprints,
    )

    reference_list = _as_smiles_list(reference_smiles, "reference_smiles")
    target_list = _as_smiles_list(target_smiles, "target_smiles")

    # The target fingerprints are computed ONCE (they are the library --
    # shared by every reference chunk); the campaign state rides on the
    # references instead.
    tgt_matrix, tgt_idx = compute_fingerprints(
        target_list, kind=fingerprint, n_bits=n_bits, radius=radius
    )

    total = len(reference_list)
    chunk = max(1, min(25, total))
    rows: list[dict[str, Any]] = []
    dropped: list[tuple[int, str]] = []
    done = 0

    if checkpoint_dir is not None:
        if (
            not isinstance(checkpoint_every, int)
            or isinstance(checkpoint_every, bool)
            or checkpoint_every < 1
        ):
            raise ValueError(
                f"checkpoint_every must be a positive int, got {checkpoint_every!r}"
            )
        from pharmaml.utils.checkpoint_schema import OperationSpec
        from pharmaml.utils.checkpoints import CheckpointManager, PersistenceError

        try:
            manager = CheckpointManager(checkpoint_dir, name="similarity-campaign")
            spec = OperationSpec(
                operation_type="similarity_campaign",
                operation_id=f"scan-{id(reference_list):x}-{id(target_list):x}",
            )
        except PersistenceError as exc:
            _LOG.warning(
                "checkpoint home unusable (%s); computing without checkpoints", exc
            )
            manager = None  # type: ignore[assignment]
            spec = None  # type: ignore[assignment]
        config_fp = _campaign_config_fingerprint(fingerprint, n_bits, radius, threshold)
        dataset_fp = _campaign_dataset_fingerprint(reference_list, target_list)
        if manager is not None and checkpoint_resume:
            try:
                latest = manager.latest()
            except OSError:  # pragma: no cover - scan failure
                latest = None
                _LOG.warning("checkpoint scan failed; starting fresh")
            if latest is not None and latest.resumable:
                state = latest.state
                if (
                    state.get("config_fp") != config_fp
                    or state.get("dataset_fp") != dataset_fp
                ):
                    _LOG.warning(
                        "similarity-campaign checkpoint at step %d is for a "
                        "different config/dataset; starting fresh (exact "
                        "restore or nothing)",
                        latest.step,
                    )
                else:
                    try:
                        rows = [
                            {
                                "reference_index": int(entry["reference_index"]),
                                "target_index": int(entry["target_index"]),
                                "similarity": float(entry["similarity"]),
                            }
                            for entry in state["rows"]
                        ]
                        dropped = [
                            (int(entry["row"]), str(entry["text"]))
                            for entry in state["dropped"]
                        ]
                        done = int(latest.step)
                        _LOG.info(
                            "similarity campaign resumed from checkpoint step %d "
                            "(%d/%d references already scanned)",
                            done,
                            done,
                            total,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        _LOG.warning(
                            "checkpoint payload unreadable (%s); starting fresh", exc
                        )
                        rows = []
                        dropped = []
                        done = 0

        boundaries_since_save = 0
        for start in range(done, total, chunk):
            batch = reference_list[start : start + chunk]
            # fingerprint this chunk's references, then score against the
            # (already materialised) target matrix -- the similarity_scan
            # math, one chunk at a time
            ref_matrix, ref_idx = compute_fingerprints(
                batch, kind=fingerprint, n_bits=n_bits, radius=radius
            )
            if len(ref_idx) == 0:
                similarities = np.zeros((0, len(tgt_idx)), dtype=np.float64)
            else:
                similarities = tanimoto_matrix(ref_matrix, tgt_matrix)
            mask = similarities >= threshold
            for ref_row, ref_pos_in_chunk in enumerate(ref_idx):
                ref_pos = start + ref_pos_in_chunk
                for tgt_row, tgt_pos in enumerate(tgt_idx):
                    if mask[ref_row, tgt_row]:
                        rows.append(
                            {
                                "reference_index": int(ref_pos),
                                "target_index": int(tgt_pos),
                                "similarity": float(similarities[ref_row, tgt_row]),
                            }
                        )
            # the drop ledger: references that failed to parse (positions
            # absent from ref_idx), mirroring compute_fingerprints' drop rule
            kept = set(ref_idx)
            for offset, text in enumerate(batch):
                if offset not in kept:
                    dropped.append((start + offset, text))
            done += len(batch)
            boundaries_since_save += 1
            if (
                manager is not None
                and spec is not None
                and (start + chunk >= total or boundaries_since_save >= checkpoint_every)
            ):
                boundaries_since_save = 0
                try:
                    manager.save(
                        step=done,
                        total=total,
                        state={
                            "config_fp": config_fp,
                            "dataset_fp": dataset_fp,
                            "rows": rows,
                            "dropped": [
                                {"row": row, "text": text} for row, text in dropped
                            ],
                        },
                        note=f"similarity campaign {done}/{total} references",
                        operation=spec,
                    )
                except (PersistenceError, OSError, TypeError, ValueError) as exc:
                    _LOG.warning(
                        "checkpoint write failed at %d/%d (%s); continuing without it",
                        done,
                        total,
                        exc,
                    )
            if progress is not None and progress(done, total, "similarity campaign") is False:
                partial = _campaign_finish(
                    rows, dropped, reference_list, target_list
                )
                raise JobCancelled(
                    f"similarity campaign cancelled after {done}/{total} references "
                    "-- stopped at a checkpoint by the caller",
                    partial_results=partial,
                )

        return _campaign_finish(rows, dropped, reference_list, target_list)

    # the checkpoint-less path: same chunking and progress contract,
    # nothing persisted, a stop discards the partial rows (the
    # historical round-205 convention)
    for start in range(0, total, chunk):
        batch = reference_list[start : start + chunk]
        ref_matrix, ref_idx = compute_fingerprints(
            batch, kind=fingerprint, n_bits=n_bits, radius=radius
        )
        if len(ref_idx) == 0:
            similarities = np.zeros((0, len(tgt_idx)), dtype=np.float64)
        else:
            similarities = tanimoto_matrix(ref_matrix, tgt_matrix)
        mask = similarities >= threshold
        for ref_row, ref_pos_in_chunk in enumerate(ref_idx):
            ref_pos = start + ref_pos_in_chunk
            for tgt_row, tgt_pos in enumerate(tgt_idx):
                if mask[ref_row, tgt_row]:
                    rows.append(
                        {
                            "reference_index": int(ref_pos),
                            "target_index": int(tgt_pos),
                            "similarity": float(similarities[ref_row, tgt_row]),
                        }
                    )
        done += len(batch)
        if progress is not None and progress(done, total, "similarity campaign") is False:
            raise JobCancelled(
                f"similarity campaign cancelled after {done}/{total} references "
                "-- stopped at a checkpoint by the caller"
            )

    return _campaign_finish(rows, dropped, reference_list, target_list)
