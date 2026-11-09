r"""GROMACS analysis-file parsing — the uploaded MD pipeline, productionized.

``upload/MD_analysis.py`` — the user's own 421-line GROMACS
post-processing script (``.xvg``/``.ndx`` parsing, unit conversion,
per-ligand statistics and cross-metric correlation heatmaps) — is the
seed of this module.  The ecosystem that file belongs to —
``MDAnalysis/mdanalysis`` and ``mdtraj/mdtraj`` for the trajectory core,
``gromacs/gromacs`` ``gmx`` tools plus ``Amber-MD/cpptraj`` /
``Amber-MD/pytraj`` for the analysis vocabulary,
``getcontacts/getcontacts`` for the contact timelines and
``Valdes-Tresanco-MS/gmx_MMPBSA`` for the binding free-energy legs —
all emit the same plain-text ``.xvg`` contract this parser honours:

* ``#`` starts a comment, ``@`` an xmgrace directive — both skipped;
* data rows are whitespace-separated numbers, the first column is the
  abscissa (time in ps unless the file says otherwise) and the second
  the metric value;
* hydrogen-bond count exports can arrive as ``.ndx``-style numeric
  tables where the time column and the integer H-bond column must be
  *found* among the columns (the uploaded script's heuristic, kept
  bit-for-bit: the time column is the first finite, monotonically
  non-decreasing, non-constant column; the H-bond column is the first
  *other* finite column whose values are near-integers).

Units are the script's own honest conversions: binding free energy
arrives in kJ/mol and converts through :data:`KJ_TO_KCAL`; a time axis
in ps is presented in ns once its maximum exceeds 100.

One deliberate upgrade over the upload: the parse is *pure*.  The
uploaded script folded a 3-point moving average into ``.ndx`` parsing;
here :meth:`XvgSeries.smoothed` carries the smoothing explicitly and
:mod:`pharmaml.md.study` applies it exactly where the script did.

Public surface:

* :func:`parse_xvg` — one ``.xvg`` text/file → :class:`XvgSeries`;
* :func:`parse_ndx_hbnum` — H-bond count tables → :class:`XvgSeries`;
* :class:`XvgSeries` — time + values with lazily-computed stats;
* :func:`metric_stats` — mean / std / CV (the uploaded CSV's columns);
* :func:`moving_average` — the uploaded 3-point H-bond smoother;
* :func:`to_kcal` / :func:`ps_to_ns` — the unit conversions;
* :func:`classify_metric` — filename → :class:`MetricInfo` (title, axis
  labels, kind, colour) for the ten standard GROMACS/MM-PBSA exports,
  plus kind inference for unknown names.

Examples
--------
>>> text = "# GROMACS\n@ s0 legend \"RMSD\"\n0.0 0.05\n1000.0 0.08\n"
>>> s = parse_xvg(text)
>>> (len(s), s.time[0], s.values[-1])
(2, 0.0, 0.08)
>>> sorted(s.stats)
['cv', 'mean', 'std']
>>> ndx = "0 1 0.5\n10 2 0.6\n20 2 0.7\n30 3 0.8\n"
>>> hb = parse_ndx_hbnum(ndx)
>>> (list(hb.time), list(hb.values))
([0.0, 10.0, 20.0, 30.0], [1.0, 2.0, 2.0, 3.0])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import MDAnalysisError

__all__ = [
    "KJ_TO_KCAL",
    "METRIC_FILES",
    "MetricInfo",
    "XvgSeries",
    "classify_metric",
    "metric_stats",
    "moving_average",
    "parse_ndx_hbnum",
    "parse_xvg",
    "ps_to_ns",
    "to_kcal",
]

KJ_TO_KCAL = 1.0 / 4.184

_KIND_COLORS: dict[str, str] = {
    "rmsd": "red",
    "rmsf": "green",
    "hbond": "blue",
    "rg": "purple",
    "sasa": "orange",
    "energy": "brown",
}


@dataclass(frozen=True)
class MetricInfo:
    """Display metadata for one standard MD metric export.

    The uploaded script carried three parallel dictionaries
    (``expected_files`` / ``plot_titles`` / ``color_map``); this class is
    their union with the colour resolved through the metric kind.
    """

    key: str
    """Stable metric key (the filename stem, e.g. ``ligand_rmsd``)."""

    kind: str
    """One of ``rmsd``/``rmsf``/``hbond``/``rg``/``sasa``/``energy``."""

    title: str
    x_label: str
    y_label: str
    color: str


METRIC_FILES: dict[str, MetricInfo] = {
    "ligand_rmsd.xvg": MetricInfo(
        "ligand_rmsd", "rmsd", "Ligand RMSD", "Time (ns)", "RMSD (Å)", "red"
    ),
    "protein_backbone_rmsd.xvg": MetricInfo(
        "protein_backbone_rmsd", "rmsd", "Protein Backbone RMSD",
        "Time (ns)", "RMSD (Å)", "red",
    ),
    "ligand_rmsf.xvg": MetricInfo(
        "ligand_rmsf", "rmsf", "Ligand RMSF", "Atom index", "RMSF (Å)", "green"
    ),
    "per_residue_rmsf.xvg": MetricInfo(
        "per_residue_rmsf", "rmsf", "Per-residue RMSF",
        "Residue index", "RMSF (Å)", "green",
    ),
    "hbnum_intermolecular.xvg": MetricInfo(
        "hbnum_intermolecular", "hbond", "Intermolecular H-bonds",
        "Time (ns)", "Number of H-bonds", "blue",
    ),
    "hbnum_intramolecular.xvg": MetricInfo(
        "hbnum_intramolecular", "hbond", "Intramolecular H-bonds",
        "Time (ns)", "Number of H-bonds", "blue",
    ),
    "protein_backbone_Rg.xvg": MetricInfo(
        "protein_backbone_Rg", "rg", "Protein Backbone Radius of Gyration",
        "Time (ns)", "Rg (Å)", "purple",
    ),
    "ligand_sasa.xvg": MetricInfo(
        "ligand_sasa", "sasa", "Ligand SASA", "Time (ns)", "SASA (Å²)", "orange"
    ),
    "protein_sasa.xvg": MetricInfo(
        "protein_sasa", "sasa", "Protein SASA", "Time (ns)", "SASA (Å²)", "orange"
    ),
    "binding_free_energy.xvg": MetricInfo(
        "binding_free_energy", "energy", "Binding Free Energy",
        "Time (ns)", "ΔG (kcal/mol)", "brown",
    ),
}


def _read_source(source: str | Path) -> str:
    """Read the parser input: a path is read, a string is used verbatim."""
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="replace")
    if isinstance(source, str):
        if "\n" in source or "\r" in source:
            return source
        # A single-line string is ambiguous; treat bare non-path text as
        # content only when it clearly is not an existing file path.
        candidate = Path(source)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace")
        return source
    raise MDAnalysisError(f"unsupported xvg source type {type(source).__name__}")


def _source_label(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.name
    if isinstance(source, str) and "\n" not in source and Path(source).exists():
        return Path(source).name
    return ""


@dataclass(frozen=True)
class XvgSeries:
    """One parsed metric series: a 1-D time axis plus 1-D values."""

    time: np.ndarray
    values: np.ndarray
    source: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if time.ndim != 1 or values.ndim != 1:
            raise MDAnalysisError("XvgSeries time/values must be 1-D arrays")
        if time.shape[0] != values.shape[0]:
            raise MDAnalysisError(
                f"time ({time.shape[0]}) and values ({values.shape[0]}) lengths differ"
            )
        if time.shape[0] == 0:
            raise MDAnalysisError("XvgSeries is empty")
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @property
    def stats(self) -> dict[str, float | None]:
        """Mean / population std / CV — the uploaded CSV's statistics."""
        return metric_stats(self.values)

    def smoothed(self, window: int = 3) -> np.ndarray:
        """The uploaded H-bond smoothing, as an explicit transform."""
        return moving_average(self.values, window=window)

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python copy for JSON serialisation (bridge payloads)."""
        return {
            "time": [float(v) for v in self.time],
            "values": [float(v) for v in self.values],
            "source": self.source,
            "label": self.label,
            "stats": self.stats,
        }


def parse_xvg(source: str | Path) -> XvgSeries:
    r"""Parse GROMACS ``.xvg`` text (or a file path) into an :class:`XvgSeries`.

    Comment (``#``) and xmgrace directive (``@``) lines are skipped; the
    first two whitespace-separated numeric columns of every other line
    become the time and value axes (later columns are ignored, exactly
    as the uploaded script read them).

    Raises :class:`~pharmaml.exceptions.MDAnalysisError` when no numeric
    data row survives — an empty metric is a broken export, and callers
    (:func:`pharmaml.md.study.read_ligand_folder`) surface it as a
    warning rather than a silent zero.

    >>> parse_xvg("# only comments\n@ s0")
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.MDAnalysisError: no numeric data rows in the .xvg input
    """
    text = _read_source(source)
    xs: list[float] = []
    ys: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            ys.append(float(parts[1]))
        except ValueError:
            continue
    if not xs:
        raise MDAnalysisError("no numeric data rows in the .xvg input")
    return XvgSeries(
        time=np.asarray(xs, dtype=float),
        values=np.asarray(ys, dtype=float),
        source=_source_label(source),
    )


def parse_ndx_hbnum(source: str | Path) -> XvgSeries:
    r"""Parse an ``.ndx``-style H-bond count table into an :class:`XvgSeries`.

    The uploaded script's column-finding heuristic, verbatim: pad ragged
    rows with NaN, pick the *time* column as the first finite,
    monotonically non-decreasing, non-constant column, and the *H-bond*
    column as the first other finite column whose values are
    near-integers with a small mean.  Non-finite rows are dropped.

    >>> parse_ndx_hbnum("[ nothing ]\njust words\n")
    Traceback (most recent call last):
        ...
    pharmaml.exceptions.MDAnalysisError: no numeric rows in the .ndx H-bond input
    """
    text = _read_source(source)
    rows: list[list[float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "@", "[", "]")):
            continue
        nums: list[float] = []
        for part in line.split():
            try:
                nums.append(float(part))
            except ValueError:
                nums.append(math.nan)
        if any(not math.isnan(v) for v in nums):
            rows.append(nums)
    if not rows:
        raise MDAnalysisError("no numeric rows in the .ndx H-bond input")

    max_cols = max(len(r) for r in rows)
    data = np.full((len(rows), max_cols), np.nan)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            if not math.isnan(value):
                data[i, j] = value

    time_col: int | None = None
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.all(np.isfinite(col)):
            diffs = np.diff(col)
            if np.all(diffs >= -1e-8) and np.ptp(col) > 0:
                time_col = j
                break

    hb_col: int | None = None
    for j in range(data.shape[1]):
        if j == time_col:
            continue
        col = data[:, j]
        if np.all(np.isfinite(col)):
            residual = np.abs(col - np.round(col))
            if np.all(residual < 1e-6) and float(np.mean(np.abs(col))) < 1000:
                hb_col = j
                break

    if time_col is None:
        time_col = 0
    if hb_col is None:
        hb_col = 1 if data.shape[1] > 1 else 0

    x = data[:, time_col]
    y = data[:, hb_col]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size == 0:
        raise MDAnalysisError("no finite (time, hbond) pairs in the .ndx input")
    return XvgSeries(
        time=x,
        values=y,
        source=_source_label(source),
        label="Number of H-bonds",
    )


def metric_stats(values: np.ndarray) -> dict[str, float | None]:
    """Mean, population standard deviation and coefficient of variation.

    NaN-aware; an empty or all-NaN input yields ``None`` for all three
    (the uploaded CSV wrote a blank CV when the mean was zero — ``None``
    is the honest typed equivalent).

    >>> metric_stats(np.array([2.0, 4.0]))["cv"] is not None
    True
    >>> round(metric_stats(np.array([2.0, 4.0]))["cv"], 4)
    0.3333
    >>> metric_stats(np.array([]))
    {'mean': None, 'std': None, 'cv': None}
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or bool(np.all(np.isnan(arr))):
        return {"mean": None, "std": None, "cv": None}
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    cv = std / mean if mean != 0.0 else None
    return {"mean": mean, "std": std, "cv": cv}


def moving_average(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving average (``np.convolve`` ``same`` mode).

    The uploaded pipeline smoothed every H-bond count series with a
    3-point window before plotting; this is that smoother, exposed.

    >>> [round(float(v), 6) for v in moving_average(np.array([1.0, 2.0, 3.0]))]
    [1.0, 2.0, 1.666667]
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise MDAnalysisError("moving_average needs a non-empty 1-D array")
    if window < 1:
        raise MDAnalysisError(f"window must be >= 1, got {window}")
    if window > arr.size:
        raise MDAnalysisError(
            f"window ({window}) exceeds the series length ({arr.size})"
        )
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="same")


def to_kcal(values: np.ndarray) -> np.ndarray:
    """Convert kJ/mol → kcal/mol (the uploaded binding-energy conversion)."""
    return np.asarray(values, dtype=float) * KJ_TO_KCAL


def ps_to_ns(time: np.ndarray) -> np.ndarray:
    """Convert a ps time axis to ns."""
    return np.asarray(time, dtype=float) / 1000.0


def classify_metric(filename: str) -> MetricInfo | None:
    """Map an export filename to its :class:`MetricInfo`, or ``None``.

    Known GROMACS/MM-PBSA filenames match exactly (the ``.ndx`` H-bond
    twins share their ``.xvg`` stems); unknown names fall back to kind
    inference from substrings (``rmsd``/``rmsf``/``hbnum``/``sasa``/
    ``energy``/``rg`` — ``energy`` is tested before ``rg`` because the
    word itself contains those letters); anything else is ``None`` and
    the study reader ignores the file.

    >>> classify_metric("ligand_rmsd.xvg").kind
    'rmsd'
    >>> classify_metric("hbnum_intermolecular.ndx").kind
    'hbond'
    >>> classify_metric("replica_energy_2.xvg").kind
    'energy'
    >>> classify_metric("random.txt") is None
    True
    """
    name = Path(filename).name
    stem = Path(filename).stem
    if name in METRIC_FILES:
        return METRIC_FILES[name]
    if f"{stem}.xvg" in METRIC_FILES:
        return METRIC_FILES[f"{stem}.xvg"]
    lowered = name.lower()
    # NOTE: "energy" is checked before "rg" — "eneRGy" contains "rg".
    for token, kind in (
        ("rmsd", "rmsd"),
        ("rmsf", "rmsf"),
        ("hbnum", "hbond"),
        ("hbond", "hbond"),
        ("sasa", "sasa"),
        ("energy", "energy"),
        ("rg", "rg"),
    ):
        if token in lowered:
            return MetricInfo(
                key=stem,
                kind=kind,
                title=stem.replace("_", " ").title(),
                x_label="Frame",
                y_label="",
                color=_KIND_COLORS[kind],
            )
    return None
