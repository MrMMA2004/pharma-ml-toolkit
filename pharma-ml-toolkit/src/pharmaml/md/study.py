r"""The MD ligand study — the uploaded ``MD_analysis.py`` pipeline as data.

The user's script walked a directory of per-ligand folders, parsed each
GROMACS/MM-PBSA export (``.xvg`` and ``.ndx``), normalised the units
(binding energy kJ→kcal, ps→ns on long time axes, smoothed H-bond
counts), computed per-metric statistics and a cross-metric correlation
heatmap per ligand, then compared ligands globally with a CSV of
mean/std/CV rows.  This module is that pipeline, productionized:

* :func:`read_ligand_folder` — one ligand folder → :class:`LigandResult`
  (known exports parsed, units normalised, unparseable files skipped
  with a logged warning — never fatal, exactly the script's tolerance);
* :func:`build_study` — a campaign root (one sub-folder per ligand, or
  a flat folder for a single ligand; ``Global_plots`` skipped) →
  :class:`MDStudy`;
* :func:`metric_correlations` — the per-ligand correlation matrix over
  the metric series (min-length aligned, the script's own rule);
* :meth:`MDStudy.summary_csv` — the script's
  ``global_summary_stats.csv`` contract, verbatim header.

The matplotlib panels the script drew become the *web* Molecular
Dynamics tab's rendering of the same data — the numbers are computed
here, once, honestly.

Examples
--------
>>> import tempfile
>>> from pathlib import Path
>>> with tempfile.TemporaryDirectory() as tmp:
...     root = Path(tmp)
...     _ = (root / "ligA").mkdir()
...     _ = (root / "ligA" / "ligand_rmsd.xvg").write_text(
...         "# t rmsd\n0 0.1\n1000 0.2\n2000 0.3\n"
...     )
...     _ = (root / "ligA" / "hbnum_intermolecular.ndx").write_text(
...         "0 1 0.5\n1000 2 0.6\n2000 3 0.7\n"
...     )
...     _ = (root / "ligA" / "notes.txt").write_text("ignore me\n")
...     study = build_study(root)
>>> len(study)
1
>>> sorted(study.ligands[0].series)
['hbnum_intermolecular', 'ligand_rmsd']
>>> rmsd = study.ligands[0].series["ligand_rmsd"]
>>> (float(rmsd.time[-1]), round(float(rmsd.values[-1]), 3))
(2.0, 0.3)
>>> labels, corr = metric_correlations(study.ligands[0].series)
>>> labels
['hbnum_intermolecular', 'ligand_rmsd']
>>> bool(np.allclose(np.diag(corr), 1.0))
True
>>> float(corr[0, 1]) > 0.5  # H-bond counts rise with the RMSD drift
True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pharmaml.exceptions import MDAnalysisError
from pharmaml.logger import get_logger
from pharmaml.md.xvg import (
    MetricInfo,
    XvgSeries,
    classify_metric,
    metric_stats,
    moving_average,
    parse_ndx_hbnum,
    parse_xvg,
    ps_to_ns,
    to_kcal,
)

__all__ = [
    "LigandResult",
    "MDStudy",
    "build_study",
    "metric_correlations",
    "read_ligand_folder",
]

_logger = get_logger("pharmaml.md.study")

_GLOBAL_DIR_NAME = "Global_plots"


@dataclass(frozen=True)
class LigandResult:
    """One ligand's parsed metric series, keyed by metric key."""

    name: str
    series: dict[str, XvgSeries]

    @property
    def metric_keys(self) -> list[str]:
        return list(self.series)

    def stats_rows(self) -> list[dict[str, Any]]:
        """Per-metric statistics rows — the uploaded CSV's payload."""
        rows: list[dict[str, Any]] = []
        for key, series in self.series.items():
            stats = metric_stats(series.values)
            rows.append({"ligand": self.name, "metric": key, "n": len(series), **stats})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "series": {key: s.to_dict() for key, s in self.series.items()},
        }


@dataclass(frozen=True)
class MDStudy:
    """A whole MD campaign: one :class:`LigandResult` per ligand folder."""

    ligands: list[LigandResult]

    def __len__(self) -> int:
        return len(self.ligands)

    @property
    def ligand_names(self) -> list[str]:
        return [ligand.name for ligand in self.ligands]

    def ligand(self, name: str) -> LigandResult:
        for ligand in self.ligands:
            if ligand.name == name:
                return ligand
        raise MDAnalysisError(
            f"no ligand named {name!r} in the study "
            f"(have: {self.ligand_names})"
        )

    def metric_series(self, metric_key: str) -> dict[str, XvgSeries]:
        """One metric across every ligand — the global-comparison panel."""
        return {
            ligand.name: ligand.series[metric_key]
            for ligand in self.ligands
            if metric_key in ligand.series
        }

    def summary_csv(self) -> str:
        """The uploaded ``global_summary_stats.csv`` contract, verbatim.

        Header ``Folder,File,Mean,Stdev,Coefficient_of_Variation``; one
        row per ligand × metric, blank CV where the mean is zero.
        """
        lines = ["Folder,File,Mean,Stdev,Coefficient_of_Variation"]
        for ligand in self.ligands:
            for key, series in ligand.series.items():
                stats = metric_stats(series.values)
                mean = "" if stats["mean"] is None else f"{stats['mean']:.6g}"
                std = "" if stats["std"] is None else f"{stats['std']:.6g}"
                cv = "" if stats["cv"] is None else f"{stats['cv']:.6g}"
                lines.append(f"{ligand.name},{key},{mean},{std},{cv}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {"ligands": [ligand.to_dict() for ligand in self.ligands]}


def _normalize_series(
    series: XvgSeries, info: MetricInfo, *, smooth_hbonds: bool
) -> XvgSeries:
    """Apply the uploaded script's unit conversions to a parsed series."""
    values = series.values
    time = series.time
    if info.kind == "energy":
        values = to_kcal(values)
    if "Time" in info.x_label and float(np.max(time)) > 100.0:
        time = ps_to_ns(time)
    if smooth_hbonds and info.kind == "hbond" and len(series) >= 3:
        values = moving_average(values, window=3)
    return XvgSeries(
        time=time, values=values, source=series.source, label=info.title
    )


def read_ligand_folder(
    folder: str | Path, *, smooth_hbonds: bool = True
) -> LigandResult:
    """Parse one ligand folder's GROMACS/MM-PBSA exports.

    Every file whose name classifies (see
    :func:`pharmaml.md.xvg.classify_metric`) is parsed — ``.ndx`` files
    through the H-bond table reader, everything else as ``.xvg`` — and
    normalised (kJ→kcal on energy metrics, ps→ns on long time axes, the
    3-point H-bond smoothing when ``smooth_hbonds``).  Unparseable
    files are skipped with a logged warning, never fatal — the
    uploaded script's tolerance, made explicit.

    A folder with no recognised export raises :class:`MDAnalysisError`
    (a ligand folder without a single metric is a wrong path, not a
    quiet empty result).
    """
    root = Path(folder)
    if not root.is_dir():
        raise MDAnalysisError(f"ligand folder {root} is not a directory")

    series: dict[str, XvgSeries] = {}
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        info = classify_metric(path.name)
        if info is None:
            continue
        try:
            parsed = parse_ndx_hbnum(path) if path.suffix == ".ndx" else parse_xvg(path)
        except MDAnalysisError as exc:
            _logger.warning("skipping %s in %s: %s", path.name, root.name, exc)
            continue
        normalized = _normalize_series(
            parsed, info, smooth_hbonds=smooth_hbonds
        )
        series[info.key] = normalized
    if not series:
        raise MDAnalysisError(
            f"no parsable GROMACS metric exports found in {root} "
            "(expected e.g. ligand_rmsd.xvg, hbnum_intermolecular.ndx)"
        )
    return LigandResult(name=root.name, series=series)


def build_study(root: str | Path, *, smooth_hbonds: bool = True) -> MDStudy:
    """Build the campaign study from a root folder.

    Sub-folders (except ``Global_plots``, the uploaded script's own
    output directory) are one ligand each; a flat folder with exports
    directly inside is a single-ligand study.  Sub-foldors that fail to
    parse are skipped with a warning so one broken ligand never sinks
    the campaign — the honest campaign rule.
    """
    base = Path(root)
    if not base.is_dir():
        raise MDAnalysisError(f"study root {base} is not a directory")
    subfolders = sorted(
        p for p in base.iterdir() if p.is_dir() and p.name != _GLOBAL_DIR_NAME
    )
    if not subfolders:
        return MDStudy([read_ligand_folder(base, smooth_hbonds=smooth_hbonds)])

    ligands: list[LigandResult] = []
    for folder in subfolders:
        try:
            ligands.append(
                read_ligand_folder(folder, smooth_hbonds=smooth_hbonds)
            )
        except MDAnalysisError as exc:
            _logger.warning("skipping ligand folder %s: %s", folder.name, exc)
    if not ligands:
        raise MDAnalysisError(
            f"no ligand folder under {base} produced a parsable result"
        )
    return MDStudy(ligands)


def metric_correlations(
    series: dict[str, XvgSeries],
) -> tuple[list[str], np.ndarray] | None:
    """Correlation matrix across one ligand's metric series.

    Series are truncated to the shortest common length (the uploaded
    script's own alignment rule) and correlated with ``np.corrcoef``;
    fewer than two non-empty series (or a common length under two)
    yields ``None`` — the caller decides what an unmeasurable
    correlation means.  A constant series produces NaN entries in its
    row, which ``np.corrcoef`` reports honestly.

    >>> from pharmaml.md.xvg import XvgSeries
    >>> a = XvgSeries(time=np.arange(4.0), values=np.array([1.0, 2.0, 3.0, 4.0]))
    >>> b = XvgSeries(time=np.arange(4.0), values=np.array([2.0, 4.0, 6.0, 8.0]))
    >>> labels, corr = metric_correlations({"a": a, "b": b})
    >>> (labels, round(float(corr[0, 1]), 6))
    (['a', 'b'], 1.0)
    >>> metric_correlations({"a": a}) is None
    True
    """
    metrics = {key: s for key, s in series.items() if len(s) > 0}
    if len(metrics) < 2:
        return None
    min_len = min(len(s) for s in metrics.values())
    if min_len < 2:
        return None
    data = np.array([s.values[:min_len] for s in metrics.values()])
    return list(metrics.keys()), np.corrcoef(data)
