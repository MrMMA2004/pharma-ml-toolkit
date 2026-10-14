"""Parsers for AutoDock4 (DLG) and Vina outputs.

Upgrades ``readerwithdetails.py`` (cluster-table regex parser with
"most populated cluster" logic), ``reader.py``/``reader_at_same_folder.py``
(iterative summary), ``checker.py`` (completion audit) and
``dlg_extracter.py``/``Prettier_extracted_dlgs.py``.

All parsers accept *text or paths* and return typed dataclasses or
DataFrames, so they are unit-testable without a docking engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pharmaml.exceptions import DockingParseError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "DockingCluster",
    "VinaPose",
    "parse_dlg_cluster_table",
    "parse_dlg_summary",
    "parse_vina_output",
    "check_docking_completion",
    "scan_docking_directory",
]


@dataclass(frozen=True, slots=True)
class DockingCluster:
    """One row of the AutoDock4 cluster table.

    Attributes
    ----------
    rank:
        Cluster rank (1 = lowest-energy cluster).
    lowest_energy:
        Best binding energy in the cluster (kcal/mol).
    mean_energy:
        Mean binding energy of the cluster (kcal/mol).
    num_in_cluster:
        Poses assigned to the cluster.
    """

    rank: int
    lowest_energy: float
    mean_energy: float
    num_in_cluster: int


@dataclass(frozen=True, slots=True)
class VinaPose:
    """One pose from a Vina output PDBQT.

    Attributes
    ----------
    mode:
        Pose rank (1-based).
    affinity:
        Predicted binding affinity (kcal/mol).
    rmsd_ub / rmsd_lb:
        RMSD upper/lower bounds vs. the best mode (Å).
    """

    mode: int
    affinity: float
    rmsd_ub: float
    rmsd_lb: float


# Cluster table lines look like:
#   "   1 |     -11.5 |   2 |     -11.4 |  100 |"
_CLUSTER_ROW = re.compile(
    r"^\s*(\d+)\s+\|\s+([-+]?\d+\.?\d*)\s+\|\s+\d+\s+\|"
    r"\s+([-+]?\d+\.?\d*)\s+\|\s+(\d+)\s+\|"
)
# Vina table lines look like:
#   "   1        -11.5        0.000        0.000"
_VINA_ROW = re.compile(
    r"^\s+(\d+)\s+([-+]?\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s*$"
)


def _read_text(source: str | Path) -> str:
    """Accept a path or raw text; return the text content."""
    if isinstance(source, Path):
        if not source.is_file():
            raise DockingParseError(f"file not found: {source}")
        return source.read_text(encoding="utf-8", errors="ignore")
    text = str(source)
    # A short single-line string is most likely a path; try to read it
    # and otherwise treat it as literal text.
    if "\n" not in text and len(text) < 260:
        candidate = Path(text)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="ignore")
    return text


def parse_dlg_cluster_table(source: str | Path) -> list[DockingCluster]:
    """Parse the ``CLUSTERING HISTOGRAM`` table of an AutoDock4 DLG.

    The DLG prints, after the docking runs, a table whose columns are
    ``rank | lowest-energy | run# | mean-energy | #in-cluster``.  This
    is exactly the regex the original ``readerwithdetails.py`` used,
    now with validation and typed results.

    Raises
    ------
    DockingParseError
        If the file/text contains no cluster table (e.g. a failed
        docking that stopped before clustering).
    """
    text = _read_text(source)
    clusters: list[DockingCluster] = []
    in_table = False
    for line in text.splitlines():
        if not in_table:
            # AutoDock writes the section title in varying case across
            # versions ("CLUSTERING HISTOGRAM" / "Clustering Histogram").
            lowered = line.lower()
            if "clus" in lowered and "histogram" in lowered:
                in_table = True
            continue
        match = _CLUSTER_ROW.match(line)
        if match:
            clusters.append(
                DockingCluster(
                    rank=int(match.group(1)),
                    lowest_energy=float(match.group(2)),
                    mean_energy=float(match.group(3)),
                    num_in_cluster=int(match.group(4)),
                )
            )
    if not clusters:
        raise DockingParseError("no cluster table found in the provided DLG source")
    return clusters


def parse_dlg_summary(source: str | Path) -> dict[str, float | int]:
    """Summarise one DLG: best cluster, most populated cluster, extremes.

    Mirrors the columns the original exported to Excel:

    * ``top_*``      -- the first table row (lowest energy overall),
    * ``best_*``     -- the most populated cluster (largest
      ``num_in_cluster``), which is often the more robust pick,
    * ``n_clusters`` -- number of clusters.
    """
    clusters = parse_dlg_cluster_table(source)
    top = clusters[0]
    best = max(clusters, key=lambda c: c.num_in_cluster)
    return {
        "n_clusters": len(clusters),
        "top_rank": top.rank,
        "top_lowest_energy": top.lowest_energy,
        "top_mean_energy": top.mean_energy,
        "top_num_in_cluster": top.num_in_cluster,
        "best_rank": best.rank,
        "best_lowest_energy": best.lowest_energy,
        "best_mean_energy": best.mean_energy,
        "best_num_in_cluster": best.num_in_cluster,
    }


def parse_vina_output(source: str | Path) -> list[VinaPose]:
    """Parse the mode table of a Vina output (stdout log or PDBQT REMARKs).

    Vina writes::

        mode | affinity | dist from best mode
            | (kcal/mol) | rmsd l.b. | rmsd u.b.
        -----+------------+----------+----------
           1       -11.5          0          0

    which the regex captures regardless of the table decoration.

    Raises
    ------
    DockingParseError
        If no pose rows are found.
    """
    text = _read_text(source)
    poses: list[VinaPose] = []
    for line in text.splitlines():
        if line.startswith(("REMARK VINA RESULT", "VINA RESULT")):
            parts = line.split(":")[-1].split()
            if len(parts) >= 3:
                poses.append(
                    VinaPose(
                        mode=len(poses) + 1,
                        affinity=float(parts[0]),
                        rmsd_lb=float(parts[1]),
                        rmsd_ub=float(parts[2]),
                    )
                )
            continue
        match = _VINA_ROW.match(line)
        if match:
            poses.append(
                VinaPose(
                    mode=int(match.group(1)),
                    affinity=float(match.group(2)),
                    rmsd_lb=float(match.group(3)),
                    rmsd_ub=float(match.group(4)),
                )
            )
    if not poses:
        raise DockingParseError("no Vina pose rows found in the provided source")
    return sorted(poses, key=lambda p: p.mode)


def check_docking_completion(
    directory: str | Path, log_name: str = "r.dlg"
) -> pd.DataFrame:
    """Audit a docking tree for missing/failed runs (upgrades checker.py).

    Every sub-directory is checked for the expected log file *and* the
    ``Successful Completion`` sentinel that AutoDock4 writes at the end
    of a healthy run.
    """
    base = Path(directory)
    rows = []
    sub_directories = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    for sub in sub_directories:
        log = sub / log_name
        if not log.is_file():
            status = "MISSING_LOG"
        else:
            content = log.read_text(encoding="utf-8", errors="ignore")
            status = "OK" if "Successful Completion" in content else "INCOMPLETE"
        rows.append({"folder": sub.name, "status": status})
    return pd.DataFrame(rows, columns=["folder", "status"])


def scan_docking_directory(
    directory: str | Path, log_name: str = "r.dlg"
) -> pd.DataFrame:
    """Build the docking summary table for every sub-directory.

    Combines the original ``reader.py`` + ``readerwithdetails.py``
    behaviour: one row per ligand folder with the top/best cluster
    columns; folders with missing or unparseable logs get explicit
    status columns instead of crashing the whole scan.
    """
    base = Path(directory)
    if not base.is_dir():
        raise DockingParseError(f"docking directory not found: {base}")

    rows: list[dict] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        log = sub / log_name
        record: dict = {"folder": sub.name}
        if not log.is_file():
            record.update({"status": "MISSING_LOG"})
        else:
            try:
                record.update(parse_dlg_summary(log))
                record.update({"status": "OK"})
            except DockingParseError as exc:
                record.update({"status": f"PARSE_ERROR: {exc}"})
        rows.append(record)
    return pd.DataFrame(rows)
