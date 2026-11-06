"""Offline DOCKSTRING target boxes and original benchmark objectives.

``dockstring/dockstring`` wraps ligand standardization, Open Babel, Vina,
pose verification and 58 receptor/search-box resources.  The toolkit already
has those preparation/execution/verification layers; this module fills the
catalog's missing *benchmark* boundary without duplicating the engine:

* all 58 upstream ``*_conf.txt`` boxes are transcribed and exposed through
  :func:`get_target` / :func:`target_frame`;
* :func:`parse_search_box` reads the same six-key config grammar;
* :func:`f2_objective`, :func:`promiscuous_ppar_objective` and
  :func:`selective_jak2_objective` port ``dockstring/benchmarks/original.py``
  including its corrected selective-JAK2 equation (``LCK median = -8.1``)
  and ``10 * (1-QED)`` penalty.

Receptor PDBQT files, Vina itself and the 260k-ligand DOCKSTRING dataset are
not bundled.  Scores are user/external-engine inputs; a target-table entry
is not a docking result or scientific validation.  NaN scores propagate just
as upstream's ``safe_dock_function`` contract requires.

Examples
--------
>>> from pharmaml.docking.dockstring import get_target, selective_jak2_objective
>>> get_target("DRD2").size
(30.0, 30.0, 32.0)
>>> round(selective_jak2_objective(jak2=-9.0, lck=-7.0, qed=0.8), 3)
-7.0
>>> round(selective_jak2_objective(jak2=-9.0, lck=-9.0, qed=0.8), 3)
-6.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "DOCKSTRING_OBJECTIVES",
    "DOCKSTRING_TARGETS",
    "DockingTarget",
    "evaluate_objective",
    "f2_objective",
    "get_target",
    "parse_search_box",
    "promiscuous_ppar_objective",
    "qed_penalty",
    "selective_jak2_objective",
    "target_frame",
]

_DOCKING_KEYS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")


@dataclass(frozen=True, slots=True)
class DockingTarget:
    """One DOCKSTRING receptor name and Vina search box (angstrom)."""

    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    def config_text(self) -> str:
        values = (*self.center, *self.size)
        return "\n".join(
            f"{key} = {value:.3f}" for key, value in zip(_DOCKING_KEYS, values)
        ) + "\n"


def parse_search_box(text: str) -> DockingTarget:
    """Parse a DOCKSTRING/Vina six-key config into an unnamed target box."""
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError("search-box config must be non-empty text")
    values: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise DataValidationError(f"invalid search-box line: {raw!r}")
        key, raw_value = (part.strip() for part in line.split("=", 1))
        if key not in _DOCKING_KEYS:
            raise DataValidationError(f"unknown search-box key {key!r}")
        if key in values:
            raise DataValidationError(f"duplicate search-box key {key!r}")
        try:
            values[key] = float(raw_value)
        except ValueError as exc:
            raise DataValidationError(f"search-box {key} must be numeric") from exc
    missing = [key for key in _DOCKING_KEYS if key not in values]
    if missing:
        raise DataValidationError("search-box config is missing: " + ", ".join(missing))
    if not all(np.isfinite(values[key]) for key in _DOCKING_KEYS):
        raise DataValidationError("search-box values must be finite")
    if any(values[key] <= 0.0 for key in ("size_x", "size_y", "size_z")):
        raise DataValidationError("search-box sizes must be positive")
    return DockingTarget(
        name="custom",
        center=(values["center_x"], values["center_y"], values["center_z"]),
        size=(values["size_x"], values["size_y"], values["size_z"]),
    )


def _entries() -> dict[str, DockingTarget]:
    return {
        name: DockingTarget(name, values[:3], values[3:])
        for name, values in _TARGET_DATA.items()
    }


def get_target(name: str) -> DockingTarget:
    """Case-sensitive upstream target lookup with an informative refusal."""
    if not isinstance(name, str) or name not in DOCKSTRING_TARGETS:
        raise DataValidationError(
            f"unknown DOCKSTRING target {name!r}; choose one of {len(DOCKSTRING_TARGETS)} indexed targets"
        )
    return DOCKSTRING_TARGETS[name]


def target_frame() -> pd.DataFrame:
    """The target registry as a flat table suitable for export/display."""
    return pd.DataFrame(
        [
            {
                "target": target.name,
                "center_x": target.center[0],
                "center_y": target.center[1],
                "center_z": target.center[2],
                "size_x": target.size[0],
                "size_y": target.size[1],
                "size_z": target.size[2],
            }
            for target in DOCKSTRING_TARGETS.values()
        ]
    )


def _qed(value: Any) -> float:
    try:
        qed = float(value)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("QED must be numeric") from exc
    if not np.isfinite(qed) or not 0.0 <= qed <= 1.0:
        raise DataValidationError("QED must be finite and between 0 and 1")
    return qed


def qed_penalty(qed: float) -> float:
    """DOCKSTRING's ``10 * (1-QED)`` optimization penalty."""
    return 10.0 * (1.0 - _qed(qed))


def f2_objective(*, f2: float, qed: float) -> float:
    """Original F2 optimization objective (lower is better)."""
    return float(f2) + qed_penalty(qed)


def promiscuous_ppar_objective(
    *, ppara: float, ppard: float, pparg: float, qed: float
) -> float:
    """Worst (maximum) PPAR score plus QED penalty; any NaN propagates."""
    values = [float(ppara), float(ppard), float(pparg)]
    if any(np.isnan(value) for value in values):
        return float("nan")
    return max(values) + qed_penalty(qed)


def selective_jak2_objective(*, jak2: float, lck: float, qed: float) -> float:
    """Corrected selective-JAK2 objective from upstream ``original.py``."""
    lck_median_score = -8.1
    return float(jak2) - min(float(lck) - lck_median_score, 0.0) + qed_penalty(qed)


DOCKSTRING_OBJECTIVES: tuple[str, ...] = ("F2", "promiscuous_PPAR", "selective_JAK2")


def evaluate_objective(name: str, scores: dict[str, float], *, qed: float) -> float:
    """Evaluate one named original objective from precomputed docking scores."""
    if name == "F2":
        if "F2" not in scores:
            raise DataValidationError("F2 objective requires an F2 docking score")
        return f2_objective(f2=scores["F2"], qed=qed)
    if name == "promiscuous_PPAR":
        missing = [key for key in ("PPARA", "PPARD", "PPARG") if key not in scores]
        if missing:
            raise DataValidationError("promiscuous PPAR requires: " + ", ".join(missing))
        return promiscuous_ppar_objective(
            ppara=scores["PPARA"], ppard=scores["PPARD"], pparg=scores["PPARG"], qed=qed
        )
    if name == "selective_JAK2":
        missing = [key for key in ("JAK2", "LCK") if key not in scores]
        if missing:
            raise DataValidationError("selective JAK2 requires: " + ", ".join(missing))
        return selective_jak2_objective(jak2=scores["JAK2"], lck=scores["LCK"], qed=qed)
    raise DataValidationError(
        f"unknown DOCKSTRING objective {name!r}; choose from {', '.join(DOCKSTRING_OBJECTIVES)}"
    )


# name: center_x, center_y, center_z, size_x, size_y, size_z
_TARGET_DATA: dict[str, tuple[float, float, float, float, float, float]] = {
    "ABL1": (15.851, 14.647, 3.904, 30.000, 30.000, 30.000),
    "ACHE": (4.521, 66.834, 65.770, 30.000, 30.000, 30.000),
    "ADAM17": (44.294, 28.123, 2.617, 30.000, 30.000, 30.000),
    "ADORA2A": (-9.373, -6.808, 55.570, 30.000, 30.000, 30.000),
    "ADRB1": (26.115, 4.496, 0.946, 30.000, 30.000, 30.000),
    "ADRB2": (2.145, 4.681, 51.903, 30.000, 30.000, 30.000),
    "AKT1": (5.507, 3.383, 18.721, 30.000, 30.000, 30.000),
    "AKT2": (22.502, -19.622, 7.413, 30.000, 30.000, 30.000),
    "AR": (27.060, 2.355, 4.947, 30.000, 30.000, 30.000),
    "BACE1": (24.495, 12.617, 22.123, 30.000, 30.000, 30.000),
    "CA2": (-5.846, 1.074, 16.803, 30.000, 30.000, 30.000),
    "CASP3": (35.109, 35.886, 33.909, 30.000, 30.000, 30.000),
    "CDK2": (1.633, 26.264, 9.276, 30.000, 30.000, 30.000),
    "CSF1R": (6.620, 30.425, 9.393, 30.000, 30.000, 30.000),
    "CYP2C9": (7.952, 32.818, -1.085, 30.000, 30.000, 30.000),
    "CYP3A4": (37.219, -14.579, 30.512, 30.000, 30.000, 30.000),
    "DHFR": (14.554, 5.102, 0.063, 30.000, 30.000, 30.000),
    "DPP4": (-18.300, -5.122, 59.878, 30.000, 30.000, 30.000),
    "DRD2": (9.250, 6.167, -7.000, 30.000, 30.000, 32.000),
    "DRD3": (8.970, 21.132, 24.193, 30.000, 30.000, 30.000),
    "EGFR": (16.837, 35.647, 91.071, 30.000, 30.000, 30.000),
    "ESR1": (30.890, -1.272, 23.403, 30.000, 30.000, 30.000),
    "ESR2": (95.847, 20.242, -4.008, 30.000, 30.000, 30.000),
    "F10": (2.383, -7.418, -12.173, 30.000, 30.000, 30.000),
    "F2": (17.261, -11.957, 21.143, 30.000, 30.000, 30.000),
    "FGFR1": (6.174, 3.432, 18.060, 30.000, 30.000, 30.000),
    "GBA": (36.269, 31.750, 0.529, 30.000, 30.000, 30.000),
    "HMGCR": (-16.274, 7.821, 44.913, 30.000, 30.000, 30.000),
    "HSD11B1": (55.248, 24.118, 38.979, 30.000, 30.000, 30.000),
    "HSP90AA1": (2.681, 12.958, 25.366, 30.000, 30.000, 30.000),
    "IGF1R": (6.095, -7.742, 21.199, 30.000, 30.000, 30.000),
    "JAK2": (114.833, 65.944, 11.536, 30.000, 30.000, 30.000),
    "KDR": (37.081, 36.130, 11.946, 30.000, 30.000, 30.000),
    "KIT": (34.300, -3.274, -27.964, 30.000, 30.000, 30.000),
    "LCK": (27.073, 39.585, 84.598, 30.000, 30.000, 30.000),
    "MAOB": (52.632, 153.977, 25.671, 30.000, 30.000, 30.000),
    "MAP2K1": (-0.988, 58.673, 32.499, 30.000, 30.000, 30.000),
    "MAPK1": (-13.754, 14.001, 41.802, 30.000, 30.000, 30.000),
    "MAPK14": (-2.061, -1.304, 24.206, 30.000, 30.000, 30.000),
    "MAPKAPK2": (37.535, 5.789, 16.718, 30.000, 30.000, 30.000),
    "MET": (-1.518, 4.049, 27.008, 30.000, 30.000, 30.000),
    "MMP13": (-5.023, 27.290, 53.665, 30.000, 30.000, 30.000),
    "NOS1": (1.235, 0.125, 59.832, 30.000, 30.000, 30.000),
    "NR3C1": (40.537, 31.424, 9.113, 30.000, 30.000, 30.000),
    "PARP1": (26.835, 11.332, 27.744, 30.000, 30.000, 30.000),
    "PDE5A": (1.678, 68.311, 83.835, 30.000, 30.000, 30.000),
    "PGR": (-2.532, -8.387, 25.099, 30.000, 30.000, 30.000),
    "PLK1": (0.141, 23.081, 66.902, 30.000, 30.000, 30.000),
    "PPARA": (15.171, 39.892, 27.766, 30.000, 30.000, 30.000),
    "PPARD": (16.075, 2.845, 39.221, 30.000, 30.000, 30.000),
    "PPARG": (3.239, 24.243, 16.108, 30.000, 30.000, 30.000),
    "PTGS2": (30.101, -23.094, -15.400, 30.000, 30.000, 30.000),
    "PTK2": (9.656, 2.763, 5.578, 30.000, 30.000, 30.000),
    "PTPN1": (46.162, 13.754, 3.076, 30.000, 30.000, 30.000),
    "REN": (-8.249, -14.835, -9.852, 30.000, 30.000, 30.000),
    "ROCK1": (52.520, 101.331, 29.676, 30.000, 30.000, 30.000),
    "SRC": (1.792, 5.091, 6.855, 30.000, 30.000, 30.000),
    "THRB": (13.230, 23.118, 46.569, 30.000, 30.000, 30.000),
}

DOCKSTRING_TARGETS: dict[str, DockingTarget] = _entries()
