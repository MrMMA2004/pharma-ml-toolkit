"""DDI qualification: ratios, GMFE, Guest limits, FDA classification.

Absorbs the qualification workflow of
``Open-Systems-Pharmacology/Qualification-DDI-P-gp`` (v1.2) — the
P-gp-mediated digoxin drug–drug-interaction network — plus the DDI
math of ``pkpdutils`` ``stats/ddi.py`` (FDA 2020 classification
thresholds) and the Cheng–Prusoff Ki derivation used by the
Itraconazole model repo:

* :func:`gmfe` — the geometric mean fold error, the qualification
  metric of every OSP evaluation report
  ``GMFE = 10 ^ (1/n · Σ |log10(pred/obs)|)``;
* :func:`guest_et_al_limits` — the Guest et al. (2011) acceptance
  limits as OSP implements them: a cone anchored at unity with slopes
  ``lower = 0.8·R + 0.2`` and ``upper = 1.25·R − 0.25`` (the solid
  lines on every OSP DDI-ratio plot; the dotted ones are the plain
  2-fold criterion, :func:`within_twofold`);
* :func:`ddi_classification` — FDA 2020 / EMA 2012 inhibitor and
  inducer strength categories from the AUC ratio;
* :func:`chen_prusoff` — ``Ki = IC50 / (1 + S/Km)``;
* :data:`P_GP_DDI_RECORDS` — the **real** 34-record digoxin P-gp DDI
  network (observed AUC/Cmax ratios for rifampicin, verapamil,
  erythromycin, clarithromycin and itraconazole as perpetrators),
  transcribed from the ``Database-for-observed-data`` v1.7 ``DDI.csv``
  that the qualification plan keys its ``ObservedDataRecordId``s into.

Examples
--------
>>> from pharmaml.pkpd.ddi import (
...     gmfe, guest_et_al_limits, ddi_classification, chen_prusoff,
...     P_GP_DDI_RECORDS, ddi_network_summary)

GMFE is symmetric in fold-error direction and 1.0 for a perfect model:

>>> round(gmfe([1.0, 1.0], [1.0, 1.0]), 6)
1.0
>>> round(gmfe([2.0], [1.0]), 4), round(gmfe([0.5], [1.0]), 4)
(2.0, 2.0)

Guest limits: a cone through unity — tight at no-effect, the classic
0.8–1.25 band at large effects:

>>> (round(guest_et_al_limits(1.0)[0], 6), round(guest_et_al_limits(1.0)[1], 6))
(1.0, 1.0)
>>> (round(guest_et_al_limits(2.0)[0], 6), round(guest_et_al_limits(2.0)[1], 6))
(1.8, 2.25)

FDA 2020 categories from the AUC ratio:

>>> ddi_classification(1.1), ddi_classification(1.5)
('no DDI', 'weak inhibitor')
>>> ddi_classification(3.0), ddi_classification(7.0)
('moderate inhibitor', 'strong inhibitor')
>>> ddi_classification(0.65), ddi_classification(0.35), ddi_classification(0.1)
('weak inducer', 'moderate inducer', 'strong inducer')

Cheng–Prusoff — exactly how the Itraconazole repo derived keto- and
N-desalkyl-itraconazole Ki values from IC50 (midazolam substrate 1
µmol/L, Km 2.73 µmol/L; IC50 7.0 → Ki 5.12; 0.44 → 0.32):

>>> round(chen_prusoff(ic50=7.0, substrate_conc=1.0, km=2.73), 3)
5.123
>>> round(chen_prusoff(ic50=0.44, substrate_conc=1.0, km=2.73), 3)
0.322

The real 34-record network: 11 rifampicin-PO studies, 3 itraconazole:

>>> summary = ddi_network_summary()
>>> (int(summary[("Rifampicin", "PO")]["n"]), int(summary[("Itraconazole", "PO")]["n"]))
(11, 3)
>>> round(summary[("Rifampicin", "PO")]["median_auc_ratio"], 3)
0.844
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.utils.validation import validate_positive

__all__ = [
    "P_GP_DDI_RECORDS",
    "DDIRecord",
    "chen_prusoff",
    "competitive_inhibition_factor",
    "ddi_classification",
    "ddi_network_summary",
    "gmfe",
    "guest_et_al_limits",
    "qualification_metrics",
    "within_guest",
    "within_twofold",
]

# FDA 2020 drug interaction guidance (= EMA 2012): AUC-ratio thresholds.
NO_DDI_LOW, NO_DDI_HIGH = 0.8, 1.25
INHIBITOR_WEAK, INHIBITOR_MODERATE, INHIBITOR_STRONG = 1.25, 2.0, 5.0
INDUCER_WEAK, INDUCER_MODERATE, INDUCER_STRONG = 0.8, 0.5, 0.2


@dataclass(frozen=True, slots=True)
class DDIRecord:
    """One observed DDI study record (digoxin P-gp network)."""

    record_id: int
    study: str
    perpetrator: str
    route: str
    mechanism: str
    auc_ratio: float | None
    cmax_ratio: float | None
    victim: str = "Digoxin"


def _p_gp_records() -> tuple[DDIRecord, ...]:
    """The 34 digoxin records of the P-gp qualification network.

    Transcribed from ``Database-for-observed-data`` v1.7 (``DDI.csv``)
    at the ``ObservedDataRecordId`` keys listed by the
    ``Qualification-DDI-P-gp`` v1.2 qualification plan; ratios with no
    reported value in the database stay ``None``.
    """
    rows = [
        (191, "Greiner 1999", "Rifampicin", "PO", "Induction", 0.697080292, 0.481481481),
        (193, "Greiner 1999", "Rifampicin", "IV", "Induction", 0.853379152, 0.834008097),
        (227, "Gurley 2008b", "Rifampicin", "PO", "Induction", 0.75308642, 0.615384615),
        (229, "Gurley 2008b", "Clarithromycin", "PO", "Mechanism-based inactivation", 1.465753425, 1.75),
        (271, "Jalava 1997", "Itraconazole", "PO", "Reversible inhibition", 1.68, 1.34),
        (314, "Kirby 2012", "Rifampicin", "PO", "Induction", 0.943396226, None),
        (316, "Kirby 2012", "Rifampicin", "PO", "Induction", 1.149425287, None),
        (337, "Larsen 2007", "Rifampicin", "PO", "Induction", 0.817346939, None),
        (396, "Reitman 2011", "Rifampicin", "PO", "Induction", 1.462, 1.49),
        (397, "Reitman 2011", "Rifampicin", "PO", "Induction", 0.683, 0.693),
        (398, "Reitman 2011", "Rifampicin", "PO", "Induction", 0.977, 0.876),
        (3028, "Tsutsumi 2002", "Erythromycin", "IV", "Reversible inhibition", 0.955, 0.989010989),
        (3029, "Tsutsumi 2002", "Clarithromycin", "IV", "Reversible inhibition", 0.98, 0.989010989),
        (3063, "Rodin 1988", "Verapamil", "PO", "Mechanism-based inactivation", 1.503184713, 1.44),
        (16159, "Johnson et al. 1987", "Verapamil", "IV", "Mechanism-based inactivation", 1.2419, None),
        (16162, "Pedersen et al. 1983", "Verapamil", "IV", "Mechanism-based inactivation", 1.41748127, None),
        (16172, "Wiebe 2020", "Verapamil", "PO", "Mechanism-based inactivation", 1.007, 1.2164),
        (16301, "Wiebe 2020", "Rifampicin", "PO", "Induction", 1.31, 2.1826),
        (16305, "Kurata 2002", "Clarithromycin", "PO", "Reversible inhibition", 0.743, 1.025),
        (16307, "Kurata 2002", "Clarithromycin", "IV", "Reversible inhibition", 0.574, 0.656),
        (16309, "Kurata 2002", "Clarithromycin", "PO", "Reversible inhibition", 0.641, 0.91),
        (16311, "Kurata 2002", "Clarithromycin", "IV", "Reversible inhibition", 0.536, 0.662),
        (16313, "Kurata 2002", "Clarithromycin", "PO", "Reversible inhibition", 0.784, 0.916),
        (16315, "Kurata 2002", "Clarithromycin", "IV", "Reversible inhibition", 0.555, 0.629),
        (16319, "Drescher 2003", "Rifampicin", "IV", "Induction", 0.727, 0.806),
        (16322, "Gurley 2006", "Rifampicin", "PO", "Induction", 0.844, 0.771),
        (16324, "Gurley 2006", "Clarithromycin", "PO", "Reversible inhibition", 1.351, 1.483),
        (16326, "Gurley 2007", "Rifampicin", "PO", "Induction", 0.84, 0.72),
        (16328, "Gurley 2007", "Clarithromycin", "PO", "Reversible inhibition", 1.567, 1.952),
        (16329, "Partanen 1996", "Itraconazole", "PO", "Reversible inhibition", None, 1.634),
        (16330, "Partanen 1996", "Itraconazole", "PO", "Reversible inhibition", None, 1.966),
        (16409, "Rengelshausen 2003", "Clarithromycin", "IV", "Reversible inhibition", 1.185185185, None),
        (16411, "Rengelshausen 2003", "Clarithromycin", "PO", "Reversible inhibition", 1.642857143, 1.833333333),
        (16638, "Pedersen 1981", "Verapamil", "IV", "Mechanism-based inactivation", 1.613336, None),
    ]
    return tuple(DDIRecord(*row) for row in rows)


P_GP_DDI_RECORDS: tuple[DDIRecord, ...] = _p_gp_records()


def gmfe(
    predicted: np.ndarray | list[float],
    observed: np.ndarray | list[float],
) -> float:
    """Geometric mean fold error — the OSP qualification metric.

    ``GMFE = 10 ^ (1/n · Σ |log10(pred/obs)|)`` — 1.0 is a perfect
    match and the metric is symmetric between over- and
    under-prediction (a 2× miss scores 2 either way).  This is the
    number every OSP evaluation report prints next to its goodness-of
    -fit plots.
    """
    p = np.asarray(predicted, dtype=float).reshape(-1)
    o = np.asarray(observed, dtype=float).reshape(-1)
    if p.size == 0 or p.size != o.size:
        raise DataValidationError("predicted and observed must be equal-length, non-empty")
    if np.any(p <= 0) or np.any(o <= 0):
        raise DataValidationError("GMFE needs strictly positive ratios")
    return float(10.0 ** np.mean(np.abs(np.log10(p / o))))


def guest_et_al_limits(observed_ratio: float) -> tuple[float, float]:
    """Guest et al. (2011) acceptance limits for a DDI ratio.

    As drawn (solid lines) on OSP DDI-ratio plots: a cone anchored at
    unity, ``lower = 0.8·R + 0.2`` and ``upper = 1.25·R − 0.25`` —
    exact at ``R = 1`` and relaxing to the 0.8–1.25 band as the effect
    grows, i.e. near-unity interactions must be predicted absolutely
    accurately while large effects tolerate proportional error.
    """
    observed_ratio = validate_positive(observed_ratio, "observed_ratio")
    return (0.8 * observed_ratio + 0.2, 1.25 * observed_ratio - 0.25)


def within_guest(predicted: float, observed: float) -> bool:
    """Whether a predicted ratio sits inside the Guest et al. limits."""
    low, high = guest_et_al_limits(observed)
    return bool(low <= predicted <= high)


def within_twofold(predicted: float, observed: float) -> bool:
    """The 2-fold criterion (dotted lines on OSP DDI plots): 0.5–2.0×."""
    return bool(0.5 * observed <= predicted <= 2.0 * observed)


def ddi_classification(auc_ratio: float) -> str:
    """FDA 2020 / EMA 2012 perpetrator category from the AUC ratio.

    Inhibitors: ``≥ 1.25`` weak, ``≥ 2`` moderate, ``≥ 5`` strong;
    inducers: ``≤ 0.8`` weak, ``≤ 0.5`` moderate, ``≤ 0.2`` strong;
    ``0.8 < ratio < 1.25`` is no DDI (pkpdutils' ``DDIThresholds``,
    which matches the agencies' cut-offs exactly).
    """
    if not math.isfinite(auc_ratio) or auc_ratio <= 0:
        raise DataValidationError(f"auc_ratio must be positive and finite, got {auc_ratio}")
    if auc_ratio >= INHIBITOR_STRONG:
        return "strong inhibitor"
    if auc_ratio >= INHIBITOR_MODERATE:
        return "moderate inhibitor"
    if auc_ratio >= INHIBITOR_WEAK:
        return "weak inhibitor"
    if auc_ratio > NO_DDI_LOW:
        return "no DDI"
    if auc_ratio > INDUCER_MODERATE:
        return "weak inducer"
    if auc_ratio > INDUCER_STRONG:
        return "moderate inducer"
    return "strong inducer"


def chen_prusoff(*, ic50: float, substrate_conc: float, km: float) -> float:
    """Cheng–Prusoff: Ki from a competitive IC50.

    ``Ki = IC50 / (1 + S/Km)`` with ``S`` the substrate concentration
    at which the IC50 was measured — exactly the derivation the
    Itraconazole model repo documents for keto- (IC50 7.0 nM → Ki
    5.12 nM) and N-desalkyl-itraconazole (0.44 → 0.32) with midazolam
    at 1 µmol/L, Km 2.73 µmol/L.
    """
    ic50 = validate_positive(ic50, "ic50")
    substrate_conc = validate_positive(substrate_conc, "substrate_conc")
    km = validate_positive(km, "km")
    return ic50 / (1.0 + substrate_conc / km)


def competitive_inhibition_factor(
    inhibitor_conc: float, ki: float
) -> float:
    """The ``1 / (1 + I/Ki)`` intrinsic-clearance multiplier.

    Under competitive inhibition the apparent ``CLint`` shrinks by
    this factor; the AUC ratio of an intravenously dosed victim with
    purely hepatic (well-stirred) clearance is bounded by it.
    """
    inhibitor_conc = validate_positive(inhibitor_conc, "inhibitor_conc")
    ki = validate_positive(ki, "ki")
    return 1.0 / (1.0 + inhibitor_conc / ki)


def ddi_network_summary(
    records: tuple[DDIRecord, ...] = P_GP_DDI_RECORDS,
) -> dict[tuple[str, str], dict[str, float]]:
    """Group the network by (perpetrator, victim route).

    Returns one entry per group with the study count and the median /
    range of the observed AUC ratios — the aggregation the
    qualification report's DDI-ratio table is built from.
    """
    groups: dict[tuple[str, str], list[DDIRecord]] = {}
    for record in records:
        groups.setdefault((record.perpetrator, record.route), []).append(record)
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, rows in groups.items():
        ratios = [r.auc_ratio for r in rows if r.auc_ratio is not None]
        entry: dict[str, float] = {"n": float(len(rows))}
        if ratios:
            entry["median_auc_ratio"] = float(np.median(ratios))
            entry["min_auc_ratio"] = float(min(ratios))
            entry["max_auc_ratio"] = float(max(ratios))
        cmax = [r.cmax_ratio for r in rows if r.cmax_ratio is not None]
        if cmax:
            entry["median_cmax_ratio"] = float(np.median(cmax))
        summary[key] = entry
    return summary


def qualification_metrics(
    observed: np.ndarray | list[float],
    predicted: np.ndarray | list[float],
) -> dict[str, float]:
    """The OSP DDI-qualification scorecard for one ratio set.

    GMFE plus the share of predictions inside the 2-fold band and
    inside the Guest et al. (2011) limits — the exact acceptance
    counting the P-gp qualification workflow reports per
    Mechanism/Perpetrator group.
    """
    o = np.asarray(observed, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if o.size == 0 or o.size != p.size:
        raise DataValidationError("observed and predicted must match and be non-empty")
    n = float(o.size)
    two_fold = sum(1.0 for pi, oi in zip(p, o) if within_twofold(float(pi), float(oi)))
    guest = sum(1.0 for pi, oi in zip(p, o) if within_guest(float(pi), float(oi)))
    return {
        "n": n,
        "gmfe": gmfe(p, o),
        "fraction_within_twofold": two_fold / n,
        "fraction_within_guest": guest / n,
    }
