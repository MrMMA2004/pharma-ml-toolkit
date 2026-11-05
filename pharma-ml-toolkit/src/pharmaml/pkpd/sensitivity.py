"""OSPSuite-R semantics: PK registry, sensitivity, individual characteristics.

Round 176's absorption of **OSPSuite-R** (Open-Systems-Pharmacology's
R API over the .NET OSPSuite core).  The R package is a thin,
well-designed marshaling layer — every heavy computation lives in
``inst/lib/*.dll`` — so exactly its *data model and conventions*
transfer:

* :class:`StandardPKParameter` + :data:`PK_PARAMETER_REGISTRY` — the
  31-entry PK-parameter table shipped in
  ``inst/lib/OSPSuite.PKParameters.xml`` (name, mode Always/Single/
  Multi, dimension — including the multi-dose ``tD1_tD2``/``tDLast``
  variants), plus the integer codes of the R ``StandardPKParameter``
  enum (0–22, ``C_max=1`` … ``Tthreshold=22``);
* :func:`sensitivity_analysis` — the OSP-Suite sensitivity workflow
  with its exact defaults (``numberOfSteps=2``, ``variationRange=
  0.1`` → the parameter runs at 90 %, 95 %, 105 %, 110 % of its
  reference) and its *normalised* value convention: −1 means "+10 %
  parameter → −10 % PK parameter" (the elasticity
  ``d ln(PK) / d ln(p)``), reported in the ``sa.csv`` schema
  (QuantityPath / Parameter / PKParameter / Value) with the
  cumulative-``totalSensitivityThreshold=0.9`` ranking filter of
  ``allPKParameterSensitivitiesFor`` (descending |value|);
* :class:`IndividualCharacteristics` +
  :func:`create_individual_characteristics` — the R layer's species /
  population / gender enums (8 species, 8 human populations), the
  value-plus-unit covariate pairs (``weightUnit="kg"``,
  ``heightUnit="cm"``, ``ageUnit="year(s)"``, gestational age
  defaulting to 40 weeks), and the validation rules (Human requires
  a population; unknown species refused with the list);
* :func:`match_paths` — the OSP-Suite pipe-path addressing with the
  ``*`` (one level) and ``**`` (recursive) wildcards, so parameter
  paths like ``Organism|Liver|Volume`` can be addressed the way the
  R API's ``getAllParametersMatching`` does.

Honest boundaries: the .NET sensitivity *solver* (CVODES adjoints),
the correlated population sampler inside the PKSimDB-backed population
factory (the seed-in/seed-out reproducibility contract transfers; the
covariance machinery does not), the user-defined-PK-parameter
dimension inference, and the whole ``rSharp`` marshaling layer.

Examples
--------
The registry — 31 parameters, the modes, the multi-dose variants:

>>> from pharmaml.pkpd.sensitivity import (
...     PK_PARAMETER_REGISTRY, StandardPKParameter, pk_parameter_frame)
>>> len(PK_PARAMETER_REGISTRY)
31
>>> int(StandardPKParameter.C_max), int(StandardPKParameter.CL)
(1, 16)
>>> PK_PARAMETER_REGISTRY["AUC_inf"].mode, PK_PARAMETER_REGISTRY["C_max"].mode
('Single', 'Always')
>>> any(name.startswith("C_max_tD") for name in PK_PARAMETER_REGISTRY)
True
>>> frame = pk_parameter_frame()
>>> frame.loc["Thalf", "dimension"]
'Time'

Sensitivity — a one-compartment IV AUC is ``D/CL``: the elasticity to
CL is exactly −1 and to V exactly 0 (the registry convention: −1 means
+10 % in → −10 % out):

>>> from pharmaml.pkpd.sensitivity import sensitivity_analysis
>>> def auc(params):
...     return 100.0 / params["CL"]
>>> res = sensitivity_analysis(auc, {"CL": 5.0, "V": 20.0}, "AUC_inf")
>>> round(res.value_for("CL"), 6)
-1.0
>>> round(res.value_for("V"), 6)
0.0

The cumulative-threshold filter keeps the parameters that carry 90 %
of the total |sensitivity|:

>>> kept = res.thresholded()
>>> list(kept["Parameter"])
['CL']

Individual characteristics — the validation rules and defaults:

>>> from pharmaml.pkpd.sensitivity import create_individual_characteristics
>>> create_individual_characteristics("Human")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: species 'Human' requires a population — one of: Asian_Tanaka_1996, BlackAmerican_NHANES_1997, European_ICRP_2002, Japanese_Population, MexicanAmericanWhite_NHANES_1997, Pregnant, Preterm, WhiteAmerican_NHANES_1997
>>> ic = create_individual_characteristics(
...     "Human", population="European_ICRP_2002", gender="FEMALE",
...     age=43.0, weight=75.0, height=175.0)
>>> (ic.age.value, ic.age.unit, ic.gestational_age.value)
(43.0, 'year(s)', 40.0)

Path wildcards:

>>> from pharmaml.pkpd.sensitivity import match_paths
>>> paths = ["Organism|Weight", "Organism|Liver|Volume",
...          "Organism|Liver|Plasma|Volume"]
>>> match_paths("Organism|*|Volume", paths)
['Organism|Liver|Volume']
>>> match_paths("Organism|**|Volume", paths)
['Organism|Liver|Volume', 'Organism|Liver|Plasma|Volume']
>>> match_paths("Organism|Weight", paths)
['Organism|Weight']
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "PK_PARAMETER_REGISTRY",
    "SPECIES",
    "Gender",
    "HumanPopulation",
    "IndividualCharacteristics",
    "PKParameterEntry",
    "StandardPKParameter",
    "create_individual_characteristics",
    "match_paths",
    "pk_parameter_frame",
    "sensitivity_analysis",
]

# ---- the R enums (OSPSuite-R R/enums.R) --------------------------------

#: Species the R layer accepts (``enums.R::Species``).
SPECIES = (
    "Beagle",
    "Dog",
    "Human",
    "Minipig",
    "Monkey",
    "Mouse",
    "Rabbit",
    "Rat",
)

#: Human populations (``enums.R::HumanPopulation``) — the PKSimDB
#: population tables the .NET individual factory draws from.
HumanPopulation = (
    "Asian_Tanaka_1996",
    "BlackAmerican_NHANES_1997",
    "European_ICRP_2002",
    "Japanese_Population",
    "MexicanAmericanWhite_NHANES_1997",
    "Pregnant",
    "Preterm",
    "WhiteAmerican_NHANES_1997",
)

#: Gender values as the R layer stores them (``enums.R::Gender``).
Gender = ("FEMALE", "MALE", "UNKNOWN")


class StandardPKParameter(IntEnum):
    """The R ``StandardPKParameter`` codes (``R/enums.R:198-222``)."""

    Unknown = 0
    C_max = 1
    C_max_norm = 2
    C_min = 3
    C_min_norm = 4
    t_max = 5
    t_min = 6
    C_trough = 7
    C_trough_norm = 8
    AUC_tEnd = 9
    AUC_tEnd_norm = 10
    AUCM_tEnd = 11
    AUC_inf = 12
    AUC_inf_norm = 13
    AUC_tEnd_inf = 14
    AUC_tEnd_inf_norm = 15
    CL = 16
    MRT = 17
    FractionAucEndToInf = 18
    Thalf = 19
    Vss = 20
    Vd = 21
    Tthreshold = 22


@dataclass(frozen=True, slots=True)
class PKParameterEntry:
    """One row of ``OSPSuite.PKParameters.xml``."""

    name: str
    display_name: str
    mode: str  # Always | Single | Multi
    dimension: str


#: The 31-entry registry shipped in ``inst/lib/OSPSuite.PKParameters.xml``
#: (counted, not guessed: ``len(PK_PARAMETER_REGISTRY) == 31``).
PK_PARAMETER_REGISTRY: dict[str, PKParameterEntry] = {
    e.name: e
    for e in [
        PKParameterEntry("C_max", "C_max", "Always", "Concentration (molar)"),
        PKParameterEntry("C_max_norm", "C_max_norm", "Always", "Concentration (mass)"),
        PKParameterEntry("C_max_tD1_tD2", "C_max_tD1-tD2", "Multi", "Concentration (molar)"),
        PKParameterEntry("C_max_tD1_tD2_norm", "C_max_tD1-tD2_norm", "Multi", "Concentration (mass)"),
        PKParameterEntry("C_max_tDLast_tEnd", "C_max_tDlast-tEnd", "Multi", "Concentration (molar)"),
        PKParameterEntry("C_max_tDLast_tEnd_norm", "C_max_tDLast-tEnd_norm", "Multi", "Concentration (mass)"),
        PKParameterEntry("t_max", "t_max", "Always", "Time"),
        PKParameterEntry("t_max_tD1_tD2", "t_max_tD1-tD2", "Multi", "Time"),
        PKParameterEntry("t_max_tDLast_tEnd", "t_max_tDlast-tEnd", "Multi", "Time"),
        PKParameterEntry("C_tEnd", "C_tEnd", "Always", "Concentration (molar)"),
        PKParameterEntry("C_trough_tD2", "C_trough_tD2", "Multi", "Concentration (molar)"),
        PKParameterEntry("C_trough_tDLast", "C_trough_tDLast", "Multi", "Concentration (molar)"),
        PKParameterEntry("AUC_tEnd", "AUC_tEnd", "Always", "AUC (molar)"),
        PKParameterEntry("AUC_tEnd_norm", "AUC_tEnd_norm", "Always", "AUC (mass)"),
        PKParameterEntry("AUC_inf", "AUC_inf", "Single", "AUC (molar)"),
        PKParameterEntry("AUC_inf_norm", "AUC_inf_norm", "Single", "AUC (mass)"),
        PKParameterEntry("AUC_tD1_tD2", "AUC_tD1-tD2", "Multi", "AUC (molar)"),
        PKParameterEntry("AUC_tD1_tD2_norm", "AUC_tD1-tD2_norm", "Multi", "AUC (mass)"),
        PKParameterEntry("AUC_tDLast_minus_1_tDLast", "AUC_tDLast_minus_1-tDLast", "Multi", "AUC (molar)"),
        PKParameterEntry("AUC_tDLast_minus_1_tDLast_norm", "AUC_tDLast_minus_1-tDLast_norm", "Multi", "AUC (mass)"),
        PKParameterEntry("AUC_inf_tD1", "AUC_inf_tD1", "Multi", "AUC (molar)"),
        PKParameterEntry("AUC_inf_tD1_norm", "AUC_inf_tD1_norm", "Multi", "AUC (mass)"),
        PKParameterEntry("AUC_inf_tDLast", "AUC_inf_tDLast", "Multi", "AUC (molar)"),
        PKParameterEntry("AUC_inf_tDLast_norm", "AUC_inf_tDLast_norm", "Multi", "AUC (mass)"),
        PKParameterEntry("MRT", "MRT", "Always", "Time"),
        PKParameterEntry("Thalf", "Thalf", "Always", "Time"),
        PKParameterEntry("Thalf_tDLast_tEnd", "Thalf_tDlast-tEnd", "Multi", "Time"),
        PKParameterEntry("FractionAucLastToInf", "FractionAucLastToInf", "Single", "Fraction"),
        PKParameterEntry("CL", "CL", "Single", "Flow per weight"),
        PKParameterEntry("Vss", "Vss", "Single", "Volume per body weight"),
        PKParameterEntry("Vd", "Vd", "Single", "Volume per body weight"),
    ]
}


def pk_parameter_frame() -> pd.DataFrame:
    """The registry as a DataFrame indexed by parameter name."""
    return pd.DataFrame(
        [
            {
                "name": e.name,
                "displayName": e.display_name,
                "mode": e.mode,
                "dimension": e.dimension,
            }
            for e in PK_PARAMETER_REGISTRY.values()
        ]
    ).set_index("name")


@dataclass(slots=True)
class SensitivityAnalysisResult:
    """Sensitivity output in the ``sa.csv`` schema.

    Columns: ``QuantityPath``, ``Parameter``, ``PKParameter``,
    ``Value`` — the CSV schema of OSPSuite-R's sensitivity export, one
    row per varied parameter.
    """

    frame: pd.DataFrame
    output_path: str
    pk_parameter: str
    number_of_steps: int
    variation_range: float

    def value_for(self, parameter: str) -> float:
        """The normalised sensitivity of one parameter (NaN→error)."""
        rows = self.frame[self.frame["Parameter"] == parameter]
        if rows.empty:
            known = ", ".join(self.frame["Parameter"])
            raise DataValidationError(
                f"no sensitivity for parameter {parameter!r} — varied: {known}"
            )
        return float(rows["Value"].iloc[0])

    def thresholded(self, total_sensitivity_threshold: float = 0.9) -> pd.DataFrame:
        """``allPKParameterSensitivitiesFor``'s cumulative-share filter.

        Ranks by |value| descending and keeps rows until the
        cumulative share of |sensitivity| crosses
        ``total_sensitivity_threshold`` (the OSP default 0.9) — the
        row that crosses the threshold is included.
        """
        if not 0 < total_sensitivity_threshold <= 1:
            raise DataValidationError(
                "total_sensitivity_threshold must be in (0, 1], got "
                f"{total_sensitivity_threshold}"
            )
        ranked = self.frame.reindex(
            self.frame["Value"].abs().sort_values(ascending=False).index
        )
        total = float(ranked["Value"].abs().sum())
        if total <= 0:
            return ranked.iloc[0:0]
        share = ranked["Value"].abs().cumsum() / total
        keep = (share - ranked["Value"].abs() / total) < total_sensitivity_threshold
        return ranked[keep]


def sensitivity_analysis(
    simulation,
    parameters: dict[str, float],
    pk_parameter: str = "AUC_inf",
    *,
    output_path: str = "Organism|PeripheralVenousBlood|Drug",
    number_of_steps: int = 2,
    variation_range: float = 0.1,
) -> SensitivityAnalysisResult:
    """Normalised parameter sensitivities — the OSP-Suite workflow.

    ``simulation(params) -> float`` maps a parameter dictionary to the
    PK metric (compute the AUC, Cmax … however your model runs).  Each
    parameter is varied symmetrically at
    ``ref·(1 ± k·variation_range/number_of_steps)`` — with the OSP
    defaults (2 steps, range 0.1) exactly the 90 %, 95 %, 105 % and
    110 % runs — and the reported value is the elasticity
    ``d ln(PK) / d ln(p)`` from a log-log regression through the
    varied points and the reference: the convention the R vignette
    documents as "−1 means +10 % parameter → −10 % PK parameter".

    Reference values must be positive (the variation is
    multiplicative); the simulation must return a positive metric
    (log-space regression).
    """
    if number_of_steps < 1:
        raise DataValidationError(
            f"number_of_steps must be >= 1, got {number_of_steps}"
        )
    if not 0 < variation_range < 1:
        raise DataValidationError(
            f"variation_range must be in (0, 1), got {variation_range}"
        )
    if not parameters:
        raise DataValidationError("parameters must not be empty")
    for name, ref in parameters.items():
        if ref <= 0:
            raise DataValidationError(
                f"reference value for {name!r} must be > 0, got {ref}"
            )

    offsets = [
        1.0 + sign * variation_range * k / number_of_steps
        for k in range(1, number_of_steps + 1)
        for sign in (-1.0, 1.0)
    ]
    rows: list[dict[str, object]] = []
    for name, ref in parameters.items():
        factors = [1.0] + offsets
        values = []
        for factor in factors:
            params = dict(parameters)
            params[name] = ref * factor
            out = float(simulation(params))
            if not math.isfinite(out) or out <= 0:
                raise DataValidationError(
                    f"simulation returned {out} for {name}={ref * factor} — "
                    "the sensitivity regression needs positive outputs"
                )
            values.append(out)
        x = np.log(np.asarray(factors) * ref)
        y = np.log(np.asarray(values))
        if float(y.max() - y.min()) == 0.0:
            slope = 0.0
        else:
            slope = float(np.polyfit(x, y, 1)[0])
        rows.append(
            {
                "QuantityPath": output_path,
                "Parameter": name,
                "PKParameter": pk_parameter,
                "Value": slope,
            }
        )
    frame = pd.DataFrame.from_records(rows, columns=["QuantityPath", "Parameter", "PKParameter", "Value"])
    return SensitivityAnalysisResult(
        frame=frame,
        output_path=output_path,
        pk_parameter=pk_parameter,
        number_of_steps=number_of_steps,
        variation_range=variation_range,
    )


# ---- individual characteristics (R/individual-characteristics.R) --------


@dataclass(frozen=True, slots=True)
class SnapshotParameter:
    """A value-plus-unit covariate pair (the R ``SnapshotParameter``)."""

    value: float
    unit: str


@dataclass(frozen=True, slots=True)
class IndividualCharacteristics:
    """The R ``IndividualCharacteristics`` — covariates for one virtual
    individual, with the seed-in/seed-out reproducibility slot the R
    population API carries."""

    species: str
    population: str | None
    gender: str | None
    weight: SnapshotParameter | None
    height: SnapshotParameter | None
    age: SnapshotParameter | None
    gestational_age: SnapshotParameter
    molecule_ontogenies: tuple[str, ...] = ()
    seed: int | None = None


def create_individual_characteristics(
    species: str,
    population: str | None = None,
    *,
    gender: str | None = None,
    weight: float | None = None,
    height: float | None = None,
    age: float | None = None,
    gestational_age: float = 40.0,
    molecule_ontogenies: list[str] | None = None,
    seed: int | None = None,
) -> IndividualCharacteristics:
    """Build and validate one individual — the R validation rules.

    ``species`` must be one of the eight the R layer accepts; **Human
    additionally requires a population** from
    :data:`HumanPopulation` (``utilities-individual.R:146-151``).
    Units follow the R defaults — ``kg``, ``cm``, ``year(s)`` and
    ``week(s)`` — and gestational age defaults to 40 weeks exactly as
    the R signature does.  Animal species carry no population; the
    derived anthropometrics come from the PK-Sim curves via
    :func:`pharmaml.pkpd.physiology.create_individual`.
    """
    if species not in SPECIES:
        known = ", ".join(SPECIES)
        raise DataValidationError(
            f"unknown species {species!r} — the R layer accepts: {known}"
        )
    if species == "Human":
        if population is None:
            known = ", ".join(HumanPopulation)
            raise DataValidationError(
                f"species 'Human' requires a population — one of: {known}"
            )
        if population not in HumanPopulation:
            known = ", ".join(HumanPopulation)
            raise DataValidationError(
                f"unknown human population {population!r} — one of: {known}"
            )
    if gender is not None and gender not in Gender:
        known = ", ".join(Gender)
        raise DataValidationError(
            f"gender must be one of {known}, got {gender!r}"
        )
    for name, value in (
        ("weight", weight),
        ("height", height),
        ("age", age),
    ):
        if value is not None and value <= 0:
            raise DataValidationError(f"{name} must be > 0, got {value}")
    if gestational_age <= 0:
        raise DataValidationError(
            f"gestational_age must be > 0, got {gestational_age}"
        )
    onts = tuple(molecule_ontogenies or ())
    return IndividualCharacteristics(
        species=species,
        population=population,
        gender=gender,
        weight=None if weight is None else SnapshotParameter(weight, "kg"),
        height=None if height is None else SnapshotParameter(height, "cm"),
        age=None if age is None else SnapshotParameter(age, "year(s)"),
        gestational_age=SnapshotParameter(gestational_age, "week(s)"),
        molecule_ontogenies=onts,
        seed=seed,
    )


# ---- path addressing (R/utilities-parameter.R) ---------------------------


def match_paths(pattern: str, paths: list[str] | tuple[str, ...]) -> list[str]:
    """Pipe-path matching with the ``*`` / ``**`` wildcards.

    ``*`` matches exactly one path level (``Organism|*|Volume`` hits
    ``Organism|Liver|Volume`` but not ``Organism|Liver|Plasma|
    Volume``); ``**`` matches one or more levels recursively.  Paths
    keep their input order.
    """
    if not pattern:
        raise DataValidationError("pattern must not be empty")
    segments = pattern.split("|")
    regex_parts: list[str] = []
    for seg in segments:
        if seg == "**":
            regex_parts.append(r"(?:[^|]+(?:\|[^|]+)*)")
        else:
            regex_parts.append(r"[^|]*".join(seg.split("*")) if "*" in seg else seg)
    regex = r"\|".join(regex_parts)
    compiled = re.compile("^" + regex + "$")
    return [p for p in paths if compiled.match(p) is not None]
