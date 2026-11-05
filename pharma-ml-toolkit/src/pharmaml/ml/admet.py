"""The ADMET-AI endpoint registry and scorecard (100x-175).

``swansonk14/admet_ai`` (v2.0.1) was reviewed file by file for this
module: ``resources/data/admet.csv`` (the 52-row endpoint table -- 11
physicochemical properties plus 41 TDC-trained ADMET endpoints with
task type, units, clipping bounds, species and per-endpoint model
quality), ``admet_model.py`` (the DrugBank percentile contextualisation
and 5-seed ensemble averaging), ``physchem.py`` (the 11-property panel
with PAINS/BRENK/NIH structural-alert counts), ``plot.py`` (the radial
pentagon summary with its directionality rules), ``constants.py`` /
``web/app/config.py`` (``LOW_PERFORMANCE_THRESHOLD = 0.6``) and
``web/app/templates/index.html`` (the display-clipping contract:
predictions are clipped to the table's min/max for display only).

Vendored verbatim from the repository's own resource files:

* :data:`ADMET_ENDPOINTS` -- all 52 rows (``category, task_type,
  units, minimum, maximum, species, dataset size``).  The 41 ML
  endpoints cover absorption (Caco2, PAMPA, HIA, Pgp,
  solubility/lipophilicity/hydration-free-energy), distribution
  (BBB, PPBR, VDss), metabolism (5 CYP inhibition + 3 CYP substrate
  assays), excretion (half-life, hepatocyte/microsome clearance) and
  toxicity (hERG, ClinTox, AMES, DILI, carcinogens, LD50, skin, the
  12 Tox21 NR/SR assays).
* :data:`ADMET_MODEL_QUALITY` -- each endpoint's headline score
  (AUROC/AUPRC for classification, R^2 for regression) so the
  low-confidence warning (``< 0.6``) can be raised exactly where
  ADMET-AI raises it.
* :data:`DRUGBANK_RADIAL_QUANTILES` -- the 1/5/10/25/50/75/90/95/99%
  quantiles of the five radial-axis properties over the 2,845-drug
  DrugBank-approved reference set, distilled from
  ``drugbank_approved.csv`` so percentile contextualisation works
  offline (percentiles are interpolated on this grid -- the
  approximation is stated in :func:`drugbank_percentile`).

The radial summary follows ``plot.py``'s exact directionality: BBB
Safe, Non-Toxic and hERG Safe are *inverted* percentiles (``100 - p``),
Soluble and Bioavailable are plain percentiles.

What is honestly absent: the 10 trained chemprop ensembles (torch
checkpoints) -- the toolkit's answer to the *model* is
:mod:`pharmaml.ml.mpnn`, the same architecture they train; the full
2,845-row DrugBank table and the ATC-class browser (a bundled
708-column resource; the quantile grid carries the five axes that are
actually displayed); and the JSME molecule editor (web-only).  The
physchem panel rides the RDKit bridge the toolkit uses everywhere,
including the three FilterCatalog alert counters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.logger import get_logger

__all__ = [
    "ADMET_ENDPOINTS",
    "ADMET_MODEL_QUALITY",
    "DRUGBANK_RADIAL_QUANTILES",
    "LOW_PERFORMANCE_THRESHOLD",
    "RADIAL_AXES",
    "admet_endpoint_frame",
    "clip_prediction",
    "drugbank_percentile",
    "low_confidence_endpoints",
    "physchem_panel",
    "radial_scores",
]

_LOG = get_logger(__name__)

#: ADMET-AI's low-confidence warning threshold (web/app/config.py).
LOW_PERFORMANCE_THRESHOLD = 0.6

#: The 52 endpoints, verbatim from resources/data/admet.csv:
#: ``(id, category, task_type, units, minimum, maximum, species, size)``.
#: ``minimum``/``maximum`` of ``None`` mean unbounded; ``size`` is the
#: TDC training-set size (``None`` for computed physchem properties).
ADMET_ENDPOINTS: tuple[tuple[Any, ...], ...] = (
    # -- Physicochemical (computed, not learned) ------------------- #
    ("molecular_weight", "Physicochemical", "regression", "Dalton", 0.0, None, "-", None),
    ("logP", "Physicochemical", "regression", "log-ratio", None, None, "-", None),
    ("hydrogen_bond_acceptors", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    ("hydrogen_bond_donors", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    ("Lipinski", "Physicochemical", "regression", "# of 4", 0.0, 4.0, "-", None),
    ("QED", "Physicochemical", "regression", "-", 0.0, 1.0, "-", None),
    ("stereo_centers", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    ("tpsa", "Physicochemical", "regression", "A^2", 0.0, None, "-", None),
    ("PAINS_alert", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    ("BRENK_alert", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    ("NIH_alert", "Physicochemical", "regression", "#", 0.0, None, "-", None),
    # -- Absorption -------------------------------------------------- #
    ("HIA_Hou", "Absorption", "classification", "-", 0.0, 1.0, "human", 578),
    ("Bioavailability_Ma", "Absorption", "classification", "-", 0.0, 1.0, "human", 640),
    ("Solubility_AqSolDB", "Absorption", "regression", "log(mol/L)", None, None, "-", 9980),
    ("Lipophilicity_AstraZeneca", "Absorption", "regression", "log-ratio", None, None, "-", 4200),
    ("HydrationFreeEnergy_FreeSolv", "Absorption", "regression", "kcal/mol", None, None, "-", 642),
    ("Caco2_Wang", "Absorption", "regression", "log(10^-6 cm/s)", None, None, "human", 906),
    ("PAMPA_NCATS", "Absorption", "classification", "-", 0.0, 1.0, "artificial membrane", 2034),
    ("Pgp_Broccatelli", "Absorption", "classification", "-", 0.0, 1.0, "human pgp / mouse cells", 1212),
    # -- Distribution ------------------------------------------------- #
    ("BBB_Martins", "Distribution", "classification", "-", 0.0, 1.0, "human", 1975),
    ("PPBR_AZ", "Distribution", "regression", "%", 0.0, 100.0, "human", 1614),
    ("VDss_Lombardo", "Distribution", "regression", "L/kg", 0.0, None, "human", 1111),
    # -- Excretion ---------------------------------------------------- #
    ("Half_Life_Obach", "Excretion", "regression", "hr", 0.0, None, "human", 665),
    ("Clearance_Hepatocyte_AZ", "Excretion", "regression", "uL/min/10^6 cells", 0.0, None, "human & rat", 1020),
    ("Clearance_Microsome_AZ", "Excretion", "regression", "uL/min/mg", 0.0, None, "human", 1102),
    # -- Metabolism --------------------------------------------------- #
    ("CYP1A2_Veith", "Metabolism", "classification", "-", 0.0, 1.0, "human", 12579),
    ("CYP2C19_Veith", "Metabolism", "classification", "-", 0.0, 1.0, "human", 12665),
    ("CYP2C9_Veith", "Metabolism", "classification", "-", 0.0, 1.0, "human", 12092),
    ("CYP2D6_Veith", "Metabolism", "classification", "-", 0.0, 1.0, "human", 13130),
    ("CYP3A4_Veith", "Metabolism", "classification", "-", 0.0, 1.0, "human", 12328),
    ("CYP2C9_Substrate_CarbonMangels", "Metabolism", "classification", "-", 0.0, 1.0, "human", 666),
    ("CYP2D6_Substrate_CarbonMangels", "Metabolism", "classification", "-", 0.0, 1.0, "human", 664),
    ("CYP3A4_Substrate_CarbonMangels", "Metabolism", "classification", "-", 0.0, 1.0, "human", 667),
    # -- Toxicity ----------------------------------------------------- #
    ("hERG", "Toxicity", "classification", "-", 0.0, 1.0, "human", 648),
    ("ClinTox", "Toxicity", "classification", "-", 0.0, 1.0, "human", 1459),
    ("AMES", "Toxicity", "classification", "-", 0.0, 1.0, "salmonella typhimurium", 7255),
    ("DILI", "Toxicity", "classification", "-", 0.0, 1.0, "human", 475),
    ("Carcinogens_Lagunin", "Toxicity", "classification", "-", 0.0, 1.0, "rodent", 278),
    ("LD50_Zhu", "Toxicity", "regression", "log(1/(mol/kg))", None, None, "rat", 7342),
    ("Skin_Reaction", "Toxicity", "classification", "-", 0.0, 1.0, "mouse", 404),
    ("NR-AR", "Toxicity", "classification", "-", 0.0, 1.0, "human", 7258),
    ("NR-AR-LBD", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6751),
    ("NR-AhR", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6542),
    ("NR-Aromatase", "Toxicity", "classification", "-", 0.0, 1.0, "human", 5815),
    ("NR-ER", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6186),
    ("NR-ER-LBD", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6948),
    ("NR-PPAR-gamma", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6443),
    ("SR-ARE", "Toxicity", "classification", "-", 0.0, 1.0, "human", 5825),
    ("SR-ATAD5", "Toxicity", "classification", "-", 0.0, 1.0, "human", 7065),
    ("SR-HSE", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6460),
    ("SR-MMP", "Toxicity", "classification", "-", 0.0, 1.0, "human", 5804),
    ("SR-p53", "Toxicity", "classification", "-", 0.0, 1.0, "human", 6767),
)

#: Headline model quality per ADMET endpoint (AUROC/AUPRC for
#: classification, R^2 for regression), verbatim from admet.csv.
ADMET_MODEL_QUALITY: dict[str, float | None] = {
    "HIA_Hou": 0.994,
    "Bioavailability_Ma": 0.716,
    "Solubility_AqSolDB": 0.817,
    "Lipophilicity_AstraZeneca": 0.771,
    "HydrationFreeEnergy_FreeSolv": 0.888,
    "Caco2_Wang": 0.707,
    "PAMPA_NCATS": 0.786,
    "Pgp_Broccatelli": 0.947,
    "BBB_Martins": 0.9,
    "PPBR_AZ": 0.589,
    "VDss_Lombardo": -1.211,
    "Half_Life_Obach": -2.386,
    "Clearance_Hepatocyte_AZ": 0.264,
    "Clearance_Microsome_AZ": 0.277,
    "CYP1A2_Veith": 0.941,
    "CYP2C19_Veith": 0.906,
    "CYP2C9_Veith": 0.908,
    "CYP2D6_Veith": 0.886,
    "CYP3A4_Veith": 0.912,
    "CYP2C9_Substrate_CarbonMangels": 0.628,
    "CYP2D6_Substrate_CarbonMangels": 0.82,
    "CYP3A4_Substrate_CarbonMangels": 0.705,
    "hERG": 0.839,
    "ClinTox": 0.928,
    "AMES": 0.882,
    "DILI": 0.881,
    "Carcinogens_Lagunin": 0.772,
    "LD50_Zhu": 0.596,
    "Skin_Reaction": 0.718,
    "NR-AR": 0.779,
    "NR-AR-LBD": 0.883,
    "NR-AhR": 0.904,
    "NR-Aromatase": 0.906,
    "NR-ER": 0.749,
    "NR-ER-LBD": 0.826,
    "NR-PPAR-gamma": 0.916,
    "SR-ARE": 0.838,
    "SR-ATAD5": 0.879,
    "SR-HSE": 0.843,
    "SR-MMP": 0.925,
    "SR-p53": 0.88,
}

#: The 1/5/10/25/50/75/90/95/99% quantiles of the five radial-axis
#: properties over the 2,845 approved drugs in drugbank_approved.csv
#: (the shipped reference set), distilled for offline percentiles.
_DRUGBANK_QUANTILE_POINTS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
DRUGBANK_RADIAL_QUANTILES: dict[str, tuple[float, ...]] = {
    "BBB_Martins": (0.0014, 0.0521, 0.1384, 0.3924, 0.718, 0.9356, 0.9804, 0.988, 0.9945),
    "ClinTox": (0.0, 0.0059, 0.0149, 0.0375, 0.0975, 0.25, 0.4503, 0.573, 0.7516),
    "Solubility_AqSolDB": (-6.8748, -5.8776, -5.3516, -4.3671, -3.1189, -1.6993, -0.4837, 0.1673, 0.8247),
    "Bioavailability_Ma": (0.0046, 0.1018, 0.319, 0.6279, 0.797, 0.8893, 0.9303, 0.9471, 0.9681),
    "hERG": (0.0012, 0.0083, 0.0158, 0.0535, 0.2951, 0.7437, 0.9319, 0.964, 0.9892),
}

#: The radial summary's five axes and their directionality (plot.py's
#: exact rules): ``invert=True`` means the axis score is ``100 - p``.
RADIAL_AXES: tuple[tuple[str, str, bool], ...] = (
    ("BBB_Martins", "Blood-Brain Barrier Safe", True),
    ("ClinTox", "Non-Toxic", True),
    ("Solubility_AqSolDB", "Soluble", False),
    ("Bioavailability_Ma", "Bioavailable", False),
    ("hERG", "hERG Safe", True),
)


def admet_endpoint_frame() -> Any:
    """The 52-endpoint registry as a DataFrame (admet.csv's columns).

    Examples
    --------
    >>> frame = admet_endpoint_frame()
    >>> len(frame)
    52
    >>> int(frame["size"].dropna().astype(int).sum())   # total training rows
    188195
    >>> frame.loc["LD50_Zhu", "units"]
    'log(1/(mol/kg))'
    """
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "id": row[0],
                "category": row[1],
                "task_type": row[2],
                "units": row[3],
                "minimum": row[4],
                "maximum": row[5],
                "species": row[6],
                "size": row[7],
                "quality": ADMET_MODEL_QUALITY.get(row[0]),
            }
            for row in ADMET_ENDPOINTS
        ]
    ).set_index("id")


def clip_prediction(endpoint_id: str, value: float) -> float:
    """Display-clip a prediction to the endpoint's table bounds.

    ADMET-AI clips *for display only* (downloads keep raw values);
    unbounded sides are passes.

    Examples
    --------
    >>> clip_prediction("PPBR_AZ", 250.0)
    100.0
    >>> clip_prediction("BBB_Martins", 0.3)
    0.3
    >>> clip_prediction("Solubility_AqSolDB", -12.0)
    -12.0
    """
    lookup = {row[0]: row for row in ADMET_ENDPOINTS}
    if endpoint_id not in lookup:
        raise DataValidationError(
            f"unknown endpoint {endpoint_id!r} -- the registry holds "
            f"{len(lookup)} endpoints"
        )
    row = lookup[endpoint_id]
    lo, hi = row[4], row[5]
    out = float(value)
    if lo is not None and out < lo:
        out = float(lo)
    if hi is not None and out > hi:
        out = float(hi)
    return out


def low_confidence_endpoints(
    threshold: float = LOW_PERFORMANCE_THRESHOLD,
) -> list[str]:
    """Endpoints whose headline model quality is below ``threshold``.

    The warning-icon rule of ADMET-AI's results table.

    Examples
    --------
    >>> low = low_confidence_endpoints()
    >>> "VDss_Lombardo" in low and "Clearance_Hepatocyte_AZ" in low
    True
    >>> "AMES" in low
    False
    """
    return sorted(
        name
        for name, quality in ADMET_MODEL_QUALITY.items()
        if quality is not None and quality < threshold
    )


def drugbank_percentile(endpoint_id: str, value: float) -> float:
    """Percentile of ``value`` among approved drugs, on the quantile grid.

    ADMET-AI computes ``scipy.stats.percentileofscore`` against the
    full 2,845-drug DrugBank reference table; offline, the percentile
    is interpolated on the vendored 9-point quantile grid of the same
    distribution -- exact at the grid points, linear between them, and
    clipped to ``[1, 99]`` at the extremes (the grid's coverage).

    Examples
    --------
    >>> round(drugbank_percentile("BBB_Martins", 0.718), 1)   # the median
    50.0
    >>> round(drugbank_percentile("BBB_Martins", 0.0014), 1)  # the 1% point
    1.0
    >>> drugbank_percentile("Solubility_AqSolDB", -3.1189)
    50.0
    >>> drugbank_percentile("hERG", 2.0)
    99.0
    """
    if endpoint_id not in DRUGBANK_RADIAL_QUANTILES:
        raise DataValidationError(
            "drugbank percentiles are vendored for the radial axes only: "
            f"{sorted(DRUGBANK_RADIAL_QUANTILES)}, got {endpoint_id!r}"
        )
    grid = DRUGBANK_RADIAL_QUANTILES[endpoint_id]
    p = float(np.interp(value, grid, _DRUGBANK_QUANTILE_POINTS) * 100.0)
    return float(min(max(p, 1.0), 99.0))


def radial_scores(values: Mapping[str, float]) -> dict[str, float]:
    """The five-axis radial summary (0-100, higher = better).

    ``values`` maps the five radial endpoint ids to predicted values
    (probabilities for the classification axes, log-solubility for
    Soluble).  Inverted axes (BBB Safe, Non-Toxic, hERG Safe) use
    ``100 - percentile`` -- a *low* predicted toxicity is a *high*
    safety score, exactly as ``plot.py`` inverts them.

    Examples
    --------
    >>> scores = radial_scores({
    ...     "BBB_Martins": 0.1384, "ClinTox": 0.0149,
    ...     "Solubility_AqSolDB": -5.3516, "Bioavailability_Ma": 0.319,
    ...     "hERG": 0.0158})
    >>> round(scores["Blood-Brain Barrier Safe"], 1)   # at the 10% point
    90.0
    >>> round(scores["Soluble"], 1)
    10.0
    """
    missing = [name for name, _, _ in RADIAL_AXES if name not in values]
    if missing:
        raise DataValidationError(f"radial_scores is missing: {missing}")
    out: dict[str, float] = {}
    for endpoint, label, invert in RADIAL_AXES:
        p = drugbank_percentile(endpoint, float(values[endpoint]))
        out[label] = 100.0 - p if invert else p
    return out


def physchem_panel(mol_or_smiles: Any) -> dict[str, float]:
    """The 11-property physicochemical panel (admet_ai/physchem.py).

    Molecular weight, Crippen LogP, H-bond acceptors/donors, the
    Lipinski count-of-four, QED (this round's formula port), stereo
    centres, TPSA, and the PAINS / BRENK / NIH structural-alert counts
    (RDKit FilterCatalog -- the bridge the toolkit uses for chemistry
    engines).  Invalid SMILES raises, matching ADMET-AI's row-drop
    behaviour.

    Examples
    --------
    >>> panel = physchem_panel("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
    >>> round(panel["molecular_weight"], 1)
    180.2
    >>> panel["hydrogen_bond_donors"]
    1.0
    >>> 0.0 <= panel["QED"] <= 1.0
    True
    >>> panel["PAINS_alert"]
    0.0
    """
    from rdkit import Chem

    from pharmaml.chem.oracles import qed_score

    mol = mol_or_smiles if hasattr(mol_or_smiles, "GetNumAtoms") else Chem.MolFromSmiles(str(mol_or_smiles))
    if mol is None:
        raise DataValidationError("physchem_panel: invalid SMILES")
    from rdkit.Chem import (
        Descriptors,
        FilterCatalog,
        rdMolDescriptors,
    )

    catalogs = {
        "PAINS_alert": FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        "BRENK_alert": FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK,
        "NIH_alert": FilterCatalog.FilterCatalogParams.FilterCatalogs.NIH,
    }
    alert_counts: dict[str, float] = {}
    for name, which in catalogs.items():
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(which)
        catalog = FilterCatalog.FilterCatalog(params)
        alert_counts[name] = float(len(catalog.GetMatches(mol)))
    mol_wt = Descriptors.MolWt(mol)  # type: ignore[attr-defined]
    mol_logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
    lipinski = sum(
        (
            mol_wt <= 500.0,
            mol_logp <= 5.0,
            rdMolDescriptors.CalcNumHBA(mol) <= 10,
            rdMolDescriptors.CalcNumHBD(mol) <= 5,
        )
    )
    return {
        "molecular_weight": float(mol_wt),
        "logP": float(mol_logp),
        "hydrogen_bond_acceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        "hydrogen_bond_donors": float(rdMolDescriptors.CalcNumHBD(mol)),
        "Lipinski": float(lipinski),
        "QED": float(qed_score(mol)),
        "stereo_centers": float(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        **alert_counts,
    }


def summarize_molecule(mol_or_smiles: Any) -> dict[str, Any]:
    """One molecule's card: physchem panel + radial summary + flags.

    The playground companion that mirrors ADMET-AI's per-molecule
    result card: the 11 computed properties, the five radial-axis
    scores (using the physchem panel's QED/LogP stand-ins where a
    learned endpoint is not available -- the honest offline
    substitution), and the low-confidence endpoint list.

    Examples
    --------
    >>> card = summarize_molecule("CCO")
    >>> sorted(card) == ["low_confidence_endpoints", "physchem", "radial"]
    True
    >>> 0.0 <= card["physchem"]["QED"] <= 1.0
    True
    """
    panel = physchem_panel(mol_or_smiles)
    radial = radial_scores(
        {
            # Offline stand-ins: logP shapes BBB/hERG-style axes, QED the
            # bioavailability-style axis, solubility via LogP inverted.
            "BBB_Martins": float(np.clip((panel["logP"] + 2.0) / 7.0, 0.0, 1.0)),
            "ClinTox": float(np.clip(1.0 - panel["QED"], 0.0, 1.0)),
            "Solubility_AqSolDB": -panel["logP"],
            "Bioavailability_Ma": panel["QED"],
            "hERG": float(np.clip((panel["logP"] + 1.0) / 8.0, 0.0, 1.0)),
        }
    )
    return {
        "physchem": panel,
        "radial": radial,
        "low_confidence_endpoints": low_confidence_endpoints(),
    }
