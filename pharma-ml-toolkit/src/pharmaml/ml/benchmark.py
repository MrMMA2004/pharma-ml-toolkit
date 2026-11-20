r"""Benchmark infrastructure: chem-aware splits, screening metrics, oracles.

Round 175 absorbed the four molecular-ML repositories' *infrastructure*
half (the model layers stay behind the honest boundary):

``chemprop/chemprop``
    (the message-passing property predictor).  v2 delegates its
    splitting to ``astartes``, but the in-repo pieces transfer:
    duplicate-grouped random splitting (``RANDOM_WITH_REPEATED_SMILES``
    -- canonical-SMILES groups so replicate measurements never leak
    across folds) and the atom-map-zeroing preprocessing before
    scaffold canonicalisation.  The uncertainty *auditors* from
    ``chemprop/uncertainty/evaluator.py`` (miscalibration area, ENCE,
    NLL, Spearman(unc, |err|)) and *calibrators*
    (``uncertainty/calibrator.py``: zelikman-interval, zscaling,
    conformal-regression) land in
    :mod:`pharmaml.ml.evaluation`.  The Ray/TPE hyperparameter grid is
    documented in :data:`CHEMPROP_SEARCH_SPACE`.

``BPS/admet_ai`` (swansonk14/admet_ai)
    (ADMET-AI).  The endpoint catalog -- 41 TDC ADME/Tox endpoints +
    11 physicochemical properties with units, valid ranges, species
    and the per-endpoint test accuracy -- transcribes verbatim from
    ``admet_ai/resources/data/admet.csv`` into
    :data:`ADMET_ENDPOINTS`; the radial five-axis "goodness"
    conversion (BBB / hERG / ClinTox inverted as ``100 - percentile``,
    Solubility / Bioavailability raw) becomes
    :func:`admet_radar_score`.

``deepchem/deepchem``
    (the deep-learning chemistry toolkit).  The splitter suite
    (``deepchem/splits/splitters.py``): :func:`scaffold_split` (the
    Murcko-group cumulative-coverage algorithm with the ``(size desc,
    first-member desc)`` ordering and float cutoffs),
    :func:`butina_split` (ECFP4/Tanimoto clustering, cutoff 0.6),
    :func:`maxmin_split` (Dice-distance MaxMin picks),
    :func:`mw_split` and :func:`cold_split` (TDC's entity-disjoint
    holdout).  The metrics: :func:`bedroc_score` (the exact
    RIE/RIEmax/RIEmin algebra, cross-checked against RDKit's
    ``CalcBEDROC``), :func:`enrichment_factor`,
    :func:`concordance_index`, :func:`pearson_r2`.  The
    ``BalancingTransformer`` class-weight scheme becomes
    :func:`balance_sample_weights`.

``mims-harvard/TDC``
    (Therapeutics Data Commons).  The oracle layer
    (``tdc/chem_utils/oracle/oracle.py``): the Ertl
    :func:`sa_score` (via RDKit's Contrib implementation -- the
    fragment-frequency table ships with RDKit, so no boundary here),
    :func:`penalized_logp` with the six ZINC normalisation constants,
    and the full GuacaMol score-modifier family
    (:func:`gaussian_modifier`, :func:`min_gaussian_modifier`,
    :func:`max_gaussian_modifier`, :func:`clipped_score_modifier`,
    :func:`smooth_clipped_score_modifier`,
    :func:`thresholded_linear_modifier`,
    :func:`squared_modifier`, :func:`absolute_modifier`).  The
    early-recognition metric :func:`range_log_auc`
    (``tdc/evaluator.py``'s log-FPR-segment AUC),
    :func:`recall_at_precision_k` /
    :func:`precision_at_recall_k`, the nM<->p conversion of
    ``tdc/utils/label.py`` and the GuacaMol top-k aggregation
    :func:`topk_score`.  The pickled oracle weights (drd2 / gsk3b /
    jnk3 / cyp3a4) are the not-vendored part -- their fingerprint
    conventions are documented in :data:`WEIGHTED_ORACLE_CONVENTIONS`.

Examples
--------
>>> import numpy as np
>>> from pharmaml.ml.benchmark import bedroc_score, enrichment_factor
>>> y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
>>> scores = np.linspace(1.0, 0.1, 10)
>>> round(float(bedroc_score(y, scores)), 6)      # matches rdkit CalcBEDROC
0.882729
>>> round(float(enrichment_factor(y, scores, fraction=0.2)), 2)   # 2 hits in top 2
1.67

>>> from pharmaml.ml.benchmark import gaussian_modifier, clipped_score_modifier
>>> round(float(gaussian_modifier(2.0, mu=2.0, sigma=1.0)), 4)
1.0
>>> clipped_score_modifier(0.5, lower=0.0, upper=1.0, floor=0.0, ceiling=1.0)
0.5

>>> from pharmaml.ml.benchmark import ADMET_ENDPOINTS
>>> int(ADMET_ENDPOINTS.shape[0])
52
>>> sorted(ADMET_ENDPOINTS["category"].unique())[:2]
['Absorption', 'Distribution']
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

_trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
# numpy 1.x spells it trapz, 2.x renamed it trapezoid -- both supported

from pharmaml.exceptions import DataValidationError

__all__ = [
    "ADMET_ENDPOINTS",
    "CHEMPROP_SEARCH_SPACE",
    "WEIGHTED_ORACLE_CONVENTIONS",
    "absolute_modifier",
    "admet_radar_score",
    "balance_sample_weights",
    "bedroc_score",
    "binarize_affinity",
    "butina_split",
    "clipped_score_modifier",
    "cold_split",
    "concordance_index",
    "duplicate_grouped_split",
    "enrichment_factor",
    "gaussian_modifier",
    "max_gaussian_modifier",
    "maxmin_split",
    "min_gaussian_modifier",
    "mw_split",
    "nm_to_p",
    "p_to_nm",
    "pearson_r2",
    "penalized_logp",
    "precision_at_recall_k",
    "range_log_auc",
    "recall_at_precision_k",
    "sa_score",
    "scaffold_split",
    "smooth_clipped_score",
    "squared_modifier",
    "thresholded_linear_modifier",
    "topk_score",
]

# ---------------------------------------------------------------------------
# the ADMET endpoint catalog (admet_ai/resources/data/admet.csv, verbatim)
# ---------------------------------------------------------------------------

_CATALOG_CSV = """
category,id,name,task_type,units,minimum,maximum,species,AUPRC,AUROC,R^2,MAE
Physicochemical,molecular_weight,Molecular Weight,regression,Dalton,0.0,inf,-,,,,
Physicochemical,logP,LogP,regression,log-ratio,-inf,inf,-,,,,
Physicochemical,hydrogen_bond_acceptors,Hydrogen Bond Acceptors,regression,#,0.0,inf,-,,,,
Physicochemical,hydrogen_bond_donors,Hydrogen Bond Donors,regression,#,0.0,inf,-,,,,
Physicochemical,Lipinski,Lipinski Rule of 5,regression,# of 4,0.0,4.0,-,,,,
Physicochemical,QED,Quantitative Estimate of Druglikeness (QED),regression,-,0.0,1.0,-,,,,
Physicochemical,stereo_centers,Stereo Centers,regression,#,0.0,inf,-,,,,
Physicochemical,tpsa,Topological Polar Surface Area (TPSA),regression,Å^2,0.0,inf,-,,,,
Physicochemical,PAINS_alert,PAINS Alert,regression,#,0.0,inf,-,,,,
Physicochemical,BRENK_alert,BRENK Alert,regression,#,0.0,inf,-,,,,
Physicochemical,NIH_alert,NIH Alert,regression,#,0.0,inf,-,,,,
Absorption,HIA_Hou,Human Intestinal Absorption,classification,-,0.0,1.0,human,0.9994842348250168,0.9940232101885412,,
Absorption,Bioavailability_Ma,Oral Bioavailability,classification,-,0.0,1.0,human,0.8699950216268804,0.7163512034578641,,
Absorption,Solubility_AqSolDB,Aqueous Solubility,regression,log(mol/L),-inf,inf,-,,,0.8166316273439673,0.6921369647155847
Absorption,Lipophilicity_AstraZeneca,Lipophilicity,regression,log-ratio,-inf,inf,-,,,0.7711741108080836,0.4069769084746869
Absorption,HydrationFreeEnergy_FreeSolv,Hydration Free Energy,regression,kcal/mol,-inf,inf,-,,,0.8876539579865786,0.7826281766672228
Absorption,Caco2_Wang,Cell Effective Permeability,regression,log(10^-6 cm/s),-inf,inf,human,,,0.7065060857230016,0.3114522384781544
Absorption,PAMPA_NCATS,PAMPA Permeability,classification,-,0.0,1.0,artificial membrane,0.9549333833403232,0.7863669156852404,,
Absorption,Pgp_Broccatelli,P-glycoprotein Inhibition,classification,-,0.0,1.0,human pgp / mouse cells,0.9580859888617924,0.9474505385046356,,
Distribution,BBB_Martins,Blood-Brain Barrier Penetration,classification,-,0.0,1.0,human,0.9589556717958928,0.9003816994239712,,
Distribution,PPBR_AZ,Plasma Protein Binding Rate,regression,%,0.0,100.0,human,,,0.5888679381304678,6.851690451293895
Distribution,VDss_Lombardo,Volume of Distribution at Steady State,regression,L/kg,0.0,inf,human,,,-1.2107082831685434,5.039386413798697
Excretion,Half_Life_Obach,Half Life,regression,hr,0.0,inf,human,,,-2.385712736530985,31.847395032315596
Excretion,Clearance_Hepatocyte_AZ,Drug Clearance (Hepatocyte),regression,uL/min/10^6 cells,0.0,inf,human & rat,,,0.2639755817093195,32.72573699277887
Excretion,Clearance_Microsome_AZ,Drug Clearance (Microsome),regression,uL/min/mg,0.0,inf,human,,,0.276741790995323,26.73759681256409
Metabolism,CYP1A2_Veith,CYP1A2 Inhibition,classification,-,0.0,1.0,human,0.9352310814957951,0.9410729533100876,,
Metabolism,CYP2C19_Veith,CYP2C19 Inhibition,classification,-,0.0,1.0,human,0.875286685765493,0.9062325677287462,,
Metabolism,CYP2C9_Veith,CYP2C9 Inhibition,classification,-,0.0,1.0,human,0.8223621973733012,0.9076776897689186,,
Metabolism,CYP2D6_Veith,CYP2D6 Inhibition,classification,-,0.0,1.0,human,0.7255343259283761,0.8861374248891597,,
Metabolism,CYP3A4_Veith,CYP3A4 Inhibition,classification,-,0.0,1.0,human,0.8733639396410531,0.9116457077306306,,
Metabolism,CYP2C9_Substrate_CarbonMangels,CYP2C9 Substrate,classification,-,0.0,1.0,human,0.3696343498175162,0.6278973930359049,,
Metabolism,CYP2D6_Substrate_CarbonMangels,CYP2D6 Substrate,classification,-,0.0,1.0,human,0.6602305465449196,0.8200422282192378,,
Metabolism,CYP3A4_Substrate_CarbonMangels,CYP3A4 Substrate,classification,-,0.0,1.0,human,0.7271589946144282,0.7048194823279468,,
Toxicity,hERG,hERG Blocking,classification,-,0.0,1.0,human,0.9061997857102244,0.8387915076676851,,
Toxicity,ClinTox,Clinical Toxicity,classification,-,0.0,1.0,human,0.6069354051558659,0.9275087330093084,,
Toxicity,AMES,Mutagenicity,classification,-,0.0,1.0,salmonella typhimurium,0.8957980386551526,0.8815869913780047,,
Toxicity,DILI,Drug Induced Liver Injury,classification,-,0.0,1.0,human,0.8777146999563998,0.8814562005384994,,
Toxicity,Carcinogens_Lagunin,Carcinogenicity,classification,-,0.0,1.0,rodent,0.608283704744014,0.7722614379084967,,
Toxicity,LD50_Zhu,Acute Toxicity LD50,regression,log(1/(mol/kg)),-inf,inf,rat,,,0.5959243623284032,0.4515061714343698
Toxicity,Skin_Reaction,Skin Reaction,classification,-,0.0,1.0,mouse,0.8730032877355487,0.7179946827366182,,
Toxicity,NR-AR,Androgen Receptor (Full Length),classification,-,0.0,1.0,human,0.5151499066330999,0.7794796279772097,,
Toxicity,NR-AR-LBD,Androgen Receptor (Ligand Binding Domain),classification,-,0.0,1.0,human,0.6721736896162589,0.8833775261961652,,
Toxicity,NR-AhR,Aryl Hydrocarbon Receptor,classification,-,0.0,1.0,human,0.6359797022426139,0.9043922118028404,,
Toxicity,NR-Aromatase,Aromatase,classification,-,0.0,1.0,human,0.4747761622876256,0.9055872473619092,,
Toxicity,NR-ER,Estrogen Receptor (Full Length),classification,-,0.0,1.0,human,0.4583187668558939,0.7488198624538707,,
Toxicity,NR-ER-LBD,Estrogen Receptor (Ligand Binding Domain),classification,-,0.0,1.0,human,0.4838179820849542,0.826487896094781,,
Toxicity,NR-PPAR-gamma,Peroxisome Proliferator-Activated Receptor Gamma,classification,-,0.0,1.0,human,0.4390021749117615,0.9163781570174218,,
Toxicity,SR-ARE,Nuclear Factor (Erythroid-Derived 2)-Like 2/Antioxidant Responsive Element,classification,-,0.0,1.0,human,0.519545643846884,0.8377680366814685,,
Toxicity,SR-ATAD5,ATPase Family AAA Domain-Containing Protein 5 (ATAD5),classification,-,0.0,1.0,human,0.3116413438983285,0.8790823594219473,,
Toxicity,SR-HSE,Heat Shock Factor Response Element,classification,-,0.0,1.0,human,0.4274974106278428,0.8426772634912544,,
Toxicity,SR-MMP,Mitochondrial Membrane Potential,classification,-,0.0,1.0,human,0.6983247428958566,0.92474050186623,,
Toxicity,SR-p53,Tumor Protein p53,classification,-,0.0,1.0,human,0.4411030404695476,0.880314338237026,,

"""

ADMET_ENDPOINTS: pd.DataFrame = pd.read_csv(io.StringIO(_CATALOG_CSV))


# ---------------------------------------------------------------------------
# deepchem / TDC splitter suite
# ---------------------------------------------------------------------------


def _murcko_smiles(smiles: str, chem: Any) -> str | None:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> dict[str, np.ndarray]:
    """The deepchem scaffold split (cumulative-coverage Murcko groups).

    Duplicate-symbol map (round 304, part12 A-6): ``scaffold_split``
    also exists at :mod:`pharmaml.ml.dataset` — the ``Dataset``-API
    library-surface variant (Datasets in and out, pluggable
    ``scaffold_fn``). THIS one is the canonical index-array variant
    the capability registry references (``ml.benchmark``); the
    family's definition sites are pinned by
    ``tests/test_architecture_hygiene.py``.

    Murcko scaffolds (chirality dropped, atom maps zeroed -- chemprop's
    preprocessing) group the molecules; groups sort by ``(size desc,
    first member desc)``; whole groups fill train, then valid, then
    test against the float cutoffs ``frac * n``.  Unparseable SMILES
    are excluded from every split and reported under ``"dropped"``.

    Examples
    --------
    >>> split = scaffold_split(["C", "CC", "CCC", "C1CC1", "C1CCC1"],
    ...                        0.4, 0.3, 0.3)
    >>> sorted(int(i) for i in split["train"])       # the two ring systems
    [3, 4]
    >>> sorted(int(i) for i in split["val"])         # acyclics share the empty scaffold
    [0, 1, 2]
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    total = frac_train + frac_val + frac_test
    if not 0.99 < total < 1.01:
        raise DataValidationError("split fractions must sum to 1")
    groups: dict[str, list[int]] = {}
    dropped: list[int] = []
    for index, smi in enumerate(smiles):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            dropped.append(index)
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        scaffold = _murcko_smiles(chem.MolToSmiles(mol), chem)
        if scaffold is None:
            dropped.append(index)
            continue
        groups.setdefault(scaffold, []).append(index)
    n_kept = sum(len(members) for members in groups.values())
    ordered = sorted(
        (members for members in groups.values()),
        key=lambda members: (len(members), members[0]),
        reverse=True,
    )
    train_cutoff = frac_train * n_kept
    val_cutoff = (frac_train + frac_val) * n_kept
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for members in ordered:
        if len(train) + len(members) > train_cutoff:
            if len(train) + len(val) + len(members) > val_cutoff:
                test.extend(members)
            else:
                val.extend(members)
        else:
            train.extend(members)
    return {
        "train": np.array(sorted(train), dtype=int),
        "val": np.array(sorted(val), dtype=int),
        "test": np.array(sorted(test), dtype=int),
        "dropped": np.array(dropped, dtype=int),
    }


def butina_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    cutoff: float = 0.6,
) -> dict[str, np.ndarray]:
    """The deepchem Butina split (ECFP4-1024 Tanimoto clustering).

    Molecules cluster by ``1 - Tanimoto`` distance (cutoff 0.6 --
    similarity above ~0.6 joins a cluster); clusters sort by size
    descending and fill train/val/test whole, like the scaffold split.

    Examples
    --------
    >>> split = butina_split(["CCO", "CC", "c1ccccc1", "CCCCO"],
    ...                      0.5, 0.25, 0.25)
    >>> len(split["train"]) + len(split["val"]) + len(split["test"])
    4
    """
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.ML.Cluster import Butina

    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    total = frac_train + frac_val + frac_test
    if not 0.99 < total < 1.01:
        raise DataValidationError("split fractions must sum to 1")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fps = []
    dropped: list[int] = []
    for index, smi in enumerate(smiles):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            dropped.append(index)
            continue
        fps.append(generator.GetFingerprint(mol))
    n = len(fps)
    if n == 0:
        raise DataValidationError("no valid molecules to split")
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - x for x in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    clusters = sorted(clusters, key=lambda members: -len(members))
    # re-map kept indices (dropped removed before clustering)
    kept = [i for i in range(len(smiles)) if i not in dropped]
    train_cutoff = frac_train * n
    val_cutoff = (frac_train + frac_val) * n
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for members in clusters:
        indices = [kept[m] for m in members]
        if len(train) + len(indices) > train_cutoff:
            if len(train) + len(val) + len(indices) > val_cutoff:
                test.extend(indices)
            else:
                val.extend(indices)
        else:
            train.extend(indices)
    return {
        "train": np.array(sorted(train), dtype=int),
        "val": np.array(sorted(val), dtype=int),
        "test": np.array(sorted(test), dtype=int),
        "dropped": np.array(dropped, dtype=int),
    }


def maxmin_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """The deepchem MaxMin diversity split (Dice distance, pure numpy).

    The test set is the ``frac_test`` share of molecules picked by the
    MaxMin greedy (each pick maximises the minimum Dice distance to the
    already-picked set); valid extends the picking; train is the rest.

    Examples
    --------
    >>> split = maxmin_split(["CCO", "CC", "c1ccccc1", "CCCCO", "CCCC"],
    ...                      0.4, 0.2, 0.4)
    >>> len(split["train"]) + len(split["val"]) + len(split["test"])
    5
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    total = frac_train + frac_val + frac_test
    if not 0.99 < total < 1.01:
        raise DataValidationError("split fractions must sum to 1")
    generator = None
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bits = []
    dropped: list[int] = []
    for index, smi in enumerate(smiles):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            dropped.append(index)
            continue
        bits.append(np.array(generator.GetFingerprint(mol), dtype=float))
    if not bits:
        raise DataValidationError("no valid molecules to split")
    matrix = np.vstack(bits)
    counts = matrix.sum(axis=1)
    intersection = matrix @ matrix.T
    dice = 2.0 * intersection / np.maximum(counts[:, None] + counts[None, :], 1e-9)
    distance = 1.0 - dice
    np.fill_diagonal(distance, -np.inf)  # self never re-picked

    n = len(bits)
    rng = np.random.default_rng(seed)
    n_test = max(1, round(frac_test * n))
    n_val = max(1, round(frac_val * n))
    picked = [int(rng.integers(n))]
    min_dist = distance[picked[0]].copy()
    while len(picked) < n_test + n_val:
        nxt = int(np.argmax(min_dist))
        picked.append(nxt)
        min_dist = np.minimum(min_dist, distance[nxt])
    picked_arr = np.array(sorted(picked))
    test_set = set(picked_arr[:n_test].tolist())
    val_set = set(picked_arr[n_test:].tolist())
    train_set = set(range(n)) - test_set - val_set
    kept = [i for i in range(len(smiles)) if i not in dropped]
    remap = {position: kept[position] for position in range(n)}
    return {
        "train": np.array(sorted(remap[i] for i in train_set), dtype=int),
        "val": np.array(sorted(remap[i] for i in val_set), dtype=int),
        "test": np.array(sorted(remap[i] for i in test_set), dtype=int),
        "dropped": np.array(dropped, dtype=int),
    }


def mw_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
) -> dict[str, np.ndarray]:
    """The deepchem molecular-weight split (extrapolation stress test).

    Sorts by exact (monoisotopic) molecular weight and slices
    contiguously -- the train set holds the lightest molecules.

    Examples
    --------
    >>> split = mw_split(["CCO", "C", "c1ccccc1"], 0.34, 0.33, 0.33)
    >>> [int(i) for i in split["train"]]             # methane is lightest
    [1]
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    from rdkit.Chem import Descriptors

    weights: list[tuple[float, int]] = []
    dropped: list[int] = []
    for index, smi in enumerate(smiles):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            dropped.append(index)
            continue
        weights.append((Descriptors.ExactMolWt(mol), index))  # type: ignore[attr-defined]
    if not weights:
        raise DataValidationError("no valid molecules to split")
    weights.sort()
    order = np.array([index for _weight, index in weights])
    n = len(order)
    train_end = int(frac_train * n)
    val_end = int((frac_train + frac_val) * n)
    return {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
        "dropped": np.array(dropped, dtype=int),
    }


def cold_split(
    frame: pd.DataFrame,
    entity_cols: Sequence[str],
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> pd.Series:
    """TDC's cold-entity split: hold out whole entities (drugs/targets).

    Rows whose entity value lands in the sampled test entities become
    ``"test"``; the validation entities draw from the remainder at
    ``val_frac / (1 - test_frac)``; everything else trains.  Returns a
    membership Series aligned with ``frame``.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"drug": ["D1", "D2", "D1", "D3", "D2"],
    ...                       "target": ["T1", "T1", "T2", "T2", "T3"]})
    >>> membership = cold_split(frame, ["drug"], test_frac=0.4, val_frac=0.2, seed=3)
    >>> set(membership.unique()) <= {"train", "val", "test"}
    True
    """
    missing = [col for col in entity_cols if col not in frame.columns]
    if missing:
        raise DataValidationError(f"cold_split: missing columns {missing}")
    rng = np.random.default_rng(seed)
    entities = frame[list(entity_cols)].drop_duplicates()
    test_entities = set(map(tuple, entities.sample(frac=test_frac, random_state=seed).values))
    remaining = entities[~entities.apply(tuple, axis=1).isin(test_entities)]
    val_entities = set(
        map(tuple, remaining.sample(frac=val_frac / (1 - test_frac), random_state=int(rng.integers(1 << 30))).values)
    )
    labels = []
    for row in frame[list(entity_cols)].itertuples(index=False):
        key = tuple(row)
        if key in test_entities:
            labels.append("test")
        elif key in val_entities:
            labels.append("val")
        else:
            labels.append("train")
    return pd.Series(labels, index=frame.index, name="split")


def duplicate_grouped_split(
    smiles: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """chemprop's ``RANDOM_WITH_REPEATED_SMILES`` split.

    Canonical SMILES define groups; identical molecules always land in
    the same fold (replicate measurements never leak).

    Examples
    --------
    >>> split = duplicate_grouped_split(["CCO", "CCO", "CC", "c1ccccc1"],
    ...                                 0.5, 0.25, 0.25, seed=0)
    >>> (0 in split["train"]) == (1 in split["train"])   # duplicates together
    True
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    canonical: list[str | None] = []
    for smi in smiles:
        mol = chem.MolFromSmiles(smi)
        canonical.append(chem.MolToSmiles(mol) if mol is not None else None)  # type: ignore[union-attr]
    groups: dict[str, list[int]] = {}
    dropped: list[int] = []
    for index, canon in enumerate(canonical):
        if canon is None:
            dropped.append(index)
        else:
            groups.setdefault(canon, []).append(index)
    unique = list(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    n = len(shuffled)
    train_end = int(frac_train * n)
    val_end = int((frac_train + frac_val) * n)
    assignment: dict[str, str] = {}
    for position, key in enumerate(shuffled):
        if position < train_end:
            assignment[unique[key]] = "train"
        elif position < val_end:
            assignment[unique[key]] = "val"
        else:
            assignment[unique[key]] = "test"
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for key, members in groups.items():
        out[assignment[key]].extend(members)
    return {
        "train": np.array(sorted(out["train"]), dtype=int),
        "val": np.array(sorted(out["val"]), dtype=int),
        "test": np.array(sorted(out["test"]), dtype=int),
        "dropped": np.array(dropped, dtype=int),
    }


# ---------------------------------------------------------------------------
# screening metrics (deepchem score_function.py + TDC evaluator.py)
# ---------------------------------------------------------------------------


def bedroc_score(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    alpha: float = 20.0,
) -> float:
    """Boltzmann-enhanced ROC (Truchon & Bayly 2007), alpha 20 default.

    ``RIE = sum_{actives} exp(-alpha*r_i/N) / (n * denom)`` with
    ``denom = (1/N)*(1-exp(-alpha))/(exp(alpha/N)-1)`` over descending
    1-based ranks, normalised between its extreme orderings.  Verified
    against RDKit's ``CalcBEDROC`` on the doctest vector.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    >>> scores = np.linspace(1.0, 0.1, 10)
    >>> round(float(bedroc_score(y, scores)), 6)
    0.882729
    """
    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    if y.shape != score.shape:
        raise DataValidationError("bedroc_score: labels and scores must align")
    n_actives = int((y == 1).sum())
    n = len(y)
    if n_actives == 0 or n == n_actives:
        return 0.0
    if alpha <= 0:
        raise DataValidationError("alpha must be positive")
    order = np.argsort(-score, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    exponent = np.exp(-alpha * ranks / n)
    denom = (1.0 / n) * (1.0 - math.exp(-alpha)) / (math.exp(alpha / n) - 1.0)
    rie = float(exponent[y == 1].sum()) / (n_actives * denom)
    ratio = n_actives / n
    rie_max = (1.0 - math.exp(-alpha * ratio)) / (ratio * (1.0 - math.exp(-alpha)))
    rie_min = (1.0 - math.exp(alpha * ratio)) / (ratio * (1.0 - math.exp(alpha)))
    return (rie - rie_min) / (rie_max - rie_min)


def enrichment_factor(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fraction: float = 0.05,
) -> float:
    """The enrichment factor over the top ``fraction`` of the ranking.

    ``EF = (actives in top floor(fraction*N) / (fraction*N)) /
    (total actives / N)`` -- how many times better than random the
    early ranking is.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    >>> scores = np.linspace(1.0, 0.1, 10)
    >>> round(float(enrichment_factor(y, scores, fraction=0.2)), 2)
    1.67
    """
    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    n = len(y)
    n_actives = int((y == 1).sum())
    if n_actives == 0:
        raise DataValidationError("enrichment_factor: no actives")
    if not 0 < fraction <= 1:
        raise DataValidationError("fraction must be in (0, 1]")
    top = max(1, math.floor(fraction * n))
    order = np.argsort(-score, kind="stable")[:top]
    hits = int((y[order] == 1).sum())
    return (hits / top) / (n_actives / n)


def concordance_index(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """The concordance index (ties in predictions count half).

    ``P(pred ranks concordant with true ranks)`` over all pairs with
    distinct true values -- the survival-time workhorse, deepchem's
    ``concordance_index``.

    Examples
    --------
    >>> round(concordance_index([1.0, 2.0, 3.0, 4.0], [0.1, 0.4, 0.35, 0.8]), 3)
    0.833
    """
    true = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if true.shape != pred.shape or len(true) < 2:
        raise DataValidationError("concordance_index: aligned arrays of length >= 2")
    pairs = 0
    correct = 0.0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            if true[i] == true[j]:
                continue
            pairs += 1
            if pred[i] == pred[j]:
                correct += 0.5
            elif (pred[i] < pred[j]) == (true[i] < true[j]):
                correct += 1.0
    if pairs == 0:
        raise DataValidationError("concordance_index: no comparable pairs")
    return correct / pairs


def range_log_auc(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fpr_range: tuple[float, float] = (0.001, 0.1),
) -> float:
    """TDC's ``range_logAUC`` -- AUC over a log-scaled FPR segment.

    Trapezoid AUC of TPR vs ``log10(FPR)`` between the range bounds,
    divided by the segment width; the random baseline on
    ``(0.001, 0.1)`` is ~0.0215 (the early-recognition regime).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(5)
    >>> y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    >>> scores = np.linspace(1.0, 0.05, 10)
    >>> value = range_log_auc(y, scores)
    >>> 0.0 < value < 1.0
    True
    """
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    lo, hi = fpr_range
    if not 0 < lo < hi <= 1:
        raise DataValidationError("fpr_range must satisfy 0 < lo < hi <= 1")
    fpr, tpr, _ = roc_curve(y, score, pos_label=1)
    tpr = np.append(tpr, np.interp([lo, hi], fpr, tpr))
    fpr = np.append(fpr, [lo, hi])
    order = np.argsort(fpr)
    fpr, tpr = fpr[order], tpr[order]
    mask = (fpr >= lo) & (fpr <= hi)
    if mask.sum() < 2:
        return 0.0
    x = np.log10(fpr[mask])
    area = float(_trapezoid(tpr[mask], x))
    return area / (math.log10(hi) - math.log10(lo))


def recall_at_precision_k(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    precision_threshold: float = 0.9,
) -> float:
    """TDC's recall at the first precision crossing (0 if unreachable)."""
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    pr, rc, _ = precision_recall_curve(y, score)
    crossing = np.where(pr >= precision_threshold)[0]
    if crossing.size == 0:
        return 0.0
    return float(rc[crossing[0]])


def precision_at_recall_k(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    recall_threshold: float = 0.9,
) -> float:
    """TDC's precision at the last recall crossing (0 if unreachable)."""
    from sklearn.metrics import precision_recall_curve

    y = np.asarray(y_true, dtype=int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    pr, rc, _ = precision_recall_curve(y, score)
    crossing = np.where(rc >= recall_threshold)[0]
    if crossing.size == 0:
        return 0.0
    return float(pr[crossing[-1]])


def pearson_r2(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> float:
    """deepchem's ``pearson_r2_score``: the SQUARED Pearson r.

    Distinct from sklearn's ``r2_score`` (which is 1 - SS_res/SS_tot);
    deepchem literature reports this one.

    Examples
    --------
    >>> round(pearson_r2([1.0, 2.0, 3.0], [1.1, 2.2, 2.9]), 4)
    0.9838
    """
    y = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if y.shape != pred.shape or len(y) < 2:
        raise DataValidationError("pearson_r2: aligned arrays of length >= 2")
    return float(np.corrcoef(y, pred)[0, 1] ** 2)


# ---------------------------------------------------------------------------
# TDC oracles: the GuacaMol modifier family + composite objectives
# ---------------------------------------------------------------------------


def gaussian_modifier(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """``exp(-0.5*((x-mu)/sigma)^2)`` -- the GuacaMol Gaussian score."""
    if sigma <= 0:
        raise DataValidationError("sigma must be positive")
    arr = np.asarray(x, dtype=float)
    result = np.exp(-0.5 * ((arr - mu) / sigma) ** 2)
    if np.isscalar(x):
        return float(result)
    return result


def min_gaussian_modifier(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """Gaussian on the clipped-above-mu values (smaller is better)."""
    arr = np.asarray(x, dtype=float)
    return gaussian_modifier(np.maximum(arr, mu), mu=mu, sigma=sigma)


def max_gaussian_modifier(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """Gaussian on the clipped-below-mu values (bigger is better)."""
    arr = np.asarray(x, dtype=float)
    return gaussian_modifier(np.minimum(arr, mu), mu=mu, sigma=sigma)


def clipped_score_modifier(
    x: float | np.ndarray,
    lower: float,
    upper: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float | np.ndarray:
    """TDC's ``ClippedScoreModifier``: linear ramp, clipped at the ends.

    Examples
    --------
    >>> float(clipped_score_modifier(0.5, lower=0.0, upper=1.0))
    0.5
    >>> float(clipped_score_modifier(2.0, lower=0.0, upper=1.0))
    1.0
    """
    if upper <= lower:
        raise DataValidationError("upper must exceed lower")
    arr = np.asarray(x, dtype=float)
    slope = (ceiling - floor) / (upper - lower)
    result = np.clip(slope * (arr - lower) + floor, floor, ceiling)
    if np.isscalar(x):
        return float(result)
    return result


def smooth_clipped_score(
    x: float | np.ndarray,
    lower_x: float,
    upper_x: float,
    floor: float = 0.0,
    ceiling: float = 1.0,
) -> float | np.ndarray:
    """TDC's ``SmoothClippedScore``: the logistic ramp between bounds."""
    if upper_x <= lower_x:
        raise DataValidationError("upper_x must exceed lower_x")
    arr = np.asarray(x, dtype=float)
    k = 4.0 / (upper_x - lower_x)
    middle = 0.5 * (lower_x + upper_x)
    result = floor + (ceiling - floor) / (1.0 + np.exp(-k * (arr - middle)))
    if np.isscalar(x):
        return float(result)
    return result


def thresholded_linear_modifier(x: float | np.ndarray, threshold: float) -> float | np.ndarray:
    """``min(x, threshold)/threshold`` -- TDC's ThresholdedLinear."""
    if threshold <= 0:
        raise DataValidationError("threshold must be positive")
    arr = np.asarray(x, dtype=float)
    result = np.minimum(arr, threshold) / threshold
    if np.isscalar(x):
        return float(result)
    return result


def squared_modifier(
    x: float | np.ndarray, target: float, coefficient: float = 1.0
) -> float | np.ndarray:
    """``1 - coefficient*(x - target)^2`` (unclipped, as in TDC)."""
    arr = np.asarray(x, dtype=float)
    result = 1.0 - coefficient * (arr - target) ** 2
    if np.isscalar(x):
        return float(result)
    return result


def absolute_modifier(x: float | np.ndarray, target: float) -> float | np.ndarray:
    """``1 - |x - target|`` (unclipped, as in TDC)."""
    arr = np.asarray(x, dtype=float)
    result = 1.0 - np.abs(arr - target)
    if np.isscalar(x):
        return float(result)
    return result


def topk_score(
    scores: Sequence[float],
    ks: Sequence[int] = (1, 10, 100),
    weights: Sequence[float] | None = None,
) -> float:
    """The GuacaMol goal-directed aggregation: weighted mean of top-k means.

    Examples
    --------
    >>> round(topk_score([0.9, 0.8, 0.7, 0.1], ks=(1, 2), weights=(1.0, 1.0)), 4)
    0.875
    """
    arr = np.sort(np.asarray(scores, dtype=float))[::-1]
    if weights is None:
        weights = [1.0] * len(ks)
    if len(ks) != len(weights):
        raise DataValidationError("topk_score: ks and weights must align")
    pieces = []
    for k in ks:
        k_eff = min(int(k), len(arr))
        pieces.append(float(arr[:k_eff].mean()) if k_eff else 0.0)
    return float(np.average(pieces, weights=weights))


# ---------------------------------------------------------------------------
# TDC label conventions
# ---------------------------------------------------------------------------


def nm_to_p(values: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
    """nM affinity to the negative-log convention ``-log10(nM*1e-9 + 1e-10)``.

    TDC's exact offset (the 1e-10 guards zero); the reverse is
    :func:`p_to_nm`.
    """
    arr = np.asarray(values, dtype=float)
    result = -np.log10(arr * 1e-9 + 1e-10)
    if np.isscalar(values):
        return float(result)
    return result


def p_to_nm(values: float | Sequence[float] | np.ndarray) -> float | np.ndarray:
    """The inverse of :func:`nm_to_p` (TDC's formula, single application)."""
    arr = np.asarray(values, dtype=float)
    result = (10.0**-arr - 1e-10) / 1e-9
    if np.isscalar(values):
        return float(result)
    return result


def binarize_affinity(
    values: Sequence[float] | np.ndarray,
    threshold: float,
    descending: bool = True,
) -> np.ndarray:
    """TDC's binarize: with ``descending`` (affinity) lower values are 1."""
    arr = np.asarray(values, dtype=float)
    if descending:
        return (arr < threshold).astype(int)
    return (arr > threshold).astype(int)


# ---------------------------------------------------------------------------
# oracles: SA score + penalized logP
# ---------------------------------------------------------------------------


def sa_score(smiles: Sequence[str]) -> np.ndarray:
    """Ertl's synthetic-accessibility score (TDC's ``sa`` oracle).

    The full pipeline -- fragment-frequency term (score1, from the
    Ertl table that ships inside RDKit's Contrib SA_Score), size/
    stereo/spiro/bridgehead/macrocycle penalties (score2), the
    fingerprint-density correction (score3) and the ``[1, 10]``
    rescale with the smoothed 8+ tail -- via
    ``rdkit.Contrib.SA_Score.sascorer`` (TDC's ``calculateScore``
    transcribed verbatim upstream; the pickled ``fpscores`` table TDC
    downloads at runtime is the same data RDKit already bundles, so no
    boundary is crossed here).

    Examples
    --------
    >>> scores = sa_score(["CC(=O)Oc1ccccc1C(=O)O", "C"])
    >>> 1.0 <= float(scores.min()) <= 10.0
    True
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as exc:
        raise DependencyError(
            "sa_score needs rdkit's Contrib SA_Score module (bundled with "
            "every standard rdkit wheel)"
        ) from exc
    out = []
    for smi in smiles:
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            raise DataValidationError(f"sa_score: unparseable SMILES {smi!r}")
        out.append(float(sascorer.calculateScore(mol)))
    return np.array(out)


def penalized_logp(smiles: Sequence[str]) -> np.ndarray:
    """TDC's ``penalized_logp`` composite with the ZINC constants.

    ``(logP - 2.4571)/1.4343 + (SA - (-3.0526))/0.8335 +
    (-cycle - (-0.0486))/0.2860`` where ``cycle`` is the largest ring
    size above 6 (else 0), negative.  The normalisation constants are
    the ZINC means/stds quoted in ``tdc/chem_utils/oracle/oracle.py``.

    Examples
    --------
    >>> values = penalized_logp(["CC(=O)Oc1ccccc1C(=O)O"])
    >>> -2.0 < float(values[0]) < 6.0
    True
    """
    from pharmaml.chem.descriptors import require_rdkit

    chem = require_rdkit()
    from rdkit.Chem import Descriptors

    logp_mean, logp_std = 2.4570953396190123, 1.434324401111988
    sa_mean, sa_std = -3.0525811293166134, 0.8335207024513095
    cycle_mean, cycle_std = -0.0485696876403053, 0.2860212110245455
    sa = sa_score(smiles)
    out = []
    for index, smi in enumerate(smiles):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            raise DataValidationError(f"penalized_logp: unparseable SMILES {smi!r}")
        logp = Descriptors.MolLogP(mol)  # type: ignore[attr-defined]
        ring_info = mol.GetRingInfo()
        cycle = 0
        for ring in ring_info.AtomRings():
            if len(ring) > 6:
                cycle = max(cycle, len(ring))
        cycle_value = -(cycle - 6) if cycle > 6 else 0
        value = (
            (logp - logp_mean) / logp_std
            + (float(sa[index]) - sa_mean) / sa_std
            + (-cycle_value - cycle_mean) / cycle_std
        )
        out.append(value)
    return np.array(out)


# ---------------------------------------------------------------------------
# the ADMET catalog helpers
# ---------------------------------------------------------------------------


def admet_radar_score(
    percentiles: Mapping[str, float] | dict[str, float],
) -> float:
    """The ADMET-AI five-axis "goodness" average (0-100).

    BBB / hERG / ClinTox enter as ``100 - percentile`` (safe = low
    percentile); Solubility / Bioavailability enter raw (higher is
    better) -- ``admet_ai/plot.py``'s radial conversion; the scalar is
    the plain mean of the five axes.

    Examples
    --------
    >>> round(admet_radar_score({"BBB_Martins": 20.0, "hERG": 10.0,
    ...                          "ClinTox": 30.0, "Solubility_AqSolDB": 60.0,
    ...                          "Bioavailability_Ma": 50.0}), 1)
    70.0
    """
    required = ("BBB_Martins", "hERG", "ClinTox", "Solubility_AqSolDB", "Bioavailability_Ma")
    missing = [key for key in required if key not in percentiles]
    if missing:
        raise DataValidationError(f"admet_radar_score: missing axes {missing}")
    axes = [
        100.0 - float(percentiles["BBB_Martins"]),
        100.0 - float(percentiles["hERG"]),
        100.0 - float(percentiles["ClinTox"]),
        float(percentiles["Solubility_AqSolDB"]),
        float(percentiles["Bioavailability_Ma"]),
    ]
    return float(np.mean(axes))


# ---------------------------------------------------------------------------
# deepchem BalancingTransformer + chemprop search space (documented data)
# ---------------------------------------------------------------------------


def balance_sample_weights(
    y: Sequence[int] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """deepchem's BalancingTransformer: per-class weights summing to N.

    Every class's rows receive weight ``N_task / count(class)`` so each
    class carries equal total mass (zero-weight samples stay zero; the
    weights REPLACE, not multiply -- the deepchem semantics).

    Examples
    --------
    >>> weights = balance_sample_weights([0, 0, 0, 0, 1])
    >>> [round(float(w), 3) for w in weights]         # 5/4 and 5/1
    [1.25, 1.25, 1.25, 1.25, 5.0]
    """
    labels = np.asarray(y, dtype=int).ravel()
    if sample_weights is None:
        base = np.ones(len(labels))
    else:
        base = np.asarray(sample_weights, dtype=float).copy()
    out = np.zeros_like(base)
    n = int((base != 0).sum())
    for class_value in np.unique(labels):
        mask = (labels == class_value) & (base != 0)
        count = int(mask.sum())
        if count:
            out[mask] = n / count
    return out


#: chemprop v2's default Ray-Tune search space (chemprop/cli/hpopt.py
#: 70-92) -- the documented hyperparameter grid, not a vendored tuner.
CHEMPROP_SEARCH_SPACE: dict[str, object] = {
    "activation": "choice(relu,leakyrelu,tanh,selu,elu,prelu)",
    "dropout": "choice(0.0 x8, 0.05..0.40 step 0.05)",
    "message_hidden_dim": "qrandint(300, 2400, q=100)",
    "depth": "randint(2, 6)",
    "aggregation": "choice(mean,sum,norm,att)",
    "aggregation_norm": "randint(1, 200)",
    "ffn_hidden_dim": "qrandint(300, 2400, q=100)",
    "ffn_num_layers": "randint(1, 2)",
    "batch_size": "choice(16, 32, 64, 128, 256)",
    "init_lr_ratio": "loguniform(1e-2, 1)",
    "max_lr": "loguniform(1e-4, 1e-2)",
    "final_lr_ratio": "loguniform(1e-2, 1)",
    "warmup_epochs": "randint(1, train_epochs // 2)",
    "num_trials": 10,
    "search_algorithm": "hyperopt (TPE)",
    "scheduler": "FIFO (ASHA optional: grace 10, factor 2)",
}

#: The TDC pickled-oracle fingerprint conventions (documented, not vendored):
#: drd2 = FCFP6 counts folded mod 2048; gsk3b / jnk3 = Morgan r=2 nBits=2048
#: bits; cyp3a4_veith = DeepPurpose CNN.  The weights are runtime downloads
#: upstream and stay behind the honest boundary.
WEIGHTED_ORACLE_CONVENTIONS: dict[str, str] = {
    "drd2": "Morgan r=3 useCounts useFeatures folded % 2048 -> sklearn pickle",
    "gsk3b": "Morgan r=2 nBits=2048 bitvect -> predict_proba[:, 1]",
    "jnk3": "Morgan r=2 nBits=2048 bitvect -> predict_proba[:, 1]",
    "cyp3a4_veith": "DeepPurpose CNN encoder -> sklearn pickle",
}


from collections.abc import Mapping

from pharmaml.exceptions import DependencyError
