"""CSV -> canonical SMILES: the batch standardization pipeline.

The owner's directive (2026-11-06, round 192): *"Add a section to
playground.py to take the smiles from the csv file and convert them to
canonical smiles (standardized and processed)"*, delivered with seven
reference repositories.  This module is the engine behind that card --
one standardization front door (``standardize_smiles``), one CSV batch
driver (``standardize_csv``), five reference chains absorbed offline
(file-by-file from shallow clones, the round-186 discipline), and the
protonation-state enumerator:

* **rdkit/rdkit -- Code/GraphMol/MolStandardize** (the subtree the
  owner linked directly): ``MetalDisconnector`` -> ``Normalizer`` ->
  ``Reionizer`` -> fragment/charge parents -> ``AssignStereochemistry``
  -> canonical SMILES.  The ``rdkit`` profile wires the module's own
  classes in the canonical order the ``MolStandardize`` docs describe,
  with the optional ``FragmentParent`` / ``ChargeParent`` /
  ``Canonicalize``-tautomer legs the owner ticks in the card.
* **mcs07/MolVS** -- ``Standardizer.standardize`` verbatim:
  ``SanitizeMol`` -> ``RemoveHs`` -> ``MetalDisconnector`` ->
  ``Normalizer`` -> ``Reionizer`` -> ``AssignStereochemistry(force,
  cleanIt)``; the fragment / charge / tautomer parents map onto the
  same RDKit classes (MolVS is *literally* what became
  ``rdMolStandardize`` -- Greg Landrum absorbed the library into
  RDKit in 2018, so this port calls the absorbed originals).
* **chembl/ChEMBL_Structure_Pipeline** -- ``standardizer.py`` step by
  step: the metal/borane *exclude flag*, valence update, SGroup strip,
  ``Kekulize``, the custom ``remove_hs_from_mol`` (the SENTINEL-isotope
  preserve rules), ``normalize_mol`` (the twelve ChEMBL SMIRKS plus
  the alkoxide pre-step), ``uncharge_mol``
  (``Uncharger(canonicalOrder=True)``), the tartrate flatten, the
  drawing-cleanup guard (a no-op without conformers -- their own
  guard), and the *solvent/salt-list fragment parent* (163 salts +
  9 solvents shipped in their ``data/`` directory, embedded here as
  :data:`CHEMBL_SALTS` / :data:`CHEMBL_SOLVENTS`).
* **datamol-io/datamol** -- ``dm.standardize_mol`` with the library's
  own defaults (``disconnect_metals=False, normalize=True,
  reionize=True, uncharge=False, stereo=True``) plus
  ``dm.standardize_smiles``; the fragment leg is *not* part of
  datamol's standardize (a documented difference -- the card applies
  the toolkit's ``LargestFragmentChooser`` front step when the owner
  ticks salts-out).
* **UnixJunkie/dimorphite_dl** -- the protonation-state enumerator:
  the 41 substructure rules of ``site_substructures.smarts``
  (:data:`PROTONATION_SMARTS`, embedded verbatim with the pKa mean /
  std columns), ``define_protonation_state`` (BOTH / PROTONATED /
  DEPROTONATED from the pH-window overlap), the list-order matching
  with the protect/unprotect discipline, the nitrogen charge special
  cases (``*``-named neutral-acid moieties, the explicit-H bond-order
  table, the ``[nH-]`` -> ``[n-]`` fix) and the ``neutralize_mol``
  reaction fixpoint that runs before the enumeration.
  :func:`protonate_smiles` is that port.
* **openbabel/openbabel** -- stands in the *cross-toolkit slot*: OB
  itself is not a pharmaml dependency (the policy
  ``chem/standardize.py`` documents: the rar MOL2 leg was dropped for
  exactly this reason), so the OB-style "second toolkit agrees" check
  is served by the InChI round-trip,
  :func:`inchi_roundtrip_check` -- the same format-neutral
  normalisation OpenBabel users reach for when two cheminformatics
  toolkits must agree on a structure.
* **rdkit/rdkit (root)** -- the ``reinvent`` profile reuses
  :func:`pharmaml.chem.standardize.reinvent_standardize_mol` (the
  REINVENT4 default chain absorbed in the round-100x-164), so the six
  profiles share one report format.

The batch driver keeps the house rules: no silent drops (every
unparseable row is a record with its error), typed errors for the
caller's mistakes (a CSV with no detectable SMILES column raises
:class:`~pharmaml.exceptions.DataValidationError` naming the tried
columns), and a duplicate register (canonical collisions are flagged
against their first row, the ``dedupe`` audit downstream feeds on).

Examples
--------
>>> from pharmaml.chem.csv_standardize import standardize_smiles
>>> standardize_smiles("CC(=O)[O-].[Na+]").canonical   # rdkit profile
'CC(=O)O'
>>> standardize_smiles("CC(=O)[O-].[Na+]", profile="molvs").canonical
'CC(=O)O'
>>> standardize_smiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
...                    profile="chembl").canonical   # aspirin sodium
'CC(=O)Oc1ccccc1C(=O)O'
>>> standardize_smiles("c1ccccc1O", profile="datamol").canonical
'Oc1ccccc1'
>>> standardize_smiles("[NH3+]CC(=O)[O-]", profile="datamol",
...                    uncharge=True).canonical
'NCC(=O)O'
>>> standardize_smiles("CC(=O)CC(C)=O", tautomer=True).canonical  # keto
'CC(=O)CC(C)=O'
>>> standardize_smiles("CC(O)=CC(C)=O", tautomer=True).canonical   # enol
'CC(=O)CC(C)=O'
>>> outcome = standardize_smiles("not-a-smiles")
>>> (outcome.canonical, outcome.status, outcome.error is not None)
(None, 'invalid', True)

The protonation enumerator (dimorphite_dl's own behaviour):

>>> from pharmaml.chem.csv_standardize import protonate_smiles
>>> protonate_smiles("CC(=O)O", min_ph=6.4, max_ph=8.4)  # physiological
['CC(=O)[O-]']
>>> protonate_smiles("CC(=O)O", min_ph=1.0, max_ph=2.0)   # acid stays
['CC(=O)O']

>>> from pharmaml.chem.csv_standardize import inchi_roundtrip_check
>>> agreed, other = inchi_roundtrip_check("CCO")
>>> (agreed, other)
(True, 'CCO')

The CSV front door (a report, not a bare frame):

>>> import pandas as pd
>>> from pharmaml.chem.csv_standardize import standardize_csv
>>> frame = pd.DataFrame({"name": ["aspirin", "sodium salt", "bad"],
...                       "SMILES": ["CC(=O)OC1=CC=CC=C1C(=O)O",
...                                  "CC(=O)[O-].[Na+]", "not-a-smiles"]})
>>> report = standardize_csv(frame, profile="rdkit")
>>> report.smiles_column
'SMILES'
>>> [(r.status, r.canonical) for r in report.records]
[('standardized', 'CC(=O)Oc1ccccc1C(=O)O'), ('standardized', 'CC(=O)O'), ('invalid', None)]
>>> report.summary()["standardized"], report.summary()["invalid"]
(2, 1)
>>> report.to_frame().shape
(3, 8)
>>> standardize_csv(pd.DataFrame({"x": [1, 2]}))   # no text column at all
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: No SMILES column detected in the CSV (looked for a name match, then a parse-rate check over: no text columns)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "CHEMBL_EXCLUDE_METALS",
    "CHEMBL_NORMALIZATION_TRANSFORMS",
    "CHEMBL_SALTS",
    "CHEMBL_SOLVENTS",
    "PROFILES",
    "PROTONATION_SMARTS",
    "StandardizationProfile",
    "StandardizationRecord",
    "StandardizationReport",
    "StandardizeOptions",
    "StandardizeOutcome",
    "detect_smiles_column",
    "inchi_roundtrip_check",
    "protonate_smiles",
    "standardize_csv",
    "standardize_smiles",
]

# ---------------------------------------------------------------------------
# The dimorphite_dl rule table (UnixJunkie/dimorphite_dl, the file
# site_substructures.smarts shipped with the release, 41 rules).  Each
# rule: (name, SMARTS, ((site, mean pKa, std), ...)).  The site number
# is the 0-based atom position inside the SMARTS string -- the
# convention the original loader indexes ``match[site]`` with.  Names
# carrying ``*`` mark nitrogen moieties whose *acidic* form is the
# neutral one (amides, imides, sulfonamides) -- the charge ladder
# inverts for them, the original ``set_protonation_charge`` logic.
# ---------------------------------------------------------------------------
PROTONATION_SMARTS: tuple[tuple[str, str, tuple[tuple[float, float, float], ...]], ...] = (
    ("*Azide", "[N+0:1]=[N+:2]=[N+0:3]-[H]", ((2, 4.65, 0.07071067811865513),)),
    ("Nitro", "[C,c,N,n,O,o:1]-[NX3:2](=[O:3])-[O:4]-[H]", ((3, -1000.0, 0.0),)),
    ("AmidineGuanidine1", "[N:1]-[C:2](-[N:3])=[NX2:4]-[H:5]", ((3, 12.025333333333334, 1.5941046150769165),)),
    ("AmidineGuanidine2", "[C:1](-[N:2])=[NX2+0:3]", ((2, 10.035538461538462, 2.1312826469414716),)),
    ("Sulfate", "[SX4:1](=[O:2])(=[O:3])([O:4]-[C,c,N,n:5])-[OX2:6]-[H]", ((5, -2.36, 1.3048043093561141),)),
    ("Sulfonate", "[SX4:1](=[O:2])(=[O:3])(-[C,c,N,n:4])-[OX2:5]-[H]", ((4, -1.8184615384615386, 1.4086213481855594),)),
    ("Sulfinic_acid", "[SX3:1](=[O:2])-[O:3]-[H]", ((2, 1.7933333333333332, 0.4372070447739835),)),
    ("Phenyl_carboxyl", "[c,n,o:1]-[C:2](=[O:3])-[O:4]-[H]", ((3, 3.463441968255319, 1.2518054407928614),)),
    ("Carboxyl", "[C:1](=[O:2])-[O:3]-[H]", ((2, 3.456652971502591, 1.2871420886834017),)),
    ("Thioic_acid", "[C,c,N,n:1](=[O,S:2])-[SX2,OX2:3]-[H]", ((2, 0.678267, 1.497048763660801),)),
    ("Phenyl_Thiol", "[c,n:1]-[SX2:2]-[H]", ((1, 4.978235294117647, 2.6137000480499806),)),
    ("Thiol", "[C,N:1]-[SX2:2]-[H]", ((1, 9.12448275862069, 1.3317968158171463),)),
    ("Phosphate", "[PX4:1](=[O:2])(-[OX2:3]-[H])(-[O+0:4])-[OX2:5]-[H]", ((2, 2.4182608695652172, 1.1091177991945305), (5, 6.5055, 0.9512787792174668))),
    ("Internal_phosphate_polyphos_chain", "[$([PX4:1](=O)([OX2][PX4](=O)([OX2])(O[H]))([OX2][PX4](=O)(O[H])([OX2])))][O:2]-[H]", ((1, 0.9, 1.0),)),
    ("Initial_phosphate_like_in_ATP_ADP", "[$([PX4:1]([OX2][C,c,N,n])(=O)([OX2][PX4](=O)([OX2])(O[H])))]O-[H]", ((1, 2.4182608695652172, 1.1091177991945305),)),
    ("Phosphonate", "[PX4:1](=[O:2])(-[OX2:3]-[H])(-[C,c,N,n:4])-[OX2:5]-[H]", ((2, 1.8835714285714287, 0.5925999820080644), (5, 7.247254901960784, 0.8511476450801531))),
    ("Phenol", "[c,n,o:1]-[O:2]-[H]", ((1, 7.065359866910526, 3.277356122295936),)),
    ("Peroxide1", "[O:1]([$(C=O),$(C[Cl]),$(CF),$(C[Br]),$(CC#N):2])-[O:3]-[H]", ((2, 8.738888888888889, 0.7562592839596507),)),
    ("Peroxide2", "[C:1]-[O:2]-[O:3]-[H]", ((2, 11.978235294117647, 0.8697645895163075),)),
    ("O=C-C=C-OH", "[O:1]=[C;R:2]-[C;R:3]=[C;R:4]-[O:5]-[H]", ((4, 3.554, 0.803339458581667),)),
    ("Vinyl_alcohol", "[C:1]=[C:2]-[O:3]-[H]", ((2, 8.871850714285713, 1.660200255394124),)),
    ("Alcohol", "[C:1]-[O:2]-[H]", ((1, 14.780384615384616, 2.546464970533435),)),
    ("N-hydroxyamide", "[C:1](=[O:2])-[N:3]-[O:4]-[H]", ((3, 9.301904761904762, 1.2181897185891002),)),
    ("*Ringed_imide1", "[O,S:1]=[C;R:2]([$([#8]),$([#7]),$([#16]),$([#6][Cl]),$([#6]F),$([#6][Br]):3])-[N;R:4]([C;R:5]=[O,S:6])-[H]", ((3, 6.4525, 0.5555627777308341),)),
    ("*Ringed_imide2", "[O,S:1]=[C;R:2]-[N;R:3]([C;R:4]=[O,S:5])-[H]", ((2, 8.681666666666667, 1.8657779975741713),)),
    ("*Imide", "[F,Cl,Br,S,s,P,p:1][#6:2][CX3:3](=[O,S:4])-[NX3+0:5]([CX3:6]=[O,S:7])-[H]", ((4, 2.4666666666666667, 1.4843629385474877),)),
    ("*Imide2", "[O,S:1]=[CX3:2]-[NX3+0:3]([CX3:4]=[O,S:5])-[H]", ((2, 10.23, 1.1198214143335534),)),
    ("*Amide_electronegative", "[C:1](=[O:2])-[N:3](-[Br,Cl,I,F,S,O,N,P:4])-[H]", ((2, 3.4896, 2.688124315081677),)),
    ("*Amide", "[C:1](=[O:2])-[N:3]-[H]", ((2, 12.00611111111111, 4.512491341218857),)),
    ("*Sulfonamide", "[SX4:1](=[O:2])(=[O:3])-[NX3+0:4]-[H]", ((3, 7.9160326086956525, 1.9842121316708763),)),
    ("Anilines_primary", "[c:1]-[NX3+0:2]([H:3])[H:4]", ((1, 3.899298673194805, 2.068768503987161),)),
    ("Anilines_secondary", "[c:1]-[NX3+0:2]([H:3])[!H:4]", ((1, 4.335408163265306, 2.1768842022330843),)),
    ("Anilines_tertiary", "[c:1]-[NX3+0:2]([!H:3])[!H:4]", ((1, 4.16690685045614, 2.005865735782679),)),
    ("Aromatic_nitrogen_unprotonated", "[n+0&H0:1]", ((0, 4.3535441240733945, 2.0714072661859584),)),
    ("Amines_primary_secondary_tertiary", "[C:1]-[NX3+0:2]", ((1, 8.159107682388349, 2.5183597445318147),)),
    ("Phosphinic_acid", "[PX4:1](=[O:2])(-[C,c,N,n,F,Cl,Br,I:3])(-[C,c,N,n,F,Cl,Br,I:4])-[OX2:5]-[H]", ((4, 2.9745, 0.6867886750744557),)),
    ("Phosphate_diester", "[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[O+0:5]-[C,c,N,n,F,Cl,Br,I:4])-[OX2:6]-[H]", ((6, 2.7280434782608696, 2.5437448856908316),)),
    ("Phosphonate_ester", "[PX4:1](=[O:2])(-[OX2:3]-[C,c,N,n,F,Cl,Br,I:4])(-[C,c,N,n,F,Cl,Br,I:5])-[OX2:6]-[H]", ((5, 2.0868, 0.4503028610465036),)),
    ("Primary_hydroxyl_amine", "[C,c:1]-[O:2]-[NH2:3]", ((2, 4.035714285714286, 0.8463816543155368),)),
    ("*Indole_pyrrole", "[c;R:1]1[c;R:2][c;R:3][c;R:4][n;R:5]1[H]", ((4, 14.52875, 4.06702491591416),)),
    ("*Aromatic_nitrogen_protonated", "[n:1]-[H]", ((0, 7.17, 2.94602395490212),)),
)

#: The dimorphite_dl neutralisation reactions (``UtilFuncs.neutralize_mol``),
#: applied to a fixpoint before the protonation sites are read -- the
#: valence patches every input passes through.
_NEUTRALIZE_REACTIONS: tuple[tuple[str, str], ...] = (
    ("[Ov1-1:1]", "[Ov2+0:1]-[H]"),
    ("[#7v4+1:1]-[H]", "[#7v3+0:1]"),
    ("[Ov2-:1]", "[Ov2+0:1]"),
    ("[#7v3+1:1]", "[#7v3+0:1]"),
    ("[#7v2-1:1]", "[#7+0:1]-[H]"),
    ("[H]-[N:1]-[N:2]#[N:3]", "[N:1]=[N+1:2]=[N:3]-[H]"),
)

#: The ChEMBL pipeline's metal exclude list (``exclude_flag.py`` -- Zn is
#: deliberately absent, their comment: most Zn compounds in ChEMBL are
#: simple salts) plus the borane rule (> 7 boron atoms).
CHEMBL_EXCLUDE_METALS: frozenset[str] = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ga", "Y",
        "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Cd", "In", "Sn", "La",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
        "Bi", "Po", "Ac", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
        "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Th", "Pa", "U", "Np", "Pu",
        "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Ge", "Sb",
    }
)

#: The twelve ChEMBL normalization SMIRKS (``standardizer.py``'s
#: ``_normalization_transforms`` -- the MolVS set with ChEMBL's own
#: additions, quoted verbatim).
CHEMBL_NORMALIZATION_TRANSFORMS: str = """\
//      Name    SMIRKS
Nitro to N+(O-)=O	[N;X3:1](=[O:2])=[O:3]>>[*+1:1]([*-1:2])=[*:3]
Diazonium N	[*:1]-[N;X2:2]#[N;X1:3]>>[*:1]-[*+1:2]#[*:3]
Quaternary N	[N;X4;v4;+0:1]>>[*+1:1]
Trivalent O	[*:1]=[O;X2;v3;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
Sulfoxide to -S+(O-)	[!O:1][S+0;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
Sulfoxide to -S+(O-) 2	[!O:1][SH1+1;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]
Trivalent S	[O:1]=[S;D2;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]
Bad amide tautomer1	[C:1]([OH1;D1:2])=;!@[NH1:3]>>[C:1](=[OH0:2])-[NH2:3]
Bad amide tautomer2	[C:1]([OH1;D1:2])=;!@[NH0:3]>>[C:1](=[OH0:2])-[NH1:3]
Halogen with no neighbors	[F,Cl,Br,I;X0;+0:1]>>[*-1:1]
Odd pyridine/pyridazine oxide structure	[C,N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[*-0:1]=[*+1:2]-[*-:3]
Odd azide	[*:1][N-:2][N+:3]#[N:4]>>[*:1][N+0:2]=[N+:3]=[N-:4]
"""

#: The ChEMBL ``solvents.smi`` (the solvent list the fragment parent
#: strips; matched by atom/bond count + substructure, their rule).
CHEMBL_SOLVENTS: tuple[tuple[str, str], ...] = (
    ("WATER", "[OH2]"),
    ("DICHLOROMETHANE", "ClCCl"),
    ("TRICHLOROMETHANE", "ClC(Cl)Cl"),
    ("ETHYL ACETATE", "CCOC(=O)C"),
    ("METHANOL", "CO"),
    ("PROPAN-2-OL", "CC(C)O"),
    ("ACETONE", "CC(=O)C"),
    ("DMSO", "CS(=O)C"),
    ("ETHANOL", "CCO"),
)

#: The ChEMBL ``salts.smi`` -- 163 salt counter-ions and co-formers,
#: the second strip list of the ChEMBL fragment parent (everything
#: removed -> keep everything, their fallback).
CHEMBL_SALTS: tuple[tuple[str, str], ...] = (
    ("Tetrafluoroboranuide", "F[B-](F)(F)F"),
    ("Arginine", "NC(CCCNC(=N)N)C(=O)O"),
    ("Deanol", "CN(C)CCO"),
    ("2-(Diethylamino)ethanol", "CCN(CC)CCO"),
    ("Ethanolamine", "NCCO"),
    ("DiMeglumine", "CNCC(O)C(O)C(O)C(O)CO"),
    ("Acetate", "CC(=O)O"),
    ("Aceturate", "CC(=O)NCC(=O)O"),
    ("Stearate", "CCCCCCCCCCCCCCCCCC(=O)O"),
    ("Adipate", "OC(=O)CCCCC(=O)O"),
    ("Aluminium", "[Al]"),
    ("Ammonium", "N"),
    ("Ascorbate", "OCC(O)C1OC(=O)C(=C1O)O"),
    ("Aspartate", "NC(CC(=O)O)C(=O)O"),
    ("Barium", "[Ba]"),
    ("Benethamine", "C(Cc1ccccc1)NCc2ccccc2"),
    ("Benzathine", "C(CNCc1ccccc1)NCc1ccccc1"),
    ("Benzoate", "OC(=O)c1ccccc1"),
    ("Besylate", "OS(=O)(=O)c1ccccc1"),
    ("Bismuth", "[Bi]"),
    ("Bromide", "Br"),
    ("Butyraldehyde", "CCCC=O"),
    ("Ethyl Butanoate", "CCCC(=O)OCC"),
    ("Calcium", "[Ca]"),
    ("Camsylate", "CC1(C)C2CCC1(CS(=O)(=O)O)C(=O)C2"),
    ("Carbonate", "OC(=O)O"),
    ("Chloride", "Cl"),
    ("Choline", "C[N+](C)(C)CCO"),
    ("Citrate", "OC(=O)CC(O)(CC(=O)O)C(=O)O"),
    ("Closylate", "OS(=O)(=O)c1ccc(Cl)cc1"),
    ("Cyclamate", "OS(=O)(=O)NC1CCCCC1"),
    ("Dichloroacetate", "OC(=O)C(Cl)Cl"),
    ("Diethylamine", "CCNCC"),
    ("Dimethylethanolamine", "CC(C)(N)CO"),
    ("Diolamine", "OCCNCCO"),
    ("Edamine", "NCCN"),
    ("Edisylate", "OS(=O)(=O)CCS(=O)(=O)O"),
    ("Epolamine", "OCCN1CCCC1"),
    ("Erbumine", "CC(C)(C)N"),
    ("Estolate", "CCCCCCCCCCCCOS(=O)(=O)O"),
    ("Esylate", "CCS(=O)(=O)O"),
    ("Ethylsulfate", "CCOS(=O)(=O)O"),
    ("Fluoride", "F"),
    ("Formate", "OC=O"),
    ("Gluceptate", "OCC(O)C(O)C(O)C(O)C(O)C(=O)O"),
    ("Gluconate", "OCC(O)C(O)C(O)C(O)C(=O)O"),
    ("Glucuronate", "OC1OC(C(O)C(O)C1O)C(=O)O"),
    ("Glutamate", "NC(CCC(=O)O)C(=O)O"),
    ("Glycerate", "OCC(O)CO"),
    ("Glycerophosphate", "OCC(O)COP(=O)(O)O"),
    ("Hexafluorophosphate", "F[P](F)(F)(F)(F)F"),
    ("Hypophosphite", "OP=O"),
    ("Iodide", "I"),
    ("Isethionate", "OCCS(=O)(=O)O"),
    ("Potassium", "[K]"),
    ("Lactate", "CC(O)C(=O)O"),
    ("Lactobionate", "OCC(O)C(OC1OC(CO)C(O)C(O)C1O)C(O)C(O)C(=O)O"),
    ("Lithium", "[Li]"),
    ("Lysine", "NCCCCC(N)C(=O)O"),
    ("Malate", "OC(CC(=O)O)C(=O)O"),
    ("Maleate", "OC(=O)/C=C/C(=O)O"),
    ("Mesylate", "CS(=O)(=O)O"),
    ("Metaphosphate", "OP(=O)=O"),
    ("Methosulfate", "COS(=O)(=O)O"),
    ("Magnesium", "[Mg]"),
    ("Monofluorophosphate", "OP(=O)(O)F"),
    ("Sodium", "[Na]"),
    ("Napadisilate", "OS(=O)(=O)c1cccc2c(cccc12)S(=O)(=O)O"),
    ("Napsylate", "OS(=O)(=O)c1ccc2ccccc2c1"),
    ("Nitrate", "O[N](=O)O"),
    ("Oxalate", "OC(=O)C(=O)O"),
    ("Palmitate", "CCCCCCCCCCCCCCCC(=O)O"),
    ("Pamoate", "OC(=O)c1cc2ccccc2c(Cc3c(O)c(cc4ccccc34)C(=O)O)c1O"),
    ("Perchlorate", "OCl(=O)(=O)=O"),
    ("Phosphanilate", "Nc1ccc(cc1)P(=O)(O)O"),
    ("Phosphate", "OP(=O)(O)O"),
    ("Picrate", "Oc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"),
    ("Piperazine", "C1CNCCN1"),
    ("Propylene Glycol", "CC(O)CO"),
    ("Saccharin", "O=C1NS(=O)(=O)c2ccccc12"),
    ("Salicylate", "OC(=O)c1ccccc1O"),
    ("Silver", "[Ag]"),
    ("Strontium", "[Sr]"),
    ("Succinate", "OC(=O)CCC(=O)O"),
    ("Sulfate", "OS(=O)(=O)O"),
    ("Sulfosalicylate", "OC(=O)c1cccc(c1O)S(=O)(=O)O"),
    ("Sulphide", "[S-2]"),
    ("Terephthalate", "OC(=O)c1ccc(cc1)C(=O)O"),
    ("Tosylate", "Cc1ccc(cc1)S(=O)(=O)O"),
    ("Triclofenate", "Oc1cc(Cl)c(Cl)cc1Cl"),
    ("Triethylamine", "CCN(CC)CC"),
    ("Trifenatate", "OC(=O)C(c1ccccc1)(c2ccccc2)c3ccccc3"),
    ("Triflutate", "OC(=O)C(F)(F)F"),
    ("Tromethamine", "NC(CO)(CO)CO"),
    ("Buciclate", "CCCCC1CCC(CC1)C(=O)O"),
    ("Butyrate", "CCCC(=O)O"),
    ("Caproate", "CCCCCC(=O)O"),
    ("Cyclotate", "CC12CCC(CC1)(C=C2)C(=O)O"),
    ("Cypionate", "OC(=O)CCC1CCCC1"),
    ("Daproate", "CN(C)CCC(=O)O"),
    ("EDTA", "OC(=O)CN(CCN(CC(=O)O)CC(=O)O)CC(=O)O"),
    ("Elaidate", "OC(=O)C/C=C/CCCCCCC"),
    ("Enanthate", "CCCCCCC(=O)O"),
    ("Etabonate", "CCOC(=O)O"),
    ("Ethanediol", "COCCO"),
    ("Etiprate", "OC(=O)CNC(=O)c1ccccc1"),
    ("Etzadroxil", "CCC(CC)C(=O)OCO"),
    ("Fostedate", "CCCCCCCCCCCCCCOP(=O)(O)O"),
    ("Furoate", "OC(=O)c1occc1"),
    ("Hybenzate", "OC(=O)c1ccccc1C(=O)c2ccc(O)cc2"),
    ("Laurate", "CCCCCCCCCCCC(=O)O"),
    ("Mebutate", "CC=C(C)C(=O)O"),
    ("Mepesuccinate", "COC(=O)CC(O)(CCCC(C)(C)O)C(=O)O"),
    ("Metazoate", "OC(=O)c1cccc(c1)S(=O)(=O)O"),
    ("Methionil", "CSCCC(N)C(=O)C"),
    ("Nicotinate", "OC(=O)c1cccnc1"),
    ("Peroxide", "OO"),
    ("Phenpropionate", "OC(=O)CCc1ccccc1"),
    ("Phenylacetate", "OC(=O)Cc1ccccc1"),
    ("Pivalate", "CC(C)(C)C(=O)O"),
    ("Propionate", "CCC(=O)O"),
    ("Tebutate", "CC(C)(C)CC(=O)O"),
    ("Trolamine", "OCCN(CCO)CCO"),
    ("Undecylate", "CCCCCCCCCCC(=O)O"),
    ("Undecylenate", "OC(=O)CCCCCCCCC=C"),
    ("Valerate", "CCCCC(=O)O"),
    ("Water", "O"),
    ("Xinafoate", "OC(=O)c1ccc2ccccc2c1O"),
    ("Zinc", "[Zn]"),
    ("Imidazole", "c1c[nH]cn1"),
    ("4-(2-Hydroxyethyl)morpholine", "OCCN1CCOCC1"),
    ("4-Acetamidobenzoic acid", "CC(=O)Nc1ccc(C(=O)O)cc1"),
    ("Camphoric acid", "CC1(C)C2CCC1(C(=O)O)C(=O)C2C(=O)O"),
    ("Capric acid", "CCCCCCCCCC(=O)O"),
    ("Caprylic acid", "CCCCCCCC(=O)O"),
    ("Cinnamic acid", "OC(=O)/C=C/c1ccccc1"),
    ("Mucic acid", "OCC(O)C(O)C(O)C(O)C(=O)O"),
    ("Gentisic acid", "O=C(O)C1=CC(O)=CC(O)=C1"),
    ("Glutaric acid", "OC(=O)CCCC(=O)O"),
    ("2-Oxoglutaric acid", "OC(=O)CCC(=O)C(=O)O"),
    ("Glycolic acid", "OCC(=O)O"),
    ("Isobutyric acid", "CC(C)C(=O)O"),
    ("Mandelic acid", "OC(C(=O)O)c1ccccc1"),
    ("Orotic acid", "O=C(O)C1=NC(=O)NC=N1"),
    ("Pyroglutamic acid", "OC(=O)C1CCC(=O)N1"),
    ("Tartrate", "OC(C(O)C(=O)O)C(=O)O"),
    ("Thiocyanic acid", "SC#N"),
    ("Methyl Iodide", "CI"),
    ("Sulfurous Acid", "OS(=O)O"),
    ("Dicyclohexylamine", "C1CCC(CC1)NC2CCCCC2"),
    ("Triflate", "OS(=O)(=O)C(F)(F)F"),
    ("Mesitylene sulfonate", "Cc1c(C)cc(C)c(S(=O)(=O)O)c1"),
    ("Malonic acid", "OC(=O)CC(=O)O"),
    ("Fluorosulfuric acid", "OS(=O)(=O)F"),
    ("Acetylsulfate", "CC(=O)OS(=O)(=O)O"),
    ("Proton", "[H]"),
    ("Rubidium", "[Rb]"),
    ("Cesium", "[Cs]"),
    ("Francium", "[Fr]"),
    ("Beryllium", "[Be]"),
    ("Radium", "[Ra]"),
    ("Glucuronate open form", "C(=O)C(O)C(O)C(O)C(O)C(=O)O"),
    ("Dimepranol", "CC(O)CN(C)C"),
)

#: The alkoxide pattern of ChEMBL's ``normalize_mol`` (Li/Na/K bonded to
#: N/O -- split into ions before the normalizer proper).
_ALKOXIDE_PATTERN = "[Li,Na,K;+0]-[#7,#8;+0]"

#: The tartrate SMARTS of ``flatten_tartrate_mol`` (stereo collapsed on
#: the two marked centres).
_TARTRATE_SMARTS = "OC(=O)C(O)C(O)C(=O)O"


# ---------------------------------------------------------------------------
# options / outcome records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StandardizeOptions:
    """Every toggle the five chains respect, one name across profiles.

    The defaults match the card's sensible reading of the directive
    (salts out, charges neutralised, groups normalised, metals
    disconnected, stereochemistry kept, tautomer untouched) -- each
    knob is the card's checkbox.

    ``protonate`` / ``min_ph`` / ``max_ph`` / ``pka_std_range`` feed the
    dimorphite_dl enumerator: when ``protonate`` is set the record's
    ``protonation_states`` carries the enumerated SMILES (the
    canonical SMILES stays the main chain's answer; the states are the
    card's second column).
    """

    profile: str = "rdkit"
    remove_fragments: bool = True
    uncharge: bool = True
    normalize: bool = True
    disconnect_metals: bool = True
    reionize: bool = True
    tautomer: bool = False
    keep_stereo: bool = True
    protonate: bool = False
    min_ph: float = 6.4
    max_ph: float = 8.4
    pka_std_range: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "remove_fragments": self.remove_fragments,
            "uncharge": self.uncharge,
            "normalize": self.normalize,
            "disconnect_metals": self.disconnect_metals,
            "reionize": self.reionize,
            "tautomer": self.tautomer,
            "keep_stereo": self.keep_stereo,
            "protonate": self.protonate,
            "min_ph": self.min_ph,
            "max_ph": self.max_ph,
            "pka_std_range": self.pka_std_range,
        }


@dataclass
class StandardizeOutcome:
    """One molecule through one chain -- the per-row answer.

    ``steps`` is the ordered chain actually executed (the card's
    tooltip / the CSV's ``steps`` column); ``flags`` marks what
    *changed* (fragments-removed, uncharged, normalized,
    metal-disconnected, tautomer-canonicalised, chembl-excluded ... --
    a diff against the input, not a mere echo of the toggles);
    ``error`` carries the parse/standardisation failure for invalid
    rows -- the no-silent-drop rule: every row answers, one way or
    another.
    """

    original: str
    canonical: str | None
    profile: str
    status: str
    steps: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    error: str | None = None
    protonation_states: list[str] = field(default_factory=list)


@dataclass
class StandardizationRecord:
    """One CSV row's outcome (identifier + provenance)."""

    row: int
    identifier: str
    original: str
    canonical: str | None
    status: str
    steps: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    protonation_states: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class StandardizationProfile:
    """A registered chain: key, label, reference repo, builder."""

    key: str
    label: str
    reference: str
    builder: Callable[..., Any]


def _require_rdkit() -> Any:
    """Import RDKit or raise the actionable dependency error."""
    try:
        from rdkit import Chem
    except ImportError as exc:  # pragma: no cover - env guard
        raise DependencyError(
            "RDKit is required for CSV SMILES standardization. From the toolkit package root run: "
            "python -m pip install -e '.[chem]'"
        ) from exc
    from rdkit import Chem

    return Chem


def _parse_smiles(smiles: str) -> Any:
    """Parse a SMILES string quietly; ``None`` when unparseable."""
    Chem = _require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    DisableLog("rdApp.error")
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:  # noqa: BLE001 - hostile strings return None anyway
        return None
    finally:
        EnableLog("rdApp.error")


def _canonical(mol: Any, isomeric: bool = True) -> str:
    """Canonical SMILES of a mol (isomeric per the stereo option)."""
    Chem = _require_rdkit()
    from rdkit.rdBase import DisableLog, EnableLog

    DisableLog("rdApp.error")
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:  # noqa: BLE001 - unsanitisable output
        return ""
    finally:
        EnableLog("rdApp.error")


def _snapshot(mol: Any) -> tuple[str, int, tuple[int, ...]]:
    """The diff basis for the change flags: canonical SMILES, fragment
    count, the sorted formal-charge multiset."""
    Chem = _require_rdkit()
    return (
        _canonical(mol),
        len(Chem.GetMolFrags(mol)),
        tuple(sorted(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    )


def _note_changes(
    before: tuple[str, int, tuple[int, ...]],
    after: tuple[str, int, tuple[int, ...]],
    flags: list[str],
) -> None:
    """Flag the semantic changes between two snapshots (the shared
    diff of every chain: fragments-removed / uncharged)."""
    if after[0] != before[0]:
        if after[1] < before[1]:
            flags.append("fragments-removed")
        if after[2] != before[2]:
            flags.append("uncharged")


# ---------------------------------------------------------------------------
# profile 1 -- rdkit/rdkit Code/GraphMol/MolStandardize (the subtree link)
# ---------------------------------------------------------------------------


def _rdkit_chain(mol: Any, options: StandardizeOptions) -> Any:
    """The MolStandardize canonical chain, per the module's own docs.

    ``MetalDisconnector`` -> ``Normalizer`` -> ``Reionizer`` -> the
    ``FragmentParent`` / ``ChargeParent`` leg (the owner's salts-out
    tick) -> ``AssignStereochemistry`` -> optional canonical tautomer.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms

    steps: list[str] = []
    flags: list[str] = []
    if options.disconnect_metals:
        before = _snapshot(mol)
        mol = rms.MetalDisconnector().Disconnect(mol)
        steps.append("MetalDisconnector")
        if _snapshot(mol) != before:
            flags.append("metal-disconnected")
    if options.normalize:
        before = _snapshot(mol)
        mol = rms.Normalizer().normalize(mol)
        steps.append("Normalizer")
        if _snapshot(mol) != before:
            flags.append("normalized")
    if options.reionize:
        mol = rms.Reionizer().reionize(mol)
        steps.append("Reionizer")
    if options.remove_fragments:
        before = _snapshot(mol)
        if options.uncharge:
            mol = rms.ChargeParent(mol)  # FragmentParent + Uncharger
            steps.append("ChargeParent")
        else:
            mol = rms.FragmentParent(mol)
            steps.append("FragmentParent")
        _note_changes(before, _snapshot(mol), flags)
    elif options.uncharge:
        before = _snapshot(mol)
        mol = rms.Uncharger().uncharge(mol)
        steps.append("Uncharger")
        _note_changes(before, _snapshot(mol), flags)
    if options.tautomer:
        before = _snapshot(mol)
        mol = rms.TautomerEnumerator().Canonicalize(mol)
        steps.append("CanonicalTautomer")
        if _snapshot(mol) != before:
            flags.append("tautomer-canonicalised")
    if options.keep_stereo:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        steps.append("AssignStereochemistry")
    else:
        Chem.RemoveStereochemistry(mol)
        steps.append("RemoveStereochemistry")
    return mol, steps, flags


# ---------------------------------------------------------------------------
# profile 2 -- mcs07/MolVS
# ---------------------------------------------------------------------------


def _molvs_chain(mol: Any, options: StandardizeOptions) -> Any:
    """MolVS ``Standardizer.standardize`` verbatim + the parents.

    ``SanitizeMol`` -> ``RemoveHs`` -> ``MetalDisconnector`` ->
    ``Normalizer`` -> ``Reionizer`` -> ``AssignStereochemistry(force,
    cleanIt=True)`` -- MolVS's own order (note sanitize *first*, the
    one step the other profiles do differently).  The fragment /
    charge legs follow the same absorbed RDKit classes MolVS itself
    became (``LargestFragmentChooser`` + ``Uncharger``).
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms

    steps: list[str] = ["SanitizeMol", "RemoveHs"]
    flags: list[str] = []
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - hostile valences
        _LOG.debug("MolVS sanitize deferred: %s", exc)
    mol = Chem.RemoveHs(mol)
    if options.disconnect_metals:
        before = _snapshot(mol)
        mol = rms.MetalDisconnector().Disconnect(mol)
        steps.append("MetalDisconnector")
        if _snapshot(mol) != before:
            flags.append("metal-disconnected")
    if options.normalize:
        before = _snapshot(mol)
        mol = rms.Normalizer().normalize(mol)
        steps.append("Normalizer")
        if _snapshot(mol) != before:
            flags.append("normalized")
    if options.reionize:
        mol = rms.Reionizer().reionize(mol)
        steps.append("Reionizer")
    if options.remove_fragments:
        before = _snapshot(mol)
        chooser = rms.LargestFragmentChooser()
        mol = chooser.choose(mol)
        steps.append("LargestFragmentChooser")
        _note_changes(before, _snapshot(mol), flags)
    if options.uncharge:
        before = _snapshot(mol)
        mol = rms.Uncharger().uncharge(mol)
        steps.append("Uncharger")
        _note_changes(before, _snapshot(mol), flags)
    if options.tautomer:
        before = _snapshot(mol)
        mol = rms.TautomerEnumerator().Canonicalize(mol)
        steps.append("CanonicalTautomer")
        if _snapshot(mol) != before:
            flags.append("tautomer-canonicalised")
    if options.keep_stereo:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        steps.append("AssignStereochemistry")
    else:
        Chem.RemoveStereochemistry(mol)
        steps.append("RemoveStereochemistry")
    return mol, steps, flags


# ---------------------------------------------------------------------------
# profile 3 -- chembl/ChEMBL_Structure_Pipeline
# ---------------------------------------------------------------------------


def _chembl_excluded(mol: Any) -> bool:
    """The ChEMBL ``exclude_flag``: a metal-list atom or a borane
    (more than 7 borons) -- standardization is skipped, the molecule
    reported as-is (their rule: depiction trouble)."""
    borons = 0
    metallic = False
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in CHEMBL_EXCLUDE_METALS:
            metallic = True
        if symbol == "B":
            borons += 1
    return metallic or borons > 7


def _chembl_remove_hs(mol: Any) -> Any:
    """ChEMBL's ``remove_hs_from_mol`` -- the SENTINEL-isotope trick.

    Preserve rules (their docstring): wedged/dashed bonds, Hs on atoms
    with tetrahedral stereo, Hs on over-valenced atoms that are not
    simply protonated, Hs on atoms with three or more ring bonds.
    Everything else goes.
    """
    from rdkit import Chem

    m = Chem.Mol(mol)
    Chem.FastFindRings(m)
    if m.NeedsUpdatePropertyCache():
        m.UpdatePropertyCache(strict=False)
    sentinel = 100
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 1 and not atom.GetIsotope():
            nbr = atom.GetNeighbors()[0]
            bnd = atom.GetBonds()[0]
            preserve = False
            if bnd.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH) or (
                bnd.HasProp("_MolFileBondStereo")
                and bnd.GetUnsignedProp("_MolFileBondStereo") in (1, 6)
            ):
                preserve = True
            else:
                nbr_valence = sum(b.GetBondTypeAsDouble() for b in nbr.GetBonds())
                default = Chem.GetPeriodicTable().GetDefaultValence(nbr.GetAtomicNum())
                is_protonated = nbr.GetFormalCharge() == 1 and nbr_valence == default + 1
                if nbr.GetChiralTag() in (
                    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                ):
                    preserve = True
                elif not is_protonated:
                    if nbr_valence > default:
                        preserve = True
                    else:
                        ring_bonds = [
                            b for b in nbr.GetBonds() if m.GetRingInfo().NumBondRings(b.GetIdx())
                        ]
                        if len(ring_bonds) >= 3:
                            preserve = True
            if preserve:
                atom.SetIsotope(sentinel)
    res = Chem.RemoveHs(m, sanitize=False)
    for atom in res.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetIsotope() == sentinel:
            atom.SetIsotope(0)
    return res


def _chembl_normalize(mol: Any) -> Any:
    """ChEMBL's ``normalize_mol``: the alkoxide split, then their
    twelve-SMIRKS normalizer (built through ``NormalizerFromData``,
    their own construction)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms

    m = Chem.Mol(mol)
    Chem.FastFindRings(m)
    alkoxide = Chem.MolFromSmarts(_ALKOXIDE_PATTERN)
    if alkoxide is not None and m.HasSubstructMatch(alkoxide):
        rw = Chem.RWMol(m)
        for match in rw.GetSubstructMatches(alkoxide):
            rw.RemoveBond(match[0], match[1])
            rw.GetAtomWithIdx(match[0]).SetFormalCharge(1)
            rw.GetAtomWithIdx(match[1]).SetFormalCharge(-1)
        m = rw
    params = rms.CleanupParameters()
    normalizer = rms.NormalizerFromData(CHEMBL_NORMALIZATION_TRANSFORMS, params)
    return normalizer.normalize(m)


def _chembl_uncharge(mol: Any) -> Any:
    """ChEMBL's ``uncharge_mol``: ``Uncharger(canonicalOrder=True)``
    then the property-cache refresh (their exact order)."""
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms

    res = rms.Uncharger(canonicalOrder=True).uncharge(mol)
    res.UpdatePropertyCache(strict=False)
    return res


def _chembl_flatten_tartrate(mol: Any) -> Any:
    """ChEMBL's ``flatten_tartrate_mol`` -- stereo collapsed on free
    tartrate fragments (the conformer-free SMILES leg: only the
    chiral tags are cleared, no drawing cleanup applies)."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    tartrate = Chem.MolFromSmarts(_TARTRATE_SMARTS)
    if tartrate is None:
        return m
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.adjustDegree = True  # type: ignore[assignment]  # the stubs mistype it
    params.adjustDegreeFlags = Chem.AdjustQueryWhichFlags.ADJUST_IGNORENONE
    query = Chem.AdjustQueryProperties(tartrate, params)
    matches = m.GetSubstructMatches(query)
    if matches:
        for match in matches:
            m.GetAtomWithIdx(match[3]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            m.GetAtomWithIdx(match[5]).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return m


def _chembl_fragment_parent(mol: Any) -> Any:
    """ChEMBL's ``get_fragment_parent_mol`` -- the solvent/salt strip.

    Their rule, quoted: split to fragments (unsanitised, RemoveHs each,
    aromaticity set), drop fragments matching the solvent list (atom
    and bond count equal + substructure match); if everything is a
    solvent, return the input; drop fragments matching the salt list
    the same way; if everything is a salt, keep everything (a pure
    salt pair like sodium acetate comes back whole -- their own
    fallback, the reason the chembl doctest uses a drug salt); the
    survivors recombine.
    """
    from rdkit import Chem

    input_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    frags = []
    for frag in input_frags:
        frag = Chem.RemoveHs(frag, sanitize=False)
        frag.UpdatePropertyCache(strict=False)
        Chem.SetAromaticity(frag)
        frags.append(frag)
    if len(frags) <= 1:
        return mol  # nothing to strip (their single-fragment fast path)

    keep = [1] * len(frags)
    for _name, smi in CHEMBL_SOLVENTS:
        solv = Chem.MolFromSmarts(smi)
        if solv is None:
            continue
        for i, frag in enumerate(frags):
            if (
                keep[i]
                and frag.GetNumAtoms() == solv.GetNumAtoms()
                and frag.GetNumBonds() == solv.GetNumBonds()
                and frag.HasSubstructMatch(solv)
            ):
                keep[i] = 0
        if not max(keep):
            break
    if not max(keep):
        return mol  # everything is a solvent: the input stands (their rule)

    survivors = [frags[i] for i, v in enumerate(keep) if v]
    keep = [1] * len(survivors)
    for _name, smi in CHEMBL_SALTS:
        salt = Chem.MolFromSmarts(smi)
        if salt is None:
            continue
        for i, frag in enumerate(survivors):
            if (
                keep[i]
                and frag.GetNumAtoms() == salt.GetNumAtoms()
                and frag.GetNumBonds() == salt.GetNumBonds()
                and frag.HasSubstructMatch(salt)
            ):
                keep[i] = 0
        if not max(keep):
            break
    if not max(keep):
        keep = [1] * len(survivors)  # everything is a salt: keep everything

    final = [survivors[i] for i, v in enumerate(keep) if v]
    if len(final) == 1:
        return final[0]
    res = final[0]
    for frag in final[1:]:
        res = Chem.CombineMols(res, frag)
    return res


def _chembl_chain(mol: Any, options: StandardizeOptions) -> Any:
    """ChEMBL ``standardize_mol`` + ``get_parent_mol`` in order."""
    from rdkit import Chem

    steps: list[str] = []
    flags: list[str] = []
    if _chembl_excluded(mol):
        return mol, ["excluded"], ["chembl-excluded"]

    working = Chem.Mol(mol)
    working.UpdatePropertyCache(strict=False)
    steps.append("UpdatePropertyCache")
    Chem.ClearMolSubstanceGroups(working)
    steps.append("ClearMolSubstanceGroups")
    try:
        working = Chem.Mol(working)
        Chem.Kekulize(working)
        steps.append("Kekulize")
    except Exception as exc:  # noqa: BLE001 - unkekulizable input
        _LOG.debug("ChEMBL kekulize skipped: %s", exc)
        steps.append("Kekulize(skipped)")
    working = _chembl_remove_hs(working)
    steps.append("RemoveHs(ChEMBL)")
    if options.normalize:
        before = _snapshot(working)
        working = _chembl_normalize(working)
        steps.append("Normalize(ChEMBL)")
        if _snapshot(working) != before:
            flags.append("normalized")
    if options.uncharge:
        before = _snapshot(working)
        working = _chembl_uncharge(working)
        steps.append("Uncharger(canonicalOrder)")
        _note_changes(before, _snapshot(working), flags)
    working = _chembl_flatten_tartrate(working)
    steps.append("FlattenTartrate")
    if options.remove_fragments:
        before = _snapshot(working)
        working = _chembl_fragment_parent(working)
        steps.append("FragmentParent(solvents+salts)")
        _note_changes(before, _snapshot(working), flags)
        if options.uncharge:
            working = _chembl_uncharge(working)
    try:
        Chem.SanitizeMol(working)
    except Exception as exc:
        _LOG.warning("ChEMBL final sanitize failed: %s", exc)
        raise
    steps.append("SanitizeMol")
    if not options.keep_stereo:
        Chem.RemoveStereochemistry(working)
        steps.append("RemoveStereochemistry")
    if options.tautomer:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rms

        before = _snapshot(working)
        working = rms.TautomerEnumerator().Canonicalize(working)
        steps.append("CanonicalTautomer")
        if _snapshot(working) != before:
            flags.append("tautomer-canonicalised")
    return working, steps, flags


# ---------------------------------------------------------------------------
# profile 4 -- datamol-io/datamol
# ---------------------------------------------------------------------------


def _datamol_chain(mol: Any, options: StandardizeOptions) -> Any:
    """datamol's ``dm.standardize_mol`` with the library defaults.

    Their defaults: ``disconnect_metals=False, normalize=True,
    reionize=True, uncharge=False, stereo=True`` -- the card's ticks
    override each (the fragment front step is the toolkit's own
    documented addition; datamol's standardize itself never touches
    fragments)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms

    steps: list[str] = []
    flags: list[str] = []
    mol = Chem.Mol(mol)  # datamol's copy_mol discipline
    if options.remove_fragments:
        before = _snapshot(mol)
        mol = rms.LargestFragmentChooser().choose(mol)
        steps.append("LargestFragmentChooser")
        _note_changes(before, _snapshot(mol), flags)
    if options.disconnect_metals:
        before = _snapshot(mol)
        mol = rms.MetalDisconnector().Disconnect(mol)
        steps.append("MetalDisconnector")
        if _snapshot(mol) != before:
            flags.append("metal-disconnected")
    if options.normalize:
        before = _snapshot(mol)
        mol = rms.Normalizer().normalize(mol)
        steps.append("Normalizer")
        if _snapshot(mol) != before:
            flags.append("normalized")
    if options.reionize:
        mol = rms.Reionizer().reionize(mol)
        steps.append("Reionizer")
    if options.uncharge:
        before = _snapshot(mol)
        mol = rms.Uncharger().uncharge(mol)
        steps.append("Uncharger")
        _note_changes(before, _snapshot(mol), flags)
    if options.tautomer:
        before = _snapshot(mol)
        mol = rms.TautomerEnumerator().Canonicalize(mol)
        steps.append("CanonicalTautomer")
        if _snapshot(mol) != before:
            flags.append("tautomer-canonicalised")
    if options.keep_stereo:
        Chem.AssignStereochemistry(mol, force=False, cleanIt=True)
        steps.append("AssignStereochemistry")
    else:
        Chem.RemoveStereochemistry(mol)
        steps.append("RemoveStereochemistry")
    return mol, steps, flags


# ---------------------------------------------------------------------------
# profile 5 -- the REINVENT4 default (absorbed round 100x-164)
# ---------------------------------------------------------------------------


def _reinvent_chain(mol: Any, options: StandardizeOptions) -> Any:
    """The REINVENT4 ``default`` chain via the existing absorption."""
    from pharmaml.chem.standardize import reinvent_standardize_mol

    steps = [
        "largest-fragment",
        "RemoveHs",
        "SaltRemover",
        "neutralize-fixpoint" if options.uncharge else "neutralize-off",
        "cleanup",
        "unwanted-patterns",
        "size/element-window",
    ]
    flags: list[str] = []
    before = _snapshot(mol)
    result = reinvent_standardize_mol(mol, neutralize=options.uncharge)
    if result is None:
        return None, steps, flags
    _note_changes(before, _snapshot(result), flags)
    if options.tautomer:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rms

        before = _snapshot(result)
        result = rms.TautomerEnumerator().Canonicalize(result)
        steps.append("CanonicalTautomer")
        if _snapshot(result) != before:
            flags.append("tautomer-canonicalised")
    return result, steps, flags


# ---------------------------------------------------------------------------
# profile 6 -- plain: sanitize + canonicalise only (the baseline)
# ---------------------------------------------------------------------------


def _plain_chain(mol: Any, options: StandardizeOptions) -> Any:
    """Sanitize + canonical SMILES, nothing else -- the honest baseline
    the other chains are measured against."""
    from rdkit import Chem

    steps = ["SanitizeMol"]
    flags: list[str] = []
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - hostile valences
        _LOG.debug("plain sanitize deferred: %s", exc)
    if options.tautomer:
        from rdkit.Chem.MolStandardize import rdMolStandardize as rms

        before = _snapshot(mol)
        mol = rms.TautomerEnumerator().Canonicalize(mol)
        steps.append("CanonicalTautomer")
        if _snapshot(mol) != before:
            flags.append("tautomer-canonicalised")
    if not options.keep_stereo:
        Chem.RemoveStereochemistry(mol)
        steps.append("RemoveStereochemistry")
    return mol, steps, flags


PROFILES: dict[str, StandardizationProfile] = {
    "rdkit": StandardizationProfile(
        "rdkit",
        "RDKit MolStandardize",
        "rdkit/rdkit · Code/GraphMol/MolStandardize",
        _rdkit_chain,
    ),
    "molvs": StandardizationProfile(
        "molvs",
        "MolVS standardize",
        "mcs07/MolVS",
        _molvs_chain,
    ),
    "chembl": StandardizationProfile(
        "chembl",
        "ChEMBL structure pipeline",
        "chembl/ChEMBL_Structure_Pipeline",
        _chembl_chain,
    ),
    "datamol": StandardizationProfile(
        "datamol",
        "datamol standardize_mol",
        "datamol-io/datamol",
        _datamol_chain,
    ),
    "reinvent": StandardizationProfile(
        "reinvent",
        "REINVENT4 default",
        "absorbed round 100x-164",
        _reinvent_chain,
    ),
    "plain": StandardizationProfile(
        "plain",
        "Plain canonicalisation",
        "sanitize + canonical SMILES only",
        _plain_chain,
    ),
}


# ---------------------------------------------------------------------------
# the dimorphite_dl protonation enumerator (UnixJunkie/dimorphite_dl)
# ---------------------------------------------------------------------------


def _define_protonation_state(
    mean: float, std: float, min_ph: float, max_ph: float
) -> str:
    """dimorphite_dl's ``define_protonation_state`` -- the pH overlap
    test on the mean ± std window: BOTH when the pKa range overlaps
    the pH range, PROTONATED when the pKa sits above, DEPROTONATED
    when below."""
    min_pka = mean - std
    max_pka = mean + std
    if min_pka <= max_ph and min_ph <= max_pka:
        return "BOTH"
    if mean > max_ph:
        return "PROTONATED"
    return "DEPROTONATED"


def _dimorphite_neutralize(mol: Any) -> Any:
    """dimorphite's ``UtilFuncs.neutralize_mol`` -- the six-reaction
    fixpoint (O- monovalent -> OH, N+H -> N, O- divalent -> O, N+
    trivalent -> N, N- divalent -> NH, the bad-azide R-N-N#N ->
    R-N=[N+]=N-H), resanitised with ``catchErrors`` and rejected when
    the sanitize string is not SANITIZE_NONE."""
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions

    reactions: list[list[Any]] = []
    for reactant, product in _NEUTRALIZE_REACTIONS:
        reactions.append([reactant, product, Chem.MolFromSmarts(reactant), None])
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)
    while True:
        current = None
        for entry in reactions:
            reactant, _product, query, rxn = entry
            if query is not None and mol.HasSubstructMatch(query):
                if rxn is None:
                    rxn = rdChemReactions.ReactionFromSmarts(reactant + ">>" + entry[1])
                    entry[3] = rxn
                current = entry[3]
                break
        if current is None:
            break
        products = current.RunReactants((mol,))
        if not products:
            break
        mol = products[0][0]
        mol.UpdatePropertyCache(strict=False)
    sanitize_result = Chem.SanitizeMol(mol, catchErrors=True)
    if sanitize_result.name != "SANITIZE_NONE":
        return None
    return mol


def _protonation_sites(mol: Any, rules: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """dimorphite's ``get_prot_sites_and_target_states`` -- list-order
    matching with the protect/unprotect discipline (an atom claimed by
    an earlier rule is never re-claimed by a later one) and the
    unique-site register."""
    sites: list[tuple[int, str, str]] = []
    protected: set[int] = set()
    for rule in rules:
        query = rule["mol"]
        if query is None or not mol.HasSubstructMatch(query):
            continue
        for match in mol.GetSubstructMatches(query):
            if any(idx in protected for idx in match):
                continue
            for site_number, state in rule["states"]:
                candidate = (match[int(site_number)], state, rule["name"])
                if candidate not in sites:
                    sites.append(candidate)
            protected.update(match)
    return sites


def _set_protonation_charge(
    mols: list[Any], idx: int, charges: list[int], site_name: str
) -> list[Any]:
    """dimorphite's ``set_protonation_charge`` verbatim -- the N charge
    ladder (``+1`` above the O/S one, undone for the ``*``-named
    moieties whose acidic form is neutral), the explicit-H table by
    bond order, and the ``[nH-]`` -> ``[n-]`` aromatic fix."""
    from rdkit import Chem

    output: list[Any] = []
    for charge in charges:
        nitrogen_charge = charge + 1
        if "*" in site_name:
            nitrogen_charge -= 1
        for mol in mols:
            try:
                copy = Chem.RemoveHs(Chem.Mol(mol))
            except Exception:  # noqa: BLE001, S112 — per-site H strips fail on hostile valences
                continue
            atom = copy.GetAtomWithIdx(idx)
            bond_total = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())
            element = atom.GetAtomicNum()
            if element == 7:
                atom.SetFormalCharge(nitrogen_charge)
                if nitrogen_charge == 1 and bond_total == 1:
                    atom.SetNumExplicitHs(3)
                elif nitrogen_charge == 1 and bond_total == 2:
                    atom.SetNumExplicitHs(2)
                elif nitrogen_charge == 1 and bond_total == 3:
                    atom.SetNumExplicitHs(1)
                elif nitrogen_charge == 0 and bond_total == 1:
                    atom.SetNumExplicitHs(2)
                elif nitrogen_charge == 0 and bond_total == 2:
                    atom.SetNumExplicitHs(1)
                elif nitrogen_charge == -1 and bond_total == 2:
                    atom.SetNumExplicitHs(0)
                elif nitrogen_charge == -1 and bond_total == 1:
                    atom.SetNumExplicitHs(1)
            else:
                atom.SetFormalCharge(charge)
                if element in (8, 16):
                    if charge == 0 and bond_total == 1:
                        atom.SetNumExplicitHs(1)
                    elif charge == -1 and bond_total == 1:
                        atom.SetNumExplicitHs(0)
            copy.UpdatePropertyCache(strict=False)
            if "[nH-]" in _canonical(copy):
                atom.SetNumExplicitHs(0)
                copy.UpdatePropertyCache(strict=False)
            output.append(copy)
    return output


def protonate_smiles(
    smiles: str,
    min_ph: float = 6.4,
    max_ph: float = 8.4,
    pka_std_range: float = 1.0,
) -> list[str]:
    """The dimorphite_dl protonation enumerator for one SMILES.

    The original pipeline: neutralise (their reaction fixpoint), strip
    Hs, canonicalise; then match the 41 rules in list order (earlier
    rules claim their atoms -- the protect discipline), decide each
    site's state from the pH window vs the pKa mean ± std
    (:func:`_define_protonation_state`), apply the charge/H ladder
    (:func:`_set_protonation_charge`) expanding BOTH sites into both
    forms, and return the unique canonical SMILES of every state.

    Parameters
    ----------
    smiles:
        The input SMILES (any salt/charge form -- the neutralisation
        fixpoint runs first).
    min_ph, max_ph:
        The pH window (dimorphite's defaults: 6.4-8.4, the
        physiological band).
    pka_std_range:
        How many std deviations of the predicted pKa to consider
        (their ``pka_precision``, default 1).

    Returns
    -------
    The sorted unique canonical SMILES of every protonation state the
    molecule can take inside the window.  Unparseable input yields an
    empty list (the caller's report records the parse failure).

    Examples
    --------
    >>> protonate_smiles("CC(=O)O", min_ph=6.4, max_ph=8.4)
    ['CC(=O)[O-]']
    >>> protonate_smiles("CC(=O)O", min_ph=1.0, max_ph=2.0)
    ['CC(=O)O']
    >>> protonate_smiles("not-a-smiles")
    []
    """
    mol = _parse_smiles(smiles)
    if mol is None:
        _LOG.warning("dimorphite protonation: unparseable SMILES %r", smiles)
        return []
    from rdkit import Chem

    mol = _dimorphite_neutralize(mol)
    if mol is None:
        _LOG.warning("dimorphite protonation: neutralisation failed %r", smiles)
        return []
    try:
        mol = Chem.RemoveHs(mol)
    except Exception:  # noqa: BLE001 - hostile valences
        return []
    mol = Chem.AddHs(Chem.Mol(mol))

    rules: list[dict[str, Any]] = []
    for name, smart, states in PROTONATION_SMARTS:
        query = Chem.MolFromSmarts(smart)
        if query is None:  # pragma: no cover - static table
            continue
        decided = [
            (site, _define_protonation_state(mean, std * pka_std_range, min_ph, max_ph))
            for site, mean, std in states
        ]
        rules.append({"name": name, "mol": query, "states": decided})

    sites = _protonation_sites(mol, rules)
    if not sites:
        return [_canonical(Chem.RemoveHs(Chem.Mol(mol)))]

    mols = [mol]
    for idx, target_state, site_name in sites:
        state_to_charge = {
            "DEPROTONATED": (-1,),
            "PROTONATED": (0,),
            "BOTH": (-1, 0),
        }
        charges = state_to_charge[target_state]
        mols = _set_protonation_charge(mols, idx, list(charges), site_name)
        if not mols:
            break

    results: set[str] = set()
    for candidate in mols:
        try:
            candidate = Chem.RemoveHs(candidate)
            candidate.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(candidate)
        except Exception:  # noqa: BLE001, S112 — some charge ladders are unsanitisable
            continue
        smi = _canonical(candidate)
        if smi:
            results.add(smi)
    return sorted(results)


# ---------------------------------------------------------------------------
# the OpenBabel slot -- the InChI round-trip
# ---------------------------------------------------------------------------


def inchi_roundtrip_check(smiles: str) -> tuple[bool, str]:
    """The cross-toolkit agreement check standing in OpenBabel's slot.

    OpenBabel is not a pharmaml dependency (the policy
    :mod:`pharmaml.chem.standardize` documents -- the rar MOL2 leg
    was dropped for exactly this reason), so the OB-style "a second,
    format-neutral toolkit agrees on this structure" test is served
    by the InChI round-trip: SMILES -> InChI -> back to RDKit ->
    canonical SMILES.  Agreement means the canonical form is stable
    under the format-neutral normalisation; disagreement is the
    tautomer/charge-normalisation warning the OB round-trip is used
    for in curation pipelines.

    Returns
    -------
    ``(agreed, other_toolkit_smiles)`` -- the flag and the InChI-side
    canonical SMILES (``""`` when the InChI conversion fails).

    Example
    -------
    >>> inchi_roundtrip_check("CCO")
    (True, 'CCO')
    >>> inchi_roundtrip_check("c1ccccc1O")
    (True, 'Oc1ccccc1')
    """
    mol = _parse_smiles(smiles)
    if mol is None:
        return False, ""
    from rdkit.Chem.inchi import MolFromInchi, MolToInchi

    try:
        inchi = MolToInchi(mol)
    except Exception:  # noqa: BLE001 - InChI failures are data, not crashes
        inchi = ""
    if not inchi:
        return False, ""
    back = MolFromInchi(inchi)
    if back is None:
        return False, ""
    other = _canonical(back)
    return other == _canonical(mol), other


# ---------------------------------------------------------------------------
# the front door
# ---------------------------------------------------------------------------


def standardize_smiles(smiles: str, **options: Any) -> StandardizeOutcome:
    """Standardise one SMILES to its canonical form -- every chain
    behind one call.

    Parameters
    ----------
    smiles:
        The input SMILES (any form: salts, charges, alternate tautomers,
        explicit Hs -- each chain handles what its reference library
        handles).
    profile:
        One of :data:`PROFILES` -- ``"rdkit"`` (MolStandardize chain,
        default), ``"molvs"``, ``"chembl"`` (the ChEMBL structure
        pipeline), ``"datamol"``, ``"reinvent"`` (REINVENT4 default),
        ``"plain"`` (sanitize + canonical only).
    remove_fragments, uncharge, normalize, disconnect_metals, reionize:
        The chain toggles (default all on -- salts out, charges
        neutralised, groups normalised).
    tautomer:
        Canonicalise the tautomer at the end (default off -- the
        heavier leg, the owner ticks it when the corpus demands it).
    keep_stereo:
        Keep stereochemistry (default on; off -> stereo-parent form).
    protonate, min_ph, max_ph, pka_std_range:
        Run :func:`protonate_smiles` after the chain; the outcome's
        ``protonation_states`` carries the enumerated forms (the
        canonical SMILES stays the main chain's answer).

    Returns
    -------
    A :class:`StandardizeOutcome` -- canonical SMILES, the executed
    steps, the change flags, or the parse error for invalid input
    (``status="invalid"``, never a silent drop).

    Examples
    --------
    >>> standardize_smiles("CC(=O)[O-].[Na+]").flags
    ['fragments-removed', 'uncharged']
    >>> standardize_smiles("C[C@H](O)C(=O)O", keep_stereo=False).canonical
    'CC(O)C(=O)O'
    >>> outcome = standardize_smiles("CC(=O)O", protonate=True,
    ...                              min_ph=11.0, max_ph=12.0)
    >>> outcome.protonation_states
    ['CC(=O)[O-]']
    >>> standardize_smiles("CC(=O)O", profile="no-such-profile")
    Traceback (most recent call last):
    ...
    pharmaml.exceptions.DataValidationError: Unknown standardization profile 'no-such-profile' (known: rdkit, molvs, chembl, datamol, reinvent, plain)
    """
    profile_name = str(options.get("profile", "rdkit"))
    if not isinstance(smiles, str) or not smiles.strip():
        return StandardizeOutcome(
            original=str(smiles),
            canonical=None,
            profile=profile_name,
            status="invalid",
            error="blank SMILES",
        )
    if profile_name not in PROFILES:
        raise DataValidationError(
            f"Unknown standardization profile {profile_name!r} "
            f"(known: {', '.join(PROFILES)})"
        )
    defaults = StandardizeOptions().as_dict()
    merged = {**defaults, **{k: v for k, v in options.items()}}
    opts = StandardizeOptions(**merged)
    mol = _parse_smiles(smiles)
    if mol is None:
        return StandardizeOutcome(
            original=smiles,
            canonical=None,
            profile=opts.profile,
            status="invalid",
            error=f"unparseable SMILES: {smiles!r}",
        )
    profile = PROFILES[opts.profile]
    steps: list[str] = []
    flags: list[str] = []
    try:
        result, steps, flags = profile.builder(mol, opts)
    except Exception as exc:  # noqa: BLE001 - the row's error, not the batch's
        _LOG.warning("standardization failed (%s) on %r: %s",
                     opts.profile, smiles, exc)
        return StandardizeOutcome(
            original=smiles,
            canonical=None,
            profile=opts.profile,
            status="invalid",
            steps=steps,
            error=str(exc),
        )
    if result is None:  # the REINVENT rejection window
        return StandardizeOutcome(
            original=smiles,
            canonical=None,
            profile=opts.profile,
            status="rejected",
            steps=steps,
            flags=flags,
            error="rejected by the REINVENT4 window (size/element/vocabulary)",
        )
    canonical = _canonical(result, isomeric=opts.keep_stereo)
    if not canonical:
        return StandardizeOutcome(
            original=smiles,
            canonical=None,
            profile=opts.profile,
            status="invalid",
            steps=steps,
            error="standardised molecule could not be canonicalised",
        )
    states: list[str] = []
    if opts.protonate:
        states = protonate_smiles(
            canonical,
            min_ph=opts.min_ph,
            max_ph=opts.max_ph,
            pka_std_range=opts.pka_std_range,
        )
        if len(states) > 1:
            flags.append("protonation-ambiguous")
    status = "unchanged" if canonical == smiles.strip() else "standardized"
    return StandardizeOutcome(
        original=smiles,
        canonical=canonical,
        profile=opts.profile,
        status=status,
        steps=steps,
        flags=flags,
        protonation_states=states,
    )


# ---------------------------------------------------------------------------
# the CSV batch driver
# ---------------------------------------------------------------------------

#: Column-name candidates for the SMILES-column heuristic (lowercase
#: substring match, in priority order).
_SMILES_COLUMN_HINTS: tuple[str, ...] = (
    "smiles",
    "smile",
    "canonical_smiles",
    "structure",
    "molfile",
    "mol",
    "smi",
)


def detect_smiles_column(frame: Any) -> str:
    """Find the SMILES column of a CSV frame -- the card's auto-pick.

    Two passes, exactly what a chemist does by eye: a name match
    (``SMILES`` / ``smiles`` / ``canonical_smiles`` / ``structure`` /
    ``smi`` -- the common corpus headers, substring +
    case-insensitive, the first hit wins in priority order), then a
    parse-rate check over the remaining object columns (up to 100
    non-null values per column, > 50 % must parse as SMILES; the
    highest rate wins).

    Raises
    ------
    :class:`~pharmaml.exceptions.DataValidationError` naming every
    column tried when nothing matches -- the caller's CSV is wrong,
    and the message says so.

    Example
    -------
    >>> import pandas as pd
    >>> detect_smiles_column(pd.DataFrame({"name": ["x"], "smiles": ["CCO"]}))
    'smiles'
    >>> detect_smiles_column(pd.DataFrame({"id": ["x"], "structure": ["CCO"]}))
    'structure'
    >>> detect_smiles_column(pd.DataFrame({"mol": ["CCO"], "junk": ["CC#"]}))
    'mol'
    """
    lowered = {str(column).lower(): column for column in frame.columns}
    for hint in _SMILES_COLUMN_HINTS:
        for lowered_name, original in lowered.items():
            if hint in lowered_name:
                return original
    tried: list[str] = []
    best_column: str | None = None
    best_rate = 0.0
    for column in frame.columns:
        if str(frame[column].dtype) != "object":
            continue
        values = frame[column].dropna().astype(str).head(100)
        values = [v for v in values if v.strip()]
        if not values:
            continue
        tried.append(str(column))
        rate = sum(1 for v in values if _parse_smiles(v) is not None) / len(values)
        if rate > 0.5 and rate > best_rate:
            best_rate = rate
            best_column = str(column)
    if best_column is None:
        raise DataValidationError(
            "No SMILES column detected in the CSV (looked for a name "
            f"match, then a parse-rate check over: {', '.join(tried) or 'no text columns'})"
        )
    return best_column


def standardize_csv(
    source: Any,
    smiles_column: str | None = None,
    id_column: str | None = None,
    **options: Any,
) -> StandardizationReport:
    """Standardise a CSV of SMILES -- the batch front door.

    Parameters
    ----------
    source:
        A path (``str`` / ``Path``) to a CSV file, or an already-loaded
        ``pandas.DataFrame``.  Paths are read with pandas' Python
        engine and separator sniffing (``sep=None`` -- comma, semicolon
        and tab corpora all load).
    smiles_column:
        The SMILES column; ``None`` -> :func:`detect_smiles_column`.
    id_column:
        An identifier column kept in every record (``None`` -> the row
        index as ``row-N``).
    profile, ...:
        The :func:`standardize_smiles` options (profile, toggles,
        protonation window).

    Returns
    -------
    A :class:`StandardizationReport` -- one record per row (no silent
    drops: unparseable rows are records with their error), the
    duplicate register (canonical collisions flagged against their
    first row), the summary counts, and the export helpers
    (:meth:`~StandardizationReport.to_frame` /
    :meth:`~StandardizationReport.write_csv`).

    Example
    -------
    >>> import pandas as pd
    >>> report = standardize_csv(pd.DataFrame(
    ...     {"SMILES": ["CCO", "CCO", "CC(=O)[O-].[Na+]"]}))
    >>> report.summary()["duplicate-rows"]
    1
    >>> [r.flags for r in report.records if "duplicate-of" in " ".join(r.flags)]
    [['duplicate-of-row-1']]
    """
    start = time.perf_counter()
    frame = _load_frame(source)
    profile_name = str(options.get("profile", "rdkit"))
    if profile_name not in PROFILES:
        raise DataValidationError(
            f"Unknown standardization profile {profile_name!r} "
            f"(known: {', '.join(PROFILES)})"
        )
    defaults = StandardizeOptions().as_dict()
    merged = {**defaults, **{k: v for k, v in options.items()}}
    opts = StandardizeOptions(**merged)
    column = smiles_column if smiles_column is not None else detect_smiles_column(frame)
    if column not in frame.columns:
        raise DataValidationError(
            f"SMILES column {column!r} not found in the CSV "
            f"(columns: {', '.join(map(str, frame.columns))})"
        )
    identifiers = None
    if id_column is not None:
        if id_column not in frame.columns:
            raise DataValidationError(
                f"Identifier column {id_column!r} not found in the CSV "
                f"(columns: {', '.join(map(str, frame.columns))})"
            )
        identifiers = frame[id_column].astype(str).tolist()

    records: list[StandardizationRecord] = []
    first_row_of: dict[str, int] = {}
    for i, raw in enumerate(frame[column].tolist()):
        identifier = (
            str(identifiers[i]) if identifiers is not None else f"row-{i + 1}"
        )
        smiles = "" if raw is None or str(raw) == "nan" else str(raw).strip()
        outcome = standardize_smiles(smiles, **options)
        flags = list(outcome.flags)
        if outcome.canonical is not None:
            first = first_row_of.get(outcome.canonical)
            if first is not None and first != i:
                flags.append(f"duplicate-of-row-{first + 1}")
            else:
                first_row_of[outcome.canonical] = i
        records.append(
            StandardizationRecord(
                row=i,
                identifier=identifier,
                original=smiles,
                canonical=outcome.canonical,
                status=outcome.status,
                steps=outcome.steps,
                flags=flags,
                protonation_states=outcome.protonation_states,
                error=outcome.error,
            )
        )
    elapsed = time.perf_counter() - start
    report = StandardizationReport(
        profile=opts.profile,
        smiles_column=column,
        id_column=id_column,
        records=records,
        elapsed=elapsed,
        options=opts.as_dict(),
    )
    summary = report.summary()
    _LOG.info(
        "CSV standardization: %d rows, %d standardized, %d invalid, "
        "%d rejected, %d duplicate(s), %.2fs (%s)",
        summary["rows"],
        summary["standardized"],
        summary["invalid"],
        summary["rejected"],
        summary["duplicate-rows"],
        elapsed,
        opts.profile,
    )
    return report


def _read_csv_any_separator(target: Any) -> Any:
    """Read a CSV by path or file-like with separator fallbacks.

    pandas' Python-engine sniffer (``sep=None``) needs a few rows of
    evidence before it commits; the card's pasted-box corpus can be a
    two-line demo, so the sniff failure falls back to the three
    corpus separators in order (comma, semicolon, tab)."""
    import pandas as pd

    last_error: Exception | None = None
    for separator in (None, ",", ";", "\t"):
        try:
            if separator is None:
                return pd.read_csv(target, sep=None, engine="python")
            return pd.read_csv(target, sep=separator)
        except Exception as exc:  # noqa: BLE001 - try the next separator
            last_error = exc
            if hasattr(target, "seek"):  # rewind the stream for retry
                target.seek(0)
    raise DataValidationError(f"Could not read the CSV: {last_error}")


def _load_frame(source: Any) -> Any:
    """Read the CSV (path, file-like or frame) -- pandas, separator
    sniffed with fallbacks (comma / semicolon / tab; the card's
    pasted-box StringIO rides the same path)."""
    if hasattr(source, "columns"):  # already a DataFrame
        return source

    if hasattr(source, "read"):  # a file-like (the pasted-box StringIO)
        return _read_csv_any_separator(source)
    path = Path(source)
    if not path.exists():
        raise DataValidationError(f"CSV file not found: {path}")
    return _read_csv_any_separator(path)


@dataclass
class StandardizationReport:
    """The batch answer -- records, summary, exports.

    Attributes
    ----------
    profile, smiles_column, id_column:
        Provenance of the run (the chain, the column auto-detected or
        given, the identifier column).
    records:
        One :class:`StandardizationRecord` per CSV row, input order.
    elapsed:
        Wall-clock seconds (the card's status line).
    options:
        The :class:`StandardizeOptions` dict as executed.
    """

    profile: str
    smiles_column: str
    id_column: str | None
    records: list[StandardizationRecord]
    elapsed: float
    options: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """The counts the card's status line shows."""
        statuses = [r.status for r in self.records]
        duplicate_rows = sum(
            1 for r in self.records if "duplicate-of" in " ".join(r.flags)
        )
        protonation_ambiguous = sum(
            1 for r in self.records if len(r.protonation_states) > 1
        )
        return {
            "rows": len(self.records),
            "standardized": statuses.count("standardized"),
            "unchanged": statuses.count("unchanged"),
            "invalid": statuses.count("invalid"),
            "rejected": statuses.count("rejected"),
            "duplicate-rows": duplicate_rows,
            "protonation-ambiguous": protonation_ambiguous,
            "elapsed": round(self.elapsed, 3),
            "profile": self.profile,
            "smiles-column": self.smiles_column,
        }

    def to_frame(self) -> Any:
        """The report as a ``pandas.DataFrame`` (the card's table and
        the CSV export's schema): identifier, original, canonical,
        status, steps, flags, protonation states, error."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "identifier": r.identifier,
                    "original": r.original,
                    "canonical": r.canonical if r.canonical is not None else "",
                    "status": r.status,
                    "steps": " -> ".join(r.steps),
                    "flags": "; ".join(r.flags),
                    "protonation_states": "; ".join(r.protonation_states),
                    "error": r.error if r.error is not None else "",
                }
                for r in self.records
            ]
        )

    def write_csv(self, path: Any) -> Path:
        """Write the report (the standardised corpus) as CSV."""
        target = Path(path)
        frame = self.to_frame()
        frame.to_csv(target, index=False)
        return target
