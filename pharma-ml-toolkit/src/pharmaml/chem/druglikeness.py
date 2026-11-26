"""Drug-likeness scoring: QED, an adapted QEPPI-style PPI score, and the bRo5 space map.

This module absorbs the *scoring* half of `AspirinCode/drug-likeness-space
<https://github.com/AspirinCode/drug-likeness-space>`_ -- a CRT/LigGPT-style
torch generator built to explore drug-likeness space (see Wang et al., "Explore
drug-likeness space with deep generative models", *Methods* 2023, the repo's own
citation).  Upstream, generated molecules were graded against three references --
QED, QEPPI and the bRO5 (beyond-Rule-of-5) compound list -- and exactly that
scoring surface lands here:

* **QED** -- quantitative estimation of drug-likeness, Bickerton, Paolini,
  Besnard, Muresan & Pollastri, "Quantifying the chemical beauty of drugs",
  *Nature Chemistry* 4:90-98 (2012), doi:10.1038/nchem.1243.  RDKit's own
  implementation is reported alongside every score below for comparison.
* **QEPPI** -- quantitative estimate of protein-protein-interaction-targeting
  drug-likeness, Kosugi & Ohue, *Int. J. Mol. Sci.* 22(20):10925 (2021),
  doi:10.3390/ijms222010925, with the conference version in Proc. IEEE CIBCB
  2021, doi:10.1109/CIBCB49929.2021.9562931.  See the honesty note: the score
  implemented here is QEPPI-*inspired*, not QEPPI itself.
* **bRO5** -- the envelope of oral drugs and clinical candidates beyond the
  Rule of 5, Doak, Over, Giordanetto & Kihlberg, "Oral druggables beyond the
  Rule of 5: insights from drugs and clinical candidates", *Chem. Biol.*
  23:105-120 (2016), doi:10.1016/j.chembiol.2015.11.001, plus the repo's list
  of 89 approved bRO5 drugs, embedded here as :data:`BRO5_CORPUS`.

Honesty note -- an adaptation, NOT the official QEPPI
-----------------------------------------------------
**QEPPI's exact published desirability breakpoints are NOT reproduced here.**
The official ``ohuelab/QEPPI`` parameter tables were not vendored; this module
implements a QEPPI-*inspired* extended drug-likeness score whose piecewise
windows follow the bRo5 review literature (Doak et al. 2016 / bRO5 envelope:
MW up to ~1000-1100, cLogP up to ~7.5, HBD up to ~6, HBA up to ~15, PSA up to
~250, rotatable bonds up to ~20).  It is an adaptation, not the official QEPPI,
and the function is deliberately named :func:`ppi_druglikeness` (NOT ``qeppi``)
to keep the name honest.  Read its score as "sits well inside the extended
bRo5 window", never as a published QEPPI value; for the real thing, vendor the
parameter tables from https://github.com/ohuelab/QEPPI.

What landed and what did not
----------------------------
The upstream repo's CRT/LigGPT torch model, its 1.7M-row ChEMBL training
corpus, and the DrugBank and iPPI-DB full datasets did **not** land: the heavy
torch stack was skipped (the toolkit's generative side already exists, see
:mod:`pharmaml.generative`), and only the 89-compound bRO5 reference list is
embedded.  :data:`BRO5_CORPUS` is a plain immutable tuple of RDKit-canonical
SMILES -- the package's no-data-files convention, the same pattern as
:data:`pharmaml.generative.corpus.DEFAULT_CORPUS`.  Stereochemistry is kept
(macrolides and cyclic peptides are not the same molecule without it); all 89
SMILES parse under RDKit and none is a duplicate, so no entry was dropped (the
doctests below re-verify that on every run -- nothing is canonicalized at
import time).

The two maps
------------
:data:`RO5_LIMITS` is the classic Lipinski/Veber line (MW 500, cLogP 5, HBD 5,
HBA 10, rotatable bonds 10); :data:`BRO5_LIMITS` is the extended bRo5 envelope
(MW 1000, cLogP 7.5, HBD 6, HBA 15, rotatable bonds 20, TPSA 250).
:func:`druglikeness_space_report` checks one molecule against both lines and
locates it among the corpus -- the README's "where does my molecule sit vs the
known bRO5 drugs" question.

Examples
--------
>>> from pharmaml.chem.druglikeness import BRO5_LIMITS, BRO5_CORPUS, DESCRIPTOR_KEYS, RO5_LIMITS
>>> len(BRO5_CORPUS), len(set(BRO5_CORPUS))       # 89 approved bRO5 drugs, no duplicates
(89, 89)
>>> DESCRIPTOR_KEYS                              # the descriptor contract
('MW', 'cLogP', 'HBD', 'HBA', 'TPSA', 'rotatable_bonds', 'aromatic_rings')
>>> sorted(RO5_LIMITS)                           # the classic line ...
['HBA', 'HBD', 'MW', 'cLogP', 'rotatable_bonds']
>>> sorted(BRO5_LIMITS)                          # ... and the extended envelope
['HBA', 'HBD', 'MW', 'TPSA', 'cLogP', 'rotatable_bonds']
>>> RO5_LIMITS["MW"], BRO5_LIMITS["MW"], BRO5_LIMITS["TPSA"]
(500.0, 1000.0, 250.0)

>>> from rdkit import Chem                       # every corpus entry parses ...
>>> mols = [Chem.MolFromSmiles(s) for s in BRO5_CORPUS]
>>> all(mol is not None for mol in mols)
True
>>> all(Chem.MolToSmiles(mol) == s for s, mol in zip(BRO5_CORPUS, mols))  # ... and is canonical
True

>>> from pharmaml.chem.druglikeness import ppi_druglikeness
>>> ppi_druglikeness("not_a_smiles") is None     # honest None on unparseable input
True
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

_LOG = get_logger(__name__)

__all__ = [
    "BRO5_CORPUS",
    "BRO5_LIMITS",
    "DESCRIPTOR_KEYS",
    "RO5_LIMITS",
    "DruglikenessSpaceReport",
    "PpiDruglikenessResult",
    "druglikeness_space_report",
    "ppi_druglikeness",
]

#: The classic Lipinski/Veber Rule-of-5 line (MW <= 500, cLogP <= 5, HBD <= 5,
#: HBA <= 10, rotatable bonds <= 10).  Deliberately a plain ``dict`` -- the GUI
#: reads it as the "classic drug-likeness" checklist.
RO5_LIMITS: dict[str, float] = {
    "MW": 500.0,
    "cLogP": 5.0,
    "HBD": 5.0,
    "HBA": 10.0,
    "rotatable_bonds": 10.0,
}

#: The extended bRo5 envelope of oral drugs / clinical candidates beyond the
#: Rule of 5 (Doak et al. 2016): MW <= 1000, cLogP <= 7.5, HBD <= 6, HBA <= 15,
#: rotatable bonds <= 20, TPSA <= 250.  Descriptors the rule is silent on
#: (``aromatic_rings``) simply have no entry here.
BRO5_LIMITS: dict[str, float] = {
    "MW": 1000.0,
    "cLogP": 7.5,
    "HBD": 6.0,
    "HBA": 15.0,
    "rotatable_bonds": 20.0,
    "TPSA": 250.0,
}

#: The seven descriptors every score and report here is computed from, in
#: canonical order (the six scored ones first, ``aromatic_rings`` last).
DESCRIPTOR_KEYS: tuple[str, ...] = (
    "MW",
    "cLogP",
    "HBD",
    "HBA",
    "TPSA",
    "rotatable_bonds",
    "aromatic_rings",
)

#: QEPPI-inspired piecewise desirability windows, keyed by descriptor:
#: ``(low, plateau_lo, plateau_hi, high)`` for :func:`_desirability` -- 0 at
#: ``low`` and ``high``, 1 on the ``[plateau_lo, plateau_hi]`` plateau.  The
#: HBD/HBA/TPSA/rotatable entries use a degenerate rise (``low == plateau_lo``
#: == 0) so small values keep full credit and only the right edge ramps down.
#: Windows follow the bRo5 review literature (Doak et al. 2016), NOT the
#: unpublished-here official QEPPI parameter tables -- see the module docstring.
_PPI_WINDOWS: dict[str, tuple[float, float, float, float]] = {
    "MW": (150.0, 250.0, 700.0, 1100.0),
    "cLogP": (-2.0, 0.0, 6.0, 8.0),
    "HBD": (0.0, 0.0, 5.0, 8.0),
    "HBA": (0.0, 0.0, 12.0, 16.0),
    "TPSA": (0.0, 0.0, 250.0, 300.0),
    "rotatable_bonds": (0.0, 0.0, 15.0, 25.0),
    "aromatic_rings": (0.0, 1.0, 4.0, 8.0),
}

#: The six descriptors whose desirabilities enter the :func:`ppi_druglikeness`
#: score (``aromatic_rings`` has a window and is reported, but is NOT scored).
_PPI_SCORED_KEYS: tuple[str, ...] = (
    "MW",
    "cLogP",
    "HBD",
    "HBA",
    "TPSA",
    "rotatable_bonds",
)

#: The 89 approved beyond-Rule-of-5 drugs of ``AspirinCode/drug-likeness-space``
#: (``data/bRO5_compounds.csv``), as RDKit-canonical SMILES (stereochemistry
#: kept; raw-string literals where a SMILES carries ``\`` stereo bonds).
#: Immutable tuple -- the no-data-files convention, see the module docstring.
BRO5_CORPUS: tuple[str, ...] = (
    (
        "C/C=C/C[C@@H](C)[C@@H](O)[C@H]1C(=O)N[C@@H](CC)C(=O)N(C)CC(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@"
        "H](C(C)C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N[C@@H](C)C(=O)N[C@H](C)C(=O)N(C)[C@@H](CC(C)C)C(=O)N"
        "(C)[C@@H](CC(C)C)C(=O)N(C)[C@@H](C(C)C)C(=O)N1C"
    ),
    (
        "COC(O)=N[C@H](C(=O)N1CCC[C@H]1c1nc2cc([C@H]3CC[C@H](c4cc5nc([C@@H]6CCCN6C(=O)[C@@H](N=C(O)"
        "OC)[C@@H](C)OC)[nH]c5cc4F)N3c3cc(F)c(N4CCC(c5ccc(F)cc5)CC4)c(F)c3)c(F)cc2[nH]1)[C@@H](C)OC"
    ),
    (
        "N=C(N)NCCC[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@@H]1CSSCCC(=O)N[C@@H](Cc2ccc(O)cc2)C(=O)N[C@@H]("
        "Cc2ccccc2)C(=O)N[C@@H](CCC(N)=O)C(=O)N[C@@H](CC(N)=O)C(=O)N1)C(=O)NCC(N)=O"
    ),
    (
        r"CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2"
        r"CC[C@@H](OCCO)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)"
        r"/C=C/C=C/C=C/1C"
    ),
    (
        r"CO[C@H]1C[C@@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N2CCCC[C@H]2C(=O)O[C@H]([C@H](C)C[C@@H]2"
        r"CC[C@@H](O)[C@H](OC)C2)CC(=O)[C@H](C)/C=C(\C)[C@@H](O)[C@@H](OC)C(=O)[C@H](C)C[C@H](C)/C="
        r"C/C=C/C=C/1C"
    ),
    (
        "COC(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)Nc1ccc([C@@H]2CC[C@@H](c3ccc(NC(=O)[C@@H]4CCCN4C(=O)[C@"
        "@H](NC(=O)OC)C(C)C)cc3)N2c2ccc(C(C)(C)C)cc2)cc1)C(C)C"
    ),
    (
        "COC(O)=N[C@H](C(=O)N1CC2(CC2)C[C@H]1c1ncc(-c2ccc3c(c2)C(F)(F)c2cc(-c4ccc5nc([C@@H]6[C@H]7C"
        "C[C@H](C7)N6C(=O)[C@@H](N=C(O)OC)C(C)C)[nH]c5c4)ccc2-3)[nH]1)C(C)C"
    ),
    (
        "COC[C@H]1C[C@@H](c2ncc(-c3ccc4c(c3)COc3cc5c(ccc6nc([C@@H]7CC[C@H](C)N7C(=O)[C@@H](N=C(O)OC"
        ")C(C)C)[nH]c65)cc3-4)[nH]2)N(C(=O)[C@H](N=C(O)OC)c2ccccc2)C1"
    ),
    (
        "COC(=O)N[C@H](C(=O)N1CCC[C@H]1c1nc(-c2ccc3c(c2)O[C@@H](c2ccccc2)n2c-3cc3cc(-c4c[nH]c([C@@H"
        "]5CCCN5C(=O)[C@@H](NC(=O)OC)C(C)C)n4)ccc32)c[nH]1)C(C)C"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H"
        "]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@](C)(F)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"
    ),
    (
        r"CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(c(/C=N/N5CCN(C6CCCC6)CC5)c(O)c4c3C2=O)NC(=O)/C("
        r"C)=C\C=C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C"
    ),
    (
        r"CO[C@H]1C[C@H](O[C@H]2[C@H](C)O[C@@H](O[C@@H]3/C(C)=C/C[C@@H]4C[C@@H](C[C@]5(CC[C@H](C)[C"
        r"@@H](C(C)C)O5)O4)OC(=O)[C@@H]4C=C(C)[C@@H](O)[C@H]5OC/C(=C\C=C\[C@@H]3C)[C@]54O)C[C@@H]2O"
        r"C)O[C@@H](C)[C@@H]1O"
    ),
    (
        "CC[C@@H]1[C@@H]2CN(C(=O)[C@H](C(C)(C)C)NC(=O)O[C@@H]3C[C@H]3CCCCC(F)(F)c3nc4ccc(OC)cc4nc3O"
        "2)[C@@H]1C(=O)N[C@]1(C(=O)NS(=O)(=O)C2(C)CC2)C[C@H]1C(F)F"
    ),
    (
        "CC1(C)CCC(CN2CCN(c3ccc(C(=O)NS(=O)(=O)c4ccc(NCC5CCOCC5)c([N+](=O)[O-])c4)c(Oc4cnc5[nH]ccc5"
        "c4)c3)CC2)=C(c2ccc(Cl)cc2)C1"
    ),
    (
        r"CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(c3C2=O)C2=NC3(CCN(CC(C)C)CC3)NC2=C(NC(=O)/C(C)=C\C="
        r"C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C4=O"
    ),
    (
        "CC(C)(C)[C@@H]1NC(=O)O[C@@H]2CCC[C@H]2OC/C=C/C(F)(F)c2nc3ccccc3nc2O[C@@H]2C[C@@H](C(=O)N[C"
        "@]3(C(=O)NS(=O)(=O)C4(C)CC4)C[C@H]3C(F)F)N(C2)C1=O"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H"
        "]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)[C@@H]2N[C@@H](COCCOC)O[C@H]([C@H]2C)["
        "C@]1(C)O"
    ),
    (
        "CCCC(=O)O[C@H]1[C@H](C)O[C@@H](O[C@H]2[C@H](N(C)C)[C@@H](O)[C@H](O[C@H]3[C@@H](CC=O)C[C@@H"
        "](C)[C@@H](O)/C=C/C=C/C[C@@H](C)OC(=O)C[C@@H](O)[C@@H]3OC)O[C@@H]2C)C[C@@]1(C)OC(=O)CC"
    ),
    (
        r"CO[C@H]1/C=C/O[C@@]2(C)Oc3c(C)c(O)c4c(O)c(c(/C=N/N5CCN(C)CC5)c(O)c4c3C2=O)NC(=O)/C(C)=C\C"
        r"=C\[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)C(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(OC"
        ")C[C@@H](C)C(=O)[C@H](C)[C@H]2N(CCCCn3cnc(-c4cccnc4)c3)C(=O)O[C@]12C"
    ),
    (
        r"C=CC[C@@H]1/C=C(\C)C[C@H](C)C[C@H](OC)[C@H]2O[C@@](O)(C(=O)C(=O)N3CCCC[C@H]3C(=O)O[C@H](/"
        r"C(C)=C/[C@@H]3CC[C@@H](O)[C@H](OC)C3)[C@H](C)[C@@H](O)CC1=O)[C@H](C)C[C@@H]2OC"
    ),
    (
        "C[C@H]1O[C@@H](O[C@H]2[C@@H](O)C[C@H](O[C@H]3[C@@H](O)C[C@H](O[C@H]4CC[C@@]5(C)[C@H](CC[C@"
        "@H]6[C@@H]5C[C@@H](O)[C@]5(C)[C@@H](C7=CC(=O)OC7)CC[C@]65O)C4)O[C@@H]3C)O[C@@H]2C)C[C@H](O"
        ")[C@@H]1O"
    ),
    (
        "CCC1=C[C@@H]2CN(C1)Cc1c([nH]c3ccccc13)[C@@](C(=O)OC)(c1cc3c(cc1OC)N(C)[C@H]1[C@@](O)(C(=O)"
        "OC)[C@H](OC(C)=O)[C@]4(CC)C=CCN5CC[C@]31[C@@H]54)C2"
    ),
    (
        "CC(C)c1nc(CN(C)C(=O)N[C@@H](CCN2CCOCC2)C(=O)N[C@H](CC[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)Cc2cc"
        "ccc2)cs1"
    ),
    (
        "C=C[C@@H]1C[C@]1(N=C(O)[C@@H]1C[C@@H]2CN1C(=O)[C@H](C(C)(C)C)N=C(O)O[C@@H]1C[C@H]1CCCCCc1n"
        "c3ccc(OC)cc3nc1O2)C(O)=NS(=O)(=O)C1CC1"
    ),
    (
        r"Cc1cnc(C(=O)N[C@H]2CCCCC/C=C\[C@@H]3C[C@@]3(C(=O)NS(=O)(=O)C3CC3)NC(=O)[C@@H]3C[C@@H](Oc4"
        r"nc5ccccc5c5ccccc45)CN3C2=O)cn1"
    ),
    (
        "CC[C@@H]1C[C@]1(N=C(O)[C@@H]1C[C@@H]2CN1C(=O)[C@H](C(C)(C)C)N=C(O)OCC(C)(C)CCCCc1cccc3c1CN"
        "(C3)C(=O)O2)C(O)=NS(=O)(=O)C1CC1"
    ),
    (
        "COc1ccc2c(O[C@H]3C[C@H]4C(=O)N(C)CCCC/C=C/[C@@H]5C[C@@]5(C(O)=NS(=O)(=O)C5CC5)N=C(O)[C@@H]"
        "4C3)cc(-c3nc(C(C)C)cs3)nc2c1C"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H"
        "]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)CN(C)[C@H](C)[C@@H](O)[C@]1(C)O"
    ),
    (
        "C=C[C@@H]1C[C@]1(NC(=O)[C@@H]1C[C@@H](Oc2ncc(OC)c3ccc(Cl)cc23)CN1C(=O)[C@@H](NC(=O)OC(C)(C"
        ")C)C(C)(C)C)C(=O)NS(=O)(=O)C1CC1"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H"
        "]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"
    ),
    (
        "COC(=O)N[C@H](C(=O)N1CCC[C@H]1c1ncc(-c2ccc(-c3ccc(-c4cnc([C@@H]5CCCN5C(=O)[C@@H](NC(=O)OC)"
        "C(C)C)[nH]4)cc3)cc2)[nH]1)C(C)C"
    ),
    (
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H"
        "]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O"
    ),
    (
        "CC(C)c1nc(CN(C)C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)[C@H](Cc2ccccc2)NC(=O)OCc2cncs2"
        ")C(C)C)cs1"
    ),
    "CCC(C)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@H]5CO[C@](Cn6cncn6)(c6ccc(Cl)cc6Cl)O5)cc4)CC3)cc2)c1=O",
    (
        "COC(=O)N[C@H](C(=O)N[C@@H](Cc1ccccc1)[C@@H](O)CN(Cc1ccc(-c2ccccn2)cc1)NC(=O)[C@@H](NC(=O)O"
        "C)C(C)(C)C)C(C)(C)C"
    ),
    (
        "CC[C@@H]([C@H](C)O)n1ncn(-c2ccc(N3CCN(c4ccc(OC[C@@H]5CO[C@@](Cn6cncn6)(c6ccc(F)cc6F)C5)cc4"
        ")CC3)cc2)c1=O"
    ),
    "O=C(NC1CCN(CCCCC2(C(=O)NCC(F)(F)F)c3ccccc3-c3ccccc32)CC1)c1ccccc1-c1ccc(C(F)(F)F)cc1",
    (
        "CCC[C@H](NC(=O)[C@@H]1[C@H]2CCC[C@H]2CN1C(=O)[C@@H](NC(=O)[C@@H](NC(=O)c1cnccn1)C1CCCCC1)C"
        "(C)(C)C)C(=O)C(=O)NC1CC1"
    ),
    (
        "CC(C)(C)N=C(O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@@H](O)[C@H](Cc1ccccc1)N=C(O)[C@H](CC(=N)O)N"
        "C(=O)c1ccc2ccccc2n1"
    ),
    "C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](OCCOCCOCCOCCOCCOCCOCCOC)CC[C@@]3(O)[C@H]1C5",
    "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
    (
        r"CO/N=C1\C[C@]2(C[C@@H]3C[C@@H](C/C=C(\C)C[C@@H](C)/C=C/C=C4\CO[C@@H]5[C@H](O)C(C)=C[C@@H]"
        r"(C(=O)O3)[C@]45O)O2)O[C@H](/C(C)=C/C(C)C)[C@H]1C"
    ),
    "CCCCCCOC(=O)N=C(N)c1ccc(NCc2nc3cc(C(=O)N(CCC(=O)OCC)c4ccccn4)ccc3n2C)cc1",
    "COC(=O)C1=C(C)NC(C)=C(C(=O)OC(C)(C)CN(C)CCC(c2ccccc2)c2ccccc2)C1c1cccc([N+](=O)[O-])c1",
    "COC(=O)C1=C(C)NC(C)=C(C(=O)OCCN2CCN(C(c3ccccc3)c3ccccc3)CC2)C1c1cccc([N+](=O)[O-])c1",
    "CCC[C@@]1(CCc2ccccc2)CC(O)=C([C@H](CC)c2cccc(NS(=O)(=O)c3ccc(C(F)(F)F)cn3)c2)C(=O)O1",
    "Cc1nn(C)cc1S(=O)(=O)NC(=O)c1ccc(-n2ccc(OCC(C)(C)C(F)(F)F)n2)nc1N1C[C@@H](C)CC1(C)C",
    r"CCCCCCO[C@@H](C)c1cccc(-c2csc(NC(=O)c3cc(Cl)c(/C=C(\C)C(=O)O)c(Cl)c3)n2)c1OC",
    (
        "COc1cc([C@@H]2c3cc4c(cc3[C@@H](O[C@@H]3O[C@@H]5CO[C@@H](C)O[C@H]5[C@H](O)[C@H]3O)[C@H]3COC"
        "(=O)[C@H]23)OCO4)cc(OC)c1O"
    ),
    "CC(C)(O)c1ccccc1CC[C@@H](SCC1(CC(=O)O)CC1)c1cccc(/C=C/c2ccc3ccc(Cl)cc3n2)c1",
    "COc1cc(N2CCC(N3CCN(C)CC3)CC2)ccc1Nc1ncc(Cl)c(Nc2ccccc2P(C)(C)=O)n1",
    "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "Cc1ccccc1-c1cc(N2CCN(C)CC2)ncc1N(C)C(=O)C(C)(C)c1cc(C(F)(F)F)cc(C(F)(F)F)c1",
    "COc1cc(C(=O)NS(=O)(=O)c2ccccc2C)ccc1Cc1cn(C)c2ccc(NC(=O)OC3CCCC3)cc12",
    "COc1cccc(N2CCN(C3=Nc4c(F)cccc4[C@H](CC(=O)O)N3c3cc(C(F)(F)F)ccc3OC)CC2)c1",
    "CO[C@@H]1[C@H](N(C)C(=O)c2ccccc2)C[C@H]2O[C@]1(C)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4",
    "COc1ccc(CN(C(=O)[C@@H](N)Cc2c(C)cc(C(N)=O)cc2C)[C@@H](C)c2nc(-c3ccccc3)c[nH]2)cc1C(=O)O",
    "Cc1c(O)cccc1C(=O)N[C@@H](CSc1ccccc1)[C@H](O)CN1C[C@H]2CCCC[C@H]2C[C@H]1C(=O)NC(C)(C)C",
    "CCC(=O)O[C@@H](O[P@](=O)(CCCCc1ccccc1)CC(=O)N1C[C@H](C2CCCCC2)C[C@H]1C(=O)O)C(C)C",
    (
        "COc1c(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)cc2c(c1OC)-c1ccc(SC)c(=O)cc1[C@@H](NC(C)="
        "O)CC2"
    ),
    "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "Cc1cc(Nc2ncc(Cl)c(=Nc3ccccc3S(=O)(=O)C(C)C)[nH]2)c(OC(C)C)cc1C1CCNCC1",
    "CCCCc1oc2ccc(NS(C)(=O)=O)cc2c1C(=O)c1ccc(OCCCN(CCCC)CCCC)cc1",
    (
        "CN(C)c1cc(CNCC(C)(C)C)c(O)c2c1C[C@H]1C[C@H]3[C@H](N(C)C)C(O)=C(C(N)=O)C(=O)[C@@]3(O)C(O)=C"
        "1C2=O"
    ),
    "COc1nc2ccc(Br)cc2cc1[C@@H](c1ccccc1)[C@@](O)(CCN(C)C)c1cccc2ccccc12",
    "CCc1nc(C(N)=O)c(Nc2ccc(N3CCC(N4CCN(C)CC4)CC3)c(OC)c2)nc1NC1CCOCC1",
    "COCCCOc1cc(C[C@@H](C[C@H](N)[C@@H](O)C[C@H](C(=O)NCC(C)(C)C(N)=O)C(C)C)C(C)C)ccc1OC",
    "COc1ccccc1Oc1c(NS(=O)(=O)c2ccc(C(C)(C)C)cc2)nc(-c2ncccn2)nc1OCCO",
    "CN1CCc2nc(C(=O)N[C@@H]3C[C@@H](C(=O)N(C)C)CC[C@@H]3NC(=O)C(=O)Nc3ccc(Cl)cn3)sc2C1",
    "COC(=O)N[C@@H](C)CNc1nccc(-c2cn(C(C)C)nc2-c2cc(Cl)cc(NS(C)(=O)=O)c2F)n1",
    "C[C@]1(COc2ccc(N3CCC(Oc4ccc(OC(F)(F)F)cc4)CC3)cc2)Cn2cc([N+](=O)[O-])nc2O1",
    "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
    "CC(C)OC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@H](n2ccc(=O)[nH]c2=O)[C@](C)(F)[C@@H]1O)Oc1ccccc1",
    "CCCCN(CCCC)CC(O)c1cc(Cl)cc2c1-c1ccc(Cl)cc1/C2=C/c1ccc(Cl)cc1",
    (
        "C[C@]12C=CC(=O)N[C@@H]1CC[C@@H]1[C@@H]2CC[C@]2(C)[C@@H](C(=O)Nc3cc(C(F)(F)F)ccc3C(F)(F)F)C"
        "C[C@@H]12"
    ),
    "Cc1cnc(Nc2ccc(OCCN3CCCC3)cc2)[nH]c1=Nc1cccc(S(=O)(=O)NC(C)(C)C)c1",
    "CC(C)(CO)c1cc2cc(NC(=O)C3(c4ccc5c(c4)OC(F)(F)O5)CC3)c(F)cc2n1C[C@@H](O)CO",
    "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1",
    "CC(C)OC(=O)OCOP(=O)(CO[C@H](C)Cn1cnc2c(N)ncnc21)OCOC(=O)OC(C)C",
    "CCCc1nc2c(C)cc(-c3nc4ccccc4n3C)cc2n1Cc1ccc(-c2ccccc2C(=O)O)cc1",
    r"CO/N=C(\C(=O)N[C@@H]1C(=O)N2C(C(=O)O)=C(/C=C\c3scnc3C)CS[C@H]12)c1csc(N)n1",
    "OCCN(CCO)c1nc(N2CCCCC2)c2nc(N(CCO)CCO)nc(N3CCCCC3)c2n1",
    "CC(C)(C)C(=O)OCOP(=O)(COCCn1cnc2c(N)ncnc21)OCOC(=O)C(C)(C)C",
    "CON(C)Cc1ccc(O)c2c1C[C@H]1C[C@H]3[C@H](N(C)C)C(O)=C(C(N)=O)C(=O)[C@@]3(O)C(O)=C1C2=O",
    "Nc1nc2c(c(=O)[nH]1)N(C=O)C(CNc1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1)CN2",
    "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1",
    "Nc1nc(=O)c2nc(CNc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)cnc2[nH]1",
    "CC(C)[C@H](N)C(=O)OCC(CO)OCn1cnc2c(=O)nc(N)[nH]c21",
)


def _require_rdkit() -> Any:
    """Import and return the RDKit ``Chem`` module or raise helpfully.

    Same lazy-import guard as :func:`pharmaml.chem.descriptors.require_rdkit`
    (kept local so this module stands alone): RDKit is an optional
    ``python -m pip install -e '.[chem]'`` extra, so importing it at module top
    would break rdkit-less installs of the rest of the package.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyError(
            "RDKit is required for drug-likeness scoring. "
            "From the toolkit package root, run:  python -m pip install -e '.[chem]'"
        ) from exc
    return Chem


def _parse_smiles(smiles: str) -> Any:
    """RDKit ``Mol`` for one SMILES, or ``None`` when it does not parse.

    Blank / whitespace-only strings count as invalid -- RDKit parses ``""``
    as a *valid* zero-atom molecule, the same trap
    :func:`pharmaml.chem.descriptors.compute_descriptors` guards against.
    """
    Chem = _require_rdkit()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _descriptors(mol: Any) -> dict[str, float]:
    """The seven :data:`DESCRIPTOR_KEYS` for one RDKit molecule.

    ``MW`` from ``Descriptors.MolWt``, ``cLogP`` from ``Crippen.MolLogP``,
    ``HBD``/``HBA``/``TPSA``/``rotatable_bonds``/``aromatic_rings`` from the
    ``rdMolDescriptors`` calculators named in the module contract.
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    # rdkit-stubs omit the dynamically-generated descriptor functors on
    # rdkit.Chem.Descriptors / rdkit.Chem.Crippen (MolWt, MolLogP among
    # them); the runtime API is stable and documented.
    return {
        "MW": float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined]
        "cLogP": float(Crippen.MolLogP(mol)),  # type: ignore[attr-defined]
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
    }


def _desirability(
    value: float,
    low: float,
    plateau_lo: float,
    plateau_hi: float,
    high: float,
) -> float:
    """Trapezoid desirability: 0 below ``low``, linear rise ``low`` ->
    ``plateau_lo``, 1 on the plateau, linear fall ``plateau_hi`` -> ``high``,
    0 above.  Used for MW/cLogP/TPSA/rotatable/aromatic (HBD and HBA use the
    same shape with a degenerate zero-width rise, i.e. a right-edge ramp only).

    Examples
    --------
    >>> from pharmaml.chem.druglikeness import _desirability
    >>> _desirability(180.0, 150.0, 250.0, 700.0, 1100.0)   # on the rise
    0.3
    >>> _desirability(450.0, 150.0, 250.0, 700.0, 1100.0)   # on the plateau
    1.0
    >>> _desirability(900.0, 150.0, 250.0, 700.0, 1100.0)   # on the fall
    0.5
    >>> _desirability(100.0, 150.0, 250.0, 700.0, 1100.0)   # below / above the window
    0.0
    >>> _desirability(1200.0, 150.0, 250.0, 700.0, 1100.0)
    0.0
    >>> _desirability(6.0, 0.0, 0.0, 5.0, 8.0)              # HBD right-edge ramp
    0.6666666666666666
    >>> _desirability(0.0, 0.0, 0.0, 5.0, 8.0)              # full credit at zero
    1.0
    """
    if value < low or value >= high:
        return 0.0
    if value < plateau_lo:
        return (value - low) / (plateau_lo - low)
    if value <= plateau_hi:
        return 1.0
    return (high - value) / (high - plateau_hi)


@dataclass
class PpiDruglikenessResult:
    """One molecule scored by :func:`ppi_druglikeness`.

    Attributes
    ----------
    smiles:
        The input SMILES, exactly as passed in.
    canonical:
        RDKit canonical SMILES of the parsed molecule.
    descriptors:
        The seven :data:`DESCRIPTOR_KEYS` values.
    desirabilities:
        Per-descriptor desirability ``d`` in ``[0, 1]`` for all seven keys
        (``aromatic_rings`` included -- informational only, see ``score``).
    score:
        Geometric mean of the SIX scored desirabilities (MW, cLogP, HBD,
        HBA, TPSA, rotatable_bonds); ``aromatic_rings`` is excluded.  A zero
        on any scored descriptor vetoes the whole score, mirroring QED's
        product semantics.
    qed:
        RDKit QED (Bickerton et al. 2012), for side-by-side comparison.
    """

    smiles: str
    canonical: str
    descriptors: dict[str, float]
    desirabilities: dict[str, float]
    score: float
    qed: float

    def as_frame(self) -> pd.DataFrame:
        """The seven descriptors with their values and desirabilities.

        One row per :data:`DESCRIPTOR_KEYS` entry (the six scored descriptors
        plus the reported-but-unscored ``aromatic_rings``), with columns
        ``descriptor``, ``value``, ``desirability``.
        """
        return pd.DataFrame(
            {
                "descriptor": list(DESCRIPTOR_KEYS),
                "value": [self.descriptors[key] for key in DESCRIPTOR_KEYS],
                "desirability": [
                    self.desirabilities[key] for key in DESCRIPTOR_KEYS
                ],
            }
        )

    def __str__(self) -> str:
        """The adapted PPI score and QED side by side."""
        return (
            f"ppi_druglikeness={self.score:.3f}  QED={self.qed:.3f}"
            f"  ({self.canonical})"
        )


def ppi_druglikeness(smiles: str) -> PpiDruglikenessResult | None:
    """Score one molecule with the adapted (QEPPI-inspired) PPI drug-likeness.

    This is the honest-name adaptation described in the module docstring --
    QEPPI's published breakpoints are NOT reproduced; the windows below follow
    the bRo5 review literature (Doak et al. 2016) instead.  Every window is a
    :func:`_desirability` trapezoid (0 / rise / 1 on the plateau / fall / 0):

    ================  =======  ============  =======
    descriptor        0 at     1 on          0 at
    ================  =======  ============  =======
    MW                150      [250, 700]    1100
    cLogP             -2       [0, 6]        8
    HBD               --       [0, 5]        8
    HBA               --       [0, 12]       16
    TPSA              --       [0, 250]      300
    rotatable_bonds   --       [0, 15]       25
    ================  =======  ============  =======

    (HBD/HBA/TPSA/rotatable_bonds are right-edge ramps only: full credit from
    zero up to the plateau top, then a linear fall -- MW keeps a rise from
    150 Da so fragments are not flattered, and the plateau is the bRo5-extended
    ~700 Da rather than QED's ~300-350 Da.)  ``score`` is the geometric mean of
    the six scored desirabilities, so a 0 anywhere vetoes, like QED's product;
    ``aromatic_rings`` gets a window (0 at 0, 1 on [1, 4], 0 at 8) but is
    reported, never scored.

    Returns ``None`` when the SMILES does not parse.

    Parameters
    ----------
    smiles:
        One SMILES string (blank or unparseable input returns ``None``).

    Returns
    -------
    PpiDruglikenessResult | None
        The scored record, or ``None`` for an unparseable SMILES.

    Examples
    --------
    >>> from pharmaml.chem.druglikeness import ppi_druglikeness
    >>> result = ppi_druglikeness("CC(=O)Oc1ccccc1C(=O)O")    # aspirin
    >>> 0.0 < result.score <= 1.0, 0.0 < result.qed <= 1.0    # both in (0, 1]
    (True, True)
    >>> result.score > 0.5                                    # comfortably drug-like
    True
    >>> round(result.score, 3), round(result.qed, 3)          # side by side
    (0.819, 0.55)
    >>> round(result.descriptors["MW"], 2), round(result.desirabilities["MW"], 2)
    (180.16, 0.3)
    >>> len(result.as_frame())                                # 6 scored + aromatic_rings
    7
    >>> str(result)
    'ppi_druglikeness=0.819  QED=0.550  (CC(=O)Oc1ccccc1C(=O)O)'

    A peptide-sized molecule (MW > 400) that the classic line rejects sits
    comfortably in the extended window -- the whole point of QEPPI:

    >>> peptide = "NC(Cc1ccccc1)C(=O)NC(C)C(=O)NC(C)C(=O)NC(CCC(=O)O)C(=O)O"
    >>> big = ppi_druglikeness(peptide)                       # Phe-Ala-Ala-Glu
    >>> round(big.score, 3), round(big.qed, 3)                # extended vs classic
    (0.833, 0.24)
    >>> ppi_druglikeness("not_a_smiles") is None
    True
    """
    from rdkit.Chem.QED import qed

    mol = _parse_smiles(smiles)
    if mol is None:
        return None
    canonical = _require_rdkit().MolToSmiles(mol)
    descriptors = _descriptors(mol)
    desirabilities = {
        key: _desirability(descriptors[key], *window)
        for key, window in _PPI_WINDOWS.items()
    }
    product = math.prod(desirabilities[key] for key in _PPI_SCORED_KEYS)
    return PpiDruglikenessResult(
        smiles=smiles,
        canonical=canonical,
        descriptors=descriptors,
        desirabilities=desirabilities,
        score=float(product ** (1.0 / len(_PPI_SCORED_KEYS))),
        qed=float(qed(mol)),
    )


@dataclass
class DruglikenessSpaceReport:
    """One molecule located in drug-likeness space by :func:`druglikeness_space_report`.

    Attributes
    ----------
    smiles:
        The input SMILES, exactly as passed in.
    canonical:
        RDKit canonical SMILES of the parsed molecule.
    per_descriptor:
        Index = :data:`DESCRIPTOR_KEYS`; columns ``value``, ``ro5_limit``,
        ``ro5_pass``, ``bro5_limit``, ``bro5_pass``, ``bro5_median``,
        ``bro5_p90``, ``percentile_in_bro5``.  Limits are NaN for
        descriptors a rule says nothing about (e.g. TPSA under the classic
        line, ``aromatic_rings`` under both) and those rows pass vacuously.
        ``percentile_in_bro5`` is the percent of corpus molecules at or
        below this molecule's value (0 = below every bRO5 drug, 100 = above
        them all).
    n_corpus:
        Number of corpus molecules the percentiles were computed against.
    ro5_violations:
        How many :data:`RO5_LIMITS` checks the molecule fails.
    bro5_violations:
        How many :data:`BRO5_LIMITS` checks the molecule fails.
    ppi_score:
        The :func:`ppi_druglikeness` score (adapted, QEPPI-inspired).
    qed:
        RDKit QED, for side-by-side comparison.
    """

    smiles: str
    canonical: str
    per_descriptor: pd.DataFrame
    n_corpus: int
    ro5_violations: int
    bro5_violations: int
    ppi_score: float
    qed: float

    def as_frame(self) -> pd.DataFrame:
        """The per-descriptor table (a copy; index = descriptor)."""
        return self.per_descriptor.copy()

    def __str__(self) -> str:
        return (
            f"Drug-likeness space: {self.canonical}\n"
            f"  RO5 violations:  {self.ro5_violations}/{len(RO5_LIMITS)} classic checks\n"
            f"  bRO5 violations: {self.bro5_violations}/{len(BRO5_LIMITS)} extended checks\n"
            f"  corpus: {self.n_corpus} bRO5 reference drugs\n"
            f"  ppi score {self.ppi_score:.3f} vs QED {self.qed:.3f}"
        )


def _corpus_matrix(corpus: Sequence[str]) -> tuple[dict[str, np.ndarray], int]:
    """Descriptor columns for a reference corpus of SMILES.

    Returns ``(matrix, n)`` where ``matrix`` maps every
    :data:`DESCRIPTOR_KEYS` name to the corpus' values as a float array.
    Entries that fail to parse are dropped with a warning -- the default
    :data:`BRO5_CORPUS` never triggers this (its doctest guarantees all 89
    entries parse), so this only fires for caller-supplied corpora.
    """
    Chem = _require_rdkit()
    rows: list[dict[str, float]] = []
    dropped: list[str] = []
    for entry in corpus:
        mol = Chem.MolFromSmiles(entry) if entry.strip() else None
        if mol is None:
            dropped.append(str(entry))
            continue
        rows.append(_descriptors(mol))
    if dropped:
        preview = ", ".join(repr(entry) for entry in dropped[:3])
        _LOG.warning(
            "Dropped %d unparseable corpus entr%s (%s%s) -- percentiles are "
            "computed over the %d parseable entries only",
            len(dropped),
            "y" if len(dropped) == 1 else "ies",
            preview,
            ", ..." if len(dropped) > 3 else "",
            len(rows),
        )
    if not rows:
        raise DataValidationError(
            "corpus contains no RDKit-parseable SMILES -- pass at least one "
            "valid reference molecule (or None for the bundled BRO5_CORPUS)"
        )
    matrix = {
        key: np.asarray([row[key] for row in rows], dtype=float)
        for key in DESCRIPTOR_KEYS
    }
    return matrix, len(rows)


def druglikeness_space_report(
    smiles: str,
    corpus: Sequence[str] | None = None,
) -> DruglikenessSpaceReport | None:
    """Locate one molecule in drug-likeness space.

    The README's "where does my molecule sit vs the known bRO5 drugs"
    question: every descriptor is checked against :data:`RO5_LIMITS` (the
    classic Lipinski/Veber line) and :data:`BRO5_LIMITS` (the extended bRo5
    envelope of Doak et al. 2016), and located among a reference corpus
    (default :data:`BRO5_CORPUS`, the 89 approved bRO5 drugs) by median,
    90th percentile and the molecule's own percentile rank.  The
    :func:`ppi_druglikeness` score and RDKit QED ride along for the verdict.

    Returns ``None`` when the SMILES does not parse.

    Parameters
    ----------
    smiles:
        One SMILES string (blank or unparseable input returns ``None``).
    corpus:
        Reference SMILES for the percentile columns; ``None`` (default) uses
        the bundled :data:`BRO5_CORPUS`.  Unparseable entries are dropped
        with a warning; a corpus with no parseable entry at all raises
        :class:`~pharmaml.exceptions.DataValidationError`.

    Returns
    -------
    DruglikenessSpaceReport | None
        The report, or ``None`` for an unparseable SMILES.

    Examples
    --------
    >>> from pharmaml.chem.druglikeness import druglikeness_space_report
    >>> report = druglikeness_space_report("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
    >>> report.n_corpus
    89
    >>> bool(report.per_descriptor["percentile_in_bro5"].between(0, 100).all())
    True
    >>> report.as_frame().shape                              # 7 descriptors x 8 columns
    (7, 8)
    >>> print(report)
    Drug-likeness space: CC(=O)Oc1ccccc1C(=O)O
      RO5 violations:  0/5 classic checks
      bRO5 violations: 0/6 extended checks
      corpus: 89 bRO5 reference drugs
      ppi score 0.819 vs QED 0.550

    A Phe-Ala-Ala-Glu tetrapeptide (MW > 400) trips the classic line on
    rotatable bonds and HBD, while staying clean inside the bRO5 envelope:

    >>> peptide = "NC(Cc1ccccc1)C(=O)NC(C)C(=O)NC(C)C(=O)NC(CCC(=O)O)C(=O)O"
    >>> report = druglikeness_space_report(peptide)
    >>> report.ro5_violations >= 1, report.bro5_violations == 0
    (True, True)
    >>> round(float(report.per_descriptor.loc["MW", "value"]), 1)     # past RO5 territory
    436.5
    >>> int(report.per_descriptor.loc["rotatable_bonds", "value"]) > 10
    True
    >>> float(report.per_descriptor.loc["rotatable_bonds", "percentile_in_bro5"]) > 50.0
    True
    >>> druglikeness_space_report("not_a_smiles") is None
    True
    """
    scored = ppi_druglikeness(smiles)
    if scored is None:
        return None
    matrix, n_corpus = _corpus_matrix(BRO5_CORPUS if corpus is None else corpus)

    ro5_violations = 0
    bro5_violations = 0
    values: list[float] = []
    ro5_limits: list[float] = []
    ro5_passes: list[bool] = []
    bro5_limits: list[float] = []
    bro5_passes: list[bool] = []
    medians: list[float] = []
    p90s: list[float] = []
    percentiles: list[float] = []
    for key in DESCRIPTOR_KEYS:
        value = scored.descriptors[key]
        corpus_values = matrix[key]
        ro5_limit = float(RO5_LIMITS.get(key, math.nan))
        bro5_limit = float(BRO5_LIMITS.get(key, math.nan))
        ro5_pass = key not in RO5_LIMITS or value <= RO5_LIMITS[key]
        bro5_pass = key not in BRO5_LIMITS or value <= BRO5_LIMITS[key]
        ro5_violations += 0 if ro5_pass else 1
        bro5_violations += 0 if bro5_pass else 1
        values.append(value)
        ro5_limits.append(ro5_limit)
        ro5_passes.append(ro5_pass)
        bro5_limits.append(bro5_limit)
        bro5_passes.append(bro5_pass)
        medians.append(float(np.median(corpus_values)))
        p90s.append(float(np.percentile(corpus_values, 90)))
        percentiles.append(float(np.mean(corpus_values <= value) * 100.0))

    per_descriptor = pd.DataFrame(
        {
            "value": values,
            "ro5_limit": ro5_limits,
            "ro5_pass": ro5_passes,
            "bro5_limit": bro5_limits,
            "bro5_pass": bro5_passes,
            "bro5_median": medians,
            "bro5_p90": p90s,
            "percentile_in_bro5": percentiles,
        },
        index=list(DESCRIPTOR_KEYS),
    )
    return DruglikenessSpaceReport(
        smiles=smiles,
        canonical=scored.canonical,
        per_descriptor=per_descriptor,
        n_corpus=n_corpus,
        ro5_violations=ro5_violations,
        bro5_violations=bro5_violations,
        ppi_score=scored.score,
        qed=scored.qed,
    )
