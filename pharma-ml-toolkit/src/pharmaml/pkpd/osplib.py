"""The OSP PBPK model library registry (40 released models, v12.3).

``Open-Systems-Pharmacology/OSP-PBPK-Model-Library`` is the release
artifact of the OSP qualification ecosystem: one directory per
substance carrying the released ``.pksim5`` PK-Sim project, the
rendered evaluation report (markdown + PDF) and its figures.  The
registry below transcribes the catalogue itself — names, release
versions, model kinds, pediatric variants — plus the GMFE
(goodness-of-fit) numbers documented in the released evaluation
reports for the models this toolkit absorbed in depth (digoxin,
itraconazole, fluconazole).

The ``.pksim5`` project files are SQLite databases for PK-Sim 12.3
and the reports are rendered by the Qualification Framework 3.7 —
neither is vendored here; this module is the *index* that a
lightweight workflow consults to know what exists, at which version,
and how accurate the released model is reported to be.

Examples
--------
>>> from pharmaml.pkpd.osplib import (
...     find_model, library_frame, library_summary, OSP_SUITE_VERSION)

The library holds 40 released models:

>>> len(library_frame())
40
>>> library_summary()["small molecule"]
32
>>> library_summary()["biologic"]
7
>>> library_summary()["endogenous tracer"]
1

Digoxin: version 2.0, small molecule, with a pediatric variant and a
documented overall GMFE of 1.46:

>>> digoxin = find_model("digoxin")
>>> (digoxin.version, digoxin.category, digoxin.pediatric)
('2.0', 'small molecule', True)
>>> from pharmaml.pkpd.osplib import DOCUMENTED_GMFE
>>> DOCUMENTED_GMFE["Digoxin"]["All"]
1.46

Verapamil is the odd one out at v2.2, Alfentanil the oldest at v3.0:

>>> find_model("Verapamil").version, find_model("Alfentanil").version
('2.2', '3.0')

Unknown substances raise rather than guess:

>>> find_model("aspirin")
Traceback (most recent call last):
...
pharmaml.exceptions.DataValidationError: no OSP library model named 'aspirin' (40 models indexed)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from pharmaml.exceptions import DataValidationError

__all__ = [
    "DOCUMENTED_GMFE",
    "OSP_PBPK_LIBRARY",
    "OSP_SUITE_VERSION",
    "QUALIFICATION_FRAMEWORK_VERSION",
    "PBPKModelEntry",
    "find_model",
    "library_frame",
    "library_summary",
]

OSP_SUITE_VERSION = "12.3"
QUALIFICATION_FRAMEWORK_VERSION = "3.7"


@dataclass(frozen=True, slots=True)
class PBPKModelEntry:
    """One released library model (one directory in the repo)."""

    name: str
    version: str
    category: str          # 'small molecule' | 'biologic' | 'endogenous tracer'
    pediatric: bool        # a <Drug>_Pediatrics.pksim5 variant exists
    notes: str = ""


def _entries() -> list[PBPKModelEntry]:
    small = [
        ("Alfentanil", "3.0", True, "opioid analgesic; CYP3A4 substrate"),
        ("Alprazolam", "2.0", False, "CYP3A4 substrate (DDI victim)"),
        ("Amikacin", "2.0", True, "aminoglycoside; renal elimination"),
        ("Atazanavir", "2.0", False, "CYP3A4 inhibitor + victim"),
        ("Carbamazepine", "2.0", False, "CYP3A4 inducer; autoinduction"),
        ("Cimetidine", "2.0", False, "histamine H2 antagonist; OCT/MATE"),
        ("Clarithromycin", "2.0", False, "CYP3A4 mechanism-based inactivator; P-gp"),
        ("Dapagliflozin", "2.0", False, "SGLT2 inhibitor"),
        ("Digoxin", "2.0", True, "P-gp probe substrate; ATP1A2 binding; GFR"),
        ("Efavirenz", "2.0", False, "CYP2B6 induction network model"),
        ("Erythromycin", "2.0", False, "CYP3A4 inactivator; P-gp inhibitor"),
        ("Ethinylestradiol", "2.0", False, "steroid; DDI network victim"),
        ("Felodipine", "2.0", False, "CYP3A4 substrate"),
        ("Fluconazole", "2.0", False, "azole; CYP2C9/3A4 inhibitor"),
        ("Fluvoxamine", "2.0", False, "SSRI; CYP1A2/2C19 inhibitor"),
        ("Itraconazole", "2.0", False, "CYP3A4/P-gp inhibitor; 4-compound chain"),
        ("Mefenamic_acid", "2.0", False, "NSAID"),
        ("Mexiletine", "2.0", False, "antiarrhythmic; CYP2D6/1A2"),
        ("Midazolam", "2.0", False, "CYP3A4 probe substrate"),
        ("Moclobemide", "2.0", False, "MAO inhibitor"),
        ("Montelukast", "2.0", True, "leukotriene antagonist; CYP2C8/3A4"),
        ("Omeprazole", "2.0", False, "proton-pump inhibitor; CYP2C19"),
        ("Propofol", "1.0", True, "anesthetic; UGT1A9/1A8"),
        ("Raltegravir", "2.0", True, "HIV integrase inhibitor; UGT1A1"),
        ("Rifampicin", "2.0", False, "pregnane-X-receptor inducer; OATP"),
        ("S-Mephenytoin", "2.0", False, "CYP2C19 probe"),
        ("Sildenafil", "2.0", False, "PDE5 inhibitor; CYP3A4"),
        ("Sufentanil", "2.0", True, "opioid; CYP3A4"),
        ("Tizanidine", "2.0", False, "CYP1A2 substrate"),
        ("Triazolam", "2.0", False, "CYP3A4 substrate"),
        ("Vancomycin", "2.0", True, "glycopeptide antibiotic; renal"),
        ("Verapamil", "2.2", False, "CYP3A4/P-gp inhibitor"),
    ]
    biologics = [
        ("7E3", "1.0", False, "anti-glycoprotein IIb/IIIa mAb (abciximab)"),
        ("BAY794620", "1.0", False, "anti-TWEAKR mAb"),
        ("CDA1", "1.0", False, "anti-CD3 mAb"),
        ("dAb2", "1.0", False, "domain antibody; FcRn-mediated"),
        ("MEDI524", "1.0", False, "anti-RSV mAb (motavizumab lineage)"),
        ("MEDI524YTE", "1.0", False, "MEDI524 with FcRn affinity engineering"),
        ("Tefibazumab", "1.0", False, "anti-CD11 mAb"),
    ]
    tracers = [
        ("Inulin", "1.0", False, "endogenous tracer; glomerular filtration"),
    ]
    entries: list[PBPKModelEntry] = []
    for name, version, ped, notes in small:
        entries.append(PBPKModelEntry(name, version, "small molecule", ped, notes))
    for name, version, ped, notes in biologics:
        entries.append(PBPKModelEntry(name, version, "biologic", ped, notes))
    for name, version, ped, notes in tracers:
        entries.append(PBPKModelEntry(name, version, "endogenous tracer", ped, notes))
    return entries


OSP_PBPK_LIBRARY: tuple[PBPKModelEntry, ...] = tuple(_entries())

# GMFE values documented in the released evaluation reports (the
# "GMFE for Goodness of fit plot for concentration in plasma" tables).
DOCUMENTED_GMFE: dict[str, dict[str, float]] = {
    "Digoxin": {
        "colonic admin.": 2.13,
        "iv": 1.45,
        "po SD": 1.30,
        "po MD": 1.64,
        "All": 1.46,
    },
    "Fluconazole": {
        "iv (building)": 1.09,
        "iv (verification)": 1.14,
        "oral (building)": 1.20,
        "oral (verification)": 1.26,
        "All": 1.18,
    },
    "Itraconazole": {
        "iv": 1.29,
        "capsule fasted": 1.74,
        "capsule fed": 1.57,
        "solution fasted": 1.54,
        "solution fed": 1.46,
        "All": 1.54,
    },
    "Hydroxy-Itraconazole": {
        "iv": 1.21,
        "capsule fasted": 1.68,
        "capsule fed": 1.95,
        "solution fasted": 1.38,
        "solution fed": 1.48,
        "All": 1.50,
    },
    "Keto-Itraconazole": {"All": 1.66},
    "N-desalkyl-Itraconazole": {"All": 1.53},
}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_-]+", "", name).lower()


def find_model(name: str) -> PBPKModelEntry:
    """Look a model up by (case/underscore-insensitive) name."""
    key = _normalize(name)
    for entry in OSP_PBPK_LIBRARY:
        if _normalize(entry.name) == key:
            return entry
    raise DataValidationError(
        f"no OSP library model named {name!r} "
        f"({len(OSP_PBPK_LIBRARY)} models indexed)"
    )


def library_frame() -> pd.DataFrame:
    """The catalogue as a DataFrame (name, version, category, pediatric)."""
    return pd.DataFrame(
        [
            {
                "name": entry.name,
                "version": entry.version,
                "category": entry.category,
                "pediatric": entry.pediatric,
                "notes": entry.notes,
            }
            for entry in OSP_PBPK_LIBRARY
        ]
    )


def library_summary() -> dict[str, int]:
    """Model counts per category."""
    counts: dict[str, int] = {}
    for entry in OSP_PBPK_LIBRARY:
        counts[entry.category] = counts.get(entry.category, 0) + 1
    return counts
