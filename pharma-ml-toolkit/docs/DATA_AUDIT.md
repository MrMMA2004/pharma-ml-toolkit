# Data Audit — the shipped example-data database (1.2.46)

The toolkit's `examples/data/` inventory is a small database: **13 CSVs, a
14-folder docking campaign, and 2 PDB structures** — 43 files, ~720 KB.
Round **100x-103** audited every one of them for **gaps** (missing cells,
ragged rows, coverage holes) and **errors** (invalid chemistry, impossible
values, duplicate identities, drifted pins) with a 526-check independent
auditor (`scratch/audit_database_103.py`) plus 20 permanent regression
guards (`tests/test_database_audit.py`).

## Verdict: 526/526 checks pass, DATABASE CLEAN

## Real errors found and fixed

| # | File | Error | Fix |
|---|------|-------|-----|
| 1 | `walkthrough_compounds.csv` | All 180 rows carried the **same invalid SMILES** `CCO(C)C` — a trivalent-oxygen string (chemically impossible), with zero diversity in the metadata column that demos "a user file's shape". | SMILES column now resamples the toolkit's own **12-molecule seed universe** (aspirin … lidocaine), drawn after the activity draw so the descriptor/census bytes stay pinned. |
| 2 | `qsar_regression.csv` | The pIC50 endpoint ranged **-4.3 .. 14.7** — IC50 from ~21 *molar* to ~20 *femtommolar*, both chemically impossible (15 rows with IC50 > 1 M, 2 rows < 10 fM). | `make_regression_table` now clips to the **assay-plausible window [3, 11]** (IC50 1 mM .. 10 pM) — exactly what a real plate does at its concentration limits ("reported as > max / < min, encoded at the boundary"). 58 rows clipped at 3, 22 at 11. |
| 3 | `pk_profile.csv` | Byte drift from its generator (1-ulp repr noise after a venv reset — the standing trap's residue). | Rebuilt byte-identical by the idempotent builder; AUC pins (linear 17.9607 / log-trapezoid 17.2536) verified through the package. |

## Designed quirks (documented audit cases — NOT errors)

These are planted on purpose so the notebooks' audit/repair sections have
something real to demo; the tests pin their exact counts:

- `screening_library.csv` — 2 planted invalid SMILES (`not_a_smiles`, `C1CC`)
- `ligand_library.csv` — 1 planted invalid SMILES + 1 empty-SMILES row
- `cellline_screen.csv` — 1 duplicate assay replicate + 1 missing LN_IC50
- `docking_campaign/` — 12 complete runs + `08_benzocaine` (MISSING_LOG) +
  `10_nicotine` (INCOMPLETE) audit cases
- `walkthrough_compounds.csv` — 12-vocab SMILES resampled across 180 rows
- `library_with_descriptors.csv` — includes real small molecules below
  100 Da (pyrrole 67.09, phenol 94.11, glycine 75.07 …)

## Invariants pinned by `tests/test_database_audit.py`

- **Structural**: no ragged rows in any CSV; missing cells match the designed
  budget exactly; trailing-newline convention (all files end with one,
  `seed_molecules.csv` deliberately doesn't); no CRLF, no stray whitespace.
- **Chemical**: every SMILES RDKit-valid except the planted pairs; the
  walkthrough vocabulary == the seed universe; `library_with_descriptors`
  agrees with RDKit recomputation (MolWt/MolLogP/TPSA).
- **Semantic**: label censuses (420/360/180-row tables), compound ids
  unique, pIC50 in [3, 11], LN_IC50 in ln(µM) range with exactly the
  planted missing/duplicate, PK AUC pins, campaign tally 12+1+1,
  aspirin dlg summary pin, both PDB byte pins.
- **Coverage**: every screened cell line has expression (and no orphans),
  campaign folders match `ligands.csv`, descriptors table covers exactly
  the screening library's 77 valid SMILES.

The provenance chain stays: every CSV regenerates byte-identically from
`tools/build_example_data.py` (idempotent — any drift, including venv
repr noise, is caught on the next build), and the generator itself now
documents the assay-window rationale in its docstring.
