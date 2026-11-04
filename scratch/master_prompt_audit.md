# MASTER_PROMPT audit — traceability matrix & coverage gap report (P4 + P6)

Evidence-based cross-reference: every row below was verified against the live
tree on 2026-11-04/05 (module existence, playground.py card wiring, test
imports, web routes). DB column reflects the P3 finding: `db/custom.db` is an
empty scaffold (User/Post, 0 rows) — no feature stores data there, so the
project's data layer is file-based (CSV/SMILES + joblib persistence).

Legend — coverage: **C** complete · **P** partial · **M** missing · **—** n/a.
Playground = playground.py cards; Web = Next.js API routes / explorer cards.

## C. Feature Traceability Matrix (P4)

| # | Feature / lineage | Original source | DB | Code (`src/pharmaml`) | Playground | Web | Verification | Action |
|---|---|---|---|---|---|---|---|---|
| 1 | Uploads: rar ML pipeline (20 descriptors, converters, Tanimoto, splitters, mergers) | `کدهای ماشین لرنینگ.rar` | — | C (chem/, ml/splits) | C (Chemistry + Utilities cards) | P (chem/features, ml/train) | tests ✓ (converters/similarity/features) | none |
| 2 | Uploads: ADA script | `Applicability_domain_analysis.py` | — | C (ml/applicability.py) | C (ML tab AD card) | C (api/ml/ad) | tests ✓ (test_applicability) | none |
| 3 | Uploads: Ali's docking scripts (pdbqt gen, SMILES→PDB, vina batch, mismatch) | `کد علی.zip` | — | C (chem/docking) | C (Docking cards 12898-13252) | C (api/docking/*) | tests ✓ (docking/docking_prep) | none |
| 4 | Uploads: new-folder Vina workflows (ligand gen, batch, readers) | `new folder.zip` | — | C (chem/docking) | C (Docking cards) | C (api/docking) | tests ✓ | none |
| 5 | Uploads: fak_ml.ipynb / look.ipynb / xxx.ipynb | upload notebooks | — | C (ml pipeline, stacking) | C (ML tab cards incl. look's OOF stacking 16825) | P | tests ✓ (ml_pipeline, notebook_variants) | none |
| 6 | PK/PD dosing, BSA, Emax, trough individualisation | toolkit native + uploads | — | C (pharma/*) | C (PK tab 13 cards) | — | tests ✓ (pk, dosage) | none |
| 7 | Cell-line / omics panels | GDSC/CCLE-style uploads | — | C (biology/, ml/drugresponse) | C (Cell-line 6 + ResponseAI 7 cards) | C (api/biology/biomarkers) | tests ✓ (cellline, pd) | none |
| 8 | Generative (SELFIES mutants, crossover, Pareto) | toolkit + uploads | — | C (generative/) | C (Generative 6 cards) | P (api/gen/surgery) | tests ✓ (generative, selfies_codec) | none |
| 9 | Foundations / DSfB / best-of-ml-python / chapters | round-165+ absorptions | — | C (ml/*, viz/statplots) | C (Foundations 9 + Bestof 7 cards) | — | tests ✓ | none |
| 10 | tensorboard / umap / fiftyone / plotly (round 186) | 4 GitHub repos | — | C (runboard/umap/curation/interactive) | C (DL cards: Run board, UMAP map, Curation, Interactive) | P | tests ✓ (runboard, umap, curation, interactive) | none |
| 11 | captum/shap XAI (rounds 185+187) | 2 GitHub repos | — | C (xai 1929L, explain 2285L) | C (DL "Attribution & attention" card + ML SHAP rows) | C (api/ml/shap + shap_bridge) | tests ✓ (explain 112) | none |
| 12 | interpretml EBM (round 187) | GitHub repo | — | C (interpret.py EBMClassifier) | **M — no GUI card** | M | tests ✓ (interpret 97) | **add Glassbox EBM card** |
| 13 | DIG GNNExplainer synthetic (round 187) | GitHub repo | — | C (geometric.py GNNClassifier+gnn_explain, 1424L) | **M — no GUI card** | M | tests ✓ (geometric 36) | **add Graph model + Graph explain cards** |
| 14 | pytorch_geometric absorption (round 187) | GitHub repo | — | C but **orphaned** (gnn.py 780L: Data/Batch/GCN/SAGE/GIN/GAT/EdgeConv/TopK/SAG) | M | M | **M — zero tests, zero importers** | **tests + wire via the Graph cards** |
| 15 | gnnxai (round 187, DIG+PyG explainers) | GitHub repos | — | C but **orphaned** (gnnxai.py 480L) | M | M | **M — zero tests, zero importers** | **tests + Graph explain card** |
| 16 | rdkit featurize bridge (round 187) | GitHub repo | — | C but **orphaned** (chem/featurize.py 174L) | M | M | **M — zero tests** | **tests + Graph model card wiring** |
| 17 | TPE / successive-halving / hyperband HPO (round 188/189 — LOST) | optuna/ray lineages | — | **M — files never committed, swallowed by rollback #4** | M | M | M | **REBUILD ml/hpo.py + tests + card** |
| 18 | Magnitude + structured pruning (round 188/189 — LOST) | torch pruning lineage | — | **M — lost in rollback #4** | M | M | M | **REBUILD ml/pruning.py + tests + card** |
| 19 | NF4 / int8 / LLM.int8() quantization (round 188/189 — LOST) | bitsandbytes lineage | — | **M — lost in rollback #4** | M | M | M | **REBUILD ml/quantization.py + tests + card** |
| 20 | ZeRO sharded harness + EarlyStopping (round 188/189 — LOST) | deepspeed lineage | — | **M — lost in rollback #4** | M | M | M | **REBUILD ml/harness.py + tests + card** |
| 21 | torchlens / alibi / Quantus fold-ins (round 189 plan) | 3 GitHub repos | — | M (torchlens ≈ xai's activation trace partially; anchors/Quantus metrics absent) | M | M | M | compact fold-ins **deferred** (see gaps) |
| 22 | Database (Prisma/SQLite) | scaffold | — (empty) | C scaffold (src/lib/db.ts, prisma/schema.prisma) | — (playground never touches SQL) | — (no route queries it) | verified 0 rows | keep as scaffold; document |
| 23 | GitHub backup + audit checkpoints | MASTER_PROMPT §12 | — | C (.zscripts/gh_backup.sh, manifests) | — | — | verified (SHA parity + clone) | continuous pushes |

## D. Coverage Gap Report (P6) — prioritized

### Critical
- **G-1 (rows 17-20): four engine modules lost to the 4th rollback.**
  Evidence: `git log --all --diff-filter=A` has no record; files absent.
  Expected: hpo (TPE γ=min(⌈0.1n⌉,25) parzen/EI, successive halving, hyperband),
  pruning (magnitude global/local + structured channel, iterative retrain),
  quantization (NF4 16-level double-quant block 64, int8 absmax, LLM.int8()
  outlier split at |x|≥6.0), harness (ZeRO stages 0-3, EarlyStopping 5
  branches). Solution: rebuild per preserved specs
  (`scratch/round189_state_after_rollback.md` §3) + tests + DL cards.
  Complexity: high (4 engines × engine+tests+card).

### High
- **G-2 (rows 12-16): round-187 engines without any execution path.**
  gnn.py/gnnxai.py/featurize.py are orphaned (no importer, no tests, no card);
  interpret's EBM + geometric's GNN have tests but no card. This violates the
  MASTER_PROMPT's central requirement ("every integrated feature has an
  appropriate and accessible execution path"). Solution: Graph model /
  Graph explain / Glassbox EBM cards in the DL tab + test files for the
  orphan trio. Complexity: medium.

### Medium
- **G-3 (row 21): torchlens/alibi/Quantus fold-ins never landed** (planned
  round 189). torchlens' core (layer activation traces) already exists in
  xai.py; the genuinely missing pieces are alibi's anchor explanations and
  Quantus' metric suite. Solution: compact numpy implementations
  (`anchor_explanation` in xai.py; `xai_metrics` complexity/robustness/
  faithfulness family) + card rows in the attribution card. Complexity:
  medium. Deferred to after G-1/G-2 if session budget runs out.

### Low
- **G-4 (row 22): Prisma/db scaffold unused** — 0 rows, no route queries it.
  Action: document as scaffold (this report); no schema migration without
  authorization (§3.4).
- **G-5: upload binaries (vina.exe etc.) are Windows executables** the Linux
  toolkit cannot execute; kept as inventory assets only. Documented, no action.
- **G-6: repo-clones/pymol tarballs** — inventory-only (graphics3d lineage
  already absorbed). No action.

## Verification plan for the fixes (P8 preview)

- G-1/G-2: new engines get doctests + dedicated test files; the four cards +
  three new cards get test_playground coverage; full gates (ruff/mypy/pytest)
  before ship; v1.2.107 release ritual + web pins sync; E2E agent-browser on
  the web app; final GitHub push + SHA verification + test clone.
