# MASTER_PROMPT audit — traceability matrix & coverage gap report (P4 + P6)

Evidence-based cross-reference: every row below was verified against the live
tree on 2026-11-04/05 (module existence, playground.py card wiring, test
imports, web routes). DB column reflects the P3 finding: `db/custom.db` is an
empty scaffold (User/Post, 0 rows) — no feature stores data there, so the
project's data layer is file-based (CSV/SMILES + joblib persistence).
Read-only re-verification on 2026-11-05: SQLite 3.40.1, `integrity_check=ok`,
both tables empty, one unique User email index, no views/triggers, and no
foreign-key constraints (including `Post.authorId`). This is a scaffold
limitation, not a migration authorization; the database was not modified.

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
| 12 | interpretml EBM (round 187) | GitHub repo | — | C (`ml/interpret.py` EBMClassifier) | C (DL **Glassbox EBM** card) | M | 99 tests + 7 doctests collected | resolved; web twin remains unimplemented and is not claimed |
| 13 | DIG GNNExplainer synthetic (round 187) | GitHub repo | — | C (`ml/geometric.py` GNNClassifier + `gnn_explain`) | C (DL **Graph model** + **Graph explain** cards) | M | 36 tests + 12 doctests collected | resolved; web twin remains unimplemented and is not claimed |
| 14 | pytorch_geometric absorption (round 187) | GitHub repo | — | C (`ml/gnn.py`: Data/Batch/GCN/SAGE/GIN/GAT/EdgeConv/TopK/SAG) | C (Graph model card) | M | adopted by `test_gnn.py` + playground | resolved |
| 15 | gnnxai (round 187, DIG+PyG explainers) | GitHub repos | — | C (`ml/gnnxai.py`) | C (Graph explain card) | M | adopted by `test_gnn.py` + playground | resolved |
| 16 | rdkit featurize bridge (round 187) | GitHub repo | — | C (`chem/featurize.py`) | C (Graph model card) | M | graph/featurize/gnnxai: 48 tests + 7 doctests collected | resolved |
| 17 | TPE / successive-halving / hyperband HPO | optuna/ray lineages | — | C (`ml/hpo.py`, rebuilt) | C (Hyperband/HPO controls) | M | 64 tests + 7 doctests collected | resolved |
| 18 | Magnitude + structured pruning | torch pruning lineage | — | C (`ml/pruning.py`, rebuilt) | C (DL pruning card) | M | 48 tests + 8 doctests collected | resolved |
| 19 | NF4 / int8 / LLM.int8() quantization | bitsandbytes lineage | — | C (`ml/quantization.py`, rebuilt) | C (DL quantization card) | M | 50 tests + 4 doctests collected | resolved |
| 20 | ZeRO sharded harness + EarlyStopping | deepspeed lineage | — | C (`ml/harness.py`, rebuilt) | C (DL training-harness card) | M | 56 tests + 5 doctests collected | resolved |
| 21 | torchlens / alibi / Quantus fold-ins (round 189 plan) | 3 GitHub repos | — | M (torchlens ≈ xai's activation trace partially; anchors/Quantus metrics absent) | M | M | M | compact fold-ins **deferred** (see gaps) |
| 22 | Database (Prisma/SQLite) | scaffold | — (empty) | C scaffold (src/lib/db.ts, prisma/schema.prisma) | — (playground never touches SQL) | — (no route queries it) | verified 0 rows | keep as scaffold; document |
| 23 | GitHub backup + audit checkpoints | MASTER_PROMPT §12 | — | C (.zscripts/gh_backup.sh, manifests) | — | — | verified (SHA parity + clone) | continuous pushes |

## D. Coverage Gap Report (P6) — prioritized

### Critical
- **G-1 (rows 17-20): RESOLVED — four engines rebuilt and accessible.**
  Original evidence: the modules were lost before commit during rollback #4.
  Current evidence: `ml/hpo.py`, `ml/pruning.py`, `ml/quantization.py`, and
  `ml/harness.py` exist, have DL execution cards, and collect 218 dedicated
  tests plus 24 doctests. The earlier rebuild specification remains in
  `scratch/round189_state_after_rollback.md` §3 for provenance.

### High
- **G-2 (rows 12-16): RESOLVED — round-187 engines have execution paths.**
  Current evidence: Graph model / Graph explain / Glassbox EBM cards are in
  `playground.py`; `test_gnn.py` adopts the former orphan trio. The combined
  graph/featurize/gnnxai lineage collects 48 tests + 7 doctests, geometric
  collects 36 + 12, and interpret collects 99 + 7. Web twins are not claimed.

### Re-verification fixes (2026-11-05, Arena branch)
- **G-7 (high): RESOLVED — headless CLI import regression.** On Python without
  `tkinter`, `ScrollFrame(ttk.Frame)` crashed before `main()`, making both
  `playground.py --probe` and `--version` unusable. The two existing CLI tests
  reproduced the failure (2 failed); an inert headless base class now permits
  import while GUI startup still returns the documented dependency error.
  Targeted result after the fix: 2 passed; both CLI commands executed.
- **G-8 (medium): RESOLVED — incomplete development dependency declaration.**
  Clean test collection failed because `tests/test_notebook_variants.py`
  imports undeclared `nbformat`. Added `nbformat>=5.10` to the dev extra and
  `nbformat==5.11.1` to the pinned QA block; all 8 requirements guards pass.
- **G-9 (low): RESOLVED — lint executable-bit drift.** `ruff check .` reported
  EXE001 for `tools/checkpoint.py`; restored its shebang-promised executable
  bit. The project lint gate now passes without weakening the rule.
- **G-10 (medium): RESOLVED — display-fixture portability.** Many GUI classes
  used `xvfb_display` without the existing `XVFB_SKIP` marker, producing 358
  setup errors on a host without Tk/Xvfb. The fixture now performs the same
  dependency check and skips explicitly. Focused evidence: all 15 Round189
  card tests skip cleanly instead of erroring; no display-capable test result
  is claimed from this environment.
- **G-13 (medium): RESOLVED — tracked environment file contradicted policy.**
  `.env` was tracked despite the manifest's exclusion claim. Its only setting
  is a local SQLite file URL (no remote credential), but environment files
  should follow one policy. Preserved the local file, removed it from the Git
  index, added `!.env.example`, and committed a safe template with the same
  local-only configuration. Recovery clones now exclude `.env` by construction.
- **Environment-only exclusions:** 21 cases need RDKit's drawing extension,
  whose native `libXrender.so.1` dependency is absent. These are deselected by
  exact node ID in the final applicable-suite run, not silently skipped in
  source. The computational decode path with `with_drawing=False`, the bridge
  validation path, all non-display tests, lint, package typing, CLI helpers,
  and package build remain executable and are verified separately.

### Medium
- **G-3 (row 21): torchlens/alibi/Quantus fold-ins never landed** (planned
  round 189). torchlens' core (layer activation traces) already exists in
  xai.py; the genuinely missing pieces are alibi's anchor explanations and
  Quantus' metric suite. Solution: compact numpy implementations
  (`anchor_explanation` in xai.py; `xai_metrics` complexity/robustness/
  faithfulness family) + card rows in the attribution card. Complexity:
  medium. Deferred to after G-1/G-2 if session budget runs out.

### Medium — newly verified, deferred for targeted follow-up
- **G-11: web lint is not clean.** `npm run lint` reports 12 errors + 1 warning
  in existing React hooks/refs code (`code-explorer`, `consensus-rescore`,
  `ml-playground`, `session-report`, `carousel`, `use-mobile`). Expected:
  rule-compliant effects and no render-time ref reads. Proposed solution:
  small per-component corrections with focused web regression tests; do not
  disable the rules. Complexity: medium. No unrelated broad refactor was made
  during the headless-playground fix.
- **G-12: dependency vulnerabilities need an upgrade review.** A clean
  temporary-lockfile `npm audit` reports 12 moderate + 5 high findings.
  Direct high findings include `prisma` and `sharp`; some available fixes are
  semver-major and `@mdxeditor/editor` is implicated transitively. Expected:
  review advisories, upgrade compatible packages deliberately, rerun web/Python
  bridge tests and production builds. Complexity: medium; status: OPEN, since
  `npm audit fix --force` would violate the conservative-change requirement.

### Low
- **G-4 (row 22): Prisma/db scaffold unused** — 0 rows, no route queries it.
  `Post.authorId` has no foreign key; retain as a documented scaffold unless a
  separately authorized schema change is commissioned.
  Action: document as scaffold (this report); no schema migration without
  authorization (§3.4).
- **G-5: upload binaries (vina.exe etc.) are Windows executables** the Linux
  toolkit cannot execute; kept as inventory assets only. Documented, no action.
- **G-6: repo-clones/pymol tarballs** — inventory-only (graphics3d lineage
  already absorbed). No action.

## Verification evidence and remaining limits (P8)

- G-1/G-2 engine and card coverage is present as itemized in the matrix.
- Current applicable suite: **3,840 passed, 634 skipped, 22 deselected, 0
  failures/errors**. The deselections are limited to native-drawing failures
  and their mixed module-doctest container; the independent correlation
  doctest and all 91 other interpret tests pass separately.
- `ruff check .`, `mypy src`, focused CLI/dependency/bridge checks, editable
  installation, and wheel/sdist builds pass. Webpack compilation and the
  source-only TypeScript check pass with offline font mocks / a scoped config.
- Unverified here: 634 optional/display skips and 21 native-drawing behaviors.
  The normal Turbopack build cannot reach Google Fonts; web lint and dependency
  audit findings are G-11/G-12. A display-capable host with the native RDKit
  prerequisites must run those cases before claiming full GUI verification.
- Backup requires a new active-branch commit, push, SHA parity check, and test
  clone; the historical `main` evidence below is not a substitute.

## 2026-11-05 addendum — G-3 resolved by the registry commission

The attached `GitHub_Links_Registry.pdf` was checked URL by URL. Its three
explicitly deferred entries are now compact, tested surfaces in `ml/xai.py`:
`model_activation_trace` (torchlens), `tabular_anchor` (alibi), and the
complexity/faithfulness/max-sensitivity trio (Quantus), with **Model graph**,
**Anchor rule**, and **Quantus metrics** actions in the existing attribution
card. The numerical APIs, display-independent handlers, native-Tk test
coverage, upstream source commits, and deliberate scope limits are recorded in
`GITHUB_LINKS_REGISTRY.md`. G-11/G-12 and the native display/RDKit limits are
unchanged; this addendum does not convert those environment exclusions into
passes.
