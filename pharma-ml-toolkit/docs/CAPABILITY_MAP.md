# Capability map — the nine-card-first audit (Phase 0)

**Baseline:** v1.2.183 (`e81985d`) · **Spec:** MASTER_PROMPT part6 §3
**Method:** every row below was derived live from the working tree —
page imports grepped, module and CLI censuses counted, tests collected —
never copied from prose.  This is the Phase-0 deliverable of the
nine-card-first enhancement architecture: the auditable
capability→card mapping and the migration plan for Phases 1-6.

## 1. What the launch architecture actually is (verified)

```
studio.py → ui.run() → MainWindow
  ├─ Navigation sidebar (18 SECTIONS, collapsible, dominant default)
  └─ QStackedWidget(18 pages)
       └─ DashboardPage → build_nine_card_grid()  ← the §5 grid is a
                                                    widget INSIDE the
                                                    dashboard page
```

The nine cards exist, are in the directive's exact order, carry live
dependency probes and real evidence previews, and every entry button
routes through `MainWindow.go_to(section)` — but they live *inside* the
Dashboard page of an 18-page sidebar application.  Part6 §2.1 requires
the inverse: the nine-card home as the primary screen with the sidebar
demoted to an optional advanced mode.  That inversion is Phase 1.

## 2. The eighteen pages and their verified backends

| Page | pharmaml modules imported (live grep) | card assignment (Phase 2 target) |
| --- | --- | --- |
| `dashboard` | — | 01 Introduction (current host of the §5 grid) |
| `data_explorer` | ml.research_audit, utils.io | 02 Datasets |
| `chemistry` | chem, chem.csv_standardize, chem.features, utils.io | 02 Datasets (chemical preparation) |
| `pkpd` | pharma.pk, pkpd.compartments, pkpd.events, pkpd.nca_ext, pkpd.ode, studio | 07 PK/PD |
| `biology` | biology | 07 PK/PD (biology sub-workspace) |
| `docking` | docking, exceptions, studio | 06 Molecular Docking |
| `model_analysis` | ml, ml.cv_results, ml.models, ml.training, utils.io | 03 Machine Learning |
| `ml_workbench` | ml, ml.evaluation, ml.models, ml.persistence, ml.tuner, studio, utils.checkpoints, utils.io | 03 Machine Learning |
| `unsupervised` | ml.manifold, ml.unsupervised, utils.io | 03 Machine Learning (exploration) |
| `deep_learning` | exceptions, ml, ml.curation, ml.neural, ml.numpy_neural, ml.runboard, ml.splits, ml.umap, ml.xai, studio, utils.io | 04 Deep Learning |
| `generative` | chem.oracles, chem.sa_scoring, generative, studio | 05 Generative AI |
| `foundations` | ml.inferential, ml.stats, profiling | 03 Machine Learning (walkthroughs) |
| `bestofml` | ml.adversarial, ml.annsearch, ml.apriori, ml.catalog, ml.privacy, ml.recommend, ml.sequences, ml.timeseries | 03 Machine Learning (sweeps) |
| `responseai` | ml.drugresponse | 07 PK/PD (response AI) |
| `image_lab` | biology, chem.descriptors, chem.similarity, studio, viz.export | 09 Visualization |
| `molecular_dynamics` | md, studio, viz, viz.export | 08 Molecular Dynamics Analysis |
| `utilities` | chem, chem.converters, docking, pharma, utils.io, viz | 02 Datasets + cross-card utilities |
| `settings_page` | — | 01 Introduction (global menu) |

Tests: 555 UI test functions across
39 UI suites; page smoke
coverage via `tests/ui/test_pages_smoke.py`.

## 3. Backend census

- `src/pharmaml/ml/`: **92 modules**,
74,074 lines.
- CLI: **62** `add_parser` sites (16 command groups).
- Library tests: 4514 test functions across
137 suites.
- `src/pharmaml/studio/` (runs + evidence): the §11 real-evidence
  backend already exists — Phase 4 extends, not replaces, it.

## 4. Backend-only capabilities (the honest C-list)

No Studio page, dialog, web card, or API route references these
modules (verified by import grep + the `test_feature_inventory.py`
word-match guard).  They have real implementations and real tests.
Phase 3 gives each its GUI workflow under the owning card:

| Module | What it does | Owning card (Phase 3 target) |
| --- | --- | --- |
| `ml/calibration.py` | probability calibration reports/recalibration | 03 Machine Learning (Phase 3) |
| `ml/comparison.py` | paired model-comparison tests + corrections | 03 Machine Learning (Phase 3) |
| `ml/conformal.py` | split-conformal regression intervals | 03 Machine Learning (Phase 3) |
| `ml/conformal_classification.py` | classification/label-conditional prediction sets | 03 Machine Learning (Phase 3) |
| `ml/coverage_diagnostics.py` | empirical-vs-nominal coverage shortfall table | 03 Machine Learning (Phase 3) |
| `ml/decision.py` | decision-curve analysis | 03 Machine Learning (Phase 3) |
| `ml/drift.py` | distribution-shift detection (PSI/quantile sketches) | 03 Machine Learning (Phase 3) |
| `ml/geometric.py` | geometric deep-learning primitives | 04 Deep Learning (Phase 3) |
| `ml/gnn.py` | GNN featuriser/trainer + XAI registry | 04 Deep Learning (Phase 3) |
| `ml/gnnxai.py` | GNN explainability | 04 Deep Learning (Phase 3) |
| `ml/harness.py` | offline sharded-training harness (ZeRO + early stopping) | 04 Deep Learning (Phase 3) |
| `ml/model_card.py` | model cards / provenance sheets | 03 Machine Learning (Phase 3) |
| `ml/multiplicity.py` | multiple-comparison corrections | 03 Machine Learning (Phase 3) |
| `ml/power.py` | fold-count power analysis | 03 Machine Learning (Phase 3) |
| `ml/pruning.py` | magnitude/structured network pruning | 04 Deep Learning (Phase 3) |
| `ml/quantization.py` | int8/fp16-style quantisation arithmetic | 04 Deep Learning (Phase 3) |
| `ml/sar.py` | MMP / activity cliffs / R-groups | 03 Machine Learning (Phase 3) |
| `ml/split_audit.py` | identifier-overlap audit across splits | 02 Datasets (Phase 3) |
| `ml/subgroups.py` | subgroup performance audits | 03 Machine Learning (Phase 3) |

The part6 §14.1 requirement 'the eight previously documented
Backend-only capability groups' maps to the first ten rows (gnn+gnnxai
count as one group, conformal+conformal_classification as one); the
nine r13-landed statistical modules extend the same list and are
handled by the same Phase-3 pass.

## 5. What counts as integrated (§3.2, restated as the gate)

A capability is GUI-integrated only when its workflow: collects and
validates inputs → invokes the real backend → reports progress and
errors → displays real results → supports inspection and export →
preserves metadata/provenance.  Import presence, a button, or a
synthetic demo is NOT integration.  Phase 6's coverage tests enforce
this against the registry, not against prose.

## 6. Migration plan (Phases 1-6)

- **Phase 1 — nine-card-first shell:** invert MainWindow so the
  nine-card home is the default screen; sidebar becomes an optional
  advanced mode; Home/Back + history; Settings via global menu;
  all 18 pages preserved as card workspaces; registry validation
  tests (exact order, Other-last, routes valid).
- **Phase 2 — full capability assignment:** every public capability
  mapped to exactly one primary card (§6.1 unique ownership); the
  C-list above reclassified as it lands; `Other` added last only if
  genuinely unassignable capabilities remain.
- **Phase 3 — backend integration completion:** build the missing GUI
  workflows for the C-list rows above (forms → execution → results →
  export), each calling the existing backend API (§3.3: one canonical
  implementation, never a GUI reimplementation).
- **Phase 4 — global search + shared workspace:** capability search /
  command palette over the registry; shared project/session state
  (datasets, models, runs, artifacts) extending `pharmaml.studio.runs`.
- **Phase 5 — refinement:** responsive polish, contextual help,
  dependency diagnostics, memory-aware jobs.
- **Phase 6 — verification:** registry-driven coverage tests,
  end-to-end workflow tests per card, dependency-matrix verification,
  documentation + screenshots, release validation.

## 7. Phase-0 gaps found (recorded, not yet fixed)

1. The nine-card grid is Dashboard-embedded, not the main screen
   (Phase 1 target).
2. `settings` is the only section no card references (must move to a
   global application menu per §2.1/§5.2).
3. The 19 backend-only modules in §4 above have no GUI route.
4. No global capability search exists yet (Phase 4).
5. No shared project/session layer beyond `studio.runs` evidence
   collectors (Phase 4).

*Generated by `scripts/build_capability_map.py` (R14, Phase 0); the
counts re-derive from the live tree on every run.*
