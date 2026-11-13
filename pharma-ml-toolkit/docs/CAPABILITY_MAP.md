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
  ├─ TopBar (Home / Back / location / jobs / Settings / Advanced)
  └─ mode stack: NineCardHome (default) → CardWorkspaces ×9 →
       QStackedWidget(18 pages, unchanged)
       (the 18-section sidebar rides only behind the Advanced toggle)
```

Since v1.2.185 (Phase 1) the Studio opens on the nine-card home;
`MainWindow.go_to(section)` remains the single routing API and all 18
pages are preserved.  Since v1.2.186 (Phase 2) the card ORDER and the
capability assignments live in the pure-Python registry
`src/pharmaml/studio/capability_registry.py` (Qt-free, §6);
`ui/nine_cards.py` asserts agreement at import time.

## 2. The eighteen pages and their verified backends

| Page | pharmaml modules imported (live grep) | card assignment (Phase 2 target) |
| --- | --- | --- |
| `dashboard` | studio | 01 Introduction (current host of the §5 grid) |
| `data_explorer` | ml.research_audit, ml.sar, ml.split_audit, studio, utils.io | 02 Datasets |
| `chemistry` | chem, chem.csv_standardize, chem.features, utils.io | 02 Datasets (chemical preparation) |
| `pkpd` | pharma.pk, pkpd.compartments, pkpd.events, pkpd.nca_ext, pkpd.ode, studio | 07 PK/PD |
| `biology` | biology | 07 PK/PD (biology sub-workspace) |
| `docking` | docking, exceptions, studio | 06 Molecular Docking |
| `model_analysis` | ml, ml.cv_results, ml.models, ml.training, utils.io | 03 Machine Learning |
| `ml_workbench` | ml, ml.calibration, ml.comparison, ml.conformal, ml.conformal_classification, ml.coverage_diagnostics, ml.cv_results, ml.decision, ml.drift, ml.evaluation, ml.model_card, ml.models, ml.multiplicity, ml.persistence, ml.power, ml.subgroups, ml.tuner, studio, utils.checkpoints, utils.io | 03 Machine Learning |
| `unsupervised` | ml.manifold, ml.unsupervised, utils.io | 03 Machine Learning (exploration) |
| `deep_learning` | exceptions, ml, ml.autograd, ml.curation, ml.geometric, ml.harness, ml.neural, ml.numpy_neural, ml.pruning, ml.quantization, ml.runboard, ml.splits, ml.umap, ml.xai, studio, utils.io | 04 Deep Learning |
| `generative` | chem.oracles, chem.sa_scoring, generative, studio | 05 Generative AI |
| `foundations` | ml.inferential, ml.stats, profiling | 03 Machine Learning (walkthroughs) |
| `bestofml` | ml.adversarial, ml.annsearch, ml.apriori, ml.catalog, ml.privacy, ml.recommend, ml.sequences, ml.timeseries | 03 Machine Learning (sweeps) |
| `responseai` | ml.drugresponse | 07 PK/PD (response AI) |
| `image_lab` | biology, chem.descriptors, chem.similarity, studio, viz.export | 09 Visualization |
| `molecular_dynamics` | md, studio, viz, viz.export | 08 Molecular Dynamics Analysis |
| `utilities` | chem, chem.converters, docking, pharma, utils.io, viz | 02 Datasets + cross-card utilities |
| `settings_page` | utils.resources | 01 Introduction (global menu) |

Tests: 636 UI test functions across
46 UI suites; page smoke
coverage via `tests/ui/test_pages_smoke.py`.

## 3. Backend census

- `src/pharmaml/ml/`: **92 modules**,
74,074 lines.
- CLI: **62** `add_parser` sites (16 command groups).
- Library tests: 4552 test functions across
139 suites.
- `src/pharmaml/studio/` (runs + evidence): the §11 real-evidence
  backend already exists — Phase 4 extends, not replaces, it.

## 4. Backend-only capabilities — now ASSIGNED (Phase 2 complete)

The Phase-0 audit found these modules referenced by no Studio page.
As of v1.2.186 every one carries its **primary card assignment**
(part6 §6.1) in the machine-checkable registry
`src/pharmaml/studio/capability_registry.py`; the 11 §6.2 validation
rules are pinned by `tests/test_capability_registry.py` (27 pins).
Status `api` means the backend and its tests are real, and the GUI
form is the Phase-3 work:

| Module | What it does | Primary card | Route (Phase-3 form) |
| --- | --- | --- | --- |

### The Other-card decision (part6 §2.3)

After the full assignment pass, **no Other card is created** (§2.3
rule 8): every genuine user-facing capability above has a logical
home — 13 modules under Machine Learning, 5 under Deep Learning, 1
under Datasets — and the remaining Studio sections (biology,
foundations, best-of-ml, responseai) are integrated pages with real
backends, already reachable.  The decision is pinned by
`TestRule3OtherPosition` so it can only change through a new,
justified assignment audit, never by drift.

## 5. What counts as integrated (§3.2, restated as the gate)

A capability is GUI-integrated only when its workflow: collects and
validates inputs → invokes the real backend → reports progress and
errors → displays real results → supports inspection and export →
preserves metadata/provenance.  Import presence, a button, or a
synthetic demo is NOT integration.  Phase 6's coverage tests enforce
this against the registry, not against prose.

Registry coverage (live): **62 capabilities**, 62 integrated; orphans 0; test gaps 0; Other card absent. Round 24 (part6 §17A/B) added `utilities.templates` (the seven workflow-template chains, every stage calling an existing backend implementation) and `utilities.pipelines` (the §17B sequential pipeline engine).

## 6. Migration plan (Phases 1-6)

- **Phase 1 — nine-card-first shell (DONE, v1.2.185):** invert MainWindow so the
  nine-card home is the default screen; sidebar becomes an optional
  advanced mode; Home/Back + history; Settings via global menu;
  all 18 pages preserved as card workspaces; registry validation
  tests (exact order, Other-last, routes valid).
- **Phase 2 — full capability assignment (DONE, v1.2.186):** every public capability
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

## 7. Gaps (1-3 CLOSED; 4-5 are the Phase-4 targets)

1. CLOSED v1.2.185 (Phase 1): the nine-card grid IS the main screen.
2. CLOSED v1.2.185 (Phase 1): Settings lives on the global top bar.
3. CLOSED v1.2.186 (Phase 2): all 19 backend-only modules carry
   primary-card assignments in the registry; GUI forms = Phase 3.
4. No global capability search exists yet (Phase 4).
5. No shared project/session layer beyond `studio.runs` evidence
   collectors (Phase 4).

*Generated by `scripts/build_capability_map.py` (R14, Phase 0); the
counts re-derive from the live tree on every run.*
