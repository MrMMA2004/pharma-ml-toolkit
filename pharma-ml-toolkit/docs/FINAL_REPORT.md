# PharmaML Studio — part8 Final Report (v1.2.212)

**Spec:** MASTER_PROMPT part8 (comprehensive audit, remediation,
feature expansion, performance engineering, nine-workspace workflow
architecture).  **Baseline:** v1.2.204 archive.  **Final state:**
v1.2.212, eight implementation rounds (r260–r267 releases shipped the
feature rounds; this round, r268, is Phase G — final verification and
this report).  Every number below was re-measured live at 1.2.212 in
this round's verification run.

Status labels: **PASS** (implemented and verified by tests), **PARTIAL**
(implemented, incompletely verified), **FAIL** (missing/broken),
**BLOCKED** (environmental limit), **NOT APPLICABLE** (with reason).

## 1. Executive summary

The part8 directive is complete: all fifteen phases (§1–§17) and all
§19 acceptance criteria are **PASS** or carry an explicitly justified
NOT APPLICABLE / BLOCKED-with-reason label.  The shipped state is a
Studio that opens on an Introduction-only home, navigates horizontally
through exactly nine workspaces in the directive's order, executes
every heavy operation off the GUI thread through one JobRunner, and
gives each workspace a sequential scientific workflow with provenance,
inspection, and export at every stage.

Final verification (this round, live): **7,219 tests green, 0
failures** — 5,370 backend (151 suites, all at 100%), 912 src
doctest contracts, 937 UI (69 suites, reconciled exactly against the
collection count).  6,334 regular tests + 915 doctest items collected
repo-wide; the 30 not run in the verification batches are the
environment-gated `checkout_only` / `web_checkout_only` profiles
(documented in §16 below).  One real defect was found and fixed during
Phase G: a stale doctest in `ml/persistence.py` (see §6).

## 2. Previous-requirement compliance matrix

| part8 requirement | Implementation | Status |
| --- | --- | --- |
| §2 traceability matrix | `docs/CAPABILITY_MAP.md` (page→backend→test, derived live) | PASS |
| §2.1 nine-card→Introduction | `ui/main_window.py` + `ui/introduction.py`; `test_nine_card_first.py` (23) | PASS |
| §2.1 horizontal 9-workspace nav | `ui/widgets/workspace_nav.py` `WORKSPACE_ORDER` (exact order); 18 pages preserved as subpages | PASS |
| §3 backend audit | 151 backend suites; audits in `docs/` (DATA_AUDIT, LEAKAGE_AUDIT, COVERAGE…) | PASS |
| §4 integration matrix | `docs/FEATURES.md` (per-module: GUI page, cards, routes, tests, grade) + `pharmaml.viz.coverage` | PASS |
| §4.1 shared figure rendering | `viz/qt_backend.py` + `ui/widgets/figure_canvas.py` (FigureCanvasQTAgg, leak-guarded) | PASS |
| §4.2 plot coverage registry | `viz/coverage.py` `viz_coverage_table()`; `docs/COVERAGE.md` | PASS |
| §5 GUI-thread freeze elimination | JobRunner + QRunnable/QThreadPool on every heavy path (load, fit, CV, HPO, predict, plots, exports) | PASS |
| §5.2 resource-aware import | preflight estimate → preview → background read with progress; `read_table_with_progress` | PASS |
| §5.3 efficient tables | `DataTableView`/`PandasTableModel` QAbstractTableModel, preview window, no eager full render | PASS |
| §5.4 job management | Job Center: names, progress, cancellation, errors, resource monitor (§17E) | PASS |
| §5.5 performance tests | `test_stress_performance.py` (17 green this round) | PASS |
| §6 manual n_jobs | `ui/widgets/njobs_field.py` editable validation (±ints, −1, 0 rejected, no clamping); `test_njobs_field.py` | PASS |
| §7 Chemistry Lab exact config | descriptor catalog + searchable selection, per-family fingerprint params (radius/bits/count), persisted metadata | PASS |
| §8 semi-supervised | `ml/semi_supervised.py`: SelfTraining, LabelPropagation, LabelSpreading; leakage-safe eval; `test_semi_supervised.py` (32) | PASS |
| §9 point inspection | ChartInspector: hover readout, nearest-point, click-select, cached per-point metadata, SMILES depiction | PASS |
| §10 crosshair removal | crosshair OFF by default, optional per-chart toggle; `test_chart_inspector.py` | PASS |
| §11 complete AD | `ml/applicability.py`: kNN distances, threshold, bootstrap uncertainty, prediction intervals, combined reliability, per-sample table; `test_applicability*.py` (62) | PASS |
| §12 nine-workspace GUI | Introduction-only startup + horizontal nav + subpages (above) | PASS |
| §13 workflow per workspace | `ui/widgets/workflow_guide.py` `WORKFLOWS` — all nine workspaces; lint suite pins steps/pages | PASS |
| §14.1 active learning | `ml/active_learning.py` (uncertainty/margin/entropy/committee strategies); ML Workbench card | PASS |
| §14.2 justified features | Proposal A (dataset provenance, r6) + Proposal B (reproducibility manifests, r7); declined list with reasons in `docs/part8_SECTION_14_2_PROPOSALS.md` | PASS |
| §15 exports | CSV/JSON/Excel + PNG/JPEG/TIFF/SVG/PDF through `viz/export.py` (16-bit TIFF, quality-controlled JPEG); chart-export buttons on every chart page | PASS |
| §16 testing strategy | layered: unit, GUI, integration, regression, performance, environment verification (this report) | PASS |
| §17 documentation | README, CHANGELOG, FEATURES, CAPABILITY_MAP, GUI_GAP_AUDIT, RELEASE_READINESS regenerated at 1.2.212; version carriers agree | PASS |
| Polars lazy scan (§5.2 optional) | evaluated; pandas pipelines retained — optional dependency not adopted (documented decision, §16 below) | NOT APPLICABLE |

## 3. Backend audit findings

The §3 audit (rounds 1–2, re-verified this round) found: input
validation gaps in two readers (fixed, `test_converters.py`), one
leakage risk in a CV splitter (fixed, `LEAKAGE_AUDIT.md`), numerical
instability in two convergence loops (guarded), and index-identity
loss in three export paths (fixed via stable sample IDs).  Current
state: 151 backend suites, 5,370 tests, 0 failures; the audit docs
under `docs/` carry the per-module evidence.

## 4. GUI audit findings

The §4 audit established the integration matrix now maintained in
`docs/FEATURES.md`: every pharmaml module lists its GUI page, cards,
routes, and pinning tests, graded A–D.  The historical gaps (plots
without export, backend functions without GUI, GUI controls without
execution paths) were closed round by round; the plot-coverage
registry (`viz/coverage.py`) records every public visualization
function and its integration status, including the honestly-listed
backend-only remainder.

## 5. Backend–GUI integration matrix

Maintained live in `docs/FEATURES.md` (per-module table) and
`docs/CAPABILITY_MAP.md` (per-page table).  Both are generated from
the working tree — page imports grepped, routes enumerated, tests
collected — and both were trued to 1.2.212 this round.

## 6. Confirmed bugs and root causes (part8-scope finds)

| Round | Bug | Root cause |
| --- | --- | --- |
| r2 | CSV read on GUI thread in 4 pages | direct pandas calls in slots → moved to JobRunner jobs |
| r3 | ChartInspector crosshair distraction | InfiniteLine pair shown on every move → default-off + opt-in |
| r5 | n_jobs fixed-list (1/2/4) in Chemistry Lab | combo without free entry → editable validated field |
| r6 | dataset reload silently reused stale frame | no per-path versioning → content_version + column_signature |
| r7 | PySide6 6.12 QWidget abort after `--doctor` | font probe left a bare QGuiApplication → builds QApplication |
| r7 | `loadFromData(format=b"PNG")` ValueError on 6.12 | upstream signature change → str/bytes dual path |
| **r8 (this round)** | **stale doctest in `ml/persistence.py`** | `bundle_metadata` grew `feature_distributions`/`feature_ranges` keys; the docstring example still asserted the old first-four sorted keys → example trued to the actual contract, 912/912 doctests green |

## 7. Fixes implemented

All §6 bugs are fixed at the source with pinning tests; no wrapper or
suppression.  The r8 fix is the docstring correction above (doctests
are tested contracts under this project's `--doctest-modules` addopts).

## 8. Semi-supervised Learning implementation details

`pharmaml.ml.semi_supervised` wraps scikit-learn's
`SelfTrainingClassifier`, `LabelPropagation`, and `LabelSpreading`
(the official implementations; the estimator-vs-base_estimator
version split is handled).  The `-1` unlabeled convention is
validated (labeled/unlabeled counts reported; legitimate classes
never converted).  `evaluate_semi_supervised` is the mandatory
leakage-safe protocol: genuine labeled holdout, pseudo-labels never
scored as truth, supervised baseline on labeled-train only, recorded
protocol.  Transductive behavior of the graph methods is declared.
32 tests pin the surface; the ML Workbench carries the workflow
(dataset → target → labeled/unlabeled definition → algorithm →
training → evaluation → visualization → export).

## 9. Additional features and their GitHub sources

| Feature | Source (verified API, per §14.3) |
| --- | --- |
| Active learning strategies | native implementation patterned on scikit-activeml (https://github.com/scikit-activeml/scikit-activeml) — uncertainty/margin/entropy/committee |
| Semi-supervised | scikit-learn `sklearn.semi_supervised` (https://github.com/scikit-learn/scikit-learn) |
| Reproducibility manifests (r7) | native; precedent studied: reprozip (https://github.com/ViDA-NYU/reprozip) — declined as too heavy, JSON sidecar instead |
| Reproducibility: joblib semantics | joblib (https://github.com/joblib/joblib) n_jobs conventions |
| Explainability | SHAP (https://github.com/shap/shap) 0.45–0.59 range pin |

## 10. GUI performance improvements

All heavy paths execute through JobRunner (QRunnable +
QThreadPool): file reads (with progress), descriptor/fingerprint
generation, model fitting, CV, HPO, prediction, plot generation for
large datasets, report generation, large exports.  Table display is
a preview-windowed QAbstractTableModel (the analysis frame is never
truncated to the view).  The Job Center exposes names, progress,
cancellation, errors, and the §17E live resource monitor.  Stress
tests (17, green) cover large CSV/Excel import, wide descriptor
matrices, fingerprint calculation, model fitting, CV, large
prediction sets, and large plot generation.

## 11. Chemistry Lab configuration improvements

Individual descriptor selection (searchable catalog, per-family
grouping, Select All/Clear All, saved reusable configuration),
independent per-family fingerprint configuration (type, bit/count,
nBits, Morgan radius — never cross-applied), manual n_jobs, and
persisted feature-generation metadata (selected descriptors, families,
radius/bits/mode, rdkit/mordred versions, failed/skipped descriptors,
feature-column order) — the §7.4 reproducibility contract.

## 12. Plot and figure integration status

`viz/coverage.py`'s registry records every public visualization
function with its integration status.  Backend Figure objects render
through the shared `FigureCanvasQTAgg` component (resize-safe,
leak-guarded, empty/loading states, export through the §17 raster
engine).  Existing Qt-native charts remain where more appropriate
with a documented mapping.  The backend-only remainder is listed
honestly rather than hidden.

## 13. Applicability Domain completion status

Complete §11 workflow: training kNN-distance distribution +
threshold (interpreted), query distances, predicted-vs-distance with
in/out-domain categories, bootstrap uncertainty + threshold,
prediction intervals where supported, combined reliability view (in/out-domain × low/high uncertainty — never invented when the
calculation was not run), and the sample-level table (SMILES, stable
ID, prediction, distance, thresholds, status).  The five concept
distinctions (AD vs uncertainty vs intervals vs confidence vs
reliability) are documented in the module and rendered distinctly.
62 tests pin the surface.

## 14. Test results with exact counts

| Layer | Suites | Tests | Result |
| --- | --- | --- | --- |
| Backend (tests/) | 151 | 5,370 | **PASS** — 0 failed, all batches 100% |
| src doctests | 206 modules | 912 items | **PASS** — 0 failed (after the r8 fix) |
| UI (tests/ui/) | 69 | 937 | **PASS** — 0 failed, count reconciled against collection |
| Carriers + phase6 + inventory + notebook variants + stress | 5 | 87+17 | **PASS** |
| **Total executed** | — | **7,219+** | **0 failures** |
| Collected, not executed | — | 30 | environment-gated profiles (see §16) |

Collection: 6,334 regular + 915 doctest items = 7,249.  Notebooks:
8/8 re-executed at 1.2.212 with embedded figures, no error outputs,
no home-path survivors.

## 15. Remaining limitations

1. **Optional dependencies stay optional**: mordred is not installed
   in this environment (its descriptor set reports unavailability
   honestly; the catalog explains the gap).  selfies was installed
   this round to run its 17 doctest contracts — still an optional
   extra, not a core dependency.
2. **Polars not adopted** (§5.2's "evaluate" directive): pandas
   pipelines + chunked progress reads cover the demonstrated needs;
   scan_csv lazy loading remains a documented future option if a
   real large-file bottleneck appears.
3. **Test-environment teardown noise**: PySide6 interpreter teardown
   in this sandbox segfaults after the last test passes (after
   `[100%]` is printed); this is an environment artifact of the
   offscreen Qt + glibc combination, not a test failure — every
   batch's completion was verified by the `[100%]` marker and dot
   counts reconciled against collection.
4. **checkout_only / web_checkout_only profiles**: 30 collected
   tests require the companion Web Console checkout or Git remote
   metadata; they are marked, documented, and excluded from the
   standalone profile by design (`docs/STANDALONE_TESTING.md`).

## 16. Dependency and compatibility changes

- **rdkit pinned to 2026.09.1** in the test environment (2026.09.2
  shows mutation-chemistry drift — `[pH]` artifacts and different
  pareto front sizes; investigated per §16.6 and resolved as a
  version-sensitive difference, not a code defect).
- **PySide6 6.8–6.12 compatibility**: both signature changes from
  6.12 are handled at the source (r7 fixes, §6 above); the toolkit
  runs on either stack.
- **selfies** added to this verification environment only (optional
  `[selfies]` extra, unchanged for users).
- numpy stays pinned 1.26.4 (mordred compatibility).

## 17. Updated architecture and workflow description

```
studio.py → ui.run() → MainWindow
  ├─ TopBar + horizontal WorkspaceNav (exact §12.1 order)
  ├─ Introduction page (only at startup)
  ├─ 9 workspaces = QTabWidget hosts over the preserved 18 pages
  │    every tab sequence follows its §13 WORKFLOWS guide
  ├─ JobRunner/QThreadPool: all heavy work off-thread
  │    Job Center: progress, cancel, errors, resource monitor
  └─ shared state: workspace registry (provenance + content_version)
       → runs registry (recorded hyperparams/seeds/n_jobs)
       → reproducibility manifests (§14.2 B)
```

## 18. Files created, modified, or removed (this round)

Modified: `VERSION`, `pyproject.toml`, `src/pharmaml/__init__.py`
(1.2.212); `src/pharmaml/ml/persistence.py` (stale doctest trued);
`docs/RELEASE_READINESS.md` (regenerated); `docs/CHANGELOG.md`,
`docs/FEATURES.md`, `docs/INDEX.md` (round entries); web mirror
`src/lib/toolkit-version.ts` + `src/lib/toolkit-stats.ts` (pins:
7,249 collected / 1,370 files / 469,011 lines, re-measured with the
census's exact walk rules); 8 example notebooks re-executed.  Created:
this report (`docs/FINAL_REPORT.md`).  Removed: nothing.

## 19. Release-readiness status

`tools/release_readiness.py` regenerated at 1.2.212: **18
dependencies installed, 0 broken; 220 suites / 6,334 tests
collected; version carriers agree** (VERSION, pyproject,
`pharmaml.__version__`, web TS mirror).  The release zip
`pharma-ml-toolkit-v1.2.212.zip` ships testzip-verified to the four
homes with the sha recorded in the release commit.

## 20. Acceptance criteria (§19) — final labels

| Criterion | Label | Evidence |
| --- | --- | --- |
| Starts on Introduction only | **PASS** | `test_opens_on_the_introduction_page`, `test_nine_card_dashboard_not_the_default` |
| Exactly nine workspaces | **PASS** | `WORKSPACE_ORDER` (9) + `test_nine_workspaces_in_the_directive_order` |
| Horizontal nav, directive order | **PASS** | `test_horizontal_bar_carries_the_exact_order`, `test_bar_is_visible_by_default` |
| 18 detailed pages preserved as subpages | **PASS** | `test_all_eightteen_pages_survive`, `test_every_section_routes_into_its_workspace` |
| Coherent sequential workflow per workspace | **PASS** | `WORKFLOWS` (9/9) + guide-lint suite |
| Mineral Design System preserved | **PASS** | design-token + theme-screenshot suites |
| GUI strings English-only | **PASS** | no RTL/Persian surfaces (audited); lint suite |
| Backend APIs inventoried | **PASS** | `docs/INDEX.md` (939 symbols at r7, regenerated) + FEATURES |
| Visualization functions inventoried | **PASS** | `viz/coverage.py` registry + `docs/COVERAGE.md` |
| GUI integration status explicit | **PASS** | FEATURES grades A–D per module |
| High-value plots connected | **PASS** | coverage registry; backend-only remainder listed |
| Backend figures render + export | **PASS** | shared FigureCanvas + export engine suites |
| No unverified integration claims | **PASS** | this report's evidence-based labels |
| CSV/Excel import off GUI thread | **PASS** | `load_path` JobRunner path + stress tests |
| Heavy calculations off GUI thread | **PASS** | JobRunner audit + worker lifecycle tests |
| Large tables displayed efficiently | **PASS** | preview-windowed model tests |
| Progress + error feedback | **PASS** | Job Center suites |
| Jobs cancellable | **PASS** | cancellation tests |
| Large-dataset performance tested | **PASS** | `test_stress_performance.py` (17) |
| Individual descriptor selection | **PASS** | catalog + selection tests |
| Fingerprint type configurable independently | **PASS** | per-family config tests |
| Morgan radius configurable | **PASS** | parameter propagation tests |
| Fingerprint bit size configurable | **PASS** | parameter propagation tests |
| Feature config reaches backend | **PASS** | end-to-end config-flow tests |
| Config metadata saved | **PASS** | §7.4 persistence tests |
| n_jobs accepts all valid nonzero ints incl. −1 | **PASS** | `test_njobs_field.py` |
| n_jobs=0 rejected | **PASS** | validation tests |
| No silent overwrite of user config | **PASS** | "no clamp" tests |
| Click shows sample metadata | **PASS** | inspector click tests |
| Actual/predicted values shown | **PASS** | readout tests |
| SMILES + stable IDs preserved | **PASS** | alignment tests (shuffled/duplicate/filtered/split) |
| Metadata aligned after split/shuffle/filter | **PASS** | same |
| Crosshair hidden by default | **PASS** | Phase-9 test |
| Hover + click-to-select functional | **PASS** | 17 inspector tests |
| Training-distance distribution visualized | **PASS** | AD view A tests |
| Distance threshold displayed + explained | **PASS** | AD interpretation tests |
| Query vs predicted + distance visualized | **PASS** | AD view B tests |
| Bootstrap uncertainty displayed | **PASS** | AD view C tests |
| Combined AD/uncertainty status | **PASS** | AD view D tests (never invented) |
| Sample-level AD inspection | **PASS** | AD view E tests |
| AD backend plotting connected | **PASS** | figure-integration tests |
| AD exports + tests complete | **PASS** | export suite + 62 AD tests |
| SelfTrainingClassifier | **PASS** | semi-supervised suite (32) |
| LabelPropagation | **PASS** | same |
| LabelSpreading | **PASS** | same |
| Unlabeled samples handled correctly | **PASS** | `-1` convention tests |
| Leakage-safe evaluation | **PASS** | `evaluate_semi_supervised` tests |
| Supervised baseline provided | **PASS** | baseline metrics in every eval |
| Pseudo-labels + confidence inspectable/exportable | **PASS** | export tests |
| Semi-supervised visualization connected | **PASS** | Workbench card tests |
| Version compatibility tested | **PASS** | estimator/base_estimator split handled + tested |
| Backend unit tests complete | **PASS** | 151 suites / 5,370 green |
| GUI tests complete | **PASS** | 69 suites / 937 green |
| Integration tests complete | **PASS** | user→GUI→backend→worker→result→export chains |
| Regression tests complete | **PASS** | preserved-behavior suites green |
| Performance tests complete | **PASS** | stress suite green |
| Documentation updated | **PASS** | §17 doc set regenerated at 1.2.212 |
| Version information consistent | **PASS** | carriers agree (readiness probe) |
| No unsupported capability advertised | **PASS** | honest-remainder listings (coverage registry, FEATURES) |

**Verdict:** every applicable §19 criterion is PASS.  The two
NOT-APPLICABLE / environment notes (Polars not adopted; 30
profile-gated tests not executed standalone) are documented above
with reasons.  The application is release-ready at v1.2.212.
