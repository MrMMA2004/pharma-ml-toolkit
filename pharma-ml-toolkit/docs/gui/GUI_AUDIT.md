# GUI Audit — the Tkinter playground mapped for the Studio redesign

Directive #16 (GUI redesign, `upload/MASTER_PROMPT part2.pdf`, 16.2/16.4):
before redesigning the interface, inventory every existing screen and map
it to its backend. This is the increment-1 audit (round 206); it is kept
current as sections migrate.

## The existing application

`playground.py` (Tkinter + ttk, ~45k lines) is the current GUI: one
`ttk.Notebook` with thirteen top-level tabs, 129 `make_card` cards, the
`WORKFLOWS` registry (11 guided workflows / 58 steps), per-card
descriptions, per-tab sample data, CSV **and** Excel I/O (v1.2.116+), and
the v1.2.119 heavy-work safety kit (Stop button, memory watchdog,
pre-flight batch guards, ML n_jobs cap, cooperative `JobCancelled`
checkpoints).

## Tab inventory → backend map → Studio status

| # | Playground tab | Cards (approx. clusters) | Backend modules | Studio page (increment 1) |
|---|----------------|--------------------------|-----------------|---------------------------|
| 1 | Home | quick-jump chips, workflow guide | `WORKFLOWS` registry | **Dashboard** (live census, dataset index) |
| 2 | Chemistry | standardize, descriptors, fingerprints, feature factory, feature recalculation | `pharmaml.chem.*` | **Chemistry Workbench** (increment 2: standardize + multi-family feature factory with the batch guard) |
| 3 | PK / PD | dose response, exposure, pregnancy PK | `pharmaml.pkpd.*` | **PK/PD Workbench** (increment 3: exact two-compartment regimen simulator + trapezoid/linear-up-log-down NCA + the NONMEM event grammar) |
| 4 | Cell Lines | expression, screening | `pharmaml.biology.*` | **Biology Workbench** (increment 4: screen audit + sensitivity matrix with quartile calls + MAD gene ranking + biomarker discovery) |
| 5 | Docking | DOCKSTRING lab, affinity, campaigns | `pharmaml.docking.*` | **Docking Workbench** (increment 5: campaign scan + completion audit + 3-column consensus rescore + DLG cluster tables + ligand library audit + the 58-box DOCKSTRING registry) |
| 6 | Data & ML | QSAR training, leaderboards, AD | `pharmaml.ml.*` | **Model Analysis** (increment 13, round 226: registry fit + curves, PLUS the Part-3 §19 CV interface ported from the playground — the Compare-models board over `compare_models_cv` with the fold-count spinner + the fold-strategy combo (auto/kfold/stratified/group/stratified_group/scaffold), the honest pre-flight refusals for scaffold/group sourcing, the formatted `mean ± std (k-fold CV)` leaderboard, and the double-click CV report dialog — fold table with train-side mirrors, per-metric aggregates with directions, notes, the PyQtGraph fold chart with the ± 1 std band edges) |
| 7 | Unsupervised | clustering, reduction | `pharmaml.ml.unsupervised` | **Unsupervised Workbench** (increment 6: PCA variance ladder + loadings, k-means/mini-batch/GMM/hierarchical clustering with the soft responsibilities, KDE density over the histogram twin, all five manifolds) |
| 8 | Deep Learning | runboard, UMAP, curation, figures | `pharmaml.ml.*` (torch) | **Deep Learning Workbench** (increment 7: the training run — DeepMLP*/ResidualMLP* (torch) or NumpyMLP* (numpy autograd) with early stopping + evaluate_model parity/overfit or per-class F1 + confusion; the runboard — RunTrace EMA loss curves + numpy_layer_stack weight histograms with the honest torch refusal; the UMAP map — umap_embed + Venna-Kaski trustworthiness; the fiftyone-style curation — DataView stage algebra + the four audits) |
| 9 | Generative | RNN/RL optimization | `pharmaml.generative.*` | **Generative Workbench** (increment 8: the evolution run — GeneticOptimizer / ReinventOptimizer / MctsOptimizer over build_objective-built objectives with the ranked leaderboard + progress curves; the Pareto front — pareto_front over the multi-objective breakdown with the trade-off scatter; molecule surgery — ring_topology_moves + scaffold_hop_crossover breeding; the GuacaMol scoring oracles) |
| 10 | Foundations | statistics, table prep | `pharmaml.utils.*`, `profiling` | **Foundations Workbench** (increment 9: compare_groups with the branch reported + two-histogram overlay, the five-distribution sampler with the theory drawn in the same value space, Welch ANOVA + Tukey HSD + Games-Howell, profile_dataset with the type census + alert rules) |
| 11 | Best-of ML | curated model cards | `pharmaml.ml.*` | **Best-of ML Workbench** (increment 10: the 924-project catalogue as queryable data + the 35-verdict coverage audit + the five native capability surfaces — DP statistics with the RDP accountant, leave-one-out collaborative filtering, the LSH recall/compression sweep, the adversarial broken-rate audit, the time-series analysis half — plus the apriori association rules) |
| 12 | Response AI | drug response | `pharmaml.ml.drugresponse` | **Response AI Workbench** (increment 11: the module's FIRST GUI surface — the Hill curve fit with every parameter against the generating truth + the log-dose AUC both ways, response metrics + rank correlations on the Data Explorer's columns or seeded pairs, the CaDRReS `DrugResponseModel` on the seeded 12x4 screen with the honest in-sample pearson, cell-line-grouped cross-validation with the ~0 mean reported honestly, the informative-prior regression, the `ResponseVAE` ELBO ledger, the `MultiOmicsFusion` per-block audit with total variance equalised) |
| 13 | Utilities | CSV standardize, converters, IO | `pharmaml.utils.io`, `chem.csv_standardize` | **Data Explorer** (tables + distributions) + **Utilities Workbench** (increment 13, round 227: the rar's others/General scripts — table merge & split over `merge_tables`/`merge_table_folder`/`split_table_into_chunks`, compare & annotate over `compare_tables`/`annotate_with_smiles` with the native reconciliation bars, the approved-drugs report over `filter_approved_drugs` (CSV + the bordered Word table), the campaign file tools over `collect_files`/`list_file_names`/`distribute_files`/`delete_named_files`/`cleanup_tree` dry-run-first, and `reindex_by_number`) |

Plus the cross-cutting concerns, all live in Studio: theme/appearance
(Settings page), navigation (sidebar + Ctrl+1..9), background
execution (JobWorker/QThreadPool with cooperative stop), state
persistence (QSettings: geometry/nav/theme), and — increment 12
(round 221, the Part-2 §16.8.2-16.8.4/16.8.7 closure) — matrix
image visualization (the **Image Lab**: `ui/widgets/image_panel.py`
+ `ui/pages/image_lab.py`, an ImageItem + HistogramLUT colormap
control + live-ROI sub-matrix statistics + PNG export, five tabs on
real project data: the 12-drug Tanimoto similarity heatmap, the
23x8 LN_IC50 response matrix, the explorer's numeric columns, the
fixed-region ROI demo, and the tanimoto timing surface).

## Preserved / redesigned / consolidated / replaced

- **Preserved (backend)**: every `pharmaml` module is reused as-is;
  Studio calls the same public API surface the playground cards call
  (`read_table`, `train_model`, `MODEL_REGISTRY`, …). No backend
  behaviour changes in this round.
- **Preserved (GUI)**: the Tkinter playground itself stays fully
  functional and shipped until every section above has a verified
  Studio page (directive 16.18 step 15).
- **Redesigned**: the presentation layer only — notebook tabs become
  sidebar navigation; card grids become dedicated pages; matplotlib
  report figures become live PyQtGraph panels.
- **Consolidated**: the Stop button + memory watchdog survive as the
  Studio's `JobWorker` cooperative-stop channel (same
  `JobCancelled` contract as v1.2.119).
- **Replaced**: nothing yet (increment 1; retirement happens only after
  each replacement is verified).

## Known issues found during the audit (16.4)

1. The Tkinter GUI blocks on every card action — long jobs freeze the
   whole window (the v1.2.119 Stop/watchdog kit mitigates the OOM
   failure mode, but the thread model remains single-threaded). The
   Studio's QThreadPool workers fix the root cause.
2. Tab order encodes history, not workflow frequency — the sidebar's
   eleven live sections are ordered by usage (dashboard → data →
   chemistry → pkpd → biology → docking → models → unsupervised →
   deep learning → generative → settings).
3. No theme or appearance control exists in Tkinter; Studio ships the
   full provider stack (Qt-Material primary, QDarkStyleSheet
   alternative, native fallback) with runtime switching.

## Part-2 (§16) completion assessment — round 226, the owner's question

*The owner asked directly: "Check if MASTER_PROMPT part2 is completely
done and retire playground.py (everything must be in studio.py)."  This
is the honest verdict against directive §16.22's acceptance criteria
(`upload/MASTER_PROMPT part2.pdf`), with the evidence.*

**Done and verified** (each with its own tests in `tests/ui/`):

- PySide6 integrated, the Studio architecture (`studio.py` → `ui/`)
  with modular pages/widgets/workers/theme (16.5/16.16).
- PyQtGraph scientific visualization: live charts everywhere, the
  Image Lab (increment 12: ImageView-style colormap/ROI/histogram over
  real project matrices) (16.8).
- Qt-Material primary + QDarkStyleSheet alternative + native fallback,
  runtime switching, ChartPalette re-colour of every plot, QSettings
  persistence (16.9–16.11, 16.15).
- Navigation, JobWorker background execution with the cooperative
  Stop channel, input validation, error handling (16.14).
- GUI documentation (`docs/gui/GUI_REDESIGN.md`, this audit) and the
  GitHub backup of all GUI sources/tests (16.20/16.21 J).

**NOT yet done — the one open parity item** (why step 15 cannot fire):

1. **Model Analysis depth** — the playground's Data & ML tab carries
   more than the registry fit + curves the Studio page had at
   increment 1 (leaderboard variants, the family benchmark, the tuner
   card, applicability-domain). Round 226 ported the newest and most
   valuable piece — the §19 CV results interface — but the remaining
   ML-tab cards still owe their Studio twin.

(Resolved in round 227, increment 13: the second former blocker —
the Utilities workflow cards — now has its dedicated Studio page,
`ui/pages/utilities.py` + `tests/ui/test_utilities_page.py`, with
every card wired to the same public backend APIs the playground
cards call and the destructive campaign paths dry-run by default.)

(Resolved in round 228, increment 14 — the retirement round: the
Model-Analysis depth landed as the **ML Workbench**
(`ui/pages/ml_workbench.py` + `tests/ui/test_ml_workbench_page.py`,
18 page tests) — the Hyperparameter-search card (RandomSearch /
TPESearch / Hyperband over every grid-bearing registry family with
the defaults baseline and the held-out verdict, the trial-ranking
board, Apply-best), the **family benchmark**
(`benchmark_feature_families` over the SMILES column), the **OOF
stacking leaderboard** (`stack_oof_regressors` with the residual
correction and the native parity scatter), the **ensemble builder**
(`build_voting` / `build_stacking`), **model persistence**
(`save_model` / `load_model` / `validate_bundle_input` predict with
the honest feature-contract audit) and the **applicability domain**
(`applicability_domain` with the reliability table, the kNN-distance
native scatter and the bootstrap-uncertainty status).  The Model
Analysis page itself grew the manual-hyperparameters table
(`train_model`'s `model_params`), the kept-run `last_run()` hand-off
and four NATIVE diagnostic figures — calibration curve, learning
curve, ROC, PR — plus the latent string-class-target crash fix the
regression-only tests had never met.)

**Therefore (round 228)**: Part-2 is **100% complete** — every
§16.22 surface (architecture, theming, visualization, workers, docs,
and ALL THIRTEEN content surfaces) is verified with its own tests,
and §16.18 step 15 **fires**: `playground.py` is retired this round
(deleted with its 980-test Tk-coupled suite: `test_playground.py`
876 + `test_playground_workflows.py` 17 + `test_job_safety.py` 31 +
`test_excel_io_and_samples.py` 26 + `test_feature_recalculation.py`
30), the Studio (`python studio.py`) is THE front end, and the pins
carry the new honest totals.


## Part-3 progress — round 229: §20 checkpoint depth

*The Part-3 backlog after Part-2's completion: §20 checkpoint depth /
§21 hover tooltips / §22 SMILES previews / §23 responsive pass / §24
SA-in-generative wiring.  Round 229 landed §20's engine and its first
Studio wiring.*

- **§20 (pause/resume/stop + checkpoints) — the engine and the
  Leaderboards tab are DONE.**  `pharmaml.utils.checkpoints`
  (`CheckpointManager`: atomic fsync+rename saves, SHA-256
  self-verification, corrupt-file quarantine with fallback-to-previous,
  rotation, strict input validation, JSON states — no pickle) is the
  durable state layer; `benchmark_feature_families` grew
  `checkpoint_dir`/`checkpoint_name`/`resume` (per-family checkpoints
  written BEFORE the stop check, signature-matched resume
  consecutive-from-the-front, write failures degrade to warnings);
  the ML Workbench Leaderboards tab checkpoints into
  `~/.pharmaml/checkpoints/family-benchmark` so its Stop button keeps
  the finished families and the next Start resumes, with the resume
  story in the status line ("checkpoint found: k/n families already
  scored — resuming there") and the tooltip.  36 engine tests + 6
  integration tests + 1 GUI loop test.
- **Remaining §20 surface (optional depth)**: the other long runners
  (the docking campaign, the generative batches, the feature factory's
  molecule loop) can adopt the same `CheckpointManager` — the engine
  is generic and they already carry the cooperative-stop channel.
- **§21 hover tooltips / §22 SMILES previews / §23 responsive pass /
  §24 SA-in-generative wiring** — unchanged, still open.
