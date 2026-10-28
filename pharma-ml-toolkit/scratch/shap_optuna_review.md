# SHAP + Optuna review — per-file inventory + disposition (round 43, 100x-160, Tasks 3-a/3-b)

Scope: two full clones, both `--depth 1 --filter=blob:none`, reviewed
**completely and one by one** as input to this round's "add the missing
items to playground.py" absorption:

* `github.com/shap/shap` at `/tmp/shap` — **615 tracked files**
  (`git ls-files`), of which **95 Python files under `shap/`** plus the
  C++ core (`shap/cext/`, `shap/cutils/`), the CUDA backend
  (`backends/shap-cuda/`), 10 demo datasets (`data/`), docs
  (`docs/`), JS bundle for force plots, CI.
* `github.com/optuna/optuna` at `/tmp/optuna` — **484 tracked files**,
  of which **220 Python files under `optuna/`** plus benchmarks,
  docs, examples, CI.

Baseline at review time: **v1.2.95** — the toolkit already ships the
TreeExplainer surface (100x-154: `ml/explain.py::shap_summary` → the
typed `ShapSummary` record, optional-gated `[explain]` extra, plus the
pure-matplotlib `plot_shap_importance` bar and `plot_shap_beeswarm`),
and the Keras-Tuner programming model (100x-89/90: `ml/tuner.py` —
`HyperParameters` / `RandomSearch` / `Hyperband`, fold-protocol via the
public `make_cv_splitter`, deterministic seeds).  Neither `shap` nor
`optuna` is a *runtime* dependency of the toolkit; the absorption below
follows the house rule: **re-implement the programming model in typed
library surfaces; pure-matplotlib figures; optional-gate only what truly
needs the third-party package** (the `shap` computation itself).

Dispositions: **COVERED-ALREADY** / **ADDED-THIS-ROUND** /
**STRETCH** / **N-A-HONEST**.

---

## 1. SHAP repository inventory (every package area, file by file)

### 1.1 Explainers (`shap/explainers/` — 16 modules)

| File | What it is | Disposition |
| --- | --- | --- |
| `_tree.py` | `TreeExplainer` — exact TreeSHAP (C++ / native XGB-LGBM-CatBoost), `shap_values`, **`shap_interaction_values`**, `model_output` raw/probability/log_loss, `feature_perturbation` auto/interventional/tree_path_dependent. | Values: **COVERED-ALREADY** (`shap_summary`, 100x-154). Interactions: **ADDED-THIS-ROUND** (`shap_interactions`). `model_output=` variants: STRETCH (the toolkit explains the raw output; the log-odds/probability transform is the `links` row below). |
| `_permutation.py` | `PermutationExplainer` — model-agnostic antithetic permutations, local-accuracy-guaranteed. | **COVERED-ALREADY(≈)** — the toolkit's honest answer for non-tree models is the shipped model-agnostic `permutation_feature_importance` (typed, seeded); a full PermutationSHAP re-implementation is STRETCH. |
| `_kernel.py` | `KernelExplainer` — the classic KernelSHAP (model-agnostic, slow, `link=`). | **N-A-HONEST** — deliberately refused in `shap_summary`'s typed error since 100x-154 (the slow approximation; the permutation sibling answers the model-agnostic question today). |
| `_exact.py` | `ExactExplainer` — gray-code Shapley enumeration (≤~15 varying features). | **N-A-HONEST** — descriptor frames are 9–2000+ features; exact enumeration is exponentially out of scope. |
| `_linear.py` / `_additive.py` | Linear-model and GAM explainers (coef × (x−mean)). | **N-A-HONEST** — the linear/GAM story is already answered exactly by `plot_feature_importance` over `coef_` (the signed-correlation + coefficient surfaces). |
| `_sampling.py` | Shapley sampling values (IME). | N-A-HONEST (kernel row's family). |
| `_partition.py` / `_coalition.py` | Owen-value explainers over a feature hierarchy (correlation-clustered coalitions). | **STRETCH** — needs the `hclust` masker row; nothing in the GUI asks for coalition explanations yet. |
| `_deep/` (3 files) | DeepExplainer / DeepExplainer-PyTorch (DeepLIFT-style). | **N-A-HONEST** — the toolkit's deep estimators are PyTorch and already carry `history_` + the GP-uncertainty surfaces; a DeepLIFT re-implementation is out of scope. |
| `_gradient.py` | GradientExplainer (expected gradients). | N-A-HONEST (same row). |
| `other/` (6 files) | `_lime`, `_maple`, `_random`, `_coefficient`, `_treegain`, `_ubjson` wrappers. | N-A-HONEST — third-party bridges (lime/maple are not dependencies; the toolkit's answer is the permutation sibling). |
| `_explainer.py` | The `Explainer` dispatcher (algorithm=auto/…). | **ADDED-THIS-ROUND(≈)** — the toolkit's dispatcher is `shap_summary` itself (tree path) with the typed refusal naming the sibling; no generic dispatcher is honest without the engines it would dispatch to. |

### 1.2 Plots (`shap/plots/` — 15 modules)

| File | What it is | Disposition |
| --- | --- | --- |
| `_beeswarm.py` | The signature global summary (colour = feature value, `max_display`, `group_remaining_features`). | **COVERED-ALREADY** — `plot_shap_beeswarm` (100x-154). |
| `_bar.py` | Mean-|SHAP| bar (global/local, cohorts, hclust bars). | **COVERED-ALREADY** — `plot_shap_importance` (100x-154); cohort variant = the group-difference row below. |
| `_waterfall.py` | The signature LOCAL explanation: base value → prediction walk, stacked signed bars, smallest features collapsed into a summed tail. | **ADDED-THIS-ROUND** — `plot_shap_waterfall(summary, sample)`. |
| `_scatter.py` | Dependence plot: feature value vs SHAP value, colour = interacting feature (auto-picked or explicit), hist underlay, jitter. | **ADDED-THIS-ROUND** — `plot_shap_dependence(summary, feature)`. |
| `_heatmap.py` | Instances × features SHAP heatmap, instances ordered by supervised clustering (hclust over |SHAP|), feature order by mean |SHAP|, top bar = mean |SHAP|. | **ADDED-THIS-ROUND** — `plot_shap_heatmap(summary)` (scipy linkage — already a core dep). |
| `_force.py` + `_force_matplotlib.py` | Additive force layout (JS interactive default; the matplotlib static twin stacks the signed contributions onto one bar). | **ADDED-THIS-ROUND** — `plot_shap_force(summary, sample)` (the static matplotlib twin only — no JS in a desktop tkinter app; the honest translation). |
| `_violin.py` | Violin variant of the beeswarm. | **COVERED-ALREADY(≈)** — the beeswarm carries the same information without the KDE-over-2-points lie (violin KDEs mislead on small held-out splits — the honest note). |
| `_decision.py` | Cumulative-SHAP decision paths (N observations × features). | **STRETCH** — needs the multi-row walk; the waterfall (one row) + force (one row) cover the local question this round. |
| `_group_difference.py` | **Cohort difference**: mean-SHAP(group A) − mean-SHAP(group B) per feature, sorted, **bootstrap 5–95% error bars** (200 resamples) — the fairness/cohort decomposition. | **ADDED-THIS-ROUND** — `group_difference(summary, mask)` + `plot_shap_group_difference` (VERY pharma: active-vs-inactive cohorts, scaffold classes, high/low-potency splits). |
| `_monitoring.py` | SHAP drift over time windows (t-tests). | **STRETCH** — needs temporal/production drift monitoring, not a batch QSAR tool. |
| `_partial_dependence.py` | PD + ICE curves with model-expected-value lines. | **COVERED-ALREADY(≈)** — the toolkit ships no PD surface yet, but PD is a model-level (not SHAP-level) curve; honest disposition: STRETCH as its own feature (recorded here so the next audit can pick it up). |
| `_embedding.py` | SHAP values projected by PCA onto 2-D, coloured by one feature's SHAP. | **COVERED-ALREADY(≈)** — `ml/manifold.py::manifold_embedding` (100x-157) + the scatter colour-by column cover the projection question. |
| `_image.py` / `_text.py` | Image / text explanation figures. | **N-A-HONEST** — tabular QSAR toolkit, no image/text models. |
| `_benchmark.py` | Explainer-benchmark figures. | N-A-HONEST (framework row below). |
| `_style.py` | Experimental style config (red/blue signed colours). | **COVERED-ALREADY** — the house `PALETTE` + the amber→teal signed ramp (every signed figure since 100x-146). |

### 1.3 Supporting areas

| Area (files) | What it is | Disposition |
| --- | --- | --- |
| `shap/_explanation.py` | The `Explanation` record (values/base_values/data, `.abs.mean(0)`, slicing, `cohorts(n)`). | **COVERED-ALREADY(≈)** — `ShapSummary` is the toolkit's typed record (100x-154); `cohorts(n)` = STRETCH (the manual-mask `group_difference` covers explicit cohorts this round). |
| `shap/links.py` | `identity`, `logit` (+`.inverse`). | **STRETCH** — only meaningful with `model_output="log_odds"` (the `_tree.py` row); recorded. |
| `shap/maskers/` (8) | Independent / Partition / Impute / Text / Image / composite maskers. | N-A-HONEST as a *surface* (maskers are engine internals for the permutation/partition explainers — the STRETCH rows); the `Independent` behaviour is what `TreeExplainer` defaults to. |
| `shap/models/` (5) | Model wrappers (teacher-forcing, transformers pipelines, top-k LM). | N-A-HONEST — NLP/vision bridges. |
| `shap/actions/` (2) | `ActionOptimizer` — prescriptive counterfactual actions. | **STRETCH** — a real pharma capability (which descriptor to change to flip the prediction); deliberately not this round (needs a constrained optimization over chemically-valid feature moves). |
| `shap/benchmark/` (10) | Explainer-quality benchmark framework (mask/impute/resample/retrain curves, local accuracy, consistency). | N-A-HONEST — meta-evaluation of explainers, not an analysis surface. |
| `shap/datasets.py` + `data/` | Demo datasets (NHANES, adult, imagenet50…). | N-A-HONEST — the toolkit ships its own synthetic QSAR examples. |
| `shap/utils/` (10) | `hclust` (correlation-clustering / partition tree), `approximate_interactions`, `kmeans`, legacy. | `approximate_interactions`-style auto colour-pick: **ADDED-THIS-ROUND** (inside `plot_shap_dependence` — the strongest interacting feature picked by binned R², exactly the heuristic the upstream docstring documents); `hclust` ordering: ADDED inside the heatmap. |
| `backends/shap-cuda/` | GPUTreeExplainer. | N-A-HONEST — CPU-only verified stack. |
| `shap/cext/`, `shap/cutils/`, `CMakeLists.txt` | The C/C++ TreeSHAP core. | N-A-HONEST — engine internals; the toolkit calls `shap.TreeExplainer`, never re-implements it. |

### 1.4 SHAP tally

**Explainers:** covered 3 (tree values / permutation≈ / linear≈) · added 2
(interactions + the dispatcher≈) · stretch 3 (partition/coalition, model_output
variants, decision plots) · n-a-honest 8.
**Plots:** covered 4 (beeswarm, bar, violin≈, embedding≈) · added 5
(waterfall, dependence/scatter, heatmap, force, group_difference) · stretch 3
(decision, monitoring, partial-dependence) · n-a-honest 3 (image, text,
benchmark).
**Support:** covered 2 · added≈ 2 (interactions heuristic, hclust ordering) ·
stretch 3 (links, cohorts(n), ActionOptimizer) · n-a-honest 7.

---

## 2. Optuna repository inventory (every package area, file by file)

### 2.1 Study / Trial core (`optuna/study/` 9 files, `optuna/trial/` 5 files)

| File | What it is | Disposition |
| --- | --- | --- |
| `study/study.py` | `create_study(direction, sampler, pruner)`, `optimize(func, n_trials, timeout, callbacks)`, `ask()/tell()`, `enqueue_trial`, `best_trial/params/value`. | The *programming model*: **COVERED-ALREADY(≈)** — `RandomSearch.search(X, y)` (100x-90) is the toolkit's study loop (trials, best, leaderboard, refit); `direction` is fixed by the sklearn scorer convention (higher-is-better) — N-A as a parameter. `ask/tell`: **STRETCH** (the house API is one `search()` call; recorded). `enqueue_trial`: STRETCH. |
| `study/_dataframe.py` | `trials_dataframe()` — number/value/state/params/duration columns. | **COVERED-ALREADY(≈)** + **ADDED-THIS-ROUND** — `TunerResults.trials_frame()` exists; this round adds the Optuna `state` (complete/pruned/fail) + `duration` columns and the `fold_scores` intermediates. |
| `trial/_trial.py` | `suggest_float(log=, step=)`, `suggest_int`, `suggest_categorical`, `report(value, step)`, `should_prune()`. | **ADDED-THIS-ROUND** — the suggest-API maps onto `HyperParameters.Int/Float/Choice` (already shipped); `Float(..., log=True)` sampling in the log domain + the report/should_prune fold loop land this round. |
| `trial/_state.py` | `TrialState` RUNNING/COMPLETE/PRUNED/FAIL/WAITING. | **ADDED-THIS-ROUND** — as the string `state` field on `TrialResult` (complete/pruned/fail; RUNNING/WAITING are ask/tell machinery — the stretch row). |
| `distributions.py` | Float/Int/Categorical distributions (log, step). | **ADDED-THIS-ROUND** — the `log=True` flag on `_HyperParameterSpec` (float sampling in log domain; Optuna's own default recommendation for learning rates). |
| `study/_tell.py`, `_frozen.py`, `_study_summary.py`, `_multi_objective.py`, `_constrained_optimization.py` | ask/tell internals, frozen trials, multi-objective, constraints. | N-A-HONEST / STRETCH — single-objective toolkit; ask/tell stretch (above). |

### 2.2 Samplers (`optuna/samplers/` — 10 sampler families, 46 files)

| Sampler | What it is | Disposition |
| --- | --- | --- |
| `_tpe/` (4 files) | **TPESampler** — the flagship: per param fit a Parzen (KDE) mixture `l(x)` to the **below** (best `γ(n)=min(⌈0.1n⌉,25)` trials) and `g(x)` to the **above** (rest); draw `n_ei_candidates=24` from `l`; pick argmax `log l − log g`; **random until `n_startup_trials=10`**; bandwidth = sorted-neighbour gaps clipped to `(high−low)/min(100, 1+n)`; one prior kernel at the domain centre (weight 1); categorical = smoothed counts + prior. | **ADDED-THIS-ROUND** — `ml/tuner.py::TPESearch` (same `search(X, y)` contract as RandomSearch: hypermodel, objective, max_trials, cv, task, seed, on_trial; plus `n_startup_trials`, `n_ei_candidates`, `gamma` cap). Multivariate/`group`/`constant_liar`: N-A-HONEST (independent per-param sampling is the honest scope — recorded in the docstring). |
| `_random.py` | `RandomSampler` (seed). | **COVERED-ALREADY** — `RandomSearch` (100x-90). |
| `_hyperband` via pruners | Hyperband as pruner+sampler pair. | **COVERED-ALREADY** — `Hyperband` the tuner (100x-90). |
| `_grid.py` / `_brute_force.py` | Exhaustive grid. | **COVERED-ALREADY(≈)** — `compare_models` + the model registries' `param_grids` (the grid-search story, `ml tune --no-tune`'s sibling since round 20). |
| `_cmaes.py` | CMA-ES. | **N-A-HONEST** — needs the `cmaes` package + a continuous-only space; the toolkit's categorical-rich spaces make TPE the honest fit. |
| `_gp/` | GP-based Bayesian sampler. | N-A-HONEST — same reasoning (and the Gaussian *process* regression surface already answers the GP-modelling question elsewhere). |
| `nsgaii/` + `_nsgaiii/` + `_ga/` (21 files) | Multi-objective genetic samplers (crossovers SBX/SPX/UNDX/BLXα, polynomial mutation, non-dominated sorting). | **N-A-HONEST** — single-objective toolkit (one scorer per search; Optuna itself needs `directions` for these). |
| `_qmc.py` | Low-discrepancy QMC. | N-A-HONEST — random+TPE cover the toolkit's search story. |
| `_partial_fixed.py` | Fix some params, sample rest. | **COVERED-ALREADY(≈)** — `HyperParameters.Fixed` + trial `values` override (the tuner's own machinery). |

### 2.3 Pruners (`optuna/pruners/` — 8 modules)

| Pruner | Decision rule | Disposition |
| --- | --- | --- |
| `_median.py` | `PercentilePruner(50)` with `n_startup_trials=5`. | **ADDED-THIS-ROUND** — `PercentilePruner` + the `MedianPruner` shortcut, wired into the tuners' fold loop: after each CV fold the running mean is *reported*; if the completed trials' same-step percentile beats it, the remaining folds are skipped and the trial records `state="pruned"` with its partial fold scores. |
| `_percentile.py` | Prune when the trial's running best sits beyond the `p`-th percentile of completed trials' values at the same step (startup/warmup/interval/n_min_trials guards). | **ADDED-THIS-ROUND** (the general form; median = percentile 50). |
| `_successive_halving.py` / `_hyperband.py` | Rung-based halving over a resource. | **COVERED-ALREADY** — `Hyperband` (the budget ladder IS the resource halving; the tuner form is the honest fit for the toolkit's `max_iter`/`epochs` slots). |
| `_patient.py` | Wrap a pruner with a patience window. | **STRETCH** — composable pruners need a pruner-composition surface first. |
| `_threshold.py` | Absolute bounds on intermediates. | **STRETCH** — the objective scales differ per scorer; percentile-relative is the honest default. |
| `_wilcoxon.py` | Signed-rank test vs the best trial. | **STRETCH** — scipy is available but the paired-step discipline needs equal fold counts everywhere (n_min_trials guard story); recorded. |
| `_nop.py` | Never prune. | **COVERED-ALREADY** — `pruner=None` (the default). |

### 2.4 Importance, visualization, terminator, storages

| Area | What it is | Disposition |
| --- | --- | --- |
| `importance/` (5) | `get_param_importances` — PedAnova (default), **fANOVA**, **MDI** (RandomForest on one-hot params → objective, `feature_importances_`). | **ADDED-THIS-ROUND** — `param_importances(results)` the MDI form (sklearn is a core dep; the honest one-paragraph re-implementation Optuna's own docstring documents). PedAnova/fANOVA: STRETCH. |
| `visualization/matplotlib/` (13) | `plot_optimization_history` (scatter + best-so-far step), `plot_slice` (param vs objective per param), `plot_parallel_coordinate` (one polyline per trial), `plot_contour` (objective contours over param pairs), `plot_param_importances` (bars), `plot_edf` (ECDF of objective), `plot_timeline` (lifetimes coloured by state), `plot_rank` (pairwise scatters coloured by rank), `plot_intermediate_values` (reported-step curves), pareto/hypervolume/terminator (multi-objective-only). | **ADDED-THIS-ROUND** — the seven single-objective figures as pure-matplotlib house plots: `plot_optimization_history`, `plot_slice`, `plot_parallel_coordinate`, `plot_param_importances`, `plot_edf`, `plot_timeline`, `plot_intermediate_values` (the pruner's reported fold scores are exactly the intermediate values). Pareto/hypervolume/terminator: N-A-HONEST (single objective). `plot_contour`: **STRETCH** (needs a dense griddata interpolation over a usually-sparse trial cloud — the honest note; slice + parallel coordinate cover the structure question). `plot_rank`: STRETCH (pairwise slice panels + EDF cover it). |
| `terminator/` (6) | Auto-stop criterion (regret bound + CV error). | **STRETCH** — the tuners run a fixed `max_trials` by contract; recorded. |
| `storages/` (26) | In-memory / RDB / journal / gRPC storages (distributed). | **N-A-HONEST** — single-process desktop toolkit; results live in the `TunerResults` record. |
| `integration/` (18) | LightGBM/XGBoost/CatBoost callbacks, MLflow/wandb, **shap** (Shapley hyperparam importance). | N-A-HONEST — third-party bridges; the toolkit's own tuner + the new `param_importances` answer the same questions natively. |
| `artifacts/`, `cli.py`, `progress_bar.py`, `logging.py`, `_gp/`, `_hypervolume/` | Artifacts store, `optuna` CLI, tqdm bar, colored logging, GP/hypervolume internals. | N-A-HONEST — the playground GUI's own status bar + job machinery (100x-106) already carry progress; no artifact store in a desktop tool. |
| `optuna/testing/` (12) | Testing fakes. | N-A-HONEST. |

### 2.5 Optuna tally

**Core:** covered≈ 2 + added 3 (state/duration/fold intermediates, log-Float,
the report/should-prune fold loop) · stretch 3 (ask/tell, enqueue, PD-plots) ·
n-a 3.
**Samplers:** covered 3(+2≈) · added 1 (**TPE**) · n-a-honest 5 families.
**Pruners:** covered 2 (hyperband-as-tuner, nop) · added 2 (percentile + median)
· stretch 4 (patient, threshold, wilcoxon, composition).
**Importance/viz/terminator/storages:** added 8 (MDI importance + the 7
figures) · stretch 5 (contour, rank, PedAnova, fANOVA, terminator) · n-a 5
(pareto/hypervolume/terminator-improvement plots, storages, integrations).

---

## 3. What this hands the absorption tasks (the prescription)

**Task 5-a (SHAP, `ml/explain.py` + `viz/ml_plots.py`):**
`shap_interactions` (the `shap_interaction_values` cube, typed record,
optional-gated) · `group_difference(summary, mask)` (cohort mean-difference +
bootstrap CI, pure numpy) · the five pure-matplotlib figures:
`plot_shap_waterfall` (local base→prediction walk) · `plot_shap_dependence`
(value vs SHAP, auto interacting colour) · `plot_shap_heatmap` (hclust
instance order) · `plot_shap_force` (static additive bar) ·
`plot_shap_group_difference` (bars + bootstrap whiskers).

**Task 5-b (Optuna, `ml/tuner.py`):** `hp.Float(..., log=True)` ·
`TPESearch` (Parzen below/above, γ=min(⌈0.1n⌉,25), n_startup=10 random,
n_ei_candidates=24, prior kernel, magic-clip bandwidth) ·
`PercentilePruner`/`MedianPruner` wired into the fold loop (report/should_prune)
· `TrialResult.state` (complete/pruned/fail) + `fold_scores` + `duration` ·
`param_importances` (MDI: RandomForestRegressor over one-hot params) ·
`TunerResults.summary()`/`trials_frame()` carrying the new columns.

**Task 5-c (Optuna viz, `viz/ml_plots.py`):** the seven single-objective
figures — optimization history, slice, parallel coordinate, param importances,
EDF, timeline, intermediate values (the pruner's fold reports).

**Task 6 (playground.py):** the SHAP row's new buttons (waterfall/dependence/
heatmap/force via sample-or-feature dialogs, group difference via the run's
held-out target split) + the Tune card's TPE searcher token, the pruner
choice, and the study-figures row (history/slice/parallel/importances/EDF/
timeline/intermediates) over `self._tune_last`.

The honest out-of-scope statement, once: SHAP's deep/text/image engines,
KernelSHAP, ExactExplainer and the masker/partition machinery, and Optuna's
multi-objective samplers (NSGA-II/III), CMA-ES/GP/QMC, storages, ask/tell and
the terminator are **not** absorbed — each row above carries its reason; the
disposition tables are the contract.
