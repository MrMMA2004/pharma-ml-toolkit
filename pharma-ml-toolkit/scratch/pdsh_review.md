# PDSH review — per-file inventory + per-notebook disposition (round 40, 100x-157, Task 1-a)

Scope: the full clone of `github.com/jakevdp/pythondatasciencehandbook` at
`scratch/pdsh` (commit `d662314 Add source material from second edition`, clean
worktree, 264 tracked files — `git ls-files`), reviewed **completely and one by
one** as input to round 40's "add the missing items to playground.py" absorption.

Method: `scratch/pdsh_extract.py` (run with `<sandbox home>`) walks every
`git ls-files` entry and, for every `.ipynb` under `notebooks/`, extracts the title
(first markdown `#` cell), per-code-cell imports and the key APIs (sklearn
estimators, `pd./np./plt./sns.` calls) by simple regex; the dispositions below were
then assigned by hand against the toolkit surface at review time
(**v1.2.92, commit `c460fdb`** — i.e. *before* this round's absorption code lands:
verified by grep, no PCA / manifold / KMeans-surface / GaussianMixture / KernelDensity /
naive-Bayes / hexbin entry point exists in `src/pharmaml/` or `playground.py` yet;
KMeans appears only as an internal building block of `HierarchicalSVR`).

Dispositions:
- **COVERED-ALREADY** — maps onto the existing toolkit surface (named below).
- **ADDED-THIS-ROUND** — the ML/viz chapters with no surface that this round's
  sibling tasks add (the prescription this review hands them).
- **STRETCH** — a real capability gap, deliberately not this round; one line on the
  plausible pharma-GUI surface.
- **N-A-HONEST** — out of scope for a pharma-ML toolkit, with the honest reason.

Count verification (the brief said "62 notebooks"; the actual clone):
`git ls-files 'notebooks/*.ipynb' | wc -l` = **67** — **66 numbered content
notebooks** (00.00 → 06.00; the 04.13 slot is absent from the 2nd-edition source,
see below) **+ `Untitled.ipynb`** (a stray empty notebook). The brief's 62 = the 66
numbered minus the four pure-link-list "Further Resources" chapters (01.08, 03.13,
04.15, 05.15); those four are explicitly asked for as N-A-HONEST rows, so **the
table below carries one row per `.ipynb` file = exactly 67 rows**, covering
everything. `notebooks/data/` holds **5 files** (the brief's "8" is the
`notebooks_v1/data/` set): `births.csv`, `president_heights.csv`,
`state-abbrevs.csv`, `state-areas.csv`, `state-population.csv`.
`notebooks/figures/` holds **36** PNGs.

---

## 1. Repository inventory (every top-level entry)

| Entry | What it is | Teachable content? |
| --- | --- | --- |
| `README.md` | Book README — badges, how to run the notebooks, links to the free HTML/PDF editions. | No — metadata. |
| `LICENSE-CODE` | MIT license for the code (© 2016 Jacob VanderPlas). | No. |
| `LICENSE-TEXT` | CC BY-NC-ND 3.0 license for the book prose. | No. |
| `environment.yml` | Conda environment for executing the book. | No. |
| `requirements.txt` | pip requirements for the book. | No. |
| `.gitignore` | Standard ignores. | No. |
| `.gitmodules` | Declares `website/plugins/ipynb` + `website/plugins/pelican-plugins` (neither gitlink is even tracked here — the submodules are not checked out). | No — site infra. |
| `tools/` (5 files) | Book build tooling: `add_book_info.py`, `add_navigation.py`, `fix_kernelspec.py`, `generate_contents.py` + README. | No — not content. |
| `website/` (28 files) | Pelican site infrastructure: `pelicanconf.py`, `publishconf.py`, `Makefile`, `fabfile.py`, `copy_notebooks.py`, `content/favicon.ico`, and `theme/` (17 templates + css/fonts). | No — site infra, not content. |
| `notebooks/` (109 files) | **THE second-edition content**: 67 `.ipynb` (66 numbered + stray `Untitled.ipynb`), `data/` (5 CSVs), `figures/` (36 PNGs), `helpers_05_08.py` (decision-tree visualization helper used by 05.08 and 06.00). | **Yes — this is the review target (Section 2).** The 5 small CSVs are the datasets the notebooks load (`state-*` for the merge/pivot chapters, `president_heights`/`births` for the NumPy/pandas chapters). The figures are pre-rendered book images, not content. |
| `notebooks_v1/` (114 files) | The **older (first) edition of the same content**: 69 `.ipynb` (the same 66 numbered + `04.13-Geographic-Data-With-Basemap.ipynb`, which the 2nd edition dropped, + `Index.ipynb` + `Untitled.ipynb`), `data/` (8 CSVs — the 5 above plus `Seattle2014.csv`, `BicycleWeather.csv`, `california_cities.csv`), `figures/` (36 PNGs), `helpers_05_08.py`. | No *additional* teachable items — it is the same book one edition back; the only chapter unique to it (04.13 Basemap geographic plotting) is deprecated-library content, out of scope. |
| `notebooks_v2/` (1 file) | `data.csv` — an 11-row `a,b` scratch CSV; a leftover of the in-progress 2nd-edition conversion. | No — stray artifact. |

Notable per-file facts inside `notebooks/`: `helpers_05_08.py` (the
`visualize_tree` / `plot_tree_interactive` ipywidgets helper for 05.08 — used
only by the book's interactive tree figures), `Untitled.ipynb` (0 cells, no
content), and the **missing 04.13 slot**: `notebooks/` jumps 04.12 → 04.14
because the 2nd-edition source dropped the Basemap chapter; `04.13-Geographic-Data-With-Basemap.ipynb`
exists only under `notebooks_v1/`. Geographic plotting is out of scope for the
toolkit either way, so no disposition row is owed (the numbered row set is 66).

---

## 2. Notebook-by-notebook review (all 67 `.ipynb` under `notebooks/`, filename order)

| # | Notebook | Title | Core teachable capabilities | Disposition (toolkit mapping) |
| --- | --- | --- | --- | --- |
| 1 | 00.00-Preface | Preface | What is data science (Venn framing); why Python; book outline + software setup. | **N-A-HONEST** — book prose; the toolkit's README/CHANGELOG/docs already carry the framing role. |
| 2 | 01.00-IPython-Beyond-Normal-Python | Jupyter: Beyond Normal Python | Chapter roadmap only. | **N-A-HONEST** — frontispiece. |
| 3 | 01.01-Help-And-Documentation | Getting Started in IPython and Jupyter | Launch shell/notebook; `?`/`??` help; tab completion; wildcard help. | **N-A-HONEST** — the GUI is a tkinter app, not IPython; the toolkit's help surface is `docs/INDEX.md` + in-app tooltips. |
| 4 | 01.02-Shell-Keyboard-Shortcuts | Keyboard Shortcuts in the IPython Shell | Nav/edit-mode key bindings. | **N-A-HONEST** — the GUI has its own bindings (Ctrl+PgDn tab cycling, Ctrl+F card finder); analog, not an absorption. |
| 5 | 01.03-Magic-Commands | IPython Magic Commands | `%timeit`, `%%time`, `%%bash`, magics ecosystem. | **N-A-HONEST** — magics don't exist outside IPython. |
| 6 | 01.04-Input-Output-History | Input and Output History | `In`/`Out` caching, `_`, `__`, `%recall`. | **N-A-HONEST**. |
| 7 | 01.05-IPython-And-Shell-Commands | IPython and Shell Commands | `!` shell escapes, `cd` magic. | **N-A-HONEST**. |
| 8 | 01.06-Errors-and-Debugging | Errors and Debugging | `%xmode`, `%debug`, pdb, reading tracebacks. | **N-A-HONEST** — the toolkit's analogue is the typed `exceptions` module: every library failure raises a typed error the GUI renders as a dialog, never a raw traceback. |
| 9 | 01.07-Timing-and-Profiling | Profiling and Timing Code | `%time`, `%timeit`, `%prun`, `%memit`, line_profiler. | **N-A-HONEST** — no profiler surface; but timing exists as first-class timing fields in the toolkit's benchmark/training/tuner results. |
| 10 | 01.08-More-IPython-Resources | More IPython Resources | Pure link list. | **N-A-HONEST** — no content. |
| 11 | 02.00-Introduction-to-NumPy | Introduction to NumPy | Chapter roadmap; numpy import convention. | **N-A-HONEST** — numpy underpins the entire toolkit. |
| 12 | 02.01-Understanding-Data-Types | Understanding Data Types in Python | Python dynamic types; fixed-type numpy arrays; array constructors. | **N-A-HONEST** — library internals; nearest surface is `clean_numeric_frame`'s dtype coercion. |
| 13 | 02.02-The-Basics-Of-NumPy-Arrays | The Basics of NumPy Arrays | Attributes, slicing, reshaping, join/split. | **N-A-HONEST** — library internals. |
| 14 | 02.03-Computation-on-arrays-ufuncs | Computation on NumPy Arrays: Universal Functions | Ufunc arithmetic/trig/exp-log; vectorization. | **N-A-HONEST** — library internals (descriptor math). |
| 15 | 02.04-Computation-on-arrays-aggregates | Aggregations: min, max, and Everything in Between | `sum`/`min`/`max`/`median`/`percentile`; NaN-aware variants. | **N-A-HONEST** — aggregates surface as the reports' summary stats (RegressionReport/ClassificationReport means etc.) and the Data & ML tab's head/shape preview; a full `describe()` stats card is not present (unchecked). |
| 16 | 02.05-Computation-on-arrays-broadcasting | Computation on Arrays: Broadcasting | Broadcasting rules. | **N-A-HONEST** — used throughout the library internally. |
| 17 | 02.06-Boolean-Arrays-and-Masks | Comparisons, Masks, and Boolean Logic | Boolean masks, `count_nonzero`, and/or logic. | **N-A-HONEST** — masking powers cleaning/imputation internally. |
| 18 | 02.07-Fancy-Indexing | Fancy Indexing | Index arrays, combined fancy+slice, `bincount`/`searchsorted`. | **N-A-HONEST** — library internals. |
| 19 | 02.08-Sorting | Sorting Arrays | `argsort`, `partition`/`argpartition`, k-nearest neighbours via sorting. | **N-A-HONEST** — argsort/partition power the ranking internals. |
| 20 | 02.09-Structured-Data-NumPy | Structured Data: NumPy's Structured Arrays | `dtype` compounds, recarray. | **N-A-HONEST** — the toolkit models tables as DataFrames. |
| 21 | 03.00-Introduction-to-Pandas | Data Manipulation with Pandas | Chapter roadmap. | **N-A-HONEST**. |
| 22 | 03.01-Introducing-Pandas-Objects | Introducing Pandas Objects | Series/DataFrame/Index anatomy. | **N-A-HONEST** — every toolkit surface passes DataFrames; the GUI previews them. |
| 23 | 03.02-Data-Indexing-and-Selection | Data Indexing and Selection | `iloc`/`loc`, masking on frames. | **N-A-HONEST** — `split_features_target`/column pickers are the controlled subset a pharma GUI needs. |
| 24 | 03.03-Operations-in-Pandas | Operating on Data in Pandas | Index alignment; ufuncs on frames. | **N-A-HONEST**. |
| 25 | 03.04-Missing-Values | Handling Missing Data | NaN semantics; `isnull`/`fillna`/`dropna`; imputation idea. | **COVERED-ALREADY** — `clean_numeric_frame` (drop/impute with a dropped-column ledger) + `build_preprocessor`'s `SimpleImputer` branch (median/mean/most-frequent). |
| 26 | 03.05-Hierarchical-Indexing | Hierarchical Indexing | MultiIndex, stack/unstack. | **STRETCH** — a MultiIndex-aware Data tab (dose × replicate hierarchy browsing). |
| 27 | 03.06-Concat-And-Append | Combining Datasets: concat and append | `pd.concat`, append, repeated-index. | **STRETCH** — dataset-stacking card (combine replicate assay CSVs). |
| 28 | 03.07-Merge-and-Join | Combining Datasets: merge and join | `pd.merge` one-to-one/many-to-many on keys. | **STRETCH** — merge-two-loaded-tables-on-a-key card. |
| 29 | 03.08-Aggregation-and-Grouping | Aggregation and Grouping | Split-apply-combine, aggregation, `apply`, pivot preview. | **STRETCH** — groupby summary card (per-compound/batch aggregate stats). |
| 30 | 03.09-Pivot-Tables | Pivot Tables | `pivot_table`, `cut`/`qcut` binning. | **STRETCH** — pivot card (rows/cols/values pickers over the loaded frame). |
| 31 | 03.10-Working-With-Strings | Vectorized String Operations | `.str` accessor, regex extraction. | **STRETCH** — string-ops card (SMILES/categorical column cleaning). |
| 32 | 03.11-Working-with-Time-Series | Working with Time Series | DatetimeIndex, resample, rolling, shifting. | **STRETCH** — pharma time series already have a home (PK tab's `plot_concentration_profile`/`plot_dose_accumulation`); a generic resample/rolling card is the stretch. |
| 33 | 03.12-Performance-Eval-and-Query | High-Performance Pandas: eval and query | `pd.eval`/`query`, numexpr. | **STRETCH** — filter-expression card over the loaded frame. |
| 34 | 03.13-Further-Resources | Further Resources | Pure link list. | **N-A-HONEST** — no content. |
| 35 | 04.00-Introduction-To-Matplotlib | Visualization with Matplotlib | pyplot vs axes; figure/axes anatomy; styles. | **N-A-HONEST** — the toolkit's house plot style already embodies it (every `viz` function). |
| 36 | 04.01-Simple-Line-Plots | Simple Line Plots | Color/linestyle, axes limits/labels, legend. | **COVERED-ALREADY** — every house plot is a styled line plot: `plot_concentration_profile`, `plot_predicted_vs_actual`, `plot_learning_curve`. |
| 37 | 04.02-Simple-Scatter-Plots | Simple Scatter Plots | Size/color encodings, legend, `plot` vs `scatter`. | **COVERED-ALREADY** — `plot_predicted_vs_actual`, `plot_gp_uncertainty`, `plot_applicability_domain`, similarity scatters. |
| 38 | 04.03-Errorbars | Visualizing Uncertainties | `errorbar`, continuous error with `fill_between`, GP confidence band. | **COVERED-ALREADY** — `plot_gp_uncertainty` + `plot_prediction_intervals` (+ `quantile_prediction_intervals`/`bootstrap_uncertainty`). |
| 39 | 04.04-Density-and-Contour-Plots | Density and Contour Plots | `contour`/`contourf`/`clabel`; visualizing grid functions. | **ADDED-THIS-ROUND** — the 2D-density read (hexbin) lands this round; the contour-grid idea is partially embodied by the existing heatmaps. |
| 40 | 04.05-Histograms-and-Binnings | Histograms, Binnings, and Density | `hist`/binning, 2D histograms (`hist2d`), `hexbin`, density norming. | **COVERED-ALREADY** — `plot_class_distribution`, `plot_descriptor_distributions`, `plot_energy_distribution`; the `hist2d`/`hexbin` 2D-density part is added this round. |
| 41 | 04.06-Customizing-Legends | Customizing Plot Legends | Legend loc/multiple legends, custom artists. | **N-A-HONEST** — the house style owns consistent legends. |
| 42 | 04.07-Customizing-Colorbars | Customizing Colorbars | Colormaps, discrete colorbars, extensions. | **N-A-HONEST** — the house heatmaps already standardize colormaps. |
| 43 | 04.08-Multiple-Subplots | Multiple Subplots | `subplot`, `GridSpec`, insets. | **N-A-HONEST** — multi-panel report figures are already assembled by `viz/report.py`. |
| 44 | 04.09-Text-and-Annotation | Text and Annotation | Arrows/annotate, transforms. | **N-A-HONEST** — house plots carry their own annotation discipline. |
| 45 | 04.10-Customizing-Ticks | Customizing Ticks | Locators/formatters. | **N-A-HONEST**. |
| 46 | 04.11-Settings-and-Stylesheets | Customizing Matplotlib: Configurations and Stylesheets | rcParams, stylesheets. | **N-A-HONEST** — the house plot style *is* the stylesheet. |
| 47 | 04.12-Three-Dimensional-Plotting | Three-Dimensional Plotting in Matplotlib | mplot3d, wireframe/surface/tri-surface. | **STRETCH** — 3D reads (e.g. interaction surfaces); mplot3d adds a dimension the 2D card grid does not need. |
| 48 | 04.14-Visualization-With-Seaborn | Visualization with Seaborn | hist/kde/joint/pair plots, facets, categorical plots. | **N-A-HONEST** — out of scope: the matplotlib house style replaces seaborn wholesale; its kde/hist/pairplot ideas are partially absorbed via the new KDE plot. |
| 49 | 04.15-Further-Resources | Further Resources | Pure link list. | **N-A-HONEST** — no content. |
| 50 | 05.00-Machine-Learning | Machine Learning | Chapter roadmap. | **N-A-HONEST** — frontispiece. |
| 51 | 05.01-What-Is-Machine-Learning | What Is Machine Learning? | Classification vs regression vs clustering vs dimensionality-reduction taxonomy; qualitative examples. | **N-A-HONEST** — conceptual prose; the taxonomy is embodied by `build_model`'s task axis (classification/regression) plus this round's clustering/dim-reduction additions. |
| 52 | 05.02-Introducing-Scikit-Learn | Introducing Scikit-Learn | Data as (n_samples, n_features) table; fit/predict API; `train_test_split`; iris/digits end-to-end; accuracy + confusion matrix; Isomap/GMM previews. | **COVERED-ALREADY** — `split_features_target` + `make_train_test_split` + `train_model`/`evaluate_model` (ClassificationReport + `plot_confusion_matrix`); the Isomap/GMM previews map onto this round's new manifold/GMM surfaces. |
| 53 | 05.03-Hyperparameters-and-Model-Validation | Hyperparameters and Model Validation | Holdout vs CV; bias-variance tradeoff; validation curve; learning curve; `GridSearchCV`; LeaveOneOut caution. | **COVERED-ALREADY** — `ml.tuner` (`make_cv_splitter` over KFold/StratifiedKFold, RandomSearch/Hyperband over `cross_val_score`) + `ml.training`'s GridSearchCV path + viz `plot_learning_curve`/`plot_cv_scores`. |
| 54 | 05.04-Feature-Engineering | Feature Engineering | Categorical features (one-hot); text features (`CountVectorizer`/`TfidfVectorizer`); imputation; derived/polynomial features. | **STRETCH** — text features and the polynomial basis are not present (stretch, not this round); imputation + one-hot categorical are COVERED via `build_preprocessor` (`SimpleImputer` + `encode_categorical` one-hot). |
| 55 | 05.05-Naive-Bayes | In Depth: Naive Bayes Classification | GaussianNB on blobs; MultinomialNB + Tfidf on 20 newsgroups; BernoulliNB (prose); when-to-use. | **ADDED-THIS-ROUND** — Gaussian/Multinomial naive-Bayes spec surface (+Bernoulli in prose); the newsgroups text pipeline rides the 05.04 text-feature stretch. |
| 56 | 05.06-Linear-Regression | In Depth: Linear Regression | Simple linear regression; basis functions (`PolynomialFeatures`); Ridge/L2; Lasso/L1 sparsity. | **COVERED-ALREADY** — `linear_regression`/`ridge`/`lasso`/`elastic_net` (+`huber_regression`) ModelSpecs over the scaled preprocessor; the polynomial basis function is a noted STRETCH (not present). |
| 57 | 05.07-Support-Vector-Machines | In Depth: Support Vector Machines | Margins; kernel trick (RBF); face-recognition pipeline (PCA + GridSearchCV + classification_report/confusion_matrix). | **COVERED-ALREADY** — `svm` ModelSpec (RBF) + tuner GridSearchCV; the eigenfaces PCA preprocessing maps onto this round's PCA surface. |
| 58 | 05.08-Random-Forests | In Depth: Decision Trees and Random Forests | Tree overfitting/depth; bagging; RF classification (digits) + regression; interactive tree visualization (helpers_05_08). | **COVERED-ALREADY** — `decision_tree`/`random_forest`/`extra_trees` ModelSpecs + `plot_feature_importance`; the ipywidgets interactive tree visualization is out of scope (GUI has no widget runtime). |
| 59 | 05.09-Principal-Component-Analysis | In Depth: Principal Component Analysis | Rotation/projection intuition; digits components; explained-variance scree (`cumsum`); noise filtering; eigenfaces. | **ADDED-THIS-ROUND** — PCA surface: fit, components, explained-variance ratios, scree + 2D projected scatter. |
| 60 | 05.10-Manifold-Learning | In Depth: Manifold Learning | LLE vs MDS; Isomap on digits; pairwise-distance preservation. | **ADDED-THIS-ROUND** — manifold surface: Isomap/LLE/MDS/SpectralEmbedding/t-SNE with the embedded 2D scatter. |
| 61 | 05.11-K-Means | In Depth: k-Means Clustering | EM intuition; MiniBatchKMeans; k-means on digits (t-SNE preview + confusion matrix); color compression. | **ADDED-THIS-ROUND** — KMeans (+MiniBatch) clustering surface (cluster labels, centers, inertia); the t-SNE preview rides the manifold surface; color compression and the SpectralClustering cameo stay stretch. |
| 62 | 05.12-Gaussian-Mixtures | In Depth: Gaussian Mixture Models | EM vs k-means; covariance types; AIC/BIC model selection; GMM as density estimator; GMM as generator. | **ADDED-THIS-ROUND** — GaussianMixture surface (covariance types, AIC/BIC selection); the BayesianGMM variant and the GMM-as-generator digits demo are noted STRETCH. |
| 63 | 05.13-Kernel-Density-Estimation | In Depth: Kernel Density Estimation | Histogram→KDE motivation; bandwidth/kernel choice; bandwidth via GridSearchCV (LeaveOneOut); digits generative preview; custom KDE estimator. | **ADDED-THIS-ROUND** — KDE surface with CV-selected bandwidth; the spheres/haversine example (1st-edition content, absent here) and the custom-estimator anatomy stay out of scope. |
| 64 | 05.14-Image-Features | Application: A Face Detection Pipeline | PatchExtractor/sliding windows; GaussianNB/LinearSVC over patches; GridSearchCV. | **N-A-HONEST** — image patch features are out of scope for a pharma-ML tool; its estimator stack is covered (naive Bayes this round; `svm`/tuner already). |
| 65 | 05.15-Learning-More | Further Machine Learning Resources | Pure link list. | **N-A-HONEST** — no content. |
| 66 | 06.00-Figure-Code | Appendix: Figure Code | Rendering code for the book's figures (bias-variance, CV, GMM covariance, EM, decision trees, LLE-vs-MDS). | **N-A-HONEST** — book-figure rendering code; the figure patterns are absorbed as the new plot functions' patterns where applicable (learning/validation-curve, covariance-type, EM-gap). |
| 67 | Untitled.ipynb | *(no markdown heading — 0 cells)* | None — stray empty notebook. | **N-A-HONEST** — no content; a leftover artifact of the book repo. |

Row count check: 67 rows — `git ls-files 'notebooks/*.ipynb'` (66 numbered
00.00–06.00 with the 04.13 slot absent + `Untitled.ipynb`). The one chapter that
exists only in `notebooks_v1/` — 04.13 Basemap geographic plotting — is out of
scope (deprecated library, no pharma-ML surface) and is covered by the inventory
note above rather than a row.

---

## 3. Gap summary

**The concrete missing items — ML chapters with NO toolkit surface at review time
(v1.2.92), which this round adds:**
- **PCA** (05.09) — components, explained-variance scree, projected scatter.
- **Manifold learning** (05.10) — Isomap / LLE / MDS / SpectralEmbedding / t-SNE.
- **K-Means** (05.11) — KMeans + MiniBatchKMeans clustering (today KMeans exists
  only *internally* inside `HierarchicalSVR`; there is no first-class clustering
  surface).
- **Gaussian mixtures** (05.12) — GMM with covariance types + AIC/BIC selection
  (BayesianGMM and GMM-as-generator = stretch, noted).
- **KDE** (05.13) — KernelDensity with CV-selected bandwidth (the haversine/sphere
  geography example is out of scope).
- **Naive Bayes** (05.05) — Gaussian/Multinomial (Bernoulli in prose).

**Viz gaps:** hexbin/2D density (04.04 + 04.05's `hist2d`/`hexbin`) — added this
round; 3D plotting (04.12) — STRETCH, not this round.

**Feature gaps:** TF-IDF/Count text features (05.04, and 05.05's newsgroups
pipeline) — STRETCH, not this round unless a sibling task explicitly picks it up;
polynomial basis functions (`PolynomialFeatures`, 05.06) — not present, STRETCH.

**Honest out-of-scope statement:** the IPython chapter (01.01–01.08: magics,
shell, history, debugging, timing/profiling) is not absorbable — the GUI is a
tkinter app, not an IPython session; the only real overlap is timing, which
already exists as timing fields in the toolkit's benchmark/training/tuner
results. The pure NumPy fundamentals (02.01–02.09) and pandas fundamentals
(03.01–03.03) are library internals a pharma-ML GUI does not re-surface (02.04's
aggregates surface only as the reports' summary stats; 03.04's missing-data
handling is the one covered surface). The pandas reshaping chapters 03.05–03.12
remain documented stretch cards (groupby summary card, pivot card, merge card,
concat card, string-ops card, resample/rolling card, filter-expression card).
The matplotlib styling chapters (04.06–04.11) and seaborn (04.14) are superseded
by the toolkit's house plot style, which already embodies consistent styling.
The preface, chapter openers, the three further-resources link lists and the
figure-code appendix carry no teachable API. `notebooks_v1/` (the older edition
of the same content), `website/` (Pelican infra), `tools/` (build tooling) and
the `figures/` directories carry no additional teachable items.

**Disposition tallies (67 rows):** COVERED-ALREADY **10** · ADDED-THIS-ROUND **7**
· STRETCH **10** · N-A-HONEST **40**.
