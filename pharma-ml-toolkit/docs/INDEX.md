# Module index

Auto-derived from each module's `__all__` exports — the authoritative quick
reference for the public API. Re-run `python scratch/build_docs_index.py`
after adding new public functions.

## `pharmaml`

*Package root: exceptions + version*

| Symbol | Kind | Summary |
|---|---|---|
| `PharmaMLError` | class | Base class for all intentional PharmaML errors. |
| `DataValidationError` | class | Raised when input data fails validation (bad columns, NaN-only |
| `DependencyError` | class | Raised when an optional dependency is required but not installed. |
| `DockingError` | class | Raised when a docking preparation or execution step fails. |
| `DockingParseError` | class | Raised when a docking log/output file cannot be parsed. |
| `UnitConversionError` | class | Raised for unknown units or dimensionally invalid conversions. |
| `__version__` | constant |  |

## `pharmaml.data`

*Bring your own data: load user CSV/TSV/Excel files*

| Symbol | Kind | Summary |
|---|---|---|
| `TARGET_CANDIDATES` | constant |  |
| `Dataset` | class | A user-supplied dataset split into features and target. |
| `DatasetSummary` | class | Human-readable audit of a loaded dataset. |
| `load_dataset` | function | Load **your own** data file and split it ready for training. |
| `summarize_dataset` | function | Audit a :class:`Dataset` for obvious data-quality problems. |
| `infer_task` | function | Decide ``'classification'`` or ``'regression'`` from a target column. |

## `pharmaml.chem`

*Cheminformatics: converters, descriptors, similarity*

| Symbol | Kind | Summary |
|---|---|---|
| `ic50_to_pic50` | function | Convert an IC50 measurement to pIC50 = -log10(IC50 [M]). |
| `ki_to_pki` | function | Convert an inhibition constant to pKi = -log10(Ki [M]). |
| `add_pic50_column` | function | Append a pIC50 column to a table of measurements. |
| `split_table_into_chunks` | function | Stream a large CSV into fixed-size part files. |
| `compute_descriptors` | function | Compute molecular descriptors for a list of SMILES. |
| `compute_fingerprints` | function | Compute fingerprints for a list of SMILES. |
| `tanimoto_from_bitarrays` | function | Tanimoto coefficient for two equal-length binary arrays. |
| `tanimoto_from_sets` | function | Tanimoto coefficient for two feature sets: |A∩B| / |A∪B|. |

## `pharmaml.ml`

*QSAR machine learning: preprocessing → training → evaluation → persistence*

| Symbol | Kind | Summary |
|---|---|---|
| `MODEL_REGISTRY` | constant |  |
| `list_models` | function | List registry entries with availability information. |
| `build_model` | function | Instantiate an unfitted estimator from the registry. |
| `clean_numeric_frame` | function | Drop non-numeric columns (except the target/keep list). |
| `build_preprocessor` | function | Build the (impute -> scale) preprocessing pipeline. |
| `split_features_target` | function | Separate features from the target column. |
| `make_train_test_split` | function | Stratified train/test split with automatic fallback. |
| `train_model` | function | Train (and optionally grid-search) one registry model. |
| `compare_models` | function | Cross-validated leaderboard over (a subset of) the registry. |
| `evaluate_model` | function | Evaluate a fitted model on held-out data. |
| `save_model` | function | Persist a fitted model plus its metadata bundle. |
| `load_model` | function | Load a model bundle saved by :func:`save_model`. |

## `pharmaml.pharma`

*Pharmaceutical calculations: units, PK, PD, dosage*

| Symbol | Kind | Summary |
|---|---|---|
| `UnitConverter` | class | Dimension-aware unit converter for pharmaceutical quantities. |
| `molar_mass` | function | Compute the molar mass (g/mol) of a chemical formula. |
| `accumulation_ratio` | function | Accumulation ratio for regular intermittent dosing. |
| `auc_trapezoid` | function | Area under the concentration-time curve (trapezoidal rule). |
| `aucm_trapezoid` | function | Area under the first-moment curve (AUMC = ∫ t·C dt, linear trapezoid). |
| `bioavailability` | function | Absolute bioavailability ``F = (AUC_ev·Dose_iv)/(AUC_iv·Dose_ev)``. |
| `clearance` | function | Systemic clearance ``CL = F·Dose/AUC``. |
| `creatinine_clearance` | function | Estimate creatinine clearance (mL/min) with the Cockcroft-Gault |
| `elimination_rate_constant` | function | Elimination rate constant ``ke = ln(2)/t½`` (1/hours). |
| `half_life` | function | Elimination half-life ``t½ = ln(2)/ke`` (hours). |
| `iv_bolus_concentration` | function | Concentration after an IV bolus: ``C(t) = C0 · e^(−ke·t)``. |
| `mrt` | function | Mean residence time ``MRT = AUMC/AUC`` (non-compartmental). |
| `oral_cmax` | function | Peak concentration after a single oral dose (1-compartment). |
| `oral_tmax` | function | Time of peak concentration for 1-compartment oral dosing. |
| `volume_of_distribution` | function | Apparent volume of distribution ``Vd = CL/ke``. |
| `emax_effect` | function | Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``. |
| `inhibitory_emax_effect` | function | Inhibitory (inverse) Emax model: ``E = E0 − Imax·C^h/(IC50^h + C^h)``. |
| `sigmoid_emax_effect` | function | Sigmoid (Hill) Emax model: ``E = E0 + Emax·C^h/(EC50^h + C^h)``. |
| `bsa_mosteller` | function | Body surface area (m²) by the Mosteller formula. |
| `dose_by_bsa` | function | BSA-scaled dose (Mosteller BSA) with an optional ceiling. |
| `dose_by_weight` | function | Weight-scaled dose (mg/kg regime) with an optional ceiling. |
| `loading_dose` | function | Loading dose to reach target concentration quickly. |
| `maintenance_dose` | function | Maintenance dose per dosing interval. |

## `pharmaml.docking`

*Molecular docking: preparation, workflow, parsing, scoring*

| Symbol | Kind | Summary |
|---|---|---|
| `GridBox` | class | Docking search box geometry. |
| `compute_grid_box` | function | Compute a grid box enclosing a reference ligand plus padding. |
| `write_gpf` | function | Write an AutoDock4 grid parameter file (GPF) for one ligand. |
| `run_autodock4` | function | Run autogrid4 + autodock4 for one ligand directory. |
| `run_vina` | function | Run a single AutoDock Vina docking. |
| `batch_run_directories` | function | Dock every prepared ligand sub-directory in parallel. |
| `parse_dlg_cluster_table` | function | Parse the ``CLUSTERING HISTOGRAM`` table of an AutoDock4 DLG. |
| `parse_dlg_summary` | function | Summarise one DLG: best cluster, most populated cluster, extremes. |
| `parse_vina_output` | function | Parse the mode table of a Vina output (stdout log or PDBQT REMARKs). |
| `scan_docking_directory` | function | Build the docking summary table for every sub-directory. |
| `ligand_efficiency` | function | Ligand efficiency ``LE = |ΔG| / N_heavy`` (kcal/mol per atom). |
| `consensus_rescore` | function | Combine several docking evidence columns into a 0–1 consensus. |

## `pharmaml.viz`

*Visual output for every domain: figures, tables, HTML reports*

| Symbol | Kind | Summary |
|---|---|---|
| `PALETTE` | constant |  |
| `PLOT_FORMATS` | constant |  |
| `close_figure` | function | Close a figure defensively (no-op when matplotlib is absent). |
| `figure_to_base64` | function | Render a figure to a ``data:`` URI for embedding in HTML reports. |
| `get_pyplot` | function | Return ``matplotlib.pyplot`` with a safe backend already selected. |
| `new_figure` | function | Create a styled ``(figure, axes)`` pair (grid + soft spines). |
| `require_matplotlib` | function | Import matplotlib lazily and pick the right backend. |
| `save_figure` | function | Save a figure to ``png``/``svg``/``pdf`` (chosen by extension). |
| `plot_class_distribution` | function | Bar chart of class balance (accepts a mapping or a ``Dataset``). |
| `plot_confusion_matrix` | function | Heat-map of predicted-vs-true class counts. |
| `plot_cv_scores` | function | Per-fold CV scores as bars with the mean line and ±std band. |
| `plot_feature_importance` | function | Top-N feature importances as horizontal bars. |
| `plot_model_comparison` | function | Leaderboard as horizontal bars with per-fold std error bars. |
| `plot_predicted_vs_actual` | function | Parity scatter for regression, with the identity reference line. |
| `plot_residuals` | function | Residuals vs predicted (left) and residual histogram (right). |
| `plot_roc_curves` | function | Receiver-operating curves — binary or one-vs-rest multiclass. |
| `plot_concentration_profile` | function | Plot one PK profile: linear + semi-log panels with AUC shading. |
| `plot_dose_accumulation` | function | One-compartment superposition of an intermittent IV regimen. |
| `plot_descriptor_distributions` | function | Histogram grid over the numeric columns of a table. |
| `plot_similarity_heatmap` | function | Heat-map of a pairwise similarity matrix. |
| `plot_energy_distribution` | function | Histogram of per-ligand best binding energies (kcal/mol). |
| `plot_top_ligands` | function | Horizontal bars of the strongest-binding ligands. |
| `HTMLReport` | class | Builder for a self-contained HTML report. |
| `TABLE_FORMATS` | constant |  |
| `export_table` | function | Write a DataFrame in a *visual* format chosen by the extension. |
| `metrics_to_dataframe` | function | Any report object as a tidy ``metric | value`` DataFrame. |
| `save_visual_bundle` | function | Render figures + tables + metrics into one directory + HTML report. |

## `pharmaml.utils.io`

*Safe tabular I/O helpers*

| Symbol | Kind | Summary |
|---|---|---|
| `read_table` | function | Read a CSV/TSV or Excel file into a DataFrame. |
| `write_table` | function | Write a DataFrame to CSV or Excel honouring the extension. |
| `ensure_directory` | function | Create ``path`` (and parents) if missing; return the resolved Path. |
| `safe_filename` | function | Sanitise a string into a portable filename. |

## `pharmaml.utils.validation`

*Shared input-validation layer*

| Symbol | Kind | Summary |
|---|---|---|
| `validate_positive` | function | Assert ``value`` is a finite, (strictly) positive number. |
| `validate_smiles_syntax` | function | Return True if ``smiles`` passes lightweight syntax checks. |
| `require_columns` | function | Assert that every name in ``columns`` exists in ``frame``. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `coerce_frame` | function | Coerce common inputs (DataFrame / dict of arrays) to a DataFrame. |

## Command-line interface

```text
pharmaml models | data inspect | ml train | ml compare | chem pic50
             | chem descriptors | pharma pk | pharma dose | docking scan
             | docking check
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
