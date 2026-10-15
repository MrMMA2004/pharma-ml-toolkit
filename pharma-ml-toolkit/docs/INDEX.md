# Module index

Auto-derived from each module's `__all__` exports — the authoritative quick
reference for the public API. Re-run `python scratch/build_docs_index.py`
after adding new public functions.
## `pharmaml`

*PharmaML — a production-ready toolkit for computational pharmacy.*

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

*Bring your own data: import user files (CSV/TSV/TXT/Excel) into PharmaML.*

| Symbol | Kind | Summary |
|---|---|---|
| `TARGET_CANDIDATES` | constant |  |
| `Dataset` | class | A user-supplied dataset split into features and target. |
| `DatasetSummary` | class | Human-readable audit of a loaded dataset. |
| `SplitResult` | class | Outcome of :func:`split_table` — the split frames plus audit info. |
| `load_dataset` | function | Load **your own** data file and split it ready for training. |
| `summarize_dataset` | function | Audit a :class:`Dataset` for obvious data-quality problems. |
| `infer_task` | function | Decide ``'classification'`` or ``'regression'`` from a target column. |
| `split_table` | function | Split a dataset table into train / (optional val) / test parts. |

## `pharmaml.chem`

*Cheminformatics helpers: conversion, descriptors, similarity, tables, standardization.*

| Symbol | Kind | Summary |
|---|---|---|
| `ic50_to_pic50` | function | Convert an IC50 measurement to pIC50 = -log10(IC50 [M]). |
| `ki_to_pki` | function | Convert an inhibition constant to pKi = -log10(Ki [M]). |
| `add_pic50_column` | function | Append a pIC50 column to a table of measurements. |
| `split_table_into_chunks` | function | Stream a large CSV into fixed-size part files. |
| `SdfWriteReport` | class | Audit of one :func:`table_to_sdf` conversion. |
| `smi_to_csv` | function | Stream a ``.smi`` file to CSV one line at a time. |
| `table_to_sdf` | function | Write a SMILES table to an SD file with 3D structures. |
| `sdf_to_table` | function | Read an SD file back into a table (SMILES, tags and titles). |
| `table_to_smi` | function | Write a SMILES table to a ``.smi`` list file. |
| `compute_descriptors` | function | Compute molecular descriptors for a list of SMILES. |
| `compute_fingerprints` | function | Compute fingerprints for a list of SMILES. |
| `compute_maccs` | function | Compute the 167-bit MACCS key fingerprints. |
| `tanimoto_from_bitarrays` | function | Tanimoto coefficient for two equal-length binary arrays. |
| `tanimoto_from_sets` | function | Tanimoto coefficient for two feature sets: |A∩B| / |A∪B|. |
| `tanimoto_matrix` | function | All-vs-all Tanimoto similarities between two bit-matrix batches. |
| `TableComparison` | class | Set reconciliation of one ligand column across two tables. |
| `compare_tables` | function | Compare one ligand-name column across two tables. |
| `annotate_with_smiles` | function | Annotate ligand-name columns with SMILES from a reference table. |
| `merge_tables` | function | Row-wise merge of several tables, ignoring the index. |
| `merge_table_folder` | function | Merge every table of one extension directly inside ``directory``. |
| `UNSUPPORTED_METALS` | constant |  |
| `PdbLigandReport` | class | Audit record for one :func:`smiles_to_pdb` conversion. |
| `canonicalize_tautomer` | function | Return the canonical-tautomer SMILES of ``smiles``. |
| `smiles_to_pdb` | function | Generate an optimised 3D PDB ligand from a SMILES string. |

## `pharmaml.ml`

*QSAR machine learning: preprocessing, training, evaluation, persistence.*

| Symbol | Kind | Summary |
|---|---|---|
| `MODEL_ALIASES` | constant |  |
| `MODEL_REGISTRY` | constant |  |
| `list_models` | function | List registry entries with availability information. |
| `build_model` | function | Instantiate an unfitted estimator from the registry. |
| `resolve_model_name` | function | Normalise a user-supplied model name to a registry key. |
| `DeepMLPClassifier` | class | Deep feed-forward neural-network classifier (PyTorch). |
| `DeepMLPRegressor` | class | Deep feed-forward neural-network regressor (PyTorch). |
| `ResidualMLPClassifier` | class | Residual (skip-connection) deep-network classifier (PyTorch). |
| `ResidualMLPRegressor` | class | Residual (skip-connection) deep-network regressor (PyTorch). |
| `clean_numeric_frame` | function | Drop non-numeric columns (except the target/keep list). |
| `build_preprocessor` | function | Build the (impute -> scale) preprocessing pipeline. |
| `has_categorical_columns` | function | Return True when ``X`` is a DataFrame with non-numeric columns. |
| `split_features_target` | function | Separate features from the target column. |
| `make_train_test_split` | function | Stratified train/test split with automatic fallback. |
| `train_model` | function | Train (and optionally grid-search) one registry model. |
| `compare_models` | function | Cross-validated leaderboard over (a subset of) the registry. |
| `evaluate_model` | function | Evaluate a fitted model on held-out data. |
| `save_model` | function | Persist a fitted model plus its metadata bundle. |
| `load_model` | function | Load a model bundle saved by :func:`save_model`. |

## `pharmaml.pharma`

*Pharmaceutical calculations: units, PK, PD, dosage design, regulatory-approval filtering.*

| Symbol | Kind | Summary |
|---|---|---|
| `UnitConverter` | class | Dimension-aware unit converter for pharmaceutical quantities. |
| `molar_mass` | function | Compute the molar mass (g/mol) of a chemical formula. |
| `PKSummary` | class | Non-compartmental PK summary for one concentration-time profile. |
| `accumulation_ratio` | function | Accumulation ratio for regular intermittent dosing. |
| `auc_trapezoid` | function | Area under the concentration-time curve (trapezoidal rule). |
| `aucm_trapezoid` | function | Area under the first-moment curve (AUMC = ∫ t·C dt, linear trapezoid). |
| `bioavailability` | function | Absolute bioavailability ``F = (AUC_ev·Dose_iv)/(AUC_iv·Dose_ev)``. |
| `clearance` | function | Systemic clearance ``CL = F·Dose/AUC``. |
| `creatinine_clearance` | function | Estimate creatinine clearance (mL/min) with the Cockcroft-Gault |
| `elimination_rate_constant` | function | Elimination rate constant ``ke = ln(2)/t½`` (1/hours). |
| `half_life` | function | Elimination half-life ``t½ = ln(2)/ke`` (hours). |
| `iv_bolus_concentration` | function | Concentration after an IV bolus: ``C(t) = C0 · e^(−ke·t)``. |
| `iv_infusion_steady_state` | function | Steady-state concentration for a constant IV infusion ``Css = R/CL``. |
| `mrt` | function | Mean residence time ``MRT = AUMC/AUC`` (non-compartmental). |
| `nca_summary` | function | Full non-compartmental analysis of one profile. |
| `oral_cmax` | function | Peak concentration after a single oral dose (1-compartment). |
| `oral_tmax` | function | Time of peak concentration for 1-compartment oral dosing. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `vd_from_dose_auc` | function | Dose/AUC ratio — and, with ``mrt``, the steady-state volume. |
| `volume_of_distribution` | function | Apparent volume of distribution ``Vd = CL/ke``. |
| `ec50_for_target_effect` | function | Inverse Emax: concentration reaching a target effect. |
| `effect_table` | function | Evaluate the sigmoid Emax model over a concentration grid. |
| `emax_effect` | function | Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``. |
| `inhibitory_emax_effect` | function | Inhibitory (inverse) Emax model: ``E = E0 − Imax·C^h/(IC50^h + C^h)``. |
| `linear_effect` | function | Linear PD model: ``E = intercept + slope·C`` (small-dose region). |
| `sigmoid_emax_effect` | function | Sigmoid (Hill) Emax model: ``E = E0 + Emax·C^h/(EC50^h + C^h)``. |
| `bsa_mosteller` | function | Body surface area (m²) by the Mosteller formula. |
| `clarks_rule` | function | Clark's rule paediatric scaling by body weight in pounds. |
| `dose_by_bsa` | function | BSA-scaled dose (Mosteller BSA) with an optional ceiling. |
| `dose_by_weight` | function | Weight-scaled dose (mg/kg regime) with an optional ceiling. |
| `infusion_rate` | function | Constant IV infusion rate achieving a steady state: ``R = CL·Css``. |
| `loading_dose` | function | Loading dose to reach target concentration quickly. |
| `maintenance_dose` | function | Maintenance dose per dosing interval. |
| `renal_dose_adjustment` | function | Linearly reduce a dose for impaired renal function. |
| `tdm_trough_adjustment` | function | Proportionally adjust a dose from a measured trough level. |
| `youngs_rule` | function | Young's rule for paediatric dose scaling (age 1–12). |
| `filter_approved_drugs` | function | Collect the drug names whose regulatory approval matches a status. |

## `pharmaml.docking`

*Molecular-docking workflows: preparation, execution, parsing, scoring*

| Symbol | Kind | Summary |
|---|---|---|
| `GridBox` | class | Docking search box geometry. |
| `compute_grid_box` | function | Compute a grid box enclosing a reference ligand plus padding. |
| `extract_pdbqt_atom_types` | function | Collect sorted unique AutoDock atom types from a PDBQT ligand. |
| `write_gpf` | function | Write an AutoDock4 grid parameter file (GPF) for one ligand. |
| `read_gpf` | function | Parse an AutoDock4 grid parameter file (GPF) into a dict. |
| `write_dpf` | function | Write an AutoDock4 docking parameter file (DPF) for one ligand. |
| `smiles_to_pdbqt` | function | Convert one SMILES to a 3D PDBQT ligand (Meeko + RDKit). |
| `LigandLibrary` | class | A validated ligand library loaded from a table file. |
| `LigandPreparationReport` | class | Audit of one :func:`prepare_ligand_library` batch. |
| `load_ligand_csv` | function | Load a ligand library from CSV/TSV (or Excel). |
| `prepare_ligand_library` | function | Prepare every ligand of a CSV library into a campaign tree. |
| `TorsionCheck` | class | Torsion-consistency audit of one prepared ligand PDBQT. |
| `check_torsion_consistency` | function | Compare TORSDOF, active REMARKs and BRANCH counts of a PDBQT. |
| `scan_torsion_consistency` | function | Audit every ``.pdbqt`` under a tree for torsion consistency. |
| `collect_files` | function | Collect matching files from a campaign tree into one folder. |
| `list_file_names` | function | List the stems of all files with ``extension`` in a directory. |
| `distribute_files` | function | Copy each file into every immediate sub-directory of ``target_dir``. |
| `cleanup_tree` | function | Delete every file not on the allowlist, bottom-up. |
| `delete_named_files` | function | Delete every file named ``filename`` anywhere under ``directory``. |
| `reindex_by_number` | function | Reindex a frame onto the full ligand-number range. |
| `scan_gpf_ligand_types` | function | Report the ``ligand_types`` of every sub-directory's ``r.gpf``. |
| `ProteinReport` | class | Audit of one :func:`prepare_protein` run. |
| `prepare_protein` | function | Import a protein and prepare a docking-ready receptor PDBQT. |
| `distribute_receptor` | function | Copy a prepared receptor into every ligand folder of a campaign. |
| `run_autodock4` | function | Run autogrid4 + autodock4 for one ligand directory. |
| `run_vina` | function | Run a single AutoDock Vina docking. |
| `run_vina_batch` | function | Dock every ``.pdbqt`` under a directory with one Vina config. |
| `parse_vina_config` | function | Parse an AutoDock Vina config file into a ``key -> value`` dict. |
| `batch_run_directories` | function | Dock every prepared ligand sub-directory in parallel. |
| `parse_dlg_cluster_table` | function | Parse the ``CLUSTERING HISTOGRAM`` table of an AutoDock4 DLG. |
| `parse_dlg_summary` | function | Summarise one DLG: best cluster, most populated cluster, extremes. |
| `parse_vina_output` | function | Parse the mode table of a Vina output (stdout log or PDBQT REMARKs). |
| `check_docking_completion` | function | Audit a docking tree for missing/failed runs (upgrades checker.py). |
| `scan_docking_directory` | function | Build the docking summary table for every sub-directory. |
| `scan_vina_directory` | function | Summarise every Vina ``.log`` under a results tree. |
| `ligand_efficiency` | function | Ligand efficiency ``LE = |ΔG| / N_heavy`` (kcal/mol per atom). |
| `ligand_efficiency_series` | function | Vectorised ligand-efficiency column for a docking summary table. |
| `consensus_rescore` | function | Combine several docking evidence columns into a 0–1 consensus. |

## `pharmaml.viz`

*Visual output for every PharmaML domain — plots, tables, HTML reports.*

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
| `plot_calibration_curve` | function | Reliability diagram — are the predicted probabilities honest? |
| `plot_class_distribution` | function | Bar chart of class balance (accepts a mapping or a ``Dataset``). |
| `plot_confusion_matrix` | function | Heat-map of predicted-vs-true class counts. |
| `plot_cv_scores` | function | Per-fold CV scores as bars with the mean line and ±std band. |
| `plot_feature_importance` | function | Top-N feature importances as horizontal bars. |
| `plot_learning_curve` | function | Training-set size vs CV score — is more data still buying accuracy? |
| `plot_model_comparison` | function | Leaderboard as horizontal bars with per-fold std error bars. |
| `plot_predicted_vs_actual` | function | Parity scatter for regression, with the identity reference line. |
| `plot_residuals` | function | Residuals vs predicted (left) and residual histogram (right). |
| `plot_roc_curves` | function | Receiver-operating curves — binary or one-vs-rest multiclass. |
| `plot_concentration_profile` | function | Plot one PK profile: linear + semi-log panels with AUC shading. |
| `plot_dose_accumulation` | function | One-compartment superposition of an intermittent IV regimen. |
| `plot_descriptor_distributions` | function | Histogram grid over the numeric columns of a table. |
| `plot_similarity_heatmap` | function | Heat-map of a pairwise similarity matrix. |
| `plot_table_comparison` | function | Overlap breakdown of a ligand-table :class:`TableComparison`. |
| `plot_energy_distribution` | function | Histogram of per-ligand best binding energies (kcal/mol). |
| `plot_receptor_atom_types` | function | Bar chart of the AutoDock atom-type census of a receptor. |
| `plot_top_ligands` | function | Horizontal bars of the strongest-binding ligands. |
| `HTMLReport` | class | Builder for a self-contained HTML report. |
| `TABLE_FORMATS` | constant |  |
| `export_table` | function | Write a DataFrame in a *visual* format chosen by the extension. |
| `metrics_to_dataframe` | function | Any report object as a tidy ``metric | value`` DataFrame. |
| `save_visual_bundle` | function | Render figures + tables + metrics into one directory + HTML report. |

## `pharmaml.utils.io`

*Robust tabular and filesystem I/O.*

| Symbol | Kind | Summary |
|---|---|---|
| `read_table` | function | Read a CSV/TSV or Excel file into a DataFrame. |
| `write_table` | function | Write a DataFrame to CSV or Excel honouring the extension. |
| `ensure_directory` | function | Create ``path`` (and parents) if missing; return the resolved Path. |
| `safe_filename` | function | Sanitise a string into a portable filename. |

## `pharmaml.utils.validation`

*Input validation guards.*

| Symbol | Kind | Summary |
|---|---|---|
| `validate_positive` | function | Assert ``value`` is a finite, (strictly) positive number. |
| `validate_smiles_syntax` | function | Return True if ``smiles`` passes lightweight syntax checks. |
| `require_columns` | function | Assert that every name in ``columns`` exists in ``frame``. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `coerce_frame` | function | Coerce common inputs (DataFrame / dict of arrays) to a DataFrame. |
| `looks_continuous` | function | Heuristic: does ``y`` look like a numeric (regression) target? |

## Command-line interface

```text
pharmaml models | data inspect | data split | data convert | ml train
             | ml compare | chem pic50 | chem descriptors | pharma pk
             | pharma dose | docking prep-ligands | docking prep-protein
             | docking scan | docking check
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
