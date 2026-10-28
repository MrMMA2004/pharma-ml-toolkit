# Module index

Auto-derived from each module's `__all__` exports — the authoritative quick
reference for the public API. Re-run `python scratch/build_docs_index.py`
after adding new public functions.
## `pharmaml`

*PharmaML — a production-ready toolkit for computational pharmacy.*

| Symbol | Kind | Summary |
|---|---|---|
| `DataValidationError` | class | Raised when input data fails validation (bad columns, NaN-only |
| `DependencyError` | class | Raised when an optional dependency is required but not installed. |
| `DockingError` | class | Raised when a docking preparation or execution step fails. |
| `DockingParseError` | class | Raised when a docking log/output file cannot be parsed. |
| `PharmaMLError` | class | Base class for all intentional PharmaML errors. |
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
| `infer_task` | function | Decide ``'classification'`` or ``'regression'`` from a target column. |
| `load_dataset` | function | Load **your own** data file and split it ready for training. |
| `split_table` | function | Split a dataset table into train / (optional val) / test parts. |
| `summarize_dataset` | function | Audit a :class:`Dataset` for obvious data-quality problems. |

## `pharmaml.chem`

*Cheminformatics helpers: conversion, descriptors, the multi-family*

| Symbol | Kind | Summary |
|---|---|---|
| `ATOMPAIR_BITS` | constant |  |
| `AVALON_BITS` | constant |  |
| `D3D_COLUMNS` | constant |  |
| `D3D_DESCRIPTOR_COUNT` | constant |  |
| `ECFP3_BITS` | constant |  |
| `ECFP3_RADIUS` | constant |  |
| `ECFP_BITS` | constant |  |
| `ERG_BITS` | constant |  |
| `ESTATE_BITS` | constant |  |
| `FAMILY_PREFIXES` | constant |  |
| `FCFP_BITS` | constant |  |
| `FEATURE_FAMILIES` | constant |  |
| `LAYERED_BITS` | constant |  |
| `MACCS_BITS` | constant |  |
| `MORGAN_BITS` | constant |  |
| `MORGAN_RADIUS` | constant |  |
| `PATTERN_BITS` | constant |  |
| `RDKIT_FP_BITS` | constant |  |
| `RISM_DESCRIPTORS` | constant |  |
| `RISM_DESCRIPTOR_COUNT` | constant |  |
| `TOPOLOGICAL_BITS` | constant |  |
| `UNSUPPORTED_METALS` | constant |  |
| `AtomEnvironment` | class | One ``(atom, radius)`` hit a Morgan bit fires on. |
| `EndpointDatasetReport` | class | Audit record for one :func:`prepare_pic50_dataset` run. |
| `FeatureReport` | class | Audit record for one :func:`compute_feature_frame` run. |
| `FeatureSelection` | class | Result of :func:`select_top_features`. |
| `MorganBitInfo` | class | Decode record for one Morgan fingerprint bit on one molecule. |
| `PdbLigandReport` | class | Audit record for one :func:`smiles_to_pdb` conversion. |
| `SdfWriteReport` | class | Audit of one :func:`table_to_sdf` conversion. |
| `TableComparison` | class | Set reconciliation of one ligand column across two tables. |
| `VarianceFilterResult` | class | Result of :func:`filter_low_variance`. |
| `add_ic50_column` | function | Append an IC50 column converted from a pIC50 column. |
| `add_pic50_column` | function | Append a pIC50 column to a table of measurements. |
| `annotate_with_smiles` | function | Annotate ligand-name columns with SMILES from a reference table. |
| `canonicalize_tautomer` | function | Return the canonical-tautomer SMILES of ``smiles``. |
| `compare_tables` | function | Compare one ligand-name column across two tables. |
| `compute_3d_descriptors` | function | Compute RDKit 3D shape descriptors for a list of SMILES. |
| `compute_descriptors` | function | Compute molecular descriptors for a list of SMILES. |
| `compute_feature_frame` | function | Compute a multi-family feature matrix for a batch of SMILES. |
| `compute_fingerprints` | function | Compute fingerprints for a list of SMILES. |
| `compute_maccs` | function | Compute the 167-bit MACCS key fingerprints. |
| `compute_rism_descriptors` | function | Compute 3D-RISM-KH-inspired solvation descriptors for SMILES. |
| `decode_morgan_bit` | function | Decode one Morgan fingerprint bit into its atom environments. |
| `family_of_column` | function | Return the feature family a frame column belongs to. |
| `filter_low_variance` | function | Cut feature columns whose variance is below ``threshold``. |
| `ic50_from_pic50_series` | function | Vectorised pIC50 -> IC50 conversion preserving NaN for bad rows. |
| `ic50_to_pic50` | function | Convert an IC50 measurement to pIC50 = -log10(IC50 [M]). |
| `ki_to_pki` | function | Convert an inhibition constant to pKi = -log10(Ki [M]). |
| `merge_table_folder` | function | Merge every table of one extension directly inside ``directory``. |
| `merge_tables` | function | Row-wise merge of several tables, ignoring the index. |
| `pic50_to_ic50` | function | Convert a pIC50 value back to an IC50 concentration. |
| `prepare_pic50_dataset` | function | Turn a raw SMILES + EC50 table into a modelling-ready pEC50 set. |
| `rank_features` | function | Rank feature columns by importance, descending. |
| `sdf_to_table` | function | Read an SD file back into a table (SMILES, tags and titles). |
| `select_top_features` | function | Select the top-K features with the rar's two-layer strategy. |
| `selection_metadata` | function | Build the JSON-ready metadata dict for a feature-selection run. |
| `signed_feature_correlations` | function | Signed correlation of every numeric column with ``target``. |
| `similarity_scan` | function | Scan reference ligands against a target library (Tanimoto >= threshold). |
| `smi_to_csv` | function | Stream a ``.smi`` file to CSV one line at a time. |
| `smiles_to_pdb` | function | Generate an optimised 3D PDB ligand from a SMILES string. |
| `split_table_into_chunks` | function | Stream a large CSV into fixed-size part files. |
| `table_to_sdf` | function | Write a SMILES table to an SD file with 3D structures. |
| `table_to_smi` | function | Write a SMILES table to a ``.smi`` list file. |
| `tanimoto_from_bitarrays` | function | Tanimoto coefficient for two equal-length binary arrays. |
| `tanimoto_from_sets` | function | Tanimoto coefficient for two feature sets: |A∩B| / |A∪B|. |
| `tanimoto_matrix` | function | All-vs-all Tanimoto similarities between two bit-matrix batches. |
| `top_matches` | function | Return the ``k`` library entries most similar to ``query``. |

## `pharmaml.ml`

*QSAR machine learning: preprocessing, training, evaluation, persistence,*

| Symbol | Kind | Summary |
|---|---|---|
| `GMM_COVARIANCE_TYPES` | constant |  |
| `KDE_KERNELS` | constant |  |
| `MANIFOLD_METHODS` | constant |  |
| `MODEL_ALIASES` | constant |  |
| `MODEL_REGISTRY` | constant |  |
| `NO_SCALE_MODELS` | constant |  |
| `SHAP_EXTRA` | constant |  |
| `SHAP_REQUIREMENT` | constant |  |
| `ApplicabilityDomainResult` | class | Outcome of a full applicability-domain analysis. |
| `BootstrapUncertainty` | class | Bootstrap-ensemble uncertainty of a fitted estimator. |
| `DeepMLPClassifier` | class | Deep feed-forward neural-network classifier (PyTorch). |
| `DeepMLPRegressor` | class | Deep feed-forward neural-network regressor (PyTorch). |
| `EmbeddingRecord` | class | Typed report of :func:`manifold_embedding`. |
| `GmmResult` | class | Typed report of :func:`gmm_cluster`. |
| `GroupDifference` | class | Cohort mean-SHAP differences with a bootstrap interval (100x-160). |
| `HierarchicalSVR` | class | Hierarchical Support Vector Regression (global + per-cluster SVRs). |
| `HyperParameters` | class | The keras-tuner container of hyperparameter definitions. |
| `Hyperband` | class | Hyperband successive halving over a resource hyperparameter. |
| `HyperparameterImportance` | class | Per-hyperparameter importances of a finished search. |
| `KMeansResult` | class | Typed report of :func:`kmeans_cluster`. |
| `KdeResult` | class | Typed report of :func:`kde_fit`. |
| `MedianPruner` | function | The median pruner: ``PercentilePruner(50)``. |
| `ModelInputReport` | class | Verdict of :func:`validate_bundle_input` — the round-trip audit. |
| `OOFStackingResult` | class | Outcome of one out-of-fold stacking run. |
| `PcaResult` | class | Typed report of :func:`pca_reduce`. |
| `PercentilePruner` | class | Prune trials whose running CV mean falls below a percentile. |
| `PermutationImportance` | class | Permutation feature importances on held-out data. |
| `QuantileIntervals` | class | Prediction intervals from gradient-boosted quantile regression. |
| `RandomSearch` | class | Random search over the hypermodel's declared space. |
| `ResidualMLPClassifier` | class | Residual (skip-connection) deep-network classifier (PyTorch). |
| `ResidualMLPRegressor` | class | Residual (skip-connection) deep-network regressor (PyTorch). |
| `ShapInteractions` | class | Pairwise SHAP interaction values of a fitted tree model (100x-160). |
| `ShapSummary` | class | SHAP values of a fitted tree model over a feature frame. |
| `TPESearch` | class | Tree-structured Parzen Estimator search — Optuna's flagship sampler. |
| `TrialResult` | class | One evaluated trial. |
| `TunerResults` | class | The outcome of one :meth:`RandomSearch.search` / Hyperband run. |
| `applicability_domain` | function | The uploaded workflow end-to-end, one call. |
| `benchmark_feature_families` | function | Cross-validated leaderboard ACROSS feature families, one model fixed. |
| `bootstrap_uncertainty` | function | Bootstrap-ensemble prediction uncertainty (the upload's step 6). |
| `build_ann_classifier` | function | sklearn MLP classifier hypermodel (the legacy ANN upgrade). |
| `build_ann_regressor` | function | sklearn MLP regressor hypermodel (the legacy ANN upgrade). |
| `build_deep_classifier` | function | Deep MLP classifier hypermodel (PyTorch, ``batchnorm`` searchable). |
| `build_deep_regressor` | function | Deep MLP regressor hypermodel (PyTorch, ``batchnorm`` searchable). |
| `build_hsvr` | function | :class:`~pharmaml.ml.hierarchical.HierarchicalSVR` hypermodel. |
| `build_model` | function | Instantiate an unfitted estimator from the registry. |
| `build_preprocessor` | function | Build the (impute -> scale) preprocessing pipeline. |
| `build_stacking` | function | Build a stacking ensemble (upgrades Stacking.py / Stacking reg.py). |
| `build_voting` | function | Build a voting ensemble (upgrades Voting.py / Voting reg.py). |
| `clean_numeric_frame` | function | Drop non-numeric columns (except the target/keep list). |
| `compare_models` | function | Cross-validated leaderboard over (a subset of) the registry. |
| `cross_model_importances` | function | Permutation importances for a WHOLE panel of fitted models. |
| `evaluate_model` | function | Evaluate a fitted model on held-out data. |
| `gmm_cluster` | function | Gaussian mixture model (PDSH 05.12) with the AIC/BIC sweep. |
| `group_difference` | function | Cohort difference of a SHAP summary, with a bootstrap interval. |
| `has_categorical_columns` | function | Return True when ``X`` is a DataFrame with non-numeric columns. |
| `kde_fit` | function | Kernel density estimation (PDSH 05.13) for ONE numeric column. |
| `kmeans_cluster` | function | k-means clustering (PDSH 05.11), with the elbow AND the silhouette. |
| `knn_applicability_domain` | function | kNN-distance applicability domain (the upload's step 4-5). |
| `list_models` | function | List registry entries with availability information. |
| `load_model` | function | Load a model bundle saved by :func:`save_model`. |
| `make_cv_splitter` | function | The tuners' fold protocol, as a public helper. |
| `make_train_test_split` | function | Stratified train/test split with automatic fallback. |
| `manifold_embedding` | function | Project a feature table onto 2-D with a manifold learner (PDSH 05.10). |
| `param_importances` | function | Random-forest (MDI) hyperparameter importances of a finished search. |
| `pca_reduce` | function | Principal component analysis (PDSH 05.09). |
| `permutation_feature_importance` | function | Permutation importances of a fitted model on held-out data. |
| `predict_with_uncertainty` | function | Predictive mean and per-row standard deviation of a GP model. |
| `quantile_prediction_intervals` | function | Central-``1 - alpha`` prediction intervals via quantile regression. |
| `resolve_model_name` | function | Normalise a user-supplied model name to a registry key. |
| `resource_bounds` | function | The declared ``[min, max]`` grid of a hypermodel's resource slot. |
| `save_model` | function | Persist a fitted model plus its metadata bundle. |
| `select_kbest_union` | function | Select the UNION of per-method top-k columns (upgrades look.ipynb). |
| `shap_interactions` | function | Pairwise SHAP interaction values of a fitted tree model over ``X``. |
| `shap_summary` | function | SHAP values of a fitted tree model over ``X``. |
| `split_features_target` | function | Separate features from the target column. |
| `stack_oof_regressors` | function | Stack registry regressors out-of-fold and rank the blenders. |
| `train_model` | function | Train (and optionally grid-search) one registry model. |
| `validate_bundle_input` | function | Audit a new dataset against a saved model bundle before inference. |

## `pharmaml.generative`

*Molecule generation from 0 to 100: objectives, genetic algorithm, RL.*

| Symbol | Kind | Summary |
|---|---|---|
| `DEFAULT_CORPUS` | constant |  |
| `GenerationResult` | class | Ranked molecules from one generative optimisation run. |
| `GeneticOptimizer` | class | Tournament-selection genetic algorithm over canonical SMILES. |
| `LogPBandObjective` | class | Gaussian band on Crippen LogP (wild octanol/water behaviour excluded). |
| `MolecularWeightBandObjective` | class | Gaussian band on average molecular weight (Da).  See |
| `MultiObjective` | class | Weighted geometric mean of several objectives, with veto semantics. |
| `NoveltyObjective` | class | Structural novelty against a reference corpus. |
| `ParetoFront` | class | The non-dominated set of one generation run (audit-ready). |
| `QedObjective` | class | RDKit QED (quantitative estimate of drug-likeness). |
| `ReinventOptimizer` | class | Policy-gradient fine-tuning of a SMILES RNN prior. |
| `ScoringObjective` | class | Base class for generation objectives: ``score(smiles) -> [0, 1]``. |
| `SimilarityObjective` | class | Tanimoto similarity (Morgan fingerprints) to one seed molecule. |
| `SmilesRnn` | class | Character-level GRU SMILES generator (CPU-only, torch optional). |
| `SubstructureObjective` | class | Hard pharmacophore constraint: 1.0 when ``smarts`` matches. |
| `TpsaBandObjective` | class | Gaussian band on topological polar surface area (Å²).  See |
| `bioisostere_moves` | function | Every distinct single-atom bioisostere mutant of ``smiles`` (sorted). |
| `bioisostere_mutate` | function | Draw one single-atom bioisostere mutant of ``smiles`` (or ``None``). |
| `pareto_front` | function | Extract the non-dominated molecules (Pareto front) of a run. |
| `ring_topology_moves` | function | Every distinct one-step ring-size mutant of ``smiles`` (sorted). |
| `ring_topology_mutate` | function | Draw one ring-size mutant of ``smiles`` with ``rng`` (or ``None``). |
| `run_genetic_optimization` | function | One-call genetic optimisation (thin :class:`GeneticOptimizer` wrapper). |
| `run_rl_optimization` | function | One-call RL optimisation (trains a prior from the corpus if needed). |
| `scaffold_hop_crossover` | function | BREED-style crossover: homologous cuts, hybrid child. |

## `pharmaml.biology`

*Cell-line biology: drug-response screens, omics and biomarkers.*

| Symbol | Kind | Summary |
|---|---|---|
| `ResponseScreen` | class | One loaded drug-response screen. |
| `audit_response_table` | function | Audit a response screen for the classic hygiene issues. |
| `call_sensitive_resistant` | function | Threshold a response vector into sensitivity calls. |
| `collapse_replicates` | function | Collapse duplicated (cell line, compound) pairs by the median. |
| `correlate_with_response` | function | Correlate every gene's expression with one drug's response. |
| `direction_of_association` | function | Translate a correlation sign into a sensitivity sentence. |
| `impute_expression` | function | Fill missing expression values with a per-gene statistic. |
| `ln_ic50_to_pic50` | function | Convert GDSC ``LN_IC50`` values to pIC50. |
| `load_response_table` | function | Load a drug-response screen from a DataFrame or csv/xlsx path. |
| `most_variable_genes` | function | Return the ``n`` most variable genes by MAD. |
| `normalize_expression` | function | Normalise an expression table: ``log2(x + 1)`` then per-gene z-score. |
| `pic50_to_ln_ic50` | function | Convert pIC50 values back to GDSC ``LN_IC50`` (the exact inverse). |
| `replicate_variability` | function | Report the replicate spread of every duplicated pair. |
| `response_matrix` | function | Pivot one metric into a cell-line x compound sensitivity matrix. |
| `sensitivity_associations` | function | Correlate every gene with one drug's response, FDR-controlled. |
| `summarize_compounds` | function | Per-compound potency statistics across the cell-line panel. |
| `top_biomarkers` | function | Trim an association table to its strongest hits. |

## `pharmaml.pharma`

*Pharmaceutical calculations: units, PK, PD, dosage design, regulatory-approval filtering.*

| Symbol | Kind | Summary |
|---|---|---|
| `PKSummary` | class | Non-compartmental PK summary for one concentration-time profile. |
| `UnitConverter` | class | Dimension-aware unit converter for pharmaceutical quantities. |
| `accumulation_ratio` | function | Accumulation ratio for regular intermittent dosing. |
| `auc_trapezoid` | function | Area under the concentration-time curve (trapezoidal rule). |
| `aucm_trapezoid` | function | Area under the first-moment curve (AUMC = ∫ t·C dt, linear trapezoid). |
| `bioavailability` | function | Absolute bioavailability ``F = (AUC_ev·Dose_iv)/(AUC_iv·Dose_ev)``. |
| `bsa_mosteller` | function | Body surface area (m²) by the Mosteller formula. |
| `clarks_rule` | function | Clark's rule paediatric scaling by body weight in pounds. |
| `clearance` | function | Systemic clearance ``CL = F·Dose/AUC``. |
| `creatinine_clearance` | function | Estimate creatinine clearance (mL/min) with the Cockcroft-Gault |
| `dose_by_bsa` | function | BSA-scaled dose (Mosteller BSA) with an optional ceiling. |
| `dose_by_weight` | function | Weight-scaled dose (mg/kg regime) with an optional ceiling. |
| `ec50_for_target_effect` | function | Inverse Emax: concentration reaching a target effect. |
| `effect_table` | function | Evaluate the sigmoid Emax model over a concentration grid. |
| `elimination_rate_constant` | function | Elimination rate constant ``ke = ln(2)/t½`` (1/hours). |
| `emax_effect` | function | Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``. |
| `filter_approved_drugs` | function | Collect the drug names whose regulatory approval matches a status. |
| `half_life` | function | Elimination half-life ``t½ = ln(2)/ke`` (hours). |
| `infusion_rate` | function | Constant IV infusion rate achieving a steady state: ``R = CL·Css``. |
| `inhibitory_emax_effect` | function | Inhibitory (inverse) Emax model: ``E = E0 − Imax·C^h/(IC50^h + C^h)``. |
| `iv_bolus_concentration` | function | Concentration after an IV bolus: ``C(t) = C0 · e^(−ke·t)``. |
| `iv_infusion_steady_state` | function | Steady-state concentration for a constant IV infusion ``Css = R/CL``. |
| `linear_effect` | function | Linear PD model: ``E = intercept + slope·C`` (small-dose region). |
| `loading_dose` | function | Loading dose to reach target concentration quickly. |
| `maintenance_dose` | function | Maintenance dose per dosing interval. |
| `molar_mass` | function | Compute the molar mass (g/mol) of a chemical formula. |
| `mrt` | function | Mean residence time ``MRT = AUMC/AUC`` (non-compartmental). |
| `nca_summary` | function | Full non-compartmental analysis of one profile. |
| `oral_cmax` | function | Peak concentration after a single oral dose (1-compartment). |
| `oral_tmax` | function | Time of peak concentration for 1-compartment oral dosing. |
| `renal_dose_adjustment` | function | Linearly scale a dose with renal function (creatinine clearance). |
| `sigmoid_emax_effect` | function | Sigmoid (Hill) Emax model: ``E = E0 + Emax·C^h/(EC50^h + C^h)``. |
| `tdm_trough_adjustment` | function | Proportionally adjust a dose from a measured trough level. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `vd_from_dose_auc` | function | Dose/AUC ratio — and, with ``mrt``, the steady-state volume. |
| `volume_of_distribution` | function | Apparent volume of distribution ``Vd = CL/ke``. |
| `youngs_rule` | function | Young's rule for paediatric dose scaling (age 1–12). |

## `pharmaml.docking`

*Molecular-docking workflows: preparation, execution, parsing, scoring*

| Symbol | Kind | Summary |
|---|---|---|
| `DEFAULT_MGL_PYTHON` | constant |  |
| `DEFAULT_PREPARE_LIGAND4` | constant |  |
| `UNSUPPORTED_LIGAND_ATOMS` | constant |  |
| `GridBox` | class | Docking search box geometry. |
| `GridParameters` | class | Parsed AutoDock4 grid parameter file (GPF) content. |
| `LigandLibrary` | class | A validated ligand library loaded from a table file. |
| `LigandPreparationReport` | class | Audit of one :func:`prepare_ligand_library` batch. |
| `ProteinReport` | class | Audit of one :func:`prepare_protein` run. |
| `TorsionCheck` | class | Torsion-consistency audit of one prepared ligand PDBQT. |
| `VinaBatchProgress` | class | Pre-flight resume census for :func:`run_vina_batch`. |
| `batch_run_directories` | function | Dock every prepared ligand sub-directory in parallel. |
| `check_docking_completion` | function | Audit a docking tree for missing/failed runs (upgrades checker.py). |
| `check_torsion_consistency` | function | Compare TORSDOF, active REMARKs and BRANCH counts of a PDBQT. |
| `cleanup_tree` | function | Delete every file not on the allowlist, bottom-up. |
| `collect_files` | function | Collect matching files from a campaign tree into one folder. |
| `compute_grid_box` | function | Compute a grid box enclosing a reference ligand plus padding. |
| `consensus_breakdown` | function | Explain a consensus ranking: per-column units plus the score. |
| `consensus_rescore` | function | Combine several docking evidence columns into a 0–1 consensus. |
| `convert_pdb_folder_to_pdbqt` | function | Batch-convert a folder of PDB files to PDBQT via MGLTools. |
| `delete_named_files` | function | Delete every file named ``filename`` anywhere under ``directory``. |
| `distribute_files` | function | Copy each file into every immediate sub-directory of ``target_dir``. |
| `distribute_receptor` | function | Copy a prepared receptor into every ligand folder of a campaign. |
| `dock_smiles_array` | function | Dock every SMILES of an array against one receptor, simultaneously. |
| `extract_pdbqt_atom_types` | function | Collect sorted unique AutoDock atom types from a PDBQT ligand. |
| `generate_ligand_structures` | function | Generate 3D ligand structures (PDB + MOL2) from a SMILES library. |
| `ligand_efficiency` | function | Ligand efficiency ``LE = |ΔG| / N_heavy`` (kcal/mol per atom). |
| `ligand_efficiency_series` | function | Vectorised ligand-efficiency column for a docking summary table. |
| `list_file_names` | function | List the stems of all files with ``extension`` in a directory. |
| `load_ligand_csv` | function | Load a ligand library from CSV/TSV (or Excel). |
| `parse_dlg_cluster_table` | function | Parse the ``CLUSTERING HISTOGRAM`` table of an AutoDock4 DLG. |
| `parse_dlg_summary` | function | Summarise one DLG: best cluster, most populated cluster, extremes. |
| `parse_vina_config` | function | Parse an AutoDock Vina config file into a ``key -> value`` dict. |
| `parse_vina_output` | function | Parse the mode table of a Vina output (stdout log or PDBQT REMARKs). |
| `prepare_ligand_library` | function | Prepare every ligand of a CSV library into a campaign tree. |
| `prepare_protein` | function | Import a protein and prepare a docking-ready receptor PDBQT. |
| `read_gpf` | function | Parse an AutoDock4 grid parameter file (GPF) into a dict. |
| `reindex_by_number` | function | Reindex a frame onto the full ligand-number range. |
| `run_autodock4` | function | Run autogrid4 + autodock4 for one ligand directory. |
| `run_vina` | function | Run a single AutoDock Vina docking. |
| `run_vina_batch` | function | Dock every ``.pdbqt`` under a directory with one Vina config. |
| `scan_docking_directory` | function | Build the docking summary table for every sub-directory. |
| `scan_gpf_ligand_types` | function | Report the ``ligand_types`` of every sub-directory's ``r.gpf``. |
| `scan_torsion_consistency` | function | Audit every ``.pdbqt`` under a tree for torsion consistency. |
| `scan_vina_directory` | function | Summarise every Vina ``.log`` under a results tree. |
| `smiles_to_pdbqt` | function | Convert one SMILES to a 3D PDBQT ligand (Meeko + RDKit). |
| `vina_batch_progress` | function | Count what a resumed Vina batch would skip and what it would dock. |
| `write_config_batch` | function | Write the ``r.gpf`` + ``r.dpf`` pair into every ligand folder. |
| `write_dpf` | function | Write an AutoDock4 docking parameter file (DPF) for one ligand. |
| `write_gpf` | function | Write an AutoDock4 grid parameter file (GPF) for one ligand. |

## `pharmaml.viz`

*Visual output for every PharmaML domain — plots, tables, HTML reports.*

| Symbol | Kind | Summary |
|---|---|---|
| `PALETTE` | constant |  |
| `PDF_THEMES` | constant |  |
| `PLOT_FORMATS` | constant |  |
| `REPORT_THEMES` | constant |  |
| `TABLE_FORMATS` | constant |  |
| `HTMLReport` | class | Builder for a self-contained HTML report. |
| `close_figure` | function | Close a figure defensively (no-op when matplotlib is absent). |
| `export_docx` | function | Write a DataFrame as a Word ``.docx`` document with a grid table. |
| `export_table` | function | Write a DataFrame in a *visual* format chosen by the extension. |
| `figure_to_base64` | function | Render a figure to a ``data:`` URI for embedding in HTML reports. |
| `get_pyplot` | function | Return ``matplotlib.pyplot`` with a safe backend already selected. |
| `metrics_to_dataframe` | function | Any report object as a tidy ``metric | value`` DataFrame. |
| `new_figure` | function | Create a styled ``(figure, axes)`` pair (grid + soft spines). |
| `plot_applicability_domain` | function | The uploaded AD workflow's combined two-panel reliability figure. |
| `plot_calibration_curve` | function | Reliability diagram — are the predicted probabilities honest? |
| `plot_class_distribution` | function | Bar chart of class balance (accepts a mapping or a ``Dataset``). |
| `plot_concentration_profile` | function | Plot one PK profile: linear + semi-log panels with AUC shading. |
| `plot_confusion_matrix` | function | Heat-map of predicted-vs-true class counts. |
| `plot_consensus_ranking` | function | Horizontal bars of ligands ranked by the 0–1 consensus. |
| `plot_correlation_heatmap` | function | Feature × feature correlation heatmap (the seaborn-free port). |
| `plot_cv_scores` | function | Per-fold CV scores as bars with the mean line and ±std band. |
| `plot_descriptor_distributions` | function | Histogram grid over the numeric columns of a table. |
| `plot_dose_accumulation` | function | One-compartment superposition of an intermittent IV regimen. |
| `plot_edf` | function | The empirical CDF of the completed trials' scores (100x-160). |
| `plot_energy_distribution` | function | Histogram of per-ligand best binding energies (kcal/mol). |
| `plot_feature_importance` | function | Top-N feature importances as horizontal bars. |
| `plot_gmm_selection` | function | AIC/BIC vs component count -- the handbook's own GMM selector. |
| `plot_gp_uncertainty` | function | Parity scatter with ``± z·σ`` error bars — the GP's honest readout. |
| `plot_hexbin_density` | function | Hexagonal binning -- the honest 2-D density for big scatter plots. |
| `plot_intermediate_values` | function | The per-trial running fold-mean curves (100x-160). |
| `plot_kde_density` | function | The KDE curve over its own bandwidth story. |
| `plot_kmeans_diagnostics` | function | Inertia elbow AND silhouette sweep -- the auditable k choice. |
| `plot_learning_curve` | function | Training-set size vs CV score — is more data still buying accuracy? |
| `plot_manifold_embedding` | function | Scatter an embedding, coloured by a target (the digits-figure pattern). |
| `plot_model_comparison` | function | Leaderboard as horizontal bars with per-fold std error bars. |
| `plot_operator_trajectory` | function | One figure per run: score trajectory + the operator mix per generation. |
| `plot_optimization_history` | function | The objective per trial + the best-so-far step line (100x-160). |
| `plot_parallel_coordinate` | function | One polyline per trial across the params + objective (100x-160). |
| `plot_param_importances` | function | The MDI hyperparameter-importance bars (100x-160). |
| `plot_pareto_front` | function | One scatter per MultiObjective run: dominated points muted, front lit. |
| `plot_pca_variance` | function | Explained-variance ladder -- the "how many components?" answer. |
| `plot_permutation_heatmap` | function | Cross-model permutation-importance heatmap (features × models). |
| `plot_permutation_importance` | function | Top-N permutation importances as signed bars with error bars. |
| `plot_pr_curves` | function | Precision-Recall curves — binary or one-vs-rest multiclass. |
| `plot_predicted_vs_actual` | function | Parity scatter for regression, with the identity reference line. |
| `plot_prediction_intervals` | function | Parity scatter with the quantile-regression prediction band. |
| `plot_receptor_atom_types` | function | Bar chart of the AutoDock atom-type census of a receptor. |
| `plot_residuals` | function | Residuals vs predicted (left) and residual histogram (right). |
| `plot_rl_introspection` | function | One figure per RL run: the score story + the prior-drift diagnostics. |
| `plot_roc_curves` | function | Receiver-operating curves — binary or one-vs-rest multiclass. |
| `plot_shap_beeswarm` | function | The SHAP beeswarm in PURE matplotlib -- ``D2.py``'s GAP-5 figure. |
| `plot_shap_dependence` | function | One feature's value vs its SHAP value, coloured by its partner (100x-160). |
| `plot_shap_force` | function | The additive force plot, static matplotlib twin (100x-160). |
| `plot_shap_group_difference` | function | Cohort mean-SHAP difference bars with bootstrap whiskers (100x-160). |
| `plot_shap_heatmap` | function | The instances x features SHAP matrix, clustered (100x-160). |
| `plot_shap_importance` | function | Top-N mean-|SHAP| horizontal bars -- ``D2.py``'s own top-20 bar. |
| `plot_shap_waterfall` | function | The LOCAL explanation: the base-value -> prediction walk (100x-160). |
| `plot_signed_correlations` | function | Top-N signed feature-target correlations as a horizontal bar chart. |
| `plot_similarity_heatmap` | function | Heat-map of a pairwise similarity matrix. |
| `plot_slice` | function | One param-vs-objective panel per hyperparameter (100x-160). |
| `plot_table_comparison` | function | Overlap breakdown of a ligand-table :class:`TableComparison`. |
| `plot_timeline` | function | Per-trial duration bars coloured by state (100x-160). |
| `plot_top_ligands` | function | Horizontal bars of the strongest-binding ligands. |
| `plot_training_history` | function | Epoch-by-epoch loss curves with the early-stopping verdict. |
| `require_matplotlib` | function | Import matplotlib lazily and pick the right backend. |
| `save_figure` | function | Save a figure to ``png``/``svg``/``pdf`` (chosen by extension). |
| `save_pdf_report` | function | Write a multi-page PDF document report (metrics, tables, figures). |
| `save_visual_bundle` | function | Render figures + tables + metrics into one directory + HTML report. |

## `pharmaml.utils.io`

*Robust tabular and filesystem I/O.*

| Symbol | Kind | Summary |
|---|---|---|
| `ensure_directory` | function | Create ``path`` (and parents) if missing; return the resolved Path. |
| `read_indexed_table_with_progress` | function | Read an INDEXED table, reporting row progress for large CSVs (100x-146). |
| `read_table` | function | Read a CSV/TSV or Excel file into a DataFrame. |
| `read_table_with_progress` | function | Read a table, reporting row progress for large CSV-like files. |
| `safe_filename` | function | Sanitise a string into a portable filename. |
| `write_table` | function | Write a DataFrame to CSV or Excel honouring the extension. |

## `pharmaml.utils.validation`

*Input validation guards.*

| Symbol | Kind | Summary |
|---|---|---|
| `coerce_frame` | function | Coerce common inputs (DataFrame / dict of arrays) to a DataFrame. |
| `looks_continuous` | function | Heuristic: does ``y`` look like a numeric (regression) target? |
| `require_columns` | function | Assert that every name in ``columns`` exists in ``frame``. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `validate_positive` | function | Assert ``value`` is a finite, (strictly) positive number. |
| `validate_smiles_syntax` | function | Return True if ``smiles`` passes lightweight syntax checks. |

## Command-line interface

```text
pharmaml models | data inspect | data split | data convert | ml train
             | ml compare | chem pic50 | chem descriptors | pharma pk
             | pharma dose | docking prep-ligands | docking prep-protein
             | docking scan | docking check | generative run
             | generative corpus
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
