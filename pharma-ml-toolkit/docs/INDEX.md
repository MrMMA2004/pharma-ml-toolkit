# Module index

Auto-derived from each module's `__all__` exports — the authoritative quick
reference for the public API. Re-run `python tools/build_docs_index.py`
after adding new public functions.
## `pharmaml`

*PharmaML — a production-ready toolkit for computational pharmacy.*

| Symbol | Kind | Summary |
|---|---|---|
| `DataValidationError` | class | Raised when input data fails validation (bad columns, NaN-only |
| `DependencyError` | class | Raised when an optional dependency is required but not installed. |
| `DockingError` | class | Raised when a docking preparation or execution step fails. |
| `DockingParseError` | class | Raised when a docking log/output file cannot be parsed. |
| `MDAnalysisError` | class | Raised for invalid molecular-dynamics inputs: non-3D coordinate |
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
| `CHEMBL_EXCLUDE_METALS` | constant |  |
| `CHEMBL_NORMALIZATION_TRANSFORMS` | constant |  |
| `CHEMBL_SALTS` | constant |  |
| `CHEMBL_SOLVENTS` | constant |  |
| `D3D_COLUMNS` | constant |  |
| `D3D_DESCRIPTOR_COUNT` | constant |  |
| `ECFP3_BITS` | constant |  |
| `ECFP3_RADIUS` | constant |  |
| `ECFP_BITS` | constant |  |
| `ELEMENTS` | constant |  |
| `ELEMENT_COLORS` | constant |  |
| `ERG_BITS` | constant |  |
| `ESTATE_BITS` | constant |  |
| `FAMILY_PREFIXES` | constant |  |
| `FCFP_BITS` | constant |  |
| `FEATURE_FAMILIES` | constant |  |
| `INTERACTION_FAMILIES` | constant |  |
| `LAYERED_BITS` | constant |  |
| `MACCS_BITS` | constant |  |
| `MORGAN_BITS` | constant |  |
| `MORGAN_RADIUS` | constant |  |
| `NEUTRALIZE_PATTERNS` | constant |  |
| `PATTERN_BITS` | constant |  |
| `PROFILES` | constant |  |
| `PROTONATION_SMARTS` | constant |  |
| `RDKIT_FP_BITS` | constant |  |
| `REINVENT_DEFAULT_ELEMENTS` | constant |  |
| `REPRESENTATIONS` | constant |  |
| `RISM_DESCRIPTORS` | constant |  |
| `RISM_DESCRIPTOR_COUNT` | constant |  |
| `SA_SCORE_LIMITATIONS` | constant |  |
| `SA_SCORE_METHOD` | constant |  |
| `TOPOLOGICAL_BITS` | constant |  |
| `UNSUPPORTED_METALS` | constant |  |
| `AtomEnvironment` | class | One ``(atom, radius)`` hit a Morgan bit fires on. |
| `AtomPrimitive` | class | One drawable atom: screen position, depth, radius, colour. |
| `AtomRecord` | class | One ATOM/HETATM record of a PDB file. |
| `BondPrimitive` | class | One drawable bond: both endpoints' screen positions and colours. |
| `Camera` | class | Yaw/pitch/zoom camera -- the ``setRotation``/``zoomTo`` concept. |
| `ElementInfo` | class | Static element data: number, weight, allowed valences, radii. |
| `EndpointDatasetReport` | class | Audit record for one :func:`prepare_pic50_dataset` run. |
| `FeatureReport` | class | Audit record for one :func:`compute_feature_frame` run. |
| `FeatureSelection` | class | Result of :func:`select_top_features`. |
| `GridSelections` | class | Register for grid selections -- mols2grid's ``SelectionRegister``. |
| `InteractionAtom` | class | One typed atom of the interaction model. |
| `InteractionCutoffs` | class | Every distance/angle threshold, exposed and documented. |
| `InteractionFingerprint` | class | The IFP: one bit per (ligand residue, protein residue, family). |
| `InteractionReport` | class | The full hit list of one :func:`detect_interactions` run. |
| `MolAtom` | class | One atom of the molecular graph. |
| `MolBond` | class | One bond: endpoints (atom indices), order 1/2/3, aromatic flag. |
| `Molecule` | class | A molecular graph plus parse bookkeeping. |
| `MoleculeGrid` | class | A DataFrame of molecules as a grid of cards -- mols2grid's |
| `MorganBitInfo` | class | Decode record for one Morgan fingerprint bit on one molecule. |
| `PdbLigandReport` | class | Audit record for one :func:`smiles_to_pdb` conversion. |
| `RepresentationSpec` | class | One representation: atom scale, bond switch, provenance. |
| `Residue` | class | A named group of interaction atoms. |
| `SAFilterResult` | class | §24.4 filtering outcome: kept AND rejected are both preserved. |
| `SAScoreBatch` | class | Batch results with the validity accounting (§24.2/24.3). |
| `SAScoreResult` | class | One molecule's SA score with full validation context (§24.2). |
| `SasaReport` | class | Solvent-accessible surface areas, per atom and total. |
| `SdfWriteReport` | class | Audit of one :func:`table_to_sdf` conversion. |
| `SelfiesBatchConversion` | class | Result of :func:`batch_smiles_to_selfies` -- per-row, never row-destructive. |
| `SelfiesRobustnessReport` | class | Validity under random corruption: SELFIES symbols vs SMILES characters. |
| `SelfiesRoundtripReport` | class | Round-trip fidelity audit of a SMILES column, the repo's own QA pattern. |
| `StandardizationProfile` | class | A registered chain: key, label, reference repo, builder. |
| `StandardizationRecord` | class | One CSV row's outcome (identifier + provenance). |
| `StandardizationReport` | class | The batch answer -- records, summary, exports. |
| `StandardizeOptions` | class | Every toggle the five chains respect, one name across profiles. |
| `StandardizeOutcome` | class | One molecule through one chain -- the per-row answer. |
| `TableComparison` | class | Set reconciliation of one ligand column across two tables. |
| `VarianceFilterResult` | class | Result of :func:`filter_low_variance`. |
| `add_ic50_column` | function | Append an IC50 column converted from a pIC50 column. |
| `add_pic50_column` | function | Append a pIC50 column to a table of measurements. |
| `align_structures` | function | CA-based alignment with iterative outlier rejection. |
| `annotate_with_smiles` | function | Annotate ligand-name columns with SMILES from a reference table. |
| `ascii_depiction` | function | Rasterise the 2D depiction of ``mol`` into a character grid. |
| `assign_secondary_structure` | function | DSSP-style secondary-structure assignment from backbone geometry. |
| `assign_types` | function | Type a residue's atoms by element and PDB atom name. |
| `average_molecular_weight` | function | Average molecular weight (daltons) from the IUPAC tables. |
| `b_factor_profile` | function | Per-residue B-factor statistics (the crystallographer's QA view). |
| `batch_smiles_to_selfies` | function | Encode a whole SMILES column, logging per-row failures. |
| `bond_angle` | function | Angle a-b-c in degrees (b is the vertex). |
| `build_coordinates_3d` | function | Rule-based 3D coordinates for ``mol`` -- ``(n_atoms, 3)`` array. |
| `build_peptide_backbone` | function | An ideal-geometry poly-alanine backbone by internal coordinates. |
| `canonical_smiles` | function | A deterministic canonical SMILES for the graph. |
| `canonicalize_tautomer` | function | Return the canonical-tautomer SMILES of ``smiles``. |
| `center_of_mass` | function | Mass-weighted centroid (unknown elements weigh as carbon). |
| `compare_tables` | function | Compare one ligand-name column across two tables. |
| `compile_selection` | function | Compile a PyMOL-style selection expression once, reuse forever. |
| `compute_3d_descriptors` | function | Compute RDKit 3D shape descriptors for a list of SMILES. |
| `compute_descriptors` | function | Compute molecular descriptors for a list of SMILES. |
| `compute_feature_frame` | function | Compute a multi-family feature matrix for a batch of SMILES. |
| `compute_fingerprints` | function | Compute fingerprints for a list of SMILES. |
| `compute_maccs` | function | Compute the 167-bit MACCS key fingerprints. |
| `compute_rism_descriptors` | function | Compute 3D-RISM-KH-inspired solvation descriptors for SMILES. |
| `contact_pairs` | function | All atom pairs within ``cutoff`` Å as ``(i, j, distance)``. |
| `decode_morgan_bit` | function | Decode one Morgan fingerprint bit into its atom environments. |
| `demo_complex` | function | A designed ligand/protein pair that triggers nine families. |
| `descriptor_card` | function | The ``info()`` popup as a text card: the descriptor set mols2grid |
| `detect_interactions` | function | Run every detector family over a ligand/protein pair. |
| `detect_smiles_column` | function | Find the SMILES column of a CSV frame -- the card's auto-pick. |
| `dihedral_angle` | function | Signed dihedral p0-p1-p2-p3 in degrees, −180..180. |
| `distance_matrix` | function | The full pairwise distance matrix ``(n, n)`` in Å. |
| `family_of_column` | function | Return the feature family a frame column belongs to. |
| `filter_by_sa` | function | Filter scored candidates by an inclusive SA range (§24.4). |
| `filter_low_variance` | function | Cut feature columns whose variance is below ``threshold``. |
| `find_rings` | function | The smallest set of smallest rings, as atom-index sets. |
| `fingerprint_diversity` | function | Mean pairwise Tanimoto *distance* across a pose series. |
| `fraction_csp3` | function | The fraction of carbon atoms that are sp3 (all-single, non-aromatic). |
| `generate_2d_coordinates` | function | Depiction coordinates — ring templates plus chain growth. |
| `grid_demo` | function | A self-contained demonstration: filter, sort, select, tooltip. |
| `has_substructure` | function | ``True`` when ``query`` matches anywhere in ``mol`` (early exit). |
| `hydrogen_bond_energies` | function | Kabsch-Sander backbone hydrogen-bond energies. |
| `ic50_from_pic50_series` | function | Vectorised pIC50 -> IC50 conversion preserving NaN for bad rows. |
| `ic50_to_pic50` | function | Convert an IC50 measurement to pIC50 = -log10(IC50 [M]). |
| `implicit_hydrogens` | function | Per-atom hydrogen counts from the valence model. |
| `inchi_roundtrip_check` | function | The cross-toolkit agreement check standing in OpenBabel's slot. |
| `kabsch_rotation` | function | The optimal rotation aligning ``mobile`` onto ``target``. |
| `ki_to_pki` | function | Convert an inhibition constant to pKi = -log10(Ki [M]). |
| `ligand_from_pdb` | function | Typed ligand residues from PDB text (HETATM records only). |
| `lipinski_profile` | function | Lipinski's own HBD/HBA counts plus the rotatable/ring/sp3 numbers. |
| `merge_table_folder` | function | Merge every table of one extension directly inside ``directory``. |
| `merge_tables` | function | Row-wise merge of several tables, ignoring the index. |
| `mol_block` | function | A V2000 MOL block for the graph — purely, no engine required. |
| `molecular_formula` | function | Hill-order molecular formula (C, H, then alphabetical). |
| `molecule_profile` | function | One SMILES -> the full graph-core profile (API + GUI surface). |
| `morgan_environments` | function | Per-atom environment identifiers, for interpretation surfaces. |
| `morgan_fingerprint` | function | A folded Morgan/ECFP-style bit fingerprint (``uint8`` 0/1 vector). |
| `parse_pdb_block` | function | Parse ATOM/HETATM records out of a PDB file's text. |
| `parse_smiles` | function | Parse a SMILES string into a :class:`Molecule` graph. |
| `perceive_bonds` | function | Distance-based bonds: radii sum × tolerance, no bonded H's. |
| `perceive_rings` | function | Aromatic rings from connectivity (spanning-tree fundamental cycles). |
| `pic50_to_ic50` | function | Convert a pIC50 value back to an IC50 concentration. |
| `point_distance` | function | Euclidean distance between two atoms (Å). |
| `prepare_pic50_dataset` | function | Turn a raw SMILES + EC50 table into a modelling-ready pEC50 set. |
| `protein_from_pdb` | function | Typed protein residues from PDB text. |
| `protonate_smiles` | function | The dimorphite_dl protonation enumerator for one SMILES. |
| `random_selfies_molecules` | function | Sample ``n`` molecules uniformly from the robust SELFIES alphabet. |
| `rank_features` | function | Rank feature columns by importance, descending. |
| `recalculate_selected_features` | function | Recalculate a frozen feature selection for NEW molecules. |
| `register` | constant |  |
| `reinvent_standardize_mol` | function | The REINVENT4 ``default`` standardization chain, on an RDKit Mol. |
| `reinvent_standardize_smiles` | function | :func:`reinvent_standardize_mol` from (and back to) a SMILES. |
| `render_ascii` | function | Render one molecule as ASCII -- the WebGL canvas as a char grid. |
| `render_primitives` | function | Depth-sorted drawing primitives (painter's algorithm order). |
| `residue_groups` | function | Group atom indices by residue, in first-appearance order. |
| `rmsd` | function | Root-mean-square deviation between two coordinate stacks. |
| `rmsf` | function | Per-atom RMS fluctuation over a frame stack. |
| `rotatable_bonds` | function | Lipinski rotatable bonds — RDKit's default (non-strict) definition. |
| `sa_score` | function | Score one SMILES string (or an RDKit molecule). |
| `sa_score_frame` | function | The §24.3 generative-results table: SMILES, SA, validity, order. |
| `sa_score_summary` | function | Direction-aware batch statistics for interfaces (§24.3). |
| `sa_scores` | function | Score a batch of SMILES with per-molecule validity (§24.2). |
| `sasa` | function | Shrake-Rupley solvent-accessible surface area (A^2), per atom. |
| `sdf_text` | function | A whole SDF text from molecules (2D coordinates, V2000 blocks). |
| `sdf_to_table` | function | Read an SD file back into a table (SMILES, tags and titles). |
| `secondary_structure_string` | function | ``assign_secondary_structure`` collapsed to the 1-letter string. |
| `select_atoms` | function | Evaluate a selection expression against an atom stack. |
| `select_spec` | function | 3Dmol.js's ``AtomSelectionSpec`` resolved to atom indices. |
| `select_string` | function | NGL's selection string grammar resolved to atom indices. |
| `select_top_features` | function | Select the top-K features with the rar's two-layer strategy. |
| `selection_metadata` | function | Build the JSON-ready metadata dict for a feature-selection run. |
| `selfies_alphabet` | function | Sorted set of SELFIES symbols appearing in an iterable of strings. |
| `selfies_available` | function | Report whether the optional ``selfies`` package is importable. |
| `selfies_feature_matrix` | function | Featurise a SMILES column as a SELFIES label / one-hot matrix. |
| `selfies_len` | function | Number of SELFIES symbols in a string (no full parse). |
| `selfies_mutants` | function | Enumerate a molecule's SELFIES-symbol point mutants (STONED-style). |
| `selfies_robustness_experiment` | function | Reproduce the SELFIES paper's Figure 3 on your own molecules. |
| `selfies_roundtrip_report` | function | Audit a SMILES column: encode → decode → canonical-SMILES equality. |
| `selfies_symbol_frequencies` | function | Symbol → occurrence count over an iterable of SELFIES strings. |
| `selfies_to_smiles` | function | Translate one SELFIES string back to SMILES. |
| `selfies_tokens` | function | Split one SELFIES string into its symbols (the tokenizer). |
| `semantic_constraint_presets` | function | The constraint presets the package ships. |
| `semantic_constraints` | function | Temporarily set the global SELFIES semantic constraints. |
| `signed_feature_correlations` | function | Signed correlation of every numeric column with ``target``. |
| `similarity_scan` | function | Scan reference ligands against a target library (Tanimoto >= threshold). |
| `smi_to_csv` | function | Stream a ``.smi`` file to CSV one line at a time. |
| `smiles_to_pdb` | function | Generate an optimised 3D PDB ligand from a SMILES string. |
| `smiles_to_selfies` | function | Translate one SMILES string to its SELFIES encoding. |
| `split_table_into_chunks` | function | Stream a large CSV into fixed-size part files. |
| `standardize_csv` | function | Standardise a CSV of SMILES -- the batch front door. |
| `standardize_smiles` | function | Standardise one SMILES to its canonical form -- every chain |
| `structure_summary` | function | The header-line facts: chains, residues, elements, B-values. |
| `substructure_matches` | function | All atom-index mappings of ``query`` inside ``mol``. |
| `superpose` | function | Least-squares superposition of ``mobile`` onto ``target``. |
| `table_to_sdf` | function | Write a SMILES table to an SD file with 3D structures. |
| `table_to_smi` | function | Write a SMILES table to a ``.smi`` list file. |
| `tanimoto_from_bitarrays` | function | Tanimoto coefficient for two equal-length binary arrays. |
| `tanimoto_from_sets` | function | Tanimoto coefficient for two feature sets: |A∩B| / |A∪B|. |
| `tanimoto_matrix` | function | All-vs-all Tanimoto similarities between two bit-matrix batches. |
| `tooltip_line` | function | Format one record's ``columns`` as tooltip key/value pairs. |
| `top_matches` | function | Return the ``k`` library entries most similar to ``query``. |
| `total_hydrogens` | function | Alias kept for reader symmetry with RDKit's ``GetTotalNumHs``. |
| `tpsa` | function | Topological polar surface area — Ertl's algorithm, transcribed. |
| `view_report` | function | The Stage as text: what the viewer would show, in numbers. |
| `write_pdb_block` | function | Render atom records back to PDB text (plus ``TER``/``END``). |

## `pharmaml.ml`

*QSAR machine learning: preprocessing, training, evaluation, persistence,*

| Symbol | Kind | Summary |
|---|---|---|
| `AMINO_ACIDS` | constant |  |
| `AUX_LOSS_WEIGHT` | constant |  |
| `CATALOG_CATEGORIES` | constant |  |
| `CATALOG_LICENSE` | constant |  |
| `CATALOG_PROJECTS` | constant |  |
| `CATALOG_SNAPSHOT` | constant |  |
| `CATALOG_SOURCE_URL` | constant |  |
| `CATEGORY_COVERAGE` | constant |  |
| `CHANGEPOINT_ALPHA` | constant |  |
| `CHANGEPOINT_MIN_SEGMENT` | constant |  |
| `CLASSIFICATION_CV_METRICS` | constant |  |
| `COVERAGE_EXTERNAL` | constant |  |
| `COVERAGE_NATIVE` | constant |  |
| `COVERAGE_OUT_OF_SCOPE` | constant |  |
| `COVERAGE_PARTIAL` | constant |  |
| `CV_STRATEGIES` | constant |  |
| `DEEPDTA_CHARCANSMISET` | constant |  |
| `DEEPDTA_CHARISOSMISET` | constant |  |
| `DEEPDTA_CHARPROTSET` | constant |  |
| `DEFAULT_ALPHA` | constant |  |
| `DEFAULT_BETA` | constant |  |
| `DISTRIBUTIONS` | constant |  |
| `ENTROPY_BINS` | constant |  |
| `GMM_COVARIANCE_TYPES` | constant |  |
| `INITIAL_POINTS_PER_DIM` | constant |  |
| `KDE_KERNELS` | constant |  |
| `MANIFOLD_METHODS` | constant |  |
| `METRIC_LABELS` | constant |  |
| `MODEL_ALIASES` | constant |  |
| `MODEL_REGISTRY` | constant |  |
| `NO_SCALE_MODELS` | constant |  |
| `QUERY_STRATEGIES` | constant |  |
| `REGRESSION_CV_METRICS` | constant |  |
| `SCRATCH_ACTIVATIONS` | constant |  |
| `SCRATCH_TASKS` | constant |  |
| `SEQUENCE_MODELS` | constant |  |
| `SGD` | class | SGD with momentum, Nesterov and (L2) weight decay. |
| `SHAP_EXTRA` | constant |  |
| `SHAP_REQUIREMENT` | constant |  |
| `SIMILARITY_METRICS` | constant |  |
| `SPLIT_STRATEGIES` | constant |  |
| `SYNTHETIC_KINDS` | constant |  |
| `ANNReport` | class | The bargain measured: recall against the exact baseline. |
| `ANOVARow` | class | One Type-II row: the term, its SS, F and p. |
| `ActiveLearningLoop` | class | The §14.1 workflow as one honest, serializable object. |
| `Adagrad` | class | Adagrad: the per-coordinate accumulator ``G += g²``. |
| `Adam` | class | Adam with bias correction (and optional AMSGrad). |
| `AdamW` | class | Adam with decoupled weight decay (the Loshchilov-Hutter fix). |
| `AdaptedModel` | class | A K-shot adapted copy of the meta model. |
| `ApplicabilityDomainResult` | class | Outcome of a full applicability-domain analysis. |
| `AttentionReport` | class | One sequence's pass through the attention machinery. |
| `BCEWithLogitsLoss` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `BaselineResult` | class | The commonsense baseline's scorecard (the chapter's first table). |
| `BatchGenerator` | class | Chollet's ``BatchGenerator`` -- sequential, ``num_batches`` counted. |
| `BayesTrial` | class | One trial: the parameters tried and the score observed. |
| `BayesianSearchReport` | class | The oracle's ledger: trials, the best, the acquisition diagnostics. |
| `BootstrapUncertainty` | class | Bootstrap-ensemble uncertainty of a fitted estimator. |
| `BroadcastReport` | class | NumPy's broadcasting verdict for two shapes. |
| `CVResult` | class | A complete §19 CV report for one model on one dataset. |
| `CapacityPoint` | class | One (width, train loss, validation loss) sample of the sweep. |
| `CapacitySweepResult` | class | Chapter 5's "reducing the network's size" figure, as data. |
| `CatalogCategory` | class | One catalogue section: slug, title, subtitle, live counts. |
| `CatalogProject` | class | One catalogue entry, with its category resolved to the title. |
| `CatalogSummary` | class | The snapshot's headline numbers. |
| `Changepoint` | class | One detected change: where, which way, how much. |
| `ChangepointResult` | class | The binary-segmentation CUSUM verdict for a series. |
| `CiResult` | class | One confidence interval: the estimate and its bounds. |
| `ComparisonReport` | class | ``compare_groups``: the whole statistics lecture, one record. |
| `CompositionReport` | class | The ε bill for a sequence of queries. |
| `ConformalIntervals` | class | Prediction intervals and optional empirical evaluation on query rows. |
| `ConformalPredictionSets` | class | Class membership per query row and optional empirical coverage. |
| `CorrelationResult` | class | The lag correlation table, with the significance band. |
| `CoverageEntry` | class | The audit's verdict for one catalogue category. |
| `CoverageGroup` | class | Empirical coverage, Wilson bounds, and optional mean size for one group. |
| `CoverageReport` | class | Overall and first-seen-ordered subgroup coverage on held-out observations. |
| `CrossEntropyLoss` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `DPNoiseScale` | class | The noise multiplier a training plan needs, by bisection. |
| `DPStatisticsReport` | class | A column's headline statistics, privatised under one ε. |
| `DTAFeatureSet` | class | Aligned paired features, optional target and explicit row accounting. |
| `DecompositionResult` | class | The additive split: observed = trend + seasonal + remainder. |
| `DeepMLPClassifier` | class | Deep feed-forward neural-network classifier (PyTorch). |
| `DeepMLPRegressor` | class | Deep feed-forward neural-network regressor (PyTorch). |
| `DescriptiveStats` | class | The descriptive lecture's full blackboard for one sample. |
| `DesignInfo` | class | The design matrix and everything about how it was built. |
| `DistributionSample` | class | Seeded draws plus the population moments to compare against. |
| `DoseResponseReport` | class | A fitted sigmoid Emax curve with its uncertainty. |
| `EmbeddingRecord` | class | Typed report of :func:`manifold_embedding`. |
| `EvaluationEstimate` | class | The pre-execution budget answer (§18.5). |
| `Factor` | class | One predictor inside a term: numeric, categorical or intercept. |
| `FewShotReport` | class | The three-arm verdict on a new task (TCRP's evaluation). |
| `FoldRecord` | class | One fold's honest bookkeeping (§19.2 "individual fold scores"). |
| `ForecastResult` | class | Everything the sequence card reports, after one run. |
| `GaussianProcessSurrogate` | class | An RBF-kernel Gaussian process over the trials seen so far. |
| `GmmResult` | class | Typed report of :func:`gmm_cluster`. |
| `GroupDifference` | class | Cohort mean-SHAP differences with a bootstrap interval (100x-160). |
| `HierarchicalSVR` | class | Hierarchical Support Vector Regression (global + per-cluster SVRs). |
| `HistogramBin` | class | One DP histogram bin: edges, true count, DP count. |
| `HyperParameters` | class | The keras-tuner container of hyperparameter definitions. |
| `Hyperband` | class | Hyperband successive halving over a resource hyperparameter. |
| `HyperparameterImportance` | class | Per-hyperparameter importances of a finished search. |
| `KMeansResult` | class | Typed report of :func:`kmeans_cluster`. |
| `KdeResult` | class | Typed report of :func:`kde_fit`. |
| `LMResult` | class | One Levenberg-Marquardt run's ledger. |
| `LSHCollection` | class | The usable engine: an :class:`LSHIndex` plus the stored vectors. |
| `LSHIndex` | class | The random-hyperplane LSH table (Charikar's SimHash engine). |
| `LSHQueryResult` | class | One approximate query: the key, the candidates, the neighbours. |
| `LabelConditionalConformalClassifier` | class | A fixed classifier with one split-conformal threshold per label. |
| `Linear` | class | ``y = x @ W.T + b`` with PyTorch's default init bounds. |
| `LinearGDResult` | class | Outcome of the from-scratch linear regression. |
| `MOTrial` | class | One multi-objective evaluation: params, the objective vector, the weights. |
| `MSELoss` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `MechanismResult` | class | One privatised query: the truth, the release, the price. |
| `MedianPruner` | function | The median pruner: ``PercentilePruner(50)``. |
| `MetaHistory` | class | The meta-training curve. |
| `MetaModel` | class | The meta-trained network (TCRP's ``observed_tissue_model``). |
| `MetricSummary` | class | One metric aggregated across folds (mean, std, every fold value). |
| `ModelInputReport` | class | Verdict of :func:`validate_bundle_input` — the round-trip audit. |
| `Module` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `MultiObjectiveStudy` | class | The optuna multi-objective study, decomposed onto one sampler. |
| `NaiveDense` | class | Chollet's ``NaiveDense`` (chapter 2), plain NumPy. |
| `NaiveSequential` | class | Chollet's ``NaiveSequential``: a stack called like a function. |
| `Neighbor` | class | One retrieved neighbour: row index and its Euclidean distance. |
| `NormalityResult` | class | Shapiro-Wilk and D'Agostino-Pearson on one sample. |
| `NumpyMLPClassifier` | class | Feed-forward classifier trained on the autograd engine. |
| `NumpyMLPRegressor` | class | Feed-forward regressor trained on the autograd engine. |
| `OLSResult` | class | Ordinary least squares through a formula. |
| `OOFStackingResult` | class | Outcome of one out-of-fold stacking run. |
| `OntologyGraph` | class | A parsed DrugCell ontology: the DAG plus its gene annotations. |
| `OntologyReport` | class | The hierarchy audit: rows per term plus the headline stats. |
| `OntologyRow` | class | One term's audit row: size, depth, layer, gene enrichment. |
| `ParamAdvice` | class | One searchable hyperparameter: its range, prior, role, and deps. |
| `Parameter` | class | A leaf tensor that always requires grad. |
| `ParetoReport` | class | The front, its spread, its indicator: what a trade-off decision reads. |
| `ParetoShortlist` | class | Selected trials and their crowding measured on the original full front. |
| `PcaResult` | class | Typed report of :func:`pca_reduce`. |
| `PercentilePruner` | class | Prune trials whose running CV mean falls below a percentile. |
| `PermutationImportance` | class | Permutation feature importances on held-out data. |
| `PositionalReport` | class | The positional encoding as an auditable object. |
| `PrimerReport` | class | The full chapter 2 walkthrough for one dataset table. |
| `QuantileIntervals` | class | Prediction intervals from gradient-boosted quantile regression. |
| `QueryRecord` | class | One AL round's bookkeeping (provenance, never labels). |
| `RDPReport` | class | The (ε, δ) guarantee of T Gaussian steps at noise multiplier σ. |
| `RMSprop` | class | RMSprop: the running quadratic mean with ``alpha`` decay. |
| `RadiusResult` | class | One radius of the audit grid: how many samples broke. |
| `RandomSearch` | class | Random search over the hypermodel's declared space. |
| `ReLU` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `Recommendation` | class | One ranked item: index, predicted rating, neighbourhood score. |
| `RecommenderEvaluation` | class | Leave-one-out hit-rate, CF versus the popularity baseline. |
| `RecurrentForecaster` | class | A from-scratch recurrent regressor over fixed-length windows. |
| `ResidualMLPClassifier` | class | Residual (skip-connection) deep-network classifier (PyTorch). |
| `ResidualMLPRegressor` | class | Residual (skip-connection) deep-network regressor (PyTorch). |
| `RobustnessAudit` | class | The verdict curve of a model under the sign-pattern family. |
| `ScratchHistory` | class | Everything the scratch training loop observed, per epoch. |
| `SearchRecommendation` | class | A preset applied and adapted to a concrete dataset/budget. |
| `SemiSupervisedEvaluation` | class | Leakage-safe honest evaluation of a semi-supervised configuration. |
| `SemiSupervisedMethod` | constant |  |
| `SemiSupervisedResult` | class | Outcome of one semi-supervised fit. |
| `SequenceHistory` | class | Per-epoch training telemetry for one recurrent fit. |
| `Sequential` | class | Modules applied in order. |
| `ShapInteractions` | class | Pairwise SHAP interaction values of a fitted tree model (100x-160). |
| `ShapSummary` | class | SHAP values of a fitted tree model over a feature frame. |
| `SharedSplitKey` | class | One cross-partition key and its row counts, in partition order. |
| `Sigmoid` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `SignificanceResult` | class | One test's verdict: statistic, p-value, decision, effect size. |
| `Softmax` | class | The container protocol: recursive ``parameters()``, ``__call__``. |
| `SpaceIssue` | class | One compatibility or sanity finding (§18.2 validation). |
| `SplitConformalClassifier` | class | Reference to a fixed classifier and its held-out calibration threshold. |
| `SplitConformalRegressor` | class | A fitted model reference and its split-conformal calibration radius. |
| `SplitOverlapReport` | class | Exact-key overlap and duplicate counts for named, nonempty partitions. |
| `TPESearch` | class | Tree-structured Parzen Estimator search — Optuna's flagship sampler. |
| `TaskFamily` | class | Seeded related regression tasks: one shared prior, per-task draws. |
| `TaskSplit` | class | One task's support/query split (the inner-loop data contract). |
| `Tensor` | class | A differentiable value: NumPy data plus reverse-mode history. |
| `TensorSummary` | class | Chapter 2's "key attributes" list for one array. |
| `Term` | class | A product of factors (``a:b``); a single factor is a main effect. |
| `TermContribution` | class | One subsystem's readout: the aux head's contribution, ranked. |
| `TrialResult` | class | One evaluated trial. |
| `TunerResults` | class | The outcome of one :meth:`RandomSearch.search` / Hyperband run. |
| `UserRecommendations` | class | A user's top-N list, with the method and neighbourhood recorded. |
| `VisibleForward` | class | One forward pass through the hierarchy. |
| `VisibleHistory` | class | The training curve plus the final agreement numbers. |
| `VisibleNetwork` | class | DrugCell's architecture in plain NumPy, training included. |
| `WilliamsResult` | class | Outcome of a leverage/Williams applicability-domain analysis. |
| `WindowDataset` | class | A windowed timeseries: the chapter's dataset object, NumPy edition. |
| `adapt` | function | Clone-and-fine-tune on a support set (the inner loop alone). |
| `adversarial_audit` | function | Run the sign-pattern family across a radius grid. |
| `adversarial_augmentation` | function | Adversarial training's closed-form half: in-ball noisy copies. |
| `advice_families` | function | The tokens with hand-authored §18.3 tables. |
| `ann_benchmark` | function | Recall@k and candidate compression of LSH against brute force. |
| `anova_ii` | function | Type-II ANOVA table from the term slices. |
| `applicability_domain` | function | The uploaded workflow end-to-end, one call. |
| `audit_split_overlap` | function | Audit exact identifier overlap across two or more named data partitions. |
| `autocorrelation` | function | The autocorrelation function: which lag carries which memory. |
| `bayesian_search` | function | The BayesianOptimizationOracle loop, minimising ``objective``. |
| `bce_with_logits` | function | Binary cross-entropy with logits, the numerically stable form. |
| `benchmark_feature_families` | function | Cross-validated leaderboard ACROSS feature families, one model fixed. |
| `bootstrap_interval` | function | The percentile bootstrap CI for any statistic (default: the mean). |
| `bootstrap_uncertainty` | function | Bootstrap-ensemble prediction uncertainty (the upload's step 6). |
| `broadcast_report` | function | Chapter 2's broadcasting rules, computed on shapes alone. |
| `brute_force_knn` | function | The exact top-k by Euclidean distance -- the benchmark baseline. |
| `build_ann_classifier` | function | sklearn MLP classifier hypermodel (the legacy ANN upgrade). |
| `build_ann_regressor` | function | sklearn MLP regressor hypermodel (the legacy ANN upgrade). |
| `build_cv_splitter` | function | Build the CV splitter plus its limitations notes (§19.3). |
| `build_deep_classifier` | function | Deep MLP classifier hypermodel (PyTorch, ``batchnorm`` searchable). |
| `build_deep_regressor` | function | Deep MLP regressor hypermodel (PyTorch, ``batchnorm`` searchable). |
| `build_hsvr` | function | :class:`~pharmaml.ml.hierarchical.HierarchicalSVR` hypermodel. |
| `build_model` | function | Instantiate an unfitted estimator from the registry. |
| `build_preprocessor` | function | Build the (impute -> scale) preprocessing pipeline. |
| `build_stacking` | function | Build a stacking ensemble (upgrades Stacking.py / Stacking reg.py). |
| `build_voting` | function | Build a voting ensemble (upgrades Voting.py / Voting reg.py). |
| `calibrate_label_conditional_classifier` | function | Calibrate one conformal score threshold per class on held-out rows. |
| `calibrate_split_conformal` | function | Calibrate finite split-conformal intervals on held-out regression data. |
| `calibrate_split_conformal_classifier` | function | Calibrate split-conformal label sets on held-out classification rows. |
| `capacity_sweep` | function | Train the same data through growing hidden widths (chapter 5). |
| `catalog_categories` | function | The 35 catalogue sections with live project counts and star totals. |
| `catalog_projects` | function | One category's projects (slug or title), or all 924, score-descending. |
| `catalog_summary` | function | The snapshot's headline numbers and license distribution. |
| `catalog_top` | function | The ``n`` highest-scoring projects in a category (or overall). |
| `chebyshev_scalarize` | function | The Tchebycheff scalarization ``max_j w_j |v_j - ideal_j|``. |
| `clean_numeric_frame` | function | Drop non-numeric columns (except the target/keep list). |
| `commonsense_baseline` | function | Predict "the last observed value" (``evaluate_naive_method``). |
| `compare_groups` | function | The course's decision flow, end to end, every branch reported. |
| `compare_models` | function | Cross-validated leaderboard over (a subset of) the registry. |
| `compare_models_cv` | function | Cross-validate a model panel into full §19 reports + the board. |
| `composition_report` | function | Sequential composition: the ε's of queries on the same rows add. |
| `contrast_matrix` | function | The coding matrix for one categorical factor. |
| `coverage_by_group` | function | Summarize boolean held-out coverage overall and within observed groups. |
| `coverage_table` | function | The category-by-category audit, in catalogue order. |
| `cross_entropy` | function | Softmax cross-entropy: ``mean(−log_softmax[i, label_i])``. |
| `cross_model_importances` | function | Permutation importances for a WHOLE panel of fitted models. |
| `cross_validate_model` | function | Cross-validate ONE registry model into a full §19 CVResult. |
| `crowding_distance` | function | NSGA-II's crowding distance (boundaries ``+inf``), per point. |
| `curve_fit` | function | The scipy-named entry point: (best params, covariance matrix). |
| `cv_result_from_dict` | function | Rebuild a :class:`CVResult` from :meth:`CVResult.to_dict` output. |
| `cv_result_to_dict` | function | §19.9 interchange: :class:`CVResult` → JSON-ready dict. |
| `dataset_fingerprint` | function | Stable sha256 fingerprint of the data a CV result belongs to. |
| `deepdta_rm2` | function | DeepDTA's ``get_rm2`` external-prediction metric. |
| `default_folds` | function | Fold-count default with its justification (§19.3). |
| `demo_dta_frame` | function | Twelve synthetic, clearly non-benchmark pairs for the offline GUI. |
| `describe_series` | function | One sample's descriptive statistics (scipy's moment machinery). |
| `describe_tensor` | function | Audit one array with chapter 2's "key attributes" vocabulary. |
| `design_matrices` | function | Expand a formula against a data frame into the design matrix. |
| `detect_changepoints` | function | Offline mean-shift changepoints: binary segmentation on the CUSUM. |
| `dominates` | function | Pareto dominance under the optuna convention (all objectives minimized). |
| `dot_product_attention` | function | Scaled dot-product attention: ``softmax(QKᵀ/√dₖ)·V``. |
| `dp_sgd_noise_scale` | function | Invert the RDP accountant: the σ that meets (ε, δ) in T steps. |
| `dp_statistics` | function | Count, sum, mean (+ optional histogram) under one ε budget. |
| `estimate_evaluations` | function | Estimate the number of evaluations a budget implies. |
| `evaluate_model` | function | Evaluate a fitted model on held-out data. |
| `evaluate_recommender` | function | Leave-one-out: hide one rating per user, rebuild, hit@N. |
| `evaluate_semi_supervised` | function | Leakage-safe honest evaluation of a semi-supervised configuration. |
| `expected_improvement` | function | The classic expected improvement over the best observed score. |
| `few_shot_report` | function | Zero-shot vs K-shot vs from-scratch, Pearson-scored. |
| `fit_dose_response` | function | Fit the sigmoid Emax (Hill) model to (dose, effect) observations. |
| `fit_ols_formula` | function | Least squares through the formula language. |
| `fit_scratch` | function | Chapter 2's full training loop, mouse-free edition. |
| `fomaml_train` | function | First-order MAML over the task family (TCRP's train loop). |
| `forecast_timeseries` | function | The chapter's end-to-end workflow on one series. |
| `format_cv_score` | function | Render ``CV Score = Mean ± Standard Deviation`` honestly. |
| `gaussian_mechanism` | function | The analytic Gaussian mechanism: (ε, δ)-DP with the classic bound. |
| `gmm_cluster` | function | Gaussian mixture model (PDSH 05.12) with the AIC/BIC sweep. |
| `gradient_check` | function | Central-difference check of ``function``'s reverse-mode gradient. |
| `group_difference` | function | Cohort difference of a SHAP summary, with a bootstrap interval. |
| `has_categorical_columns` | function | Return True when ``X`` is a DataFrame with non-numeric columns. |
| `hierarchy_report` | function | The peeling audit: per-term rows and the hierarchy's shape. |
| `hypermodel_from_advice` | function | Compile a recommendation into a tuner-searchable hypermodel. |
| `hypervolume_2d` | function | The exact 2-D dominated hypervolume under minimisation. |
| `kde_fit` | function | Kernel density estimation (PDSH 05.13) for ONE numeric column. |
| `kmeans_cluster` | function | k-means clustering (PDSH 05.11), with the elbow AND the silhouette. |
| `knn_applicability_domain` | function | kNN-distance applicability domain (the upload's step 4-5). |
| `label_sequence` | function | Encode a protein with DeepDTA's exact labels, truncation and padding. |
| `label_smiles` | function | Encode SMILES with DeepDTA's exact labels, truncation and zero padding. |
| `laplace_mechanism` | function | The Laplace mechanism: exact (ε, 0)-differential privacy. |
| `levenberg_marquardt` | function | Levenberg-Marquardt least squares, plain NumPy, deterministic. |
| `leverage_williams` | function | Leverage/Williams applicability domain (the regression companion). |
| `linear_regression_gd` | function | Batch gradient descent on the squared error (DSBC's centrepiece). |
| `list_models` | function | List registry entries with availability information. |
| `load_model` | function | Load a model bundle saved by :func:`save_model`. |
| `log_softmax` | function | Log-softmax, the stable way (x − logsumexp). |
| `lower_confidence_bound` | function | ``mean − β·std`` -- UCB's minimisation form (keras-tuner's pick). |
| `make_cv_splitter` | function | The tuners' fold protocol, as a public helper. |
| `make_strategy_split` | function | Strategy-aware train/test split, drop-in for the preprocessing twin. |
| `make_train_test_split` | function | Stratified train/test split with automatic fallback. |
| `manifold_embedding` | function | Project a feature table onto 2-D with a manifold learner (PDSH 05.10). |
| `mann_whitney_test` | function | The rank-based test for the non-normal cases. |
| `mean_confidence_interval` | function | The t-based CI for the mean: ``mean ± t·s/√n``. |
| `minimal_attack_radius` | function | Bisect the smallest radius at which some pattern breaks the sample. |
| `mse_loss` | function | Mean squared error (no gradient through the target). |
| `multi_head_attention` | function | Multi-head self-attention over one sequence, seeded projections. |
| `murcko_scaffold_key` | function | Canonical Bemis–Murcko scaffold SMILES for one molecule. |
| `no_grad` | function | Disable graph construction (the inference-mode switch). |
| `normality_report` | function | Both normality tests (the course's gate before choosing a test). |
| `one_hot_sequence` | function | DeepDTA's dense ``(max_length, 25)`` one-hot protein encoding. |
| `one_hot_smiles` | function | DeepDTA's dense ``(max_length, alphabet_size)`` SMILES one-hot. |
| `one_sample_ttest` | function | Is this sample's mean different from ``mu0``? |
| `paired_dta_features` | function | Build an aligned ECFP + protein matrix from a paired DTA table. |
| `paired_ttest` | function | Did the paired differences move away from zero? |
| `param_importances` | function | Random-forest (MDI) hyperparameter importances of a finished search. |
| `pareto_indices` | function | The non-dominated indices of a point cloud (ascending input order). |
| `parse_formula` | function | Parse a formula into ``(response, terms)``. |
| `parse_ontology` | function | Parse the ``drugcell_ont.txt`` three-column format. |
| `partial_autocorrelation` | function | The partial ACF via Durbin-Levinson: lag-k credit, shorter lags removed. |
| `pca_reduce` | function | Principal component analysis (PDSH 05.09). |
| `permutation_feature_importance` | function | Permutation importances of a fitted model on held-out data. |
| `predict_scratch` | function | Predict with a trained scratch model (standardisation reapplied). |
| `predict_with_uncertainty` | function | Predictive mean and per-row standard deviation of a GP model. |
| `preset_styles` | function | The three §18.2 preset styles, in increasing cost order. |
| `protein_feature_frame` | function | Return aligned AAC (and optionally dipeptide) features per sequence. |
| `quantile_prediction_intervals` | function | Central-``1 - alpha`` prediction intervals via quantile regression. |
| `rdp_epsilon` | function | The RDP accountant: T plain Gaussian steps -> (ε, δ). |
| `recommend_itembased` | function | Item-based collaborative filtering: the similarity-weighted sum. |
| `recommend_search` | function | The §18.2 recommendation: preset + dataset/budget adaptation. |
| `recommend_userbased` | function | User-based collaborative filtering: the k-neighbour weighted mean. |
| `recommendation_from_json` | function | Load a recommendation exported by :func:`recommendation_to_json`. |
| `recommendation_to_json` | function | Serialize a recommendation (§18.5's export-as-JSON). |
| `resolve_model_name` | function | Normalise a user-supplied model name to a registry key. |
| `resource_bounds` | function | The declared ``[min, max]`` grid of a hypermodel's resource slot. |
| `sample_distribution` | function | Draw ``n`` seeded samples from one of the course's distributions. |
| `save_model` | function | Persist a fitted model plus its metadata bundle. |
| `scaffold_aware_coverage` | function | Empirical coverage summarized per Murcko scaffold. |
| `scaffold_groups` | function | Scaffold key per SMILES, in input order (``[""]`` for failures). |
| `search_catalog` | function | Case-insensitive substring search over names and descriptions. |
| `seasonal_decomposition` | function | Classical additive decomposition: centered-MA trend, phase-average seasonal. |
| `select_kbest_union` | function | Select the UNION of per-method top-k columns (upgrades look.ipynb). |
| `self_attention_report` | function | Run one sequence through multi-head self-attention and read it. |
| `shap_interactions` | function | Pairwise SHAP interaction values of a fitted tree model over ``X``. |
| `shap_summary` | function | SHAP values of a fitted tree model over ``X``. |
| `sigmoid` | function |  |
| `similarity_matrix` | function | User-user similarities over co-rated items only. |
| `sinusoidal_positions` | function | The classic fixed positional encoding, ``(seq_len, dim)``. |
| `sliding_windows` | function | Cut a series (or channel matrix) into overlapping windows. |
| `softmax` | function | Row-stable softmax (subtract-max). |
| `split_features_target` | function | Separate features from the target column. |
| `split_indices` | function | Core dispatcher: ``(train_indices, test_indices)`` for a strategy. |
| `stack_oof_regressors` | function | Stack registry regressors out-of-fold and rank the blenders. |
| `synthetic_ratings` | function | A seeded rating matrix with genuine latent structure. |
| `synthetic_series` | function | Seeded synthetic series: the no-data fallback for the whole card. |
| `tanh` | function |  |
| `tensor_primer_report` | function | Walk a table through chapter 2's tensor vocabulary. |
| `train_model` | function | Train (and optionally grid-search) one registry model. |
| `train_semi_supervised` | function | Fit one of the three semi-supervised methods on ``X`` / ``y``. |
| `tune_model_multi` | function | Cross-validated multi-objective HPO over the sklearn registry models. |
| `two_sample_ttest` | function | Two independent samples: Welch's t (default) or Student's t. |
| `validate_bundle_input` | function | Audit a new dataset against a saved model bundle before inference. |
| `validate_space` | function | Compatibility-check a search space before optimization starts. |
| `validate_unlabeled` | function | Validate a target vector under the ``-1`` unlabeled convention. |
| `validate_values` | function | Compatibility-check one drawn parameter combination. |
| `window_features` | function | Fixed non-overlapping windows, one feature row each (tsfresh-lite). |

## `pharmaml.generative`

*Molecule generation from 0 to 100: objectives, genetic algorithm, RL.*

| Symbol | Kind | Summary |
|---|---|---|
| `DEFAULT_CORPUS` | constant |  |
| `DIVERSITY_FILTERS` | constant |  |
| `RAW_COMPONENTS` | constant |  |
| `SCAFFOLD_PENALTY_FUNCTIONS` | constant |  |
| `TRANSFORM_REGISTRY` | constant |  |
| `CustomAlertsObjective` | class | REINVENT4's ``CustomAlerts`` blocklist filter. |
| `DiversityFilter` | class | Base class: the scaffold memory every concrete filter shares. |
| `DiversityFilteredObjective` | class | Any objective through a REINVENT4 diversity filter. |
| `GenerationResult` | class | Ranked molecules from one generative optimisation run. |
| `GeneticOptimizer` | class | Tournament-selection genetic algorithm over canonical SMILES. |
| `GroupCountObjective` | class | REINVENT4's ``GroupCount`` component (``comp_group_count.py``). |
| `IdenticalMurckoScaffoldFilter` | class | Penalise repeated Murcko scaffolds (``identical_murcko_scaffold``). |
| `IdenticalTopologicalScaffoldFilter` | class | The generic-scaffold variant (``identical_topological_scaffold``). |
| `LogPBandObjective` | class | Gaussian band on Crippen LogP (wild octanol/water behaviour excluded). |
| `MatchingSubstructureObjective` | class | REINVENT4's ``MatchingSubstructure`` penalty (``0.5 * (1 + match)``). |
| `MctsOptimizer` | class | ChemTSv2-style MCTS over SMILES prefixes, guided by a language model. |
| `MolecularWeightBandObjective` | class | Gaussian band on average molecular weight (Da).  See |
| `MultiObjective` | class | Weighted geometric mean of several objectives, with veto semantics. |
| `NoFilter` | class | The omitted-``[diversity_filter]`` behaviour: score untouched. |
| `NoveltyObjective` | class | Structural novelty against a reference corpus. |
| `ParetoFront` | class | The non-dominated set of one generation run (audit-ready). |
| `PenalizeSameSmilesFilter` | class | Multiply every *repeat* SMILES (``penalize_same_smiles``). |
| `QedObjective` | class | RDKit QED (quantitative estimate of drug-likeness). |
| `RawComponentObjective` | class | A REINVENT4 component: raw descriptor value pushed through a transform. |
| `ReinventOptimizer` | class | Policy-gradient fine-tuning of a SMILES RNN prior. |
| `SAFilteredCandidates` | class | Section 24.4 outcome: kept AND rejected are both preserved. |
| `SAScoreObjective` | class | Ertl synthetic-accessibility score as a generation reward (part3 §24.5). |
| `ScaffoldSimilarityFilter` | class | Bucket by *similar* scaffolds (``scaffold_similarity``). |
| `ScoreTransform` | class | Base class of the REINVENT4 transform DSL. |
| `ScoringObjective` | class | Base class for generation objectives: ``score(smiles) -> [0, 1]``. |
| `SimilarityObjective` | class | Tanimoto similarity (Morgan fingerprints) to one seed molecule. |
| `SmilesRnn` | class | Character-level GRU SMILES generator (CPU-only, torch optional). |
| `SubstructureObjective` | class | Hard pharmacophore constraint: 1.0 when ``smarts`` matches. |
| `TpsaBandObjective` | class | Gaussian band on topological polar surface area (Å²).  See |
| `bioisostere_moves` | function | Every distinct single-atom bioisostere mutant of ``smiles`` (sorted). |
| `bioisostere_mutate` | function | Draw one single-atom bioisostere mutant of ``smiles`` (or ``None``). |
| `make_diversity_filter` | function | Build a diversity filter by its REINVENT4 TOML ``type`` key. |
| `make_transform` | function | Build a transform by registry name (REINVENT4's ``get_transform``). |
| `pareto_front` | function | Extract the non-dominated molecules (Pareto front) of a run. |
| `ring_topology_moves` | function | Every distinct one-step ring-size mutant of ``smiles`` (sorted). |
| `ring_topology_mutate` | function | Draw one ring-size mutant of ``smiles`` with ``rng`` (or ``None``). |
| `run_genetic_optimization` | function | One-call genetic optimisation (thin :class:`GeneticOptimizer` wrapper). |
| `run_mcts_optimization` | function | One-call MCTS convenience (mirrors ``run_genetic_optimization``). |
| `run_rl_optimization` | function | One-call RL optimisation (trains a prior from the corpus if needed). |
| `scaffold_hop_crossover` | function | BREED-style crossover: homologous cuts, hybrid child. |
| `weighted_arithmetic_mean` | function | REINVENT4's weighted arithmetic mean (``custom_sum`` alias). |
| `weighted_geometric_mean` | function | REINVENT4's weighted geometric mean (``custom_product`` alias). |

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
| `EVID` | class | Event-identifier (EVID) codes for dosing/observation data records. |
| `GFR_MATURATION` | constant |  |
| `OSP_PK_PARAMETERS` | constant |  |
| `SOLVER_DEFAULTS` | constant |  |
| `EventRecord` | class | One row of a NONMEM-style dosing/observation data set. |
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
| `dubois_bsa` | function | Body surface area (m2) by the Du Bois 1916 formula. |
| `ec50_for_target_effect` | function | Inverse Emax: concentration reaching a target effect. |
| `effect_table` | function | Evaluate the sigmoid Emax model over a concentration grid. |
| `elimination_rate_constant` | function | Elimination rate constant ``ke = ln(2)/t½`` (1/hours). |
| `emax_effect` | function | Simple (hyperbolic) Emax model: ``E = E0 + Emax·C/(EC50 + C)``. |
| `expand_events` | function | Materialize ``addl``/``until`` rows into an explicit event list. |
| `filter_approved_drugs` | function | Collect the drug names whose regulatory approval matches a status. |
| `gfr_maturation` | function | Glomerular filtration rate (L/min) from post-menstrual age (PK-Sim model). |
| `half_life` | function | Elimination half-life ``t½ = ln(2)/ke`` (hours). |
| `infusion_rate` | function | Constant IV infusion rate achieving a steady state: ``R = CL·Css``. |
| `infusion_windows` | function | Zero-order infusion windows ``(start, end, cmt, rate)`` for rate > 0 records. |
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
| `pk_sensitivity` | function | PK parameter elasticities by the OSPSuite-R sensitivity protocol. |
| `renal_dose_adjustment` | function | Linearly scale a dose with renal function (creatinine clearance). |
| `sigmoid_emax_effect` | function | Sigmoid (Hill) Emax model: ``E = E0 + Emax·C^h/(EC50^h + C^h)``. |
| `steady_state_amounts` | function | Periodic steady-state amounts for repeated bolus dosing (mrgsolve steady_bolus). |
| `tdm_trough_adjustment` | function | Proportionally adjust a dose from a measured trough level. |
| `validate_concentration_series` | function | Validate a paired time/concentration series for PK math. |
| `vd_from_dose_auc` | function | Dose/AUC ratio — and, with ``mrt``, the steady-state volume. |
| `volume_of_distribution` | function | Apparent volume of distribution ``Vd = CL/ke``. |
| `youngs_rule` | function | Young's rule for paediatric dose scaling (age 1–12). |

## `pharmaml.docking`

*Molecular-docking workflows: preparation, execution, parsing, scoring*

| Symbol | Kind | Summary |
|---|---|---|
| `CARBOXYLATE_OXYGENS` | constant |  |
| `DEFAULT_MGL_PYTHON` | constant |  |
| `DEFAULT_PREPARE_LIGAND4` | constant |  |
| `DOCKSTRING_OBJECTIVES` | constant |  |
| `DOCKSTRING_TARGETS` | constant |  |
| `INTERACTION_CLASSES` | constant |  |
| `UNSUPPORTED_LIGAND_ATOMS` | constant |  |
| `AutomorphismCache` | class | spyrmsd's pose-list economy: enumerate the automorphisms once. |
| `ContactAtom` | class | One ATOM/HETATM record, parsed for contact geometry. |
| `ContactRecord` | class | One ligand-receptor atom pair within the cutoff, classified. |
| `DockingTarget` | class | One DOCKSTRING receptor name and Vina search box (angstrom). |
| `GridBox` | class | Docking search box geometry. |
| `GridParameters` | class | Parsed AutoDock4 grid parameter file (GPF) content. |
| `LigandLibrary` | class | A validated ligand library loaded from a table file. |
| `LigandPreparationReport` | class | Audit of one :func:`prepare_ligand_library` batch. |
| `PocketReport` | class | The outcome of :func:`extract_pocket` / :func:`crop_beyond`. |
| `PoseEvaluationReport` | class | DiffDock's ``evaluate.py`` per-complex metrics, one pose set. |
| `ProteinReport` | class | Audit of one :func:`prepare_protein` run. |
| `TorsionCheck` | class | Torsion-consistency audit of one prepared ligand PDBQT. |
| `VinaBatchProgress` | class | Pre-flight resume census for :func:`run_vina_batch`. |
| `batch_run_directories` | function | Dock every prepared ligand sub-directory in parallel. |
| `best_pose_rmsd` | function | DiffDock's evaluation loop: one reference vs *many* poses. |
| `centroid_distance` | function | Euclidean distance between the two geometries' centroids. |
| `check_docking_completion` | function | Audit a docking tree for missing/failed runs (upgrades checker.py). |
| `check_torsion_consistency` | function | Compare TORSDOF, active REMARKs and BRANCH counts of a PDBQT. |
| `classify_interaction` | function | Classify one ligand-receptor atom pair by distance geometry. |
| `cleanup_tree` | function | Delete every file not on the allowlist, bottom-up. |
| `collect_files` | function | Collect matching files from a campaign tree into one folder. |
| `compute_contacts` | function | All ligand-receptor atom pairs within ``cutoff``, classified. |
| `compute_grid_box` | function | Compute a grid box enclosing a reference ligand plus padding. |
| `consensus_breakdown` | function | Explain a consensus ranking: per-column units plus the score. |
| `consensus_rescore` | function | Combine several docking evidence columns into a 0–1 consensus. |
| `contact_matrix` | function | Minimum-distance grid: receptor residues x ligand atoms. |
| `convert_pdb_folder_to_pdbqt` | function | Batch-convert a folder of PDB files to PDBQT via MGLTools. |
| `crop_beyond` | function | DiffDock's ``utils/utils.py::crop_beyond``: the residue-level crop. |
| `delete_named_files` | function | Delete every file named ``filename`` anywhere under ``directory``. |
| `dihedrals` | function | The torsion angles (radians) around :func:`torsion_bonds`. |
| `distribute_files` | function | Copy each file into every immediate sub-directory of ``target_dir``. |
| `distribute_receptor` | function | Copy a prepared receptor into every ligand folder of a campaign. |
| `dock_smiles_array` | function | Dock every SMILES of an array against one receptor, simultaneously. |
| `engine_argv` | function | Resolve an engine and validate its launch prefix; no execution occurs. |
| `evaluate_objective` | function | Evaluate one named original objective from precomputed docking scores. |
| `extract_pdbqt_atom_types` | function | Collect sorted unique AutoDock atom types from a PDBQT ligand. |
| `extract_pocket` | function | The MOAD pocket selection: residues near the ligand. |
| `f2_objective` | function | Original F2 optimization objective (lower is better). |
| `generate_ligand_structures` | function | Generate 3D ligand structures (PDB + MOL2) from a SMILES library. |
| `get_target` | function | Case-sensitive upstream target lookup with an informative refusal. |
| `hungarian_rmsd` | function | The Hungarian-assignment RMSD (``spyrmsd/hungarian.py``). |
| `kabsch_rmsd` | function | The RMSD after the optimal Kabsch superposition. |
| `kabsch_transform` | function | The Kabsch rigid transformation superposing A onto B. |
| `ligand_efficiency` | function | Ligand efficiency ``LE = |ΔG| / N_heavy`` (kcal/mol per atom). |
| `ligand_efficiency_series` | function | Vectorised ligand-efficiency column for a docking summary table. |
| `list_file_names` | function | List the stems of all files with ``extension`` in a directory. |
| `load_ligand_csv` | function | Load a ligand library from CSV/TSV (or Excel). |
| `locate_engine_binary` | function | Locate an engine, but do not claim it is runnable on the current host. |
| `min_intra_distance` | function | The minimum pairwise atom distance (the self-clash metric). |
| `parse_dlg_cluster_table` | function | Parse the ``CLUSTERING HISTOGRAM`` table of an AutoDock4 DLG. |
| `parse_dlg_summary` | function | Summarise one DLG: best cluster, most populated cluster, extremes. |
| `parse_dockstring_search_box` | function | Parse a DOCKSTRING/Vina six-key config into an unnamed target box. |
| `parse_vina_config` | function | Parse an AutoDock Vina config file into a ``key -> value`` dict. |
| `parse_vina_output` | function | Parse the mode table of a Vina output (stdout log or PDBQT REMARKs). |
| `perturb_torsions` | function | Apply torsion updates to a geometry (``modify_conformer_torsion_angles``). |
| `plain_rmsd` | function | The plain, no-superposition RMSD (``spyrmsd.rmsd``). |
| `plif_bits` | function | The 0/1 fingerprint twin of :func:`plif_matrix`. |
| `plif_matrix` | function | Per-residue interaction-class counts (the PLIF table). |
| `pose_clusters` | function | Greedy RMSD clustering of poses (the confidence approximation). |
| `pose_evaluation_report` | function | The full DiffDock evaluation on one reference vs N poses. |
| `pose_rmsd_matrix` | function | The pairwise in-place (or Kabsch) RMSD matrix of a pose set. |
| `prepare_ligand_library` | function | Prepare every ligand of a CSV library into a campaign tree. |
| `prepare_protein` | function | Import a protein and prepare a docking-ready receptor PDBQT. |
| `promiscuous_ppar_objective` | function | Worst (maximum) PPAR score plus QED penalty; any NaN propagates. |
| `qcp_rmsd` | function | The QCP RMSD (Theobald 2005, ``spyrmsd/qcp.py`` verbatim). |
| `qed_penalty` | function | DOCKSTRING's ``10 * (1-QED)`` optimization penalty. |
| `randomize_pose` | function | DiffDock's ``randomize_position``: the diffusion initial state. |
| `read_gpf` | function | Parse an AutoDock4 grid parameter file (GPF) into a dict. |
| `read_pdb_atoms` | function | ATOM/HETATM records of a plain PDB text or file. |
| `read_pdbqt_atoms` | function | ATOM/HETATM records of a PDBQT text or file, one MODEL at a time. |
| `reindex_by_number` | function | Reindex a frame onto the full ligand-number range. |
| `rmsd_wrapper` | function | spyrmsd's ``rmsdwrapper``: one reference vs a list of molecules. |
| `rotatable_masks` | function | The bonds and their rotation masks (``get_transformation_mask``). |
| `run_autodock4` | function | Run autogrid4 + autodock4 for one ligand directory. |
| `run_vina` | function | Run a single AutoDock Vina docking. |
| `run_vina_batch` | function | Dock every ``.pdbqt`` under a directory with one Vina config. |
| `scan_docking_directory` | function | Build the docking summary table for every sub-directory. |
| `scan_gpf_ligand_types` | function | Report the ``ligand_types`` of every sub-directory's ``r.gpf``. |
| `scan_torsion_consistency` | function | Audit every ``.pdbqt`` under a tree for torsion consistency. |
| `scan_vina_directory` | function | Summarise every Vina ``.log`` under a results tree. |
| `selective_jak2_objective` | function | Corrected selective-JAK2 objective from upstream ``original.py``. |
| `smiles_to_pdbqt` | function | Convert one SMILES to a 3D PDBQT ligand (Meeko + RDKit). |
| `symmetry_corrected_rmsd` | function | The symmetry-corrected RMSD (spyrmsd's ``symmrmsd``). |
| `target_frame` | function | The target registry as a flat table suitable for export/display. |
| `torsion_bonds` | function | DiffDock's rotatable bonds (``get_transformation_mask`` rule). |
| `vina_batch_progress` | function | Count what a resumed Vina batch would skip and what it would dock. |
| `write_config_batch` | function | Write the ``r.gpf`` + ``r.dpf`` pair into every ligand folder. |
| `write_dpf` | function | Write an AutoDock4 docking parameter file (DPF) for one ligand. |
| `write_gpf` | function | Write an AutoDock4 grid parameter file (GPF) for one ligand. |

## `pharmaml.md`

*Molecular-dynamics analysis — trajectories, metrics, interactions.*

| Symbol | Kind | Summary |
|---|---|---|
| `KJ_TO_KCAL` | constant |  |
| `METRIC_FILES` | constant |  |
| `CVTable` | class | The ``COLVAR`` equivalent: named CV series on one time axis. |
| `InteractionFingerprint` | class | One pose's interaction record: counts, bits and residue pairs. |
| `LigandResult` | class | One ligand's parsed metric series, keyed by metric key. |
| `MDStudy` | class | A whole MD campaign: one :class:`LigandResult` per ligand folder. |
| `MetricInfo` | class | Display metadata for one standard MD metric export. |
| `ProjectionResult` | class | A fitted linear projection: mean, components, explained variance. |
| `Topology` | class | Per-atom metadata: names, residues and optional masses. |
| `Trajectory` | class | A stack of frames: ``(n_frames, n_atoms, 3)`` coordinates + topology. |
| `XvgSeries` | class | One parsed metric series: a 1-D time axis plus 1-D values. |
| `anm_fluctuations` | function | Per-atom ANM fluctuations summed over every non-trivial mode. |
| `anm_hessian` | function | The 3N×3N anisotropic network Hessian. |
| `anm_modes` | function | The slowest non-trivial ANM modes: ``(eigenvalues, mode vectors)``. |
| `build_study` | function | Build the campaign study from a root folder. |
| `center_of_mass` | function | Geometric centre, or centre of mass when weights are given. |
| `classify_metric` | function | Map an export filename to its :class:`MetricInfo`, or ``None``. |
| `contact_map` | function | Symmetric contact-frequency matrix over the trajectory. |
| `cross_correlation` | function | Normalised GNM fluctuation cross-correlation matrix. |
| `cv_angle` | function | ``ANGLE`` — the angle at the middle atom per frame (degrees). |
| `cv_coordination` | function | ``COORDINATION`` — the fraction of groupA–groupB pairs within ``R_0``. |
| `cv_dihedral` | function | ``DIHEDRAL`` — the signed torsion per frame (degrees). |
| `cv_distance` | function | ``DISTANCE`` — the two-atom separation per frame (Å). |
| `cv_gyration` | function | ``GYRATION`` — the per-frame radius of gyration (Å). |
| `cv_rmsd` | function | ``RMSD`` — per-frame RMSD (Å) after optimal superposition. |
| `cv_table` | function | Assemble named CV series onto one validated time axis. |
| `dihedral` | function | Torsion angle (degrees) of the bond ``p1–p2`` for atoms ``p0..p3``. |
| `find_hbonds` | function | Detect one frame's H-bonds: (donor, acceptor) index pairs, sorted. |
| `gnm_fluctuations` | function | Per-atom GNM mean-square fluctuations (relative, kT/γ units). |
| `hbond_timeline` | function | Per-frame H-bond counts — the ``hbnum`` series over a trajectory. |
| `interaction_fingerprint` | function | Compute a pose's protein–ligand interaction fingerprint. |
| `kabsch_rotation` | function | Optimal rotation ``R`` with ``(mobile − mean)·R ≈ (reference − mean)``. |
| `kirchhoff_matrix` | function | The GNM Kirchhoff matrix over the contact graph. |
| `metric_correlations` | function | Correlation matrix across one ligand's metric series. |
| `metric_stats` | function | Mean, population standard deviation and coefficient of variation. |
| `moving_average` | function | Centered moving average (``np.convolve`` ``same`` mode). |
| `native_contact_fraction` | function | Per-frame fraction of native contacts kept (the Q-value). |
| `pairwise_distances` | function | Full ``(n, n)`` distance matrix for one frame. |
| `parse_ndx_hbnum` | function | Parse an ``.ndx``-style H-bond count table into an :class:`XvgSeries`. |
| `parse_plumed_cvs` | function | Parse a ``plumed.dat`` snippet and compute its CVs over ``traj``. |
| `parse_xvg` | function | Parse GROMACS ``.xvg`` text (or a file path) into an :class:`XvgSeries`. |
| `ps_to_ns` | function | Convert a ps time axis to ns. |
| `radial_distribution_function` | function | Pair radial distribution function g(r) between two point sets. |
| `radius_of_gyration` | function | Radius of gyration (Å): mass-weighted RMS distance to the centre. |
| `radius_of_gyration_trajectory` | function | Per-frame radius of gyration (``gmx gyrate``). |
| `read_ligand_folder` | function | Parse one ligand folder's GROMACS/MM-PBSA exports. |
| `resolve_atom_indices` | function | Normalise an optional atom selection into a validated index array. |
| `rmsd` | function | RMSD after optimal superposition (Å, per-atom root mean square). |
| `rmsd_trajectory` | function | Per-frame RMSD (Å) to the reference, after optimal superposition. |
| `rmsf` | function | Per-atom root-mean-square fluctuation (Å) about the aligned mean. |
| `salt_bridge_timeline` | function | Per-frame salt-bridge counts — the ionic-contact series. |
| `salt_bridges` | function | Charged-opposite pairs within ``cutoff`` Å — the salt-bridge list. |
| `superpose` | function | Return the mobile coordinates optimally fitted onto the reference. |
| `synthetic_hbond_trajectory` | function | Deterministic H-bond demo: two donor–H···acceptor pairs that form |
| `synthetic_salt_bridge_trajectory` | function | Deterministic salt-bridge demo: two charged pairs that form and |
| `synthetic_trajectory` | function | Deterministic demo trajectory: an alpha-helix with a flexible loop. |
| `tica` | function | Time-lagged independent component analysis of a ``(T, D)`` matrix. |
| `to_kcal` | function | Convert kJ/mol → kcal/mol (the uploaded binding-energy conversion). |
| `trajectory_pca` | function | PCA of the trajectory coordinates (essential dynamics). |

## `pharmaml.viz`

*Visual output for every PharmaML domain — plots, tables, HTML reports.*

| Symbol | Kind | Summary |
|---|---|---|
| `EXPORT_EXTENSIONS` | constant |  |
| `EXPORT_FILTERS` | constant |  |
| `PALETTE` | constant |  |
| `PDF_THEMES` | constant |  |
| `PLOT_FORMATS` | constant |  |
| `REPORT_THEMES` | constant |  |
| `TABLE_FORMATS` | constant |  |
| `TIFF_COMPRESSIONS` | constant |  |
| `ExportFormat` | class | The four first-class raster/vector export formats (§17.1). |
| `ExportManager` | class | Batch export with the §17.4 manager contract. |
| `ExportResult` | class | What an export actually produced -- the honest report card. |
| `HTMLReport` | class | Builder for a self-contained HTML report. |
| `ImageExportOptions` | class | Per-export configuration (§17.4's reusable export manager knobs). |
| `close_figure` | function | Close a figure defensively (no-op when matplotlib is absent). |
| `export_docx` | function | Write a DataFrame as a Word ``.docx`` document with a grid table. |
| `export_figure` | function | Export a matplotlib figure to any §17 format. |
| `export_image` | function | Export an image (array / PIL image / existing file) to any §17 format. |
| `export_matrix` | function | Export a scientific matrix -- the §17.2-specialised path. |
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
| `plot_contact_map` | function | Residue × ligand-atom minimum-distance grid of a docked pose. |
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
| `plot_mcts_introspection` | function | One figure per MCTS run: score story, search growth, dedup economy. |
| `plot_md_correlation` | function | Cross-metric Pearson heatmap — the uploaded script's closing panel. |
| `plot_md_series` | function | Line chart of parsed MD metrics — the uploaded panel, typed. |
| `plot_model_comparison` | function | Leaderboard as horizontal bars with per-fold std error bars. |
| `plot_operator_trajectory` | function | One figure per run: score trajectory + the operator mix per generation. |
| `plot_optimization_history` | function | The objective per trial + the best-so-far step line (100x-160). |
| `plot_parallel_coordinate` | function | One polyline per trial across the params + objective (100x-160). |
| `plot_param_importances` | function | The MDI hyperparameter-importance bars (100x-160). |
| `plot_pareto_front` | function | One scatter per MultiObjective run: dominated points muted, front lit. |
| `plot_pca_variance` | function | Explained-variance ladder -- the "how many components?" answer. |
| `plot_permutation_heatmap` | function | Cross-model permutation-importance heatmap (features × models). |
| `plot_permutation_importance` | function | Top-N permutation importances as signed bars with error bars. |
| `plot_plif_heatmap` | function | Residue × interaction-class PLIF counts (or the 0/1 twin). |
| `plot_pr_curves` | function | Precision-Recall curves — binary or one-vs-rest multiclass. |
| `plot_predicted_vs_actual` | function | Parity scatter for regression, with the identity reference line. |
| `plot_prediction_intervals` | function | Parity scatter with the quantile-regression prediction band. |
| `plot_receptor_atom_types` | function | Bar chart of the AutoDock atom-type census of a receptor. |
| `plot_residuals` | function | Residuals vs predicted (left) and residual histogram (right). |
| `plot_rl_introspection` | function | One figure per RL run: the score story + the prior-drift diagnostics. |
| `plot_roc_curves` | function | Receiver-operating curves — binary or one-vs-rest multiclass. |
| `plot_sa_distribution` | function | Raw Ertl SA-score histogram for one generation run (section 24.3). |
| `plot_selfies_robustness` | function | The robustness experiment's two validity curves, one figure. |
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
| `plot_study_pareto` | function | Plot completed trials and the full-dimensional Pareto front. |
| `plot_table_comparison` | function | Overlap breakdown of a ligand-table :class:`TableComparison`. |
| `plot_timeline` | function | Per-trial duration bars coloured by state (100x-160). |
| `plot_top_ligands` | function | Horizontal bars of the strongest-binding ligands. |
| `plot_training_history` | function | Epoch-by-epoch loss curves with the early-stopping verdict. |
| `require_matplotlib` | function | Import matplotlib lazily and pick the right backend. |
| `save_figure` | function | Save a figure to ``png``/``svg``/``pdf`` (chosen by extension). |
| `save_pdf_report` | function | Write a multi-page PDF document report (metrics, tables, figures). |
| `save_visual_bundle` | function | Render figures + tables + metrics into one directory + HTML report. |
| `sniff_format` | function | Identify an image file from its magic bytes alone. |
| `validate_export` | function | Reopen an exported file and verify it kept its promises. |

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

## `pharmaml.utils.checkpoints`

*Durable checkpoints for long-running jobs (MASTER_PROMPT Part-3 §20).*

| Symbol | Kind | Summary |
|---|---|---|
| `Any` | class | Special type indicating an unconstrained type. |
| `CheckpointInfo` | class | One validated checkpoint record (the contents of one JSON file). |
| `CheckpointManager` | class | Write, validate, rotate and clear checkpoint files for one job. |
| `ENVELOPE_KEY` | constant |  |
| `OperationSpec` | class | The identity of one resumable job (§13's "complete state" header). |
| `Path` | class | PurePath subclass that can make system calls. |
| `PersistenceError` | class | Raised when saving/loading model artifacts fails. |
| `SCHEMA` | constant |  |
| `annotations` | constant |  |
| `dataclass` | function | Add dunder methods based on the fields defined in the class. |
| `default_directory` | function | The GUI's per-job checkpoint home: ``~/.pharmaml/checkpoints/<name>/``. |
| `hashlib` | module | hashlib module - A common interface to many hash functions. |
| `json` | module | JSON (JavaScript Object Notation) <https://json.org> is a subset of |
| `os` | module | OS routines for NT or Posix depending on what system we're on. |
| `time` | module | This module provides various functions to manipulate time values. |

## Command-line interface

```text
pharmaml models | data inspect | data split | data convert | ml train
             | ml compare | chem pic50 | chem descriptors | pharma pk
             | pharma dose | docking prep-ligands | docking prep-protein
             | docking scan | docking check | generative run
             | generative corpus
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
