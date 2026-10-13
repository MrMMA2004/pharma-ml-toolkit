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
| `molar_mass` | constant |  |
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

## `pharmaml.utils.io`

*Safe tabular I/O helpers*

| Symbol | Kind | Summary |
|---|---|---|

## `pharmaml.utils.validation`

*Shared input-validation layer*

| Symbol | Kind | Summary |
|---|---|---|

## Command-line interface

```text
pharmaml models | ml train | ml compare | chem pic50 | chem descriptors
             | pharma pk | pharma dose | docking scan | docking check
```

See `pharmaml/cli.py` for every flag, and `README.md` for the full manual.
