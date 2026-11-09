# RAR → PharmaML Coverage Matrix

Every file inside the original `کدهای ماشین لرنینگ.rar` (118 recovered
files), audited **one by one** and mapped to the PharmaML surface that
upgrades it. Status legend:

- ✅ **covered** — the rar script's core functionality ships in the
  toolkit (library function, CLI command, or notebook stage), hardened
  and typed.
- ✚ **covered (this audit)** — a gap found by the one-by-one audit and
  closed in this release (see CHANGELOG).
- ⚠ **absorbed** — the script's *behaviour* is covered by a strictly
  better path; the literal file format/output is not reproduced
  (explained per row).

The rar's near-duplicate layout (a script per estimator, a copy per
folder) is the point of the upgrade: one registry, one pipeline.

## Classification (`Classification/`)

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `RF.py` | `ml.build_model('random_forest')` + `train_model` (GridSearchCV, stratified split) + `evaluate_model` + `ml.persistence.save_model`; viz: `plot_confusion_matrix`, `plot_roc_curves`; the fpdf PDF report → `viz.save_pdf_report` (title + metrics, one page per table, one page per figure; `save_visual_bundle(..., pdf_report=True)`); CLI `ml train` | ✅ |
| `SVM.py` | `ml.build_model('svm')` (probability=True path pinned by tests) | ✅ |
| `XGboost.py` | `ml.build_model('xgboost')` (optional-gated; string class labels handled by the `_IntegerLabelClassifier` wrapper since 1.2.44 — the registry tour found xgboost was the one family refusing them) | ✅ |
| `LightGBM.py` | `ml.build_model('lightgbm')` | ✅ |
| `CATboost.py` | `ml.build_model('catboost')` (verbose=0) | ✅ |
| `ADAboost.py` | `ml.build_model('adaboost')` (determinism pinned) | ✅ |
| `KNN.py` | `ml.build_model('knn')` | ✅ |
| `Decision tree.py` | `ml.build_model('decision_tree')` | ✅ |
| `Logistic regression.py` | `ml.build_model('logistic_regression')` | ✅ |
| `Ridge.py` | `ml.build_model('ridge_classifier')` | ✅ |
| `Lasso.py` | `ml.build_model('lasso_logistic')` — L1 *logistic* classifier, liblinear | ✚ |
| `Elastic net.py` | `ml.build_model('elastic_net_logistic')` — elasticnet *logistic* classifier, saga + `l1_ratio` grid | ✚ |
| `ANN.py` | `ml.build_model('ann')` (MLP) and `'dnn'` / `'dnn_residual'` (PyTorch, optional) | ✅ |
| `Stacking.py` | `ml.build_stacking('classification', [...])` (LogisticRegression+KNN+SVC stack of the rar, any members) | ✅ |
| `Voting.py` | `ml.build_voting('classification', [...], voting='soft'/'hard', weights=...)` | ✅ |
| `ANN/ANN class.py` | same as `ANN.py` (class wrapper) → `train_model` returns the fitted pipeline + metadata | ✅ |
| `ANN/reuse.py` | `ml.persistence.load_model` (bundle: fitted estimator + provenance metadata) → `restored.predict(...)`; the rar's separate `_pipeline.joblib` + `_metadata.json` pair is one version-stamped artifact | ✅ |
| `Logestic Regression/Logistic regression class done.py` | `train_model` + `save_model` (combined bundle: pipeline + metadata in one step) | ✅ |
| `Logestic Regression/reuse.py` | `ml.persistence.load_model` → `restored.predict(...)` | ✅ |
| `Stacking/stacking class.py` | `build_stacking` + `train_model` | ✅ |
| `Stacking/reuse.py` | `ml.persistence` bundle path | ✅ |
| `Descriptor/Descriptor classification.py` | `chem.compute_descriptors` + `chem.compute_fingerprints` (incl. **avalon**, this audit) → `clean_numeric_frame` → `train_model`/`compare_models`; the full chain is the ML notebook §2 | ✚ |
| `Descriptor/Descriptor reuse.py` | cached-descriptor reuse → `compute_descriptors` once, then `train_model` on the cached frame + `persistence` | ✅ |
| `Descriptor/Descriptor reuse(اگه اونیکی کار نکرد).py` | same as above (fallback variant) | ✅ |
| `BBB/BBB_Prediction_model.ipynb` | the whole notebook is the ML pipeline: `compute_descriptors` (rdkit_all) → NaN handling (imputer in `build_preprocessor`) → stratified split → `compare_models(['logistic_regression','random_forest','xgboost'])` → tuning → `save_model`; ROC AUC in `ClassificationReport` | ✅ |
| `BBB/tuned_random_forest_bbb_prediction_model.pkl` | `ml.persistence.save_model` bundles (joblib + JSON sidecar, version-stamped) | ⚠ model weights are dataset-specific and intentionally not shipped |
| root `Descriptor classification.py` | same as `Classification/Descriptor/...` | ✚ |

## Regression (`Regression/`)

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `RF.py` | `ml.build_model('random_forest', 'regression')` + `RegressionReport` (R², MSE, MAE, RMSE), `plot_predicted_vs_actual` | ✅ |
| `SVM.py` / `SVM/SVM reg.py` / `SVM/reuse.py` | `ml.build_model('svm', 'regression')` + persistence | ✅ |
| `XGboost.py`, `LightGBM.py`, `CATboost.py`, `ADAboost.py`, `KNN.py`, `Decision tree.py` | registry twins of the classification rows | ✅ |
| `Linear regression.py` | `ml.build_model('linear_regression')` | ✅ |
| `Ridge.py` / `Lasso.py` / `Elastic net.py` / `Huber.py` | `build_model('ridge'/'lasso'/'elastic_net'/'huber_regression', 'regression')` | ✅ |
| `ANN.py` / `ANN2.py` | `build_model('ann'/'dnn'/'dnn_residual', 'regression')` | ✅ |
| `Stacking.py` / `Stacking/Stacking reg.py` / `Stacking/reuse.py` | `build_stacking('regression', [...])` | ✅ |
| `Voting.py` / `Voting/Voting reg.py` / `Voting/reuse.py` | `build_voting('regression', [...])` (VotingRegressor) | ✅ |
| `Descriptor/Descriptor regression.py` / `Descriptor reuse.py` | descriptor+fingerprints → regression pipeline (as the classification twin) | ✚ |

## Chemistry & data utilities (root, `others/Ml/`, `others/General/`)

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `Convert IC50(nM) to pIC50.py` | `chem.parse_endpoint` (strips `<`/`>` qualifiers, never fabricates) + `chem.ic50_to_pic50` + `add_pic50_column`; `pharma.units` for the unit math; CLI `chem pic50` | `ic50_to_pic50`×2 `add_pic50_column`×2 |
| `Convert CSV to SDF.py` | `chem.table_to_sdf` — AddHs → ETKDG (fixed seed) → UFF, every column becomes an SD tag, invalid SMILES counted not lost | `table_to_sdf`×2 |
| `Convert SMI to CSV.py` | `chem.smi_to_csv` (streaming, BOM for Excel) — also `others/General/Convert SMI to CSV.py` | `smi_to_csv`×2 |
| `others/Ml/Sdf reading.ipynb` | `chem.sdf_to_table` (tags → columns; the notebook's PandasTools.LoadSDF path) | `sdf_to_table`×2 |
| `others/Ml/Descriptor calculator.py` / `Descriptoredited.py` | `chem.compute_descriptors(descriptor_set='rdkit_all')` | ✅ |
| `others/Ml/Mordred_calculation.py` | `chem.compute_descriptors(descriptor_set='mordred')` — Error/Missing → NaN with an aggregate warning (100x-52) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `others/Ml/ECFP_calculation.py` | `chem.compute_fingerprints(kind='morgan')` (ECFP-style bits, radius & n_bits) | `compute_fingerprints`×5 |
| `others/Ml/Morgan_calculator.py` | `chem.compute_fingerprints(kind='morgan', as_counts=True/False)` | `compute_fingerprints`×5 |
| `Tanimoto calculator.py` / `others/Ml/Tanimoto calculator.py` | `chem.tanimoto_matrix` (ref × target) + `similarity_scan` (long format, threshold) + `top_matches` | ✅ |
| `Tanimoto-edited.py` / `others/Ml/Tanimotoedited.py` | the chunked/threshold variant → `similarity_scan(threshold=...)` | ✅ |
| `Non-numerical data remover.py` | `ml.clean_numeric_frame` (target/keep_cols survive, dropped names logged); cell-level coercion is the mordred NaN path + `biology.normalize_expression` | `clean_numeric_frame`×2 |
| `Split CSV file.py` / `others/General/Split CSV file.py` | `chem.split_table_into_chunks` (flat memory, explicit output dir) | ✅ |
| `others/General/CSV_merger.py` | `chem.merge_tables(first_columns=[...])` (the column-ordering contract) | `merge_tables`×2 |
| `others/General/filemerger.py` | `chem.merge_table_folder` (CSV/Excel folder → one file); CLI `chem merge` | `merge_table_folder`×2 |
| `others/General/common_between_two_excel.py` | `chem.compare_tables(...).common` / `.only_in_a` / `.only_in_b` (name-normalized set operations; TableComparison) | `compare_tables`×2 |
| `others/compare/notcommon.py` | `chem.compare_tables(...).only_in_a / .only_in_b` differences report; CLI `chem compare` | `compare_tables`×2 |
| `others/compare/smilesgetter.py` | `chem.annotate_with_smiles` (join SMILES onto a ligand-name column, extension/case normalization) | `annotate_with_smiles`×2 |
| `others/General/approved.py` | `pharma.filter_approved_drugs` (17 TSVs → approved names) + `viz.export_docx` (any table as a real Word document — the rar wrote a bare .docx name list) | `filter_approved_drugs`×2 `export_docx`×3 |
| `others/General/Nameextracter.py` | `docking.list_file_names` (file inventory → table) | `list_file_names`×2 |
| `others/General/adder.py` | `docking.distribute_files` + `docking.distribute_receptor` (copy binaries/receptor into every campaign subfolder) | `distribute_files`×2 |
| `others/General/delete.py` | `docking.cleanup_tree` (keep-list + pattern, dry-run default) | `cleanup_tree`×2 |
| `others/General/deleter.py` | `docking.delete_named_files` | `delete_named_files`×2 |
| `others/General/extaacter.py` | `docking.collect_files(pattern='r.dlg')` with unique-name resolution | `collect_files`×2 |
| `others/General/Prettier_extracted_dlgs.py` / `others/Prettier_extracted_dlgs.py` / `others/Docking/Prettier_extracted_dlgs.py` | `docking.reindex_by_number` (align sparse per-folder rows onto a dense 0..N index) | ✅ |
| `others/General/reader_at_same_folder.py` / `others/reader_at_same_folder.py` | `docking.scan_docking_directory` (single-level scan) | ✅ |
| `others/Ml/New Text Document.txt`, `others/Docking/New Text Document.txt` | empty placeholders | ⚠ nothing to cover |

## Docking (`others/Docking/`, `others/`)

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `main.py` / `others/main.py` | `docking.batch_run_directories` (autogrid4 → autodock4 per campaign folder, idempotent resume, ProcessPool); `workflow.run_autodock4` | ✅ |
| `checker.py` | `docking.check_docking_completion` + `scan_docking_directory` ("Successful Completion" audit) | ✅ |
| `gpfer.py` / `others/gpfer.py` | `docking.write_gpf` (+ `extract_pdbqt_atom_types`, `compute_grid_box` for the center) | ✅ |
| `gpferreceptorni.py` | `write_gpf` with an explicit `receptor_types` list (Ni-type receptors included) | ✅ |
| `gpfreader.py` | `docking.read_gpf` + `scan_gpf_ligand_types` (per-folder ligand_types summary) | ✅ |
| `dpfer.py` / `others/dpfer.py` | `docking.write_dpf` (DPF from ligand atom types, TORSDOF, coords) | ✅ |
| `reader.py` / `readerwithdetails.py` / `reader_at_same_folder.py` / others | `docking.parse_dlg_cluster_table` + `parse_dlg_summary` (cluster rank / lowest / mean / num-in-cluster; the top-cluster highlight) | ✅ |
| `others/Reader for alphabetic folders/dlg_extracter_not_numeral.py` / `readerwithdetails.py` | `scan_docking_directory` accepts non-numeric folder names (pattern-based, not `\d+`-only) | ✅ |
| `others/dlg_extracter.py` | `collect_files(pattern='r.dlg')` | `collect_files`×2 |
| `smilestopdbqt.py` / `smiqt.py` | `docking.smiles_to_pdbqt` (Meeko, AD4 atom types) | ✅ |
| `ligand_preparation_autodock4.py` / `others/ligand_preparation_autodock4.py` | `docking.prepare_ligand_library` (CSV → campaign folders → PDBQT with torsion reports) | ✅ |
| `ligand_pdbqt_extracter.py` | `collect_files` (numbered .pdbqt collection with unique-name resolution) | ✅ |
| `ligand generation/niggaqt.py` | `prepare_ligand_library` (the modern Meeko path replaces the MGLTools pythonsh subprocess; PDB → PDBQT) | ⚠ MGLTools itself is deprecated tooling — Meeko is the supported path |
| `ligand generation/nigga.py` / `nigga2.py` | `prepare_ligand_library` + `chem.table_to_sdf` (multi-conformer 3D: ETKDG embeds + UFF, unsupported-metal filtering via validation notes) | ⚠ mol2 output not produced (PDBQT/SDF are the docking formats) |
| `ligand generation/niggamatch.py` / `mismatch2.py` | `docking.check_torsion_consistency` + `scan_torsion_consistency` (TORSDOF vs active REMARKs vs BRANCHes) | ✅ |

## Vina (`others/Vina/`)

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `vina_reader.py` | `docking.parse_vina_output` (best affinity per log) + `scan_vina_directory` | ✅ |
| `vinar.py` | `docking.run_vina` (config-driven single docking) + `parse_vina_config` (receptor/grid params from conf) | ✅ |
| `vinar_multiple.py` | `docking.run_vina_batch` (folder of ligand PDBQTs, per-ligand logs) | ✅ |

## Rescoring & shared

| Rar file | PharmaML surface | Status |
| --- | --- | --- |
| `others/Readme.txt` | `README.md` (the toolkit's own documentation) | ✅ |

---

## Audit summary

- **118 files** audited one by one (112 `.py`, 2 `.ipynb`, 1 `.pkl`, 3 `.txt`);
  every file has a row above.
- **Two functional gaps found and closed in 1.2.23**: penalized
  *logistic* classification (the rar's `Classification/Lasso.py` and
  `Classification/Elastic net.py` — 2 files) gained the registry entries
  `lasso_logistic` and `elastic_net_logistic`; Avalon fingerprints (the
  rar's Descriptor wrappers — 7 files) gained the `avalon` kind on
  `compute_fingerprints`, bit and count flavours.
- **Two more "absorbed" outputs became first-class in 1.2.24**: the
  fpdf **PDF model report** (every rar model script's signature output)
  is now `viz.save_pdf_report` + `save_visual_bundle(pdf_report=True)`,
  and the **.docx export** (the rar's `approved.py`) is now
  `viz.export_docx` — fpdf was declared in the `reports` extra since
  1.0 but never wired to a function until the audit asked what the ZIP
  was missing.
- **Three rows intentionally absorbed** (dataset-specific `.pkl`
  weights, `.mol2` output — needs Open Babel, and the deprecated
  MGLTools bridge; plus the empty placeholders, which contain nothing)
  — each with the reason stated in its row.
- The near-duplicate explosion (≈30 model scripts × 2 tasks + folder
  copies) collapses into **24 registry models + 2 ensemble builders**
  with one training/evaluation/persistence pipeline.
- **One family beyond the rar arrived in 1.2.41**: `extra_trees` —
  extremely randomized trees (random thresholds on the full sample,
  `bootstrap=False`), classification **and** regression, aliases
  `et` / `ert` / `extratrees`. It has no rar original; it is the
  classic tree ensemble the rar never shipped, and it slots into the
  same registry/tuning/persistence pipeline unchanged.
- **A second family beyond the rar arrived in 1.2.42**:
  `gradient_boosting` — Friedman's classic GBM (sklearn's
  `GradientBoostingClassifier` / `GradientBoostingRegressor`, aliases
  `gb` / `gradientboosting` / `friedman_gbm` / `sk_gbm`),
  classification **and** regression. The rar reached for gradient
  boosting only through the engineered implementations
  (`XGboost.py` / `LightGBM.py`); the sklearn ancestor — the ancestor
  the web playground has mirrored all along — was missing from the
  registry until now. The `gbm` alias stays lightgbm's.
- **A third family beyond the rar arrived in 1.2.43**: `pls` —
  partial least squares regression (sklearn's `PLSRegression`,
  NIPALS, aliases `plsr` / `pls_regression` /
  `partial_least_squares`), **regression only** — the registry
  refuses the classification side with the standard typed error.
  It has no rar original, yet no QSAR toolkit ships without it:
  latent directions that maximize descriptor–target covariance are
  the classic answer to wide, collinear descriptor blocks with few
  observations. The web playground's hand-rolled NIPALS mirror was
  verified against sklearn to machine epsilon (predictions 2.2e-15,
  coefficients 1.1e-15) — the first family to arrive on both sides
  on the same day, with no mirror gap.

---

## Notebook & playground coverage (100x-101)

The second audit axis: not "does the toolkit cover the rar" but "do the
**notebooks and playground.py** cover the **toolkit**". Full matrix in
`docs/LEAKAGE_AUDIT.md` §E; the summary:

| Surface | Coverage | Guarantee |
| --- | --- | --- |
| 26 registry families in `playground.py` | all 26, dynamic | the model combobox lists `list_models(task, only_available=True)` — a new family joins with zero edits (pls appeared the day it landed, gaussian_process the day after) |
| every registry family's tunable knobs in `playground.py` (1.2.48) | all grids, dynamic | the "Manual hyperparameters" card builds its fields from the selected model's own `param_grid` + estimator defaults — a new knob joins with zero edits, and `train_model(model_params=...)` pins the edited ones (searching only the blank slots) |
| long-job progress in `playground.py` (1.2.48) | searches + epochs + leaderboards | `on_trial` / `on_epoch` / `on_model` callbacks feed the status bar's determinate progress + live ETA; zero-argument jobs keep the marquee |
| 26 registry families in the notebooks | all 26 as of 1.2.47 | ML workflow §11 "the full registry tour" walks `list_models()` for both tasks and fits every available family (39 entries) on the notebook's own splits, closing with the cumulative fit-budget row — self-updating, doubles as a full-registry smoke test |
| 9 module groups (`chem`, `ml`, `docking`, `pharma`, `biology`, `generative`, `viz`, `utils`, `data`) | each has a dedicated workflow notebook | ml + walkthrough + tuning (ml/chem), docking, pk (pharma), cellline (biology), generative, deep learning (ml.torch) |
| Ensemble builders (`build_stacking`, `build_voting`) | walkthrough + ml notebook | `compare_models`-adjacent cells |
| Optional engines (xgboost / lightgbm / catboost / torch) | notebook-conditional + tour-conditional | listed honestly when absent (`requires` in the tour table), fitted when present |

The 13 families that had **no** notebook demonstration before this
round (adaboost, catboost, elastic_net_logistic, extra_trees,
gradient_boosting, lasso, lasso_logistic, lightgbm, pls, ridge,
ridge_classifier, svm, xgboost) are all in §11's tour now — fitted,
scored on held-out rows, timed.

---

## Playground coverage (100x-110) — the standing guarantee

The third audit axis, asked for by the user and now a standing
contract: **every file of the rar must have a GUI surface in
`playground.py`**, not just a library/CLI one. The 118-file matrix
above maps rar → toolkit; this matrix maps toolkit → playground tab.
It is the checklist future rounds re-run whenever the playground
gains or moves a card.

| Rar workflow family (files) | playground.py surface since 1.2.50 |
| --- | --- |
| ~30 model scripts × 2 tasks (Classification/ + Regression/) | Data & ML → *Train a model* (26-family registry combobox, dynamic) + *Manual hyperparameters* + *Tuner* + *Results* + *Leaderboard* |
| `Stacking.py` (×5) | Data & ML → *Ensemble builder* (`build_stacking`) |
| `Voting.py` (×5) | Data & ML → *Ensemble builder* (`build_voting`, soft/hard + weights) |
| every `reuse.py` / `class done.py` (×8) | Data & ML → *Model persistence* (save bundle / load / audit / predict) |
| every model script's PDF report | Data & ML → *Save report…* + `include report.pdf` (`save_pdf_report` via the visual bundle) |
| `Descriptor *.py` (×7) + `others/Ml/*calculator*.py` (×6) | Chemistry → *Feature factory* (descriptor sets + fingerprint kinds, top-K ranking) |
| `BBB_Prediction_model.ipynb` | Data & ML end-to-end: load → train → tune → save |
| `Convert IC50(nM) to pIC50.py` | Chemistry → *Affinity conversion* (single value) **and** *Structure & table converters* → `append pIC50 column` (batch, censored-honest) |
| `Convert CSV to SDF.py` | Chemistry → *Structure & table converters* → `CSV table → 3D SDF` |
| `Convert SMI to CSV.py` (×2) | same card → `.smi list → CSV` |
| `others/Ml/Sdf reading.ipynb` | same card → `SDF → table` (+ `table → .smi list` inverse) |
| `Tanimoto calculator.py` (×2) | Chemistry → *Library similarity scan* → `matrix: references × library` |
| `Tanimoto-edited.py` (×2) | same card → `scan: pairs ≥ threshold` (and `top-k`) |
| `Non-numerical data remover.py` | Data & ML → *Drop non-numeric columns* + *Save cleaned CSV…* |
| `Split CSV file.py` (×2) | Utilities → *Table merge & split* → `split into chunks` |
| `CSV_merger.py` | same card → `merge picked files` (first-columns contract) |
| `filemerger.py` | same card → `merge whole folder` |
| `common_between_two_excel.py` + `notcommon.py` | Utilities → *Table compare & annotate* → `compare a column` |
| `smilesgetter.py` | same card → `annotate names with SMILES` |
| `approved.py` | Utilities → *Approved drugs → report* (CSV + `export_docx`) |
| `extaacter.py` / `dlg_extracter.py` / `ligand_pdbqt_extracter.py` | Utilities → *Campaign file tools* → `collect files` (name ∪ pattern) |
| `Nameextracter.py` | same card → `list file names` |
| `adder.py` | same card → `distribute files into folders` (and Docking → receptor card's *distribute*) |
| `deleter.py` | same card → `delete named files` |
| `delete.py` | same card → `cleanup tree (keep list)` |
| `Prettier_extracted_dlgs.py` (×3) | Utilities → *Reindex by number* |
| `others/General/reader_at_same_folder.py` (×2) + all `reader*.py` | Docking → *Docking results* (campaign scan / DLG cluster table / Vina table) |
| `main.py` (×2) | Docking → *Campaign runner* (`batch_run_directories`, autogrid4 + autodock4) |
| `checker.py` | Docking → *Grid, torsion & completion audit* → `folder: completion check` |
| `gpfer.py` / `gpferreceptorni.py` / `dpfer.py` (×4) | Docking → *Grid box & docking configs* (`write_gpf` + `write_dpf`) |
| `gpfreader.py` | *Grid, torsion & completion audit* → `GPF file → parameters` **and** `folder: ligand_types per sub-dir` |
| `niggamatch.py` / `mismatch2.py` | same card → `folder: torsion consistency` |
| `smilestopdbqt.py` / `smiqt.py` | Docking → *SMILES → PDBQT (single ligand)*; the array card + *Ligand library* cover the batch paths |
| `ligand_preparation_autodock4.py` (×2) | Docking → *Ligand library → PDBQT* (`prepare_ligand_library`) |
| `smiles array docking` (the 100x-108 request) | Docking → *SMILES array docking* (`dock_smiles_array`, both engines, simultaneous) |
| `vinar.py` | Docking → *Campaign runner* → `Vina single ligand` (`run_vina`) |
| `vinar_multiple.py` | Docking → *Vina config batch* (`run_vina_batch`, simultaneous workers) |
| `vina_reader.py` | Docking → *Docking results* → Vina mode (`parse_vina_output` / `scan_vina_directory`) |
| `others/Readme.txt` | `README.md` + the home tab's *What lives where* guide |
| `New Text Document.txt` (×2) | empty placeholders — nothing to cover |
| `tuned_random_forest_bbb_prediction_model.pkl` | ⚠ dataset-specific weights are not shipped (the *save bundle* card writes fresh ones) |
| `ligand generation/nigga*.py` MGLTools/`.mol2` paths | ⚠ deprecated tooling absorbed by the Meeko paths (see rows above) |

Audit method: the rar re-extracted to a clean tree (118 files, `unrar
x`), each file opened, its calls matched against
`grep -n "<library function>" playground.py`; every zero-hit family
in the table above gained its card in 1.2.50. Re-run the grep audit
whenever a card moves — the matrix is the contract.

## Per-file playground coverage (100x-111) — the tightened guarantee

The user sharpened the standing contract to **each file**: "check that
playground.py covers the entire contents (each file) of the zip and
rar files." The family matrix above proved every *workflow family*
had a card; the per-file audit re-ran the grep evidence **file by
file** (118 rows) and found two behaviours the family lens had
absorbed too coarsely — both closed in 1.2.51:

- `others/Vina/vina_reader.py`'s **folder** walk → the Docking
  results card's new *Scan Vina results folder…* button
  (`scan_vina_directory`: best affinity per log, strongest binder
  first, crashed logs keep NaN rows).
- the `gpfer.py` / `gpferreceptorni.py` / `dpfer.py` **batch loop**
  (one config pair per campaign sub-folder, custom receptor types)
  → the new `docking.write_config_batch` + the grid card's *Batch
  GPF/DPF for campaign…* button and *receptor types* field.

The zip side is verified standing every release: the downloaded
archive's `playground.py` is byte-identical to the working tree, zero
rar-content leakage, `testzip` clean.

The matrix below is the contract — re-run
`scratch/probe_coverage_111_perfile.py` (fresh `unrar x` → per-file
grep) whenever a card moves; a **ZERO-HIT** row is a regression.

| Rar file | Playground surface | Grep evidence |
| --- | --- | --- |
| `Classification/ADAboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/ANN.py` | Data & ML → Train a model (ann) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/ANN/ANN class.py` | Data & ML → Train a model (ann) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/ANN/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Classification/BBB/BBB_Prediction_model.ipynb` | Data & ML end-to-end (load→train→compare→save) | `list_models`×2 `build_model`×4 `train_model`×4 `compare_models`×2 |
| `Classification/BBB/tuned_random_forest_bbb_prediction_model.pkl` | Data & ML → Model persistence (fresh bundles; weights not shipped) | `save_model`×6 |
| `Classification/CATboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Decision tree.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Descriptor/Descriptor classification.py` | Chemistry → Feature factory | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Classification/Descriptor/Descriptor reuse(اگه اونیکی کار نکرد).py` | Chemistry → Feature factory (cached frame) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Classification/Descriptor/Descriptor reuse.py` | Chemistry → Feature factory (cached frame) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Classification/Elastic net.py` | Data & ML → Train a model (elastic_net_logistic) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/KNN.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Lasso.py` | Data & ML → Train a model (lasso_logistic) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/LightGBM.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Logestic Regression/Logistic regression class done.py` | Data & ML → Train + Save bundle | `list_models`×2 `build_model`×4 `train_model`×4 `save_model`×6 |
| `Classification/Logestic Regression/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Classification/Logistic regression.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/RF.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Ridge.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/SVM.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Classification/Stacking.py` | Data & ML → Ensemble builder (stacking) | `build_stacking`×2 `build_voting`×2 |
| `Classification/Stacking/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Classification/Stacking/stacking class.py` | Data & ML → Ensemble builder (stacking) | `build_stacking`×2 `build_voting`×2 |
| `Classification/Voting.py` | Data & ML → Ensemble builder (voting) | `build_stacking`×2 `build_voting`×2 |
| `Classification/XGboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Convert CSV to SDF.py` | Chemistry → Structure & table converters (CSV→3D SDF) | `table_to_sdf`×2 |
| `Convert IC50(nM) to pIC50.py` | Chemistry → Affinity conversion + converters (append pIC50) | `ic50_to_pic50`×2 `add_pic50_column`×2 |
| `Convert SMI to CSV.py` | Chemistry → Structure & table converters (.smi→CSV) | `smi_to_csv`×2 |
| `Descriptor classification.py` | Chemistry → Feature factory | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Descriptor regression.py` | Chemistry → Feature factory | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Non-numerical data remover.py` | Data & ML → Drop non-numeric columns | `clean_numeric_frame`×2 |
| `Regression/ADAboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/ANN.py` | Data & ML → Train a model (ann) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/ANN2.py` | Data & ML → Train a model (ann) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/CATboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Decision tree.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Descriptor/Descriptor regression.py` | Chemistry → Feature factory | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Regression/Descriptor/Descriptor reuse.py` | Chemistry → Feature factory (cached frame) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `Regression/Elastic net.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Huber.py` | Data & ML → Train a model (huber_regression) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/KNN.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Lasso.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/LightGBM.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Linear regression.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/RF.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/Ridge.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/SVM.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/SVM/SVM reg.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Regression/SVM/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Regression/Stacking.py` | Data & ML → Ensemble builder (stacking) | `build_stacking`×2 `build_voting`×2 |
| `Regression/Stacking/Stacking reg.py` | Data & ML → Ensemble builder (stacking) | `build_stacking`×2 `build_voting`×2 |
| `Regression/Stacking/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Regression/Voting.py` | Data & ML → Ensemble builder (voting) | `build_stacking`×2 `build_voting`×2 |
| `Regression/Voting/Voting reg.py` | Data & ML → Ensemble builder (voting) | `build_stacking`×2 `build_voting`×2 |
| `Regression/Voting/reuse.py` | Data & ML → Model persistence (load) | `load_model`×2 |
| `Regression/XGboost.py` | Data & ML → Train a model (registry) | `list_models`×2 `build_model`×4 `train_model`×4 |
| `Split CSV file.py` | Utilities → Table merge & split (split into chunks) | `split_table_into_chunks`×2 |
| `Tanimoto calculator.py` | Chemistry → Library similarity scan (matrix) | `tanimoto_matrix`×2 `similarity_scan`×2 `top_matches`×2 |
| `Tanimoto-edited.py` | Chemistry → Library similarity scan (scan ≥ threshold) | `similarity_scan`×2 |
| `others/Docking/New Text Document.txt` | empty placeholder (absorbed) | — |
| `others/Docking/Prettier_extracted_dlgs.py` | Utilities → Reindex by number | `reindex_by_number`×2 |
| `others/Docking/checker.py` | Docking → Grid, torsion & completion audit | `check_docking_completion`×2 |
| `others/Docking/dpfer.py` | Docking → Grid & configs (single) + **Batch GPF/DPF** (1.2.51) | `write_dpf`×2 `write_config_batch`×2 |
| `others/Docking/gpfer.py` | Docking → Grid & configs (single) + **Batch GPF/DPF** (1.2.51) | `write_gpf`×2 `write_config_batch`×2 |
| `others/Docking/gpferreceptorni.py` | Docking → **receptor types field** + Batch GPF/DPF (1.2.51) | `write_gpf`×2 `write_config_batch`×2 `receptor_types`×10 |
| `others/Docking/gpfreader.py` | Docking → Audit (GPF → parameters, ligand_types per folder) | `read_gpf`×2 `scan_gpf_ligand_types`×2 |
| `others/Docking/ligand generation/mismatch2.py` | Docking → Audit (folder: torsion consistency) | `check_torsion_consistency`×0 `scan_torsion_consistency`×2 |
| `others/Docking/ligand generation/nigga.py` | Docking → Ligand library + Chemistry → CSV→3D SDF (Meeko path; mol2 absorbed) | `prepare_ligand_library`×2 `table_to_sdf`×2 |
| `others/Docking/ligand generation/nigga2.py` | Docking → Ligand library + Chemistry → CSV→3D SDF (Meeko path; mol2 absorbed) | `prepare_ligand_library`×2 `table_to_sdf`×2 |
| `others/Docking/ligand generation/niggamatch.py` | Docking → Audit (folder: torsion consistency) | `check_torsion_consistency`×0 `scan_torsion_consistency`×2 |
| `others/Docking/ligand generation/niggaqt.py` | Docking → Ligand library → PDBQT (MGLTools absorbed → Meeko) | `prepare_ligand_library`×2 |
| `others/Docking/ligand_pdbqt_extracter.py` | Utilities → Campaign file tools (collect) | `collect_files`×2 |
| `others/Docking/ligand_preparation_autodock4.py` | Docking → Ligand library → PDBQT | `prepare_ligand_library`×2 |
| `others/Docking/main.py` | Docking → Campaign runner | `batch_run_directories`×2 |
| `others/Docking/reader.py` | Docking → Docking results (DLG cluster table) | `parse_dlg_cluster_table`×2 |
| `others/Docking/reader_at_same_folder.py` | Docking → Docking results (campaign scan) | `scan_docking_directory`×2 |
| `others/Docking/readerwithdetails.py` | Docking → Docking results (DLG cluster + summary) | `parse_dlg_cluster_table`×2 `parse_dlg_summary`×2 |
| `others/Docking/smilestopdbqt.py` | Docking → SMILES → PDBQT (single) + Ligand library | `smiles_to_pdbqt`×2 |
| `others/Docking/smiqt.py` | Docking → SMILES → PDBQT (single) + Ligand library | `smiles_to_pdbqt`×2 |
| `others/General/CSV_merger.py` | Utilities → Table merge & split (merge picked files) | `merge_tables`×2 |
| `others/General/Convert SMI to CSV.py` | Chemistry → Structure & table converters (.smi→CSV) | `smi_to_csv`×2 |
| `others/General/Nameextracter.py` | Utilities → Campaign file tools (list file names) | `list_file_names`×2 |
| `others/General/Prettier_extracted_dlgs.py` | Utilities → Reindex by number | `reindex_by_number`×2 |
| `others/General/Split CSV file.py` | Utilities → Table merge & split (split into chunks) | `split_table_into_chunks`×2 |
| `others/General/adder.py` | Utilities → Campaign file tools (distribute files) | `distribute_files`×2 |
| `others/General/approved.py` | Utilities → Approved drugs → report (CSV + DOCX) | `filter_approved_drugs`×2 `export_docx`×3 |
| `others/General/common_between_two_excel.py` | Utilities → Table compare & annotate (compare a column) | `compare_tables`×2 |
| `others/General/delete.py` | Utilities → Campaign file tools (cleanup tree) | `cleanup_tree`×2 |
| `others/General/deleter.py` | Utilities → Campaign file tools (delete named files) | `delete_named_files`×2 |
| `others/General/extaacter.py` | Utilities → Campaign file tools (collect files) | `collect_files`×2 |
| `others/General/filemerger.py` | Utilities → Table merge & split (merge whole folder) | `merge_table_folder`×2 |
| `others/General/reader_at_same_folder.py` | Docking → Docking results (campaign scan) | `scan_docking_directory`×2 |
| `others/Ml/Descriptor calculator.py` | Chemistry → Feature factory (rdkit_all) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `others/Ml/Descriptoredited.py` | Chemistry → Feature factory (rdkit_all) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `others/Ml/ECFP_calculation.py` | Chemistry → Feature factory (morgan fingerprints) | `compute_fingerprints`×5 |
| `others/Ml/Mordred_calculation.py` | Chemistry → Feature factory (mordred set) | `compute_descriptors`×2 `compute_fingerprints`×5 |
| `others/Ml/Morgan_calculator.py` | Chemistry → Feature factory (morgan counts) | `compute_fingerprints`×5 |
| `others/Ml/New Text Document.txt` | empty placeholder (absorbed) | — |
| `others/Ml/Sdf reading.ipynb` | Chemistry → Structure & table converters (SDF → table) | `sdf_to_table`×2 |
| `others/Ml/Tanimoto calculator.py` | Chemistry → Library similarity scan (matrix) | `tanimoto_matrix`×2 |
| `others/Ml/Tanimotoedited.py` | Chemistry → Library similarity scan (scan) | `similarity_scan`×2 |
| `others/Prettier_extracted_dlgs.py` | Utilities → Reindex by number | `reindex_by_number`×2 |
| `others/Reader for alphabetic folders/dlg_extracter_not_numeral.py` | Utilities → Campaign file tools (collect; pattern-based folders) | `scan_docking_directory`×2 |
| `others/Reader for alphabetic folders/readerwithdetails.py` | Docking → Docking results (DLG cluster table) | `parse_dlg_cluster_table`×2 |
| `others/Readme.txt` | README.md + Home → What lives where (absorbed) | — |
| `others/Vina/vina_reader.py` | Docking → Docking results (**Scan Vina results folder**, 1.2.51) | `parse_vina_output`×2 `scan_vina_directory`×2 |
| `others/Vina/vinar.py` | Docking → Campaign runner (Vina single ligand) | `run_vina`×2 |
| `others/Vina/vinar_multiple.py` | Docking → Vina config batch | `run_vina_batch`×2 |
| `others/adder.py` | Utilities → Campaign file tools (distribute) | `distribute_files`×2 |
| `others/compare/notcommon.py` | Utilities → Table compare & annotate (compare a column) | `compare_tables`×2 |
| `others/compare/smilesgetter.py` | Utilities → Table compare & annotate (annotate with SMILES) | `annotate_with_smiles`×2 |
| `others/dlg_extracter.py` | Utilities → Campaign file tools (collect) | `collect_files`×2 |
| `others/dpfer.py` | Docking → Grid & configs (single) + **Batch GPF/DPF** (1.2.51) | `write_dpf`×2 `write_config_batch`×2 |
| `others/gpfer.py` | Docking → Grid & configs (single) + **Batch GPF/DPF** (1.2.51) | `write_gpf`×2 `write_config_batch`×2 |
| `others/ligand_preparation_autodock4.py` | Docking → Ligand library → PDBQT | `prepare_ligand_library`×2 |
| `others/main.py` | Docking → Campaign runner (batch_run_directories) | `batch_run_directories`×2 |
| `others/reader_at_same_folder.py` | Docking → Docking results (campaign scan) | `scan_docking_directory`×2 |

## Per-file notebook coverage (100x-113) — the third axis, now with evidence

The GUI axis (100x-111) stays the binding contract; the notebook axis
now carries the same per-file evidence. Run
`scratch/probe_coverage_113_notebooks.py` after any notebook or
mapping change — it reuses the canonical FILE_MAP and greps every
executed notebook's **code cells** (markdown prose does not count: an
API tour means import/call surface, not a mention).

First verdict: 108/118 covered, 7 absent (the whole Stacking/Voting
family). Closed in 1.2.52:

- the 7 ensemble files (`Classification/Stacking*.py`, `Voting*.py`,
  `Regression/Stacking*.py`, `Voting*.py`) → the new *Meta-ensembles*
  subsection of `complete_ml_workflow.ipynb` §11
  (`build_voting` soft+weighted, `build_stacking` with the logistic /
  ridge blenders, classification and regression both);
- the data-utility families gained a dedicated tour
  (`pharmaml_walkthrough.ipynb` §9 "Data utilities"): converters
  round-trips, merge/compare/annotate, split ↔ merge-back,
  clean-numeric-frame, name listing, approved filter, similarity
  scan/top-k, GPF reading, batch config writing (Ni types) and the
  Vina results-folder walk.

Standing verdict after 1.2.52: **118/118 — notebook-covered 115,
absorbed 3 (documented non-goals), absent 0.** A NOTEBOOK-ABSENT row
is a rot alarm, not a contract breach: the GUI axis remains the
zero-tolerance one.

## Data provenance round-trip (100x-115, extended 100x-116/117)

The fourth axis closes the standing 100x-104 item: **every data file the
notebooks load is the file the builder writes.** Three legs, pinned in
`tests/test_data_provenance.py` (the binding contract) and re-runnable as
one probe (`scratch/probe_provenance_roundtrip_115.py`):

1. **Notebook-source scan** — every `DATA_DIR / "<name>"` reference in the
   eight executed notebooks' code cells resolves to a top-level
   `examples/data` entry. A stale path (a rename the notebooks missed)
   fails loudly. Standing verdict: 15/15 top-level entries referenced
   across 36 code-cell references, zero stale, zero deep.
2. **No orphans** — every top-level entry under `examples/data` is loaded
   by at least one executed notebook (13 CSVs + `lysozyme_1aki.pdb` +
   `docking_campaign/`). The inventory can never silently grow a file the
   demos forgot.
3. **Builder `--check`** — the non-mutating drift audit
   (`scratch/build_example_data.py --check`): every owned file re-generated
   in memory and byte-compared (42 files), missing files and orphans
   flagged (`disk == owned`, 43 files under `examples/data`), PDB size
   pins checked without a network fetch, and the full `verify()` content
   pins run — exit 1 with the drift list on any deviation. The
   planted-drift guard proves the gate: one flipped byte → exactly one
   drift line, exit 1, and the audit never repairs the file.

Re-run rule: after touching `examples/data/`, any notebook data cell, or
the builder, run the probe (or the pytest file) — a FAIL row is a
regression, not a warning.

### 100x-116 extensions

Two additions, same probe file, same pytest module:

4. **Documentation surfaces** — the resolve-or-fail contract extended to
   the CLI and the README: every `examples/data/...` path mentioned by
   `src/pharmaml/cli.py` (help text / docstrings) and by `README.md`
   must resolve on disk **and** be builder-owned. Prose promises
   ("ships as ...") bind exactly like code-block paths; a stray file or
   a stale promise after a rename fails the suite (3 new tests:
   CLI resolves, README resolves, both inside the builder inventory).
5. **Drift attribution** — every drift line now names the owning builder
   function — `... differs from the builder output (owner:
   build_qsar_multiclass_and_regression)` — and the new
   `STATS["owners"]` maps all 44 owned paths to their 11 builder
   functions (`@_owned` decorator tracks the running builder). A drift
   report is an actionable pointer ("rerun that function, or re-extract
   its notebook cell"), not just a file name; the planted-drift guard
   pins the attribution on both the byte-flip and the missing-file
   branches.

### 100x-117 extensions

6. **The `docs/*.md` guides join leg 4** — every markdown file under
   `docs/` (COVERAGE, DATA_AUDIT, INDEX, LEAKAGE_AUDIT) is now a
   documentation surface: an `examples/data/...` path a guide promises
   must resolve and be builder-owned, exactly like cli.py/README
   promises. The guides legitimately promise zero data paths today
   (they document the audit, not the inventory), so the scan carries a
   synthetic pattern-rot guard — a planted reference (with the
   sentence-punctuation and directory-slash variants) must extract
   exactly the clean name, keeping the scanner honest while the real
   surfaces stay empty (2 new tests).
7. **The walkthrough §9 heat-map figure is a failing check** —
   `tests/test_notebook_variants.py::
   test_walkthrough_similarity_heatmap_is_committed` decodes the §9
   similarity heat-map's committed `image/png` output, checks the PNG
   magic bytes, and pins a minimum size (a 5x5 labelled matrix renders
   tens of kilobytes, never bytes). Before this round the figure's
   existence was drift-checked only by notebook re-execution; now the
   suite fails if the demo's picture silently vanishes or becomes a
   stub (1 new test).

## User uploads (100x-118, extended 100x-141) — the contract extends past the two archives

The standing guarantee is "playground.py covers the entire contents of
the zip and rar files" — and a file the user hands over later binds
exactly the same way. The 2026-10 upload
`Applicability_domain_analysis.py` joins as **file #119**:

| Uploaded file | PharmaML surface | Status |
| --- | --- | --- |
| `Applicability_domain_analysis.py` | `ml.applicability_domain` + `knn_applicability_domain` + `bootstrap_uncertainty` (estimator-agnostic clones, half the upload's fits, seeded resamples, self-distance excluded); `viz.plot_applicability_domain` (the combined two-panel figure); CLI `ml ad` (`--data` split mode and the upload's `--bundle/--train-data/--test-data` flow); the playground **Applicability domain & prediction reliability** card (model source = last trained run or loaded bundle + its CSVs, k/percentile/bootstrap knobs, SMILES-named reliability table, the `Applicability_Domain_Results.csv` save); ML notebook §14 Trust | ✚ |

The probe axis: `scratch/probe_coverage_111_perfile.py` grew an
**uploads axis** (the upload's 8 mapped GUI symbols all hit;
zero-hit/missing-upload fails the probe) and
`scratch/probe_coverage_113_notebooks.py` requires its notebook surface
(binding — an uploaded file must tour in an executed notebook). The
verdict line now reads 118 rar files (115 covered, 3 absorbed,
zero-hit 0) **+ 1 upload covered = 119/119**.

The 2026-10 "new folder.zip" upload (audited file by file in
100x-141) extends the axis by **13 more files** (the `vina.exe` /
`vina_split.exe` / license `.rtf` binaries inside the zip are external
engines, absorbed like the rar's own engine payloads):

| Uploaded file | PharmaML surface | Status |
| --- | --- | --- |
| `ligand generation/nigga.py` | `docking.generate_ligand_structures` (the whole pipeline: tautomer → ETKDG conformers → MMFF/UFF → lowest-energy PDB with CONECT + OpenBabel MOL2, the Fe/Zn/Cu/Mn/Co/Ni/Mo/W skip list, the process-pool workers) + the playground **Ligand structures (CSV → PDB + MOL2)** card | ✚ 100x-141 |
| `ligand generation/niggaqt.py` | `docking.convert_pdb_folder_to_pdbqt` (prepare_ligand4.py over a folder of PDBs into numbered sub-folders, per-ligand timeout, conversion_log.txt) + the playground **PDB → PDBQT batch (MGLTools)** card | ✚ 100x-141 |
| `ligand generation/mismatch2.py` | `docking.check_torsion_consistency` + `scan_torsion_consistency` + the audit card's torsion mode | already covered |
| `ligand generation/niggamatch.py` | same surface (the numeric-sort variant needed none of its own) | already covered |
| `Vina/vina.conf` | `docking.parse_vina_config` + the grid box / config batch cards | already covered |
| `Vina/log.txt` | `docking.parse_vina_output` + `scan_vina_directory` + the results card | already covered |
| `Vina_contol/adder.py` | `docking.distribute_files` + the receptor distribution card | already covered |
| `Vina_contol/batch.py` | `docking.run_vina_batch(skip_existing=True)` — the resume | ✚ 100x-141 |
| `Vina_contol/conf.txt` | `docking.parse_vina_config` | already covered |
| `Vina_contol/niggavina.py` | `run_vina_batch` parallel + the skip-existing resume (the empty-log rule) | ✚ resume 100x-141 |
| `Vina_contol/vinar.py` | `docking.run_vina` + the Vina config batch card | already covered |
| `Vina_contol/vinar_multiple.py` | `docking.run_vina_batch(max_parallel=…)` | already covered |
| `Vina_contol/vina_reader.py` | `docking.parse_vina_output` + `scan_vina_directory` + the Vina results card | already covered |

The uploads verdict line now reads **14/14 upload files covered**
(119 rar + 14 upload = 133 surfaces, zero-hit 0).

## Library coverage (100x-119) — the fourth axis: the toolkit itself in the GUI

The user's next sharpening — "playground.py content is incomplete" —
opened a fourth audit axis: the rar mapping above proves the *original
archive* is covered, but the **library had outgrown the GUI**. The
audit walked every public symbol of every `pharmaml` module and grepped
`playground.py` for a surface, separating user-facing features from
internal helpers (preprocessor plumbing, validators used inside other
functions, `dataclass`/logger infrastructure — those ride along inside
the covered calls).

First verdict: ~30 user-facing capabilities with no GUI surface, all
closed in **1.2.58**:

| Library capability | playground.py surface since 1.2.58 |
| --- | --- |
| `pharma.nca_summary` + `auc_trapezoid` + `aucm_trapezoid` + `mrt` + `vd_from_dose_auc` + `validate_concentration_series` | PK / PD → *Non-compartmental analysis (NCA)* card (profile figure + CSV export) |
| `pharma.infusion_rate` + `iv_infusion_steady_state` + `volume_of_distribution` + `bioavailability` | PK / PD → *IV infusion & distribution* card |
| `pharma.tdm_trough_adjustment` | PK / PD → *TDM trough adjustment* card |
| `pharma.sigmoid_emax_effect` + `inhibitory_emax_effect` + `linear_effect` + `ec50_for_target_effect` (+ `effect_table` semantics) | PK / PD → *PD model explorer* card (table + curve + reverse EC50) |
| `viz.plot_concentration_profile` + `plot_dose_accumulation` | the NCA card's figure + the infusion card's repeated-dosing plot |
| `chem.canonicalize_tautomer` + `chem.smiles_to_pdb` (`PdbLigandReport`) | Chemistry → *Tautomer & 3D structure* card |
| `chem.compute_maccs` | Chemistry → *MACCS key fingerprint* card |
| `viz.plot_similarity_heatmap` | Chemistry → similarity scan card's *Plot heatmap* |
| `viz.plot_calibration_curve` | Data & ML → Results card's *Calibration curve* |
| `viz.plot_model_comparison` | Data & ML → Leaderboard card's *Plot leaderboard* |
| `biology.replicate_variability` + `summarize_compounds` | Cell Lines → screen card's two readout buttons |
| `viz.plot_receptor_atom_types` | Docking → receptor card's *Plot atom types* |
| `viz.plot_top_ligands` | Docking → results card's *Plot top ligands* |

## Library coverage (100x-120) — round 5: the last stragglers

The re-run of the same audit after 1.2.58 found seven more user-facing
capabilities without a GUI surface (and a stale version pin); all
closed in **1.2.59**:

| Library capability | playground.py surface since 1.2.59 |
| --- | --- |
| `viz.plot_descriptor_distributions` | Chemistry → Feature factory → *Plot distributions* |
| `viz.plot_learning_curve` | Data & ML → Results → *Learning curve* (CV-folds knob) |
| `viz.plot_table_comparison` | Utilities → Table compare → *Plot reconciliation* |
| `docking.parse_vina_config` | Docking → Vina config batch → *Parse config* (dump + form fill) |
| `biology.most_variable_genes` | Cell Lines → Expression table → *Rank variable genes (MAD)* |
| `pharma.effect_table` | PK / PD → PD model explorer — the sigmoid table renders `effect_table`'s lists |
| `docking.run_autodock4` | Docking → Campaign runner — the *AutoDock4 single folder* engine + resume checkbox |

Remaining zero-hit symbols (38, down from 46) are the documented
non-goals: internal
plumbing (`build_preprocessor`, `split_features_target`,
`has_categorical_columns` — inside `train_model`), single-object
companions the composed call already covers
(`knn_applicability_domain`/`bootstrap_uncertainty` inside the AD card's
`applicability_domain`; `correlate_with_response` inside
`sensitivity_associations`; `run_genetic_optimization`/`run_rl_optimization`
inside the generative tab's optimizer classes; `check_torsion_consistency`
inside the audit card's `scan_torsion_consistency`; `aucm_trapezoid`
inside `nca_summary`; `save_pdf_report` inside `save_visual_bundle`),
pure helpers (`tanimoto_from_sets`,
`safe_filename`, validators), dataclass result containers riding inside
composed calls (`PKSummary`, `TableComparison`, `TrialResult`,
`ModelInputReport`, the deep/residual MLP estimator classes behind
`build_deep_*`/`build_residual_*`, `SmilesRnn` behind the RNN trainer),
and the report-bundle figures the GUI renders as tables or saves in the
bundle (`plot_residuals`, `plot_roc_curves`, `plot_cv_scores`,
`plot_feature_importance`, `plot_class_distribution` — all inside
`save_visual_bundle`). The audit method is
re-runnable: walk `inspect.getmembers(pharmaml.<module>)` and grep
`playground.py` — a user-facing zero-hit is the next round's todo.

## Library coverage (100x-121) — round 6: UnitConverter + export_table

The sixth audit re-run (same method, after 1.2.59) re-examined the 38
documented non-goals one by one and found two more genuinely
user-facing capabilities; both closed in **1.2.60**:

| Library capability | playground.py surface since 1.2.60 |
| --- | --- |
| `pharma.UnitConverter` | PK / PD → *Unit converter* card (dimension picker + editable unit comboboxes + factor; the honest `UnitConversionError` names every supported unit) |
| `viz.export_table` | every table save in the window — one shared dialog offering `.csv` / `.tsv` / `.md` / `.json` / `.html` (all 11 save sites migrated) |

Round 6 also rode three content/UI items: the Deep Learning tab's
size-vs-score learning curve (`plot_learning_curve` over the full
dataset — the deep twin of the ML Results button), the parsed
`parse_dlg_summary` cluster verdict on the single-folder AutoDock4
line, and the Ctrl+F finder's session-persisted last query. Remaining
zero-hit symbols (36, down from 38): the documented non-goals above
minus `UnitConverter` and `export_table`.

## Library coverage (100x-122) — round 7: the reference objectives + molecule surgery

The seventh audit re-run (same method, after 1.2.60; every claimed
absorption verified in source this time — `knn_applicability_domain`
and `bootstrap_uncertainty` confirmed inside `applicability_domain`,
`correlate_with_response` inside `sensitivity_associations`) found
five more genuinely user-facing capabilities, all closed in **1.2.61**:

| Library capability | playground.py surface since 1.2.61 |
| --- | --- |
| `generative.SimilarityObjective` | Generative → objective combobox (*Similarity to reference SMILES*) + the swapped-in reference row |
| `generative.NoveltyObjective` | Generative → objective combobox (*Novelty vs corpus*; the card's corpus is the reference pool) |
| `generative.SubstructureObjective` | Generative → objective combobox (*Contains substructure (SMARTS)*) + the swapped-in SMARTS row |
| `generative.ring_topology_moves` | Generative → *Ring topology moves* card (the standalone one-step ring-size mutant enumerator) |
| `generative.scaffold_hop_crossover` | Generative → *Breed two molecules* card (the standalone two-parent crossover) |

Round 7 also composed `viz.save_pdf_report` into a first-class
surface — `File ▸ Save session report (PDF)…` walks the new
session-figure registry (one live entry per figure host) — and fixed
the latent session-restore bug where a restored RL selection left the
GA parameter block visible. Remaining zero-hit by the round-7 audit
script (44, down from 51 at the round's start — the new script is
stricter than the round-5/6 one: module constants like `ECFP_BITS`
and internal helpers are counted, which the older "38 → 36" numbers
excluded): the documented non-goal families — dataclass result
containers riding inside composed calls (`PdbLigandReport`,
`TableComparison`, `PKSummary`, `TrialResult`, `BootstrapUncertainty`,
`ModelInputReport`, `GridParameters`, `TorsionCheck`, `HTMLReport`),
module constants absorbed by the cards that use their values
(`ECFP_BITS`, `RISM_DESCRIPTORS`, `MODEL_ALIASES`, `TABLE_FORMATS`,
`PLOT_FORMATS`, …), internal plumbing (`build_preprocessor`,
`split_features_target`, `tanimoto_from_sets`, `figure_to_base64`,
`get_pyplot`, `require_matplotlib`, `aucm_trapezoid`), verified
absorptions (`knn_applicability_domain` and `bootstrap_uncertainty`
inside `applicability_domain`; `correlate_with_response` inside
`sensitivity_associations`; `check_torsion_consistency` behind the
torsion scan; `Deep/ResidualMLP*` behind the DL architecture
combobox; `ring_topology_mutate` behind the GA's topology mutation;
`run_genetic/rl_optimization` behind the class-based optimiser cards;
`SmilesRnn` inside `ReinventOptimizer`; `ScoringObjective` the
abstract base every concrete objective subclasses).

### Round 8 (100x-123) — the audit re-run and the GUI's own content gaps

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: 244 public symbols, 44 zero-hit — every
one re-examined and re-verified in source this round (the GA/RL
`run_*_optimization` functions are thin wrappers around the
`GeneticOptimizer`/`ReinventOptimizer` classes the cards drive
directly; `SmilesRnn` is the prior engine inside `ReinventOptimizer`;
`RISM_DESCRIPTORS` is the RISM card's ten meanings; `MODEL_ALIASES`
powers the model-name resolution every text entry feeds). The library
axis stayed at its floor — no new library-symbol gaps — so the round
closed the two genuine gaps on the GUI's OWN content surfaces:

| GUI surface | The 1.2.62 fix |
| --- | --- |
| the session PDF report | figures only → **figures + every filled table** (`_live_session_tables` walks the live `ttk.Treeview`s, climbs to the card shell for provenance, and ships real text pages through `save_pdf_report(tables=…)`) |
| the two surgery cards | text summaries only → **2D structure grids** (`molecule_grid_figure`: one RDKit panel per mutant / parent / viable child, honest unparseable-slot and empty-grid notes) |

The 44 zero-hit symbols are unchanged from round 7's classification
(containers, constants, plumbing, verified absorptions — all
documented above); the honest next library-axis frontier is the
notebook axis, which stays binding on the rar files only.

### Round 9 (100x-124) — HTMLReport surfaces, the library axis holds

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: 244 public symbols, 44 zero-hit. The
genuine find this round: **`pharmaml.viz.HTMLReport` — a documented,
user-facing, self-contained HTML report builder — had NO caller
anywhere** (not the playground, not the CLI, not the library's own
compositions; even its `add_metrics` metric-card path was dead code).
Every other zero-hit symbol was re-verified in source:
`correlate_with_response` is the engine `sensitivity_associations`
calls (biomarkers.py:141), `aucm_trapezoid` feeds `pk_summary` and
the AUCM ratio (pk.py:246/409), `tanimoto_from_sets` is the set
primitive under the bitarray/Mol similarity the cards drive,
`check_torsion_consistency` runs inside `scan_torsion_consistency`.

| Surface | The 1.2.63 fix |
| --- | --- |
| File ▸ Save session report (HTML)… | `HTMLReport`'s first caller: the same figure registry + table walk as the PDF, plus `add_metrics` census cards and an `add_text` provenance block — one self-contained `.html` (figures base64-embedded) |
| Generative ▸ Results card | **Draw top molecules** — the leaderboard's top 12 as a 2D structure grid, each panel carrying its total score as an xlabel score row (`molecule_grid_figure(scores=…)`) |
| Web ▸ Cell Line tab | the MAD variable-genes disclosure (`src/lib/variable-genes.ts`, parity-guarded against `most_variable_genes`) |

The 44 zero-hit count is unchanged — `HTMLReport` was already
classified "builder with no surface" and now HAS its surface (the
count drops to 43 by the audit's grep, the honest direction), and
`figure_to_base64` moves from plumbing to "the HTML report's embed
path". The remaining 43 are the round-7/8 classification: containers,
constants, verified absorptions.

### Round 10 (100x-125) — the front draws its molecules; the web twin gains the learning curve

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: 244 public symbols, 43 zero-hit — the
documented floor, unchanged, with the round-124 absorptions
spot-verified in source AGAIN this round (`bootstrap_uncertainty` is
the engine the AD card's `sensitivity_associations`/bootstrap
ensemble drives, `ModelInputReport` is what `validate_bundle_input`
returns at the bundle-audit card, `TrialResult` rows are the tuner
card's trials CSV export, `ring_topology_mutate` is the GA's topology
mutation operator). The library axis holds. The round's two genuine
gaps were the ones round 124's handover named — on the GUI's own
content surfaces and the web twin:

| Surface | The 1.2.64 fix |
| --- | --- |
| Generative ▸ Pareto front card | **Draw front** — the front's members as a colour-coded 2D structure grid (`molecule_grid_figure(colors=…)`): boundary corners (crowding +inf) accented teal, interior trade-offs muted neutral, objective values as every panel's score row. `Plot front` restores the scatter |
| Web ▸ ML Training tab | **Learning curve** — `POST /api/ml/learning-curve` + `src/lib/learning-curve.ts` + the verdict-strip/band-chart card: the desktop figure's data semantics (10 % … 100 % in 8 steps, k-fold CV per size, mean ± std bands, the 0.15 gap verdict), parity-guarded by `probe_lc_parity_125.{py,ts}` + `TestLearningCurveContract` (fractions bit-identical, verdict strings byte-identical) |

The 43 zero-hit are unchanged: containers, constants, and verified
absorptions — the classification that has held for three consecutive
re-audits. The rar/upload/notebook axes: 119/119, zero-hit 0, all
three probes green after the round's edits.

### Round 11 (100x-126) — the HTML report gains a theme engine; the Pareto card gains its own export

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **244 public symbols, 43 zero-hit — the
documented floor, byte-identical to round 125's final set.** Two
fresh-eye suspects from the 43 were re-verified as absorptions IN
SOURCE this round: `check_torsion_consistency`/`TorsionCheck` are the
single-file engine behind the ligand-audit card's
`scan_torsion_consistency` folder walk (the audit's own per-file probe
shows the pairing: `check_torsion_consistency:0
scan_torsion_consistency:2` on niggamatch.py / mismatch2.py), and
`PKSummary`/`aucm_trapezoid` are the return type + internal
integrator behind the PK card's `summary.aucm` readouts
(playground.py:6501/6568). The library axis holds; the round's
genuine work was the two UI gaps round 125's handover named:

| Surface | The 1.2.65 fix |
| --- | --- |
| `pharmaml.viz.HTMLReport` + every report | **Theme engine** — `theme="auto"/"light"/"dark"` baked as `data-theme`, CSS rebuilt on custom properties (auto keeps the original media-query behaviour palette-for-palette), and a per-reader toggle button (auto → light → dark, remembered via `localStorage`) in every report's top-right — verified end-to-end in a real browser incl. the saved-`light`-overrides-baked-`dark` reload |
| Generative ▸ Pareto front card | **Save figure…** — the per-card export of the figure THE CARD shows (grid or scatter), the honest distinction from Ctrl+S which saves the last figure drawn anywhere; guard path included |

The 43 zero-hit are unchanged (the same containers, constants, and
verified absorptions). The rar/upload/notebook axes: 119/119,
zero-hit 0, all three probes green after the round's edits. The
GUI-axis census holds at 68 live cards (the new button joined an
existing card).

### Round 12 (100x-127) — the per-card export on every Generative figure card; the web twin's session report

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **245 public symbols, 44 zero-hit.** The
+1 symbol over round 125's floor is `REPORT_THEMES` itself — exported
with the round-126 theme engine and until now consumed only inside
the library (`HTMLReport.__init__` validation); this round gives the
GUI a real caller (the session-report bake validates the app's theme
against it, a drifted value falls back to `auto`), so the 44th
zero-hit is a wiring decision made honest rather than a gap. The
library axis otherwise holds; the round's genuine work was round
126's handover list:

| Surface | The 1.2.66 fix |
| --- | --- |
| Generative ▸ Results / Ring topology / Breed cards | **Save figure…** on every remaining Generative figure card — the per-card export the Pareto card shipped in 1.2.65, generalised into `_save_card_figure` (the two-host Results card exports whichever of the card's own figures drew LAST, via the new monotonic `_pg_figure_seq` render stamp; still strictly narrower than the global Ctrl+S) |
| `pharmaml.viz.REPORT_THEMES` | **A real GUI caller** — the HTML report bake validates against the library's public contract instead of trusting the Tk theme variable |
| Web ▸ Playground | **The session report twin** — a "Session report" card above the sub-tabs (live census + one-click download), collecting the persisted session state into a section payload that `/api/report/session` + `scripts/report_bridge.py` compose into REAL matplotlib figures and assemble through the SAME self-contained themed `HTMLReport` the desktop saves; the learning-curve result moved into the store so the collector can walk it |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits. The GUI-axis census holds at 68 live
cards (the three new buttons joined existing cards).

### Round 13 (100x-128) — the per-card export closes app-wide

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **245 public symbols, 43 zero-hit —
byte-identical to round 127's final floor** (the REPORT_THEMES wiring
of 1.2.66 removed the 44th; every remaining member a previously
verified source absorption — torsion, PKSummary/aucm,
correlate_with_response and the rest of the standing classification).
The library axis holds; the round's genuine work was round 127's
handover item (b), taken to completion:

| Surface | The 1.2.67 fix |
| --- | --- |
| Every remaining figure card (15) | **Save figure…** on every figure card that lacked it — Data & ML (results / leaderboard / AD), Deep Learning (results + the TWO-host training history, newest stamp wins), PK / PD (IV profile / Emax / NCA / accumulation / family), Chemistry (feature factory / similarity scan), Docking (receptor / results), Utilities (reconciliation). Each saves the figure the card's OWN host shows through `save_figure` (PNG/SVG/PDF), never the global Ctrl+S last-drawn-anywhere figure; honest per-card guards everywhere |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.67 via
both drivers). The GUI-axis census holds at 68 live cards (the DL
Results card's new row is a frame INSIDE the card, not a new card).

### Round 14 (100x-129) — the PDF's dark ink + the web twin's card finder

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **245 public symbols, 43 zero-hit —
byte-identical to rounds 127/128's documented floor.** The library
axis holds; the round's genuine work was round 128's handover items
(a) and (d), both taken to completion:

| Surface | The 1.2.68 fix |
| --- | --- |
| `save_pdf_report(..., theme=)` | The PDF session report's **dark-ink variant**, riding the View ▸ theme radios the way the HTML report's palette does: a dark app writes near-black pages with light ink + accent headings, figures on white plates; a light app writes the classic paper — byte-identical (zero colour operators, pinned). New public `PDF_THEMES = ("light", "dark")` (no `auto`: the ink is frozen at write time, the caller resolves — the playground maps its radios through `_resolve_auto_theme`). |
| The web twin's card finder | `Ctrl/Cmd+Shift+F` (the browser owns plain Ctrl+F) opens a fuzzy dialog over the 24 always-rendered Playground cards (catalog: `card-directory.ts`); the jump switches top tab + sub-tab + PK inner tab, scrolls, and flashes the card. Title-first, panel-scoped, mount-gated resolution (the exit-animation race — an Overview blurb contains "Torsion QA" — caught by the E2E probe and fixed); a rename surfaces as an honest "Card not found" toast. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits. The GUI-axis census holds at 68 live
cards (the round touched no card topology). The web catalog walk:
24/24 entries through the live finder UI (probe_card_finder_129.sh).

### Round 15 (100x-130) — the two-sided finder recency row + the web Surgery sub-tab

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–129's documented floor.** The library
axis holds; the round's genuine work was round 129's handover items
(c) and (d), both taken to completion:

| Surface | The 1.2.69 fix |
| --- | --- |
| The finder's recency row (both sides) | Every Ctrl+F jump is remembered — most-recent-first, de-duplicated, capped at `_FINDER_RECENT_MAX = 6` — and the dialog renders one-click "Recently jumped" chips, filtered through `_live_cards()` (a vanished shell is honestly omitted). The memory persists with the session (`finder_recent` text key, JSON-encoded; tolerant restore). The web twin's Ctrl+Shift+F finder gained the same memory: localStorage `pharmaml.cardFinder.recent`, only jumps that actually LANDED are remembered, chips carry their sub-tab tag. |
| The web Surgery sub-tab | The mirror of the desktop Generative tab's two surgery cards: *Ring topology moves* + *Breed two molecules* over `scripts/surgery_bridge.py` (the stdin→stdout JSON bridge pattern — real `pharmaml.generative` operators, real RDKit, the desktop's own captioned-panel structure grids as base64 PNGs, honest null rows, per-card Download PNG). Route `/api/gen/surgery` (zod discriminated union, the bridge's caps). The card-finder catalog grew to 26 entries; the drift probe + walk re-ran green. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.69 via
both drivers). The GUI-axis census holds at 68 live cards (the round
touched no card topology). The web catalog walk: 26/26 entries
through the live finder UI (probe_card_finder_129.sh, the two Surgery
entries joined).

### Round 16 (100x-131) — the web session report's per-section control + the surgery feed; both finders' clear-recent

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–130's documented floor.** The library
axis holds; the round's genuine work was the two longest-standing
handover items — round 129's (a) and round 130's (b) — plus the
finders' (d), all taken to completion:

| Surface | The 1.2.70 fix |
| --- | --- |
| The web session report's per-section control | Every group the session actually has content for renders a toggle chip ("Include in report"): ML training, learning curve, PK / PD, NCA data, run history, surgery — a group with nothing to ship renders no chip. The census line and the download walk the SAME filtered collector (`collectSessionReport(snapshot, options)`), so "what will ship" is always the truth; a fully deselected session disables the button honestly. Skips persist in localStorage (`pharmaml.reportSections`, tolerant restore). |
| The surgery feed | A run on either Surgery card persists to the store (`surgeryResult`, sessionStorage + tolerant mode-discriminated validation, the latest run winning like `mlResult`), and the collector walks it: the run's metrics card, its SMILES list as a real table (a failed crossover an honest "(no viable crossover)" cell), and the captioned structure grid through a NEW `image` section kind on the report bridge — decoded, PNG-magic-checked, 2 MB decoded cap, re-composed full-bleed via the public `HTMLReport.add_figure` contract. The route's zod schema accepts the kind. |
| Both finders' clear-recent | One click on the row's dashed "Clear" chip forgets every remembered jump: the desktop empties the session-persisted `finder_recent` key and collapses the row in place; the web twin clears its localStorage and hides the strip live. The next open renders no row at all (the honest-empty rule); no row, no Clear button. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.70 via
both drivers). The GUI-axis census holds at 68 live cards (the round
touched no card topology). The web catalog walk: 26/26 entries (the
round touched no catalog topology). 15 new tests: TestReview131Gui
(7, desktop) + TestSessionReportSections (8, web parity incl. the
bridge's surgery-image round trip).

### Round 17 (100x-132) — the report order: the desktop picklist + the web twin's reorderable chips

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–131's documented floor.** The library
axis holds; round 131's handover named the round's work — (a) the
desktop report's per-section control and (c) the web report's walking
order — and settled (b) by inspection (every surgery grid renders
through `_render_figure`, so the desktop report already walks them):

| Surface | The 1.2.71 fix |
| --- | --- |
| The desktop pre-save picklist | Both report menus (PDF + HTML) consult a "Session report contents" dialog before the file dialog: every page the walk found (one figure page per live render, one text page per filled table) is offered with a checkbox — grouped, numbered, all on by default — and only the kept pages ship. A LIVE census footer recounts on every flip; at zero pages Save disables AND the confirm command re-checks (a keyboard/programmatic activation of the disabled button stays a no-op). Cancel aborts with no save started. |
| The web walking order | The section chips reorder: drag (GripVertical handle + primary drop-caret cue) or the hover/focus micro-arrows (labelled, extended hit areas — the keyboard/touch path); the order persists per browser (`pharmaml.reportSectionOrder`, tolerant restore, equality removes the key) and a dashed "Reset walking order" chip appears only while drifted. The census line confesses "in your chosen walking order". |
| The ordered collector | `collectSessionReport`/`sessionReportCensus` take an optional third `order` argument; the walk was refactored into six pure per-group builders assembled under `sanitizeSessionOrder` (unknown ids dropped, duplicates collapsed first-wins, missing groups appended canonical-last so a stored order predating a new group never drops it). Whole groups move — a group's internal order never re-cuts — and the census is order-invariant (pinned byte-identical to the 131 baseline). |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.71 via
both drivers: 64 TOTAL 190 s; 26 "26-R7 driver OK"). The GUI-axis
census holds at 68 live cards (the round touched no card topology).
The web catalog walk: 26/26 entries (the round touched no catalog
topology). 11 new tests: TestReview132Gui (7, desktop — the six
pre-132 report-menu classes gained the `picklist_accept_all` bypass)
+ TestSessionReportOrder (4, web parity incl. the sanitizer and the
order-invariant census).

### Round 18 (100x-133) — the picklist's per-page reorder + the registry's captions + the web twin's PREVIEW

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–131's documented floor.** The library
axis holds; round 132's handover named all three of the round's
items — (a) the desktop picklist's per-page reorder, (c) the surgery
feed reaching the picklist's labels, (b) the web report's preview —
and all three shipped:

| Surface | The 1.2.72 fix |
| --- | --- |
| The picklist's reorder | Every row of the pre-save dialog carries ▲ / ▼ buttons that move the page among its OWN kind's neighbours (a figure never crosses into the table block — both walkers ship the kinds as separate blocks). The rows ride in the WALKING order (table pages first, then figure pages — the exact sequence `save_pdf_report` / the HTML builder write them), a muted note under the header says so, the numbering rewrites itself after every move, the boundary buttons disable honestly, and the returned lists carry the dialog's order — the saved report walks its pages the way the checklist read them (pinned end-to-end into the written HTML). |
| The registry's captions | `_render_figure` takes an optional `caption` overriding the job label in the session-figure registry; the three generative structure grids pass their counts ("Ring topology moves — 6 mutant(s)", "Scaffold-hop litter — 4/6 viable children", "Top 12 molecules — the last run's leaderboard") so the picklist's rows and the reports' captions say what each page holds. |
| The web PREVIEW | A "Preview order" button opens a dialog rendering the ordered sections before any file is written — the exact payload the download would POST: numbered entries with kind badges, metric chips, tables capped at 6 rows (+ the honest "+N more rows"), figure SKETCHES as inline SVG from the same trace data (lines → polylines, bands → filled polygons + mean line, class scatters → stride-capped dots), and the surgery grids' real PNGs direct. The collector exports the sketch's pure helpers (`figureSketchBounds`, `PREVIEW_TABLE_ROWS`, `PREVIEW_SCATTER_POINTS`) so the probe pins the arithmetic. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.72 via
both drivers: 64 TOTAL 210 s; 26 "26-R7 driver OK"). The GUI-axis
census holds at 68 live cards (the round touched no card topology).
The web catalog walk: 26/26 entries (the round touched no catalog
topology). 13 new tests: TestReview133Gui (9, desktop) +
TestSessionReportPreview (4, web parity incl. the bounds arithmetic
and the section stream).

### Round 19 (100x-134) — the picklist's order memory + both twins' count-prefixed page names + the web preview's hover INSPECT

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–133's documented floor.** The library
axis holds; round 133's handover named all three of the round's
items — (b) the desktop picklist remembering the last custom order
per session, (c) the PDF twin riding the caption feed further, (a)
the web preview's sketches growing a hover inspect — and all three
shipped:

| Surface | The 1.2.73 fix |
| --- | --- |
| The picklist's order memory | A completed Save stores the dialog's per-kind label order (`_report_page_order`, all labels — kept or not), and the next Save's picklist re-opens pre-arranged in it (stale labels skipped, new pages appended in walk order) — the web twin's persisted walking order, mirrored per session. A muted confession line renders when the memory applied; the dialog's own "Reset to walking order" button restores the walk order AND forgets the memory, enabled exactly while the order has drifted (a remembered arrival or a manual ▲ / ▼ move), disabled at the walk order. Cancel never touches the memory. |
| The count-prefixed page names | The PDF's pages ("figure N · <caption>" / "table N · <caption>") and the HTML twin's captions/headings read the same — numbered in the picklist's own walking order, so both files' page headings read the way the picklist's numbered rows did. The HTML twin's 100x-124 " (2)" caption dedupe retired with it (the position prefix separates same-titled pages on its own). |
| The web preview's hover INSPECT | Every trace group in the preview's inline SVG sketch carries a native `<title>` — name, point count, x/y ranges (fmtCell-compacted); a band reports its CENTER mean; a classified scatter its class split; the empties confess; a degenerate axis reads its single value. The hovered trace thickens (group-hover emphasis) so the tooltip's subject is never ambiguous, the cursor confesses (`cursor-help`), and the caption says it out loud. The wording lives in the pure exported `traceInspectText` so the probe pins it verbatim. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.73 via
both drivers: 64 TOTAL 206 s; 26 "26-R7 driver OK"). The GUI-axis
census holds at 68 live cards (the round touched no card topology).
The web catalog walk: 26/26 entries (the round touched no catalog
topology). 12 new tests: TestReview134Gui (9, desktop) +
TestSessionReportInspect (3, web parity incl. the TS probe's wording
pins); the standing index pins updated for the count prefix
(TestReview123's PDF page name, TestReview124's retired dedupe,
TestReview132's three table-heading pins, the preview caption's
hover confession).

### Round 20 (100x-135) — the picklist's order memory across restarts + the PDF title page's Contents + the web preview's keyboard inspect

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–134's documented floor.** The library
axis holds; round 134's handover named all three of the round's
items — (a) the desktop picklist's memory persisting across
sessions, (b) the web preview's inspect growing a keyboard path, and
(c) the PDF twin's title page listing the numbered page names — and
all three shipped:

| Surface | The 1.2.74 fix |
| --- | --- |
| The order memory across restarts | The picklist's remembered order now rides the auto-saved session file (`report_page_order`, the JSON-in-a-text trick): `_collect_session` serialises the two real kinds with labels coerced; `_restore_session` replants it before the first picklist ever opens, tolerantly (a corrupt/foreign value resets per kind, never the restore); the Reset button's forget is symmetric (the close-time auto-save writes the empty state, the next launch starts at the walk order); `persist_session=False` keeps the per-process memory. |
| The PDF title page's Contents | `save_pdf_report` writes a mini table-of-contents under the subtitle — every page name in the walking order (tables first), each line the SAME string the page's own heading prints (a shared helper, no drift). The count-prefixed names make it read like the picklist's numbered rows; the empty report skips the section; a shared name lists twice (two pages, no silent dict-merge drop). |
| The web preview's keyboard inspect | Every trace group in the preview's sketch is focusable (Tab reaches it; the inspect text rides `role="group"`'s aria-label), focus thickens the trace like hover does, and a readout strip under the sketch carries the FULL inspect text (a polite `role="status"` live region — the native tooltip never shows on keyboard focus). Enter PINS the readout (Space toggles too, Escape clears, one pin at a time, a pin outlives its focus). The caption names both paths. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.74 via
both drivers: 64 TOTAL 212 s; 26 "26-R7 driver OK"). The GUI-axis
census holds at 68 live cards (the round touched no card topology).
The web catalog walk: 26/26 entries (the round touched no catalog
topology). 18 new tests: TestReview135Gui (9, desktop) +
TestSessionReportKeyboard (6, web parity incl. the E2E-caught Escape
layering) + 3 new TestSavePdfReport
cases (library); the standing caption pins updated for the round's
dual-path wording.


### Round 21 (100x-136) — the File menu's Clear remembered order + the HTML twin's Contents block + the web preview's persistent pin

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–135's documented floor.** The library
axis holds; round 135's handover named all three of the round's
items — (a) the HTML twin's header growing the same CONTENTS block
the PDF title page gained, (b) the web preview's pin persisting per
browser, and (c) the desktop picklist's Clear-remembered-order
affordance in the FILE menu — and all three shipped:

| Surface | The 1.2.75 fix |
| --- | --- |
| The File menu's Clear remembered order | File ▸ "Clear remembered report order…" (right after its forgetting sibling) drops both kinds of `_report_page_order` and writes the session file through IMMEDIATELY (the Reset's forget rides the close-time auto-save; this one does not wait — it survives a crash). The honest messages: "Nothing to forget yet" for a memory that never existed, the persist_session=False scope named, and the failed-write confession ("the next launch restores the remembered order after all"). The next picklist opens in the walking order with its confession quiet. |
| The HTML twin's Contents block | `HTMLReport` collects every table/figure page name in the walking order (the string the page itself prints — the `<h2>` heading / the `<figcaption>` caption; heading-less blocks list their positional fallback `table N` / `figure N`) and renders a themed `<nav class='contents' aria-label='Contents'>` panel under the subtitle — the PDF title page's 100x-135 mini table-of-contents, mirrored. No table/figure pages → the block is skipped (the noise rule). `save_visual_bundle`'s report.html gains it automatically. |
| The web preview's persistent pin | The pin rides an external store over localStorage (`src/lib/pinned-trace-store.ts`, keyed by the figure section's title — the web twin of the desktop's `report_page_order` session-file key): useSyncExternalStore subscribes (no restore effect — the react-hooks/set-state-in-effect rule stays clean), the snapshot reads a lazily-hydrated mirror, writes move the key AND the mirror and notify, SSR hydrates against null. A stored pin is range-validated at read time, a restored pin confesses ("pinned from your last visit"), an empty map removes the whole key. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.75 via
both drivers: 64 TOTAL 208 s; 26 "26-R7 driver OK"). The GUI-axis
census holds at 68 live cards (the round touched no card topology).
The web catalog walk: 26/26 entries (the round touched no catalog
topology). 19 new tests: TestReview136Gui (11, desktop incl. the
HTML-twin Contents end-to-end) + TestHTMLReportContents (5, library)
+ TestSessionReportPinStore (3, web parity: the TS probe pins + the
component/store static pins); one standing pin updated (the toggle's
computed form) and one joined the import line
(useSyncExternalStore).

### Round 22 (100x-137) — the order memory's export/import pair + the HTML twin's clickable Contents anchors + the web preview's Contents twin

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0 after the standing /tmp re-extract;
notebooks probe green), then the round-7 audit script re-ran: **246
public symbols, 43 zero-hit — byte-identical to rounds 127–135's
documented floor.** The library axis holds; round 136's handover named
all three of the round's items — (a) the web preview's dialog showing
its own numbered-contents line (the file's Contents block mirrored),
(b) the desktop picklist's memory growing a cross-session
export/import (the report_page_order key the session already
serialises), and (c) the HTML twin's Contents lines becoming clickable
anchors — and all three shipped:

| Surface | The 1.2.76 fix |
| --- | --- |
| The order memory's export/import pair | File ▸ "Export remembered order…" writes the STANDALONE artifact — a tiny JSON (`dump_report_order` normalises) carrying the format marker, the kind tag and the two label lists: the preference WITHOUT the session's inputs (a full Export session drags every input along). "Nothing to share yet" for a memory that never existed (never an empty file). File ▸ "Import remembered order…" adopts a colleague's file through `load_report_order` (a foreign/corrupt file is rejected with an honest error and the live memory NEVER touched; the empty order imports as the honest reset), replaces the in-process memory, and writes the session file through immediately — the Clear item's discipline, mirrored for adoption (the imported order survives a crash; the next launch re-opens pre-arranged). Proven across two fresh apps in scratch/shoot137.py. |
| The HTML twin's clickable anchors | Every table/figure page rides an id-carrying wrapper (`<section class='page' id='pg-N' tabindex='-1'>` for tables, `<figure class='page' id='pg-N' tabindex='-1'>` for figures — the figcaption's legitimate parent) and every Contents line links to it (`<li><a href='#pg-N'>name</a></li>`): a click jumps the reader to the page. Ids are minted at add time in lockstep with the names (href and target can never drift); hover/focus re-inks the line (accent + underline), the jumped-to page flashes an accent ring via `:target`, scroll-margin-top gives the landing breathing room. Fragment links only — the self-contained contract holds; the PDF twin's ink is frozen, its lines stay plain text. No caller change (the session report + save_visual_bundle gained it automatically). |
| The web preview's Contents twin | The preview dialog mirrors the downloaded file's header block as a STICKY strip: exactly the table/figure/image titles in walking order (metrics are title-page material and never listed; a metrics-only report hides the strip, the noise rule), each line a labelled button (the section's overall number + kind icon + title) that JUMPS the dialog to its section — smooth scroll, a brief accent ring on the arrival (the HTML twin's :target ring, mirrored), scroll-mt clearing the pinned strip. The derivation is pure (`sessionReportContents` in src/lib/session-report.ts, probe-pinned by scratch/probe_session_report_137.ts: walking order, metrics never listed, overall indices with gaps, kinds, the noise rule, the file-twin parity). |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.76 via
both drivers). The GUI-axis census holds at 68 live cards (the round
touched no card topology). The web catalog walk: 26/26 entries (the
round touched no catalog topology). 17 new tests: TestReview137Gui
(12, desktop: the normalisers + export + import incl. the pre-arranged
re-open, the foreign rejection, the honest empty, the persist-off
scope, the docstring pins, the File-menu wiring) +
TestPreviewContentsTwin (3, web parity: the TS probe's derivation
verdicts + the collector export surface + the component's strip/jump/
ring/margin/confession pins) + 2 anchor tests in test_viz.py (href↔id
resolution + fragment-only links; the anchor CSS on both themes).
Standing pins updated where the feature evolved: the Contents `<li>`
pins across test_viz/test_playground/smoke136 (now anchor-wrapped),
and the bundle test's anchor resolution.

**The round's fourth deliverable — the build-machine-path leak found
and fixed.** The round's own ZIP verification re-ran the 1.2.44
full-archive scan (LEAKAGE_AUDIT.md's contract: zero /home/<user>/
matches) and found 42 leaking entries — the 1.2.44 scrub had regressed
silently as rounds 120+ added scratch drivers with absolute paths, and
the scan had fallen out of the standing bump checklist. The fix:
scratch/scrub_home_paths_137.py (portable run hints, file-relative
Path targets, the junit xmls' site-packages-neutral rewrite) +
tests/test_leakage.py (the guard that FAILS the suite if any absolute
home prefix ever returns — the pattern assembled at runtime, the scrub
tool excluded as a pattern-documenter). All scrubbed drivers
re-verified (SMOKE134/136/137_OK; venv_repair_9.sh rides $HOME).

### Round 23 (100x-138) — the report's return leg + the strip's kind census + the picklist confession's provenance

The binding axes were re-proven first (per-file rar + upload probe:
118 + 1 = 119/119, zero-hit 0 after the standing /tmp re-extract; the
self-test green under Xvfb :99; notebooks probe green), then the
round-7 audit script re-ran: **246 public symbols, 43 zero-hit —
byte-identical to rounds 127–135's documented floor.** The library
axis holds; round 137's handover named all three of the round's items
— (a) the HTML twin's Contents growing a "back to contents" link on
each page wrapper, (b) the web preview's Contents strip showing the
kind badges' counts, and (c) the desktop's imported order confessing
its provenance in the picklist itself — and all three shipped:

| Surface | The 1.2.77 fix |
| --- | --- |
| The report's RETURN leg | Every table/figure page wrapper in `HTMLReport`'s output ends in a small "↑ Contents" link back to the header's block: the nav carries `id='contents'` + `tabindex='-1'` (a landing aid, never a tab stop) and flashes the same accent ring the jumped-to pages do — the one-way anchor became a ROUND TRIP, so a report thirty pages deep is one click from the index. The link is minted at ADD time with the wrapper (a wrapper exists exactly when at least one Contents line does, so the target can never be missing); prose-only reports render neither nav nor links (the noise rule's other half — the static CSS rides every report like every other rule); fragment links only, the self-contained contract holds; the PDF's ink stays frozen. No caller change — the playground's HTML twin and `save_visual_bundle`'s report.html gain it automatically. |
| The strip's KIND CENSUS | The web preview's Contents strip header carries the census line — "2 figures · 1 table · 1 structure grid" — derived from the strip's OWN entries (`sessionContentsCensus` + `contentsCensusText` in src/lib/session-report.ts, pure, probe-pinned by scratch/probe_session_report_138.ts): metrics are excluded BY CONSTRUCTION (the footer's census counts metric cards, this one never does — the strip confesses only its own lines), the wording rides the footer's singular/plural discipline, a census with no images drops the leg entirely, and a metrics-only payload hides the whole strip as before. |
| The confession's PROVENANCE | An order that arrived through File ▸ Import remembered order says so in the picklist itself — "The imported order from \<file\> was applied — Reset to walking order forgets it" (the basename, `_truncate_caption_text`-truncated for very long names, the DIRECTORY never leaking) — until the reader completes their own Save (the order becomes theirs; the wording returns to "Your last saved order was applied"), resets, or clears. The empty import confesses nothing (an empty memory is nobody's import); a foreign file never touches the live memory OR its provenance; and a fresh LAUNCH restores from the session file — which IS the last saved order — so a restart never claims an import (the honest restart contract, test-pinned through a seeding app's write-through and a fresh launch's restore). |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.77 via
both drivers — the re-execution's home-path leak in two notebook
outputs caught by the round-137 leakage guard and scrubbed before the
clean suite re-run). The GUI-axis census holds at 68 live cards (the
round touched no card topology). The web catalog walk: 26/26 entries
(the round touched no catalog topology). 20 new tests:
TestReview138Gui (12, desktop: the provenance attribute's life cycle —
import/empty/foreign/clear — the confession's wording incl. the
truncation and the Save-retirement, the Reset forget, the restart
contract, the docstring pins) + TestPreviewContentsCensus (3, web
parity: the TS probe's census verdicts + the collector export surface
+ the component's census-line pins) + TestHTMLReportBackLinks (5,
test_viz.py: the wrappers' return links incl. the captionless figure,
the theme-variable styling, the noise rule's other half, the bundle's
automatic gain). Standing pins updated where the markup evolved: the
nav string (now id+tabindex-carrying) and the :target selector (now
nav-covering) across test_viz/smoke136/smoke137/shoot136/shoot137.

### Round 24 (100x-139) — the return link's pointer confession + the census's clickable legs + the strip's order confession

| What shipped | The contract |
| --- | --- |
| The return link's POINTER confession | The "↑ Contents" link every HTML page wrapper ends in carries `title='Back to the contents block at the top of the report'` — the pointer world gets the same confession the aria world always had. ONE string, minted once (`_BACK_CONTENTS_LINK` in `pharmaml.viz/report.py`), so the table and figure wrappers can never drift apart (the lockstep discipline the page ids follow); the playground composes through `HTMLReport` and carries no inline twin (test-pinned). No caller change: the desktop's save path, `save_visual_bundle`, and the web bridge's `/api/report/session` all gained it automatically; the PDF twin's ink stays frozen. |
| The census's CLICKABLE legs | The web preview's census line ("2 figures · 1 table · 1 structure grid") became interactive: each kind's count is a button that SPOTLIGHTS that kind's strip lines — the other kinds dim to 40% opacity, never hidden (every dimmed line stays exactly as clickable as before, jumps included); the active leg carries `aria-pressed` with accent + dotted-underline styling; clicking the active leg again clears it; the spotlight resets when the dialog closes (a fresh open starts fresh — a view aid, never a filter on what ships). A zero-count kind renders as plain text (nothing to spotlight); the legs' labels are the same text the plain line prints (`contentsCensusLegs`, pure, probe-pinned — the visual census can never drift from the aria one); a metrics leg can never exist (excluded by construction upstream). |
| The strip's ORDER confession | The web preview's confession line appends "Your last saved walking order was applied — Reset walking order forgets it." while the browser's persisted walking order is in force (`orderConfessionText`, pure, probe-pinned — the web twin of the desktop picklist's memory label); the canonical order confesses NOTHING (the desktop's "nothing remembered: stay quiet" noise rule, mirrored). The desktop's own wording stays the anchor this round mirrors — test-pinned on the picklist's method source. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.78 via
both drivers — the re-execution's home-path leak in two notebook
outputs scrubbed before the suite ran, the standing round-138 trap).
The GUI-axis census holds at 68 live cards (the round touched no card
topology). The web catalog walk: 26/26 entries (the round touched no
catalog topology). 8 new tests: TestReview139Gui (5, desktop: the
docstring pins, the picklist-wording anchor, the no-inline-link
discipline, the library constant's exact string, the end-to-end save
path shipping the tooltip on both wrappers) + TestPreviewContentsSpotlight
(3, web parity: the TS probe's legs + confession verdicts, the
collector export surface, the component's spotlight + confession
pins). Standing pins updated where the markup evolved: the return-link
string (now title-carrying) across test_viz's TestHTMLReportBackLinks
(one shared BACK_LINK class constant), smoke138, shoot138, and the
report.py doctest.

### Round 25 (100x-140) — the forward links' pointer confession + the census's keyboard path + the confession's in-dialog reset

| What shipped | The contract |
| --- | --- |
| The forward links' POINTER confession | Every Contents line in the HTML twin carries `title='Jump to this page in the report'` — the same tooltip discipline the return link gained in 100x-139, applied symmetrically to the other leg: the aria world always knew where the jump goes (the line's own text names the page), a hovering pointer gets told now. The tooltip is minted at the ONE place the lines are minted (the render join), so href and tooltip can never drift (the lockstep discipline the ids and names have always had); the playground carries no inline markup twin (test-pinned on the markup form, not the English phrase — the docstring may quote the wording). No caller change: the desktop's save path, `save_visual_bundle`, and the web bridge's `/api/report/session` all gained it automatically; the PDF twin's ink stays frozen. |
| The census's KEYBOARD path | The web preview's spotlight gained a keyboard twin: digits 1/2/3 toggle the census line's 1st/2nd/3rd leg (figure / table / structure grid — the digits count the line's legs, clickable or plain, so the mapping is stable whatever the session lists), 0 clears, every other key PASSES untouched (Tab, Escape — the pin-then-close layering's own — and anything a modifier rides). The map is PURE (`spotlightKeyAction` in src/lib/session-report.ts, probe-pinned: every verdict incl. the zero-count leg's swallowed no-op and the out-of-range digit's pass) and the dialog's keydown consumes the SAME verdict a click consumes — the two paths cannot disagree. Guards: modifiers and text fields keep their keys; the legs carry `aria-keyshortcuts` with their digit; the census paragraph's aria-label + title confess the keys. |
| The confession's in-dialog RESET | The strip's order confession grew its way out INSIDE the dialog: a "Reset now" link beside the sentence restores the canonical walking order without closing the preview — the strip, the section list and the download all take it immediately, through the SAME single `resetOrder` source the card's dashed chip uses (the card's chip stays the other way out, for when the dialog is closed). The confession retires honestly on reset (the canonical order confesses nothing, as always); the desktop picklist's own in-dialog Reset button — always there — is the parity this round brought the web twin up to. |

The rar/upload/notebook axes: 119/119, zero-hit 0, all three probes
green after the round's edits (notebooks re-executed with 1.2.79 via
both drivers — the re-execution's home-path leak in two notebook
outputs scrubbed before the suite ran, the standing round-138 trap;
the junit140.xml scrubbed after the final run, the standing order).
The GUI-axis census holds at 68 live cards (the round touched no card
topology). The web catalog walk: 26/26 entries (the round touched no
catalog topology). 11 new tests: TestContentsForwardTooltips (5,
test_viz.py: every line carries the tooltip, the round trip confesses
in BOTH directions, the noise rule's mirrored half, the bundle's
automatic gain, the CSS comment's contiguous discipline pins) +
TestReview140Gui (3, desktop: the docstring pins, the one-place
minting — markup-form pin, the end-to-end save path shipping the
forward tooltip on every line) + TestPreviewKeyboardSpotlight (3, web
parity: the TS probe's every verdict, the collector export surface,
the component's keydown + keyshortcuts + in-dialog reset pins).
Standing pins updated where the markup evolved: the Contents-line
form (now title-carrying) across test_viz (12), test_playground (7),
smoke136/137/139, shoot137/138/139, and the report.py doctest.

### Round 26 (100x-141) — the "new folder.zip" upload audited file by file: the nigga.py pipeline + the niggaqt.py converter + the Vina batch's resume

| What shipped | The contract |
| --- | --- |
| The file-by-file audit | The user's new upload (22 entries; 13 script/config/log files + 6 engine binaries + folders) was checked one file at a time against the toolkit's surfaces: 9 of the 13 were already covered (the torsion pair, the Vina configs/logs, the receptor distribution, the Vina runners and reader — most of them rar twins the toolkit had already absorbed), and the audit isolated the three genuine gaps: nigga.py's structure pipeline, niggaqt.py's MGLTools converter (previously "absorbed: MGLTools deprecated" — the re-delivery says otherwise), and the batch.py/niggavina.py resume. The probe's uploads axis now maps all 13 files (14/14 with the AD upload); the engine binaries are absorbed like the rar's own payloads. |
| `generate_ligand_structures` + the Ligand structures card | nigga.py whole: tautomer canonicalisation (best-effort, confessed per row), the Fe/Zn/Cu/Mn/Co/Ni/Mo/W skip list (`UNSUPPORTED_LIGAND_ATOMS`, per-row reason, never batch-fatal), ETKDG multi-conformer embedding, MMFF minimisation with the UFF fallback, the lowest-energy conformer kept, the one-model PDB written by the original's own writer (HEADER/COMPND/ATOM/CONECT/END — the connectivity prepare_ligand4.py reads), and the OpenBabel/Gasteiger MOL2 twin (graceful without OpenBabel — the PDB lands, the row confesses). The workers knob rides a spawn-context ProcessPoolExecutor (the GUI's worker thread makes fork the documented deadlock risk). The card: library picker reuse, conformers/workers/MOL2 knobs, report tree, CSV save. |
| `convert_pdb_folder_to_pdbqt` + the MGLTools batch card | niggaqt.py whole: prepare_ligand4.py over every .pdb of a folder into the numbered-sub-folder tree (1/, 2/, 3/…), the 30-second per-ligand timeout (one subprocess.run(timeout=...) instead of the original's polling loop), the conversion_log.txt with its SUCCESSFUL/TIMED OUT/FAILED sections, and pre-flight path checks (a typo'd MGLTools path is a typed error before any subprocess fires). The card: source/output pickers, both MGLTools paths pre-filled with the original's Windows defaults, the timeout knob, report tree, CSV save. |
| `run_vina_batch(skip_existing=True)` + the resume checkbox | batch.py's "already docked" skip and niggavina.py's empty-log rule, unified: a ligand is skipped only when its _out.pdbqt AND non-empty log both exist (the zero-byte log marks a crashed job and is re-docked); skipped rows return skipped=True/ok=True/seconds=0.0 and the vbatch card's table shows "skip" with the summary confessing the count. The checkbox rides the card next to the workers knob. |

The axes after the round: rar 118 (115 covered + 3 absorbed, zero-hit
0) + uploads 14/14 — the probe verdict line reads "every rar + upload
file has a playground.py surface (or a documented absorption)". 30 new
tests: TestLigandStructureGeneration (6) + TestMglToolsConversion (6)
+ TestVinaBatchResume (4) + TestReview141Gui (14); the docs-index
symbol test re-synced (275 symbols). The version chain moved to
1.2.80 across all four carriers.

### Round 27 (100x-142) — the engineering review: a one-by-one audit of every tab, nine latent bugs fixed, and the picked-paths session + the generate→MGL relay + the pre-flight resume census

| What shipped | The contract |
| --- | --- |
| The one-by-one audit | The instruction ("review the codebase, troubleshoot potential issues, improve the engineering details") became a tab-by-tab review of playground.py (module helpers, chrome, session machinery, Data & ML, Deep Learning, Generative; the Docking tab in parallel). The audit's harvest: one real bug (the text-map drift), five risks (the Ctrl+O race, the _poll_jobs ordering, the finder's title-only keys, the AD k-guard's condition, the shadowed tuner parameter), three polish items (fractional parse_int truncation, the _copy label clobber, _require's unguarded next()) — all nine fixed, each with its own regression test or source-level pin. |
| The picked PATHS ride the session | `_SESSION_PATH_ATTRS` (27 pure picker attributes across Docking / conversion / Utilities) persist through the session file's new `paths` key and restore only when the folder still EXISTS on this machine (a mailed session degrades to the honest un-picked state; a live pick is never overwritten; dataset paths needing a reload stay out on purpose — a path without its frame would be a lie the label tells). `_refresh_restored_path_labels()` re-renders every picker's info label after the restore, reusing the four cards that own refresh helpers. |
| `vina_batch_progress` + the Count remaining button | The pre-flight resume census: `VinaBatchProgress(total, docked, remaining, collisions)` applies the SAME skip rule and the SAME collision-disambiguated output names as `run_vina_batch(skip_existing=True)` — the stem logic lives in one shared `_batch_output_names` helper, so "N/M already docked · K to go" is a promise the run keeps. The vbatch card's button reports it (confessing name collisions); toggling the resume checkbox with both folders picked runs the census on its own (the trace degrades to a no-op during session restore and never fires while busy). |
| The generate → MGLTools relay | The ligand-structures card's "Send PDBs → MGLTools" button: the structure output folder becomes the converter's PDB source, the PDBQT output folder is pre-filled BESIDE it (never inside, never over a live pick), the card's info label refreshes, and the finder's own jump + flash lands on the MGL card — the nigga.py → niggaqt.py relay in one click. |
| `_dl_load_frame` (the Apply-best fix) | `_ml_tune_apply` → dnn reloaded the ORIGINAL CSV (resurrecting every row the Clean card removed — the winner then trained on data it was never tuned on). The split-out `_dl_load_frame` adopts the LIVE cleaned frame; `_dl_load_path` is a thin wrapper. |

The axes after the round: unchanged (rar 118, uploads 14/14, zero-hit
0) — the round was engineering depth, not upload breadth. 41 new
tests: TestVinaBatchProgress (7, test_campaign.py — the census counts
the skip rule rows, the zero-byte log means redo, the census AGREES
with the run, collisions, the typed error, the frozen dataclass, the
package export) + TestReview142Gui (33, test_playground.py — the
docstring pins, the seven-key text map + the ownership walk + the
live Chemistry reset/copy confessions, the fractional parse_int
rejection, the Ctrl+O busy guard, the _poll_jobs ordering pin, the
_copy busy skip, the _require fallback, the AD k-guard pin, the
notify rename pin, the finder's keyed recency + legacy tolerance +
jump recording, the session-path collect/restore/existence/overwrite
guards + the round-trip + the allowlist pins, the bridge's handover +
no-overwrite + warning + live button, the census button + auto trace
+ quiet no-path case, the live-frame adoption + the Apply-best source
pin, and the library export + wiring pins). The docs-index symbol
test re-synced (277 symbols). The version chain moved to 1.2.81
across all four carriers.

### Round 28 (100x-143) — the second engineering review: every file-dialog CSV loader off the main thread, the Utilities restored-path labels, and the audit's remaining findings

| What shipped | The contract |
| --- | --- |
| The five-loader port | The Data & ML / Deep Learning / feature-factory / Cell Lines screen / Cell Lines expression file-dialog flows all ran `read_table` / `pd.read_csv` on the MAIN thread — a multi-hundred-megabyte CSV froze the window while every long calculation already showed a progress bar (the round-27 audit's own §4 finding, plus the Cell Lines twins the fresh audit added). Each tab gained a `_load_file` job twin over `_start_job("loading {name}")` (the Docking tab's proven pattern); the bundled examples stay synchronous (shipped, tiny — the 46 test fixture call sites depend on it), and each tab's sync core and job render share ONE adoption body (`_ml_adopt_frame`, `_dl_load_frame`, `_feat_adopt_frame`, `_bio_adopt_screen`, `_bio_adopt_expr_frame`) so the two paths can never drift. The Cell Lines collapse checkbox is captured on the main thread (a Tk variable is never read from a worker). |
| The Utilities restored-path labels | Round 27's `_SESSION_PATH_ATTRS` restored seven Utilities paths, but `_refresh_restored_path_labels()` walked only the Docking cards + the converter pair — the four Utilities info labels kept their launch placeholders while the run silently used the restored paths (the fresh audit's one real bug; round 27's "every restored path re-renders its picker's info label" promise, finally kept). The refresh now runs the three cards' own mode-aware helpers and writes the reindex picker's full-path text. |
| "SDF → table" without the phantom output file | The converter's run guard demanded an output file for every non-pIC50 mode, but the SDF branch returns an in-memory table (tree + "Save table CSV…") and never writes it — the guard now exempts SDF mode (the pIC50 exemption extended to its in-memory twin), and picking an output for either in-memory mode gets the same honest notice. |
| The MACCS mixed-parse label | A list where some molecules parsed and some failed lost its computed summary to a false "no MACCS keys" clobber. The summary now wins with the dropped count riding alongside; the full clobber is reserved for the nothing-parsed case where it is actually true. |
| Two advisory dialogs joined the showwarning convention | The surgery card's "Type a SMILES in the molecule row first." and the rescore card's "Name at least one evidence column." were showerror where every sibling card shows a warning for missing input; parse errors keep their showerror. |
| The notebook census tour | `complete_docking_workflow.ipynb` §12 gained the census + workers>1 cell (the round-27 worklog's own documentation candidate): `vina_batch_progress` on a three-ligand tree (one docked, one crashed via a zero-byte log, one untouched → "1/3 already docked · 2 to go") and `run_vina_batch(..., max_parallel=2, skip_existing=True)` showing the honest engine gate; the recap table gained the Resume census row. The web finder's localStorage memory was audited against the desktop's "tab:title" discipline and verified already stronger (keys on unique catalog ids, tab label always shown) — no change needed. |

The axes after the round: unchanged (rar 118, uploads 14/14, zero-hit 0)
— the second engineering-depth round. 21 new tests: TestReview143Gui
(test_playground.py — the five job loaders end-to-end through the real
dialog monkeypatch pattern, the empty-CSV warning via the render path,
the busy guard, the synchronous-example design pin, the main-thread
collapse capture pin, the shared-adoption-body source pins, the
Utilities label re-render probe, the SDF guard exemption + in-memory
notice + still-firing twin, both MACCS label verdicts, and the two
showwarning convention pins) + the rescore-gates standing contract
updated to the new warning. The notebook re-executed and its
`_no_output` twin regenerated. The version chain moved to 1.2.82
across all four carriers.

### Round 29 (100x-144) — the third engineering review: the multi-file session picks, the determinate loader bar, and the audit's parity/normaliser fixes

| What shipped | The contract |
| --- | --- |
| The `paths_multi` session key | The three Utilities MULTI-file pickers (the merge & split tables, the approval exports, the distribute-files list — `askopenfilenames` selections) now ride the session file as an always-present key with the single paths' own discipline: per-file existence filtering (a vanished file drops out, the survivors keep their order), never overwriting a live pick, and a `_SESSION_PATH_LIST_ATTRS` allowlist exactly three entries wide. `dump_session` drops a non-list value instead of char-splitting it into single-character "paths" (the audit's §5); `load_session` sanitises foreign payloads to shorter honest lists; the approval card's inline label re-renders after a restore in the picker's own "first three + … (+N)" format (the merge/tools labels already rode their mode-aware helpers from round 28). |
| `read_table_with_progress` (new public reader) | CSV-like files at/above the 8 MiB threshold stream in 50k-row chunks with `(rows_done, rows_estimated, "")` reported per chunk — the estimate from the file's own bytes-per-line average (64 KiB sample), degrading to an honest `None` for a giant first line. Small files and Excel pass straight through `read_table` (no reports, no behavioural difference). `_finalise_columns` is ONE shared rule for both readers, and ragged edge cases delegate to the sync reader's own verdict — the audit's §3 parity break (a sniffed `;` separator choking on a ragged body raised where `read_table` silently returned its mangled comma frame) is closed by delegation, so the two readers can never disagree on the same bytes. |
| The four loaders report | The Data & ML / Deep Learning / feature-factory job loaders take the reporter (`work(report)` → `read_table_with_progress(path, report)`) and the Cell Lines screen forwards it through `load_response_table(..., report=...)` — a new optional callback that never fires for DataFrame callers. A large CSV now carries a determinate bar ("loading x — 900000/1499898 · ~1 s left") instead of the open-ended marquee; the expression-matrix loader keeps the marquee by design (its indexed read has its own fallback semantics). |
| The chained-job guard | `_poll_jobs` clears `_on_done` only when the app is idle — a render callback that chains straight into a follow-up job (round 27's own documented scenario) keeps its callback instead of having it silently wiped one line later (the audit's §2; round 27's fix was one line short of its own promise). Pinned by a direct `_poll_jobs` exercise: the chained callback survives and still fires; the idle twin still clears. |
| The Apply-best invariant (the handover's item b) | Stated where it lives: `_ml_adopt_frame` sets frame and path TOGETHER and nothing ever clears the frame alone, so the relay's `elif _ml_path` branch is a safety net for programmatic plants, never a state a normal load can reach — pinned by a source-order test (the adoption body is the ONLY `self._ml_path = path` site, the Clean card swaps the frame alone). |

The axes after the round: unchanged (rar 118, uploads 14/14, zero-hit 0)
— the third engineering-depth round. 27 new tests: TestReadTableWithProgress
(test_io.py ×11 — the small-file plain-reader passthrough, the
chunk/estimate contract, dtype-and-value parity across three delimiters
via `assert_frame_equal`, the header-only twin, the ragged-semicolon
parity verdict, the malformed-file clean error, the missing/Excel
routes, the giant-first-line degradation, the estimator's two honest
sides, the duplicate-header refusal, the export surface) + the biology
pair (the 12 MB streamed screen with exact final report, the
DataFrame-never-fires pin) + TestReview144Gui (test_playground.py ×14 —
the round docstring pins, the paths_multi round trip/sanitise/collect,
the vanished-file filter with order, the live-pick guard, the foreign
allowlist, the approval label format, the allowlist contents, the
reporting-loader source pins, the live determinate-bar test over a
genuine 10 MB CSV with a throttled-but-real reader, the chained-job
pair, the source guard pin, the Apply-best invariant pin). The
docs-index symbol test re-synced (278 symbols). The version chain
moved to 1.2.83 across all five carriers (the VERSION file included).

### Round 30 (100x-145) — the "کد علی" upload audited file by file: D2.py's data prep, its permutation importance, and the radius-3 ECFP family

| What shipped | The contract |
| --- | --- |
| The one-by-one verdict over the 8 uploaded files | `pdbqtgeneration.py` — byte-identical to the audited `niggaqt.py` (the MGLTools PDB → PDBQT batch card, 100x-141). `pdbfromsmile.py` — the STRICTER older twin of `nigga.py` (the LIGAND STRUCTURES card implements the tolerant version: auto-generated names, first-column SMILES fallback — a superset). `mismatchcheck.py` — the grid/torsion/completion audit card's TORSDOF-vs-active-REMARKs-vs-BRANCH rule, upgraded with MGLTools/Meeko convention auto-detection. `niggavinamulti.py` — the Vina config batch card's thread-pool + resume story (`run_vina_batch(skip_existing=True)` + the pre-flight census). `conf.txt` — the config-batch card's parsed-keys surface. `vina.exe` — an external engine binary (absorbed like the rar's own engines, never shipped). `Ml/pred.py` + `Ml/D2.py` — the round's genuine work, below. |
| `prepare_pic50_dataset` (chem/converters.py) | `Ml/D2.py`'s `process_data` upgraded: auto-detected or explicit SMILES/endpoint columns (the upload's own `Ligand SMILES` / `EC50 (nM)` spellings first), censored `<`/`>` magnitudes kept AND counted (the original stripped them in silence), the `0 < EC50 ≤ upper_limit` cap (default 1e9 nM — D2.py's own), the duplicate-SMILES collapse as a parameter (`median`/`mean`/`first`/`keep`; the original hard-coded the median), the conversion through `ic50_to_pic50` (any unit), and a typed `EndpointDatasetReport` whose buckets satisfy the accounting identity (valid + missing-SMILES + unparsable + out-of-range == rows in; rows out == valid − duplicates collapsed). |
| The EC50 → pEC50 prep card (Data & ML) | The Dataset card's third row: the duplicate-rule combobox + the prep button + the save button. The audit lands in the status line ("5 rows in → 2 out · 1 censored magnitudes used · 1 duplicates collapsed (median) …"), one click adopts the prepared set as the active dataset with pEC50 pre-picked as the target, and declining keeps the loaded frame untouched. |
| `permutation_feature_importance` (ml/evaluation.py) | `Ml/D2.py`'s interpretability step, typed: a `PermutationImportance` record (names/means/stds/scoring/repeats/n) with `as_frame()` and first-declared-wins `ranking()`. Works with bare estimators AND fitted pipelines (the preprocessing reruns on every shuffle, so the measure reflects the whole model), defaults D2.py's own choices (r2/accuracy, 5 repeats, seed 42), pins `n_jobs=1` (the nested-parallelism trap the original's comment confessed), and raises the house `ModelError` for every bad input (task, repeats, row mismatch, names mismatch). |
| `plot_permutation_importance` (viz/ml_plots.py) | The top-N signed horizontal bars with ±1 std whiskers — the spread the original threw away — and NEGATIVE means (a shuffle that HELPED: a noise feature) in the palette's red. Accepts the record or its `as_frame()` twin, rides the shared `_check_top_n` contract, and labels the axis honestly ("importance (Δ score when shuffled)"). |
| The Permutation-importance button (Results card) | Runs on the last trained model's held-out rows with a repeats knob, replaces the parity figure like the calibration/learning-curve buttons before it, and rides the card's existing Save-figure export. Fractional/zero repeats are named in a dialog before anything runs. |
| The `ecfp3` family (chem/features.py) | `Ml/D2.py`'s R3 fingerprint half as a first-class family: radius-3 Morgan at the toolkit's 2048-bit width, prefix `ecfp3_bit_`, canonical order right after `ecfp`, constants `ECFP3_BITS`/`ECFP3_RADIUS` exported. The honest divergence note: D2.py's R2 half ran 4096 bits where the toolkit's `ecfp` runs 2048 — combining `ecfp + ecfp3` covers the dual-radius story. The feature factory gained the checkbox ("ECFP3 bits — radius 3 (2048)", ticked by default) and the width preview/families session map pick it up for free. |
| The ragged-semicolon doctest (io.py) | Round 29's open candidate (c): `read_table_with_progress`'s docstring now SHOWS the parity verdict the tests pinned — a sniffed `;` separator choking on a ragged body delegates to the sync reader's own mangled-frame verdict instead of raising past it. |

The axes after the round: rar 118, uploads 22/22 (the "کد علی" zip's 8
files join the 14 audited before — six already-covered verdicts
documented above, two surfacing as this round's features), zero-hit 0.
37 new tests: TestPreparePic50Dataset (test_converters.py ×12),
TestPermutationImportance (test_ml_pipeline.py ×7), the permutation
viz tests (test_viz.py ×5), TestEcfp3Family (test_features.py ×4),
TestReview145Gui (test_playground.py ×15). Four standing family-list
pins updated for ecfp3. docs/INDEX.md re-synced (285 symbols). The
version chain moved to 1.2.84 across all five carriers.

### Round 31 (100x-146) — the fourth engineering-details round: the indexed progressive reader, the chunk-boundary guard, the loader phase notes, the permutation Save-CSV twin

| What shipped | The contract |
| --- | --- |
| The one-by-one audit (subagent, read-only) | Round-145's own additions end to end (the prep card, the permutation button, the ecfp3 family, the four library functions) plus the never-audited module tail (print_banner / self_test / main and the 18438+ helpers). Harvest: 1 real parity break + 1 stale-state risk + 4 polish items, every finding probe-verified; 10 candidate concerns disproven with reasons (the session-ride of the new widgets, the ecfp3 map escape, the prep adoption invariant, the permutation worker purity, the needs-run guard ordering, the chunked double-reporting, three io candidates, the hidden-tail-edit census). |
| `_guard_ragged_boundaries` (utils/io.py) | The audit's headline finding fixed: pandas' chunked C parser silently field-truncates a ragged row landing exactly on a chunk boundary (the plain reader raises for the same bytes) — the guard scans the physical lines once and rejects such files with the sync reader's own verdict. Scope honestly bounded: quote-bearing and interior-blank-line files skip the guard (lines cannot align to records there; pandas' own mid-chunk raises still cover them), short rows need none (both parsers pad with NaN), and the boundary arithmetic holds for chunk_rows=1 (the first-run catch). Wired into BOTH drain attempts of the separator ladder. |
| `read_indexed_table_with_progress` (utils/io.py) | The new public indexed twin: the expression loader's plain `pd.read_csv(index_col=...)` pair with determinate progress. The index is preserved through the chunked concat; the label-missing ValueError fires at the reader's first chunk, and only that empty-chunks case re-drains with the positional fallback (a LATER ValueError propagates for its honest verdict); no separator sniffing (comma on both paths — a semicolon table is one mangled column below AND above the threshold); small files keep the plain pair byte-for-byte. |
| The expression-matrix loader (Cell Lines) | The last open-ended marquee in the loader story (standing "by design" since round 28) streams through the new reader: a wide matrix carries the determinate rows/ETA bar; the `cell_line` label index with its positional fallback keeps its exact semantics. |
| The screen loader's phase notes (Cell Lines) | After the streaming read, the status bar carries "auditing rows" and (box ticked) "collapsing replicates" counter-style notes — the AUDIT/COLLAPSE phases are never a silently frozen bar on a huge screen. The collapse flag is still captured on the main thread (the round-143 pin, updated to the new if/else shape). |
| The permutation Save-CSV twin (Results card) | "Save CSV…" writes the last run's table (feature, importance_mean, importance_std) through the shared save-frame dialog; "Save figure…" keeps the picture. The table retires when a new model TRAINS (it follows the run, not the dataset). |
| The stale derived-state fix (Data & ML) | The audit's §2: loading a new dataset now clears `_ml_cleaned_frame` and `_ml_prep_frame` — the Clean/prep Save buttons used to write the PREVIOUS dataset's table while claiming to describe the new one. |
| The out_col collision guard (chem/converters.py) | The audit's §3: `prepare_pic50_dataset(out_col="SMILES")` used to silently clobber the SMILES column via a duplicate dict key — the three reserved result names are refused up front with a typed error. |
| The honesty polish | The module docstring's stale "8-family, 10,000+-column" → "9-family, ~12,860-column"; `selection_metadata`'s "six fingerprint families" → seven; the Save-figure docstring/empty-hint name the permutation figure; the prep card gates pandas like its Clean-card sibling. |

The axes after the round: rar 118, uploads 22/22 (unchanged — no new
upload this round), zero-hit 0. 37 new tests: TestGuardRaggedBoundaries
(test_io.py ×8), TestReadIndexedTableWithProgress (test_io.py ×9),
the out_col collision pair (test_converters.py ×3 parametrized),
TestReview146Gui (test_playground.py ×17). One standing pin updated
(the round-143 collapse-capture contract, its line deliberately
changed by the phase notes). docs/INDEX.md re-synced (286 symbols).
The version chain moved to 1.2.85 across all five carriers.

### Round 32 (100x-148) — the one-by-one notebook review round: both uploads' missing halves shipped (20 families, PR curves, the IC50 triage, the family benchmark)

| What shipped | The contract |
| --- | --- |
| The cell-by-cell notebook review | Both freshly uploaded notebooks read cell by cell (xxx.ipynb: 35 cells — the 3-class classification pipeline; fak_ml.ipynb: 183 cells — the BindingDB SYK pIC50 regression), each capability mapped against the toolkit (~70% covered), every gap below shipped this round. |
| Ten datamol fingerprint families | maccs (167), fcfp + fcfp_count (2048 each), topological + topological_count (the TORSION generator, datamol's own mapping — a first draft used the path generator and silently duplicated rdkit_fingerprint), rdkit_count (numBitsPerFeature=1, datamol's count default), pattern, layered (2048 each), estate (79 counts), erg (315 floats, ±inf→NaN). Every family byte-exact against dm.to_fp (TestFakmlFamilies, tests/test_features.py ×10 incl. the ethanol-torsion zero-vector where datamol's own fp_to_array crashes). |
| shape_3d (xxx.ipynb cell 6's 3D block) | compute_3d_descriptors: ETKDGv3 seed-42 conformers, MMFF94s→UFF fallback (typing failures included), lowest-energy conformer, the nine rdMolDescriptors shape descriptors; failed pipelines keep NaN rows; single atoms embed as honest degenerate zeros (probed — [Cl-], [He] all embed on this build). |
| pic50_to_ic50 / add_ic50_column | fak_ml's dm.molar.log_to_molar inverse with exponent arithmetic (integer pIC50s land on exact powers; datamol itself returns 999.9999999999999 for 6.0); series + table twins, NaN/±inf honesty, unit validation (tests/test_converters.py ×21). |
| plot_pr_curves | xxx.ipynb cell 28's OvR PR curves (previously absent toolkit-wide) with average precision per class and the no-skill prevalence baselines the notebook's own plot omitted (tests/test_viz.py ×5). |
| ClassificationReport.per_class | The notebook's per-class precision/recall/f1/support table, str(label) keys, backward-compatible default None (tests/test_multiclass.py ×5). |
| sgd_classifier | The notebook's 13th classifier; hinge/log_loss/modified_huber grid; early_stopping deliberately NOT the factory default (breaks screens < 2×n_classes — pinned in notes + tests, tests/test_ml_pipeline.py ×6). Registry counts 28 → 29. |
| benchmark_feature_families | fak_ml's test_fp loop (one LGBM per fingerprint) as a compare_models-convention family leaderboard over the 18 2D families (3D families opt-in), progress callback, NaN rows sink (tests/test_ml_pipeline.py ×11). |
| The playground GUI | The family picker at 20 ticked checkboxes (honest widths, library-order labels), live ROC + PR buttons on the Results card, the per-class metrics line, the Predict card's pIC50→IC50 button + potency-threshold triage (≤100/300/500/1000 nM, full table preserved), the Leaderboard's family-benchmark row, sgd on the model picker (TestReview148Gui, tests/test_playground.py ×15). |
| Version / docs | 1.2.86 across the five carriers; CHANGELOG [1.2.86]; docs/INDEX.md re-synced (301 symbols — the new chem/ml/viz exports ride the package __all__); README test pins updated. |

### Round 33 (100x-150) — the fifth engineering review: the audit's three findings + the board's Save-table twin

| What shipped | The contract |
| --- | --- |
| The static audit (Task 1-a) | The round-31 additions (the least-reviewed code in the tree — that round's subagent entries were lost to context deadlines) audited line by line: 3 real defects found, 10 suspected findings disproven and recorded (scratch/audit_150.txt — the honesty ledger), every other area verified clean (the 3D descriptors, all ten fingerprint families, prefix uniqueness, per_class, plot_pr_curves, sgd_classifier, the family bitmap, docs INDEX). |
| benchmark_feature_families' hoisted guard | The docstring's promised "fewer aligned rows than cv" ModelError was unreachable (the per-family except swallowed it) — a 2-row dataset with cv=5 degraded into a 100%-NaN board. The compare_models 27-A guard is hoisted above the loop; the per-family drop path honestly keeps its NaN row (test_ml_pipeline.py ×2 new). |
| pic50_to_ic50's OverflowError | 10**(exponent−p) raises for finite pIC50 ≲ −299 (probed at the boundary: −299.25 finite, −299.26 raises) — one absurd cell killed a whole add_ic50_column job with a raw traceback. Now returns the mathematical limit inf (mirroring the underflow end's silent 0.0); inf correctly FAILS the potency triage (test_converters.py ×3 new incl. the boundary + the triage verdict). |
| The add_*_column collision guards | add_ic50_column/add_pic50_column silently clobbered on out_col collision (round-146's guard stopped one function short); both refuse now ("appends, never overwrites"), and the GUI's two call sites drop the stale column first so a re-click is an explicit recompute (test_converters.py ×6 parametrized + test_playground.py ×2 new). |
| The board's "Save table…" button | Every frame that fills the shared leaderboard (compare_models, benchmark_feature_families, the OOF stacking card, hyperparameter searches) exports through the five-format dialog — the permutation card's 100x-146 twin, completed (test_playground.py ×4 new incl. the write-through). |
| The environment repairs | The venv's 11th reset (rdkit/torch/mordred) repaired — foreground pip (the sandbox reaps background installs mid-download); the corrupted file modes (every tracked file executable → 115 EXE002 ruff errors) restored from the git index; the dev server now rides a double-fork daemon (scratch/dev_daemon_150.py) the session reaper cannot kill. |
| Version / docs | 1.2.88 across the five carriers; CHANGELOG [1.2.88]; README pins 2965; suite 2971 collected = 2965 passed + 6 skipped / 0 failures (+17 over 1.2.87). |

### Round 34 (100x-151) — the sixth engineering review: the D2.py prediction intervals ported honestly

| What shipped | The contract |
| --- | --- |
| The static audit (Task 1-a) | The round-33 additions audited line by line: 3 real defects + 1 note-grade polish found, 17 suspected findings disproven and recorded (scratch/audit_151.txt + scratch/probe_audit_151.py — the honesty ledger, leak-scan clean). The hoisted benchmark guard, the inf convention end-to-end (triage semantics included), the collision guards, both recompute drops, the board export index handling all verified clean. |
| The tune render's missing `_board_frame` (Finding 1, RISK) | The board's Save-table twin promised FOUR fillers; the hyperparameter render filled only the tree — right after a search the button lied or exported the PREVIOUS run's stale leaderboard (a cross-KIND stale export). The render now builds the trials frame (full configuration text, float scores, ranked 1..n, failed trials last); pinned behaviorally ×2 + a four-filler source pin (tests/test_playground.py TestReview151Gui ×3). |
| The lex-side overflow net (Finding 2) | pic50_to_ic50's `float()` net caught (TypeError, ValueError) but a Python bigint (10**400 through an object-dtype column) raises OverflowError — the same absurd-cell class the pow-side guard contains. OverflowError joined the net → None (test_converters.py ×1 new + the column round-trip). |
| The blank out_col guards (Finding 3) | out_col=None/""/whitespace passed all three append siblings' collision guards and pandas appended a None-named column. All three (add_ic50_column, add_pic50_column, prepare_pic50_dataset) refuse blank labels with the same typed error (test_converters.py ×9 parametrized). |
| quantile_prediction_intervals | The D2.py uncertainty block, ported honestly: the upload trained three IDENTICAL plain forests (no quantile loss — RF has no quantile mode in sklearn), so its advertised 90% interval had width 0.0 and coverage 0.0% (probed). The port fits GradientBoostingRegressor(loss="quantile") at α/2 / 0.5 / 1−α/2 — typed QuantileIntervals (coverage, mean width, crossings counted AND repaired by min/max swap), deterministic-by-default honesty (subsample=1.0 has no stochastic element; the seed works only when subsample < 1 — pinned so the knob cannot regress into a placebo), progress callback ×3 with the swallow convention (tests/test_applicability.py TestQuantilePredictionIntervals ×18). |
| plot_prediction_intervals | The GP ±σ plot's twin for bracket-shaped uncertainty: parity scatter of the median fit, the band through the test rows sorted by actual value (the upload's calibration-style choice, kept deliberately), coverage + mean-width box, crossed-bounds guard (tests/test_viz.py TestPredictionIntervalsPlot ×8 incl. the end-to-end from the record). |
| The GUI twin | The Results card's "Prediction intervals" button (regression-only gate, α entry, the honest coverage/width/crossing note) + "Save intervals…" (the five-format table export, retiring on the next train) — TestPredictionIntervalsGui ×8, smoke151 (84.6% live coverage on 52 held-out rows) + shoot151 (download/pg151-ml-intervals.png). |
| The notebook-ritual by-catch | The committed notebook outputs printed 1.2.87 against a 1.2.88 tree — round 33's re-execution ritual had run BEFORE the version bump (a latent ordering defect, surfaced by this round's suite). The ritual ran after the 1.2.89 bump; all 119 variant tests green. |
| Version / docs | 1.2.89 across the five carriers; CHANGELOG [1.2.89]; README pins 3012; suite 3018 collected = 3012 passed + 6 skipped / 0 failures (+47 over 1.2.88); docs/INDEX.md re-synced (307 symbols); README gained the "Prediction intervals" section before hsvr. |

### Round 35 (100x-152) — the seventh engineering review: four D2.py audit gaps closed + the untracked-files wiring

| What shipped | The contract |
| --- | --- |
| The environment repair (12th reset) | The sandbox reset lost rdkit/torch/mordred/meeko/gemmi/datamol/ruff/mypy AND the pip executable — foreground `python -m pip` repair (the standing recipe: never background), numpy pinned back to 1.26.4, torch 2.14.1+cpu, editable reinstall; the tracked files' +x corruption restored from the git index (117 EXE002 → 0, with two never-tracked files chmod'd directly); Xvfb :99 respawned. |
| The static audit (Task 1-a) | The round-34 additions audited line by line: the quantile-intervals FEATURE verified clean (guard parity, honest coverage/width/crossing, seed-honesty, thread-safe wiring, five-format save twin), 20 suspected findings disproven and recorded (scratch/audit_152.txt + probe_audit_152.py). The round's real risk was elsewhere: the untracked strays. |
| The untracked-files finding (Finding 1, the red-suite bomb) | bioisostere.py had shipped in every ZIP since its round but was never git-added and never wired: imported by nothing, its own doctest importing `from pharmaml.generative import bioisostere_moves` and FAILING — a live failure in every bare `pytest` (3019 collected / 1 failed, .pytest_cache/lastfailed the witness). The module's five doctest claims verified live with RDKit (sound), so it was WIRED, not deleted: package exports + the GA's mutation="bioisostere" mode + 7 tests. __main__.py committed alongside; the web orphan data-file-preview.tsx left untracked (zero importers, superseded). |
| decode_morgan_bit + MorganBitInfo (GAP 2) | The bit-index → atom-environment decoding, typed: every (atom, radius) hit decoded with THE HIT'S OWN radius (the upload passed the fingerprint radius — a real decode bug for any bit firing below max radius, fixed and pinned), radius-0 hits KEPT (the upload dropped them all silently), the DrawMorganBit SVG legend as .drawing. 61 tests in tests/test_interpret.py across the whole bundle + 3 doctests (real molecule, real bit). |
| signed_feature_correlations + the two plots (GAP 2) | Every numeric column vs the target, |r| descending with sign kept, pearson AND spearman; undefined correlations dropped never zeroed (the upload appended a fake 0.0 wherever pearsonr raised). plot_signed_correlations (the signed top-N bar, the permutation-importance colour convention) + plot_correlation_heatmap (pure matplotlib, the diverging house palette, zero-centred, grey undefined cells, annotations only ≤ 12 columns — the upload called seaborn). |
| cross_model_importances + plot_permutation_heatmap (GAP 3) | The features × models matrix: one PermutationImportance per fitted model keyed by name (same seed for every model — cross-model differences reflect the models, not the shuffle luck, pinned by the same-estimator-twice test; a failing model is NAMED in the typed error), and the heatmap with a grey "no vote" cell for a feature a model never saw — honestly absent instead of a fake zero. 5 runner tests + 10 heatmap tests. |
| filter_low_variance + VarianceFilterResult (GAP 6) | The xxx.ipynb VarianceThreshold(0.01) step: rank_features RANKS by variance, this CUTS by it, same fillna(0).var() definition so the twins can never disagree (pinned); the dropped ledger carries every cut's variance as evidence; non-numeric columns pass through with a warning. A mid-round test caught the first cut silently DROPPING the carrier columns it promised to pass through — fixed before ship (the test that caught it ships too). 12 tests. |
| RegressionReport's train side + r2_gap (GAP 7) | evaluate_model(..., X_train=, y_train=) fills pearson_r / train_rmse / train_r2 / train_pearson_r / r2_gap (the overfit gap as one signed number); all-or-nothing pair (half-passed raises), regression-only (classification rejects with the typed error), the same NaN/inf/length guard family the test side carries, __str__/as_dict extended (the notebooks' str-pinned outputs re-executed through the standing ritual). 11 tests incl. the unpruned-tree gap > 0.3 use case. |
| The GUI (+782 lines) | Four surfaces: the regression results' "Pearson r / train R² / R² gap (train−test)" line (tolerant of older persisted reports, **train_side at all three regression evaluate sites, the CV-search site deliberately skipped); the Results card's THIRD button row (Correlations… / Correlation heatmap / Decode ECFP bit… — the two-row discipline held, defaults derived from the last importance run and the loaded dataset); the leaderboard's Cross-model importance…; the feature factory's Filter low variance… row with frame adoption (the prep card's pattern; the adoption body now also retires _stack_last — the stale-state belt-and-braces, the updated source pin documents it). 20 GUI tests (TestOverfitGapGui / TestInterpretabilityGui / TestCrossModelImportanceGui / TestVarianceFilterGui). |
| The close-out | 1.2.90 across the five carriers; CHANGELOG [1.2.90]; README gained the interpretability-bundle section + the overfit-gap bullet; docs/INDEX.md 307 → 319 symbols; notebooks re-executed AFTER the bump (the round-33 ordering lesson, 196 s, 119 variant tests green); the leakage guard's mid-round catch (four resurrected untracked scratch files from the environment reset) scrubbed through the standing script. |

### Round 36 (100x-153) — the eighth engineering review: the round-35 additions audited, one red + three ambers closed, the web interpretability twin

| What shipped | The contract |
| --- | --- |
| The audit (Task 1-a, subagent completed by the orchestrator after its context-deadline death — the standing pattern) | The round-35 additions audited with six adversarial probes (scratch/probe_audit_153_{a,c,d,e,f,g2}.py, verdicts in scratch/audit_153.txt): interpret.py, the three plots, filter_low_variance's edges, the GA bioisostere wiring's statistics bookkeeping, evaluation's train side, cross_model_importances' seed discipline, the GUI's third button row. ~20 suspicions DISPROVEN (the GA's "draws 37 ≠ 30" was the auditor's wrong expected-formula — the breeding loop's duplicate-rejection retries are attempt-based accounting exactly as the _OperatorBook docstring documents; the seed alignment, every typed guard probed, and the signed-bars twin's honesty all hold). |
| The RED: the GUI heatmap's target join is now SWAP-PROOF | The handler joined the target from the SESSION frame even though the run tuple carries the honest y_test — a same-shaped swapped dataset (RangeIndex covering the split's indices) silently passed the "rows align" guard and paired the new dataset's y with the old run's features (probe g2 read a-y = +0.17, noise, where the run's own target gives +1.00). The join now takes the run's OWN y_test (aligned by construction, swap-proof, and no longer racing a concurrent dataset load — the 100x-142 concern retired with the consultation itself); the GUI regression test reproduces the exact natural swap flow and pins the +1.00 cell (tests/test_playground.py TestInterpretabilityGui). |
| The ambers: filter_low_variance ×2 | (1) inf-poisoned columns CUT, not silently kept: after the fillna(0) convention a NaN variance can only arise from inf/-inf cells — a column every downstream estimator rejects; the ledger entry is NaN (undefined — the same verdict rank_features already reports, the twins stay in agreement), the warning names the poison. (2) The threshold guard fires BEFORE float() coerces — float(True) is 1.0 and float('0.01') is 0.01, but neither is an honest threshold (the house guard family rule). 4 new tests (incl. the ranker-agreement pin). |
| The amber: signed_feature_correlations skips complex columns | pandas counts complex128 "numeric" and the dtype-float cast silently discards the imaginary part (an unread ComplexWarning) — a half-honest correlation; complex columns now ride the non-numeric skip path with the naming warning. 1 new test. |
| The web interpretability twin (the standing candidate) | The ML playground's results column gained the Feature–target correlations card: signed Pearson-r bars (teal pulls the target up, rose pulls it down, undefined dropped never zeroed) + the compact top-N correlation matrix (rose→white→teal diverging scale, grey undefined cells, −1/+1 gradient legend), both computed over the run's OWN held-out split IN the same request that trained the model — inherently swap-proof. The pure core is the shipped pearsonCorrelation in src/lib/ml-math.ts (returns NaN never a fake 0; rejects length mismatch rather than silently truncating), pinned against scipy.stats.pearsonr by the 100x-153 parity pair (scratch/verify_corr_parity_153.py + scratch/verify_corr_parity_153.ts, 11-case seeded grid, worst |Δr| = 8.3e-17, NaN-agreement on every undefined case) wired into tests/test_web_parity.py as TestCorrelationParity. |
| The close-out | 1.2.91 across the five carriers; CHANGELOG [1.2.91]; README pins 3149 → 3157 (×2) + the interpretability-web-twin bullet; suite 3162 collected = 3157 passed + 5 skipped / 0 failures (+8); ruff clean (the round also fixed a round-35 leftover: genetic.py's bioisostere import order); mypy src/ clean (69 files); notebooks re-executed AFTER the bump (200 s; the first attempt's DeadKernelError was the dev server's 1.5 GB — the standing OOM ritual, dev server stopped first). |

### Round 37 (100x-154) — the ninth engineering review: the round-36 web twin audited ADVERSARIALLY, a fake-zero RED closed in BOTH correlation twins, the classification bundle, and SHAP (the last D2.py gap)

| What shipped | The contract |
| --- | --- |
| The audit (Task 1-a, orchestrator-run — the three-round subagent-death pattern retired) | The round-36 additions audited with adversarial numerics probes the 100x-153 parity grid never asked (scratch/probe_audit_154_{a,b,c}.py, verdicts in scratch/audit_154.txt): pearsonCorrelation's extreme-scale behavior, the live route's bundle on every path, the parity grid's breadth, the GUI join's ndarray fallback. ~10 suspicions DISPROVEN (the ?? 0 reachability, duplicate React keys, matrix truncation, the <2 guard reachability, stable-sort ties, spearman honesty, the Series non-covering join, big_n drift, the empty-state note, JSON NaN). Two self-catches recorded: the first transcription of the fix dropped the mean subtraction (the uncentered cosine is NOT Pearson r — the new parity pair caught it in minutes) and the audit probe's own TS runner SHARED /tmp paths with the parity twin (a stale output poisoned the cross-language check — the same stale-shared-state bug class the audit hunts). |
| The fake-zero RED (BOTH twins) | A constant 1e300-scale column produced a fabricated r = 0.0 — web pearsonCorrelation AND desktop signed_feature_correlations (np.corrcoef, the identical arithmetic) — while scipy honestly returns NaN + ConstantInputWarning; LIVE in the route (ridge + a 1e300-constant CSV column rendered "big +0.0000 (weak)"). The fix: the offset-then-scale recipe in BOTH languages (first-element re-centring is exact under Sterbenz — a constant column collapses to exact zeros at ANY scale; max-deviation scaling keeps every product finite; the scaled-mean subtraction is required — the offset alone leaves the n(x̄−x₀)(ȳ−y₀) cross term). Pinned: the const_huge/const_tiny/const_plain fake-zero cases must be NaN on both sides. |
| The lost-signal REDs (BOTH twins) | Extreme-scale VARYING columns (1e300 overflow / 1e-300 underflow) were silently dropped where scipy reports a defined r; the recipe keeps them, matching scipy bit-for-bit (2.8e-17). The desktop drop-note wording gains "numerically undefined at extreme scales". 3 new test_interpret tests (the fake-zero regression, the recovered features vs scipy, the huge-offset analytic-truth pin). |
| The honest tolerance story + the NEW parity pair | scratch/verify_corr_parity_154.py + .ts: a 14-case adversarial grid (huge_magnitude/huge_offset/near_constant/tiny_values/alternating/two_points/big_n/neg_zero + the constant fake-zero pins + mismatch/one_point/nan_pair), both sides running the SHIPPED code, wired as TestCorrelationParityAdversarial in tests/test_web_parity.py. huge_offset: the irreducible input-quantization floor gets 5e-3 vs scipy PLUS the analytic-truth pin |recipe−1.0| ≤ 2e-3 (the recipe is STRICTLY closer to truth than scipy: 0.9990 vs 0.9968); near_constant: scipy's own NearConstantInputWarning territory at 1e-7 vs scipy PLUS the analytic-truth pin at 1e-12. Worst |dr| on the default cases: 2.78e-17. The 153 pair's mirror updated to the shipped recipe (its docstring claims line-by-line mirroring) — still green at 2.2e-16. |
| The classification bundle (the parity gap) | The web route's runClassification now builds the correlation bundle over the run's OWN held-out rows (the point-biserial r against the 0/1 label — the desktop twin's own no-task-guard read); the CorrelationCard gains the task prop + the point-biserial note. LIVE-VERIFIED: the synthetic moons bundle (x2 −0.7626 / x1 +0.5429), the 2-feature CSV path, the 1e300-constant column now DROPPED, the 1e300-VARYING column now reported. |
| The GUI ndarray fallback + its pin | The heatmap join's fallback requires exact length (the old values[:len(frame)] silently truncated a LONGER y_test); pinned end-to-end — the library's typed length guard fires first at the handler level (defense in depth), 1 new GUI test. |
| SHAP — the LAST unabsorbed D2.py gap (its lines 472-528) | pharmaml.ml.explain (shap_summary → the typed ShapSummary record: names, mean|SHAP|, the full values matrix, the feature values, the base value; TreeExplainer for LightGBM/XGBoost/CatBoost/sklearn trees; the DependencyError carries the exact pip install 'pharmaml[explain]' command; non-tree estimators and multiclass get the typed error naming the permutation_feature_importance sibling — never a silent KernelExplainer fallback) + viz plot_shap_importance (the upload's own top-20 bar) + plot_shap_beeswarm (PURE matplotlib: jittered (sample, feature) points coloured by the feature's own value on the house amber→teal ramp, seeded jitter, base-value caption — the plotting layer never needs shap). The pharmaml[explain] extra (shap>=0.45,<0.60, runtime-verified on numpy 1.26.4; requirements.txt documents the resolver-warning story). 25 new tests (tests/test_explain.py: the record, both figures, the seeded-jitter determinism, the guard simulation, the real TreeExplainer path incl. the binary positive-class read and every typed refusal) + 4 GUI tests (both buttons end-to-end, the no-run guard, the non-tree sibling pointer). The D2.py upload is now FULLY absorbed. |
| The close-out | 1.2.92 across the five carriers; CHANGELOG [1.2.92] with the verification paragraph; README pins 3157 → 3196 (×2) + the SHAP section; suite 3201 collected = 3196 passed + 5 skipped / 0 failures (+39: 25 explain + 4 GUI-SHAP + 1 GUI-ndarray + 3 interpret + 2 parity + 4 explain doctests) after the notebook freshness guard fired and the eight notebooks were re-executed against the bumped tree (200 s); ruff clean (the 14th venv reset's +x corruption de-exec'd: 595 tracked files restored from the git index — the tab-delimited ls-files recipe, no double-prefix); mypy src/ clean (70 files); web pins 3157 → 3196 (page.tsx ×1, overview.tsx ×3, download-panel.tsx ×3) + tsc(src) clean after the dying session's uncompilable data-file-preview.tsx duplicate was folded into data-file-view.tsx (row-index column, empty-cell em-dash, cell max-width truncation) and deleted; the two SHAP screenshots VLM-reviewed 5/5; zip v1.2.92 (635 entries, 12.95 MB, testzip clean, leak-scan clean, no nested zips); the D2.py upload fully absorbed. |

## The Python Data Science Handbook review (round 40, 100x-157)

The instruction: *"Check the contents and files in this GitHub link
completely and one by one and add the missing items to playground.py:
jakevdp/pythondatasciencehandbook."* The clone (264 tracked files,
commit d662314) was reviewed **completely and one by one** — the full
method + the per-notebook table (all 67 `.ipynb` under `notebooks/`)
are reproduced here; the extraction tooling and the raw output live at
`scratch/pdsh_extract.py` / `scratch/pdsh_review.md`. The dispositions
were assigned against v1.2.92 **before** this round's absorption code
landed (grep-verified), so COVERED-ALREADY rows name the surfaces that
already existed and ADDED-THIS-ROUND rows are exactly what v1.2.93
ships: `ml/unsupervised.py` (PCA / K-Means / GMM / KDE),
`ml/manifold.py` (Isomap / LLE / MDS / Spectral / t-SNE), the
naive-Bayes ModelSpecs (`gaussian_nb` / `multinomial_nb` /
`bernoulli_nb` + the `NO_SCALE_MODELS` carve-out), the six
`viz/ml_plots` figures, and the playground's Unsupervised tab.

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

### Round-40 verification rows

| What | Where it landed |
| --- | --- |
| `ml/unsupervised.py` | `pca_reduce` / `kmeans_cluster` / `gmm_cluster` / `kde_fit` + the four typed records; 33 tests (tests/test_unsupervised.py) + the module doctest. |
| `ml/manifold.py` | `manifold_embedding` (Isomap / LLE×3 / MDS / Spectral / t-SNE) + `EmbeddingRecord`; 20 tests (tests/test_manifold.py) + the module doctest. |
| Naive Bayes (05.05) | `gaussian_nb` / `multinomial_nb` / `bernoulli_nb` ModelSpecs + `NO_SCALE_MODELS` (multinomial trains without the scaler — counts stay counts); 9 tests (tests/test_ml_pipeline.py::TestNaiveBayes). |
| The six figures | `plot_pca_variance`, `plot_kmeans_diagnostics`, `plot_gmm_selection`, `plot_kde_density`, `plot_manifold_embedding`, `plot_hexbin_density`; 12 tests (tests/test_viz.py::TestUnsupervisedPlots) + 3 new doctests. |
| The Unsupervised tab | playground.py `_build_unsupervised_tab` (21,009 → 21,562 lines): shared dataset + column tree, ten methods, per-method params, the record's summary, canonical figure, 2-D scatter twin, CSV export; 11 GUI tests (tests/test_playground.py::TestUnsupervisedTab). |
| The artifacts | download/pg157-unsup-{pca-variance,isomap-embedding,kmeans-diagnostics,gmm-selection,kde-density,hexbin}.png — VLM-reviewed 6/6 PASS. |

## The DearPyGui review (round 41, 100x-158) — improving playground.py's GUI

Repository: <https://github.com/hoffstadt/DearPyGui> (MIT). Reviewed from a
`--depth 1 --filter=blob:none` clone (266 tracked files: `src/` 109 — the C++
core, `thirdparty/` 65 — Dear ImGui + ImPlot + imnodes + the UTF/font stack,
`docs/` 44 — the Sphinx sources, `dearpygui/` 10 — the Python wrapper +
`_dearpygui.pyi` API stubs + `demo.py`, `.github/` 14, `scripts/` 6, `sandbox/`
4, `assets/` 3, `testing/` 2). The feature review below is grounded in the
README's feature list, the `docs/source/documentation/*.rst` topics and the
`_dearpygui.pyi` API surface (`add_tooltip`, `add_table`, `add_plot` with
`box_select`/`no_inputs`/`crosshairs`/query flags, `set_global_font_scale`,
`add_collapsing_header`, `add_node_editor`, `add_theme` + `add_theme_color` +
`add_theme_style`, `add_status_bar`, `add_loading_indicator`, `add_file_dialog`,
`add_progress_bar`, the filter-set and init-file docs).

The review's premise: DearPyGui is a **GPU-rendered, immediate-mode-under-the-
hood framework** — the toolkit's GUI is a tkinter app and stays one (a rewrite
would discard 21k lines of battle-tested UI for zero pharma-ML capability).
What the review mines instead is DearPyGui's *UX vocabulary* — the affordances
its users take for granted — and asks which ones tkinter can deliver honestly.

### Feature-by-feature disposition

| DearPyGui feature (source) | Disposition | Where it landed |
| --- | --- | --- |
| Interactive plots — pan, zoom, box-select, query rects, live coordinate readout (`docs/plots.rst`: Pan/Zoom/Zoom Region/Zoom Extents + Querying) | **ADDED-THIS-ROUND** | Every embedded TkAgg figure now carries a styled `NavigationToolbar2Tk` (home/pan/zoom/save + the readout label); the matplotlib 3.10 bump is what makes this possible on Tcl 9 stacks (3.9's `_tkagg` could not talk to Tcl 9 — those users got static PNG figures). |
| `crosshairs` / cross-highlight under the cursor (plot flag, `_dearpygui.pyi`) | **ADDED-THIS-ROUND** | `_attach_crosshair`: two dashed guides via `add_artist` (never `add_line` — that joins the data limits and autoscale-drifts the panel), hidden on leave, colour-follows-theme. |
| Tables — row stripes, hover row highlight (`mvThemeCol_TableRowBgAlt` / `TableRowBgHovered`) | **ADDED-THIS-ROUND** | `attach_table_chrome` on all 29 Treeviews: zebra stripes + hover highlight; the 150 ms poll re-stripes on row-count change so all ~59 populate paths gain the stripes unedited. |
| Sortable table columns (dpg table demo behaviour) | **ADDED-THIS-ROUND** | Heading-click sort, numeric-aware (thousands separators tolerated, mixed columns stay lexicographic), direction toggling with ↑/↓ arrows, other columns' arrows cleared. |
| `set_global_font_scale` (`_dearpygui.pyi`) | **ADDED-THIS-ROUND** | `View ▸ Font size` (9–12 pt): the named Tk fonts reconfigure live and `_build_style` re-derives every bold/large variant from the base. |
| `add_tooltip` per item (`docs/tooltips.rst`) | COVERED-ALREADY | The `Tooltip` class (75 attachment sites since earlier rounds). |
| Themes — `add_theme` + color/style components, light/dark | COVERED-ALREADY | The two named palettes + `apply_theme` + the live recolour walk (View ▸ Follow system / Light / Dark). |
| `add_status_bar`, `add_progress_bar`, async jobs | COVERED-ALREADY | The status bar + indeterminate/determinate progress + the thread/queue job machinery. |
| `add_file_dialog` | COVERED-ALREADY | tkinter file dialogs (the session's pick-path memory). |
| `add_loading_indicator` | COVERED-ALREADY(≈) | The status-bar progress bar serves the same role; a modal spinner would *block* the Tk loop — the honest Tk translation of "busy" is the non-blocking bar. |
| `add_collapsing_header` | STRETCH | A `CollapsibleCard` wrapper is feasible (header toggles the body) but the tabs' card flow is scroll-designed; retrofit deferred until a tab is actually too tall. |
| Dockspace / `primary_window` docking | N-A-HONEST | tkinter has `PanedWindow` but the toolkit's card-per-scroll-page layout is the opposite paradigm; a dock rewrite would discard the scroll UX without adding analysis capability. |
| Node editor (`add_node_editor`, imnodes) | N-A-HONEST | No pipeline-graph surface in a pharma-ML toolkit whose workflows are forms over typed library calls. |
| GPU rendering, 1M-point 60 fps plots | N-A-HONEST | The plots are matplotlib publication figures (paper exports); interactivity is a lens on them, not a rendering engine swap. |
| Filter sets (`docs/filter-set.rst`) | COVERED-ALREADY(≈) | The descriptor/family search rows + the Ctrl+F card finder. |
| Init files (`.ini` window persistence) | COVERED-ALREADY | The session file (geometry + inputs + preferences). |
| Built-in demo / developer tools (theme inspector, metrics, debugger) | STRETCH | The dependency report + `--self-test` cover the introspection role; a live theme inspector has no user demand. |

**Disposition tallies (18 rows):** ADDED-THIS-ROUND **5** · COVERED-ALREADY(≈)
**7** · STRETCH **3** · N-A-HONEST **3**.

### Round-41 verification rows

| What | Where it landed |
| --- | --- |
| Figure toolbars + crosshair | playground.py `_attach_figure_toolbar` / `_attach_crosshair` / `_style_figure_toolbar` / `_apply_toolbar_visibility` (View ▸ Figure toolbars, session-persisted); theme-flip retint via the retint walker's new Button/Checkbutton/Label branches. |
| Table chrome | playground.py `attach_table_chrome` / `sweep_table_chrome` / `refresh_table_chrome` / `_sort_tree_column` / `retag_tree_zebra` + the `STRIPE`/`HOVER` palette keys; the card-finder dialog's per-dialog tree attaches at creation. |
| Font scale | `View ▸ Font size` + `_apply_font_size` + `_build_style`'s derived sizes; session-persisted via `_font_var`. |
| The matplotlib 3.10 bump | `requirements.txt` verified-stack note (3.9.2 → 3.10.9: Tcl 9 stacks get live TkAgg canvases instead of PNG fallbacks — the toolbar's enabling change). |
| Tests | tests/test_playground.py::TestDearPyGuiAffordances (16 tests: zebra, sweep + prune, numeric/thousands/case/blank sorting, arrow toggling + cross-column clearing, hover + stripe restore, theme retint, toolbar attach/style/toggle/replace, crosshair + no-drift, font scale, session persistence) + the seven figure-landing pins updated to the toolbar-aware contract. |

## The DearPyGui review, second pass (round 42, 100x-159) — closing the stretch rows

The instruction arrived again (*"Improve the playground.py GUI:
DearPyGui"*) the round after the first pass shipped — and the interim
state demanded it: the round-41 session had died mid-close-out (a
UUID-message commit, no worklog entry, the notebook re-execution and
three green-suite fixes never ran; see the round-42 worklog post-mortem).
This round closed that debt AND the disposition table's two STRETCH
rows, translated honestly to tkinter:

| DearPyGui feature (source) | Round-41 disposition | Round-42 disposition | Where it landed |
| --- | --- | --- | --- |
| `add_collapsing_header` | STRETCH ("deferred until a tab is actually too tall") | **ADDED-THIS-ROUND** | `make_card(collapsible=True, key=...)`: a ▾/▸ chevron on the header folds the card to its heading (inputs keep their values, only the body hides). Retrofitted onto the two static Home reference cards (Dependency status, What lives where) + all thirteen PK-tab calculators — the tallest tab, fold the calculators you are not using. The fold persists with the session (`collapsed_cards` text key, the tolerant JSON trick), a finder jump unfolds the target first, and the per-tab reset restores the launch look (other tabs keep their folds). |
| Built-in demo / developer tools (metrics, theme inspector) | STRETCH ("no user demand") | **ADDED-THIS-ROUND** | View ▸ **GUI metrics…**: the live widget census by Tk class (the `_walk_widgets` walk), the card / tree / toolbar / job totals, the active preferences and the session file's state — a snapshot at open, honestly labelled — over the **palette inspector**: one colour chip + hex value per palette key (all 18), re-tinted live on theme flips (`_refresh_metrics_dialog`). |
| Keyboard accelerators & key handlers (`add_item_handler` + key bindings, `docs/keybinds`) | *(the row the dying session's tallies promised but its table never carried)* | **COVERED-ALREADY** | The full accelerator layer predates the review: Ctrl+O/S/Shift+S/Shift+R/F/Q, Ctrl+PgDn/PgUp tab cycling, the Ctrl+F card finder, the hint row that advertises them all (`shortcut_hints_text`) — pinned by `TestShortcutHints` since 100x-65/119/120. |
| Dockspace / node editor / GPU rendering | N-A-HONEST | N-A-HONEST (unchanged) | A tkinter card-per-scroll-page app is the opposite paradigm; the reasoning of round 41 stands. |

**Round-41 tally reconciliation:** the shipped Round-41 table carried 17
rows (5 ADDED · 7 COVERED · 2 STRETCH · 3 N-A) while its tally line
claimed 18 rows / 3 STRETCH — the dying session never reconciled the
two. With the keyboard row now documented above (COVERED-ALREADY), the
review's real disposition is 18 rows: 5 ADDED · 8 COVERED · 2 STRETCH ·
3 N-A — and the two stretch rows are closed by this round.

### Round-42 verification rows

| What | Where it landed |
| --- | --- |
| Collapsible cards | playground.py `_COLLAPSE_CARDS` / `_set_card_expanded` / `_toggle_collapsible_card` / `collapsed_card_keys` + the fifteen retrofitted call sites; session key `collapsed_cards`; finder unfold in `_jump_to_card`; per-tab unfold in `_reset_tab`. |
| GUI metrics + palette inspector | playground.py `_gui_metrics_snapshot` / `_show_gui_metrics` / `_populate_metrics_palette` / `_refresh_metrics_dialog` (View ▸ GUI metrics…); the retint walker's new `Toplevel` branch follows flips too. |
| The round-41 menu fix | `_menu_tints` was re-assigned AFTER the font cascade's append — the font submenu silently lost its theme tint on every flip; the list now carries it. |
| Tests | tests/test_playground.py::TestDearPyGuiRoundTwo (14 tests: the registry, fold/unfold + chevron, quiet noops, finder unfold, session round-trip + corrupt/unknown restore, per-tab reset unfold, dialog census/prefs, census-vs-walk parity, palette chips, theme-flip live retint, reopen-lift/destroy-prune, the font-cascade tint pin). |

## The SHAP + Optuna review (round 43, 100x-160) — two repos, one absorption

The instruction: *"Check the contents and files in this GitHub links
completely and one by one and add the missing items to playground.py:
github.com/shap/shap, github.com/optuna/optuna."*  Both repos cloned
(`--depth 1 --filter=blob:none`) and reviewed file by file — SHAP: 615
tracked files (95 Python modules, the C++ TreeSHAP core, ten demo
datasets); Optuna: 484 tracked files (220 Python modules across
samplers / pruners / importance / visualization / storages).  The
per-area inventory and the full disposition tables live in
`scratch/shap_optuna_review.md` (the pdsh_review.md pattern); the
baseline at review time was v1.2.95, which already shipped the
TreeExplainer surface (100x-154) and the Keras-Tuner programming model
(100x-89/90).  The honest premise, stated once: **neither package
becomes a runtime dependency** — the absorption re-implements the
programming models in typed library surfaces, keeps every figure pure
matplotlib, and optional-gates only the one thing that truly needs the
third-party package (the `shap` computation itself, behind the
`[explain]` extra, exactly as 100x-154 established).

### SHAP (shap/shap) — the disposition

| SHAP capability (source) | Disposition | Where it landed |
| --- | --- | --- |
| `TreeExplainer.shap_values` | COVERED-ALREADY (100x-154) | `ml/explain.py::shap_summary` → the typed `ShapSummary` record. |
| `shap.plots.beeswarm` / `_bar` | COVERED-ALREADY (100x-154) | `viz/ml_plots.py::plot_shap_beeswarm` / `plot_shap_importance`. |
| `shap.plots.waterfall` — the local base→prediction walk | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_shap_waterfall(summary, sample)`: signed cumulative bars (teal up / amber down), dashed end-lines at the base value and the prediction, the tail collapsed into one summed bar; the GUI asks which held-out row. |
| `shap.plots.scatter` (dependence, auto interaction colour) | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_shap_dependence(summary, feature)`: value-vs-SHAP scatter coloured by the strongest interacting feature (the binned-R² heuristic — the upstream's own `approximate_interactions` recipe, honestly documented). |
| `shap.plots.heatmap` | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_shap_heatmap(summary)`: instances × top features, diverging red→white→teal, rows ordered by complete-linkage clustering over \|SHAP\| (scipy — a core dep), the thin mean-\|SHAP\| bar on top. |
| `shap.plots.force` (the matplotlib static twin) | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_shap_force(summary, sample)`: one stacked bar of the row's signed contributions, base and prediction annotated — the honest static translation for a desktop app (the JS layout is N-A). |
| `shap.plots.group_difference` (cohort decomposition + bootstrap CI) | **ADDED-THIS-ROUND** | `ml/explain.py::group_difference(summary, mask)` (pure numpy: per-feature cohort mean difference + 200-resample 5–95% bootstrap bounds, the upstream's own recipe) + `viz/ml_plots.py::plot_shap_group_difference`. The GUI splits the run's held-out rows at their own target median — the "why does the high-potency cohort score differently" figure. |
| `TreeExplainer.shap_interaction_values` (the exact pairwise cube) | **ADDED-THIS-ROUND** | `ml/explain.py::shap_interactions(model, X)` → the typed `ShapInteractions` record (diagonal = main effects, symmetric off-diagonal = pairwise; `as_frame`/`ranking`/`__str__`). Same optional gate, same refusals as `shap_summary`. |
| `PermutationExplainer` / `permutation_feature_importance` | COVERED-ALREADY(≈) | The shipped model-agnostic sibling (typed, seeded) answers the non-tree question — the standing honest answer since 100x-154. |
| `_violin` / `_embedding` / `_style` | COVERED-ALREADY(≈) | The beeswarm carries the violin's information without KDE-over-2-points lies; `ml/manifold.py` + colour-by covers embeddings; the house PALETTE is the style system. |
| `KernelExplainer` / `ExactExplainer` / `_linear` / `_additive` / `_sampling` | N-A-HONEST | The slow approximation is deliberately refused with the typed error naming the sibling; exact enumeration is exponential; linear/GAM attribution is answered exactly by the coefficient surfaces. |
| `_deep/` / `_gradient` / `image` / `text` maskers & plots | N-A-HONEST | Tabular QSAR toolkit — no NLP/vision/deep-explainer surface. |
| `model_output=` transforms / `links.py` (logit) / `cohorts(n)` auto-tree / `_decision` / `_monitoring` / `_partial_dependence` / `_partition`+`_coalition` / `ActionOptimizer` / benchmark framework / GPUTreeExplainer / datasets | STRETCH | Each row's reason in `scratch/shap_optuna_review.md` §1 — the four highest-value stretches (logit links, decision plots, PD+ICE, the counterfactual ActionOptimizer) are the recorded candidates for a later round. |

**SHAP tally:** 12 ADDED (5 figures + 2 records + 2 computations +
interactions + the auto-interaction heuristic + hclust ordering) · 6
COVERED(≈) · 13 N-A-HONEST · 9 STRETCH.

### Optuna (optuna/optuna) — the disposition

| Optuna capability (source) | Disposition | Where it landed |
| --- | --- | --- |
| `TPESampler` — the flagship sampler | **ADDED-THIS-ROUND** | `ml/tuner.py::TPESearch`: same `search(X, y)` contract as RandomSearch; per param, Parzen (KDE) mixtures over the below (best `γ(n)=min(⌈0.1n⌉, 25)` trials) and above sets — neighbour-gap bandwidths, magic clip, one prior kernel at the domain centre — `n_ei_candidates=24` candidates drawn from the below mixture, argmax of `log pdf_below − log pdf_above`; `n_startup_trials=10` uniform bootstrap; deterministic per seed. Honest deviations (independent per-param, pruned excluded from mixtures, clipped-normal truncation) documented in the class docstring. |
| `MedianPruner` / `PercentilePruner` (report + should_prune) | **ADDED-THIS-ROUND** | `ml/tuner.py::PercentilePruner` + the `MedianPruner` shortcut, wired into the tuners' fold loop: after each fold the running mean is REPORTED, and a trial below the percentile of the completed trials' same-fold values is cut short — `state="pruned"`, partial `fold_scores` kept. |
| `trial._state` (COMPLETE/PRUNED/FAIL) + `trials_dataframe` | **ADDED-THIS-ROUND** | `TrialResult.state` / `resolved_state` / `pruned` (the legacy `failed` semantics preserved by derivation) + `fold_scores` + `duration`; `TunerResults.trials_frame()` carries state/duration/n_folds; `summary()` counts the pruned line. |
| `suggest_float(log=True)` | **ADDED-THIS-ROUND** | `HyperParameters.Float(..., log=True)` — log-domain sampling (Optuna's own recommendation for learning rates), the step+log and non-positive-bound refusals. |
| `get_param_importances` (MDI evaluator) | **ADDED-THIS-ROUND** | `ml/tuner.py::param_importances(results)` — Optuna's own one-paragraph recipe: RandomForestRegressor over the one-hot-encoded params → objective, `feature_importances_` summed back per param (`np.add.at`), normalized; the typed `HyperparameterImportance` record. |
| `plot_optimization_history` | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_optimization_history`: completed dots + the best-so-far step line + pruned trials as open amber markers at their last intermediate (the honest addition over the upstream). |
| `plot_slice` / `plot_parallel_coordinate` | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_slice` (one panel per param, categorical tick labels) / `plot_parallel_coordinate` (per-trial polylines, score-coloured on the house ramp, the param ranges under the axes). |
| `plot_param_importances` | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_param_importances` — the MDI bars, record or DataFrame form. |
| `plot_edf` / `plot_timeline` / `plot_intermediate_values` | **ADDED-THIS-ROUND** | `viz/ml_plots.py::plot_edf` (ECDF + Q1/median/Q3) / `plot_timeline` (the honest stacked-duration Gantt for a sequential search, coloured by state) / `plot_intermediate_values` (the pruner's running fold-mean curves, pruned dashed). |
| `RandomSampler` / Hyperband / grid search | COVERED-ALREADY | `RandomSearch` (100x-90), `Hyperband` (100x-90), the registries' `param_grids`. |
| `plot_contour` / `plot_rank` / PedAnova / fANOVA / `plot_pareto_front`+hypervolume (multi-objective) / ask/tell / `enqueue_trial` / terminator / storages (RDB/journal/gRPC) / integrations (mlflow, wandb, LightGBM callbacks…) / NSGA-II/III / CMA-ES / GP / QMC samplers | N-A-HONEST or STRETCH | Single-objective, single-process, sklearn-native toolkit — each row's reason in `scratch/shap_optuna_review.md` §2 (contour/rank = STRETCH: sparse trial clouds; the multi-objective family = N-A without `directions`). |

**Optuna tally:** 11 ADDED · 3 COVERED · 5 STRETCH · 10 N-A-HONEST.

### Round-43 verification rows

| What | Where it landed |
| --- | --- |
| SHAP computations | `ml/explain.py`: `ShapInteractions` + `shap_interactions` (the optional-gated cube) + `GroupDifference` + `group_difference` (pure numpy, seeded bootstrap); `ml/__init__.py` exports. |
| SHAP figures | `viz/ml_plots.py`: `plot_shap_waterfall` / `plot_shap_dependence` (with `_auto_interaction`) / `plot_shap_heatmap` / `plot_shap_force` / `plot_shap_group_difference`; `viz/__init__.py` exports. |
| Optuna tuner | `ml/tuner.py`: `TPESearch` + the `_parzen_*`/`_tpe_*` machinery, `PercentilePruner`/`MedianPruner` + `_evaluate_trial_folds` (the report/should-prune fold loop), `TrialResult` states/intermediates/duration, `HyperparameterImportance` + `param_importances`, `Float(log=True)`; `ml/__init__.py` + CLI `--searcher tpe`. |
| Optuna figures | `viz/ml_plots.py`: the seven study figures over `_unwrap_trials` (TunerResults or bare list, duck-typed); `viz/__init__.py` exports. |
| The GUI | playground.py: the five SHAP buttons (two rows — waterfall/force/dependence/heatmap + group difference, with the `_ask_shap_row`/`_ask_shap_feature` dialogs), the Tune card's `tpe` searcher + pruner combo + the two study-figure button rows (history/slice/parallel/importances + EDF/timeline/intermediates), the pruned-trial leaderboard rows (partial score marked `·`), the `on_trial` pruned note. |
| **The crash the round found on the way** | `viz/_compat.py::_ipython_shell_running` — the old `"IPython" not in sys.modules` notebook guard was fooled by `import shap` (which pulls IPython in), so a shap-using process silently skipped the headless Agg pin; pyplot resolved TkAgg under `$DISPLAY` and every WORKER-THREAD figure became an off-main-thread Tk window creation — the intermittent `Illegal instruction` the new surfaces exposed (the old beeswarm tests only survived full-suite runs because an earlier test had forced Agg directly). The fix: only a RUNNING shell (`IPython.get_ipython()` non-None) keeps the inline backend. Pinned by `test_viz.py::TestBackendGuard`. |
| Tests | tests/test_explain.py (+~370: `TestShapInteractions`, `TestGroupDifference`, `TestShapPlotsRoundTwo`) · tests/test_tuner.py (+~600: `TestLogFloat`, `TestTPESearch`, `TestPruners`, `TestPrunedSearch`, `TestTrialStates`, `TestParamImportances`, `TestOptunaStudyPlots`) · tests/test_playground.py::`TestShapOptunaAbsorptionGui` (12: the token/button census, the five figures end-to-end with the ask-dialog fixture, the guards, the seven study figures, a real TPE+median search) · tests/test_viz.py::`TestBackendGuard` (3). |

## Round 185 (re-ship, v1.2.105) — the XAI absorption's verification rows

*The four-repo absorption round (captum · tf-keras-vis · PAIR-code/saliency · bertviz → `ml/xai.py` + the Attribution & attention card), re-implemented after the third rollback and re-shipped lean after the first zip walked the research clones into a 1.15 GB archive.*

| What | Where it landed |
| --- | --- |
| The attribution registry | `ml/xai.py::ATTRIBUTION_METHODS` + `input_x_gradient` / `integrated_gradients` (zero baseline, midpoint steps, `convergence_delta`) / `smoothgrad_attributions` (NoiseTunnel averaging) / `occlusion_attributions` (predict-only, every engine); `numpy_layer_stack` snapshots the numpy engine's `linears` (the `_act_np` probe recovers the activation token empirically — the bare-`_MLPNetwork` path silently defaulted to relu before). |
| The sign split + palettes | `positive_negative_split`, `diverging_hex` (blue-negative/red-positive), `grayscale_hex` (percentile-clipped monochrome) — PAIR-code/saliency's two visualizers as Tkinter hex. |
| Layer traces + attention | `layer_activation_trace` / `layer_activation_summary` (units, min/mean/max, dead-unit share) over the numpy stack; the bertviz-style layer × head grid on the toolkit's own `ml.attention` + `token_pairs`. |
| The GUI | playground.py's **Attribution & attention** card: method combobox over the four-method registry, row selector + IG steps, the ranked per-feature table (push sign + share), the colorized strip, the layer-activation table + grayscale unit map, the attention grid canvas + pair digest; the DL session round-trip pin moved 18 → 24 inputs. |
| Tests | `TestXaiHelpers` (11 pure pins incl. the finite-difference gradient check and the playground↔registry key sync) + `TestXaiAbsorptionGui` (7 headless card drives). |

## Round 186 (v1.2.106) — tensorboard · umap · fiftyone · plotly.py, absorbed offline

*The instruction's fourth arrival: four more repositories checked file by file (treeless shallow clones: tensorboard 1,934 files · umap 349 · fiftyone 7,168 · plotly.py 1,992) and absorbed into the Deep Learning tab's rows 6-9.*

| What | Where it landed |
| --- | --- |
| The event model (`summary_v2`) | `ml/runboard.py::RunTrace` — `add_scalar` / `add_histogram` (stats + 30 fixed-width buckets) / `add_text` / `add_embedding` (checkpoint semantics) / `add_hparams`; `tags`/`scalars`/`histograms`/... accumulator queries; deterministic `to_json`/`from_json`. |
| The reservoir (`reservoir.py`) | `ReservoirBucket` — Vitter's `randint(0, num_seen)` draw with TensorBoard's `always_keep_last` replacement, per-tag determinism; `reservoir_sample` one-shot; every RunTrace event list bounded by it. |
| The dashboards' math | `smooth_series` (the scalars slider's bias-corrected EMA), `histogram_stats_frame` (the overlay view's per-step stats), `RunBoard.compare_table` (the compare-runs strip: last/best/mean per run, loss-named tags take min), `session_groups` (the hparams plugin's grouping). |
| The projector's click | `nearest_neighbours` — cosine similarity over a labelled matrix, query row first, k neighbours after. |
| The UMAP algorithm | `ml/umap.py` — `smooth_knn_dist` (64-iteration bisection on `log2(k)·bandwidth`, rho from `local_connectivity`), `compute_membership_strengths`, `fuzzy_simplicial_set` (the `A + A^T − A·A^T` union over both directed strengths), `find_ab_params` (the same `curve_fit`; a=1.577/b=0.895 pinned), `_optimize_layout_euclidean` (attraction `−2ab d^{2b−1}/(a d^{2b}+1)`, repulsion `2γb/((0.001+d²)(a d^{2b}+1))`, ±4 clip, `epochs_per_sample = w_max/w`, linear alpha decay, negative_sample_rate), `_spectral_init` (sklearn over the fuzzy graph, honest random fallback), `UmapRecord.transform` (weighted-average init, fitted side frozen). |
| Trustworthiness | `trustworthiness` — sklearn's exact inverted-index algorithm, pinned bit-for-bit at k ∈ {1, 3, 5, 10}. |
| The view algebra (`core/stages.py`) | `ml/curation.py::SampleSet`/`Sample`/`DataView` — `filter` (the FilterField op family), `match`, `sort_by`, `limit`, `skip`, `shuffle`, `take`, `select`/`exclude`, `exists` (None/NaN/NaT), `has_tag`, `select_fields`, `concat`; every stage a NEW view, the chain as provenance; `tag`/`untag` the mutating side. |
| The brain methods | `find_duplicates` (identical-field groups, keep=first), `find_outliers` (zscore + IQR fences), `uniqueness` (1 − mean k-NN cosine), `sort_by_similarity`, `evaluate_classification` (confusion, per-class P/R/F1, mistakes). |
| The express layer | `viz/interactive.py` — `hovertext` per trace (validated, JSON round-trip), `TEMPLATES` (plotly / plotly_white / ggplot2 / seaborn: colour cycle + paper + ink), `express_figure` + `express_scatter/line/bar/histogram` (colour splits, hover columns, labels; FD bins numpy-side), `apply_template`; the SVG renderer paints `<title>` tooltips and honours the template palette. |
| The GUI | playground.py's four cards: **Run board** (loss curves + smoothing spin, weight histogram + per-layer stats strip, hparams table), **UMAP map** (knobs, target-coloured map, trustworthiness title, nearest rows, CSV export), **Data curation** (filter chains + the four audits), **Interactive figure** (kind/x/y/colour/template, hover columns, ASCII preview, hover-HTML export). |
| Tests | `tests/test_runboard.py` (36) · `tests/test_umap.py` (27, incl. the sklearn parity pins) · `tests/test_curation.py` (26) · `tests/test_interactive.py` (20) · `TestRound186Gui` (13 headless drives) — 122 new, suite 4,032 collected across 147 files. |
| Fixed on the way | The SELFIES samplers' PYTHONHASHSEED determinism bug (the default alphabet iterated as a raw SET in three call sites — same seed, different molecules per process; the flaky doctest gate's root cause), the EXE002 exec-bit sweep (385 files), the round-185 lint debt (S110/RUF046/RUF059/UP032/I001), and the zip-walk lesson (research clones deleted before shipping — RESUME.md rule 7). |

## Round 222 (v1.2.135) — the Part-3 §17 image-export engine

*The Part-3 directive's first section made TIFF and JPEG first-class export formats throughout the application — the engine below carries the section's scientific-integrity rules in code, and every GUI export path routes through it.*

| What | Where it landed |
| --- | --- |
| The engine | `viz/export.py` — `export_image` (array / PIL image / existing file → PNG/TIFF/JPEG by extension), `export_matrix` (+ multi-page TIFF stacks, per-page bit-depth checks), `export_figure` (matplotlib → raster through Agg, or native vectorial SVG), `export_widget_image` (Qt-rendered bytes). |
| TIFF, the scientific path (§17.2) | `uint16` → genuine `I;16` (bit-identical reopen, pinned by test); lossless `tiff_lzw` / `tiff_deflate` / `none`; `save_all`+`append_images` stacks with `n_frames` validation; dpi resolution tags. |
| JPEG, the presentation path (§17.3) | quality 1-95 (validated); 16-bit sources REFUSE without explicit `downcast_16bit=True` (warning recorded); RGBA/LA/PA composited onto the chosen background (never a silent alpha drop); float matrices refuse without `normalize_float=True`. |
| Validation as part of the export (§17.4) | every write reopened + `sniff_format` magic-byte check (PNG / JPEG / TIFF II+MM / SVG) + dimension check before success; `validate_export` (never raises — reports `ok`/`reason`) catches extension/content lies and wrong sizes; extension must MATCH format (§17.4's no-silent-extension rule). |
| The reusable manager (§17.4) | `ExportManager` — batch to a directory with a `{name}` pattern, per-write validation, JSON `manifest.json` (name/format/size/mode/compression/warnings); `EXPORT_FILTERS` the §17.1 dialog strings (`.tif .tiff`, `.jpg .jpeg`). |
| The GUI bridge | `ui/widgets/image_export.py` (`grab_png_bytes` offscreen-safe + `export_widget_grab`); `ImagePanel.export_image` (with the `export_png` boolean shim) and the new `PlotPanel.export_image` — the GUI renders, the engine decides format policy. |
| The Image Lab row | `ui/pages/image_lab.py` — format combo (TIFF default), JPEG quality spin 1-95, TIFF compression combo (LZW/DEFLATE/None), the three-filter dialog, the extension-stamping rule, and the honest status line (format/dimensions/bytes + every integrity warning; failures surface the engine's error). |
| Dependency honesty | Pillow optional exactly like matplotlib (`reports` extra; verified stack pins 11.3.0) — importing is free, exporting without it raises the actionable `DependencyError`. |
| Tests | `tests/test_export_images.py` (29 collected items: the 16-bit roundtrip, the three compressions parametrised, quality monotonicity, every refusal, the stack, the liar/dimension/missing/corrupt validation cases, the manager manifest + override, the Pillow-free error shape) · `tests/ui/test_export_gui.py` (7 offscreen: both panels' three formats at the exact rendered size, the compat shim, the control-row contract, the result-card status, the extension stamping, the honest FAILURE). 36 items + 7 doctests; honest totals 5,575 = 4,790 collect + 785 doctest. |

## Round 229 (v1.2.142) — the Part-3 §20 checkpoint engine

*The directive's pause/resume/stop section became durable state: long jobs write self-verifying checkpoints after every completed unit, Stop keeps the finished work on disk, and the next Start resumes instead of recomputing.*

| What | Where it landed |
| --- | --- |
| The engine | `utils/checkpoints.py` — `CheckpointManager.save` (atomic: temp + `fsync` + `os.replace` + directory `fsync`; SHA-256 over the canonical record), `latest`/`list` (newest first, corrupt files QUARANTINED to `*.json.corrupt` with the resume falling back to the previous valid checkpoint), `clear`, rotation (`keep` newest), `default_directory` (`~/.pharmaml/checkpoints/<name>/`), JSON states (no pickle — a malicious checkpoint cannot execute on load). |
| The benchmark wiring (§20) | `ml/training.py::benchmark_feature_families` — `checkpoint_dir`/`checkpoint_name`/`resume` params; after EVERY completed row (success or NaN-swallowed failure, BEFORE the stop check) the board-so-far is saved; a matching signature (task + cv + metric + resolved model + family list + SMILES digest) resumes at the first unscored family — consecutive-from-the-front, any gap stops the reuse; write failures degrade to warnings, never kill the sweep. The enumerate-`start=`-offsets-only-not-skips bug the smoke caught: the resume must SLICE `family_list[start_position-1:]`, or reused rows recompute and re-save at shifted steps. |
| The Studio wiring | `ui/pages/ml_workbench.py` — the Leaderboards benchmark checkpoints into `~/.pharmaml/checkpoints/family-benchmark` (an unwritable home degrades to fresh runs, never fails the page); the job surfaces the resume story before the run; the status line leads with "checkpoint found: k/n families already scored — resuming there" when a checkpoint matched; the Start button's tooltip carries the Stop→resume contract. |
| Tests | `tests/test_checkpoints.py` (36: the round-trip, unicode, JSON-document shape, step/state/name/keep validation, directory-on-demand, rotation keep=1/2, newest-first listing, truncated/tampered/foreign quarantine + fallback, re-quarantine naming, all-corrupt → None, foreign-name invisibility, clear) · `tests/test_ml_pipeline.py` +6 (cancel keeps finished families, resume computes only the unscored families — featuriser-counting spy, resumed board == uninterrupted board with the ORIGINAL wall-clock fit times, signature mismatch starts fresh, `resume=False` recomputes, an unwritable checkpoint path never kills the board) · `tests/ui/test_ml_workbench_page.py` +1 (the two-Start GUI loop: run 1 fills the checkpoint, run 2 reuses every family and the status leads with the resume note). |

## Round 230 (v1.2.143) — the Part-3 §24 SA engine (completing the interrupted round's ship)

*The directive's synthetic-accessibility section landed end-to-end — and the round also finished the work an interrupted session had committed half-done: the syntax fix, the true-value pins, the docs, and the release ritual.*

| What | Where it landed |
| --- | --- |
| The engine (§24.2) | `chem/sa_scoring.py` — the Ertl–Schuffenhauer score on the REAL fragment model (the RDKit wheel's own `Contrib/SA_Score/fpscores.pkl.gz`, loaded lazily; aspirin 1.58, caffeine 2.30 — the reference Contrib values): `sa_scores` (batch rows keep original + canonical SMILES, score, easy/moderate/hard band, per-row parse errors — invalid rows never crash), `filter_by_sa` (the max-SA bound flags rejects in place — the §24.4 never-discard contract), `sa_score_summary` (mean ± std, range, the honest caveat). |
| The oracle upgrade | `chem/oracles.py::sa_score` delegates to the same lazy loader (Contrib-less builds degrade honestly to complexity-only ranking) and uses the non-deprecated `rdFingerprintGenerator` API; `penalized_logp` moved onto the real SA scale (aspirin 1.137, ethanol -0.258 — the old -7.3-era pins were the degraded scorer's artefacts; the doctest and GUI pins trued). |
| The generative wiring (§24.5) | `generative/scoring.py::SAScoreObjective` — the window bounds become the easy/hard reward anchors; "SA score (ease of synthesis)" selectable in the Studio's Evolve objective combo; SA as ONE signal, never the sole target. |
| The Studio tab (§24.3) | `ui/pages/generative.py` — the "SA score" engine tab: paste molecules (four seeded defaults incl. a taxane-style complex), the optional max-SA bound ([1, 10]), the ranked table (rank / SMILES / canonical / score / band / status) with rejects kept in place and the summary line. |
| The web bridge | `/api/chem/sa-score` (zod: ≤ 50 SMILES, min/max bounds) over `scripts/sa_score_bridge.py` (stdin→stdout JSON in the venv) feeding the Explorer Playground `sa-score-card`; the card-directory + probe pins updated. |
| Infrastructure | ROOT `conftest.py` preloads the round-206 EGL stub for EVERY pytest invocation — pytest-qt imports `PySide6.QtGui` at `pytest_configure` (before any `tests/` conftest), so EGL-less containers INTERNALERRORed every non-`tests/ui` run (the library chunks AND the doctest chunk). |
| Tests | `tests/test_sa_score.py` (37: the reference values, bands, filtering, summary, invalid rows, the Contrib-less degradation) · `tests/ui/test_generative_page.py` +6 (the SA tab defaults with the true easiest-first pins — the ibuprofen core 1.520 leads, not aspirin 1.580 — the 2-kept/2-rejected filter split, invalid rows, empty input, the bad bound, the objective selection with `_obj_lo`/`_obj_hi`) · `tests/test_web_parity.py` (the bridge + card pins). |

## Round 231 (v1.2.144) — the Part-3 §22 SMILES previews + the Qt console-noise filter *(COVERAGE row reconstructed in round 232 — the interrupted round shipped the code but not its docs rows)*

| What | Where it landed |
| --- | --- |
| SMILES previews (§22) | `ui/widgets/smiles_preview.py` — `MoleculeRenderer` (MolDraw2DCairo at 2× device pixels, 128-entry FIFO cache, honest invalid-SMILES refusal — §22.4 never a made-up molecule), `MoleculePopup` (350 ms dwell, §22.2), `SmilesDetailDialog` (§22.5: formula / weight / the Ertl SA score with method attribution / canonical-SMILES copy); wired onto the Generative tab's shared table by COLUMN NAME so every bound frame gains it. |
| Qt console-noise filter | `ui/qt_noise_filter.py` — the user-reported v1.2.142 Windows noise (QFontDatabase fonts-dir advice + the Qt 6.8 QStyleHints unique-connection chatter ×4) diagnosed benign; the message handler installs BEFORE the QApplication in `ui.run()`; `PHARMAML_STUDIO_QT_VERBOSE=1` escape hatch. |
| Tests | +17: `tests/ui/test_smiles_preview.py` (12, on real renders) · `tests/ui/test_qt_noise_filter.py` (5, incl. a live `qInstallMessageHandler` round-trip); the chunk runner updated. |

## Round 232 (v1.2.145) — the Windows black-window fix + the Part-3 §21 slice 1 (hover inspection)

*The user reported `studio.py` showing console noise with NO window on Windows — root-caused to the headless bootstrap forcing the offscreen platform there; and every chart gained hover data inspection.*

| What | Where it landed |
| --- | --- |
| The Windows fix (user report) | `ui/headless.py` — the displayless probe tested only X11's `DISPLAY`, which NEVER exists on Windows (nor covers Wayland): every Windows launch silently forced `QT_QPA_PLATFORM=offscreen` — the app ran with an invisible window, `app.exec()` blocked, no GUI appeared. Now platform-aware: offscreen ONLY on Linux-class platforms with neither `DISPLAY` nor `WAYLAND_DISPLAY`; Windows/macOS keep native; explicit `QT_QPA_PLATFORM` still wins; the applied-dict records `"native"` for observability. |
| Hover inspection (§21 slice 1) | `ui/widgets/chart_inspector.py::ChartInspector` on every `PlotPanel` — crosshair, nearest-point snapping (bar bodies are whole hover targets on bars/histograms; 8 %-of-range resolution-independent tolerance), the readout line of REAL values with page-attached per-point metadata (`meta=` on line/scatter/bars, `attach_metadata()`; histograms read exact bin boundaries + count, labelled bars their category), per-point metadata caching + debounced text rebuild (§21.6), failing providers degrade to plain values, the context menu (per-series visibility toggles / inspection switch / reset-view), click-to-select with the `pointSelected(label, index)` drill-down signal (§21.5). |
| PlotPanel API | `set_inspection_enabled()` / `inspection_enabled()` / `attach_metadata()`; the readout QLabel has a fixed minimum height (no layout jump — the §23 seed). |
| Tests | `tests/ui/test_headless_platform.py` (7: the Windows/macOS/Wayland/X11/displayless/explicit/idempotence decision matrix) · `tests/ui/test_chart_inspector.py` (14: readout real values, nearest-of-two, far-hover, cached-once metadata, failing metadata, attach-after-plot, histogram bins, bar categories, crosshair wiring, click-select emit, disabled no-op, hidden-series skip, clear reset, placeholder naming); the chunk runner grows chunk 4c. |
| Docs repaired | The interrupted round 231's missing rows reconstructed: CHANGELOG 1.2.144 entry, this COVERAGE row, GUI_AUDIT's §22 progress — plus README's Windows-native note, the §21 section, GUI_AUDIT's round-232 progress and CHANGELOG 1.2.145. |

## Round 233 (v1.2.146) — the Part-3 §23 slice 1: the responsive pass

*The splitter audit lands as wired policy, exports gain a legibility floor, and matrix images lock aspect — display geometry only, the data untouched (§23.5 pinned by test).*

| What | Where it landed |
| --- | --- |
| The §23.2 splitter policy | `ui/widgets/responsive.py::configure_splitter` — no collapsible children, hard per-child minimum extents (`MIN_CHILD_PX` 320; the main window's nav 200), stretch factors routing window growth to the chart side. Wired onto ALL 20 splitters: the 15 page splitters, the Image Lab's four tab splitters, the ML Workbench's two, and the main window's nav/content splitter (its nav side was previously collapsible — the whole navigation could be dragged away). |
| Export legibility floor (§23.3) | `ui/widgets/image_export.py` — `grab_png_bytes(min_size=…)` / `export_widget_grab(min_size=…)` re-render a squeezed `QGraphicsView` from its **vector scene** at floor scale: `LEGIBILITY_FLOOR` 640×480 (`ui/widgets/responsive.py`), text/pens/curves re-drawn crisp (never bitmap-stretched), aspect preserved (§23.4), the pyqtgraph chrome re-filled via `backgroundBrush()` (scene.render skips the view-layer painting). `PlotPanel.export_image` / `ImagePanel.export_image` gained the `min_size` passthrough (default `None` = the §17-tested widget-owns-its-size contract); the Image Lab's user export button passes the floor. |
| Aspect lock (§23.4) | `ui/widgets/image_panel.py` — `setAspectLocked(True)`: matrix pixels stay square under every resize (a 100×50 correlation renders 2:1 as shaped, fit-to-view, never stretched); the §22 renderer's canvas-aspect contract is pinned alongside (RDKit letterboxes inside the exact requested canvas). |
| Tests | `tests/ui/test_responsive_policy.py` (+13): the policy contract (horizontal/vertical/empty-noop), the structural audit over every live splitter in a constructed main window (≥19 wired, all non-collapsible, all minimum extents), the floor (upscale-meets-floor + aspect-preserved, already-legible noop, no-floor contract), §23.5 data-integrity (arrays + view ranges byte-identical across an export), the ImagePanel aspect lock, the §22 canvas-aspect contract. Chunk runner `scratch/run_suite_chunks_233.py` (4d gains the new suite). |

## Round 235 (v1.2.147) — the Part-3 §23 slice 2: the scroll policy

*The tallest forms ride a scroll region now — sideways overflow made impossible, a fitting page visually unchanged, display geometry only (§23.5).*

| What | Where it landed |
| --- | --- |
| The scroll policy | `ui/widgets/responsive.py::wrap_in_scroll` + `SCROLL_MIN_PX` (240) — widget-resizable, horizontal scrollbars OFF, vertical as needed, frameless + palette-blended, hard region minimum. Wired onto ALL five ML Workbench tab bodies at the addTab sites (probe evidence: tuner 479 px, Leaderboards 775 px vs ~450 px budget at the 860×560 minimum window). |
| Tests | `tests/ui/test_responsive_policy.py` (+3, 13→16): the wrapper contract, the §23.5 remount invariants (same rows, never more space required), and the structural audit that fails if a future tab skips the wrapper. Chunk runner `scratch/run_suite_chunks_235.py`. |

## Round 236 (v1.2.148) — the Part-3 §21.5 molecule readouts

*Chart points that are molecules read out as molecules — hover carries SMILES/formula/weight, click opens the detail view, nothing fabricated.*

| What | Where it landed |
| --- | --- |
| The forwarding signal | `ui/widgets/plot_widget.py::PlotPanel.point_selected` — re-emits the inspector's click-to-select `(label, index)` so pages never reach into the inspector for drill-downs. |
| The SA chart | `ui/pages/generative.py::_render_sa_score` — rank vs score, one point per scored molecule; hover metadata via the lazy provider (`SMILES`, RDKit `formula`, `MW`); click → the §22.5 `SmilesDetailDialog` with a foreign-label guard; invalid rows stay table-only. |
| The Pareto front | `_render_front` — the non-dominated scatter carries the same hover metadata per front member. |
| Tests | `tests/ui/test_generative_page.py` (+4): the forwarding pin, the hover readout with real numbers (C10H14O / 150.22 g/mol for the ibuprofen core), the click-to-detail + guard, the front hover. |

## Round 237 (v1.2.149) — the Qt import guard + `studio.py --doctor` + the ui/ lint cleanup

*A broken PySide6 install answers with a diagnosis and a fix recipe, not a traceback wall; one command audits the whole launch environment.*

| What | Where it landed |
| --- | --- |
| The import guard | `ui/qt_import_guard.py` + the guarded `QApplication` import in `ui/__init__.py::run` — the caught `ImportError` prints `qt_import_guidance()` (diagnosis, the two `sys.executable`-qualified fix commands for the whole PySide6/shiboken6 wheel family, VC++-redist + conda notes) and exits 2; the missing-module family takes the plain-install branch. |
| The doctor | `ui/doctor.py` + `--doctor` dispatch in `ui.run` — 16 checks (python floor, PySide6/shiboken6 with wheel-metadata versions matched before the import probe, the exact `QApplication` import, GUI providers, science stack), a fix under every failure, exit 0/1, read-only. |
| Lint cleanup | `ui/` + `studio.py` now ruff- and mypy-clean: 19 ruff findings + 7 mypy errors from rounds 233–236 cleared (dead imports/vars, unsorted blocks, `int(ceil())`, the `_SeriesRecord` shadow, the rdkit `Descriptors.MolWt` ignore per repo convention, the `ndarray \| None` metric guard in model_analysis); `studio.py` carries its executable bit. src/ (90) + tests/ (57) recorded as backlog. |
| Tests | `tests/ui/test_qt_doctor.py` (+7): both guidance branches, report format + exit codes, the live 16-check collection, the `--doctor` CLI subprocess, and two PYTHONPATH-shadowed broken-PySide6 end-to-end legs (exit 2 + recipe; doctor diagnoses without a traceback). |
