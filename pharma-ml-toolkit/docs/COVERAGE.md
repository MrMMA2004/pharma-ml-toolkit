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

## User uploads (100x-118) — the contract extends past the two archives

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
