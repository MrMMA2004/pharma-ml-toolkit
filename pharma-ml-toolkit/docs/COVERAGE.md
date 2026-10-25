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
