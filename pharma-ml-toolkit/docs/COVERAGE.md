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
| `Convert IC50(nM) to pIC50.py` | `chem.parse_endpoint` (strips `<`/`>` qualifiers, never fabricates) + `chem.ic50_to_pic50` + `add_pic50_column`; `pharma.units` for the unit math; CLI `chem pic50` | ✅ |
| `Convert CSV to SDF.py` | `chem.table_to_sdf` — AddHs → ETKDG (fixed seed) → UFF, every column becomes an SD tag, invalid SMILES counted not lost | ✅ |
| `Convert SMI to CSV.py` | `chem.smi_to_csv` (streaming, BOM for Excel) — also `others/General/Convert SMI to CSV.py` | ✅ |
| `others/Ml/Sdf reading.ipynb` | `chem.sdf_to_table` (tags → columns; the notebook's PandasTools.LoadSDF path) | ✅ |
| `others/Ml/Descriptor calculator.py` / `Descriptoredited.py` | `chem.compute_descriptors(descriptor_set='rdkit_all')` | ✅ |
| `others/Ml/Mordred_calculation.py` | `chem.compute_descriptors(descriptor_set='mordred')` — Error/Missing → NaN with an aggregate warning (100x-52) | ✅ |
| `others/Ml/ECFP_calculation.py` | `chem.compute_fingerprints(kind='morgan')` (ECFP-style bits, radius & n_bits) | ✅ |
| `others/Ml/Morgan_calculator.py` | `chem.compute_fingerprints(kind='morgan', as_counts=True/False)` | ✅ |
| `Tanimoto calculator.py` / `others/Ml/Tanimoto calculator.py` | `chem.tanimoto_matrix` (ref × target) + `similarity_scan` (long format, threshold) + `top_matches` | ✅ |
| `Tanimoto-edited.py` / `others/Ml/Tanimotoedited.py` | the chunked/threshold variant → `similarity_scan(threshold=...)` | ✅ |
| `Non-numerical data remover.py` | `ml.clean_numeric_frame` (target/keep_cols survive, dropped names logged); cell-level coercion is the mordred NaN path + `biology.normalize_expression` | ✅ |
| `Split CSV file.py` / `others/General/Split CSV file.py` | `chem.split_table_into_chunks` (flat memory, explicit output dir) | ✅ |
| `others/General/CSV_merger.py` | `chem.merge_tables(first_columns=[...])` (the column-ordering contract) | ✅ |
| `others/General/filemerger.py` | `chem.merge_table_folder` (CSV/Excel folder → one file); CLI `chem merge` | ✅ |
| `others/General/common_between_two_excel.py` | `chem.compare_tables(...).common` / `.only_in_a` / `.only_in_b` (name-normalized set operations; TableComparison) | ✅ |
| `others/compare/notcommon.py` | `chem.compare_tables(...).only_in_a / .only_in_b` differences report; CLI `chem compare` | ✅ |
| `others/compare/smilesgetter.py` | `chem.annotate_with_smiles` (join SMILES onto a ligand-name column, extension/case normalization) | ✅ |
| `others/General/approved.py` | `pharma.filter_approved_drugs` (17 TSVs → approved names) + `viz.export_docx` (any table as a real Word document — the rar wrote a bare .docx name list) | ✅ |
| `others/General/Nameextracter.py` | `docking.list_file_names` (file inventory → table) | ✅ |
| `others/General/adder.py` | `docking.distribute_files` + `docking.distribute_receptor` (copy binaries/receptor into every campaign subfolder) | ✅ |
| `others/General/delete.py` | `docking.cleanup_tree` (keep-list + pattern, dry-run default) | ✅ |
| `others/General/deleter.py` | `docking.delete_named_files` | ✅ |
| `others/General/extaacter.py` | `docking.collect_files(pattern='r.dlg')` with unique-name resolution | ✅ |
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
| `others/dlg_extracter.py` | `collect_files(pattern='r.dlg')` | ✅ |
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
