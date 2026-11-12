# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.181.zip`
Size: 9224767 bytes; SHA-256: `9b8fbde309ee17e42a228b44505ee5a9dcd8f7711f1a560321a0b82b6b495556`
Entries: 621; files: 567

## Checks

- path_and_bounds: **passed**
- crc: **passed**
- version_carriers: **passed**
- manifest: **passed**
- excluded_paths: **passed**
- engine_checksums: **passed**
- isolated_restore: **passed**
- archive_stability: **passed**

Example-data files: 45 recursive, 16 top-level. Documentation consistency not fully checked.

## Bundled engine checksum references

- `engines/autodock4.exe`: MD5 `052576a87455c30df8ef8bdd951d65ab`, reference match: True; not executed
- `engines/autogrid4.exe`: MD5 `9ef694a508215d5c46ab072fba59e035`, reference match: True; not executed
- `engines/vina.exe`: MD5 `51fd15a78c71da12958d55609c3720c7`, reference match: True; not executed

## Test profiles (recorded evidence; not executed by this audit)

- **R11 audit battery (hpo ×4, generative-search proofs + regression, similarity ×2, descriptor-batch, docking-batch proofs + docking + campaign + engine-discovery + docking-prep regression, preprocessing proofs, schema, tooling, numpy-neural, pipeline, notebook-variants, carriers + doctests)** — `python -m pytest tests/test_hpo.py tests/test_hpo_presets.py tests/test_multiobjective_hpo.py tests/test_hpo_resumability.py tests/test_generative_search_resumability.py tests/test_generative.py tests/test_generative_result_sa.py tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_docking_batch_resumability.py tests/test_docking.py tests/test_campaign.py tests/test_engine_discovery_part5.py tests/test_docking_prep.py tests/test_preprocessing_resumability.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/ml/hpo.py src/pharmaml/generative/genetic.py src/pharmaml/chem/similarity.py src/pharmaml/docking/smiles_docking.py src/pharmaml/ml/preprocessing.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py --doctest-modules`
  1273 passed, 0 failed, 2 skipped, 0 blocked, 0 not-executed (collected: 1275)
  Notes: run on main at the 1.2.181 bump, AFTER the notebook ritual completed (the round-8 scheduling rule); ALL SIX §13 proof suites ride this battery (94 tests across descriptor-batch, similarity-campaign, hpo, generative-search, docking-batch, preprocessing), plus the doctests of every touched module (including the new apply_preprocessor_chunked docstring example)
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.181 in 168s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness + variants guards green in the audit battery
- **Offscreen GUI battery (round-5 recorded evidence)** — `python -m pytest tests/ -k gui --no-header -q`
  0 passed, 0 failed, 0 skipped, 0 blocked, 532 not-executed (collected: 532)
  Notes: not executed this round; the round-5 recorded evidence (532/532 green at 1.2.174) carries the GUI leg; R11 touches no GUI code (the preprocessing chunk loop is headless)

### Gates

- ruff (full src + all seven touched suites): **passed** — preprocessing.py, checkpoint_schema.py and all touched test suites clean
- mypy src/pharmaml: **passed** — 217 files, zero errors
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.181 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.181, twins stripped, guard green
- §13 registry completeness (this round's milestone): **passed** — all six candidate classes registered with pinned proof nodes; PENDING_OPERATION_TYPES empty; the unknown-type gate armed (time_travel refused in the proof suite)

### Backup and recovery

- status: verified this round (push below re-verifies the remote)
- evidence: main carries R10 (4123cad) verified on the remote at round start; the R11 commit pushed to origin/main; the v1.2.181 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- The §13 candidate sequence is COMPLETE (all six classes registered), but the registered resumability is library-side only: the GUI does not yet surface JobCancelled.partial_results or offer Pause/Resume buttons for any registered class
- The preprocessing exactness contract assumes the sklearn pipeline's fit is deterministic given the table (verified for SimpleImputer/StandardScaler/OneHotEncoder in this round's recon; a future transformer with non-deterministic fit would break the refit-on-resume contract and must not be added to build_preprocessor without a new proof
- The checkpoint payload is the transformed rows as JSON floats — bit-exact in CPython (repr round-trip), but the payload size grows linearly with the table (a 1M-row x 500-col table would checkpoint ~4 GB of JSON); the intended use is the large-but-not-huge QSAR table regime
- A resume refits the pipeline on the full table before skipping completed chunks: the fit cost is re-paid on every resume (seconds for QSAR-scale tables; the alternative — serializing fitted sklearn state — was rejected as version-fragile)
- chunk-aligned cancel: the partial ledger holds whole chunks; a stop mid-chunk (not possible via the progress contract, which fires on boundaries) would need row-level ledgering
- The offscreen GUI battery was not re-run for 181 (round-5 recorded evidence carries it; R11 touches no GUI code)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
