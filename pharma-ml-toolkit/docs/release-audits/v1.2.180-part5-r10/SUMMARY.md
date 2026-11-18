# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.180.zip`
Size: 9176666 bytes; SHA-256: `896f91f312ae5eb8c312215eb985ffd73e51a81bb55f7ec38af6d4f849b94ee6`
Entries: 616; files: 563

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

- **R10 audit battery (hpo ×4, generative-search proofs + regression, similarity ×2, descriptor-batch, docking-batch proofs + docking + campaign + engine-discovery + docking-prep regression, schema, tooling, numpy-neural, pipeline, notebook-variants, carriers + doctests)** — `python -m pytest tests/test_hpo.py tests/test_hpo_presets.py tests/test_multiobjective_hpo.py tests/test_hpo_resumability.py tests/test_generative_search_resumability.py tests/test_generative.py tests/test_generative_result_sa.py tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_docking_batch_resumability.py tests/test_docking.py tests/test_campaign.py tests/test_engine_discovery_part5.py tests/test_docking_prep.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/ml/hpo.py src/pharmaml/generative/genetic.py src/pharmaml/chem/similarity.py src/pharmaml/docking/smiles_docking.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py --doctest-modules`
  1260 passed, 0 failed, 2 skipped, 0 blocked, 0 not-executed (collected: 1262)
  Notes: run on main at the 1.2.180 bump, AFTER the notebook ritual completed (the round-8 scheduling rule); the 11-test docking-batch proof suite rides this battery, as do the docking regression suites (test_docking.py, test_campaign.py, test_engine_discovery_part5.py, test_docking_prep.py) guarding the historical behaviour of the touched module
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.180 in 171s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness + variants guards green in the audit battery
- **Offscreen GUI battery (round-5 recorded evidence)** — `python -m pytest tests/ -k gui --no-header -q`
  0 passed, 0 failed, 0 skipped, 0 blocked, 532 not-executed (collected: 532)
  Notes: not executed this round; the round-5 recorded evidence (532/532 green at 1.2.174) carries the GUI leg; R10 touches no GUI code (the docking batch loop is headless)

### Gates

- ruff (full src + all six resumability suites + schema suite): **passed** — smiles_docking.py, checkpoint_schema.py and all touched test suites clean
- mypy src/pharmaml: **passed** — 217 files, zero errors
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.180 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.180, twins stripped, guard green

### Backup and recovery

- status: verified this round (push below re-verifies the remote)
- evidence: main carries R9 (afb1553) verified on the remote at round start; the R10 commit pushed to origin/main; the v1.2.180 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- One §13 operation class remains pending (preprocessing) — it needs its own exact-restore proof round
- The docking-batch exactness contract is PRESERVE, not re-execute: finished ligands' rows are restored verbatim from the ledger, and the bit-identity proof runs against a deterministic fake engine (a real Vina/AutoDock4 run is stochastic by nature and cannot be bit-repeated; the §14 archive audit pins the real engine checksums separately)
- The preserved rows carry the original campaign-root paths (folder/output): a resume into a DIFFERENT out_dir keeps pointing at the original folders — the realistic resume is the same out_dir (the campaign folders are the durable artifacts), and the proof suite's fresh twins compare path-independent columns
- The ligand ledger is keyed by index+smiles and the fingerprint covers the full smiles list, but the on-disk campaign folders are NOT fingerprinted: a resume after the user manually deleted a finished ligand's folder preserves the row (with its now-dangling folder path) rather than re-docking it
- keep_folders=False sweeps the campaign root but not the checkpoint ledger: a resume then restores rows whose folder paths were recorded at sweep time; the interaction is sound (the sweep path is fingerprinted) but subtle
- The threaded stop lands at wave barriers, not instantly: a stop request during a wave waits for that wave's subprocesses (minutes per ligand with real engines)
- The GUI does not yet surface JobCancelled.partial_results or offer a resume action for any registered class — the API is library-side only
- The offscreen GUI battery was not re-run for 180 (round-5 recorded evidence carries it; R10 touches no GUI code)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
