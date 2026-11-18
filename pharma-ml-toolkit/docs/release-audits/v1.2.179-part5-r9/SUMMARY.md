# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.179.zip`
Size: 9129575 bytes; SHA-256: `4146a0ea5a9124fac85095ed1a767adfc4d8a81444b978a98b52c81d553ca1f7`
Entries: 611; files: 559

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

- **R9 audit battery (hpo ×4, generative-search proofs + regression, similarity ×2, descriptor-batch, schema, tooling, numpy-neural, pipeline, notebook-variants, carriers + doctests)** — `python -m pytest tests/test_hpo.py tests/test_hpo_presets.py tests/test_multiobjective_hpo.py tests/test_hpo_resumability.py tests/test_generative_search_resumability.py tests/test_generative.py tests/test_generative_result_sa.py tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/ml/hpo.py src/pharmaml/generative/genetic.py src/pharmaml/chem/similarity.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py --doctest-modules`
  982 passed, 0 failed, 2 skipped, 0 blocked, 0 not-executed (collected: 984)
  Notes: run on main at the 1.2.179 bump, AFTER the notebook ritual completed (round-8's lesson: never overlap the battery with the ritual rewriting examples/); the 10-test generative-search proof suite rides this battery, as do the 322-test generative regression suites (test_generative.py + test_generative_result_sa.py) guarding the historical GA behaviour
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.179 in 173s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness + variants guards green in the audit battery
- **Offscreen GUI battery (round-5 recorded evidence)** — `python -m pytest tests/ -k gui --no-header -q`
  0 passed, 0 failed, 0 skipped, 0 blocked, 532 not-executed (collected: 532)
  Notes: not executed this round; the round-5 recorded evidence (532/532 green at 1.2.174) carries the GUI leg; R9 touches no GUI code (the GA loop is headless)

### Gates

- ruff (full src + the four resumability suites): **passed** — genetic.py, checkpoint_schema.py and all four proof suites clean; full src/pharmaml clean
- mypy src/pharmaml: **passed** — 217 files, zero errors
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.179 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.179, twins stripped, guard green

### Backup and recovery

- status: verified this round (push below re-verifies the remote)
- evidence: main carries R8 (b23c4e6) verified on the remote at round start; the R9 commit pushed to origin/main; the v1.2.179 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- Two §13 operation classes remain pending (docking_batch, preprocessing) — each needs its own exact-restore proof round
- The resumed-vs-uninterrupted comparison necessarily excludes metadata.elapsed_s (wall-clock by nature); every substance field — smiles, scores, history, breakdown, all other metadata keys — is asserted bit-identical
- The config fingerprint covers the objective's class identity, not its instance data (e.g. a SimilarityObjective built against a different reference keeps the same class name); corpus/seed_smiles are NOT in the fingerprint either — a resume against a different corpus replays the completed generations but continues from the checkpointed population (documented in the plan doc as a known limit for docking_batch to close)
- The cooperative stop attaches the partial GenerationResult in both modes — a deliberate, strictly richer deviation from the round-205 None convention for checkpoint-less loops, pinned by its own test
- The GUI does not yet surface JobCancelled.partial_results or offer a resume action for any registered class — the API is library-side only
- The offscreen GUI battery was not re-run for 179 (round-5 recorded evidence carries it; R9 touches no GUI code)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
