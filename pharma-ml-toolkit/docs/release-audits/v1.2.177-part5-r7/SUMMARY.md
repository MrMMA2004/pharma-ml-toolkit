# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.177.zip`
Size: 9039779 bytes; SHA-256: `4899aa82a62890dd5ecfd8237f8e75e2db7682ecb97381952063ea747d02d82d`
Entries: 601; files: 551

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

- **R7 focused battery (similarity/descriptor-batch/schema/tooling/numpy-neural/pipeline/cli/notebook-variants/carriers + doctests)** — `python -m pytest tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_cli_features.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/chem/similarity.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py src/pharmaml/exceptions.py --doctest-modules`
  564 passed, 0 failed, 5 skipped, 0 blocked, 0 not-executed (collected: 569)
  Notes: run on main at the 1.2.177 bump; zero failures; the 11-test similarity-campaign proof suite rides this battery
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.177 in 400s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness guards green (119/119)
- **Offscreen GUI battery (round-5 recorded evidence)** — `python -m pytest tests/ -k gui --no-header -q`
  0 passed, 0 failed, 0 skipped, 0 blocked, 532 not-executed (collected: 532)
  Notes: not executed this round; the round-5 recorded evidence (532/532 green at 1.2.174) carries the GUI leg; R7 touches no GUI code (the similarity campaign loop is headless)

### Gates

- ruff (full src + the two resumability suites): **passed** — similarity.py, checkpoint_schema.py and both proof suites clean; full src/pharmaml clean
- mypy src/pharmaml: **passed** — 217 files, zero errors
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.177 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.177, twins stripped, guard green

### Backup and recovery

- status: verified this round (fresh clone from origin/main after a sandbox recycle; push below re-verifies the remote)
- evidence: main cloned at f0f40a1 (v1.2.176) after the recycle; the R7 commit pushed to origin/main; the v1.2.177 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- Four §13 operation classes remain pending (hpo, generative_search, docking_batch, preprocessing) — each needs its own exact-restore proof round
- The GUI does not yet surface JobCancelled.partial_results or offer a resume action for similarity campaigns or descriptor batches — the API is library-side only this round
- The offscreen GUI battery was not re-run for 177 (round-5 recorded evidence carries it; R7 touches no GUI code)
- Notebook re-execution environment pins: fpdf2 2.8.5, meeko 0.8.0, gemmi 0.7.5 (per requirements.txt)
- The similarity campaign checkpoints the long-format result rows, not the raw similarity matrix — a resume replays scored pairs; the trade-off keeps checkpoint size linear in kept pairs (a threshold near 0 against a large library makes checkpoints large)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
