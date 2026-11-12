# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.178.zip`
Size: 9083864 bytes; SHA-256: `90628d97dd15f532a2bd99ce87be81cec8d381fff5aaa51447420edc87406a0b`
Entries: 606; files: 555

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

- **R8 focused battery (hpo/presets/multiobjective/resumability×3/similarity/schema/tooling/numpy-neural/pipeline/notebook-variants/carriers + doctests)** — `python -m pytest tests/test_hpo.py tests/test_hpo_presets.py tests/test_multiobjective_hpo.py tests/test_hpo_resumability.py tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/ml/hpo.py src/pharmaml/chem/similarity.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py --doctest-modules`
  647 passed, 0 failed, 2 skipped, 0 blocked, 0 not-executed (collected: 649)
  Notes: run on main at the 1.2.178 bump, after the notebook ritual completed (an earlier concurrent run hit 8 notebook-variants failures from the ritual rewriting examples/ mid-read — rerun clean); zero failures; the 12-test HPO proof suite rides this battery
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.178 in 188s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness + variants guards green (119/119 in the focused battery)
- **Offscreen GUI battery (round-5 recorded evidence)** — `python -m pytest tests/ -k gui --no-header -q`
  0 passed, 0 failed, 0 skipped, 0 blocked, 532 not-executed (collected: 532)
  Notes: not executed this round; the round-5 recorded evidence (532/532 green at 1.2.174) carries the GUI leg; R8 touches no GUI code (the HPO trial loop is headless)

### Gates

- ruff (full src + the three resumability suites): **passed** — hpo.py, checkpoint_schema.py and all three proof suites clean; full src/pharmaml clean (4 findings fixed on the way: encode() simplifications, dead test code, quoted annotation)
- mypy src/pharmaml: **passed** — 217 files, zero errors
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.178 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.178, twins stripped, guard green

### Backup and recovery

- status: verified this round (push below re-verifies the remote)
- evidence: main carries R7 (2954bca) verified on the remote at round start; the R8 commit pushed to origin/main; the v1.2.178 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- Three §13 operation classes remain pending (generative_search, docking_batch, preprocessing) — each needs its own exact-restore proof round
- The RNG-state lesson generalises: generative_search and docking_batch will need the same audit of what their samplers/engines carry in-memory beyond the result log
- The GUI does not yet surface JobCancelled.partial_results or offer a resume action for any registered class — the API is library-side only
- The offscreen GUI battery was not re-run for 178 (round-5 recorded evidence carries it; R8 touches no GUI code)
- Notebook re-execution environment pins: fpdf2 2.8.5, meeko 0.8.0, gemmi 0.7.5 (per requirements.txt)
- tune_model's dataset fingerprint digests X/y content as contiguous arrays; a resume against a differently-ordered but row-identical frame intentionally refuses (exact restore or nothing)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
