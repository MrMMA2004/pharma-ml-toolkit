# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.201.zip`
Size: 10758027 bytes; SHA-256: `4d0336c296ccac24ad5fc1a95eeacc0c07f3401e12e90a58c8461f327a5ae1b0`
Entries: 771; files: 711

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

- **review-fix regression batch 1 (touched pages)** — `pytest tests/ui/test_model_analysis_page.py tests/ui/test_chemistry_page.py tests/ui/test_pkpd_page.py tests/ui/test_biology_page.py tests/ui/test_deep_learning_page.py`
  76 passed, 3 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 79)
- **review-fix regression batch 2 (touched pages)** — `pytest tests/ui/test_docking_page.py tests/ui/test_foundations_page.py tests/ui/test_unsupervised_page.py tests/ui/test_generative_page.py tests/ui/test_bestofml_page.py tests/ui/test_responseai_page.py`
  43 passed, 2 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 45)
- **review-fix regression batch 3 (windows + dialogs)** — `pytest tests/ui/test_foundations_page.py tests/ui/test_docking_page.py tests/ui/test_molecular_dynamics_page.py tests/ui/test_main_window.py tests/ui/test_nine_cards.py`
  71 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 71)
- **web parity (session report + explorer contracts)** — `pytest tests/test_web_parity.py`
  78 passed, 0 failed, 1 skipped, 0 blocked, 0 not-executed (collected: 79)
- **version carriers + notebook freshness gates** — `pytest tests/test_version_carriers.py tests/test_notebook_variants.py`
  126 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 126)
- **torch-dependent retest (after torch-cpu install)** — `pytest tests/ui/test_deep_learning_page.py tests/ui/test_generative_page.py`
  58 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 58)

### Gates

- torch-dependent failures were environmental, not regressions: **passed** — the 5 failures from batches 1-2 (deep_learning torch backend x3, generative RL/MCTS x2) all PASS after installing torch-cpu 2.15.0 — they required the pharmaml[deep] extra, absent from the venv at first sweep time
- tsc --noEmit on src/: **passed** — zero errors under src/; 18 pre-existing errors in examples/, scratch/, skills/ verified identical on the pre-change tree
- notebook freshness at 1.2.201: **passed** — tools/reexec_notebooks.py re-executed all 8 executed notebooks (meeko+gemmi+torch reinstalled first); every notebook reports error-outputs=0 and home-path-survivors=0; twins regenerated with tools/strip_notebook_outputs.py; the drift guard test_notebook_variants passes 126/126 with the carrier test

### Backup and recovery

- status: not verified in this round
- detail: no crash-recovery drill was run; the zip builder's no-clobber publication and CRC/leak checks passed

### Runtime / platform / scientific limitations

- The full 5,000+ test suite was not run end-to-end this round (chucked spot-batches over every touched page + the web parity + carrier/notebook gates instead).
- No GUI screenshot walk was re-shot at 1.2.201; the R31/R32 fixes were live-verified headlessly per page instead.
- The web console's Next build was not exercised (tsc only) — no code paths changed beyond the reviewed fixes.

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
