# Archive audit — not a release certification

Archive: `pharma-ml-toolkit-v1.2.182.zip`
Size: 9267990 bytes; SHA-256: `36a2f453bf510760fb51c34e635d5b517bd88b0a4a5ce84ea721a9e08754c872`
Entries: 625; files: 570

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

- **R12 full offscreen UI battery (all 37 suites, run in two halves)** — `python -m pytest $(ls tests/ui/test_*.py | sort | head -19)  # then | tail -18`
  537 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 537)
  Notes: 291 + 246 across the two halves; includes the 5 new round-12 tests (3 worker + 2 page). HONEST CAVEAT: the first single-process full run was OOM-killed at ~42% (exit 137) when it collided with the notebook ritual in the same 4 GB sandbox; the two-half reruns (clean memory, ritual finished) are the recorded evidence. This is a sandbox memory limit, not a product finding.
- **R12 audit battery (§13 proofs ×6 + docking/chemistry regression + schema/tooling/pipeline/carriers + doctests)** — `python -m pytest tests/test_hpo.py tests/test_hpo_presets.py tests/test_multiobjective_hpo.py tests/test_hpo_resumability.py tests/test_generative_search_resumability.py tests/test_generative.py tests/test_generative_result_sa.py tests/test_similarity.py tests/test_similarity_campaign_resumability.py tests/test_descriptor_batch_resumability.py tests/test_docking_batch_resumability.py tests/test_docking.py tests/test_campaign.py tests/test_engine_discovery_part5.py tests/test_docking_prep.py tests/test_preprocessing_resumability.py tests/test_checkpoint_schema.py tests/test_checkpoint_tooling.py tests/test_numpy_neural_checkpoints.py tests/test_ml_pipeline.py tests/test_notebook_variants.py tests/test_version_carriers.py src/pharmaml/ml/hpo.py src/pharmaml/generative/genetic.py src/pharmaml/chem/similarity.py src/pharmaml/docking/smiles_docking.py src/pharmaml/ml/preprocessing.py src/pharmaml/utils/checkpoint_schema.py src/pharmaml/utils/checkpoints.py --doctest-modules`
  1273 passed, 0 failed, 2 skipped, 0 blocked, 0 not-executed (collected: 1275)
  Notes: run on main at the 1.2.182 bump, AFTER the notebook ritual completed (the round-8 scheduling rule)
- **Ship safety + release portability** — `python -m pytest tests/test_checkpoint_ship_safety.py tests/test_release_portability.py`
  16 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 16)
  Notes: post-bump sanity on main
- **Notebook re-execution at ship (8/8)** — `python tools/reexec_notebooks.py && python tools/strip_notebook_outputs.py`
  8 passed, 0 failed, 0 skipped, 0 blocked, 0 not-executed (collected: 8)
  Notes: 8/8 executed at pharmaml 1.2.182 in 199s; zero error outputs, zero home-path survivors; the 7 no-output twins regenerated; freshness + variants guards green in the audit battery

### Gates

- ruff (worker, chemistry page, both touched test suites): **passed** — all four round files clean
- mypy on the touched files: **passed** — ui/workers/job_worker.py + ui/pages/chemistry.py introduce ZERO new errors; the 7 errors mypy reports under the transitive ui/ tree are pre-existing (nine_cards.py, docking.py, dashboard.py) — proven identical on the stashed pre-R12 tree and excluded from this round's scope
- Four-carrier version pin (round-263): **passed** — VERSION, pyproject, __version__, SHIPPED_TOOLKIT_VERSION all 1.2.182 in one change
- Notebook freshness: **passed** — 8/8 re-executed at 1.2.182, twins stripped, guard green
- Back-compat pin (this round's core risk): **passed** — the cancelled(float) contract is untouched: test_historical_cancelled_handlers_stay_byte_identical (worker) + test_feature_stop_without_payload_keeps_the_historic_line (page); the full 537-test UI battery across all 15 JobRunner pages

### Backup and recovery

- status: verified this round (push below re-verifies the remote)
- evidence: main carries R11 (ddd93bd) verified on the remote at round start; the R12 commit pushed to origin/main; the v1.2.182 zip written to all four homes
- power_loss_recovery: not verified by this round

### Runtime / platform / scientific limitations

- The preserved-partial channel has ONE consumer page (chemistry feature factory, the R6 descriptor batch); the other §13-registered backends are not yet reachable from GUI pages (similarity campaigns, HPO, generative search, docking batches, chunked preprocessing run library-side or CLI-side only), so their partials ride the new channel only when a page adopts it
- The preserved line reports counts, not the resume affordance: a Stop mid-batch does not yet offer a Resume button (the checkpoints exist library-side; the GUI-side resume wiring is future work)
- The channel emits the payload as-is (object role): a malicious/corrupt payload is bounded by the page's isinstance guards (the chemistry page ignores non-(frame, report) tuples), but no schema validation rides the signal
- The first full-battery single-process run OOM-killed at ~42% when it collided with the notebook ritual in the 4 GB sandbox; the recorded evidence is the two-half rerun — future rounds should keep the battery and the ritual strictly sequential (this round's honest scheduling lesson, added to the worklog)

See `report.json` for the sorted per-file manifest, checksums, recorded profile evidence and limitations.
