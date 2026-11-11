# Part 5 — phase 0 baseline and restore audit

This is an **incremental, partial assessment**, not a full release certification.
The requested v1.2.173 ZIP and v1.2.173-specific master prompt were not supplied
in the attachment for this task. The checkout is a Git monorepo rooted at
`/home/user/pharma-ml-toolkit`; the Python toolkit and Studio are in the inner
`pharma-ml-toolkit/` directory and the companion Next.js web source is at the
repository root. The branch began at main commit `732f5538b5c69aa3997a5686b4aaaf47a6fed767`;
its prior landing-page change was committed as `8c1ca49` before this work.
The actual source `VERSION`, Python package version and web version are
**1.2.172**. The root citation intentionally has no version.

The available checked-in **v1.2.172** ZIP is 8,883,247 bytes with 531 entries
(486 files), SHA-256
`c2676af8db050b8f31c40bd36216e3de7ba4b8216a7067e321c421e6c5eb9857`.
Its CRC check passed. It has 45 example-data files recursively (16 top-level,
29 nested), and its three Windows engine binaries match the recorded MD5s in
`engines/README.md`. They were **not executed**: this Linux x86-64 environment
has no Wine. Presence, PE headers and checksums do not prove runtime behavior.
The [machine-readable baseline report](PART5_BASELINE_V1.2.172.json) includes
all archive file sizes and SHA-256 hashes, engine hashes, exclusions, smoke-test
scope and outstanding profiles. It describes the old checked-in archive, **not**
a new archive containing this patch.

Before editing, the archive was copied to the workspace backup directory
`/home/user/pharma-part5-backup/` and its SHA-256 matched the source. This is
not an off-site disaster-recovery backup. The original checked-in ZIP was not
modified or rebuilt. The prior branch work was already committed and pushed.

## Findings and completion matrix

Statuses refer only to the evidence in this phase. No unrun tests are counted as
passed. The PDF's requirements span several phases; this table is not a claim
that the entire prompt has been completed.

| Area | Status | Evidence / remaining work |
| --- | --- | --- |
| Bounded ZIP validation and exception rollback | **Partially implemented** | Root-level and historical prefixed archives are checked for unsafe or colliding paths, special files, size/ratio limits, CRC and required version carriers. Existing destinations are backed up and verified; caught install, mkdir, journal and verification failures roll back. Fault tests compare file bytes, modes and mtimes. Actual disk exhaustion, power loss, process kill and restart recovery remain untested. |
| Recovery information | **Partially implemented** | A sibling private recovery directory contains staged files, backup hashes and a journal; incomplete rollback retains it. Crash recovery is manual, not an automatic or crash-atomic transaction. Concurrent writers are unsupported. |
| Real archive smoke | **Completed and verified** | The available v1.2.172 ZIP was dry-run and restored into an isolated temporary directory; every restored member's SHA-256 matched the ZIP and an unrelated user file survived. This does not test the unavailable v1.2.173 archive. |
| Release ZIP QA report | **Partially implemented** | A static machine-readable report and manifest were produced for the existing v1.2.172 ZIP, not a new release build. Full profile results, standalone test suite and a reusable release-report generator remain outstanding. |
| Standalone release portability and canonical builders | **Partially implemented** | Original data, index and notebook-stripping generators now ship in `tools/`; checkout-only CLI forwarders preserve their historical paths. An explicitly selected targeted standalone profile and isolated candidate ZIP regression passed; the full release suite remains untested. See the continuation evidence below and [standalone testing](STANDALONE_TESTING.md). |
| Dependency profiles and CI | **Partially implemented** | The focused restore regression module was added to the existing Python CI matrix. GUI, Meeko/docking, chemistry, notebook, web parity, standalone and full-release profiles have not been provisioned or run in this phase. |
| Estimator validation / scikit-learn compatibility | **Not implemented** | `_estimator_type` assumptions remain in `ml/conformal.py` and `ml/dalex.py`; the PDF's 1.8-specific failure has not yet been reproduced locally. Pinned scikit-learn is 1.5.2; the declared range remains `>=1.3`. |
| Engine discovery and actual docking | **Not implemented** | `docking/workflow.py` has engine lookup APIs; the current engine README and feature inventory make conflicting claims. Search-order and runtime behavior were not changed or fully tested. No binary was executed. |
| Documentation and scientific feature inventory | **Partially implemented** | The example count mismatch and current-Tkinter and bundled-engine claims were located; the broader reconciliation and feature inventory have not been completed. |
| Dashboard, visualization and resumable jobs | **Not implemented** | Existing Studio architecture is preserved. No GUI behavior or scientific output was changed or tested. |
| v1.2.173-specific archive and validation | **Blocked by an external dependency/source** | The requested v1.2.173 ZIP and its version-specific prompt were not provided in this task. Do not treat v1.2.172 observations as v1.2.173 results. |
| Standalone-only absence of web source and Git | **Partially implemented** | Two checkout-only checks are skipped only when `--standalone-release` is explicitly selected. The isolated targeted selection passed; broader tests and web-parity behavior are not certified. |

Source inspection also found that the prior restore implementation staged files
but then used sequential `copy2` without backups or rollback. A later failure
could leave a mixture of old and new files. `--restore --dry-run` could also
materialize a missing Git-tracked ZIP, contrary to its no-write description.
Both paths are covered by this phase's regression tests. Invalid or missing
archive metadata now causes a refusal, including in dry-run; older synthetic
unit-test fixtures were updated to represent valid project metadata.

## Environment and commands

Linux x86-64, Python 3.11.2, Git 2.39.5; isolated environment: NumPy 1.26.4,
pandas 2.2.3, scikit-learn 1.5.2, SciPy 1.14.1, openpyxl 3.1.5, pytest 9.2.0,
Ruff 0.17.1. RDKit, Meeko, gemmi, PySide6, pyqtgraph and pytest-qt were not
installed in that environment. Bun and Wine were not available on PATH.

Run from the inner toolkit directory with the pinned core requirements and
pytest installed:

```bash
python -m pytest tests/test_checkpoint_restore_transaction.py -o addopts="-q --tb=short"
python -m pytest tests/test_checkpoint_tooling.py -o addopts="-q --tb=short"
python -m pytest tests/test_checkpoints.py tests/test_docs_index.py tests/test_conformal.py -o addopts="-q --tb=short"
ruff check tools/checkpoint.py tests/test_checkpoint_restore_transaction.py tests/test_checkpoint_tooling.py
```

Observed locally: **45 passed, 0 failed, 0 skipped** in the restore regression
module; **22 passed, 0 failed, 0 skipped** in adjacent checkpoint tooling;
**83 passed, 0 failed, 0 skipped** across checkpoint utilities, docs index and
conformal regression on scikit-learn 1.5.2; the root GitHub-page validator
passed and its unittest module had **10 passed, 0 failed, 0 skipped**; Ruff
passed. Full backend, optional chemistry/docking, GUI, notebooks, web parity,
standalone ZIP tests, release build and full suite: **not executed**. The
isolated actual-archive restore/byte comparison is a smoke check, not a full
standalone ZIP test profile. See [RESUME](RESUME.md#release-restore-safety-part-5-phase-0)
for constraints and manual recovery guidance.

## Continuation — standalone portability (targeted, not release certification)

The baseline v1.2.172 ZIP was restored into a separate temporary directory. With
RDKit 2026.9.1, the original provenance and checkpoint selection produced **27
passed, 5 failed, 0 skipped**: three failures depended on the excluded data
builder, one on excluded scratch forwarders, and one on Git origin metadata.
The data builder itself passed `--check` in the checkout with RDKit 2026.9.1;
RDKit 2024.9.6 instead reported drift in two outputs. No example data was
changed to force a match.

The existing toolkit-local generators were moved to `tools/`, with checkout
command-line shims at their former paths. The separately located top-level
historical index script is not the original toolkit generator and was not used.
Notebook source references were updated in both full and no-output variants,
without regenerating or fabricating execution outputs. The published notebook
execution script does not regenerate twins itself; documentation now gives the
separate stripping command. The standalone option skips exactly the two marked
checkout-only tests; the default still executes them.

A **temporary candidate**, not a published release, was made using the release
builder's file-selection rules without running its publication/deletion routine.
Its ZIP had 513 files and passed CRC. In its isolated extraction, the package
import resolved to the extracted `src/` tree. The provenance, checkpoint, index,
and notebook-variant selection reported **179 passed, 0 failed, 2 skipped**;
the skipped tests are the checkout-only forwarders and origin checks. The data
builder `--check` reported 44 files byte-identical, 46 owned, no orphans and
passing verify pins (the synthetic inventory intentionally includes invalid
SMILES and failed docking examples). The same selection in the Git checkout
reported **181 passed, 0 failed, 0 skipped**. The new portability regression
reported **2 passed, 0 failed, 0 skipped**, including an isolated candidate and
proof that those two checks fail rather than silently skip without the profile.
These results use Python 3.11.2, scikit-learn 1.5.2, RDKit 2026.9.1 and nbformat.
The pre-existing archive and its backup were not overwritten. This is not the
requested v1.2.173 archive, the full standalone suite, a real engine test, or a
release build. The CI addition for this regression is pending its next run.
A combined checkout run including the 45 restore fault tests and the new
portability regression reported **228 passed, 0 failed, 0 skipped**. The old
scratch data-builder command also passed `--check`; both the original archive
and its backup still have SHA-256
`c2676af8db050b8f31c40bd36216e3de7ba4b8216a7067e321c421e6c5eb9857`.
