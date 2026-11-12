# Read-only release archive audit

`tools/audit_release.py` audits an **existing** archive; it does not build,
publish, replace or delete a release. From the toolkit root:

```bash
python tools/audit_release.py /path/to/pharma-ml-toolkit-v1.2.172.zip --output-dir /path/to/new-audit-directory
```

## Folding in recorded test-profile evidence (part5-continuation round 5, §14)

The `--profile-results <json>` option folds a **separately recorded** evidence
file into the report — the audit itself still never runs tests, engines,
backups or recoveries; everything from the file is labelled *recorded
evidence; not executed by this audit*. The file must carry
`"schema": "pharmaml-profile-results/2"` and is validated fail-closed for
honest labelling before any output is written:

- every profile needs a unique non-empty name, a recorded command, and all
  five outcome counts (`passed` / `failed` / `skipped` / `blocked` /
  `not_executed`, each an int ≥ 0);
- the counts must sum to the recorded `collected` total (the audit
  recomputes it — a drifted total refuses the whole file);
- unknown top-level, per-profile or outcome keys are rejected (a typo must
  never silently drop evidence);
- gate outcomes must be `passed` / `failed` / `not run`.

Profiles are recorded **separately and never merged** — no cross-profile
total exists anywhere in the report, and a failing or not-executed profile
does not fail the archive audit (it is honestly recorded, not certified).
`SUMMARY.md` renders the per-profile counts, the gates table, the
backup/recovery status and the runtime/platform/scientific limitations
from the same single source (`report.json`). See
`docs/release-audits/v1.2.174-part5-r5-*` for a shipped example (input
evidence, machine-readable report, rendered summary).

The output directory must not already exist. Its `report.json` records the
archive size and SHA-256, entry and file counts, sorted per-file sizes and
SHA-256 hashes, version carriers, exclusion-policy hits, example-data count,
and the three engine MD5s against the references in the archive's own engine
README. It uses the checkpoint restore validator's limits and path/collision
rules, verifies CRC, and performs a restore to a private temporary tree, checking
every restored member and an unrelated sentinel file. The archive hash is
rechecked afterward. Failed and unexecuted checks are labelled separately;
`SUMMARY.md` provides a human-readable overview. A failed archive check exits
nonzero and still writes a report when possible.

This is **not a release certification**. The report explicitly says test
profiles and engines were not executed and backup/recovery was not verified by
this command. It does not compare the archive to current checkout sources,
verify the absent companion-web carrier, audit all documentation/scientific
claims, or prove power-loss recovery. MD5 comparison checks agreement with the
archive's own references, not their independent authenticity. A successful
isolated restore does not replace fault-injection tests, independently verified
backup, full test profiles or an actual engine invocation. Consult the
[Part 5 audit](PART5_PHASE0_AUDIT.md) for the separate observed profile and
backup results.

The audit extracts only to its own temporary directory. Do not invoke the
existing publishing ritual merely to obtain an audit: it has overwrite and
retention behavior and requires an independently reviewed safe release plan.

## Checkpoint build and publication limits

The checkpoint tool's individual ZIP build step now uses a private staging
directory, CRC-checks the result and refuses to replace an existing output;
publication uses an exclusive hard link and fails closed if the filesystem
does not support it. It does not clean or reuse the old fixed staging directory.
This is **not** a safety certification for either publishing ritual: checkpoint
`--commit` also edits version carriers, writes safe copies and retires old
archives, while `tools/build_zip.py` can scrub the source tree, replace files
and delete predecessor copies. Do not run either against valuable releases
until their full publication and retention behavior is independently reviewed
and protected by verified backups. Neither step is crash-durable or guarded
against hostile concurrent changes.
