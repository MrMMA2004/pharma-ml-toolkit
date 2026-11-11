# Read-only release archive audit

`tools/audit_release.py` audits an **existing** archive; it does not build,
publish, replace or delete a release. From the toolkit root:

```bash
python tools/audit_release.py /path/to/pharma-ml-toolkit-v1.2.172.zip --output-dir /path/to/new-audit-directory
```

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
