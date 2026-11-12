# Interrupted release: read-only inventory and manual recovery

A multi-home build or ship can stop after creating some final names; a rerun
will refuse to overwrite them. Neither publisher is a crash-atomic transaction.
Stop concurrent publishers and preserve the working tree, logs, archive copies,
and any restore recovery directories before deciding what to do. Do not delete
or overwrite a release to make a rerun succeed. An interrupted build's temporary
candidate may no longer exist; a newly built ZIP is not necessarily byte-identical.

If a candidate is available and its provenance has been established independently,
inspect explicitly named destinations **without modifying them**:

```sh
python tools/release_recovery_status.py /path/to/preserved-candidate.zip \
  /path/to/home-one/pharma-ml-toolkit-vX.Y.Z.zip \
  /path/to/home-two/pharma-ml-toolkit-vX.Y.Z.zip
```

The command checks candidate CRC, reports its SHA-256 and identifies missing,
matching, different, non-regular and symlink destinations. Exit 0 means all
listed files matched; exit 1 means some do not; exit 2 means the inventory could
not be performed. It does not follow final-path symlinks, publish, repair, remove,
validate source provenance or prove scientific correctness. Hard links to the
candidate are flagged: they are not independent backups. The inventory assumes
no concurrent changes and does not protect against hostile filesystem races.

Preserve mismatching copies and investigate rather than selecting a winner by
version or timestamp. If copies match, verify the intended source, archive audit,
backup location, and all homes independently; choose any manual recovery action
only after review and authorization. A missing copy should be restored with an
exclusive no-clobber method and verified by hash; never replace existing files.
Keep a verified independent backup outside all publication homes, preferably on
separate storage. A matching copy on the same disk is not off-site or protection
against disk loss. Existing retention rules only retire predecessors with a
matching independent configured backup; they are not a backup strategy.

Restore failures are separate: preserve `.restore-recovery-*`, its `journal.json`
and `backups/`, plus the current tree. Ordinary exceptions have tested rollback;
power loss, process termination and incomplete rollback require human review.
Do not automatically replay a journal or delete a recovery directory. Audit the
journal, target files and backup hashes first and preserve evidence of any conflict.
See [release audit](RELEASE_AUDIT.md) and [test profiles](TEST_PROFILES.md).
