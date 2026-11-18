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
version or timestamp. Verify source provenance, archive audit, backup location,
and all homes independently; review and authorize any recovery action first.

`tools/resume_release.py` offers an explicitly invoked recovery operation after
that review. By default it is a **read-only plan**. Supply the trusted candidate,
a separate regular-file backup outside all publication homes, the independently
recorded SHA-256, and every intended destination (all parents must already exist;
all destination names must be the same versioned release name):

```sh
python tools/resume_release.py --candidate /path/to/candidate.zip \
  --backup /path/outside/homes/backup.zip --sha256 <verified-64-char-sha256> \
  /path/home-one/pharma-ml-toolkit-vX.Y.Z.zip \
  /path/home-two/pharma-ml-toolkit-vX.Y.Z.zip
```

Once the plan, provenance and backup are reviewed, repeat with `--execute` to
publish **only missing** names using verified staged copies and exclusive links.
Existing matching copies are left untouched; any mismatching file, symlink,
aliased backup or invalid ZIP aborts preflight. The command never builds,
commits, retires predecessors or replaces existing paths. Already-published
copies remain if a later step fails; stop and inventory again. It requires hard
link support in each home, assumes no concurrent writers, and is neither
crash-atomic nor power-loss durable. CRC and matching hashes do not prove archive
safety, source provenance, or release readiness. A controlled SIGKILL rehearsal
in synthetic homes exercised one interruption and successful recovery; it is not
proof of all failure modes. Do not use this operation to choose among conflicting
archives. Keep a verified independent backup outside all publication homes,
preferably on separate storage. A matching copy on the same disk is not off-site or protection
against disk loss. Existing retention rules only retire predecessors with a
matching independent configured backup; they are not a backup strategy.

Restore failures are separate: preserve `.restore-recovery-*`, its `journal.json`
and `backups/`, plus the current tree. Ordinary exceptions have tested rollback;
power loss, process termination and incomplete rollback require human review.
Do not automatically replay a journal or delete a recovery directory. Audit the
journal, target files and backup hashes first and preserve evidence of any conflict.
See [release audit](RELEASE_AUDIT.md) and [test profiles](TEST_PROFILES.md).
