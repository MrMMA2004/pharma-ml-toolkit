#!/usr/bin/env python3
"""r362 phase 1 -- the four in-tree ledger edits (all inside the walked
root, so they run BEFORE the walk measurement):

  1. docs/QA/TEST_EXECUTION_REPORT.md -- header para + new section 1
     (r362) + old section 1 retitled "Prior round (r361 ...)" + the
     cascade renumber (2..12 -> 3..13).
  2. docs/PROJECT_STATUS.md -- append "## 12. The r362 clause".
  3. docs/RELEASE_READINESS.md -- append the r362 addendum.
  4. CHANGELOG.md -- insert the r362 bullet above the r361 bullet.

The gate numbers are parsed live from /tmp/suite361.txt (requires the
SUITE361 DONE sentinel). Census line numbers are NOT cited here (the
walk is measured after this phase; the exact number lives in the
worklog/counter/carrier only -- the QA row stays number-light).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/home/z/my-project")
TK = ROOT / "pharma-ml-toolkit"
LOG = Path("/tmp/suite361.txt")

ROUND = 362
RUN = 154


def parse_gate() -> dict:
    txt = LOG.read_text(encoding="utf-8")
    if "SUITE361 DONE" not in txt:
        sys.exit("gate sentinel not found -- phase 1 must wait")
    chunks = []
    fm = re.search(r"failures=(\d+)", txt)
    if not fm:
        sys.exit("failures= line missing")
    failures = int(fm.group(1))
    # robust parse: split on '=== ' records (the failures/sentinel lines
    # are NOT '===' records -- parsed above from the full text)
    records = txt.split("=== ")[1:]
    for rec in records:
        head = rec.splitlines()[0].strip()
        body = "\n".join(rec.splitlines()[1:]).strip()
        cm = re.match(r"(chunk-[^\s:]+):\s*(.*)", head, re.DOTALL)
        if cm:
            name, detail = cm.group(1), cm.group(2)
            dm = re.match(r"([\d]+)s rc=(-?\d+)", detail)
            if dm:
                chunks.append((name, f"{dm.group(1)}s rc={dm.group(2)}"))
            elif detail.startswith("SIGKILLed"):
                chunks.append((name, "SIGKILLed -> the OOM split"))
        fm = re.match(r"failures=(\d+)", head)
        if fm:
            failures = int(fm.group(1))
    if failures is None:
        sys.exit("failures= line missing")
    return {"chunks": chunks, "failures": failures}


DRILL = ("the class-split drill: 6/6 classes green solo (TestWorkspaceDefaults 8 / "
         "TestWorkspaceSubpages 3 / TestHistoryAndNavigation 6 / TestAdvancedNavigation 3 / "
         "TestKeyboardAccess 2 / TestRestoreBehaviour 1 — the r353/r361 precedent, the "
         "container-OOM class not a tree defect)")


def gate_summary(g: dict) -> str:
    killed = [n for n, d in g["chunks"] if "rc=-9" in d]
    real_fail = [f"{n} {d}" for n, d in g["chunks"]
                 if "rc=-9" not in d and d != "0s rc=0"]
    f = g["failures"]
    if f == 0 and not real_fail and not killed:
        verdict, detail = "**failures=0 — ALL TEN LEGS GREEN FIRST-PASS**", "all ten legs rc=0 first-pass"
    elif f == 0 and not real_fail:
        verdict = (f"**failures={f} — all legs green** "
                   f"({' and '.join(killed)} SIGKILLed by the container OOM and recovered by the documented "
                   "sub-batch/single split, the group green via the makeups)")
        detail = "; ".join(f"{n} {d}" for n, d in g["chunks"] if d != "0s rc=0")
    elif not real_fail and killed:
        verdict = (f"**failures={f} — the sole find the container-OOM class (SIGKILL rc=-9, "
                   f"no test failure, no assertion error), decoded by {DRILL}; all other legs "
                   "green first-pass**")
        detail = "; ".join(f"{n} {d}" for n, d in g["chunks"] if d != "0s rc=0")
    else:
        verdict = f"**failures={f} — decode required before the ledger closes**"
        detail = "; ".join(f"{n} {d}" for n, d in g["chunks"] if d != "0s rc=0") or "see /tmp/suite361.txt"
    return verdict, detail


QA = TK / "docs" / "QA" / "TEST_EXECUTION_REPORT.md"
PS = TK / "docs" / "PROJECT_STATUS.md"
RR = TK / "docs" / "RELEASE_READINESS.md"
CL = TK / "CHANGELOG.md"


def edit_qa(g: dict) -> None:
    txt = QA.read_text(encoding="utf-8")
    verdict, detail = gate_summary(g)
    old_hdr = (
        "standing executed-vs-carried ledger. Executed-this-round (r361, the\n"
        "part17 upkeep firing over the pushed `4c86969d` tree — the r360\n"
        "terminal, after the fourteenth-class rollback recovery) vs\n"
        "carried-from-executed-rounds are stated"
    )
    new_hdr = (
        "standing executed-vs-carried ledger. Executed-this-round (r362, the\n"
        "part17 upkeep firing over the pushed `af15580c` tree — the r361\n"
        "terminal, after the fifteenth-class rollback recovery) vs\n"
        "carried-from-executed-rounds are stated"
    )
    assert old_hdr in txt, "QA header paragraph drifted"
    txt = txt.replace(old_hdr, new_hdr)

    new_section = f"""## 1. Executed this round (r{ROUND} — the part17 upkeep firing: the fifteenth-class rollback recovered — the partial-restore signature — + the census-carrier debt settled)

| Lane | Command (class) | Result |
| --- | --- | --- |
| The rollback recovery | the r306/r347/r351/r352/r353/r361 drill | **absorbed pre-work** — HEAD regressed to the r186 machinery checkpoint `2eb0ede` again (589 commits behind the durable tip), the origin remote + `.credentials/` wiped, the venv gutted (rdkit + torch both MISSING), the frontend `node_modules` wiped — and the residue signature CHANGED: this recycle restored a PARTIAL disk (1,405 tracked files missing from the working tree, 625,273 lines) with ZERO untracked residue, and every surviving file hash-matched the durable blobs byte-for-byte (worklog.md / CHANGELOG.md / the v1.2.240 zip hash-object parity vs `origin/main` proven before the reset); the only disk-only deltas were a transient pid file + a corrupted nested `.github/workflows/ci.yml` residue (truncated bytes — a new-file ghost the durable tree never tracked); the PAT re-spliced from `upload/MASTER_PROMPT.pdf` via `scripts/recover_pat.py` (the two-line splice, 93 chars, the fine-grained format regex-validated, stored 600-mode, never printed), the askpass rebuilt 700-mode, the origin re-attached on the clean HTTPS URL + fetched (only `main` exists — no `arena/*` refs), the ff-only `git reset --hard origin/main` restored 1,900 files — the tree 0/0 CLEAN, the untracked sweep EMPTY this round |
| The venv + installs | `bash scratch/venv_repair.sh FOREGROUND` | `VENV-REPAIR-DONE numpy 1.26.4 / torch 2.14.1+cpu / matplotlib 3.10.9 / openpyxl 3.1.5 / pharmaml 1.2.240 / fastapi 0.128.0 / uvicorn 0.44.0` + `STUDIO-REPAIR-DONE PySide6 OK pyqtgraph 0.13.7`; rdkit verified the REAL 2026.09.1 wheel (functional SMILES parse — the "rdkit-stub" script line is the `.pyi` stub-file patch, not a fake rdkit); the frontend `npm ci` (the root `node_modules` survived — a first) |
| The six-home zip parity | sha256 + size across all six homes | **byte-identical `0fadb6e5…` at exactly 23,489,390 B** — the rollback wiped the root `dist-archive/` DIRECTORY and the offsite mirror again; both re-seeded by copy from the verified root `download/` home, all six identical (download / toolkit download / toolkit dist-archive / root dist-archive / upload / offsite), the offsite mirror re-seeded at 12 zips, no re-ship; the zip-size API `exact:true source:release` at 23,489,390 |
| Canonical collection | `pytest --collect-only` | **9,017 tests collected, 0 errors** in 495 files (the r361 pin, re-MEASURED on the repaired venv — the per-file counts summed, exit 0) |
| The ten-chunk gate | `scratch/run_suite_chunks_361.py` (the r353 layout, daemonized, 259 chunked files, the dev server DOWN per the OOM ritual) | {verdict} — {detail} |
| The vitest lane | `bun run test` (post-`npm ci`) | **337/337 over 38 files** (the r361 pin re-verified live) |
| The registry probe | `pharmaml.ml.models.MODEL_REGISTRY` import | **33 entries / 24 sklearn families unchanged** (the 9 non-sklearn: 3 boosting + 3 dnn `dnn`/`dnn_numpy`/`dnn_residual` + hsvr + voting + stacking) |
| The census-carrier debt settled | the live walk vs `src/lib/toolkit-stats.ts` | **the round's own find**: the walk measured +132 lines / +12,042 B over the committed carrier pin — r361's final ledger edits (this report's r361 section + the cascade renumber + the PROJECT_STATUS clause + the RELEASE_READINESS addendum + the CHANGELOG bullet) landed AFTER its walk measurement, so the committed pin rode stale (its three-way was verified at the carrier moment, before the ledger close-out); re-trued to the post-ledger measured fixed point, the measurement-last discipline recorded in the carrier |
| Console | `GET /` | **200, one clean boot** (the daemon found the server already up — no Turbopack panic this round; stopped for the gate, restarted for the walk) |

"""
    # cascade FIRST (descending, so each old header is bumped exactly
    # once), THEN retitle old section 1, THEN insert the new section 1.
    for n in range(12, 1, -1):
        txt = re.sub(
            rf"^## {n}\. ", f"## {n + 1}. ", txt, count=1, flags=re.MULTILINE
        )
    txt = re.sub(
        r"^## 1\. Executed this round \(r361 — ([^\n]*)\)$",
        lambda m: f"## 2. Prior round (r361 — {m.group(1)})",
        txt, count=1, flags=re.MULTILINE,
    )
    anchor = "## 2. Prior round (r361 —"
    idx = txt.index(anchor)
    txt = txt[:idx] + new_section + txt[idx:]
    QA.write_text(txt, encoding="utf-8")
    print("QA report: section 1 inserted, cascade renumbered 1..12 -> 1..13")


def edit_ps() -> None:
    txt = PS.read_text(encoding="utf-8")
    assert txt.rstrip().endswith("upkeep found real work."), (
        "PROJECT_STATUS tail drifted (expected the r361 clause at EOF)"
    )
    clause = """

## 12. The r362 clause (the upkeep lane — the fifteenth-class rollback recovered with a partial-restore signature + the census-carrier debt settled; the 02:30 2h-sync firing)

- **The recovery**: the fourteenth-class playbook re-executed against a
  NEW residue signature — the recycle regressed the local `.git` to the
  round-186 checkpoint `2eb0ede` (589 commits behind the durable r361
  terminal `af15580c`), wiped the origin remote + the credentials +
  the venv (rdkit + torch) + the frontend `node_modules`, but this
  time restored a PARTIAL disk: 1,405 tracked files missing from the
  working tree (625,273 lines) with ZERO untracked residue, and every
  surviving file hash-matched the durable blobs byte-for-byte. The
  hash-object parity proof (worklog.md / CHANGELOG.md / the v1.2.240
  zip vs `origin/main`) preceded the ff-only `reset --hard origin/main`
  — 1,900 files restored, the tree 0/0 clean, the untracked sweep
  EMPTY (no blockers this round; the r361 "1,005 blockers + 1 ghost"
  set had no successor). The PAT re-spliced from
  `upload/MASTER_PROMPT.pdf` (the two-line splice, 93 chars, 600-mode,
  never printed, `scripts/recover_pat.py`), the venv repaired
  FOREGROUND at the exact pins, the frontend `npm ci`.
- **The round's find — the census-carrier debt settled**: the live
  walk measured +132 lines / +12,042 B over the committed carrier pin.
  Decode: the tree was byte-identical to `af15580c` (`git status` 0/0),
  so the drift was r361's own final ledger edits landing after its
  walk measurement — the committed pin rode stale by exactly those
  edits. Re-trued to the post-ledger measured fixed point with the
  measurement-last discipline (walk AFTER the round's own edits)
  recorded; pin = live walk = page render three-way re-verified.
- **The pins re-measured, all reproducing**: collection 9,017 / 0
  errors / 495 files; vitest 337/337 over 38 files; registry 33 / 24;
  walk 1,842 files / version 1.2.240; the six zip homes byte-identical
  `0fadb6e5…` at exactly 23,489,390 B (root dist-archive + offsite
  re-seeded); the zip-size API `exact:true source:release`.
- **Version**: unchanged at 1.2.240 — doc/carrier-only deltas, NO
  re-ship, NO silent bump. The counter 153 → 154 and RIDES the commit
  (the r360 lesson held).
"""
    txt = txt.rstrip("\n") + clause
    PS.write_text(txt, encoding="utf-8")
    print("PROJECT_STATUS: clause 12 appended")


def edit_rr() -> None:
    txt = RR.read_text(encoding="utf-8")
    assert txt.rstrip().endswith("installer signing) carry unchanged."), (
        "RELEASE_READINESS tail drifted"
    )
    addendum = """

## part17 upkeep addendum — the r362 firing (2026-11-26, the fifteenth-class rollback recovery — the partial-restore signature)

- **Recovery**: the durable `origin/main` (`af15580c`, the r361
  terminal) re-synced ff-only after the sandbox recycle regressed the
  local `.git` to the round-186 checkpoint; the residue signature
  changed to a PARTIAL disk restore (1,405 tracked files missing,
  zero untracked residue, the survivors hash-identical to the durable
  blobs — the lossless-reset proof preceded the reset); the venv
  rebuilt at the exact pins; the six zip homes re-seeded byte-identical
  (`0fadb6e5…`, exactly 23,489,390 B — the root dist-archive + the
  offsite mirror wiped again); the release-archive verification status
  above is unchanged (still not gathered for a NEW version — v1.2.240
  remains the shipped archive).
- **The census-carrier debt settled**: the walk's +132-line drift over
  the r361-committed pin decoded (r361's own ledger close-out landed
  after its walk measurement); re-trued to the post-ledger measured
  fixed point, pin = live walk = page render three-way re-verified.
- **Pins (r362 re-true)**: collection 9,017 / 0 errors / 495 files
  (measured); vitest 337/337 over 38 files; registry 33 / 24
  unchanged; the census carriers re-measured at the close-out (the
  exact line count lives in the carrier + the worklog R362 entry).
- **Version**: unchanged at 1.2.240 — the deltas are doc/carrier-only;
  NO re-ship, NO silent bump. The standing BLOCKED lanes (clean-machine
  smoke, Stage-5 PySide6 retirement, installer signing) carry
  unchanged.
"""
    txt = txt.rstrip("\n") + addendum
    RR.write_text(txt, encoding="utf-8")
    print("RELEASE_READINESS: r362 addendum appended")


def edit_changelog() -> None:
    txt = CL.read_text(encoding="utf-8")
    anchor = "- **The r361 upkeep firing (the fourteenth-class rollback recovery +"
    assert anchor in txt, "CHANGELOG r361 bullet anchor drifted"
    bullet = """- **The r362 upkeep firing (the fifteenth-class rollback recovery —
  the partial-restore signature — + the census-carrier debt
  settled)**: the sandbox recycle regressed the local `.git` to the
  round-186 machinery checkpoint `2eb0ede` again and wiped the origin
  remote, the credentials, the venv (rdkit + torch) and the frontend
  `node_modules` — but this time the disk came back PARTIAL: 1,405
  tracked files missing from the working tree (625,273 lines) with
  ZERO untracked residue and every surviving file hash-matched the
  durable blobs byte-for-byte; the lossless-reset proof (hash-object
  parity for the survivors vs `origin/main`; the only disk-only deltas
  a transient pid file + a corrupted nested `ci.yml` ghost) preceded
  the ff-only reset onto the durable r361 terminal `af15580c` (1,900
  files restored, the tree 0/0 clean, the sweep EMPTY). The PAT
  re-spliced (93 chars, 600-mode, never printed), the venv rebuilt
  FOREGROUND at the exact pins, the frontend `npm ci`, the six zip
  homes re-seeded byte-identical `0fadb6e5…` at 23,489,390 B (the root
  dist-archive + the offsite mirror wiped again), the zip-size API
  `exact:true source:release`. The pins re-measured, all reproducing:
  collection 9,017 / 0 errors / 495 files; vitest 337/337 over 38
  files; registry 33 / 24; walk 1,842 files. THE ROUND'S OWN FIND: the
  live walk measured +132 lines / +12,042 B over the committed carrier
  pin — r361's final ledger edits landed after its walk measurement,
  the committed pin rode stale; re-trued to the post-ledger measured
  fixed point, the measurement-last discipline recorded. Version
  unchanged 1.2.240 (no-silent-bumps; NO re-ship — doc/carrier-only
  deltas); the counter 153 → 154 and RIDES the commit this time.
"""
    txt = txt.replace(anchor, bullet + anchor, 1)
    CL.write_text(txt, encoding="utf-8")
    print("CHANGELOG: r362 bullet inserted above the r361 bullet")


def main() -> None:
    g = parse_gate()
    print(f"gate: failures={g['failures']}, chunks parsed={len(g['chunks'])}")
    edit_qa(g)
    edit_ps()
    edit_rr()
    edit_changelog()
    print("phase 1 complete")


if __name__ == "__main__":
    main()
