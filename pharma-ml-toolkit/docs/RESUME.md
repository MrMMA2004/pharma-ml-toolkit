# RESUME — the anti-rollback protocol

_Last rebuilt 2026-11-02 after the rollback that ate rounds 180–182.
Updated 2026-11-03 after a second rollback ate round 184, and again the
same evening after a THIRD rollback ate the first round-185 ship._

## 2026-11-07 — Arena chart-readability addendum (v1.2.115)

The active branch remains `arena/2697c755-pharma-ml-toolkit`. Shared Matplotlib
figures now use a 9×6 layout, 120 display DPI, consistent typography and quiet
grid/spine styling. File/HTML exports default to 220/160 DPI with tight padded
bounds and reproducible SVG/PDF metadata. Playground hosts default to 390 px,
live figures preserve aspect ratio on resize, and the PNG fallback uses the
same high-resolution policy. Usage is documented in
`docs/CHART_READABILITY.md`; `examples/chart_readability.py` regenerates PNG,
SVG, PDF and HTML evidence under the ignored `artifacts/` tree.

The final applicable non-GUI suite exits 0 with 3,486 passed, 44 skipped and
24 deselected; 769 source doctests collect with zero failures (three skips).
The RL-inclusive visualization suite has 219 passed and six skipped. All eight
executed notebooks ran at 1.2.115, their no-output twins were regenerated and
all 119 notebook-variant guards pass. Native `test_playground.py` and RDKit
XRender-dependent groups were excluded because this image has no working
Tk/display, Xvfb or `libXrender`; no native GUI/X11 execution is claimed.

## 2026-11-06 — Arena research-catalog addendum (v1.2.114)

The active Arena branch is `arena/2697c755-pharma-ml-toolkit`; no older Arena
branch is part of this ship. The 140-repository disposition lives in
`docs/GITHUB_RESEARCH_CATALOG_AUDIT.md`. The new computation is covered by 27
focused tests and three module doctest groups. CI-style Ruff and mypy pass;
all eight notebooks executed at 1.2.114; the full non-Tk suite exits 0 on the
pinned NumPy 1.26.4 stack. This minimal Python build has no `tkinter` and the
image has no Xvfb, so the four real-Tk and three stub-callback regression
tests are retained but skipped, never reported as executed. Headless RDKit
rendering required a temporary external XRender loader shim because the image
also lacks `libXrender.so.1`; the shim is not part of the archive.

## The standing zip-safety rule (2026-11-03, user instruction)

*"After each round, save the zip file in a safe place so that rollback
does not occur."*  Baked into `tools/checkpoint.py`'s ship ritual —
every shipped zip now lands in THREE homes before the commit:

1. `download/` (repo root) — the historical home, retention keeps 2;
2. `pharma-ml-toolkit/dist-archive/` — the SAFE COPY home, inside the
   toolkit tree so the ship commit's `-- pharma-ml-toolkit` pathspec
   carries it in the SAME commit as the code (repo-root `download/`
   sits outside that pathspec — the round-184 lesson).  Retention
   keeps 4 here;
3. `/home/<user>/pharma-ml-toolkit-zips/` — off-repo, immune to any git
   accident (amend, retention sweep, reset).

**The third rollback's lesson (2026-11-03 ~20:15): the physical
limit is REAL.**  A whole-filesystem snapshot rollback reverts
`/home/<user>` too — all three homes AND the off-repo copy were lost
together with the round-185 commits.  When that happens the round must
be re-implemented (the conversation context is the last surviving
carrier — verify contracts empirically before trusting any
reconstruction).  The three homes still pay off for the COMMON case
(git-level accidents, partial snapshots): whenever the snapshot
postdates the ship commit, BOTH code and zip survive together, and
`--restore`/`--doctor` consider all three homes (the off-repo copy
first).

## Rollback ledger

* **First (2026-11-02 ~22:00)** — reverted to round-179 (v1.2.103,
  HEAD `8085d76`); ate rounds 180–182 and every untracked zip.
  Survived: git history + tracked files.  Drove the original rules.
* **Second (2026-11-03 ~17:00)** — reverted to between round-183's
  retention commit (`249bff2`) and the round-184 ship; ate `e938407`
  (v1.2.105 docking campaign runner), `d03cdde`, `3f94306`,
  `engines/*.exe`, the v1.2.105 archive AND the tracked v1.2.104
  archive (the snapshot predated its commit).  Drove the three-home
  rule.
* **Third (2026-11-03 ~20:15)** — reverted to the round-183 snapshot
  `249bff2` minutes after the first v1.2.105 ship; ate the round-185
  commits, all three zip homes and the off-repo copy together.
  Round 185 was re-implemented from conversation context the same
  evening (the seeded attention demo reproduced its numbers
  byte-for-byte — the reconstruction-fidelity check).

## The standing guard

A cron job fires every 30 minutes (Job `495807`) and runs, minimally:

```
cd /home/<user>/my-project/pharma-ml-toolkit
pgrep -f checkpoint.py || python tools/checkpoint.py --light
```

`--light` = doctor + commit work-in-progress (a UUID-message commit
under the toolkit tree only).  The guard never develops features; it
only protects them.  CAUTION: a guard turn can land while a
development turn is mid-flight (model concurrency usually blocks it,
but not always) — after any surprise file mutation, re-verify the
tree's contracts before shipping.

## Doctor verdicts

* `ROLLBACK DETECTED` — the working tree's `VERSION` sits below the
  newest surviving archive.  Recovery:
  `python tools/checkpoint.py --restore` (unpacks the
  newest zip over the tree and light-commits), then `--doctor` again.
  If ALL homes are gone (the third-rollback case), re-implement from
  context — `--restore` has nothing to unpack.
* `MISSING packages: …` — the venv was reset too.  Install exactly:
  `pip install rdkit selfies shap torch meeko gemmi ruff mypy`
  (run it with `setsid nohup … &` — plain background jobs are reaped
  with the tool session).
* The venv ALSO loses the editable install on every rollback:
  `cd pharma-ml-toolkit && pip install -e .
  --no-deps --no-build-isolation` — otherwise notebook kernels and any
  cwd≠toolkit import fail with `No module named 'pharmaml'`.
* `PROBLEMS: version mismatch` — fix `VERSION` / `pyproject.toml` /
  `src/pharmaml/__init__.py` before shipping.

## Rules (each one earned the hard way)

1. **Commit the shipped zip.**  Tracked zips survived the rollback;
   untracked ones did not.  Retention: newest two stay tracked, older
   ones are `git rm`-ed (history keeps the blobs).
2. **Never run full pytest inside the ship ritual.**  The full suite
   needs 10–15 min on this 2-CPU box; a 590 s ritual timeout once
   killed a ship silently.  The ritual gates are doctests + ruff +
   mypy + GUI self-test only; run full pytest in chunks separately
   when a round touches broad surfaces.
3. **`PYTHONPATH` must be absolute** when the cwd will change
   (`the toolkit's absolute src/ path`) — pytest's
   `pythonpath=["src"]` ini option only holds in-process.
4. **Notebooks print the version at execution time.**  After a version
   bump, re-execute every example notebook or
   `test_notebook_variants` fails on stale outputs.
5. **Background processes die with the tool session.**  Use the
   double-fork daemon recipe (`scratch/xvfb_daemon_158.py`,
   `scratch/dev_daemon.py` under the repo root, the
   `run_full_suite_1NN.py` runners) or `setsid nohup … < /dev/null &`
   for pip/Xvfb/long jobs.
6. **Xvfb on `:99`** is started by `scratch/xvfb_daemon_158.py` when a
   GUI drive or the self-test needs a display.  A rollback reaps the
   daemon — restart it before any GUI test.
7. **Delete shallow clones before shipping** — repo clones under
   `scratch/` ride the zip walk into a 141 MB archive (round 185's
   first ship; deleted and re-shipped lean at 16.9 MB).
8. **Modal dialogs block headless drives** — the tests' `quiet_dialogs`
   fixture patches `showwarning`/`showerror`/`showinfo`; never call a
   card handler in a test without it.

## Ship ritual (end of every round)

```
python tools/checkpoint.py --commit "round NNN: <message>"
```

Gates (doctests → ruff → mypy → GUI self-test) then version re-pin,
zip build (outside `download/`, testzip'd, moved in), the three-home
safe-copy, retention sweep and the round commit.  Before it: bump
`VERSION`, write the CHANGELOG entry, run the round's targeted pytest
chunks.

## Verification status after the round-185 re-ship (2026-11-03, v1.2.105)

The XAI absorption (captum · tf-keras-vis · PAIR-code/saliency ·
bertviz, offline) re-implemented after the third rollback:
`TestXaiHelpers` (11 pure pins incl. the finite-difference gradient
check and the playground↔registry key sync), `TestXaiAbsorptionGui`
(7 headless card drives), the Deep Learning session round-trip pin
moved 18 → 24 inputs, ship gates doctests/ruff/mypy/GUI self-test,
zip testzip'd and safe-copied to all three homes.  One genuine fix
found during the re-implementation: the bare-`_MLPNetwork` path of
`numpy_layer_stack` silently defaulted the activation token to
`relu` — the `_act_np` callable probe now recovers it empirically.
