# RESUME — the anti-rollback protocol

_Last rebuilt 2026-11-02 after the rollback that ate rounds 180–182._

## What happened

The sandbox can silently revert the whole filesystem to an earlier
snapshot (observed 2026-11-02 ~22:00): the toolkit tree went back to
the round-179 state (v1.2.103, HEAD `8085d76`) while the conversation
carried on — three rounds of work, `tools/checkpoint.py`, this file
and every untracked zip (v1.2.98–v1.2.103) vanished. **What survived:
git history and git-TRACKED files.** That single observation drives
every rule below.

## The standing guard

A cron job fires every 30 minutes (Job `495807`) and runs, minimally:

```
cd /home/z/my-project/pharma-ml-toolkit
pgrep -f checkpoint.py || /home/z/.venv/bin/python tools/checkpoint.py --light
```

`--light` = doctor + commit work-in-progress (a UUID-message commit
under the toolkit tree only).  The guard never develops features; it
only protects them.

## Doctor verdicts

* `ROLLBACK DETECTED` — the working tree's `VERSION` sits below the
  newest surviving archive.  Recovery:
  `/home/z/.venv/bin/python tools/checkpoint.py --restore` (unpacks the
  newest zip over the tree and light-commits), then `--doctor` again.
* `MISSING packages: …` — the venv was reset too.  Install exactly:
  `/home/z/.venv/bin/pip install rdkit selfies shap torch meeko gemmi ruff mypy`
  (run it with `setsid nohup … &` — plain background jobs are reaped
  with the tool session).
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
   (`/home/z/my-project/pharma-ml-toolkit/src`) — pytest's
   `pythonpath=["src"]` ini option only holds in-process.
4. **Notebooks print the version at execution time.**  After a version
   bump, re-execute every example notebook or
   `test_notebook_variants` fails on stale outputs.
5. **Background processes die with the tool session.**  Use the
   double-fork daemon recipe (`scratch/xvfb_daemon_158.py`) or
   `setsid nohup … < /dev/null &` for pip/Xvfb/long jobs.
6. **Xvfb on `:99`** is started by `scratch/xvfb_daemon_158.py` when a
   GUI drive or the self-test needs a display.

## Ship ritual (end of every round)

```
/home/z/.venv/bin/python tools/checkpoint.py --commit "round NNN: <message>"
```

Gates (doctests → ruff → mypy → GUI self-test) then version re-pin,
zip build (outside `download/`, testzip'd, moved in), retention sweep
and the round commit.  Before it: bump `VERSION`, write the CHANGELOG
entry, run the round's targeted pytest chunks.

## Verification status after the 23:30 guard (2026-11-02, round 183 / 100x-169)

The four-bug round (v1.2.104) is committed as `c200bd6` (light commit).
Test slices verified green this turn: `test_tuner.py` (exit 0),
`TestTunerCard` (incl. the 3 new card tests), `test_explain.py`
(SHAP +4), `test_rism.py` + `test_features.py` (tqdm wiring).

**CLOSED (the "Continue" turn, same night):**

1. `test_cli_features.py` — exit 0 in its own 590 s window.
2. `test_playground.py` — fully verified in per-class chunks: the
   first ~62% green in the OOM-attempt run, then chunks
   `TestReview130Gui…136Gui`, `137…141`, `148…151`+
   `TestPredictionIntervalsGui`, `TestOverfitGap/Interpretability/
   CrossModelImportance/VarianceFilter/UnsupervisedTab`, and
   `TestDearPyGui*` + `TestShapOptuna/Reinvent4DiffDockAbsorption` —
   all exit 0.  The whole-file OOM at ~62% is a box RAM limit, not a
   code failure; chunking is the standing workaround.
3. `encode_class_target` now has its pytest pair in
   `TestPureHelpers` (multiclass/binary/list + numeric
   pass-through), green.

Standing environment notes (kept):

4. GUI tests need `DISPLAY=:99` (Xvfb up) and
   `PYTHONPATH=<abs>/src`; without DISPLAY the suite errors en masse
   (not a code failure).
5. Ship ritual will re-pin pyproject/`__init__` to 1.2.104 (they lag
   by design pre-ship) and build the v1.2.104 zip (none since v1.2.90).
6. Bash-tool output can silently EAT bracket-letter pairs (``[m`` →
   nothing): a line displayed as ``np.asarray(apping[`` was actually
   ``np.asarray([mapping[``.  When Bash output shows impossible code,
   cross-check with the Read tool or an ast/tokenizer dump before
   "fixing" anything.

