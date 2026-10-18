# 100x-49-A — fix-subagent (viz + ml carried contract gaps)

Full entry: see /home/z/my-project/worklog.md (Task ID: 100x-49-A, appended at the end).

## One-paragraph summary
Closed the carried 100x-41-B/41-C contract gaps after re-deriving every repro from scratch (the /tmp findings were lost to the sandbox reset): unified top_n contract (`int | None = 15`; None=all rows; <=0/non-int -> DataValidationError) on plot_top_ligands + plot_feature_importance; explicit-missing energy_column -> typed error naming it (None keeps the fallback + message); six raw-error corners -> typed (labels arity, string similarity frame, bins<=0 x2, non-numeric leaderboard cells, junk receptor dict, classes_-less stub x2); duplicated-column guards in plot_descriptor_distributions + plot_model_comparison ("duplicate column names: [...]"); train_model all-NaN cv_scores -> warning naming fold scores (still returns); docstring-only contracts for y NaN/±inf through training, r2 single-sample folds, and mordred BCUT2D NaN preservation; evaluation ±inf verified already fixed (100x-42-B).

## Key facts for successors
- Evidence lives in /tmp/task49a/ (probe_viz.py 37 cases, probe_ml.py 9 cases, PRE/POST/FINAL runs x2 byte-identical, junitxml baselines + finals, newtests_PREFIX_*.xml = 13+4 failed/passed pre-fix).
- Counts: test_viz.py 99 -> 116, test_ml_pipeline.py 90 -> 91 items, test_descriptors.py 48 -> 48; ruff + mypy clean; doctest subsets green x2.
- Second unguarded `bins` is in docking_plots.plot_energy_distribution (NOT ml_plots — its n_bins was already typed; grep-verified).
- mordred on this numpy>=2 venv returns Error objects, NOT NaN, for many descriptors (BCUT2D NaN census = 0) — that's why the BCUT2D note is docstring-only, no doctest.
- ENV lessons: (a) sandbox reset flips +x on shebang-less py files -> ruff EXE002; chmod -x restores (git core.fileMode=false so it's invisible); (b) repo root is the OUTER /home/z/my-project — `git show HEAD:src/...` fails AFTER `>` truncates the target; use `HEAD:./src/...`.
- Carried: partial-NaN cv folds still poison cv_mean silently (only all-NaN warns — binding scope); ±inf through make_train_test_split documented, not guarded (decision rule); R4-c mordred Error-object (full-deps CI).
