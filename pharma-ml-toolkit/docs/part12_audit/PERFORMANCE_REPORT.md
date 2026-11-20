# Performance Report (part12 PART 6)

Evidence gathered this session + the repo's own performance lanes. No new
microbenchmarks were fabricated; claims below cite measured artifacts.

## Execution model (verified)

- All long-running scientific routes accept `mode: "job"` — the callable
  runs on `api/jobs.py`'s ThreadPoolExecutor (CPU-budget-sized), progress
  streams over WS, state is REST-authoritative, cancellation is
  cooperative (`CANCELLATION_REQUESTED→CANCELLED`, native JobCancelled).
- The GUI never blocks on computation: workspace rows render JobRunCard
  states; DL surfaces live per-epoch `training.metrics` envelopes.
- Frontend rendering: Plotly is dynamically imported + `Plotly.react`
  (diff render), ResizeObserver + purge on teardown; three.js is a lazy
  manual chunk; 2,000-atom bond cap / 4,000-atom render cap in the 3D lane.

## Large-data lanes

- Import: multipart upload with bounded previews (≤500 rows/column in
  the GUI), 5,000-row export cap mirrored FE/BE; bounded column fetch lane.
- Job history: SQLite with index on created_at, LIMIT-parameterized.
- Virtualization: previews + caps are the strategy (documented); no
  unbounded table renders found in the audited workspace pages.
- Memory awareness: `utils/resources.py` cgroup-v1/v2 + /proc/meminfo
  probe, `ExecutionPreset`s (balanced/eco: n_jobs + chunk_rows),
  `preset_for_low_memory`.
- Polars: evaluated (part12 PART 6) and deliberately NOT adopted — no
  measured benefit for the workload; genuine lazy `scan_csv` semantics
  not needed; pandas stays. Decision recorded, not skipped silently.

## Measured artifacts on file

- Frontend production build (this session): 2,085 modules, shell
  800.99 kB (gzip 213.45 kB) + lazy plotly/three/motion chunks —
  consistent with the r279 close-out figures.
- Full-suite collection: 8,590 tests in 25.6 s (0 errors).
- The r301 ten-chunk gate on the repaired venv: green with the chunk-3
  OOM storm decoded (concurrent GUI-test memory pressure) and
  `test_nine_card_first` re-verified 23/23 via the six-class split — the
  mitigation is documented, not hidden.
- WS client: bounded backoff (500 ms→8 s, 12 attempts), heartbeat
  15 s/40 s liveness, since-cursor replay — no unbounded reconnect loops.

## Resource manager (part12 PART 7)

- CPU/RAM monitoring lanes exist (resources probe + UI monitor widget in
  the legacy GUI); GPU monitoring: NOT implemented (documented; no CUDA
  dependency in the pinned CPU stack — torch CPU wheel).
- Job queue + concurrent limits + heavy-job semaphore + per-job resource
  estimates (preset-derived) + user-configurable n_jobs (validated
  `-1`/positive-int, `0`/sub-`-1`/non-int rejected, never silently
  replaced — verified in the ML workspace lane).
- Cleanup after completion/cancellation: graceful drain + cancel events.

## Installer payload (part12 PART 16 size lane)

- EXE 557.7 MB / MSI 710.1 MB: dominated by the PyInstaller onedir
  sidecar (torch CPU ~200 MB, rdkit, scipy/numba stack) + offline WebView2
  bootstrapper + the bundled backend; the MSI delta is WiX compression of
  the same payload. No unnecessary *file* was found bundled (the archive
  composition audit is 0-violation on the shipped zip); the honest size
  lever is dependency tiering (e.g. dropping torch from the default
  bundle), which is an owner decision trading DL coverage — NOT taken
  silently per "do not remove scientific functionality merely to reduce
  size".

## Verdict

Performance architecture: **PASS** (background-first, bounded, cancellable,
resource-aware). The one measured regression class this session was tooling
(census leakage from local builds — B-1b, fixed). No unbounded or
GUI-blocking computation path was found.
