# PERFORMANCE BASELINES — part14 §13 deliverable #5

The part14 §10 benchmark matrix instantiated HONESTLY: measured legs
carry numbers + hardware context; unmeasured legs carry
NOT-MEASURED, never a placeholder number (§16.15 forbids fabricated
benchmarks). One benchmark result is never a universal performance
guarantee.

Seeded: round 329 (2026-11-22).

## Measured (this environment, named runs)

| Metric | Value | Context | Evidence |
| --- | --- | --- | --- |
| Web console landing render | HTTP 200 (r331, the rebuilt console) | Linux sandbox, Next.js dev server, Chromium/agent-browser; 1440px | `p14_r331_console_restore.txt` + `p14_r331_browser_e2e.txt` |
| zip-size API latency (release path) | sub-second; answer `23,489,390 / v1.2.240 / exact:true / source:release` (the r329 re-ship terminal bytes; re-verified r330 ×3 and r331) | same; fstat-only path, zero spawns | `p14_r331_console_restore.txt` |
| API regression suite wall time | 108.88s (449 tests + 1 skip) on the r331 rebuilt venv | sandbox venv; includes model-training fixtures | `p14_r331_api_suite_run2.txt` |
| Trusted-origin matrix wall time | 5.83s (34 tests) | same | `p14_r331_trusted_origins_run.txt` |
| Census collection wall time | 23.45s (8,589 = 7,670 tests/ + 919 doctests, 0 errors) | same | `p14_r331_collection_census.txt` |

## NOT-MEASURED (the §10 desktop matrix — rides the Windows box)

Startup time; backend startup; time-to-first-usable-UI; REST latency
(Windows loopback); WS latency; dataset import/filtering at
small/medium/large; ML training time (representative workloads);
model loading; Plot rendering; 3D scene load + frame rate; SQLite
query time; memory consumption; CPU; GPU utilization; shutdown time.

- Required environment: the physical Windows box (§9.5; VM numbers
  flagged as VM). Representative small/medium/large workloads per
  workspace; hardware + software configuration recorded per run.
- No numbers exist in this document for these rows — by rule.

## Investigation legs (§10 tail)

Memory leaks / unbounded caches / CPU saturation during long runs:
**BLOCKED** on the 45–60-min stability lane (the box). The JobManager
resource-snapshot API (`GET /api/v1/system/resources`, the r324
deliverable — read-only host/process/queue snapshot, GPU honestly
absent) is the in-app instrument those legs will read on the box.

— seeded round 329; the measured rows re-pinned round 331 (the
r329 rows' evidence stands in `scratch/p14_r329_*`); numbers only
ever land with named evidence.
