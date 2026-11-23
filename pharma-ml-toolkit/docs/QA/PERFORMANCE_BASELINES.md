# PERFORMANCE BASELINES — part15 §13 deliverable (refreshed round 342)

The part15 §12 measurement matrix instantiated HONESTLY: measured
lanes carry numbers + hardware context; unmeasured lanes carry
NOT-MEASURED, never a placeholder number (§16.15 forbids fabricated
benchmarks). One benchmark result is never a universal performance
guarantee.

Seeded: round 329 (part14); re-pinned r331; **refreshed round 342
(part15 roadmap #4) — the release-era rows re-measured at the v1.2.240
byte-frozen tree + the new backend startup/latency/memory/shutdown
lanes**. Wall-clock rows on this box are load-dependent — the r331
pins are kept for drift comparison, both honestly labeled.

## Measured (this environment, named runs)

| Metric | r342 value (2026-11-23) | r331 pin | Context | Evidence |
| --- | --- | --- | --- | --- |
| Web console landing render | HTTP 200 | HTTP 200 | Linux sandbox, Next.js dev server; 1440px | `scratch/p15_r342_zip_size.txt` window + the r342 browser pass |
| zip-size API latency | 0.308 s total; answer `23,489,390 / v1.2.240 / exact:true / source:release` | sub-second, same answer | same; fstat-only path, zero spawns | `scratch/p15_r342_zip_size.txt` |
| **Backend startup (NEW)** | **4.138 s spawn → PHARMAML-READY** (port-0 bind, machine-readable handshake) | — | sandbox venv (py 3.12.16, pharmaml 1.2.240, 2 cpus), fresh `PHARMAML_STUDIO_DATA` | `scratch/p15_r342_backend_perf.txt` |
| **Health latency (NEW)** | **first 82.5 ms** (RDKit scientific-core probe), **warm 0.9 ms** (min/median, 3 probes) | — | same, `GET /api/v1/health` | `scratch/p15_r342_backend_perf.txt` |
| **Idle backend memory (NEW)** | **RSS 228.4 MiB, +0.00 MiB drift over 10 s idle, 6 threads** | — | same, the r324 read-only instrument `GET /api/v1/system/resources` (psutil source), jobs_active=0 | `scratch/p15_r342_backend_perf.txt` |
| **Graceful shutdown (NEW)** | **0.665 s SIGTERM → exit 0** (JobManager drain honored) | — | same | `scratch/p15_r342_backend_perf.txt` |
| Trusted-origin matrix wall time | 10.955 s (34/34, exit 0) | 5.83 s (34) | sandbox venv; same suite — box load differs | `scratch/p15_r342_trusted_origins.txt` |
| API regression suite wall time | 117.495 s (461 passed + 1 skip = 462 collected, exit 0) | 108.88 s (449 + 1 skip) | sandbox venv; the tree grew +13 tests since r331 (the r339/r341 WS lanes) | `scratch/p15_r342_api_suite.txt` |
| Census collection wall time | 33.236 s (8,602 items, 0 errors, 472 files) | 23.45 s (8,589) | sandbox venv | `scratch/p15_r342_census.txt` |

The r342 backend lanes rode the instrumented entry
`python -m pharmaml.api` (the part10 §10.1 structured startup
handshake — port-0 bind, one machine-readable stdout line), the
in-app r324 resource-snapshot API, and a faithful SIGTERM; the
driver is `scratch/p15_r342_backend_perf.py` (re-runnable).

## NOT-MEASURED (the §12 desktop matrix — rides the Windows box)

Frontend startup time; time-to-first-usable-UI; REST latency
(Windows loopback); WS latency; dataset import/filtering at
small/medium/large; ML training time (representative workloads);
model loading; Plot rendering; 3D scene load + frame rate; SQLite
query time; CPU/GPU utilization under load; the long-run stability
lane. The large-dataset/import-filtering rows are desktop-app
workflows (§9.5) — the sandbox venv cannot host them honestly, so
they stay NOT-MEASURED here rather than borrowing a stand-in.

- Required environment: the physical Windows box. Representative
  small/medium/large workloads per workspace; hardware + software
  configuration recorded per run.
- No numbers exist in this document for these rows — by rule.

## Investigation legs (§12 tail)

Memory leaks / unbounded caches / CPU saturation during long runs:
**BLOCKED** on the 45–60-min stability lane (the box). The JobManager
resource-snapshot API (`GET /api/v1/system/resources`, the r324
deliverable — read-only host/process/queue snapshot, GPU honestly
absent) is the in-app instrument those legs will read on the box;
the r342 idle lane above is its first measured use in the baselines.

— refreshed round 342 (the 134th recorded run); numbers only ever
land with named evidence.
