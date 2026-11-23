# IMPROVEMENT ROADMAP — part15 §13 deliverable #9

The prioritized backlog over the r342 terminal, ordered per the
part15 §16-C remediation priority (security → startup/communication →
data integrity → API/WS correctness → frontend runtime → scientific
correctness → performance → UX/a11y → docs). Each item names its
evidence gap — nothing here is speculative (§10 rule: no improvement
without an audit-named problem).

## Now executable in this sandbox (the next campaign rounds)

| # | Item | Rationale (audit-named) | Shape |
| --- | --- | --- | --- |
| 1 | The remaining part15 QA surface upkeep — keep the SS13 doc set trued each round (test counts, CI verdicts, defect rows) | the docs are now the current-state layer; stale numbers are the exact failure mode §3.3 audits | small per-round doc deltas |
| 2 | ~~The GUI-lane regression surface for the WS origin guard on the frontend side~~ **DELIVERED r341** — the D-6 client lane landed: 6 fail-first vitest legs over `services/websocket/client.ts` (the 4403/4401 policy closes are terminal with a surfaced `terminalReason`, never a reconnect storm; the auth-lane token/cursor retry pin) + the banner surfacing test + the Python drift guard `test_ws_client_treats_policy_close_codes_as_terminal`; the D-8 register row closed with executed evidence | D-6 pinned the backend; the frontend lane was the named gap | a vitest lane over `services/websocket/client.ts` |
| 3 | ~~The root console's dev-script portability note~~ **DELIVERED r342** — the PowerShell-safe dev script set landed: `dev:portable` (`next dev -p 3000`, the POSIX `2>&1 \| tee` pipe dropped), `build:portable` (`next build && node scripts/copy-standalone.cjs` — pure-Node `fs.cpSync` replaces `cp -r`), `start:portable` (`node scripts/start-standalone.cjs` — NODE_ENV set in-process, faithful exit-code passthrough, replaces the `NODE_ENV=production bun …` env-prefix syntax); the POSIX defaults untouched (the sandbox's own dev contract rides them); executed verification over fake targets: the copy leg (happy + the missing-source exit-1 path), the launcher legs (NODE_ENV pin + the 7/0 exit passthrough + the missing-target surfacing); documented dev-only | §6's own review named the Bun/pipe reliance | small `package.json` script variants, documented as dev-only |
| 4 | ~~Performance baselines refresh~~ **DELIVERED r342** — PERFORMANCE_BASELINES.md refreshed at the v1.2.240 tree: the release-era rows re-measured fresh (console 200, zip-size 0.308 s exact:true release, trusted-origin 34/34 in 10.955 s, API suite 461+1 skip of 462 in 117.495 s, census 8,602/0/472 in 33.236 s — the r331 pins kept for drift comparison, wall-clock honestly load-labeled) + the NEW backend lanes (startup 4.138 s spawn→PHARMAML-READY via the structured handshake, health first 82.5 ms / warm 0.9 ms, idle RSS 228.4 MiB zero-drift over 10 s via the r324 read-only instrument, graceful shutdown 0.665 s exit 0); the driver `scratch/p15_r342_backend_perf.py` re-runnable; the large-dataset/import rows stay honestly NOT-MEASURED (desktop-app lanes, §9.5) | PERFORMANCE_BASELINES.md is release-era; §10 asks for current measurements | re-measured baselines + doc update |

## Riding the owner's Windows box (evidence-gated)

| # | Item | Rationale | Carrier |
| --- | --- | --- | --- |
| 5 | D-4/D-5 conversion (cargo fmt/clippy/test) | the register's compile-blocked rows | `test-rust.ps1` |
| 6 | The SS11 execution pass (all nine scripts, transcripts) | the honest BLOCKED legs | `WINDOWS_VALIDATION_GUIDE.md` |
| 7 | The interactive packaged-app checklist rows (§A/§D/§E/§G) | the WebView2 legs no runner can fully cover | `MANUAL_TEST_CHECKLIST.md` part15 layer |
| 8 | Installer signing | B-4 | certificate + `windows-release.yml` |
| 9 | The new-version ship lane (v1.2.241+) | opens when B-1..B-5 clear | the standing release ritual |

## Deliberately NOT on the roadmap (the constraint rails)

- No framework migration (no C#/WinUI, no Tauri replacement — §2).
- No PySide6 retirement before its gates (B-5).
- No speculative caching of scientific results (§10: caching only with
  explicit, tested invalidation rules).
- No new external workflow engine (the r322 in-tree decision stands).
- No version bumps without owner approval (§5; v1.2.240 byte-frozen).

— trued round 342 (the 134th recorded run); the sandbox backlog is now empty (#1 upkeep rides every round) — the remaining lanes are the Windows-box evidence-gated set below.
