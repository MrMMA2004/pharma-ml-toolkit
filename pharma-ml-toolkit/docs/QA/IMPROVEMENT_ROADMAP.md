# IMPROVEMENT ROADMAP — part15 §13 deliverable #9

The prioritized backlog over `0132d409` (round r340), ordered per the
part15 §16-C remediation priority (security → startup/communication →
data integrity → API/WS correctness → frontend runtime → scientific
correctness → performance → UX/a11y → docs). Each item names its
evidence gap — nothing here is speculative (§10 rule: no improvement
without an audit-named problem).

## Now executable in this sandbox (the next campaign rounds)

| # | Item | Rationale (audit-named) | Shape |
| --- | --- | --- | --- |
| 1 | The remaining part15 QA surface upkeep — keep the SS13 doc set trued each round (test counts, CI verdicts, defect rows) | the docs are now the current-state layer; stale numbers are the exact failure mode §3.3 audits | small per-round doc deltas |
| 2 | The GUI-lane regression surface for the WS origin guard on the frontend side (the websocket client's reconnect-with-origin behavior) | D-6 pinned the backend; the frontend client suite covers auth but not yet a hostile-origin reconnect scenario | a vitest lane over `services/websocket/client.ts` |
| 3 | The root console's dev-script portability note → a PowerShell-safe dev script set (optional; console is sandbox-only today) | §6's own review named the Bun/pipe reliance | small `package.json` script variants, documented as dev-only |
| 4 | Performance baselines refresh (backend startup, memory idle, large-dataset lanes) at the v1.2.240 tree | PERFORMANCE_BASELINES.md is release-era; §10 asks for current measurements | re-measured baselines + doc update |

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

— trued round 340 (the 132nd recorded run) at `0132d409`.
