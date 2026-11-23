# RELEASE ACCEPTANCE REPORT — part14 §13 deliverable #8

The part14 §14 acceptance gates, honestly stated at round 329
(2026-11-22); the r331 addendum re-pins G/D on this round's
evidence; the r333 addendum records the owner-directed release
(build + publish legs executed); the r336 addendum records the
owner's CI directive — BLK-1's installer-mechanics half now executes
per release via `windows-install-smoke.yml` (the first dispatch
GREEN END-TO-END at run 45086526069, the r336 addendum). Supersedes the audit's §H snapshot
(r326): Gate B has moved since (the incident fixed at source +
regression-pinned; the packaged-app legs still ride the box).

## Gate-by-gate

| Gate | Status | Basis |
| --- | --- | --- |
| A — Repository integrity | **PASS (this round)** | HEAD verified; the r328 payload + r329 hardening pushed to origin/main (`ad5f00a7..a75f98eb..013b3ff4`); the token-sweep ghost reset away (reflog-resident, never pushed — push protection proved it); tree understood (8 intended modifications + documented untracked residue); version carriers consistent at 1.2.240; the QA deliverable set now reconciled with the implementation |
| B — Backend and security | **PARTIAL (upgraded from FAIL)** | "Tauri origins handled correctly" — TRUE at source + regression level now (D-1/D-2 fixed, 34/34 pinned, r329 re-pin; D-3 honesty fixed, 9 tests; D-4/D-5 source-fixed, compile-blocked). Authentication/loopback-bind verified. The PACKAGED-app legs: the process-lifecycle half (launch → the sidecar child spawn → the graceful close → the no-orphan check) + the INSTALLED sidecar's live contract VERIFIED GREEN in CI (run 45086526069, the r336 install-smoke lane — all 16 steps: both installers' launches, zero orphans after the graceful closes, the handshake/health/guard/shutdown contract on the installed bundle); the real WebView2 VISUAL round-trip still rides the box — the gate does not close without it |
| C — Frontend and scientific | **PARTIAL** | vitest 119/119 + tsc clean re-pinned; web console E2E green (render/title/interaction/footer, zero page/console errors, pins pair r329); the nine-workspace desktop legs ride the box |
| D — Automated tests | **PASS (the sandbox scope)** | tests/api 449+1s 0 failed (r331 re-pin: 108.88s, `p14_r331_api_suite_run2.txt`); census 8,589/0 errors (r331, `--doctest-modules` re-trued); trusted-origins 34/34; ruff clean + mypy clean (r331, after the mode-wall restoration); vitest 119/119 + tsc clean; CI 19 jobs live at the pushed HEAD (verdict closed GREEN at r331: a6cc830b 19/19); the Rust unit legs stay compile-blocked (§16.14) |
| E — Performance and stability | **BLOCKED** | unchanged — the 45–60-min + resource lanes need the physical box; no numbers fabricated (PERFORMANCE_BASELINES.md). The r336 CI lane records a bounded RUNNER-SCOPE resource sample per release — runner numbers, honestly never the desktop baseline (the baseline stays box-pinned) |
| F — Windows release | **PARTIAL (upgraded from BLOCKED, r333; the CI-lane addendum r336)** | the build + publish legs EXECUTED: the v1.2.240 installer set built by the CI pipeline (run 45047077848, 53/53 steps) and attached to the public release (EXE 558,022,042 B + MSI 710,553,078 B + manifest + SHA256SUMS + SBOM, the manifest's commit byte-verified as the r333 terminal) over the owner's explicit release directive; the standing lanes remain: the clean-machine smoke's DESKTOP half (the GUI click-through, the offline WebView2 bootstrap, desktop-stability fidelity — BLK-1's installer-mechanics half is CI-EXECUTED and GREEN since r336 via `windows-install-smoke.yml`, run 45086526069 all 16 steps: both silent installs + the installed-tree verification + the INSTALLED sidecar's live contract + the lifecycle + both uninstalls + the v1.2.231→v1.2.240 cross-scope upgrade with the retention marker verified; the lane also surfaced the real installer findings — the data-dir/installdir collision on the NSIS lane + the per-user/per-machine scope split across the two MSI generations), Stage-5 PySide6 retirement (BLK-4), installer signing (BLK-3 — the v1.2.240 installers ship UNSIGNED, the v1.2.231 precedent) |
| G — Release integrity | **PARTIAL (basis re-pinned r331)** | release homes at the CURRENT terminal bytes (v1.2.240 / 23,489,390 B / sha-256 `0fadb6e5…`, the r329 same-version re-ship, six-home parity + offsite exact) + live exact:true zip-size API re-verified r330 (×3) and r331; the CI verdict queue CLOSED — a6cc830b 19/19 check-runs success (checked r331), the r329 pair closed GREEN at r330; SBOM/pins standing from r324–r325 |

## The honest verdict

No production-ready declaration: gate E's desktop-stability lanes
remain BLOCKED on owner-side hardware; F's remaining desktop legs
(the GUI click-through, the offline WebView2 bootstrap) + B's visual
WebView2 round-trip + G's box legs stay box-pinned; the Rust suites
compile-blocked. The r336 CI lane executes BLK-1's
installer-mechanics half per release — GREEN on the published
v1.2.240 assets (run 45086526069), runner-scope evidence, honestly
never a desktop pass. The standing convention holds — no silent
version bump; the shipped fixed point
stays v1.2.240 (23,489,390 B, the r329 same-version re-ship) with
the published installer set standing at tag v1.2.240.

## What changed this round (the delta log)

- The credential lane restored → **the first successful pushes of
  the part14 era** (the r328 payload + the r329 hardening).
- The QA deliverable set seeded (§13's ten documents: this set +
  PART14_PHASE1_AUDIT + DEFECT_REGISTER + the consolidated execution
  report embedded in the worklog round entry).
- Every sandbox-executable gate re-pinned at the arrival; the census
  re-proven at r328 byte-parity.

## Round 331 addendum — the queued items closed

- The CI verdict queue **closed**: the r330 terminal `a6cc830b` —
  **19/19 check-runs success** (checked this round via the
  authenticated API); the r329 pair had closed GREEN at r330.
- The **twenty-first arrival (deepest-class rollback: history reset
  to the r186 checkpoint, origin detached, PAT + venv gutted, the
  Studio frontend wiped, system `libEGL.so.1` gone)** recovered in
  full before any gate work: origin re-attached, reset onto
  `a6cc830b`, the ghost pair + the old-layout `.github` duplicate +
  the stale counter copy swept, the venv rebuilt pin-exact (the
  numpy-1.x wheel class documented), the mode wall (101 files'
  rollback +x bits) restored — ruff clean again.
- Every gate re-pinned at r329/r330 parity on the rebuilt stack
  (trusted-origins 34/34, tests/api 449+1s, census 8,589/0, vitest
  119/119, tsc clean, ruff/mypy clean, the console + zip-size API
  exact at 23,489,390 B, the browser E2E zero errors + footer
  geometry exact + the r331 pins pair).

— recorded round 329; addendum round 331; nothing above converts
absence of evidence into a pass (§16.14/§16.15).
