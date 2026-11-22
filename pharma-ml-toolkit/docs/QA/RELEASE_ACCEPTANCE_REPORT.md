# RELEASE ACCEPTANCE REPORT — part14 §13 deliverable #8

The part14 §14 acceptance gates, honestly stated at round 329
(2026-11-22). Supersedes the audit's §H snapshot (r326): Gate B has
moved since (the incident fixed at source + regression-pinned; the
packaged-app legs still ride the box).

## Gate-by-gate

| Gate | Status | Basis |
| --- | --- | --- |
| A — Repository integrity | **PASS (this round)** | HEAD verified; the r328 payload + r329 hardening pushed to origin/main (`ad5f00a7..a75f98eb..013b3ff4`); the token-sweep ghost reset away (reflog-resident, never pushed — push protection proved it); tree understood (8 intended modifications + documented untracked residue); version carriers consistent at 1.2.240; the QA deliverable set now reconciled with the implementation |
| B — Backend and security | **PARTIAL (upgraded from FAIL)** | "Tauri origins handled correctly" — TRUE at source + regression level now (D-1/D-2 fixed, 34/34 pinned, r329 re-pin; D-3 honesty fixed, 9 tests; D-4/D-5 source-fixed, compile-blocked). Authentication/loopback-bind verified. The PACKAGED-app legs (real WebView2 round-trip, process lifecycle) remain BLOCKED on the box — the gate does not close without them |
| C — Frontend and scientific | **PARTIAL** | vitest 119/119 + tsc clean re-pinned; web console E2E green (render/title/interaction/footer, zero page/console errors, pins pair r329); the nine-workspace desktop legs ride the box |
| D — Automated tests | **PASS (the sandbox scope)** | tests/api 449+1s 0 failed; census 8,589/471/0 (r328 byte-parity); ruff/mypy clean; CI 19 jobs live at the pushed HEAD (async verdict checked at round close); the Rust unit legs stay compile-blocked (§16.14) |
| E — Performance and stability | **BLOCKED** | unchanged — the 45–60-min + resource lanes need the physical box; no numbers fabricated (PERFORMANCE_BASELINES.md) |
| F — Windows release | **BLOCKED** | the three standing lanes (clean-machine smoke, Stage-5 PySide6 retirement, installer signing) |
| G — Release integrity | **PARTIAL** | release homes at exact ledger bytes (v1.2.240 / 22,449,138 B, six-home parity r325) + live exact:true zip-size API re-verified r329; CI verdict at HEAD pending close; SBOM/pins standing from r324–r325 |

## The honest verdict

No production-ready declaration: gates E/F remain BLOCKED on
owner-side environments; B/G close only with the box legs; the Rust
suites compile-blocked. The release lanes ride each continuation
until the owner-side gates clear (the standing convention — no
premature publish, no silent version bump; the shipped fixed point
stays v1.2.231 with its assets intact).

## What changed this round (the delta log)

- The credential lane restored → **the first successful pushes of
  the part14 era** (the r328 payload + the r329 hardening).
- The QA deliverable set seeded (§13's ten documents: this set +
  PART14_PHASE1_AUDIT + DEFECT_REGISTER + the consolidated execution
  report embedded in the worklog round entry).
- Every sandbox-executable gate re-pinned at the arrival; the census
  re-proven at r328 byte-parity.

— recorded round 329; nothing above converts absence of evidence
into a pass (§16.14/§16.15).
