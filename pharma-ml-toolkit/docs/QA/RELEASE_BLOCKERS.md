# RELEASE BLOCKERS — part15 §13 deliverable #8

The honest blocked-release register over `0132d409` (round r340).
Nothing here is concealed (§19 rule); the published v1.2.240 release
stands as the byte-frozen, smoke-passed release (the r334 ruling), and
these are the gates a FUTURE version (v1.2.241+) must clear before its
ship. The version never moves silently (§5).

| ID | Gate | Status | What clears it |
| --- | --- | --- | --- |
| B-1 | Rust toolchain legs — `cargo fmt --check` / `clippy` / `cargo test` (the D-4/D-5 register rows) | **BLOCKED** (sandbox has no Rust toolchain; source + 11 unit cases delivered) | `scripts/windows/test-rust.ps1` green on the owner's box → both rows convert to plain FIXED |
| B-2 | The nine SS11 PowerShell scripts (execution) | **BLOCKED** (no PowerShell in this sandbox; source-delivered, PS 5.1-compatible) | the owner-side runs with transcripts (`WINDOWS_VALIDATION_GUIDE.md`) |
| B-3 | Clean-machine Windows acceptance gates E/F on real Win10/11 desktops (the interactive packaged-app legs: §A integration, §D/E interactive charts/3D, §G performance profiles, third-party AV) | **PARTIALLY VERIFIED** — the automated smoke lane is green end-to-end on clean runners (ninth dispatch `45086526069`, all 16 steps); the owner-desktop legs remain **NOT TESTED** | the owner's manual pass over `MANUAL_TEST_CHECKLIST.md` part15 §A–H rows |
| B-4 | Installer digital signing | **BLOCKED** (certificate-gated; the installers ship unsigned with SHA256SUMS digest verification) | an owner-provided code-signing certificate wired into `windows-release.yml` |
| B-5 | PySide6 Stage-5 retirement (the legacy Studio surface) | **BLOCKED** by design (§2 rule: do not retire until migration-parity + clean-Windows acceptance genuinely pass) | the B-3 owner pass + the parity checklist (PROJECT_STATUS Stage-5 lane) |
| B-6 | Disk-full / low-disk behavior lane | **BLOCKED** (needs a controlled low-disk environment; never simulated on the live box) | a dedicated low-disk test pass (the §4.5 environmental leg) |
| B-7 | ECC install-state on the OWNER's Windows box | standing directive (sandbox re-installed + check GREEN each round since r335; the Windows-side install is the owner's environmental choice per part15 §9) | optional: the owner may install ECC locally; the repo never depends on it |

The standing release verdict (unchanged this round): **v1.2.240 is the
published, smoke-passed release** — five assets intact (EXE
558,022,042 B / MSI 710,553,078 B / SHA256SUMS / RELEASE_MANIFEST /
sbom.cdx.json), byte-frozen (no re-ship without owner approval), CI
19/19 at HEAD, six-home zip parity `0fadb6e5…a63e2cb3`. The new-version
lane opens the day B-1..B-5 clear.

— trued round 340 (the 132nd recorded run) at `0132d409`.
