# WINDOWS VALIDATION GUIDE — part15 §13 deliverable #7

How the owner's Windows 10/11 box executes the legs this Linux sandbox
cannot. Written round r340 (the 132nd recorded run) over `0132d409`;
the script set itself was delivered round r339 (SS11, the part15 §13
minimum list, PowerShell 5.1-compatible).

## 1. The script set — `pharma-ml-toolkit/scripts/windows/`

All nine scripts are source-delivered, PS 5.1-compatible (no admin
required unless a row says so), print tool versions, use the correct
working directory (they resolve the repo root themselves), return
meaningful exit codes, and preserve transcripts under
`pharma-ml-toolkit/logs/windows-tests/`. No script ever prints a token
or a secret; none performs destructive actions.

| Script | What it proves | Notes |
| --- | --- | --- |
| `doctor.ps1` | prerequisite census (python, node/npm, rust/cargo, tauri CLI, git) + per-lane runnable verdicts | run this FIRST |
| `test-backend.ps1` | pytest collection + focused/full runs, ruff, mypy | the Python suite lanes |
| `test-frontend.ps1` | `npm ci` + tsc + vitest + `vite build` in the NESTED frontend (`pharma-ml-toolkit/frontend/`) — never the root console | §7's correct-directory rule |
| `test-rust.ps1` | `cargo fmt --check` + `cargo clippy` + `cargo test` in `src-tauri/` — **the D-4/D-5 carrier**: a green run converts both register rows to plain FIXED | needs the Rust toolchain |
| `test-tauri.ps1` | tauri conf parse + capabilities/permissions audit + `cargo check` | the capability review leg of §12 |
| `test-api-security.ps1` | the three security suites + optional 5.1-compatible LIVE probes: hostile-origin 403, no-token 401, trusted+token 200 | probes are opt-in (backend must be running) |
| `test-persistence.ps1` | the isolated-temp-DB persistence suites | never touches user data |
| `test-installers.ps1` | published-digest verification of downloaded installers against SHA256SUMS + RELEASE_MANIFEST cross-check — verifies, NEVER installs | no admin needed |
| `collect-diagnostics.ps1` | consolidated PASS/FAIL/BLOCKED report over all lanes | the §13 "consolidated diagnostic report" |
| `validate-clean-machine.ps1` | **the part19 Phase 8 clean-machine suite** (r368): system census (Windows/CPU/RAM/GPU/WebView2/Python-Node-Rust presence), installer SHA-256 + Authenticode status (BLOCKED honestly while signing is certificate-gated), the live sidecar handshake/health/guard/shutdown leg, the nine-workspace API execution matrix over synthetic data, artifact export + backup/restore, and the upgrade/uninstall leg (requires the prior installer present) | run on a GENUINELY clean box; JSON + Markdown reports land in `logs/windows-tests/clean-machine/`; BLOCKED legs are honest, never silent |

## 2. The recommended order

1. `doctor.ps1` — the census; it names any missing prerequisite and the
   lanes that stay BLOCKED because of it.
2. `test-backend.ps1` → `test-frontend.ps1` → `test-rust.ps1` (D-4/D-5)
   → `test-tauri.ps1`.
3. `test-api-security.ps1` (start the backend first if using the live
   probes; the scripts' transcripts record what actually ran).
4. `test-persistence.ps1`, `test-installers.ps1`.
5. `collect-diagnostics.ps1` — the consolidated verdict; paste its output
   into the round directive when reporting back (it distinguishes
   PASS/FAIL/BLOCKED — the §13 rule).

## 3. The installer smoke lane (already automated on GitHub)

The `windows-install-smoke.yml` workflow auto-runs on every PUBLISHED
release and is dispatchable at will: it installs both installers on a
clean windows-2022 runner, verifies the digests, the installed tree,
the version, the INSTALLED SIDECAR's live contract (handshake +
authenticated health + the refused guard + graceful shutdown), the app
host launch with the sidecar child, resource samples, graceful closes
with zero orphans, uninstalls, and the cross-scope upgrade — all 16
steps. The ninth dispatch (`45086526069`) was green end-to-end against
the published v1.2.240 assets. The owner's local box adds what the
runner cannot: real Win10/11 desktops, third-party AV panels, physical
GPU/driver profiles, and the interactive packaged-app legs of the
manual checklist (§A/§D-interactive/§E/§G rows).

## 4. Honest status

None of the nine scripts was EXECUTED in this sandbox (no PowerShell
here) — they are source-delivered per part15 §13/§16 and honestly
BLOCKED for execution; their first owner-side runs (with transcripts)
are the evidence that flips the corresponding checklist rows. The D-4/D-5
register rows convert to plain FIXED only on `test-rust.ps1`'s first
green `cargo test` run — never before.

— delivered r339 (SS11); guide written r340 at `0132d409`.
