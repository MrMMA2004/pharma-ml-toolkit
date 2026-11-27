# WINDOWS EXE/MSI AUDIT — MASTER_PROMPT part20 deliverable C (round 370)

## 1. Build workflow (the ACTIVE lane — verified line-level this round)

`.github/workflows/windows-release.yml` (repo root; triggers:
`workflow_dispatch` + `push: tags: v*`; `contents: write`; windows-2022,
120 min) builds BOTH installers from the pushed tag:

1. checkout the tag (no ref override → the tag's exact commit);
   tag==VERSION==tauri.conf.json==pyproject quadruple validated first;
2. Python 3.12 venv + `requirements.txt` (frozen pins: rdkit, torch
   2.14.1+cpu, meeko, gemmi, mordred…);
3. PyInstaller onedir freeze via `packaging/windows/pharmaml_backend.spec`
   (entry `backend_entry.py`; **Qt-free**: PySide6/pytest/dev tools
   excluded by spec; datas: engines/, LICENSE, VERSION, meeko) →
   `build/windows-backend/pharmaml-backend/pharmaml-backend.exe`;
4. sidecar staged into `src-tauri/resources/backend` (bundle resource —
   NOT an externalBin; tauri.conf.json `resources` maps it);
5. Vite production frontend build;
6. `cargo tauri build` → **NSIS** `PharmaML Studio_v{ver}_x64-setup.exe`
   + **WiX** `…_x64_en-US.msi` (WebView2 offline embedded; currentUser
   install mode);
7. staged to `release/windows/` as `PharmaML_Studio_Setup_v{ver}.exe` +
   `PharmaML_Studio_v{ver}.msi` + RELEASE_MANIFEST.json + SHA256SUMS.txt +
   `sbom.cdx.json` (CycloneDX 1.6);
8. artifact upload `pharmaml-studio-windows-${{ github.sha }}` + release-asset
   attach on tag builds.

The Inno-Setup lane (`packaging/windows/build_windows.ps1` + `installer.iss`
+ `pharmaml.spec`) is RETIRED (banner-marked "RETIRED PACKAGING LANE
(r368/part19 Phase 2)"; zero workflow references; the 26 retirement guards
pin: production must freeze the Qt-free `pharmaml_backend.spec` and never
reference the legacy lane — guards 26/26 green this round).

## 2. Source commit, version, provenance (part20 §7.1 — finding B revalidated)

| Item | Value | Evidence |
| --- | --- | --- |
| Tag | v1.2.243 | git tag list + release API |
| Release build commit | `902dd89b3dbf029595694967df1a9e46ab310e77` | the r367 release ladder (first build 45905675582 died at step 10 on the moved ui/ reads → fix commit 902dd89b → tag re-pinned → run **45907478532 SUCCESS**) |
| Build workflow run | 45907478532 (SUCCESS) | r367 ledger; push CI over 902dd89b green (CI 45907471751 + Push 45907472523) |
| main HEAD (this audit) | bd53f7b3 (+ this round's part-20 fixes) | git |
| **Post-release source delta** | **16 commits / 172 files / +3,540 −2,064** | `git diff --stat 902dd89b..bd53f7b3` (38 backend, 27 docs, 17 examples, 15 web-console components, 11 web lib, 11 tests, 7 scratch, 7 legacy, 3 frontend, 2 scripts, 2 packaging, 1 worklog, 1 github-page test, zip re-carried) |

**Classification of the post-tag changes (do they affect the shipped
installers?):**
* **backend (38 files)** — the r368/r369 rounds' test re-aiming, guard
  extension, and docs-adjacent source edits; no shipped-behavior defect
  fix among them (the r368 F-1/F-3 fixes are frontend), but the scientific
  tree did move after the freeze.
* **frontend (3 files)** — the r368 **F-1 fix** (three Datasets chemistry
  buttons' in-flight busy guards) and **F-3 fix** (StageHeader dead icon
  prop) — real UI fixes NOT in the shipped v1.2.243 installers.
* **packaging (2 files)** — retirement banners only.
* **this round (r370)** — the D-9 test-pin fix (turns the failing CI
  chemistry lane green — a test, not shipped code), the D-10 Cargo.toml
  version-carrier alignment (metadata; the crate version was stale at
  1.2.229 across the r229→r243 releases — the tauri.conf.json carrier that
  actually drives the installer version was always correct), and the QA
  docs.

**Verdict:** the v1.2.243 installers are internally consistent
(manifest==tag commit==VERSION) but **lag main by the 16-commit delta**
(r368 F-1/F-3 UI fixes + the r370 CI/metadata fixes) — registered as
**D-15 (P2, OPEN)**: the next release cycle (owner-tagged, per the
no-release-without-authorization rule) must carry them. No asset was
overwritten and no release was published this round.

## 3. Packaged contents (verified from the build spec + r367/r369 asset evidence)

Main executable (Tauri host, WebView2 shell) · frontend dist assets ·
the frozen pharmaml-backend.exe (onedir: scientific backend modules,
native Python extensions/DLLs, rdkit/torch-cpu/numpy/pandas/sklearn stack) ·
engines/ + meeko data (docking resources) · VERSION/LICENSE · config ·
SQLite/FileVault runtime homes under user-data · logging. Qt/PySide6
EXCLUDED from the shipped payload (spec + 26/26 guards). WebView2 runtime is
embedded offline (the workflow pins the offline installer — no network
fetch at install time). Asset sizes (EXE 558,600,414 B / MSI 711,110,134 B)
investigated and justified in WINDOWS_CLEAN_INSTALL_REPORT.md §3
(torch-CPU the largest contributor; no reduction without breaking the
offline guarantee).

## 4. Installation/upgrade/uninstall (runner evidence — r367 install-smoke run 45912713199, all 20 steps)

Executed on windows-2022 (GitHub-hosted) over the PUBLISHED assets:
SHA256SUMS + manifest cross-check (commit==tag) · Defender scans of both
installers · MSI silent install + the INSTALLED sidecar live-probe
(PHARMAML-READY handshake, token held in-memory, authenticated health 200,
guard refusal, graceful shutdown — the `verify_sidecar.py` wire contract) ·
NSIS install + lifecycle + uninstall · **upgrade 1.2.242 → 1.2.243 with
the user-data marker retained** · evidence bundle uploaded. Per part20 §7.3:
a successful installer exit code was never treated as the proof — the
sidecar live-probe + host launch legs are the functional evidence.

**Not runner-covered (honestly bounded):** interactive install wizard,
paths-with-spaces installs under standard user permissions on physical
machines, Win10-vs-Win11 physical matrix — these ride
`scripts/windows/validate-clean-machine.ps1` (the clean-machine lane,
STATUS **NOT_RUN until the owner executes it** — never conflated with
runner evidence) plus `scripts/windows/test-installers.ps1` (the
integrity-only leg: digests vs published SHA256SUMS + manifest
cross-check, exit 0/1/2).

## 5. Offline operation

PASS on the carried evidence: WebView2 offline embedded; loopback-only CSP
(`connect-src ipc: http://ipc.localhost http://127.0.0.1:* ws://127.0.0.1:*` —
no remote origins); the backend binds 127.0.0.1 itself; no remote service
in the startup path (the startup handshake is local stdout). No
undocumented remote dependency found in the freeze spec or the host code.

## 6. Platform compatibility claims — honestly bounded

| Leg | Evidence | State |
| --- | --- | --- |
| GitHub windows-2022 runner | r367 build + install-smoke + r369 asset re-poll | PASS |
| Clean VM / physical machine | validate-clean-machine.ps1 (delivered r368) | NOT_RUN — owner lane |
| Win10/Win11 physical matrix | none | BLOCKED (no Windows infra in this sandbox — reported, not fabricated) |
| This sandbox (Linux) | — | EXE/MSI builds + local installs BLOCKED by environment; never claimed |

## 7. Signing and integrity

**Authenticode: BLOCKED — certificate-gated.** No certificate exists or
was fabricated; the installers are NOT signed and this is never claimed
otherwise. Integrity is independently verified: SHA256SUMS + RELEASE_MANIFEST
(manifest==tag commit==version) generated by the workflow and cross-checked
by both the install-smoke lane and the local integrity leg; SBOM
(CycloneDX 1.6, 23,634 B) carried per release. Release evidence retained:
source commit, run IDs, manifest, checksums, SBOM (the r367/r369 ledgers).

## 8. Remaining blockers

1. **D-15** — installers lag main by the post-tag fixes (next release
   cycle; owner-authorized tag push is the trigger).
2. **Signing** — BLOCKED until a certificate arrives through an authorized
   channel (status stays explicit; no fake).
3. **Clean-machine + physical Win10/11 validation** — owner-executed lane
   (suite delivered, STATUS NOT_RUN).
4. **Fresh build from this revision** — BLOCKED in this sandbox (no
   Windows, no cargo); the workflow is the reproducible path (tag push).
