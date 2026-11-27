# WINDOWS CLEAN INSTALL REPORT (part19, Phase 8 — Round 368, 2026-11-27)

## 1. The verification ladder — runner vs clean-machine, honestly separated

| Lane | What it proves | Status |
|---|---|---|
| **GitHub-hosted runner** (windows-install-smoke, run 45912713199 at v1.2.243) | MSI silent install + the INSTALLED sidecar's handshake/health/guard/shutdown probe, host launch + lifecycle legs, Defender scans, NSIS install/uninstall, upgrade 1.2.242→1.2.243 with user-data marker retention — 20 steps SUCCESS | Verified by GitHub Actions (historical, r367 ledger); **not equivalent to a clean-machine check** — the runner image carries preinstalled toolchains and known baseline state |
| **Clean machine** (the owner's genuinely clean Windows box) | Fresh/offline install, no-Python/no-Node/no-Rust startup, WebView2-offline availability, workspace execution, exports, backup/restore, upgrade, uninstall/reinstall on a machine the runner cannot emulate | **NOT_RUN** — Windows-only; a reproducible PowerShell suite now exists (§2) so the owner can execute it; until the owner runs it, this row stays honestly NOT_RUN (never silently flipped to PASS) |

## 2. The reproducible clean-machine validation suite (delivered this round)

`pharma-ml-toolkit/scripts/windows/validate-clean-machine.ps1` — PowerShell 5.1-compatible
(matching the `scripts/windows/` house rules: resolves the repo root itself, per-leg exit
semantics, transcripts under `logs/windows-tests/clean-machine/`, never prints/persists a token
or user secret, disposable synthetic data only, never touches real application data).

Legs and what each collects (the part19 Phase 8 collection list):

* **census** — Windows version/arch, CPU + cores, memory total/free, GPU when present, WebView2
  Evergreen runtime state (registry channels), Python/Node/Rust presence (each reported with the
  note that absence is EXPECTED for the installed app — the sidecar bundles its own runtime, the
  frontend ships pre-built, the host ships compiled).
* **integrity** — per-installer SHA-256 + size; Authenticode status reported honestly (an
  unsigned installer maps to BLOCKED-with-reason: signing remains certificate-gated per the
  standing register — the hash is the integrity anchor); the size census note.
* **sidecar** — launches the installed `pharmaml-backend.exe` directly, captures the single
  `PHARMAML-READY {port, token, pid, version}` handshake line (token held in memory only),
  authenticated `GET /api/v1/health` 200, unauthenticated refusal, graceful shutdown — the same
  wire contract `verify_sidecar.py` pins on the runner.
* **workspaces** — the nine-workspace execution matrix over the live API with synthetic data:
  dataset import + preview, ML registry + logistic train + predict, DL catalog + 2-epoch DNN
  train (CPU), semi-supervised lane, applicability-domain lane, PK/PD catalog + NCA calculator,
  MD catalog (analysis-only), visualization stat figure, docking engine census (read-only —
  execution lanes honestly gated on the machine's engines).
* **artifacts** — figure + table export lanes, SQLite backup + restore round-trip (disposable
  store; ephemeral-mode refusal honestly maps to BLOCKED-with-reason), trained-model registry.
* **lifecycle** — upgrade/uninstall/reinstall: BLOCKED-with-reason by design (requires the prior
  release's installer in place; the runner's upgrade leg is the verified reference).

Output: machine-readable JSON + human-readable Markdown per run (both under
`logs/windows-tests/clean-machine/`). Overall verdict PASS/FAIL with BLOCKED legs listed and
never silently skipped.

**Execution status this round: NOT_RUN (BLOCKED — Windows-only; this sandbox is Linux).** The
suite is delivered for the owner's clean box; the report rows in §1 remain honest until the
owner executes it.

## 3. Installer size investigation (EXE ≈ 558 MB / MSI ≈ 711 MB)

The payload composition (from the release pipeline's build flow — the r367 ledger + the SBOM +
the spec):

1. **The frozen FastAPI sidecar** (PyInstaller onedir from `pharmaml_backend.spec`) — embeds the
   Python runtime and the **full offline scientific stack**: torch-CPU (the single largest
   contributor), RDKit (+ its data files), scikit-learn/SciPy/pandas/NumPy, the boosting trio,
   SHAP, matplotlib, openpyxl, selfies, meeko/gemmi. This is deliberate: the part9 rule — "do not
   silently omit a capability to shrink the installer" — every optional feature group the nine
   workspaces exercise ships offline.
2. **The Tauri host + the pre-built React frontend** (WebView2-rendered) — small relative to the
   Python payload.
3. **WebView2 offline capability** — the Evergreen offline bootstrapper bundled for the no-network
   install lane.
4. **What is NOT in the payload:** the Qt/PySide6 stack — the production sidecar spec EXCLUDES the
   whole Qt family (`pharmaml_backend.spec` excludes PySide6/shiboken6/pyqtgraph/qt_material/
   qdarkstyle; pinned by `tests/test_legacy_retirement.py::test_sidecar_spec_excludes_the_qt_stack`
   and the new r368 production-workflow guards). The retired Inno-Setup lane (which DID embed
   PySide6/Qt) is banner-marked RETIRED and referenced by no release workflow.

**Verdict on reduction:** the realistic reduction levers are (a) trimming torch to the CPU-only
wheel — already done (the +cpu wheel is pinned first precisely so the resolver never pulls the
GPU-flavoured wheel, "which would triple the installer size"); (b) excluding Qt — already done;
(c) dropping required scientific packages — NOT acceptable (the offline-install and
required-dependency constraints of Phase 8); (d) moving distributables out of Git into release
assets — the Phase 10 recommendation. No further reduction is available without sacrificing the
offline guarantee or the scientific feature set; **the sizes are honest and justified**.
(Build-time note: the build machine's `installer-requirements.txt` also installs the Qt stack —
harmless to payload size since the spec excludes it from the freeze; a future pass could prune
the build-venv to speed CI, registered as INFORMATIONAL in Phase 10.)

## 4. Required updates to adjacent documents

* `RELEASE_READINESS.md` — the clean-machine row remains NOT_RUN with the new suite named as the
  executable path (this round's edit).
* The `WINDOWS_VALIDATION_GUIDE.md` script table gains the new suite as the tenth script (the
  owner-facing entry point for the whole clean-machine matrix).

## 5. Standing constraint

Windows-only operations were not simulated, per the work-safety rule: every Windows-specific
claim above that was not executed is marked NOT_RUN/BLOCKED with its reproducible path.
