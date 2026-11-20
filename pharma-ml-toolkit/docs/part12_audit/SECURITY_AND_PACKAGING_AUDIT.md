# Security and Packaging Audit (part12 PART 16)

Baseline v1.2.231 · audited 2026-11-20. Verified against source and the
live release this session.

## 1. Runtime security posture (verified PASS)

| Control | Implementation | Verdict |
| --- | --- | --- |
| Loopback binding | Backend binds port-0 on 127.0.0.1 it owns; Tauri `assert_loopback` refuses non-`http://127.0.0.1:{port}` handshakes; REST guard validates Host/Origin | PASS |
| Session token | `secrets.token_urlsafe(32)` generated INSIDE the backend (never argv/env/logs); memory-only in host + frontend; constant-time verify; WS carries it via query/header (browser header limitation documented) | PASS |
| CSP | `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src ipc: http://127.0.0.1:* ws://127.0.0.1:*; object-src 'none'; frame-ancestors 'none'` | PASS (offline-strict) |
| IPC surface | ONE Tauri command (`backend_endpoint`); no fs/shell/dialog commands; `core:default` capabilities only | PASS |
| Upload limits | Bounded multipart + byte caps on artifact store; bounded previews (≤500 rows/col); 5,000-row export cap | PASS |
| File access | FileVault manages payloads; no arbitrary FS paths from the frontend; the explorer's source viewer is path-constrained (`toolkit-fs`) | PASS |
| Dependency trust | Pinned stack (numpy 1.26.4, torch 2.14.1+cpu, rdkit 2026.9.1, shap 0.54.1 with the documented `--no-deps` metadata-lie note) | PASS |
| Token hygiene (this session) | PAT recovery via askpass shim; token never in git config/URLs/ps/output (recovery scripts print length + sha12 only); no PAT in commits/pushed files | PASS |

## 2. Packaging audit

| Item | Verdict | Evidence |
| --- | --- | --- |
| PyInstaller bundle completeness | PASS | Release run 44693818307 step 13 verified the frozen sidecar (handshake + authenticated health + guard refusal + CTRL_BREAK exit 0) on the runner |
| Tauri resource paths | PASS | Sidecar staged via `resources/backend → backend/`; release assets published |
| WebView2 offline | PASS (bundle) / BLOCKED (execution) | `webviewInstallMode: offlineInstaller`; runtime verify needs the clean machine |
| EXE/MSI sizes | EXE 557,655,583 B · MSI 710,122,853 B — verified on the live release this session (r301's three v1.2.230 size quotes are historical supersets, documented) | PASS |
| SHA-256 verification | PASS | `SHA256SUMS.txt` + release manifest attached; digest-verified artifact downloads in-app (X-Artifact-SHA256 + local recompute) |
| Release provenance | PASS | Tag → run → commit chain verified live (fire 3 at `3da0ff96`, all 26 steps) |
| Source-archive integrity | **PARTIAL → fixed this round** | The shipped v1.2.231 source zip omitted exactly one real source file (`frontend/src/services/artifacts/client.ts` — B-1); scoped exclusions fixed the builder/streamer; this round's archive re-ships complete |
| SBOM | PASS (CLOSED r304) | `packaging/sbom/sbom.cdx.json` (CycloneDX 1.6, 40 components, validated + reproducible) ships in the archive; the Windows workflow step 22b regenerates it on the runner + self-registers its hash into SHA256SUMS.txt |
| Vulnerability scanning | PARTIAL | Pinned-stack policy + `requirements.txt` snapshot; no automated scanner lane yet (owner decision) |
| Digital signing | BLOCKED | No certificate available |
| Installer integrity validation | PASS | Static packaging validation step 20 (26/26) + in-archive version gate |
| Uninstall/upgrade/user-data preservation | BLOCKED | Clean-machine lane (see WINDOWS_CLEAN_INSTALL_REPORT.md) |

## 3. SQLite data location + backup/restore

- `%LOCALAPPDATA%\PharmaML Studio` (XDG/`PHARMAML_STUDIO_DATA` respected);
  WAL journal, per-op connections, `busy_timeout`, schema migrations
  idempotent+transactional, `recover_registry()` with honest
  `startup_recovery.dropped`.
- Backup/restore: online-backup API bundle + vault payloads, artifact-registered
  with sha256; restore validates integrity + schema before swapping; live
  registry rebuilt from the backup (the documented replace semantics —
  now stated in the GUI's confirm step too, B-2).

## 4. Release pipeline hardening (verified)

- 26-step ladder with test gates BEFORE packaging (steps 8–11: tsc,
  frontend contract guards, backend units, API/database suite).
- Version-carrier validation at step 2 (the eighth-carrier guard born from
  fire 1 of v1.2.231); plain-version CHANGELOG header rule enforced by
  `probe_versions` (born from fire 2).
- Root-level workflow registration (round-293 rule) with drift guards;
  the malformed nested ci.yml removed this round (B-3).
- Provenance: BLAS-exposed rounding (6-decimal float64) keeps census pins
  byte-stable across the mixed Intel/AMD runner fleet (r298/r299).

## 5. Verdict

Runtime security: **PASS** across every control audited. Packaging:
**PASS** — the source-archive omission fixed at the r302 re-ship (the
d3673da7 archive carries it), SBOM CLOSED by r304 (re-verified in-tree
at the r307 reconciliation), and the external-gated items remain
honestly BLOCKED (signing certificate-gated; clean-machine no-machine).
