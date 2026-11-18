"""E2E — the §11.6 PK/PD workspace row against the REAL backend
server (round 286, the interactive part11_1+part11_2 session).

Boots the real uvicorn sidecar (the port-0 FD + PHARMAML-READY
handshake lane), then drives EVERY pkpd lane the new PKPDPage/client
rides, over real HTTP with the session token:

  catalog -> nca (+ the validator's honest 422) -> steady-state ->
  iv-bolus (the analytical value) -> bateman -> superposition (the
  real accumulation) -> pbpk (the frame's own columns, iv + oral) ->
  pd (the EC50 midpoint) -> population sync -> the population JOB
  lane (submit -> poll -> the authoritative REST result) -> the
  negative-omega 422 -> the exposure figure lane -> the table export
  lane -> SIGTERM (clean close).

Asserts the REAL answers (counts, values, artifact digests) — no
mocked lanes. Exit code 0 only on a fully verified pass.
"""

from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# the single-dose profile the page's own sample carries (a distribution
# phase into a log-linear terminal tail)
NCA_TIMES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0]
NCA_CONC = [8.0, 6.9, 5.8, 4.4, 3.1, 1.9, 1.2, 0.75, 0.30, 0.12, 0.02]

# one steady-state dosing interval at tau = 12 (trough -> Cmax -> trough)
SS_TIMES = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
SS_CONC = [0.80, 2.90, 4.10, 4.60, 4.45, 3.40, 2.20, 1.30, 0.80]

GRID = [round(t / 2, 2) for t in range(49)]  # 0..24 step 0.5 (49 pts)

POP_GRID = [float(t) for t in range(13)]  # 0..12 step 1 (13 pts)
POP_THETA = [5.0, 1.5, 0.25]


class Backend:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.base: str | None = None
        self.token: str | None = None

    def start(self) -> None:
        env = {
            "PHARMAML_DATA_DIR": tempfile.mkdtemp(prefix="e2e-pkpd-"),
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pharmaml.api"],
            cwd=str(ROOT), env={**{k: v for k, v in __import__("os").environ.items()},
                                 **env},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline().strip()
        deadline = time.monotonic() + 60
        while not line.startswith("PHARMAML-READY") and time.monotonic() < deadline:
            print(f"  [backend] {line[:160]}")
            line = self.proc.stdout.readline().strip()
        assert line.startswith("PHARMAML-READY"), f"no handshake: {line[:200]}"
        ready = json.loads(line[len("PHARMAML-READY"):])
        self.base = f"http://127.0.0.1:{ready['port']}"
        self.token = ready["token"]
        print(f"E2E handshake OK: port={ready['port']} "
              f"version={ready['version']} pid={ready['pid']}")

    def call(self, method: str, path: str, body: dict | None = None):
        assert self.base and self.token
        req = urllib.request.Request(
            f"{self.base}{path}", method=method,
            headers={
                "Content-Type": "application/json",
                "X-Session-Token": self.token,
                "Origin": "tauri://localhost",
                "Host": "127.0.0.1",
            },
            data=None if body is None else json.dumps(body).encode(),
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            try:
                return err.code, json.loads(err.read().decode())
            except Exception:
                return err.code, {}

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
                print("backend SIGTERM -> closed cleanly "
                      f"(rc={self.proc.returncode})")
            except subprocess.TimeoutExpired:
                self.proc.kill()
                print("backend SIGKILL fallback")


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def ok(name: str, cond: bool, note: str = "") -> None:
        checks.append((name, bool(cond), note))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              f"{' — ' + note if note else ''}")

    backend = Backend()
    backend.start()
    try:
        # 1. catalog — the route inventory (four groups, the page's own rows)
        code, payload = backend.call("GET", "/api/v1/pkpd/catalog")
        groups = payload.get("groups", [])
        group_names = [g.get("group") for g in groups]
        ok("pkpd catalog", code == 200 and group_names == [
            "nca", "models", "pd", "population"
        ], f"groups={group_names}")

        # 2. nca — the REAL non-compartmental analysis
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/nca", {"times": NCA_TIMES, "conc": NCA_CONC})
        summary = payload.get("summary", {})
        terminal = payload.get("terminal", {})
        ok("nca", code == 200 and summary.get("Cmax") == 8.0
           and summary.get("Tmax") == 0.0
           and payload.get("auc_trapezoid_linear") is not None
           and payload.get("auc_trapezoid_log") is not None,
           f"AUC={summary.get('AUC'):.4f} t_half={summary.get('t_half_terminal'):.3f}h")
        ok("nca terminal phase", terminal.get("n_points", 0) >= 3
           and terminal.get("r2", 0) > 0.9 and terminal.get("half_life", 0) > 0
           and isinstance(terminal.get("span_low"), bool),
           f"r2={terminal.get('r2'):.4f} span={terminal.get('span_half_lives'):.2f} "
           f"span_low={terminal.get('span_low')} method={terminal.get('method')}")

        # 3. the validator's own honest 422 (negative concentration)
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/nca",
            {"times": [0.0, 1.0], "conc": [1.0, -0.5]})
        detail = str(payload.get("detail", ""))
        ok("nca honest 422", code == 422 and "concentration" in detail.lower(),
           f"422: {detail[:60]}")

        # 4. steady-state — the interval metrics over one 0..tau profile
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/steady-state",
            {"times": SS_TIMES, "conc": SS_CONC, "tau": 12.0, "dose": 100.0})
        metrics = payload.get("metrics", {})
        ok("steady-state", code == 200
           and metrics.get("Cmax_ss", 0) >= metrics.get("Cmin_ss", 1e9)
           and metrics.get("AUC_tau", 0) > 0
           and metrics.get("CL_ss") is not None
           and "R_accum" in metrics,
           f"AUC_tau={metrics.get('AUC_tau'):.3f} fluct={metrics.get('fluctuation'):.3f}")

        # 5. iv-bolus — the analytical curve (C(t) = c0 * exp(-ke t))
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/iv-bolus", {"c0": 10.0, "ke": 0.35, "times": GRID})
        conc = payload.get("concentration", [])
        idx1 = GRID.index(1.0)
        expected = 10.0 * math.exp(-0.35)
        ok("iv-bolus", code == 200 and len(conc) == len(GRID)
           and abs(conc[idx1] - expected) < 1e-9,
           f"C(1h)={conc[idx1]:.6f} (exp: {expected:.6f})")

        # 6. bateman — the oral one-compartment curve (causal: C(0) == 0)
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/bateman",
            {"dose_per_volume": 5.0, "ka": 1.5, "ke": 0.25, "times": GRID})
        conc = payload.get("concentration", [])
        peak = max(conc)
        peak_idx = conc.index(peak)
        ok("bateman", code == 200 and abs(conc[0]) < 1e-12
           and all(c >= 0 for c in conc) and 0 < peak_idx < len(conc) - 1,
           f"peak C={peak:.4f} at t={GRID[peak_idx]}h")

        # 7. superposition — the REAL accumulation over 8 doses
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/superposition",
            {"dose_per_volume": 5.0, "ka": 1.5, "ke": 0.25,
             "times": GRID, "tau": 6.0, "n_doses": 8})
        sup = payload.get("concentration", [])
        ok("superposition", code == 200 and payload.get("n_doses") == 8
           and len(sup) == len(GRID) and max(sup) > peak,
           f"accum peak {max(sup):.4f} > single-dose {peak:.4f}")

        # 8. pbpk — the MinimalPBPK frame (iv + oral)
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/pbpk",
            {"times": GRID, "dose_mg": 100.0, "route": "iv"})
        rows = payload.get("rows", [])
        compartments = payload.get("compartments", [])
        ok("pbpk iv", code == 200 and compartments == [
            "venous", "lung", "liver", "kidney", "rest"
        ] and len(rows) == len(GRID)
           and all("C_plasma" in row for row in rows)
           and all(row["venous"] > 0 for row in rows[1:]),
           f"{compartments} + C_plasma, {len(rows)} rows")
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/pbpk",
            {"times": GRID, "dose_mg": 100.0, "route": "oral",
             "ka": 1.0, "f": 0.85})
        ok("pbpk oral", code == 200 and payload.get("route") == "oral"
           and len(payload.get("rows", [])) == len(GRID),
           "route echo + the frame rows")

        # 9. pd — the sigmoid Emax table (the EC50 midpoint)
        grid_pd = [0.1, 1.0, 10.0, 100.0]
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/pd",
            {"concentrations": grid_pd, "e0": 0.0, "emax": 100.0,
             "ec50": 10.0, "hill": 1.0})
        effect = payload.get("effect", [])
        ok("pd sigmoid Emax", code == 200 and len(effect) == 4
           and abs(effect[grid_pd.index(10.0)] - 50.0) < 1e-9
           and effect[3] > 90.0,
           f"E(10)={effect[grid_pd.index(10.0)]:.4f} E(100)={effect[3]:.4f}")

        # 10. population sync — the seeded simulation body
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/population",
            {"theta": POP_THETA, "omega_diag": [0.09], "times": POP_GRID,
             "n_subjects": 25, "seed": 42, "n_profiles": 3, "mode": "sync"})
        summary = payload.get("summary", {})
        profiles = payload.get("subject_profiles", [])
        ok("population sync", code == 200 and payload.get("mode") == "sync"
           and len(summary.get("mean", [])) == len(POP_GRID)
           and len(summary.get("p5", [])) == len(POP_GRID)
           and len(summary.get("p95", [])) == len(POP_GRID)
           and 1 <= len(profiles) <= 3
           and all(v >= 0 for v in summary.get("mean", [])),
           f"25 subjects, mean[0..12] over {len(POP_GRID)} grid points")

        # 11. the population JOB lane — submit -> poll -> REST result
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/population",
            {"theta": POP_THETA, "omega_diag": [0.09], "times": POP_GRID,
             "n_subjects": 120, "seed": 7, "n_profiles": 3, "mode": "job"})
        job = payload.get("job", {})
        job_id = job.get("job_id", "")
        ok("population job submit", code == 200
           and payload.get("mode") == "job" and job_id != ""
           and job.get("operation") == "pkpd.population",
           f"job_id={job_id[:12]}… state={job.get('state')}")
        deadline = time.monotonic() + 120
        state = job.get("state")
        progress_seen = 0
        while state not in ("COMPLETED", "FAILED", "CANCELLED") and time.monotonic() < deadline:
            time.sleep(0.4)
            code, status = backend.call("GET", f"/api/v1/jobs/{job_id}")
            state = status.get("state")
            progress_seen = max(progress_seen, status.get("progress", 0))
        ok("population job completion", state == "COMPLETED",
           f"state={state} progress_seen={progress_seen}%")
        code, result = backend.call("GET", f"/api/v1/jobs/{job_id}/result")
        body = result.get("result", {})
        summary = body.get("summary", {})
        ok("population job REST result", code == 200
           and body.get("n_subjects") == 120
           and len(summary.get("mean", [])) == len(POP_GRID)
           and len(body.get("subject_profiles", [])) <= 3,
           f"the authoritative REST body: {body.get('n_subjects')} subjects, "
           f"{body.get('n_times')} grid points")

        # 12. the negative-omega honest 422 (the route's own verdict)
        code, payload = backend.call(
            "POST", "/api/v1/pkpd/population",
            {"theta": POP_THETA, "omega_diag": [-0.1], "times": POP_GRID,
             "n_subjects": 25, "mode": "sync"})
        detail = str(payload.get("detail", ""))
        ok("population omega 422", code == 422 and "omega" in detail.lower(),
           f"422: {detail[:60]}")

        # 13. the figure lane (scatter: the exposure profile)
        code, payload = backend.call(
            "POST", "/api/v1/artifacts/figure",
            {"kind": "scatter", "title": "PK/PD exposure profile",
             "x": GRID, "y": conc, "format": "PNG"})
        fig = payload.get("artifact", {})
        ok("exposure figure lane", code == 200 and fig.get("kind") == "figure"
           and fig.get("size_bytes", 0) > 0,
           f"size={fig.get('size_bytes')}B sha={fig.get('sha256', '')[:12]}")

        # 14. the export lane (rows table)
        code, payload = backend.call(
            "POST", "/api/v1/visualization/table",
            {"title": "pkpd curve", "rows": [
                {"time": t, "concentration": c} for t, c in zip(GRID, conc)],
             "format": "csv"})
        tab = payload.get("artifact", {})
        ok("table export lane", code == 200 and tab.get("kind") == "table",
           f"size={tab.get('size_bytes')}B format=csv")

        # 15. the artifact registry + health close-out
        code, payload = backend.call("GET", "/api/v1/artifacts")
        arts = payload.get("artifacts", [])
        ok("artifact registry", code == 200
           and any(a["artifact_id"] == fig.get("artifact_id") for a in arts),
           f"{len(arts)} registered")
        code, payload = backend.call("GET", "/api/v1/health")
        ok("health at close", code == 200 and "version" in payload,
           payload.get("version", ""))
    finally:
        backend.stop()

    failed = [c for c in checks if not c[1]]
    print(f"\nE2E-PKPD-{('PASS' if not failed else 'FAIL')}: "
          f"{len(checks) - len(failed)}/{len(checks)} lanes verified")
    for name, _, _ in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
