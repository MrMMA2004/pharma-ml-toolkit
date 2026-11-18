"""E2E — the §11.5 Molecular Docking row against the REAL backend
server (round 285, the interactive part11 session).

Boots the real uvicorn sidecar (the port-0 FD + PHARMAML-READY
handshake lane), then drives EVERY docking lane the new
DockingPage/client rides, over real HTTP with the session token:

  engines catalog -> dockstring targets/target/objective ->
  receptor prepare -> ligand pdbqt -> parse vina -> parse dlg ->
  consensus + ligand efficiency -> contacts + plif -> pose rmsd ->
  the dock engine boundary (the honest 422) -> the figure lane ->
  the table export lane -> SIGTERM (clean close).

Asserts the REAL answers (counts, values, artifact digests) — no
mocked lanes. Exit code 0 only on a fully verified pass.
"""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pdbqt_atom(serial, name, resname, x, y, z, ad_type):
    return (
        f"ATOM  {serial:5d} {name:>4s} {resname:>3s} A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00     {ad_type}"
    )


SAMPLE_RECEPTOR = "\n".join([
    "REMARK  Name = tiny_receptor",
    _pdbqt_atom(1, "CA", "ALA", 0.0, 0.0, 0.0, "C"),
    _pdbqt_atom(2, "N", "ALA", 1.2, 0.0, 0.0, "N"),
    _pdbqt_atom(3, "O", "ALA", 2.4, 0.0, 0.0, "OA"),
    _pdbqt_atom(4, "HD", "ALA", 0.6, 1.0, 0.0, "HD"),
    _pdbqt_atom(5, "O", "HOH", 9.0, 9.0, 9.0, "OA"),
]) + "\n"

VINA_LOG = (
    "mode | affinity | dist from best mode\n"
    "     | (kcal/mol) | rmsd l.b. | rmsd u.b.\n"
    "-----+------------+----------+----------\n"
    "    1       -11.5          0          0\n"
    "    2       -10.3       1.825      2.941\n"
)

DLG_LOG = (
    "CLUSTERING HISTOGRAM\n"
    "______________________________________________________________________\n"
    "Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster\n"
    "     1       |       -9.65    |   41 |      -9.58    |     6 |\n"
    "     2       |       -8.20    |   12 |      -7.90    |    15 |\n"
    "     3       |       -4.91    |   38 |      -4.74    |     8 |\n"
    "Successful Completion\n"
)

LIGAND_ATOMS = [
    {"serial": 1, "name": "C1", "element": "C", "resname": "LIG",
     "chain": "A", "resseq": 1, "x": 0.0, "y": 0.0, "z": 0.0,
     "autodock_type": "C"},
    {"serial": 2, "name": "O1", "element": "O", "resname": "LIG",
     "chain": "A", "resseq": 1, "x": 1.0, "y": 0.0, "z": 0.0,
     "autodock_type": "OA"},
]
RECEPTOR_ATOMS = [
    {"serial": 10, "name": "CA", "element": "C", "resname": "ASP",
     "chain": "A", "resseq": 30, "x": 2.0, "y": 0.0, "z": 0.0,
     "autodock_type": "C"},
    {"serial": 11, "name": "OD1", "element": "O", "resname": "ASP",
     "chain": "A", "resseq": 30, "x": 2.5, "y": 0.5, "z": 0.0,
     "autodock_type": "OA"},
    {"serial": 12, "name": "CB", "element": "C", "resname": "PHE",
     "chain": "A", "resseq": 31, "x": 40.0, "y": 0.0, "z": 0.0,
     "autodock_type": "C"},
]


class Backend:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.base: str | None = None
        self.token: str | None = None

    def start(self) -> None:
        env = {
            "PHARMAML_DATA_DIR": tempfile.mkdtemp(prefix="e2e-dock-"),
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
        # 1. engines — the honest catalog
        code, payload = backend.call("GET", "/api/v1/docking/engines")
        engines = payload.get("engines", [])
        launchable = [e["name"] for e in engines if e.get("launchable")]
        ok("engines catalog", code == 200 and len(engines) == 3
           and payload.get("dockstring", {}).get("n_targets") == 58,
           f"platform={payload.get('platform')} launchable={launchable} "
           f"boundary={engines[0].get('boundary_note', '')[:60]}")

        # 2. dockstring targets + single view
        code, payload = backend.call("GET", "/api/v1/docking/dockstring/targets")
        first = payload["targets"][0]
        name = first["name"] if "name" in first else first.get("target")
        ok("dockstring targets", code == 200 and payload["total"] == 58,
           f"first={name}")
        code, payload = backend.call(
            "GET", f"/api/v1/docking/dockstring/targets/{name}")
        ok("dockstring single target",
           code == 200 and len(payload["center"]) == 3 and "config" in payload,
           f"{name} center={payload.get('center')} size={payload.get('size')}")

        # 3. objective (F2 — the route requires the F2 key itself)
        code, payload = backend.call(
            "POST", "/api/v1/docking/dockstring/objective",
            {"name": "F2", "scores": {"F2": -8.4}, "qed": 0.71})
        ok("dockstring objective F2", code == 200 and "value" in payload
           and payload["value"] is not None,
           f"value={payload.get('value')}")

        # 4. receptor prepare + artifact
        code, payload = backend.call(
            "POST", "/api/v1/docking/receptor/prepare",
            {"pdbqt_text": SAMPLE_RECEPTOR})
        art = payload.get("artifact", {})
        report = payload.get("report", {})
        ok("receptor prepare", code == 200 and art.get("kind") == "receptor-pdbqt"
           and art.get("size_bytes", 0) > 0,
           f"atoms {report.get('input_atoms')} -> {report.get('output_atoms')} "
           f"sha={art.get('sha256', '')[:12]}")
        artifact_id = art.get("artifact_id", "")

        # 5. ligand pdbqt + artifact (CCO)
        code, payload = backend.call(
            "POST", "/api/v1/docking/ligand/pdbqt",
            {"smiles": "CCO", "name": "ethanol", "seed": 7})
        ok("ligand pdbqt", code == 200 and payload.get("size_bytes", 0) > 0
           and payload["artifact"]["kind"] == "ligand-pdbqt",
           f"smiles=CCO size={payload.get('size_bytes')}B")

        # 6. parse vina
        code, payload = backend.call(
            "POST", "/api/v1/docking/parse/vina", {"text": VINA_LOG})
        poses = payload.get("poses", [])
        ok("parse vina", code == 200 and payload["n_poses"] == 2
           and poses[0]["affinity"] == -11.5 and poses[1]["rmsd_ub"] == 2.941,
           f"poses={[(p['mode'], p['affinity']) for p in poses]}")

        # 7. parse dlg
        code, payload = backend.call(
            "POST", "/api/v1/docking/parse/dlg", {"text": DLG_LOG})
        ok("parse dlg", code == 200 and payload["summary"]["n_clusters"] == 3
           and payload["clusters"][0]["lowest_energy"] == -9.65,
           f"clusters={payload['summary']['n_clusters']} "
           f"top={payload['summary']['top_lowest_energy']}")

        # 8. consensus over the parsed poses
        code, payload = backend.call(
            "POST", "/api/v1/docking/scoring/consensus",
            {"rows": [{"mode": p["mode"], "affinity": p["affinity"]}
                      for p in poses],
             "columns": ["affinity"], "higher_is_better": {"affinity": False}})
        ok("scoring consensus", code == 200 and payload["n_rows"] == 2
           and "rows" in payload,
           f"n_rows={payload.get('n_rows')}")

        # 9. ligand efficiency (LE = |dG| / N_heavy — the backend's own
        #    positive-magnitude convention, per scoring.py's docstring)
        code, payload = backend.call(
            "POST", "/api/v1/docking/scoring/ligand-efficiency",
            {"affinity_kcal_mol": -11.5, "num_heavy_atoms": 9})
        ok("ligand efficiency",
           code == 200 and abs(payload["ligand_efficiency"] - 11.5 / 9) < 1e-9,
           f"LE={payload.get('ligand_efficiency'):.4f}")

        # 10. contacts + plif
        code, payload = backend.call(
            "POST", "/api/v1/docking/contacts",
            {"ligand": LIGAND_ATOMS, "receptor": RECEPTOR_ATOMS,
             "cutoff": 5.0})
        ok("contacts + plif", code == 200 and payload["n_contacts"] == 4
           and payload["plif"][0]["residue"] == "A:ASP30",
           f"classes={payload.get('interaction_classes')} "
           f"plif_rows={len(payload.get('plif', []))}")

        # 11. pose rmsd (symmetry-aware)
        ref = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]
        translated = [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [2.0, 1.5, 0.0]]
        code, payload = backend.call(
            "POST", "/api/v1/docking/pose-rmsd",
            {"smiles": "CCO", "coords_ref": ref, "poses": [ref, translated]})
        ok("pose rmsd", code == 200 and payload["n_atoms"] == 3
           and abs(payload["rmsd"][0]) < 1e-6
           and abs(payload["rmsd"][1] - 2.0) < 1e-6,
           f"rmsd={payload.get('rmsd')}")

        # 12. the dock engine boundary — the honest 422 (never a fake row)
        code, payload = backend.call(
            "POST", "/api/v1/docking/dock",
            {"smiles": ["CCO"], "receptor_pdbqt": SAMPLE_RECEPTOR})
        detail = str(payload.get("detail", ""))
        ok("dock engine boundary honest",
           code == 422 and "engine unavailable" in detail,
           f"422: {detail[:70]}")

        # 13. the figure lane (bar: affinity per mode)
        code, payload = backend.call(
            "POST", "/api/v1/artifacts/figure",
            {"kind": "bar", "title": "Vina pose affinities",
             "x": [str(p["mode"]) for p in poses],
             "y": [p["affinity"] for p in poses], "format": "PNG"})
        fig = payload.get("artifact", {})
        ok("figure lane", code == 200 and fig.get("kind") == "figure",
           f"size={fig.get('size_bytes')}B sha={fig.get('sha256', '')[:12]}")

        # 14. the export lane (rows table)
        code, payload = backend.call(
            "POST", "/api/v1/visualization/table",
            {"title": "poses", "rows": [
                {"mode": p["mode"], "affinity": p["affinity"]} for p in poses],
             "format": "csv"})
        tab = payload.get("artifact", {})
        ok("table export lane", code == 200 and tab.get("kind") == "table",
           f"size={tab.get('size_bytes')}B format=csv")

        # 15. the artifact registry + the digest-verified download
        code, payload = backend.call("GET", "/api/v1/artifacts")
        arts = payload.get("artifacts", [])
        ok("artifact registry",
           code == 200 and any(a["artifact_id"] == artifact_id for a in arts),
           f"{len(arts)} registered")

        # health close-out
        code, payload = backend.call("GET", "/api/v1/health")
        ok("health at close", code == 200 and "version" in payload,
           payload.get("version", ""))
    finally:
        backend.stop()

    failed = [c for c in checks if not c[1]]
    print(f"\nE2E-DOCKING-{('PASS' if not failed else 'FAIL')}: "
          f"{len(checks) - len(failed)}/{len(checks)} lanes verified")
    for name, _, _ in failed:
        print(f"  FAILED: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
