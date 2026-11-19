"""part10 row four — the docking / pkpd / md / visualization /
applicability-domain API surfaces over the REAL backend.

Every route is exercised against the real pharmacometrics, docking,
MD, viz and AD layers (no mocks): the engines catalog reports the
honest Windows-PE boundary of this host, the managed jobs publish
REAL progress telemetry on the bus (per bootstrap member, per MD
stage, per population batch, per report section), cancellation lands
CANCELLED (never FAILED), and every figure/table/receptor/ligand
artifact round-trips byte-for-byte through the download contract.
"""

from __future__ import annotations

import io
import math
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _csv_bytes(rows: int = 24) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        buf.write(f"{round(0.1 * i + 0.5, 3)},{(i * 7) % 5 * 0.25},{i % 2}\n")
    return buf.getvalue().encode()


def _upload(client, data: bytes | None = None, name: str = "toy.csv",
            target: str = "y", mode: str = "sync") -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, data or _csv_bytes(), "text/csv")},
        data={"target": target, "mode": mode},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _train(client, dataset_id: str, model: str = "random_forest",
           mode: str = "sync", **extra) -> dict:
    body = {
        "dataset_id": dataset_id,
        "model": model,
        "tune": False,
        "n_jobs": -1,
        "mode": mode,
        **extra,
    }
    r = client.post("/api/v1/ml/train", headers=auth_headers(), json=body)
    assert r.status_code == 200, r.text
    return r.json()["model"]


def _wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if job["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never finished: {job}")


def _job_result(client, job_id: str) -> dict:
    r = client.get(
        f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
    )
    assert r.status_code == 200, r.text
    return r.json()["result"]


def _bus_events(client) -> list[dict]:
    return [
        e.model_dump(mode="json", exclude_none=True)
        for e in client.app.state.bus.replay(0)
    ]


def _pdbqt_atom(serial: int, name: str, resname: str, x: float, y: float,
                z: float, atom_type: str) -> str:
    return (
        f"ATOM  {serial:5d} {name:>4s} {resname:>3s} A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}    0.000 {atom_type:>2s}"
    )


SAMPLE_RECEPTOR_PDBQT = "\n".join([
    "REMARK  Name = tiny_receptor",
    _pdbqt_atom(1, "CA", "ALA", 0.0, 0.0, 0.0, "C"),
    _pdbqt_atom(2, "N", "ALA", 1.2, 0.0, 0.0, "N"),
    _pdbqt_atom(3, "O", "ALA", 2.4, 0.0, 0.0, "OA"),
    _pdbqt_atom(4, "HD", "ALA", 0.6, 1.0, 0.0, "HD"),
    _pdbqt_atom(5, "O", "HOH", 9.0, 9.0, 9.0, "OA"),
]) + "\n"

SAMPLE_VINA_OUTPUT = (
    "Estimated Free Energy of Binding   = -10.3 kcal/mol\n"
    "Modes | Affinity | dist from best mode\n"
    "      |          | rmsd l.b. | rmsd u.b.\n"
    "------+----------+----------+----------\n"
    "   1      -10.3          0          0\n"
    "   2       -9.6      1.842      2.301\n"
    "   3       -9.4      2.105      2.777\n"
    "------+----------+----------+----------\n"
)

SAMPLE_DLG = """\
AutoDock4 log file (synthetic)

________
CLUSTERING HISTOGRAM
__________________
______________________________________________________________________
Cluster Rank | Lowest Energy | Run | Mean Energy | # in Cluster
_____________|_______________|_____|_____________|________________
     1       |     -11.5     |  3  |    -11.2    |      7 |
     2       |     -10.8     |  9  |    -10.5    |     34 |
____________|_______________|_____|_____________|________________
"""


# ---------------------------------------------------------------- docking

class TestDockingCatalogs:

    def test_engines_catalog_honest_boundary(self, client):
        r = client.get("/api/v1/docking/engines", headers=auth_headers())
        assert r.status_code == 200
        payload = r.json()
        names = {e["name"] for e in payload["engines"]}
        assert names == {"vina", "autogrid4", "autodock4"}
        # bundled Windows binaries, no wine on this host: not launchable
        assert all(e["bundled"] for e in payload["engines"])
        if os.name == "nt":
            # r297: on a Windows host the bundled PEs are NATIVE -- the
            # catalog's launchable flag is honestly True there (the design's
            # whole Windows deployment: the sha256-verified engines/vina.exe
            # + friends, tracked in this very repository) and the boundary
            # note says so. The runner carries all three.
            assert all(e["launchable"] for e in payload["engines"])
            assert all(
                e["boundary_note"] == "engine resolves and launches on this host"
                for e in payload["engines"]
            )
        else:
            assert all(not e["launchable"] for e in payload["engines"])
            assert any("Windows" in (e["boundary_note"] or "")
                       for e in payload["engines"])
        assert payload["dockstring"]["n_targets"] == 58
        assert payload["dockstring"]["objectives"] == [
            "F2", "promiscuous_PPAR", "selective_JAK2"
        ]

    def test_dockstring_targets_and_single_view(self, client):
        r = client.get(
            "/api/v1/docking/dockstring/targets", headers=auth_headers()
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["total"] == 58
        first = payload["targets"][0]
        assert set(first) == {
            "target", "center_x", "center_y", "center_z",
            "size_x", "size_y", "size_z",
        }
        name = first["target"]
        r = client.get(
            f"/api/v1/docking/dockstring/targets/{name}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        view = r.json()
        assert view["name"] == name
        assert len(view["center"]) == 3 and len(view["size"]) == 3
        assert "center_x" in view["config"]

    def test_dockstring_unknown_target_422(self, client):
        r = client.get(
            "/api/v1/docking/dockstring/targets/NOT_A_TARGET",
            headers=auth_headers(),
        )
        assert r.status_code == 422

    def test_dockstring_objective_real_math(self, client):
        r = client.post(
            "/api/v1/docking/dockstring/objective",
            headers=auth_headers(),
            json={"name": "F2", "scores": {"F2": -9.5}, "qed": 0.8},
        )
        assert r.status_code == 200
        # F2 + 10 * (1 - qed) = -9.5 + 2.0
        assert r.json()["value"] == pytest.approx(-7.5)

    def test_dockstring_objective_unknown_422(self, client):
        r = client.post(
            "/api/v1/docking/dockstring/objective",
            headers=auth_headers(),
            json={"name": "bogus", "scores": {}, "qed": 1.0},
        )
        assert r.status_code == 422
        assert "F2" in r.json()["detail"]


class TestDockingPreparation:

    def test_receptor_prepare_pdbqt_mode(self, client):
        r = client.post(
            "/api/v1/docking/receptor/prepare",
            headers=auth_headers(),
            json={"pdbqt_text": SAMPLE_RECEPTOR_PDBQT},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        report = payload["report"]
        assert report["mode"] == "pdbqt"
        assert report["input_atoms"] == 5
        # pdbqt mode re-types records as-is: the HOH water is stripped,
        # the polar hydrogen is kept (charges kept, no Meeko step)
        assert report["waters_removed"] == 1
        assert report["hydrogens_removed"] == 0
        assert report["output_atoms"] == 4
        assert report["chains"] == ["A"]
        assert "A" in report["atom_types"] or "C" in report["atom_types"]
        # no sandbox paths leak into the payload
        assert "/home/" not in report["source"]
        assert "/tmp" not in report["output_pdbqt"]
        artifact = payload["artifact"]
        assert artifact["kind"] == "receptor-pdbqt"
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert b"ATOM" in dl.content

    def test_ligand_pdbqt_real_embedding(self, client):
        r = client.post(
            "/api/v1/docking/ligand/pdbqt",
            headers=auth_headers(),
            json={"smiles": "CCO", "name": "ethanol", "seed": 0},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["smiles"] == "CCO"
        assert payload["size_bytes"] > 100
        artifact = payload["artifact"]
        assert artifact["kind"] == "ligand-pdbqt"
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert b"ATOM" in dl.content and b"ROOT" in dl.content

    def test_ligand_pdbqt_invalid_smiles_422(self, client):
        r = client.post(
            "/api/v1/docking/ligand/pdbqt",
            headers=auth_headers(),
            json={"smiles": "not-a-smiles-((("},
        )
        assert r.status_code == 422


class TestDockingParsingScoring:

    def test_parse_vina_table(self, client):
        r = client.post(
            "/api/v1/docking/parse/vina",
            headers=auth_headers(),
            json={"text": SAMPLE_VINA_OUTPUT},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_poses"] == 3
        assert payload["poses"][0] == {
            "mode": 1, "affinity": -10.3, "rmsd_ub": 0.0, "rmsd_lb": 0.0
        }
        assert payload["poses"][1]["rmsd_ub"] == pytest.approx(2.301)

    def test_parse_vina_no_table_422(self, client):
        r = client.post(
            "/api/v1/docking/parse/vina",
            headers=auth_headers(),
            json={"text": "nothing here but words\n"},
        )
        assert r.status_code == 422

    def test_parse_dlg_cluster_table(self, client):
        r = client.post(
            "/api/v1/docking/parse/dlg",
            headers=auth_headers(),
            json={"text": SAMPLE_DLG},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["summary"]
        assert len(payload["clusters"]) == 2
        first = payload["clusters"][0]
        assert first["rank"] == 1
        assert first["lowest_energy"] == pytest.approx(-11.5)
        assert first["num_in_cluster"] == 7

    def test_consensus_breakdown_real_normalization(self, client):
        rows = [
            {"ligand": "A", "energy": -11.0, "size": 30},
            {"ligand": "B", "energy": -9.0, "size": 10},
        ]
        r = client.post(
            "/api/v1/docking/scoring/consensus",
            headers=auth_headers(),
            json={
                "rows": rows, "columns": ["energy", "size"],
                "higher_is_better": {"energy": False, "size": True},
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_rows"] == 2
        assert set(payload["rows"][0]) == {
            "energy_norm", "size_norm", "consensus"
        }
        # best on both evidences -> consensus 1.0, worst -> 0.0
        assert payload["rows"][0]["consensus"] == pytest.approx(1.0)
        assert payload["rows"][1]["consensus"] == pytest.approx(0.0)

    def test_consensus_unknown_column_422(self, client):
        r = client.post(
            "/api/v1/docking/scoring/consensus",
            headers=auth_headers(),
            json={"rows": [{"a": 1}], "columns": ["b"]},
        )
        assert r.status_code == 422

    def test_ligand_efficiency(self, client):
        r = client.post(
            "/api/v1/docking/scoring/ligand-efficiency",
            headers=auth_headers(),
            json={"affinity_kcal_mol": -8.5, "num_heavy_atoms": 17},
        )
        assert r.status_code == 200
        assert r.json()["ligand_efficiency"] == pytest.approx(0.5)

    def test_contacts_and_plif(self, client):
        ligand = [
            {"serial": 1, "name": "C1", "element": "C", "resname": "LIG",
             "chain": "A", "resseq": 1, "x": 0.0, "y": 0.0, "z": 0.0,
             "autodock_type": "C"},
            {"serial": 2, "name": "O1", "element": "O", "resname": "LIG",
             "chain": "A", "resseq": 1, "x": 1.0, "y": 0.0, "z": 0.0,
             "autodock_type": "OA"},
        ]
        receptor = [
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
        r = client.post(
            "/api/v1/docking/contacts",
            headers=auth_headers(),
            json={"ligand": ligand, "receptor": receptor, "cutoff": 5.0},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        # ASP 30 is within 5 A of both ligand atoms; PHE 31 (40 A away) is not
        assert payload["n_contacts"] == 4
        assert all(
            c["receptor_residue"] == "A:ASP30" for c in payload["contacts"]
        )
        assert payload["interaction_classes"] == [
            "salt_bridge", "hbond", "hydrophobic", "vdw"
        ]
        assert payload["plif"][0]["residue"] == "A:ASP30"
        assert payload["plif"][0]["total"] >= 1

    def test_pose_rmsd_symmetry_aware(self, client):
        ref = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]
        translated = [[2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [2.0, 1.5, 0.0]]
        r = client.post(
            "/api/v1/docking/pose-rmsd",
            headers=auth_headers(),
            json={"smiles": "CCO", "coords_ref": ref, "poses": [ref, translated]},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_atoms"] == 3 and payload["n_poses"] == 2
        assert payload["rmsd"][0] == pytest.approx(0.0)
        assert payload["rmsd"][1] == pytest.approx(2.0)

    def test_pose_rmsd_bad_smiles_422(self, client):
        r = client.post(
            "/api/v1/docking/pose-rmsd",
            headers=auth_headers(),
            json={
                "smiles": "(((",
                "coords_ref": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "poses": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            },
        )
        assert r.status_code == 422

    def test_dock_engine_boundary_422(self, client, monkeypatch):
        """The honest host boundary: bundled Windows PE + no wine."""
        from pharmaml.exceptions import DockingError

        # r297: the boundary is forced EXPLICITLY -- on the POSIX sandbox it
        # arose naturally (bundled PE + no wine), but the Windows runner
        # carries the bundled engines as NATIVE binaries, so engine_argv
        # succeeds there BY DESIGN and this route's honest-refusal contract
        # needs the refusal injected to be verified on every host.
        def _unavailable(name: str, engine_dir=None):
            raise DockingError(
                f"engine unavailable: bundled Windows PE without wine "
                f"({name!r} not launchable on this host)"
            )

        monkeypatch.setattr(
            "pharmaml.api.routes.docking.engine_argv", _unavailable
        )
        r = client.post(
            "/api/v1/docking/dock",
            headers=auth_headers(),
            json={
                "smiles": ["CCO", "CCC"],
                "receptor_pdbqt": SAMPLE_RECEPTOR_PDBQT,
            },
        )
        assert r.status_code == 422
        assert "docking engine unavailable" in r.json()["detail"]
        assert "Windows" in r.json()["detail"]


# ------------------------------------------------------------------- pkpd

class TestPkPdCalculators:

    def test_nca_summary(self, client):
        r = client.post(
            "/api/v1/pkpd/nca",
            headers=auth_headers(),
            json={
                "times": [0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
                "conc": [0.5, 5.0, 4.0, 3.0, 1.5, 0.5],
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["summary"]["Cmax"] == pytest.approx(5.0)
        assert payload["summary"]["Tmax"] == pytest.approx(1.0)
        assert payload["summary"]["AUC"] > 0.0
        assert payload["auc_trapezoid_linear"] > 0.0
        assert payload["auc_trapezoid_log"] is not None
        assert payload["terminal"]["n_points"] >= 3
        assert payload["terminal"]["half_life"] > 0.0

    def test_nca_non_ascending_times_422(self, client):
        r = client.post(
            "/api/v1/pkpd/nca",
            headers=auth_headers(),
            json={"times": [0.0, 2.0, 1.0], "conc": [1.0, 2.0, 3.0]},
        )
        assert r.status_code == 422

    def test_steady_state_metrics(self, client):
        from pharmaml.pkpd.compartments import bateman, superposition

        # grid starts at 0.5: the profile must be strictly positive
        # (Cmin drives the swing ratio), so the t=0 dose point is left out
        grid = np.arange(0.5, 12.5, 0.5)

        def unit(t):
            clamped = np.clip(np.asarray(t, dtype=float), 0.0, None)
            return np.asarray(bateman(10.0, 1.2, 0.25, clamped))

        ss = superposition(grid, unit, 6.0, 4)
        r = client.post(
            "/api/v1/pkpd/steady-state",
            headers=auth_headers(),
            json={
                "times": [float(t) for t in grid],
                "conc": [float(v) for v in ss],
                "tau": 6.0, "dose": 40.0, "lambda_z": 0.25,
            },
        )
        assert r.status_code == 200, r.text
        metrics = r.json()["metrics"]
        assert set(metrics) == {
            "AUC_tau", "Cavg", "Cmax_ss", "Cmin_ss", "Ctrough",
            "fluctuation", "swing", "CL_ss", "R_accum",
        }
        assert metrics["R_accum"] == pytest.approx(
            1.0 / (1.0 - math.exp(-0.25 * 6.0))
        )
        assert metrics["CL_ss"] == pytest.approx(40.0 / metrics["AUC_tau"])

    def test_iv_bolus(self, client):
        r = client.post(
            "/api/v1/pkpd/iv-bolus",
            headers=auth_headers(),
            json={"c0": 100.0, "ke": 0.25, "times": [0.0, 1.0, 4.0]},
        )
        assert r.status_code == 200
        values = r.json()["concentration"]
        assert values[0] == pytest.approx(100.0)
        assert values[1] == pytest.approx(100.0 * math.exp(-0.25))
        assert values[2] == pytest.approx(100.0 * math.exp(-1.0))

    def test_bateman_curve(self, client):
        times = [float(t) for t in range(11)]
        r = client.post(
            "/api/v1/pkpd/bateman",
            headers=auth_headers(),
            json={"dose_per_volume": 10.0, "ka": 1.0, "ke": 0.25, "times": times},
        )
        assert r.status_code == 200
        values = r.json()["concentration"]
        assert len(values) == 11
        assert values[0] == pytest.approx(0.0)
        assert max(values) > 4.0
        assert values[-1] < max(values)

    def test_superposition_accumulates(self, client):
        times = [float(t) for t in range(0, 49, 2)]
        r = client.post(
            "/api/v1/pkpd/superposition",
            headers=auth_headers(),
            json={
                "dose_per_volume": 10.0, "ka": 1.2, "ke": 0.25,
                "times": times, "tau": 6.0, "n_doses": 8,
            },
        )
        assert r.status_code == 200
        values = r.json()["concentration"]
        assert len(values) == len(times)
        # repeated dosing accumulates above the single-dose maximum
        assert max(values) > 6.0
        assert values[-1] > 1.0

    def test_pbpk_iv_and_oral(self, client):
        times = [float(t) for t in range(25)]
        r = client.post(
            "/api/v1/pkpd/pbpk",
            headers=auth_headers(),
            json={"times": times, "dose_mg": 100.0, "route": "iv"},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_points"] == 25
        columns = set(payload["rows"][0])
        assert {"time", "venous", "lung", "liver", "kidney", "rest",
                "C_plasma"} <= columns
        assert payload["rows"][0]["C_plasma"] > 0.0  # iv bolus in venous
        r = client.post(
            "/api/v1/pkpd/pbpk",
            headers=auth_headers(),
            json={"times": times, "dose_mg": 100.0, "route": "oral",
                  "ka": 1.0, "f": 0.85},
        )
        assert r.status_code == 200
        oral = r.json()["rows"]
        assert oral[0]["C_plasma"] == pytest.approx(0.0)  # nothing absorbed yet
        assert max(row["C_plasma"] for row in oral) > 0.0

    def test_pd_effect_table(self, client):
        r = client.post(
            "/api/v1/pkpd/pd",
            headers=auth_headers(),
            json={"concentrations": [0.1, 1.0, 10.0, 100.0],
                  "e0": 0.0, "emax": 100.0, "ec50": 10.0, "hill": 1.0},
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["n_points"] == 4
        assert payload["concentration"] == [0.1, 1.0, 10.0, 100.0]
        # sigmoid Emax at the EC50 is exactly half of the Emax
        assert payload["effect"][2] == pytest.approx(50.0)
        assert payload["effect"][3] > 90.0


class TestPkPdPopulation:

    def test_population_job_real_batches(self, client):
        times = [float(t) for t in range(25)]
        r = client.post(
            "/api/v1/pkpd/population",
            headers=auth_headers(),
            json={
                "theta": [10.0, 1.0, 0.25], "omega_diag": [0.09],
                "times": times, "n_subjects": 120, "seed": 3,
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "pkpd.population"
        assert job["workspace"] == "pkpd"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = _job_result(client, job["job_id"])
        assert result["n_subjects"] == 120
        assert result["n_times"] == 25
        assert len(result["summary"]["mean"]) == 25
        assert all(math.isfinite(v) for v in result["summary"]["mean"])
        assert len(result["subject_profiles"]) == 3
        # REAL per-batch telemetry: ceil(120 / 50) = 3 progress events
        progress = [
            e for e in _bus_events(client)
            if e["event_type"] == "job.progress" and e["job_id"] == job["job_id"]
        ]
        assert len(progress) == 3
        assert progress[-1]["payload"]["stage"] == "simulated 120/120 subjects"

    def test_population_cancel_never_failed(self, client):
        r = client.post(
            "/api/v1/pkpd/population",
            headers=auth_headers(),
            json={
                "theta": [10.0, 1.0, 0.25], "omega_diag": [0.09],
                "times": [float(t) for t in range(201)],
                "n_subjects": 20000, "seed": 1,
            },
        )
        assert r.status_code == 200
        job_id = r.json()["job"]["job_id"]
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            job = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            ).json()
            if job["state"] == "RUNNING":
                break
            time.sleep(0.01)
        cancel = client.post(
            f"/api/v1/jobs/{job_id}/cancel", headers=auth_headers()
        )
        assert cancel.status_code == 200
        final = _wait_job(client, job_id)
        assert final["state"] == "CANCELLED", final
        assert final["cancellation"] == "cooperative"

    def test_population_bad_theta_422(self, client):
        r = client.post(
            "/api/v1/pkpd/population",
            headers=auth_headers(),
            json={"theta": [1.0, 2.0], "omega_diag": [0.09],
                  "times": [0.0, 1.0], "n_subjects": 10},
        )
        assert r.status_code == 422


# --------------------------------------------------------------------- md

class TestMdAnalysis:

    def test_synthetic_metric_bundle(self, client):
        r = client.post(
            "/api/v1/md/synthetic",
            headers=auth_headers(),
            json={"n_frames": 40, "seed": 5, "n_residues": 10},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_frames"] == 40
        assert payload["n_atoms"] == 30
        assert len(payload["rmsd"]) == 40
        assert payload["rmsd"][0] == pytest.approx(0.0)
        assert len(payload["rmsf"]) == 30
        assert len(payload["radius_of_gyration"]) == 40
        assert len(payload["native_contact_fraction"]) == 40
        assert all(0.0 <= v <= 1.0 for v in payload["native_contact_fraction"])
        assert 0.0 <= payload["contact_map"]["strong_pair_fraction"] <= 1.0
        assert len(payload["pca"]["explained_variance_ratio"]) == 2
        assert len(payload["pca"]["projections"]) == 40

    def test_analyze_inline_coords(self, client):
        from pharmaml.md.trajectory import synthetic_trajectory

        traj = synthetic_trajectory(n_frames=30, seed=11)
        r = client.post(
            "/api/v1/md/analyze",
            headers=auth_headers(),
            json={"coords": traj.coords.tolist()},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_frames"] == 30
        assert len(payload["rmsd"]) == 30
        assert len(payload["rmsf"]) == traj.n_atoms
        assert payload["pca"]["method"] == "pca"

    def test_analyze_bad_shape_422(self, client):
        r = client.post(
            "/api/v1/md/analyze",
            headers=auth_headers(),
            json={"coords": [[[0.0, 0.0], [1.0, 1.0]]]},
        )
        assert r.status_code == 422

    def test_xvg_parse_and_stats(self, client):
        text = "# gmx rms\n@ s0 legend \"RMSD\"\n0.0 0.0\n1.0 0.5\n2.0 1.0\n"
        r = client.post(
            "/api/v1/md/xvg", headers=auth_headers(),
            json={"text": text, "moving_average_window": 3},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_points"] == 3
        assert payload["time"] == [0.0, 1.0, 2.0]
        assert payload["values"] == [0.0, 0.5, 1.0]
        assert payload["stats"]["mean"] == pytest.approx(0.5)
        assert payload["stats"]["cv"] is not None
        assert len(payload["moving_average"]) == 3

    def test_xvg_no_data_422(self, client):
        r = client.post(
            "/api/v1/md/xvg", headers=auth_headers(),
            json={"text": "# only comments\n@ and decorators\n"},
        )
        assert r.status_code == 422

    def test_enm_fluctuations(self, client):
        rng = np.random.default_rng(7)
        coords = rng.normal(scale=3.0, size=(12, 3)).tolist()
        r = client.post(
            "/api/v1/md/enm", headers=auth_headers(),
            json={"coords": coords, "cutoff": 8.0},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_atoms"] == 12
        assert len(payload["fluctuations"]) == 12
        assert all(math.isfinite(v) and v >= 0.0 for v in payload["fluctuations"])

    def test_pca_projections(self, client):
        rng = np.random.default_rng(9)
        coords = (rng.normal(size=(30, 9, 3)) * 2.0).tolist()
        r = client.post(
            "/api/v1/md/pca", headers=auth_headers(),
            json={"coords": coords, "n_components": 2},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_components"] == 2
        assert len(payload["projections"]) == 30
        assert all(len(row) == 2 for row in payload["projections"])
        assert sum(payload["explained_variance_ratio"]) <= 1.0 + 1e-9

    def test_study_job_real_figures(self, client):
        r = client.post(
            "/api/v1/md/study",
            headers=auth_headers(),
            json={"n_frames": 60, "seed": 3, "n_residues": 10,
                  "title": "API study"},
        )
        assert r.status_code == 200
        job = r.json()["job"]
        assert job["operation"] == "md.study"
        assert job["workspace"] == "molecular_dynamics"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = _job_result(client, job["job_id"])
        assert result["metrics"]["n_frames"] == 60
        assert len(result["artifacts"]) == 2
        for artifact in result["artifacts"]:
            assert artifact["kind"] == "figure"
            dl = client.get(
                f"/api/v1/artifacts/{artifact['artifact_id']}/download",
                headers=auth_headers(),
            )
            assert dl.status_code == 200
            assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"
        progress = [
            e for e in _bus_events(client)
            if e["event_type"] == "job.progress" and e["job_id"] == job["job_id"]
        ]
        assert len(progress) >= 4
        assert progress[-1]["payload"]["progress"] == 95


# ----------------------------------------------------------- visualization

class TestVisualization:

    def test_catalog(self, client):
        r = client.get(
            "/api/v1/visualization/catalog", headers=auth_headers()
        )
        assert r.status_code == 200
        payload = r.json()
        assert len(payload["families"]) == 5
        assert payload["palette"]
        assert payload["export_formats"] == ["PNG", "SVG", "TIFF", "JPEG"]

    def test_stat_box_with_outliers(self, client):
        values = [1.0] * 4 + [2.0] * 4 + [100.0]
        r = client.post(
            "/api/v1/visualization/stat",
            headers=auth_headers(),
            json={"kind": "box", "values": values},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_points"] == 9
        assert payload["stats"]["outliers"] == [100.0]
        assert payload["artifact"]["kind"] == "figure"
        dl = client.get(
            f"/api/v1/artifacts/{payload['artifact']['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_stat_box_clean_data_no_fliers(self, client):
        # the matplotlib 3.10.9 empty-fliers fix: zero outliers renders
        values = [float(i) for i in range(40)]
        r = client.post(
            "/api/v1/visualization/stat",
            headers=auth_headers(),
            json={"kind": "box", "values": values},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["stats"]["outliers"] == []
        assert payload["artifact"]["kind"] == "figure"

    def test_stat_regression_slope(self, client):
        x = [float(i) for i in range(20)]
        y = [2.0 * i + 1.0 for i in range(20)]
        r = client.post(
            "/api/v1/visualization/stat",
            headers=auth_headers(),
            json={"kind": "regression", "x": x, "y": y, "n_boot": 0},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["stats"]["slope"] == pytest.approx(2.0)
        assert payload["stats"]["intercept"] == pytest.approx(1.0)
        assert payload["stats"]["slope_ci"] is None  # n_boot=0, honestly

    def test_stat_missing_values_422(self, client):
        r = client.post(
            "/api/v1/visualization/stat",
            headers=auth_headers(),
            json={"kind": "histogram"},
        )
        assert r.status_code == 422

    def test_concentration_profile_figure(self, client):
        times = [float(t) for t in range(13)]
        conc = [0.0, 5.0, 4.2, 3.5, 3.0, 2.2, 1.6, 1.1, 0.8, 0.5, 0.3, 0.2, 0.1]
        r = client.post(
            "/api/v1/visualization/concentration-profile",
            headers=auth_headers(),
            json={"times": times, "conc": conc, "auc": 22.0},
        )
        assert r.status_code == 200, r.text
        artifact = r.json()["artifact"]
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"
        assert dl.headers["ETag"] == f'"{artifact["sha256"]}"'

    def test_concentration_profile_length_mismatch_422(self, client):
        r = client.post(
            "/api/v1/visualization/concentration-profile",
            headers=auth_headers(),
            json={"times": [0.0, 1.0], "conc": [1.0]},
        )
        assert r.status_code == 422

    def test_dose_accumulation_figure(self, client):
        r = client.post(
            "/api/v1/visualization/dose-accumulation",
            headers=auth_headers(),
            json={"dose": 100.0, "tau": 6.0, "ke": 0.25, "vd": 50.0,
                  "n_doses": 6},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["tau"] == 6.0
        dl = client.get(
            f"/api/v1/artifacts/{payload['artifact']['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_dose_accumulation_both_halflife_and_ke_422(self, client):
        r = client.post(
            "/api/v1/visualization/dose-accumulation",
            headers=auth_headers(),
            json={"dose": 100.0, "tau": 6.0, "ke": 0.25, "t_half": 2.77,
                  "vd": 50.0},
        )
        assert r.status_code == 422

    def test_md_series_two_panels(self, client):
        rmsd = "# t rmsd\n0.0 0.0\n1.0 0.4\n2.0 0.9\n"
        rg = "# t rg\n0.0 1.1\n1.0 1.2\n2.0 1.25\n"
        r = client.post(
            "/api/v1/visualization/md-series",
            headers=auth_headers(),
            json={"series": {"rmsd": rmsd, "rg": rg}, "title": "MD panels"},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_series"] == 2
        assert payload["series"]["rmsd"]["stats"]["mean"] == pytest.approx(
            (0.0 + 0.4 + 0.9) / 3.0
        )
        dl = client.get(
            f"/api/v1/artifacts/{payload['artifact']['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_table_xlsx_artifact(self, client):
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        r = client.post(
            "/api/v1/visualization/table",
            headers=auth_headers(),
            json={"rows": rows, "format": "xlsx"},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["rows"] == 2
        artifact = payload["artifact"]
        assert artifact["kind"] == "table"
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:2] == b"PK"  # xlsx = zip magic
        assert "spreadsheetml" in dl.headers["Content-Type"]
        assert dl.headers["X-Artifact-Kind"] == "table"

    def test_report_job_selfcontained_html(self, client):
        times = [float(t) for t in range(9)]
        conc = [0.0, 5.0, 4.0, 3.0, 2.0, 1.2, 0.8, 0.4, 0.2]
        r = client.post(
            "/api/v1/visualization/report",
            headers=auth_headers(),
            json={
                "title": "API report",
                "metrics": {"AUC": 123.4, "Cmax": 5.0},
                "tables": [{"heading": "demo", "rows": [{"a": 1}, {"a": 2}]}],
                "concentration_profile": {
                    "times": times, "conc": conc,
                    "caption": "profile panel",
                },
            },
        )
        assert r.status_code == 200
        job = r.json()["job"]
        assert job["operation"] == "visualization.report"
        assert job["workspace"] == "imagelab"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = _job_result(client, job["job_id"])
        assert result["figures"] == 1
        artifact = result["artifact"]
        assert artifact["kind"] == "report"
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        html = dl.content
        assert b"API report" in html
        assert b"AUC" in html
        assert b"demo" in html
        assert b"<img" in html  # the figure is embedded, self-contained
        progress = [
            e for e in _bus_events(client)
            if e["event_type"] == "file.progress" and e["job_id"] == job["job_id"]
        ]
        assert len(progress) >= 2
        assert progress[-1]["payload"]["stage"] == "report artifact registered"


# --------------------------------------------------- applicability domain

class TestApplicabilityDomain:

    def test_catalog(self, client):
        r = client.get(
            "/api/v1/applicability-domain/catalog", headers=auth_headers()
        )
        assert r.status_code == 200
        payload = r.json()
        assert len(payload["entry_points"]) == 3
        assert payload["result_columns"] == [
            "name", "predicted", "prediction_sd", "knn_distance",
            "inside_ad", "low_uncertainty", "reliable",
        ]

    def test_knn_inside_outside(self, client):
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=0.0, scale=0.5, size=(15, 2))
        cluster_b = rng.normal(loc=8.0, scale=0.5, size=(15, 2))
        train = np.vstack([cluster_a, cluster_b]).tolist()
        test = [
            [0.1, 0.2], [8.2, 7.9], [-0.3, 0.4],  # inside the clusters
            [50.0, 50.0], [60.0, -60.0],          # far outside
        ]
        r = client.post(
            "/api/v1/applicability-domain/knn",
            headers=auth_headers(),
            json={"x_train": train, "x_test": test, "k": 3},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["n_test"] == 5
        assert payload["n_inside"] == 3
        assert payload["distance_threshold"] > 0.0
        assert len(payload["frame"]) == 5
        flags = [row["inside_ad"] for row in payload["frame"]]
        assert flags == [True, True, True, False, False]

    def test_knn_feature_mismatch_422(self, client):
        r = client.post(
            "/api/v1/applicability-domain/knn",
            headers=auth_headers(),
            json={"x_train": [[0.0, 1.0]] * 10, "x_test": [[0.0, 1.0, 2.0]]},
        )
        assert r.status_code == 422

    def _prepare_model(self, client):
        ds = _upload(client, data=_csv_bytes(rows=40))
        model = _train(client, ds["dataset_id"])
        return ds, model

    def test_analyze_job_bootstrap_telemetry(self, client):
        _ds, model = self._prepare_model(client)
        x_test = [[0.5, 0.5], [1.0, 0.25], [4.0, 2.0]]
        r = client.post(
            "/api/v1/applicability-domain/analyze",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"], "x_test": x_test,
                "k": 3, "n_models": 5,
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "ad.analyze"
        assert job["workspace"] == "machine-learning"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = _job_result(client, job["job_id"])
        analysis = result["analysis"]
        assert analysis["n_models"] == 5
        assert analysis["n_test"] == 3
        assert analysis["uncertainty_kind"] is not None
        assert len(analysis["frame"]) == 3
        assert all(row["predicted"] is not None for row in analysis["frame"])
        # REAL per-member bootstrap telemetry: 5 training.progress events
        training = [
            e for e in _bus_events(client)
            if e["event_type"] == "training.progress"
            and e["job_id"] == job["job_id"]
        ]
        assert len(training) == 5
        assert training[-1]["payload"]["stage"] == "bootstrap member 5/5"
        assert training[-1]["payload"]["progress"] == 99

    def test_analyze_sync_uncertainty_off(self, client):
        _ds, model = self._prepare_model(client)
        r = client.post(
            "/api/v1/applicability-domain/analyze",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_test": [[0.5, 0.5]], "k": 3, "uncertainty": False,
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        analysis = r.json()["analysis"]
        assert analysis["n_models"] is None
        assert analysis["uncertainty_kind"] is None
        # the distance half still ran: threshold and inside counts exist
        assert analysis["distance_threshold"] > 0.0
        row = analysis["frame"][0]
        assert row["knn_distance"] is not None
        assert row["predicted"] is None  # no bootstrap half, honestly

    def test_analyze_with_figure(self, client):
        _ds, model = self._prepare_model(client)
        r = client.post(
            "/api/v1/applicability-domain/analyze",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"], "x_test": [[0.5, 0.5], [3.0, 1.0]],
                "k": 3, "n_models": 4, "figure": True, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        artifact = r.json()["artifact"]
        assert artifact["kind"] == "figure"
        dl = client.get(
            f"/api/v1/artifacts/{artifact['artifact_id']}/download",
            headers=auth_headers(),
        )
        assert dl.status_code == 200
        assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"

    def test_analyze_unknown_model_404(self, client):
        r = client.post(
            "/api/v1/applicability-domain/analyze",
            headers=auth_headers(),
            json={"model_id": "mdl-bogus", "x_test": [[0.5, 0.5]]},
        )
        assert r.status_code == 404

    def test_analyze_feature_mismatch_422(self, client):
        _ds, model = self._prepare_model(client)
        r = client.post(
            "/api/v1/applicability-domain/analyze",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "x_test": [[0.5, 0.5, 0.5]], "k": 3,
            },
        )
        assert r.status_code == 422

    def test_domain_box(self, client):
        metadata = {
            "feature_ranges": {
                "x1": {"min": 0.0, "max": 10.0},
                "x2": {"min": 0.0, "max": 5.0},
            }
        }
        rows = [{"x1": 5.0, "x2": 2.0}, {"x1": 50.0, "x2": 2.0},
                {"x1": -1.0, "x2": 2.0}]
        r = client.post(
            "/api/v1/applicability-domain/domain-box",
            headers=auth_headers(),
            json={"metadata": metadata, "rows": rows},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["checked"] is True
        assert payload["n_rows"] == 3
        assert payload["n_out"] == 2
        assert payload["fraction_out"] == pytest.approx(2.0 / 3.0)
        assert payload["above"]["x1"] == 1
        assert payload["below"]["x1"] == 1
        assert payload["out"] == [False, True, True]

    def test_domain_box_no_ranges_honest_false(self, client):
        r = client.post(
            "/api/v1/applicability-domain/domain-box",
            headers=auth_headers(),
            json={"metadata": {"model": "no box here"}, "rows": [{"x1": 1.0}]},
        )
        assert r.status_code == 200
        payload = r.json()
        assert payload["checked"] is False
        assert "feature_ranges" in payload["reason"]
