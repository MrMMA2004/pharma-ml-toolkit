"""The part11 Datasets/ML workflow's API and artifact surface.

Test-alongside-every-change, exactly like the opening suite: every
route here runs the REAL backend (load_dataset / profile_frame /
split_table / train_model / list_models / save_model / write_table /
the §17 figure export stack) and the file transfer contracts are
verified byte-for-byte:

* the upload contract (multipart, extension + size guards, sync and
  job modes, JSON-safe views with NaN-as-null previews);
* the download contract (exact Content-Length, Content-Type,
  Content-Disposition, strong ETag = sha256, X-Artifact-SHA256,
  HEAD support, magic bytes per format);
* the model bundle round-trip (download → load_model → same estimator
  metadata + identical predictions — the save/reload parity clause);
* the session-scoped stores (fresh per app, bounded eviction).
"""

from __future__ import annotations

import hashlib
import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pharmaml.api import create_app
from pharmaml.api.stores import ArtifactStore

TEST_TOKEN = "test-session-token-0123456789"


@pytest.fixture()
def client():
    app = create_app(token=TEST_TOKEN)
    with TestClient(app, base_url="http://127.0.0.1") as tc:
        yield tc


def auth_headers():
    return {"X-Session-Token": TEST_TOKEN}


def _csv_bytes(rows: int = 24, blank_cell_at: int | None = None) -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(rows):
        x1 = "" if i == blank_cell_at else round(0.1 * i + 0.5, 3)
        buf.write(f"{x1},{(i * 7) % 5 * 0.25},{i % 2}\n")
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
    if mode == "sync":
        return r.json()["dataset"]
    job = r.json()["job"]
    final = _wait_job(client, job["job_id"])
    assert final["state"] == "COMPLETED", final
    # the job's result IS the dataset view (the sync payload analog)
    return client.get(
        f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
    ).json()["result"]


def _train(client, dataset_id: str, model: str = "random_forest",
           mode: str = "sync", **extra) -> dict:
    body = {
        "dataset_id": dataset_id,
        "model": model,
        "tune": False,
        "n_jobs": -1,  # the Phase-Eight passthrough clause
        "mode": mode,
        **extra,
    }
    r = client.post("/api/v1/ml/train", headers=auth_headers(), json=body)
    assert r.status_code == 200, r.text
    if mode == "sync":
        return r.json()["model"]
    job = r.json()["job"]
    final = _wait_job(client, job["job_id"])
    assert final["state"] == "COMPLETED", final
    result = client.get(
        f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
    ).json()["result"]
    return result["model"]


def _wait_job(client, job_id: str, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(
            f"/api/v1/jobs/{job_id}", headers=auth_headers()
        ).json()
        if body["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.1)
    pytest.fail(f"job {job_id} timed out")


def _figure(client, **body) -> dict:
    payload = {"kind": "scatter", "x": [1.0, 2.0, 3.0, 4.0],
               "y": [1.0, 4.0, 9.0, 16.0]}
    payload.update(body)
    r = client.post(
        "/api/v1/artifacts/figure", headers=auth_headers(), json=payload
    )
    assert r.status_code == 200, r.text
    return r.json()["artifact"]


# ---------------------------------------------------------------- upload


class TestDatasetUpload:
    def test_sync_upload_registers_and_reports(self, client):
        ds = _upload(client)
        assert ds["dataset_id"].startswith("ds-")
        assert ds["shape"] == [24, 3]
        assert ds["target"] == "y"
        assert ds["task"] == "classification"
        assert ds["classes"] == [0, 1]
        assert ds["class_counts"] == {"0": 12, "1": 12}
        assert ds["feature_names"] == ["x1", "x2"]
        assert ds["n_features"] == 2
        assert ds["dropped_columns"] == []
        assert ds["source_filename"] == "toy.csv"
        assert ds["size_bytes"] > 0
        assert len(ds["preview"]["rows"]) == 10
        assert ds["preview"]["n_rows"] == 24
        assert ds["summary"]["n_rows"] == 24

    def test_xlsx_upload(self, client):
        frame = pd.DataFrame(
            {"x1": [float(i) for i in range(24)],
             "x2": [float(i % 5) for i in range(24)],
             "y": [i % 2 for i in range(24)]}
        )
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="data")
        ds = _upload(client, data=buffer.getvalue(), name="toy.xlsx")
        assert ds["shape"] == [24, 3]
        assert ds["task"] == "classification"

    def test_job_mode_upload_completes(self, client):
        ds = _upload(client, mode="job")
        assert ds["dataset_id"].startswith("ds-")
        listed = client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"]
        assert any(d["dataset_id"] == ds["dataset_id"] for d in listed)

    def test_unsupported_extension_is_415(self, client):
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": ("bad.exe", b"MZ", "application/octet-stream")},
            data={"target": "y"},
        )
        assert r.status_code == 415
        assert "allowed" in r.json()["detail"]

    def test_oversize_upload_is_413(self, client, monkeypatch):
        from pharmaml.api.routes import datasets as datasets_route

        monkeypatch.setattr(datasets_route, "MAX_UPLOAD_BYTES", 16)
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": ("big.csv", b"a,b,y\n" + b"1,2,0\n" * 10, "text/csv")},
            data={"target": "y"},
        )
        assert r.status_code == 413

    def test_empty_upload_is_422(self, client):
        r = client.post(
            "/api/v1/datasets",
            headers=auth_headers(),
            files={"file": ("empty.csv", b"", "text/csv")},
            data={"target": "y"},
        )
        assert r.status_code == 422

    def test_upload_requires_token(self, client):
        r = client.post(
            "/api/v1/datasets",
            files={"file": ("toy.csv", _csv_bytes(), "text/csv")},
            data={"target": "y"},
        )
        assert r.status_code == 401


# ------------------------------------------------------------- dataset views


class TestDatasetViews:
    def test_list_and_get(self, client):
        ds = _upload(client)
        listed = client.get(
            "/api/v1/datasets", headers=auth_headers()
        ).json()["datasets"]
        assert any(d["dataset_id"] == ds["dataset_id"] for d in listed)
        got = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}",
            headers=auth_headers(),
        )
        assert got.status_code == 200
        assert got.json()["dataset_id"] == ds["dataset_id"]
        assert got.json()["summary"]["n_rows"] == 24

    def test_preview_limit_and_clamp(self, client):
        ds = _upload(client)
        r = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/preview",
            headers=auth_headers(), params={"limit": 3},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["returned"] == 3
        assert body["n_rows"] == 24
        assert body["columns"] == ["x1", "x2", "y"]
        assert client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/preview",
            headers=auth_headers(), params={"limit": 0},
        ).status_code == 422

    def test_preview_nan_becomes_null(self, client):
        ds = _upload(client, data=_csv_bytes(blank_cell_at=3))
        body = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/preview",
            headers=auth_headers(), params={"limit": 5},
        ).json()
        assert body["rows"][3][0] is None
        assert body["rows"][0][0] is not None

    def test_delete_then_404(self, client):
        ds = _upload(client)
        r = client.delete(
            f"/api/v1/datasets/{ds['dataset_id']}", headers=auth_headers()
        )
        assert r.status_code == 200
        assert r.json()["evicted"] is True
        assert client.get(
            f"/api/v1/datasets/{ds['dataset_id']}",
            headers=auth_headers(),
        ).status_code == 404

    def test_unknown_dataset_404(self, client):
        assert client.get(
            "/api/v1/datasets/ds-doesnotexist", headers=auth_headers()
        ).status_code == 404


class TestDatasetColumn:
    """§11.1's chemistry-stage bridge: bounded column extraction."""

    def test_column_values_and_shape_fields(self, client):
        ds = _upload(client)
        r = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/x1",
            headers=auth_headers(),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["column"] == "x1"
        assert body["n_rows"] == 24
        assert body["returned"] == 24
        assert body["truncated"] is False
        assert body["dtype"].startswith("float")
        assert len(body["values"]) == 24
        assert isinstance(body["values"][0], float)

    def test_column_limit_and_truncated_flag(self, client):
        ds = _upload(client)
        r = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/x1",
            headers=auth_headers(), params={"limit": 5},
        )
        assert r.status_code == 200
        body = r.json()
        # §17.3: the bound is explicit, never a silent truncation
        assert body["returned"] == 5
        assert body["n_rows"] == 24
        assert body["truncated"] is True
        assert body["values"] == body["values"][:5]

    def test_column_nan_becomes_null(self, client):
        ds = _upload(client, data=_csv_bytes(blank_cell_at=3))
        body = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/x1",
            headers=auth_headers(),
        ).json()
        assert body["values"][3] is None
        assert body["values"][0] is not None

    def test_column_unknown_dataset_404(self, client):
        assert client.get(
            "/api/v1/datasets/ds-doesnotexist/column/x1",
            headers=auth_headers(),
        ).status_code == 404

    def test_column_unknown_column_404(self, client):
        ds = _upload(client)
        assert client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/nope",
            headers=auth_headers(),
        ).status_code == 404

    def test_column_bad_limit_422(self, client):
        ds = _upload(client)
        assert client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/x1",
            headers=auth_headers(), params={"limit": 0},
        ).status_code == 422
        assert client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/x1",
            headers=auth_headers(), params={"limit": 501},
        ).status_code == 422

    def test_column_cap_mirrors_the_chemistry_engine(self):
        # the bound is the chemistry request models' own max_length —
        # one source per engine limit, drift-pinned here
        import re
        from pathlib import Path

        chemistry = (
            Path(__file__).resolve().parents[2]
            / "src" / "pharmaml" / "api" / "routes" / "chemistry.py"
        ).read_text(encoding="utf-8")
        caps = re.findall(
            r"smiles: list\[str\] = Field\(.*?max_length=(\d+)", chemistry
        )
        assert caps and all(c == "500" for c in caps), caps
        from pharmaml.api.routes.datasets import CHEMISTRY_MAX_SMILES
        assert CHEMISTRY_MAX_SMILES == 500

    def test_column_feeds_the_chemistry_descriptors_route(self, client):
        """The §11.1 chain end-to-end: upload → column → descriptors."""
        buf = io.StringIO()
        buf.write("smiles,y\n")
        for smi, y in [("CCO", "0"), ("CCC", "1"), ("CCN", "0")]:
            buf.write(f"{smi},{y}\n")
        ds = _upload(
            client,
            data=buf.getvalue().encode(),
            name="tiny.csv", target="y",
        )
        column = client.get(
            f"/api/v1/datasets/{ds['dataset_id']}/column/smiles",
            headers=auth_headers(),
        ).json()
        assert column["values"] == ["CCO", "CCC", "CCN"]
        r = client.post(
            "/api/v1/chemistry/descriptors",
            headers=auth_headers(),
            json={
                "smiles": column["values"],
                "descriptor_set": "morgan",
                "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        result = r.json()["result"]
        assert result["shape"][0] == 3
        assert result["rows"][0][0] == "CCO"


# ------------------------------------------------------ profile / split


class TestDatasetProfile:
    def test_sync_profile(self, client):
        ds = _upload(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/profile",
            headers=auth_headers(), json={"mode": "sync"},
        )
        assert r.status_code == 200
        profile = r.json()["profile"]
        assert profile["n"] == 24
        assert profile["n_cells"] == 72
        assert set(profile["variables"]) == {"x1", "x2", "y"}

    def test_job_profile(self, client):
        ds = _upload(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/profile",
            headers=auth_headers(), json={"mode": "job"},
        )
        assert r.status_code == 200
        job = r.json()["job"]
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED"
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["profile"]["n"] == 24


class TestDatasetSplit:
    def test_split_shapes_and_balance(self, client):
        ds = _upload(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(), json={"test_size": 0.2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["target"] == "y"
        assert body["stratified"] is True
        sides = body["sides"]
        assert sides["train"]["rows"] == 19
        assert sides["test"]["rows"] == 5
        assert sides["train"]["target_balance"] is not None
        assert sides["test"]["target_balance"] is not None

    def test_split_with_val(self, client):
        ds = _upload(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.2, "val_size": 0.25},
        )
        assert r.status_code == 200
        sides = r.json()["sides"]
        assert set(sides) == {"train", "val", "test"}
        assert sides["train"]["rows"] + sides["val"]["rows"] + \
            sides["test"]["rows"] == 24

    def test_split_unknown_dataset_404(self, client):
        r = client.post(
            "/api/v1/datasets/ds-doesnotexist/split",
            headers=auth_headers(), json={"test_size": 0.2},
        )
        assert r.status_code == 404

    def test_split_stratification_boundary_is_422_not_500(self, client):
        # the live r281 E2E caught this: sklearn's "test fold smaller
        # than the class count" surfaced as a raw 500 — now a clean 422
        buf = io.StringIO()
        buf.write("x1,y\n")
        for i in range(12):
            buf.write(f"{i % 12 * 0.25},{i % 4}\n")
        ds = _upload(client, data=buf.getvalue().encode(),
                     name="tiny-strat.csv", target="y")
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "stratify": True, "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "classes" in r.json()["detail"]

    def test_split_bad_fraction_422(self, client):
        ds = _upload(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(), json={"test_size": 1.5},
        )
        assert r.status_code == 422


# ------------------------------------------------- DS-1: the strategy lane


def _scaffold_csv() -> bytes:
    """12 rows in two scaffold families (the DS-1 fail-first fixture).

    8 ring-family SMILES (aspirin/phenol → Murcko ``c1ccccc1``) and 4
    acyclic SMILES (ethanol → the empty scaffold key): a random split
    would happily scatter the ring family across both sides — the
    scaffold strategy must keep each family wholly on one side.
    """
    buf = io.StringIO()
    buf.write("smiles,grp,x,y\n")
    for i in range(8):
        smi = "CC(=O)Oc1ccccc1C(=O)O" if i % 2 == 0 else "c1ccccc1O"
        buf.write(f"{smi},{'A' if i < 4 else 'B'},{i},{i % 2}\n")
    for i in range(4):
        buf.write(f"CCO,C,{8 + i},{i % 2}\n")
    return buf.getvalue().encode()


def _scaffold_csv_val() -> bytes:
    """16 rows in THREE scaffold families (the val-carve fixture).

    The two-family fixture above cannot answer a val request: the
    first carve leaves a single-group train remainder and the engine
    honestly refuses the second carve. Three families (benzene,
    pyridine, acyclic) give the val stage a two-group remainder to
    carve honestly.
    """
    buf = io.StringIO()
    buf.write("smiles,grp,x,y\n")
    families = [
        ("CC(=O)Oc1ccccc1C(=O)O", 6, "A"),
        ("c1ccncc1O", 5, "B"),
        ("CCO", 5, "C"),
    ]
    i = 0
    for smi, count, grp in families:
        for _ in range(count):
            buf.write(f"{smi},{grp},{i},{i % 2}\n")
            i += 1
    return buf.getvalue().encode()


class TestDatasetSplitStrategy:
    """DS-1 — the ml/splits engine vocabulary behind the split route."""

    def _scaffold_ds(self, client) -> dict:
        return _upload(client, data=_scaffold_csv(),
                       name="scaffold-fixture.csv", target="y")

    def test_scaffold_split_keeps_family_whole(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "scaffold", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategy"] == "scaffold"
        assert body["smiles_column"] == "smiles"  # the auto-pick
        assert body["target"] == "y"  # the record-fallback (no explicit target)
        sides = body["sides"]
        assert sum(s["rows"] for s in sides.values()) == 12
        census = body["group_census"]
        assert census is not None
        # the leakage verdict: no scaffold key on two sides
        assert census["shared_groups"] == []
        # the ring family ('c1ccccc1', 8 rows) lands wholly on ONE side
        ring_counts = {
            name: dict(c.get("top_groups", [])).get("c1ccccc1", 0)
            for name, c in census.items()
            if isinstance(c, dict)
        }
        nonzero = [n for n, c in ring_counts.items() if c > 0]
        assert nonzero and len(nonzero) == 1
        assert ring_counts[nonzero[0]] == 8

    def test_scaffold_split_with_val_stays_whole(self, client):
        ds = _upload(client, data=_scaffold_csv_val(),
                     name="scaffold-val-fixture.csv", target="y")
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "val_size": 0.25,
                  "strategy": "scaffold", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body["sides"]) == {"train", "val", "test"}
        assert body["group_census"]["shared_groups"] == []
        assert sum(s["rows"] for s in body["sides"].values()) == 16

    def test_val_carve_on_single_group_remainder_is_422(self, client):
        # two families only: the first carve leaves a single-group
        # remainder and the engine REFUSES the group-honest val carve —
        # the honest answer, never a silently degraded val side
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "val_size": 0.25,
                  "strategy": "scaffold", "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "one group/scaffold" in r.json()["detail"]

    def test_group_split_census(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "group",
                  "group_column": "grp", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategy"] == "group"
        assert body["group_column"] == "grp"
        assert body["group_census"]["shared_groups"] == []
        assert sum(s["rows"] for s in body["sides"].values()) == 12

    def test_temporal_split_is_positional(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "temporal", "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategy"] == "temporal"
        assert body["sides"]["train"]["rows"] == 9
        assert body["sides"]["test"]["rows"] == 3
        # order-as-given: the test side IS the last three rows (x = 9, 10, 11)
        test_x = [row[2] for row in body["sides"]["test"]["preview"]["rows"]]
        assert test_x == [9, 10, 11]

    def test_kennard_stone_split(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "kennard_stone",
                  "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["strategy"] == "kennard_stone"
        assert sum(s["rows"] for s in body["sides"].values()) == 12

    def test_scaffold_split_job_mode(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "scaffold", "mode": "job"},
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["strategy"] == "scaffold"
        assert result["group_census"]["shared_groups"] == []

    def test_unknown_strategy_is_422(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "nope", "mode": "sync"},
        )
        assert r.status_code == 422, r.text

    def test_group_strategy_requires_group_column_422(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "group", "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "explicit group_column" in r.json()["detail"]

    def test_group_column_not_found_is_422(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "group",
                  "group_column": "nope", "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "not found" in r.json()["detail"]

    def test_smiles_column_not_found_is_422(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "scaffold",
                  "smiles_column": "nope", "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "not found" in r.json()["detail"]

    def test_kennard_stone_without_numeric_columns_is_422(self, client):
        buf = io.StringIO()
        buf.write("smiles,grp,y\n")
        for i in range(12):
            buf.write(f"CCO,{'A' if i % 2 else 'B'},{i % 2}\n")
        ds = _upload(client, data=buf.getvalue().encode(),
                     name="strings-only.csv", target="y")
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.25, "strategy": "kennard_stone",
                  "mode": "sync"},
        )
        assert r.status_code == 422, r.text
        assert "numeric feature column" in r.json()["detail"]

    def test_legacy_split_response_shape_unchanged_plus_nulls(self, client):
        ds = self._scaffold_ds(client)
        r = client.post(
            f"/api/v1/datasets/{ds['dataset_id']}/split",
            headers=auth_headers(),
            json={"test_size": 0.2, "mode": "sync"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # the legacy lane answers exactly as before plus explicit nulls
        assert body["strategy"] is None
        assert body["group_census"] is None
        assert body["stratified"] is True
        assert set(body["sides"]) == {"train", "test"}


# ------------------------------------------- ML-3: the train-lane split section


class TestTrainSplitStrategy:
    """ML-3 — the ml/splits engine vocabulary behind POST /ml/train.

    Same fail-first fixture as DS-1 (12 rows, two scaffold families): a
    carve that ignored the strategy would scatter the 8-row ring family
    across both sides — the scaffold carve must keep each family whole,
    and the fit must see the carved train side only.
    """

    def _train_split(self, client, split: dict, tune: bool = False):
        ds = _upload(client, data=_scaffold_csv(),
                     name="scaffold-fixture.csv", target="y")
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": ds["dataset_id"],
                "model": "random_forest",
                "tune": tune,
                "n_jobs": 1,
                "mode": "sync",
                "split": split,
            },
        )
        return ds, r

    def _latest_train_run(self, client) -> dict:
        body = client.get(
            "/api/v1/runs", headers=auth_headers(),
            params={"family": "ml.train"},
        ).json()
        assert body["runs"], "no ml.train run record found"
        return client.get(
            f"/api/v1/runs/{body['runs'][0]['run_id']}",
            headers=auth_headers(),
        ).json()

    def test_scaffold_split_keeps_family_whole(self, client):
        # the fail-first fixture: the 8-row ring family must land on ONE
        # side and the estimator must train on the carved train side
        _, r = self._train_split(
            client, {"strategy": "scaffold", "test_size": 0.25})
        assert r.status_code == 200, r.text
        body = r.json()
        split = body["split"]
        assert split["strategy"] == "scaffold"
        assert split["smiles_column"] == "smiles"  # the honest auto-pick
        assert split["group_column"] is None
        assert split["test_size"] == 0.25
        assert split["n_rows"] == 12
        assert split["sides"]["train"]["rows"] == 8
        assert split["sides"]["heldout"]["rows"] == 4
        # the leakage verdict: no scaffold key on two sides
        census = split["group_census"]
        assert census is not None
        assert census["shared_groups"] == []
        # the ring family ('c1ccccc1', 8 rows) lands wholly on ONE side
        ring_counts = {
            name: dict(side.get("top_groups", [])).get("c1ccccc1", 0)
            for name, side in census.items()
            if isinstance(side, dict)
        }
        nonzero = [n for n, count in ring_counts.items() if count > 0]
        assert nonzero and len(nonzero) == 1
        assert ring_counts[nonzero[0]] == 8
        # the carve is real: the fit saw the train side only
        assert body["model"]["n_train_rows"] == split["sides"]["train"]["rows"]

    def test_split_block_pinned_in_the_run_record(self, client):
        _, r = self._train_split(
            client, {"strategy": "scaffold", "test_size": 0.25})
        assert r.status_code == 200, r.text
        record = self._latest_train_run(client)
        assert record["status"] == "completed"
        split = record["cross_validation"]["split"]
        assert split["strategy"] == "scaffold"
        assert split["smiles_column"] == "smiles"
        assert split["sides"]["train"]["rows"] == 8
        assert split["sides"]["heldout"]["rows"] == 4
        assert split["group_census"]["shared_groups"] == []
        # the model artifact row count IS the carved train side
        assert record["model_artifact"]["n_train_rows"] == 8

    def test_refused_group_split_records_the_failed_run(self, client):
        _, r = self._train_split(
            client, {"strategy": "group", "test_size": 0.25})
        assert r.status_code == 422, r.text
        assert "explicit group_column" in r.json()["detail"]
        # the §15.2 failure exit: the refused echo is traceable
        record = self._latest_train_run(client)
        assert record["status"] == "failed"
        split = record["cross_validation"]["split"]
        assert split["strategy"] == "group"
        assert split["group_column"] is None
        assert split["sides"] is None
        assert "group_column" in (split["error"] or "")

    def test_group_strategy_with_column_census(self, client):
        _, r = self._train_split(
            client,
            {"strategy": "group", "group_column": "grp", "test_size": 0.25},
        )
        assert r.status_code == 200, r.text
        split = r.json()["split"]
        assert split["strategy"] == "group"
        assert split["group_column"] == "grp"
        assert split["smiles_column"] is None
        assert split["group_census"]["shared_groups"] == []
        assert split["sides"]["train"]["rows"] + \
            split["sides"]["heldout"]["rows"] == 12

    def test_temporal_train_side_is_positional(self, client):
        _, r = self._train_split(
            client, {"strategy": "temporal", "test_size": 0.25})
        assert r.status_code == 200, r.text
        split = r.json()["split"]
        # order-as-given: the first 9 rows train, the last 3 hold out
        assert split["sides"]["train"]["rows"] == 9
        assert split["sides"]["heldout"]["rows"] == 3
        assert r.json()["model"]["n_train_rows"] == 9

    def test_no_split_section_is_the_legacy_full_frame(self, client):
        ds = _upload(client, data=_scaffold_csv(),
                     name="scaffold-fixture.csv", target="y")
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": ds["dataset_id"],
                "model": "random_forest",
                "tune": False, "n_jobs": 1, "mode": "sync",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # the legacy lane answers exactly as before: no split key at all
        assert "split" not in body
        assert body["model"]["n_train_rows"] == 12
        record = self._latest_train_run(client)
        assert record["cross_validation"]["split"] is None

    def test_smiles_column_not_found_is_422(self, client):
        _, r = self._train_split(
            client,
            {"strategy": "scaffold", "smiles_column": "nope",
             "test_size": 0.25},
        )
        assert r.status_code == 422, r.text
        assert "not found" in r.json()["detail"]

    def test_kennard_stone_without_numeric_is_422(self, client):
        buf = io.StringIO()
        buf.write("smiles,grp,y\n")
        for i in range(12):
            buf.write(f"CCO,{'A' if i % 2 else 'B'},{i % 2}\n")
        ds = _upload(client, data=buf.getvalue().encode(),
                     name="strings-only.csv", target="y")
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": ds["dataset_id"], "model": "random_forest",
                "tune": False, "n_jobs": 1, "mode": "sync",
                "split": {"strategy": "kennard_stone", "test_size": 0.25},
            },
        )
        assert r.status_code == 422, r.text
        assert "numeric feature column" in r.json()["detail"]

    def test_train_split_job_mode_relays_the_block(self, client):
        ds = _upload(client, data=_scaffold_csv(),
                     name="scaffold-fixture.csv", target="y")
        r = client.post(
            "/api/v1/ml/train",
            headers=auth_headers(),
            json={
                "dataset_id": ds["dataset_id"], "model": "random_forest",
                "tune": False, "n_jobs": 1, "mode": "job",
                "split": {"strategy": "scaffold", "test_size": 0.25},
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        assert result["split"]["strategy"] == "scaffold"
        assert result["split"]["group_census"]["shared_groups"] == []
        assert result["model"]["n_train_rows"] == 8


# ------------------------------------------------------------ ml catalog


class TestMLCatalog:
    def test_catalog_lists_models(self, client):
        r = client.get("/api/v1/ml/models", headers=auth_headers())
        assert r.status_code == 200
        models = r.json()["models"]
        assert models
        for row in models:
            assert set(row) == {"name", "task", "available", "requires",
                                "notes"}
        assert any(m["name"] == "random_forest" for m in models)

    def test_catalog_task_filter(self, client):
        models = client.get(
            "/api/v1/ml/models", headers=auth_headers(),
            params={"task": "classification"},
        ).json()["models"]
        assert models
        assert all(m["task"] in ("classification", "both") for m in models)

    def test_catalog_bad_task_422(self, client):
        r = client.get(
            "/api/v1/ml/models", headers=auth_headers(),
            params={"task": "clasification"},
        )
        assert r.status_code == 422


# ------------------------------------------------------------- ml train


class TestMLTrain:
    def test_sync_train_random_forest(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        assert model["model_id"].startswith("mdl-")
        assert model["name"] == "random_forest"
        assert model["task"] == "classification"
        assert model["features"] == ["x1", "x2"]
        assert model["n_train_rows"] == 24
        assert model["target"] == "y"
        assert model["fit_seconds"] >= 0
        assert isinstance(model["best_params"], dict)
        trained = client.get(
            "/api/v1/ml/trained", headers=auth_headers()
        ).json()["models"]
        assert any(m["model_id"] == model["model_id"] for m in trained)

    def test_train_tuned_cv_reports_folds(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"], tune=True, cv=2, n_jobs=1)
        assert model["cv_scores"] is not None
        assert len(model["cv_scores"]) == 2
        assert model["cv_mean"] is not None

    def test_train_job_mode(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"], mode="job")
        assert model["model_id"].startswith("mdl-")

    def test_train_unknown_dataset_404(self, client):
        r = client.post(
            "/api/v1/ml/train", headers=auth_headers(),
            json={"dataset_id": "ds-doesnotexist", "model": "random_forest"},
        )
        assert r.status_code == 404

    def test_train_unknown_model_422(self, client):
        ds = _upload(client)
        r = client.post(
            "/api/v1/ml/train", headers=auth_headers(),
            json={"dataset_id": ds["dataset_id"],
                  "model": "definitely_not_a_model"},
        )
        assert r.status_code == 422
        assert "definitely_not_a_model" in r.json()["detail"]


# ----------------------------------------------------------- ml predict


class TestMLPredict:
    def test_predict_inline_rows(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "rows": [{"x1": 0.5, "x2": 0.25},
                            {"x1": 1.1, "x2": 0.5}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["n_rows"] == 2
        assert len(body["predictions"]) == 2
        assert all(p in (0, 1) for p in body["predictions"])
        assert "probabilities" not in body

    def test_predict_with_probabilities(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "rows": [{"x1": 0.5, "x2": 0.25}],
                  "predict_proba": True},
        )
        assert r.status_code == 200
        proba = r.json()["probabilities"]
        assert proba["classes"] == [0, 1]
        assert len(proba["rows"]) == 1
        assert len(proba["rows"][0]) == 2
        assert abs(sum(proba["rows"][0]) - 1.0) < 1e-6

    def test_predict_on_dataset(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "dataset_id": ds["dataset_id"]},
        )
        assert r.status_code == 200
        assert r.json()["n_rows"] == 24

    def test_predict_unknown_model_404(self, client):
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": "mdl-doesnotexist",
                  "rows": [{"x1": 0.5, "x2": 0.25}]},
        )
        assert r.status_code == 404

    def test_predict_source_xor_422(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        base = {"model_id": model["model_id"]}
        neither = client.post(
            "/api/v1/ml/predict", headers=auth_headers(), json=base
        )
        assert neither.status_code == 422
        both = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={**base, "rows": [{"x1": 1.0, "x2": 1.0}],
                  "dataset_id": ds["dataset_id"]},
        )
        assert both.status_code == 422

    def test_predict_feature_mismatch_422(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "rows": [{"zzz": 1.0}]},
        )
        assert r.status_code == 422

    # ----------------------------- part11 §22.5 — the evaluation block

    def test_predict_dataset_mode_carries_evaluation(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "dataset_id": ds["dataset_id"]},
        )
        assert r.status_code == 200
        body = r.json()
        evaluation = body["evaluation"]
        assert evaluation["target"] == "y"
        assert evaluation["task"] == "classification"
        assert evaluation["smiles_col"] is None  # the toy CSV has none
        assert evaluation["n_rows_available"] == body["n_rows"] == 24
        assert evaluation["n_rows_evaluated"] == 24
        assert len(evaluation["rows"]) == 24
        first = evaluation["rows"][0]
        # the stable frame-index identity, the aligned actual, the same
        # prediction the top-level array reports
        assert first["name"] == 0
        assert first["actual"] == 0
        assert first["predicted"] == body["predictions"][0]
        # classification carries no residual — the honest null
        assert first["residual"] is None
        assert first["smiles"] is None

    def test_predict_inline_rows_have_no_evaluation(self, client):
        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "rows": [{"x1": 0.5, "x2": 0.25}]},
        )
        assert r.status_code == 200
        # inline rows have no aligned actuals to report — the block is
        # a dataset-mode lane, never a fabricated one
        assert "evaluation" not in r.json()

    def test_evaluation_regression_residual_math(self, client):
        buf = io.StringIO()
        buf.write("x1,x2,y\n")
        # 32 rows with all-distinct numeric targets: infer_task's
        # max_class_count=20 boundary crossed -> regression
        for i in range(32):
            buf.write(f"{round(0.25 * i + 0.5, 3)},{(i * 3) % 7 * 0.5},"
                       f"{round(2.0 * (0.25 * i + 0.5) + (i * 3) % 7 * 0.25 + 0.01 * i, 3)}\n")
        csv = buf.getvalue().encode()
        ds = _upload(client, data=csv, name="reg.csv", target="y")
        assert ds["task"] == "regression"
        model = _train(client, ds["dataset_id"], task="regression")
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "dataset_id": ds["dataset_id"]},
        )
        assert r.status_code == 200
        evaluation = r.json()["evaluation"]
        assert evaluation["task"] == "regression"
        for row in evaluation["rows"]:
            # the residual IS actual - predicted, recomputed from the
            # block's own fields (the parity plot's y-axis minus x)
            assert row["residual"] is not None
            assert abs(
                row["residual"] - (row["actual"] - row["predicted"])
            ) < 1e-9

    def test_evaluation_carries_the_smiles_column(self, client):
        buf = io.StringIO()
        buf.write("smiles,x1,x2,y\n")
        for i in range(16):
            smiles = "C" * (i % 4 + 1) + "O"  # toy parseable structures
            buf.write(f"{smiles},{round(0.2 * i + 0.5, 3)},"
                       f"{(i * 5) % 3 * 0.5},{i % 2}\n")
        csv = buf.getvalue().encode()
        ds = _upload(client, data=csv, name="smiles.csv", target="y")
        assert "smiles" in ds["dropped_columns"]
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "dataset_id": ds["dataset_id"]},
        )
        assert r.status_code == 200
        evaluation = r.json()["evaluation"]
        assert evaluation["smiles_col"] == "smiles"
        first = evaluation["rows"][0]
        assert first["smiles"] == "CO"
        # the identity travels with the structure: the §22.5 click
        # payload has name + actual + predicted + residual + smiles
        assert first["name"] == 0
        assert first["actual"] == 0
        assert "predicted" in first

    def test_evaluation_bound_is_reported_never_silent(self, client):
        buf = io.StringIO()
        buf.write("x1,x2,y\n")
        for i in range(600):
            buf.write(f"{round(0.01 * i + 0.5, 3)},{(i * 7) % 5 * 0.25},{i % 2}\n")
        csv = buf.getvalue().encode()
        ds = _upload(client, data=csv, name="big.csv", target="y")
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "dataset_id": ds["dataset_id"]},
        )
        assert r.status_code == 200
        body = r.json()
        evaluation = body["evaluation"]
        # the full prediction set stays authoritative (§17.3: the
        # bound is on the plotted sample, never on the result)
        assert body["n_rows"] == 600
        assert evaluation["n_rows_available"] == 600
        assert evaluation["n_rows_evaluated"] == 500
        assert len(evaluation["rows"]) == 500


# --------------------------------------------------------- figure artifacts


class TestArtifactFigure:
    def test_png_download_contract(self, client):
        art = _figure(client, format="PNG")
        assert art["artifact_id"].startswith("art-")
        assert len(art["sha256"]) == 64
        assert art["size_bytes"] > 0
        assert art["content_type"] == "image/png"
        assert art["download_url"].endswith(f"/{art['artifact_id']}/download")
        r = client.get(art["download_url"], headers=auth_headers())
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert int(r.headers["content-length"]) == len(r.content)
        assert r.headers["etag"] == f'"{art["sha256"]}"'
        assert r.headers["x-artifact-sha256"] == art["sha256"]
        assert r.headers["x-artifact-kind"] == "figure"
        assert "attachment" in r.headers["content-disposition"]
        assert r.content.startswith(b"\x89PNG\r\n")
        assert hashlib.sha256(r.content).hexdigest() == art["sha256"]

    def test_all_four_formats_magic_bytes(self, client):
        magics = {
            "PNG": b"\x89PNG\r\n",
            "SVG": b"<?xml",
            "TIFF": b"II*\x00",
            "JPEG": b"\xff\xd8\xff",
        }
        for fmt, magic in magics.items():
            art = _figure(client, format=fmt)
            r = client.get(art["download_url"], headers=auth_headers())
            assert r.status_code == 200, fmt
            assert r.content.startswith(magic), fmt
            assert r.headers["content-type"].startswith("image/")
            assert art["meta"]["format"] == fmt

    def test_scatter_requires_y_422(self, client):
        r = client.post(
            "/api/v1/artifacts/figure", headers=auth_headers(),
            json={"kind": "scatter", "x": [1.0, 2.0, 3.0]},
        )
        assert r.status_code == 422

    def test_mismatched_lengths_422(self, client):
        r = client.post(
            "/api/v1/artifacts/figure", headers=auth_headers(),
            json={"kind": "scatter", "x": [1.0, 2.0, 3.0],
                  "y": [1.0, 2.0]},
        )
        assert r.status_code == 422

    def test_bar_and_histogram_kinds(self, client):
        bar = client.post(
            "/api/v1/artifacts/figure", headers=auth_headers(),
            json={"kind": "bar", "x": ["a", "b", "a", "c"],
                  "y": [1.0, 2.0, 3.0, 4.0]},
        )
        assert bar.status_code == 200
        assert bar.json()["artifact"]["meta"]["figure_kind"] == "bar"
        hist = client.post(
            "/api/v1/artifacts/figure", headers=auth_headers(),
            json={"kind": "histogram", "x": [1.0, 2.0, 2.0, 3.0, 5.0]},
        )
        assert hist.status_code == 200
        assert hist.json()["artifact"]["meta"]["figure_kind"] == "histogram"

    def test_delete_artifact_then_download_404(self, client):
        art = _figure(client)
        r = client.delete(
            f"/api/v1/artifacts/{art['artifact_id']}",
            headers=auth_headers(),
        )
        assert r.status_code == 200
        assert client.get(
            art["download_url"], headers=auth_headers()
        ).status_code == 404

    def test_head_download_headers(self, client):
        art = _figure(client)
        r = client.head(art["download_url"], headers=auth_headers())
        assert r.status_code == 200
        assert r.headers["x-artifact-sha256"] == art["sha256"]
        assert int(r.headers["content-length"]) == art["size_bytes"]


# --------------------------------------------------------- table artifacts


class TestArtifactTable:
    def test_csv_roundtrip(self, client):
        ds = _upload(client)
        r = client.post(
            "/api/v1/artifacts/table", headers=auth_headers(),
            json={"dataset_id": ds["dataset_id"], "format": "csv"},
        )
        assert r.status_code == 200
        art = r.json()["artifact"]
        assert art["kind"] == "table"
        dl = client.get(art["download_url"], headers=auth_headers())
        assert dl.status_code == 200
        assert dl.headers["content-type"].startswith("text/csv")
        frame = pd.read_csv(io.BytesIO(dl.content))
        assert list(frame.columns) == ["x1", "x2", "y"]
        assert len(frame) == 24
        assert frame["y"].tolist() == [i % 2 for i in range(24)]

    def test_xlsx_roundtrip(self, client):
        ds = _upload(client)
        r = client.post(
            "/api/v1/artifacts/table", headers=auth_headers(),
            json={"dataset_id": ds["dataset_id"], "format": "xlsx"},
        )
        assert r.status_code == 200
        art = r.json()["artifact"]
        dl = client.get(art["download_url"], headers=auth_headers())
        assert "spreadsheetml" in dl.headers["content-type"]
        frame = pd.read_excel(io.BytesIO(dl.content))
        assert len(frame) == 24

    def test_table_unknown_dataset_404(self, client):
        r = client.post(
            "/api/v1/artifacts/table", headers=auth_headers(),
            json={"dataset_id": "ds-doesnotexist"},
        )
        assert r.status_code == 404


# --------------------------------------------------------- model artifacts


class TestArtifactModel:
    def test_model_bundle_roundtrip(self, client, tmp_path):
        from pharmaml.ml.persistence import load_model

        ds = _upload(client)
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/artifacts/model", headers=auth_headers(),
            json={"model_id": model["model_id"]},
        )
        assert r.status_code == 200
        art = r.json()["artifact"]
        assert art["kind"] == "model"
        assert art["filename"].endswith(".joblib")
        assert art["meta"]["model_name"] == "random_forest"
        assert art["meta"]["features"] == ["x1", "x2"]
        dl = client.get(art["download_url"], headers=auth_headers())
        assert dl.status_code == 200
        assert dl.headers["content-type"] == "application/octet-stream"
        target = tmp_path / "downloaded.joblib"
        target.write_bytes(dl.content)
        estimator, metadata = load_model(target)
        assert metadata["name"] == "random_forest"
        assert metadata["task"] == "classification"
        assert list(metadata["features"]) == ["x1", "x2"]
        # the parity clause: the downloaded bundle predicts identically
        probe = pd.DataFrame({"x1": [0.5, 1.1], "x2": [0.25, 0.5]})
        session_predictions = client.post(
            "/api/v1/ml/predict", headers=auth_headers(),
            json={"model_id": model["model_id"],
                  "rows": [{"x1": 0.5, "x2": 0.25},
                           {"x1": 1.1, "x2": 0.5}]},
        ).json()["predictions"]
        assert estimator.predict(probe).tolist() == session_predictions

    def test_model_artifact_unknown_404(self, client):
        r = client.post(
            "/api/v1/artifacts/model", headers=auth_headers(),
            json={"model_id": "mdl-doesnotexist"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------- lifecycle


class TestArtifactLifecycle:
    def test_list_contains_artifacts(self, client):
        _figure(client)
        listed = client.get(
            "/api/v1/artifacts", headers=auth_headers()
        ).json()["artifacts"]
        assert listed

    def test_bounded_eviction(self, client):
        # honest cap enforcement: a small store evicts oldest-first
        client.app.state.artifact_store = ArtifactStore(max_items=3)
        ids = []
        for i in range(5):
            art = _figure(client, x=[float(i), float(i + 1)],
                          y=[float(i), float(i + 1)])
            ids.append(art["artifact_id"])
        listed = client.get(
            "/api/v1/artifacts", headers=auth_headers()
        ).json()["artifacts"]
        assert len(listed) == 3
        assert ids[0] not in [a["artifact_id"] for a in listed]
        assert ids[-1] in [a["artifact_id"] for a in listed]
        assert client.get(
            f"/api/v1/artifacts/{ids[0]}/download",
            headers=auth_headers(),
        ).status_code == 404

    def test_stores_are_session_scoped(self):
        fresh = create_app(token=TEST_TOKEN)
        with TestClient(fresh, base_url="http://127.0.0.1") as tc:
            body = tc.get(
                "/api/v1/datasets", headers=auth_headers()
            ).json()
            assert body == {"datasets": []}
            trained = tc.get(
                "/api/v1/ml/trained", headers=auth_headers()
            ).json()
            assert trained == {"models": []}
            arts = tc.get(
                "/api/v1/artifacts", headers=auth_headers()
            ).json()
            assert arts == {"artifacts": []}


# ------------------------------------------------------------- ml explain
# part11 §11.2's Explainability/Uncertainty stage — the r289 queue row
# closed: POST /ml/explain rides the REAL shap_summary engine (the
# tree-family SHAP lane), bounded preview matrices, the honest
# non-tree refusal, and the importance + beeswarm figure artifacts.


class TestMLExplain:

    def _prepare(self, client, rows: int = 40, model: str = "random_forest"):
        ds = _upload(client, data=_csv_bytes(rows=rows))
        trained = _train(client, ds["dataset_id"], model=model)
        return ds, trained

    def test_explain_sync_ranking_and_preview(self, client):
        ds, trained = self._prepare(client, rows=40)
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={
                "model_id": trained["model_id"],
                "dataset_id": ds["dataset_id"],
                "max_rows": 30,
            },
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["mode"] == "sync"
        explain = payload["explain"]
        assert explain["engine"] == "shap_summary"
        assert explain["model_id"] == trained["model_id"]
        assert explain["task"] in ("classification", "regression")
        assert explain["model_name"]
        assert explain["n_rows_available"] == 40
        assert explain["n_samples"] == 30  # the bounded sample, honest
        assert explain["n_features"] == 2
        assert explain["base_value"] is not None
        ranking = explain["ranking"]
        assert len(ranking) == 2
        assert ranking[0]["mean_abs_shap"] >= ranking[1]["mean_abs_shap"]
        # the preview carries the top-ranked columns, one row per
        # explained sample
        preview = explain["preview"]
        assert preview["features"] == [
            row["feature"] for row in ranking
        ]
        assert len(preview["values"]) == 30
        assert len(preview["values"][0]) == 2
        assert len(preview["feature_values"]) == 30
        assert explain["artifacts"] == []

    def test_explain_defaults_to_the_models_training_dataset(self, client):
        _ds, trained = self._prepare(client)
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={"model_id": trained["model_id"], "max_rows": 24},
        )
        assert r.status_code == 200, r.text
        explain = r.json()["explain"]
        assert explain["n_rows_available"] == 40
        assert explain["n_samples"] == 24

    def test_explain_job_mode_progress_and_result(self, client):
        _ds, trained = self._prepare(client)
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={
                "model_id": trained["model_id"],
                "max_rows": 24,
                "mode": "job",
            },
        )
        assert r.status_code == 200, r.text
        job = r.json()["job"]
        assert job["operation"] == "ml.explain"
        assert job["workspace"] == "ml"
        final = _wait_job(client, job["job_id"])
        assert final["state"] == "COMPLETED", final
        result = client.get(
            f"/api/v1/jobs/{job['job_id']}/result", headers=auth_headers()
        ).json()["result"]
        explain = result["explain"]
        assert explain["n_samples"] == 24
        assert explain["engine"] == "shap_summary"
        assert len(explain["ranking"]) == 2

    def test_explain_figures_download_byte_for_byte(self, client):
        _ds, trained = self._prepare(client)
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={
                "model_id": trained["model_id"],
                "max_rows": 24,
                "figure": True,
            },
        )
        assert r.status_code == 200, r.text
        artifacts = r.json()["explain"]["artifacts"]
        assert [a["filename"] for a in artifacts] == [
            "shap-importance.png", "shap-beeswarm.png"
        ]
        for artifact in artifacts:
            dl = client.get(
                f"/api/v1/artifacts/{artifact['artifact_id']}/download",
                headers=auth_headers(),
            )
            assert dl.status_code == 200
            assert dl.content[:8] == b"\x89PNG\r\n\x1a\n"
            assert dl.headers["X-Artifact-SHA256"] == hashlib.sha256(
                dl.content
            ).hexdigest()

    def test_explain_non_tree_model_refused_honestly(self, client):
        _ds, trained = self._prepare(client, model="svm")
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={"model_id": trained["model_id"]},
        )
        assert r.status_code == 422
        detail = str(r.json()["detail"])
        # the engine's own refusal names the engine family, never a
        # silent fallback
        assert "TreeExplainer" in detail

    def test_explain_unknown_model_404(self, client):
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={"model_id": "mdl-unknown"},
        )
        assert r.status_code == 404

    def test_explain_missing_dataset_404(self, client):
        ds, trained = self._prepare(client)
        evict = client.delete(
            f"/api/v1/datasets/{ds['dataset_id']}", headers=auth_headers()
        )
        assert evict.status_code == 200, evict.text
        r = client.post(
            "/api/v1/ml/explain",
            headers=auth_headers(),
            json={"model_id": trained["model_id"]},
        )
        assert r.status_code == 404
        assert "training dataset" in r.json()["detail"]

    def test_explain_max_rows_bounds_validated(self, client):
        _ds, trained = self._prepare(client)
        for bad in (0, -1, 501):
            r = client.post(
                "/api/v1/ml/explain",
                headers=auth_headers(),
                json={"model_id": trained["model_id"], "max_rows": bad},
            )
            assert r.status_code == 422, bad
