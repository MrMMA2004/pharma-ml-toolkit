"""part10's queued row — the datasets / ml / artifacts API surface.

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
