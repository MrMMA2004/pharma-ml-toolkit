"""part21 §3/§4 — the evaluation-sample diagnostics on /ml/predict.

The part21 shared ML/DL evaluation-visualization contract: every route
here runs the REAL backend (train + predict over the same uploaded
dataset), then asserts the evaluation block's own sample-derived
diagnostics — the regression residual histogram/stats/Q–Q and the
classification per-class metrics + ROC/PR curves + the binary
reliability diagram — against invariants re-derived from the payload's
own fields (never a golden-file of invented numbers).

The scientific rails under test (part21 §4):

* the curves describe the SAME bounded sample the parity chart plots
  (``n_rows_evaluated``), never a second evaluation of the model's
  authoritative set;
* a probability-less answer attaches no curve fields (the honest
  absence — never a fabricated curve);
* a class with no positives or no negatives in the sample carries no
  ROC/PR curve for that class;
* the residual histogram counts sum to the numeric residual count;
  the Q–Q arrays are equal-length over the ordered sample;
* inline-row predictions carry no evaluation block at all (unchanged
  from the §22.5 contract).
"""

from __future__ import annotations

import io

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


def _upload(client, csv: bytes, name: str, target: str = "y") -> dict:
    r = client.post(
        "/api/v1/datasets",
        headers=auth_headers(),
        files={"file": (name, csv, "text/csv")},
        data={"target": target, "mode": "sync"},
    )
    assert r.status_code == 200, r.text
    return r.json()["dataset"]


def _train(client, dataset_id: str, model: str = "random_forest",
           task: str | None = None) -> dict:
    body: dict = {
        "dataset_id": dataset_id,
        "model": model,
        "mode": "sync",
    }
    if task is not None:
        body["task"] = task
    r = client.post(
        "/api/v1/ml/train", headers=auth_headers(), json=body,
    )
    assert r.status_code == 200, r.text
    return r.json()["model"]


def _predict(client, model_id: str, dataset_id: str,
             proba: bool = False) -> dict:
    r = client.post(
        "/api/v1/ml/predict",
        headers=auth_headers(),
        json={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "predict_proba": proba,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _regression_csv() -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(40):
        buf.write(
            f"{round(0.25 * i + 0.5, 3)},{(i * 3) % 7 * 0.5},"
            f"{round(2.0 * (0.25 * i + 0.5) + (i * 3) % 7 * 0.25 + 0.01 * i, 3)}\n"
        )
    return buf.getvalue().encode()


def _binary_csv() -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(30):
        label = 1 if (i % 3 == 0 or i % 7 == 0) else 0
        buf.write(f"{round(0.1 * i + 0.5, 3)},{(i * 5) % 6 * 0.25},{label}\n")
    return buf.getvalue().encode()


def _multiclass_csv() -> bytes:
    buf = io.StringIO()
    buf.write("x1,x2,y\n")
    for i in range(36):
        label = ["low", "mid", "high"][i % 3]
        buf.write(f"{round(0.15 * i + 0.5, 3)},{(i * 4) % 5 * 0.3},{label}\n")
    return buf.getvalue().encode()


class TestRegressionDiagnostics:
    def test_residual_diagnostics_attached(self, client):
        ds = _upload(client, _regression_csv(), "reg.csv")
        assert ds["task"] == "regression"
        model = _train(client, ds["dataset_id"], task="regression")
        body = _predict(client, model["model_id"], ds["dataset_id"])
        evaluation = body["evaluation"]
        assert evaluation["task"] == "regression"
        diag = evaluation["residual_diagnostics"]
        assert diag is not None
        rows = evaluation["rows"]
        numeric = [r for r in rows if r["residual"] is not None]
        # the histogram covers exactly the sample's own numeric residuals
        assert sum(diag["histogram"]["counts"]) == len(numeric)
        assert len(diag["histogram"]["bin_edges"]) == 21
        # the Q–Q arrays are equal-length over the ordered residuals
        qq = diag["qq"]
        assert qq is not None
        assert len(qq["theoretical"]) == len(qq["empirical"]) == len(numeric)
        # the summary stats describe those same residuals
        stats = diag["stats"]
        assert stats["n"] == len(numeric)
        residuals = sorted(r["residual"] for r in numeric)
        assert stats["min"] == residuals[0]
        assert stats["max"] == residuals[-1]
        # the residual math stays the §22.5 identity: actual - predicted
        for row in numeric:
            assert abs(
                row["residual"] - (row["actual"] - row["predicted"])
            ) < 1e-9

    def test_classification_carries_no_residual_diagnostics(self, client):
        ds = _upload(client, _binary_csv(), "bin.csv")
        model = _train(client, ds["dataset_id"])
        body = _predict(client, model["model_id"], ds["dataset_id"])
        evaluation = body["evaluation"]
        assert evaluation["task"] == "classification"
        # the honest absence — never a fabricated histogram
        assert "residual_diagnostics" not in evaluation


class TestClassificationCurves:
    def test_binary_curves_attached(self, client):
        ds = _upload(client, _binary_csv(), "bin.csv")
        model = _train(client, ds["dataset_id"])
        body = _predict(client, model["model_id"], ds["dataset_id"],
                        proba=True)
        evaluation = body["evaluation"]
        classes = body["probabilities"]["classes"]
        assert len(classes) == 2
        n = evaluation["n_rows_evaluated"]

        metrics = evaluation["class_metrics"]
        assert metrics is not None
        assert metrics["n"] == n
        assert len(metrics["rows"]) == 2
        assert sum(r["support"] for r in metrics["rows"]) == n
        # the model's own class ordering is preserved
        assert [r["label"] for r in metrics["rows"]] == [
            str(c) for c in classes
        ]
        # accuracy recomputed from the payload's own rows
        correct = sum(
            1
            for r in evaluation["rows"]
            if str(r["actual"]) == str(r["predicted"])
        )
        assert abs(metrics["accuracy"] - correct / n) < 1e-9

        roc = evaluation["roc_curves"]
        assert roc is not None and len(roc) == 2
        for curve in roc:
            assert curve["fpr"][0] == 0.0
            assert curve["fpr"][-1] == 1.0
            assert curve["tpr"][-1] == 1.0
            assert 0.0 <= curve["auc"] <= 1.0
            assert len(curve["fpr"]) == len(curve["tpr"])
        # a real binary ROC sits strictly between the degenerate ends
        assert 0.0 < roc[0]["auc"] < 1.0

        pr = evaluation["pr_curves"]
        assert pr is not None and len(pr) == 2
        for curve in pr:
            assert curve["recall"][-1] == 1.0
            assert 0.0 <= curve["average_precision"] <= 1.0
            assert curve["prevalence"] == curve["n_pos"] / n

        cal = evaluation["calibration"]
        assert cal is not None
        assert cal["n"] == n
        assert cal["positive_class"] == str(classes[-1])
        assert sum(b["count"] for b in cal["bins"]) == n
        for b in cal["bins"]:
            assert 0.0 <= b["observed"] <= 1.0
            assert 0.0 <= b["mean_predicted"] <= 1.0

    def test_multiclass_carries_per_class_curves_no_calibration(
        self, client,
    ):
        ds = _upload(client, _multiclass_csv(), "multi.csv")
        model = _train(client, ds["dataset_id"])
        body = _predict(client, model["model_id"], ds["dataset_id"],
                        proba=True)
        evaluation = body["evaluation"]
        classes = body["probabilities"]["classes"]
        n = evaluation["n_rows_evaluated"]
        assert len(classes) == 3

        metrics = evaluation["class_metrics"]
        assert len(metrics["rows"]) == 3
        assert sum(r["support"] for r in metrics["rows"]) == n

        # per-class OvR curves: one per class THAT HAS both positives
        # and negatives in the sample (all three here by construction)
        roc = evaluation["roc_curves"]
        assert sorted(c["label"] for c in roc) == sorted(
            str(c) for c in classes
        )
        pr = evaluation["pr_curves"]
        assert len(pr) == 3
        # the reliability diagram is the binary case's own lane — the
        # honest absence for multiclass, never a misleading aggregate
        assert "calibration" not in evaluation

    def test_no_proba_no_curves_but_metrics_present(self, client):
        ds = _upload(client, _binary_csv(), "bin.csv")
        model = _train(client, ds["dataset_id"])
        body = _predict(client, model["model_id"], ds["dataset_id"],
                        proba=False)
        evaluation = body["evaluation"]
        # the metrics need only the aligned rows — they ride always
        assert evaluation["class_metrics"] is not None
        # the curves need the probability block — the honest absence
        assert "roc_curves" not in evaluation
        assert "pr_curves" not in evaluation
        assert "calibration" not in evaluation

    def test_inline_rows_still_carry_no_evaluation(self, client):
        ds = _upload(client, _binary_csv(), "bin.csv")
        model = _train(client, ds["dataset_id"])
        r = client.post(
            "/api/v1/ml/predict",
            headers=auth_headers(),
            json={
                "model_id": model["model_id"],
                "rows": [{"x1": 0.5, "x2": 0.25}, {"x1": 1.1, "x2": 0.5}],
            },
        )
        assert r.status_code == 200
        assert "evaluation" not in r.json()

    def test_probabilities_align_with_evaluation_rows(self, client):
        """The curve lanes' position-alignment premise, checked live."""
        ds = _upload(client, _binary_csv(), "bin.csv")
        model = _train(client, ds["dataset_id"])
        body = _predict(client, model["model_id"], ds["dataset_id"],
                        proba=True)
        evaluation = body["evaluation"]
        probabilities = body["probabilities"]
        n = evaluation["n_rows_evaluated"]
        assert len(probabilities["rows"]) == evaluation["n_rows_available"]
        # the argmax of each sampled probability row is the row's own
        # prediction (the curves are only honest under this alignment)
        classes = probabilities["classes"]
        for i in range(n):
            row = probabilities["rows"][i]
            winner = max(range(len(classes)), key=lambda j: row[j])
            assert str(classes[winner]) == str(
                evaluation["rows"][i]["predicted"]
            )
