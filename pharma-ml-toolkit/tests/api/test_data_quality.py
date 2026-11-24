"""The part16 §15.4 data-quality lane -- the API proof.

The directive's "Data Quality and Scientific Validation" checklist,
made mechanical.  The scan must SEE every defect class it names
without ever TOUCHING the data:

* **the chemistry pair** -- the two-stage SMILES validation (grammar
  parse, then sanitization) plus the canonical-form duplicate census
  (:class:`TestChemistryPair`);
* **completeness/integrity** -- missing values, duplicate rows,
  non-finite descriptors (:class:`TestCompletenessIntegrity`);
* **the leakage family** -- the ID-like advisory, identical column
  pairs, target leakage, and the train/test overlap advisory
  (:class:`TestLeakageFamily`);
* **distribution/range** -- class imbalance, constant columns, the
  pinned unit ranges (:class:`TestDistributionRange`);
* **the honest surface** -- the executed/skipped check list, the
  record's own target by default, the SMILES override + disable,
  the read-only flag (:class:`TestHonestSurface`);
* **immutability** -- the profile and view are byte-identical
  before and after the scan (§15.4: never silently alter
  user-provided scientific data) (:class:`TestReadOnlyGuarantee`);
* **job mode + the 404** (:class:`TestJobModeAndErrors`);
* **the REAL demo dataset** -- the scan over the shipped
  walkthrough data (:class:`TestRealDatasetScan`).
"""

from __future__ import annotations

import time

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


# ---------------------------------------------------------------- the frames

#: 10 rows: the chemistry/integrity frame -- a 9:1 class imbalance,
#: a duplicate row, a duplicate molecule on a distinct row, an
#: unparseable SMILES, a negative molecular weight, an infinite
#: descriptor, a missing molecular weight.
CHEMISTRY_CSV = """smiles,mw,activity
CCO,46.07,1
CCO,46.07,1
CCO,,1
CCN,44.08,1
CCC,44.10,1
CCCO,60.10,1
CCOC,60.10,1
C1COCCO1,88.15,1
not_a_smiles,-5.0,1
CCN,inf,0
"""

#: 12 rows: the leakage frame -- a unique-per-row id column, a
#: constant column, a target, and two identical copies of it.
LEAKAGE_CSV = """id,activity,mirror,copy,flat
1,1,1,1,0
2,1,1,1,0
3,1,1,1,0
4,1,1,1,0
5,1,1,1,0
6,1,1,1,0
7,1,1,1,0
8,1,1,1,0
9,1,1,1,0
10,1,1,1,0
11,1,1,1,0
12,0,0,0,0
"""


def _upload_csv(
    client, name: str, csv_text: str, target: str = "activity"
) -> str:
    files = {"file": (name, csv_text.encode("utf-8"), "text/csv")}
    response = client.post(
        "/api/v1/datasets", files=files,
        data={"mode": "sync", "target": target},
        headers=auth_headers(),
    )
    assert response.status_code == 200, response.text
    return response.json()["dataset"]["dataset_id"]


def _quality(client, dataset_id: str, **payload):
    return client.post(
        f"/api/v1/datasets/{dataset_id}/quality",
        json=payload or {"mode": "sync"},
        headers=auth_headers(),
    )


def _codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


# ---------------------------------------------------------------- the lanes

class TestChemistryPair:
    """§15.4 SMILES validation + molecular structure validation."""

    def test_invalid_smiles_is_an_error_with_examples(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {
            f["code"]: f for f in report["findings"]
        }
        assert "invalid_smiles" in findings
        finding = findings["invalid_smiles"]
        assert finding["severity"] == "error"
        assert finding["category"] == "chemistry"
        assert finding["columns"] == ["smiles"]
        assert finding["evidence"]["invalid_count"] == 1
        assert "not_a_smiles" in finding["evidence"]["examples"]

    def test_the_smiles_column_is_auto_detected_and_reported(
        self, client,
    ):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        assert report["smiles_column"] == "smiles"
        checks = {c["check"]: c for c in report["checks"]}
        assert checks["smiles_validation"]["status"] == "executed"
        assert checks["molecular_structure_validation"]["status"] == (
            "executed"
        )

    def test_duplicate_molecules_by_canonical_form(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert "duplicate_molecules" in findings
        # CCO appears 3x (one exact-row duplicate + one distinct row),
        # CCN 2x -- the census counts rows sharing canonical forms.
        assert findings["duplicate_molecules"]["evidence"][
            "duplicate_rows"
        ] >= 5


class TestCompletenessIntegrity:
    """§15.4 missing-value reports + duplicate detection + descriptors."""

    def test_missing_values_per_column(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert "missing_values" in findings
        per_column = findings["missing_values"]["evidence"][
            "per_column"
        ]
        assert per_column["mw"] == 1

    def test_duplicate_rows(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert findings["duplicate_rows"]["evidence"][
            "duplicate_rows"
        ] == 2  # the two rows of the one duplicated pair

    def test_non_finite_and_out_of_range_descriptors(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert "non_finite_values" in findings
        assert findings["non_finite_values"]["severity"] == "error"
        assert "out_of_range_values" in findings
        range_finding = findings["out_of_range_values"]
        assert range_finding["columns"] == ["mw"]
        assert "molecular weight" in range_finding["evidence"]["rule"]
        assert range_finding["evidence"]["violations"] >= 2  # -5 and inf


class TestLeakageFamily:
    """§15.4 data leakage warnings + target leakage + separation."""

    def test_target_leakage_over_identical_copies(self, client):
        dataset_id = _upload_csv(client, "leak.csv", LEAKAGE_CSV)
        report = _quality(
            client, dataset_id, target="activity",
        ).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert "target_leakage" in findings
        leaking = {tuple(f["columns"]) for f in report["findings"]
                   if f["code"] == "target_leakage"}
        assert ("mirror", "activity") in leaking
        assert ("copy", "activity") in leaking

    def test_id_like_column_advisory(self, client):
        dataset_id = _upload_csv(client, "leak.csv", LEAKAGE_CSV)
        report = _quality(
            client, dataset_id, target="activity",
        ).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert findings["id_like_column"]["columns"] == ["id"]
        assert findings["id_like_column"]["severity"] == "info"

    def test_duplicate_columns_and_constant_column(self, client):
        dataset_id = _upload_csv(client, "leak.csv", LEAKAGE_CSV)
        report = _quality(
            client, dataset_id, target="activity",
        ).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert {"mirror", "copy"} <= set(
            findings["duplicate_columns"]["columns"]
        )
        assert findings["constant_columns"]["columns"] == ["flat"]

    def test_train_test_overlap_advisory_rides_duplicates(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        codes = _codes(report)
        assert "train_test_overlap_risk" in codes
        overlap = {f["code"]: f for f in report["findings"]}[
            "train_test_overlap_risk"
        ]
        assert overlap["category"] == "leakage"


class TestDistributionRange:
    """§15.4 class imbalance warnings + the pinned unit ranges."""

    def test_class_imbalance_at_ninety_percent_majority(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        findings = {f["code"]: f for f in report["findings"]}
        assert "class_imbalance" in findings
        assert findings["class_imbalance"]["evidence"][
            "majority_fraction"
        ] == 0.9

    def test_the_unit_vocabulary_is_exact_and_pinned(self, client):
        from pharmaml.data_quality import SCIENTIFIC_RANGES

        assert set(SCIENTIFIC_RANGES) == {
            "mw", "molecular_weight", "mass", "logp", "logd",
            "conc", "concentration", "dose", "time", "hour",
        }


class TestHonestSurface:
    """The executed/skilled check list + the overrides + read-only."""

    def test_chemistry_check_honestly_skipped_without_a_smiles_column(
        self, client,
    ):
        dataset_id = _upload_csv(client, "leak.csv", LEAKAGE_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        assert report["smiles_column"] is None
        checks = {c["check"]: c for c in report["checks"]}
        assert checks["smiles_validation"]["status"] == "skipped"
        assert checks["molecular_structure_validation"]["status"] == (
            "skipped"
        )
        assert "no SMILES column detected" in checks[
            "smiles_validation"
        ]["detail"]

    def test_the_check_list_covers_the_directive_vocabulary(
        self, client,
    ):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        keys = {c["check"] for c in report["checks"]}
        assert keys == {
            "smiles_validation",
            "molecular_structure_validation",
            "duplicate_detection",
            "invalid_descriptor_detection",
            "missing_value_report",
            "data_leakage_warnings",
            "target_leakage_detection",
            "class_imbalance_warnings",
            "train_test_separation_check",
            "unit_validation",
            "scientific_input_validation",
        }
        assert all("label" in c for c in report["checks"])

    def test_the_read_only_flag_and_advisory_ride_every_report(
        self, client,
    ):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        report = _quality(client, dataset_id).json()["quality"]
        assert report["read_only"] is True
        assert "never modifies" in report["advisory"]

    def test_smiles_column_override_and_disable(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        # the override points the pair at a non-SMILES column
        report = _quality(
            client, dataset_id, smiles_column="activity",
        ).json()["quality"]
        assert report["smiles_column"] == "activity"
        assert "invalid_smiles" in _codes(report)
        # the disable (empty string) skips the chemistry pair
        report = _quality(
            client, dataset_id, smiles_column="",
        ).json()["quality"]
        assert report["smiles_column"] is None
        assert "invalid_smiles" not in _codes(report)


class TestReadOnlyGuarantee:
    """§15.4's hard rule: never silently alter user data."""

    def test_the_profile_and_view_are_unchanged_by_the_scan(
        self, client,
    ):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        before = client.post(
            f"/api/v1/datasets/{dataset_id}/profile",
            json={"mode": "sync"}, headers=auth_headers(),
        ).json()["profile"]
        view_before = client.get(
            f"/api/v1/datasets/{dataset_id}", headers=auth_headers()
        ).json()
        _quality(client, dataset_id)  # the scan itself
        after = client.post(
            f"/api/v1/datasets/{dataset_id}/profile",
            json={"mode": "sync"}, headers=auth_headers(),
        ).json()["profile"]
        view_after = client.get(
            f"/api/v1/datasets/{dataset_id}", headers=auth_headers()
        ).json()
        assert before == after
        assert view_before == view_after

    def test_the_scan_is_deterministic(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        first = _quality(client, dataset_id).json()["quality"]
        second = _quality(client, dataset_id).json()["quality"]
        assert first == second


class TestJobModeAndErrors:
    """The async lane + the honest 404."""

    def test_job_mode_delivers_the_same_report(self, client):
        dataset_id = _upload_csv(client, "chem.csv", CHEMISTRY_CSV)
        response = _quality(client, dataset_id, mode="job")
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        job_id = job["job_id"]
        assert job["operation"] == "datasets.quality"
        state = None
        for _ in range(120):
            state = client.get(
                f"/api/v1/jobs/{job_id}", headers=auth_headers()
            ).json()["state"]
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            time.sleep(0.25)
        assert state == "COMPLETED"
        result = client.get(
            f"/api/v1/jobs/{job_id}/result", headers=auth_headers()
        ).json()["result"]
        report = result["quality"]
        assert report["read_only"] is True
        assert "invalid_smiles" in _codes(report)

    def test_unknown_dataset_is_a_clean_404(self, client):
        response = _quality(client, "dataset-does-not-exist")
        assert response.status_code == 404
        assert "Unknown dataset_id" in response.json()["detail"]


class TestRealDatasetScan:
    """The shipped demo data through the same gate."""

    def test_the_walkthrough_dataset_scans_clean(self, client):
        response = client.post(
            "/api/v1/samples/walkthrough-classification/load",
            headers=auth_headers(),
        )
        assert response.status_code == 200, response.text
        dataset_id = response.json()["dataset"]["dataset_id"]
        report = _quality(
            client, dataset_id,
        ).json()["quality"]
        assert report["rows"] > 0
        assert report["read_only"] is True
        # the real shipped data carries no error-class chemistry
        # defects (the walkthrough set is curated); advisory-class
        # findings are honest, errors are the defect signal.
        errors = [
            f for f in report["findings"] if f["severity"] == "error"
        ]
        chemistry_errors = [
            f for f in errors if f["category"] == "chemistry"
        ]
        assert chemistry_errors == []

    def test_the_record_target_is_used_by_default(self, client):
        response = client.post(
            "/api/v1/samples/walkthrough-classification/load",
            headers=auth_headers(),
        )
        assert response.status_code == 200, response.text
        body = response.json()["dataset"]
        dataset_id = body["dataset_id"]
        record_target = body["target"]
        report = _quality(client, dataset_id).json()["quality"]
        assert report["target"] == record_target
