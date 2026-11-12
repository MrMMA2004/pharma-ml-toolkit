"""Tests for :mod:`pharmaml.ml.model_card`.

The card is generated from a real saved bundle, so these tests build one
with :func:`pharmaml.ml.persistence.save_model` and assert that every
recorded fact (identity, metrics, feature contract, applicability box,
provenance, and the mandatory limitations section) reaches the rendered
Markdown and HTML.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.model_card import (
    build_model_card,
    model_card_html,
    model_card_markdown,
    write_model_card,
)
from pharmaml.ml.persistence import save_model


@pytest.fixture()
def bundle(tmp_path):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = 2.0 * frame["a"] - frame["b"]
    model = LinearRegression().fit(frame, y)
    path = tmp_path / "lin.joblib"
    save_model(
        model,
        path,
        name="lin",
        task="regression",
        features=["a", "b", "c"],
        best_params={"fit_intercept": True},
        metrics={"rmse": 0.0, "r2": 1.0},
        target="y",
    )
    return str(path), frame.assign(y=y)


class TestBuildModelCard:
    def test_identity_and_provenance(self, bundle):
        path, _ = bundle
        card = build_model_card(path)
        assert card["name"] == "lin"
        assert card["task"] == "regression"
        assert card["target"] == "y"
        assert card["n_features"] == 3
        assert card["metrics"] == {"rmse": 0.0, "r2": 1.0}
        assert card["best_params"] == {"fit_intercept": True}
        assert card["provenance"]["sklearn_version"]
        assert card["provenance"]["pharmaml_version"]
        assert card["generated_at"]

    def test_estimator_params_are_json_safe(self, bundle):
        path, _ = bundle
        card = build_model_card(path)
        assert isinstance(card["estimator_params"], dict)
        json.dumps(card)  # must not raise

    def test_without_training_frame_domain_is_none(self, bundle):
        path, _ = bundle
        card = build_model_card(path)
        assert card["applicability_domain"] is None
        assert card["training_rows"] is None

    def test_with_training_frame_records_box_and_rows(self, bundle):
        path, frame = bundle
        card = build_model_card(path, training_frame=frame)
        assert card["training_rows"] == 60
        box = card["applicability_domain"]
        assert set(box) == {"a", "b", "c"}
        assert box["a"]["min"] <= box["a"]["max"]

    def test_training_frame_must_be_a_dataframe(self, bundle):
        path, _ = bundle
        with pytest.raises(DataValidationError, match="DataFrame"):
            build_model_card(path, training_frame=[1, 2, 3])


class TestRender:
    def test_markdown_has_the_required_sections(self, bundle):
        path, frame = bundle
        md = model_card_markdown(build_model_card(path, training_frame=frame))
        assert "# Model card: lin" in md
        assert "## Metrics (held-out)" in md
        assert "## Applicability domain" in md
        assert "## Intended use and limitations" in md
        assert "## Provenance" in md
        assert "| rmse | 0 |" in md

    def test_markdown_states_missing_domain_honestly(self, bundle):
        path, _ = bundle
        md = model_card_markdown(build_model_card(path))
        assert "Not computed" in md

    def test_html_is_a_real_document(self, bundle):
        path, frame = bundle
        html = model_card_html(build_model_card(path, training_frame=frame))
        assert html.startswith("<!DOCTYPE html>")
        assert "<h1>Model card: lin</h1>" in html
        assert "<table>" in html
        assert "<li>" in html

    def test_html_escapes_injection(self, tmp_path):
        model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
        path = tmp_path / "x.joblib"
        save_model(model, path, name="<script>alert(1)</script>", task="regression")
        html = model_card_html(build_model_card(path))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestWriteModelCard:
    def test_default_markdown_path(self, bundle, tmp_path):
        path, _ = bundle
        out = tmp_path / "card.md"
        written = write_model_card(path, out)
        assert written == out and out.is_file()
        assert out.read_text(encoding="utf-8").startswith("# Model card")

    def test_html_suffix_selects_html(self, bundle, tmp_path):
        path, _ = bundle
        out = tmp_path / "card.html"
        write_model_card(path, out)
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_json_suffix_round_trips(self, bundle, tmp_path):
        path, _ = bundle
        out = tmp_path / "card.json"
        write_model_card(path, out)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["name"] == "lin"

    def test_parent_directory_is_created(self, bundle, tmp_path):
        path, _ = bundle
        out = tmp_path / "nested" / "deep" / "card.md"
        write_model_card(path, out)
        assert out.is_file()
