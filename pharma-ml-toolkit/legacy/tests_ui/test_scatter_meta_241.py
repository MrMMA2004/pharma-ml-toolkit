"""Round 241 -- Part-3 §21 completion: per-point hover meta on the
remaining scatter surfaces.

The parity / residual / fold / dose / cluster / sample / embedding
scatters now carry per-point metadata providers (the §21.5 contract
``_molecule_meta`` pioneered on the generative page): only facts the
plot itself cannot tell -- the 0-based source row, the signed
residual, the original-scale dose, the fold's delta against the mean,
the cluster's member count -- computed lazily per point, cached by the
inspector (§21.6), and never raising: out-of-range indexes answer
``{}``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from ui.pages.deep_learning import _embedding_meta
from ui.pages.deep_learning import _parity_meta as _dl_parity_meta
from ui.pages.model_analysis import _fold_meta, _parity_meta, _residual_meta
from ui.pages.pkpd import _sample_meta
from ui.pages.responseai import _dose_meta, _pair_residual_meta, _row_meta
from ui.pages.unsupervised import _cluster_meta
from ui.widgets.plot_widget import PlotPanel


@pytest.fixture()
def panel(studio_app):
    panel = PlotPanel()
    yield panel
    panel.close()


class TestParityMeta:
    def test_row_and_signed_residual(self):
        meta = _parity_meta(np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.0, 2.5]))
        assert meta(0) == {"row": 0, "residual": 0.5}
        assert meta(1) == {"row": 1, "residual": 0.0}
        assert meta(2) == {"row": 2, "residual": -0.5}

    def test_out_of_range_answers_empty(self):
        meta = _parity_meta(np.array([1.0]), np.array([1.0]))
        assert meta(-1) == {}
        assert meta(1) == {}
        assert meta(10**6) == {}

    def test_nan_inputs_do_not_raise(self):
        meta = _parity_meta(np.array([1.0, np.nan]), np.array([2.0, 4.0]))
        point = meta(1)
        assert point["row"] == 1
        assert math.isnan(point["residual"])

    def test_page_duplicates_share_one_contract(self):
        a = _parity_meta(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        b = _dl_parity_meta(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        assert a(1) == b(1) == {"row": 1, "residual": 1.0}


class TestResidualMeta:
    def test_discloses_the_actual_value(self):
        meta = _residual_meta(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert meta(0) == {"row": 0, "actual": 1.0}
        assert meta(1) == {"row": 1, "actual": 2.0}
        assert meta(7) == {}


class TestFoldMeta:
    def test_one_based_fold_and_signed_delta(self):
        meta = _fold_meta(np.array([0.8, 0.9, 0.7]), mean=0.8)
        assert meta(0) == {"fold": 1, "delta vs mean": 0.0}
        assert meta(1) == {"fold": 2, "delta vs mean": pytest.approx(0.1)}
        assert meta(2) == {"fold": 3, "delta vs mean": pytest.approx(-0.1)}
        assert meta(3) == {}


class TestSampleMeta:
    def test_one_based_sample_number(self):
        meta = _sample_meta(3)
        assert meta(0) == {"sample": 1}
        assert meta(2) == {"sample": 3}
        assert meta(3) == {}
        assert meta(-2) == {}


class TestDoseMeta:
    def test_original_scale_dose_not_log10(self):
        meta = _dose_meta(np.array([1e-2, 1.0, 10.0]))
        assert meta(0) == {"dose": 0.01, "row": 0}
        assert meta(2) == {"dose": 10.0, "row": 2}
        assert meta(3) == {}


class TestPairResidualMeta:
    def test_metrics_and_screen_pairs_share_the_factory(self):
        meta = _pair_residual_meta(np.array([5.0, 6.0]), np.array([5.0, 7.0]))
        assert meta(0) == {"row": 0, "residual": 0.0}
        assert meta(1) == {"row": 1, "residual": 1.0}
        assert meta(-1) == {}


class TestRowMeta:
    def test_correlated_pair_row_only(self):
        meta = _row_meta(4)
        assert meta(3) == {"row": 3}
        assert meta(4) == {}


class TestEmbeddingMeta:
    def test_row_points_at_the_source_table(self):
        meta = _embedding_meta(6)
        assert meta(5) == {"row": 5}
        assert meta(6) == {}


class TestClusterMeta:
    def test_source_row_and_member_count(self):
        labels = np.array([0, 1, 1, 0, 1, 2])
        meta_one = _cluster_meta(labels, 1)
        assert meta_one(0) == {"row": 1, "cluster members": 3}
        assert meta_one(2) == {"row": 4, "cluster members": 3}
        meta_zero = _cluster_meta(labels, 0)
        assert meta_zero(1) == {"row": 3, "cluster members": 2}
        assert meta_zero(2) == {}

    def test_out_of_range_cluster_answers_empty(self):
        meta = _cluster_meta(np.array([0, 1]), 7)
        assert meta(0) == {}


class TestWiring:
    def test_parity_meta_flows_through_the_inspector(self, panel):
        test = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 2.0, 2.5])
        panel.scatter(test, pred, label="held-out", meta=_parity_meta(test, pred))
        assert panel._inspector.inspect_at((3.0, 2.5)) == ("held-out", 2)
        text = panel._readout.text()
        assert "row = 2" in text
        assert "residual = -0.5" in text  # 2.5 - 3.0, exact (§21.3)

    def test_dose_meta_flows_through_the_inspector(self, panel):
        doses = np.array([0.01, 10.0])
        panel.scatter(
            np.log10(doses),
            np.array([0.2, 0.9]),
            label="measured",
            meta=_dose_meta(doses),
        )
        assert panel._inspector.inspect_at((1.0, 0.9)) == ("measured", 1)
        text = panel._readout.text()
        assert "dose = 10" in text  # the ORIGINAL scale, not the log10 x
        assert "row = 1" in text

    def test_cluster_meta_flows_through_the_inspector(self, panel):
        labels = np.array([0, 1, 1, 0, 1])
        xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ys = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mask = labels == 1
        panel.scatter(
            xs[mask],
            ys[mask],
            label="cluster 1",
            meta=_cluster_meta(labels, 1),
        )
        assert panel._inspector.inspect_at((4.0, 4.0)) == ("cluster 1", 2)
        text = panel._readout.text()
        assert "row = 4" in text  # the flat source row, not the mask slot
        assert "cluster members = 3" in text
