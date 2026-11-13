"""Deep Learning Efficiency + GNN lab tests — part6 Phase 3, batch 3.

The Deep Learning cluster's GUI workflows (the Efficiency lab and the
GNN lab, added to the Deep Learning Workbench in v1.2.189) over the
canonical backends: ``ml.pruning``'s magnitude mask + Zhu-Gupta ramp,
``ml.quantization``'s three-method report, ``ml.harness``'s ZeRO
Table-3 arithmetic, and ``ml.geometric``'s motif dataset →
GNNClassifier → gnn_explain chain.  Everything runs offscreen with
real inputs and real backends; the asserted numbers are checked
against the same engines' own docstring examples.
"""

from __future__ import annotations

import time


def _drain(pool_wait: float = 180.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_page():
    from ui.pages.deep_learning import DeepLearningWorkbenchPage

    return DeepLearningWorkbenchPage()


# -- construction ------------------------------------------------------------------


def test_labs_construct_with_defaults(studio_app) -> None:
    page = _fresh_page()
    # the Efficiency lab's defaults
    assert page._eff_params.value() == 256
    assert page._eff_sparsity.value() == 0.5
    assert page._eff_scope.currentText() == "local"
    assert page._eff_steps.value() == 4
    assert page._eff_world.value() == 4
    assert page._eff_millions.value() == 1.0
    # the GNN lab's defaults
    assert page._gnn_motif.currentText() == "house"
    assert page._gnn_graphs.value() == 120
    assert page._gnn_layer.currentText() == "gcn"
    assert page._gnn_epochs.value() == 200
    assert page._gnn_sparsity.value() == 0.7
    assert page._gnn_seed.value() == 27
    # both labs' buttons ride the page's busy-guard registry
    assert page._eff_prune_btn in page._buttons()
    assert page._gnn_train_btn in page._buttons()


# -- the efficiency lab workflows ----------------------------------------------------


def test_prune_workflow_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._eff_params.setValue(64)
    page._eff_sparsity.setValue(0.5)
    page._eff_scope.setCurrentText("global")
    page._eff_steps.setValue(3)
    page._start_eff_prune()
    _settle(qtbot, qapp)

    assert "one-shot 50.0%" in page._report.text()
    assert "ramp" in page._report.text()
    assert "final 50.0%" in page._report.text()
    assert "2 Linear layers" in page._report.text()
    bound = page._table.model()._frame
    assert bound is not None
    # one-shot rows + ramp-final rows, both over 2 layers
    assert len(bound) == 4
    assert set(bound["kind"]) == {"one-shot", "ramp-final"}
    # a global 0.5 prune zeroes half of every layer's weights
    one_shot = bound[bound["kind"] == "one-shot"]
    assert abs(float(one_shot["sparsity"].mean()) - 0.5) < 1e-6


def test_quantization_workflow_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._eff_params.setValue(128)
    page._start_eff_quant()
    _settle(qtbot, qapp)

    status = page._report.text()
    assert "weight 128x8" in status
    assert "nf4 4.15625 bpw" in status
    assert "int8_channel 8.0 bpw" in status
    bound = page._table.model()._frame
    assert bound is not None
    assert list(bound["method"]) == ["nf4", "int8", "int8_channel"]
    # the ordering the backends' own docstrings pin: per-channel int8
    # beats per-tensor int8 beats nf4 on relative error
    by_method = bound.set_index("method")["relative_error"]
    assert by_method["int8_channel"] < by_method["int8"] < by_method["nf4"]
    assert float(bound.loc[bound["method"] == "nf4", "bits_per_weight"].iloc[0]) == 4.156
    # int8 keeps its 8 bits per weight at both granularities
    assert float(bound.loc[bound["method"] == "int8", "bits_per_weight"].iloc[0]) == 8.0


def test_zero_workflow_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._eff_millions.setValue(1.0)
    page._eff_world.setValue(4)
    page._start_eff_zero()
    _settle(qtbot, qapp)

    status = page._report.text()
    assert "1M params" in status
    assert "world size 4" in status
    assert "saved" in status
    bound = page._table.model()._frame
    assert bound is not None
    assert list(bound["stage"]) == [0, 1, 2, 3]
    assert list(bound["partitioned"]) == [
        "nothing", "optimizer", "grads+optimizer", "params+grads+optimizer",
    ]
    # the harness's Table-3 arithmetic: fp16 params + grads (2 bytes
    # each) + the 12-byte Adam state pair per parameter, replicated at
    # stage 0: 1M params -> 2/2/12 MB (16 MB total per rank)...
    zero = bound.set_index("stage")
    assert float(zero.loc[0, "params_MB"]) == 2.0
    assert float(zero.loc[0, "grads_MB"]) == 2.0
    assert float(zero.loc[0, "optimizer_MB"]) == 12.0
    assert float(zero.loc[0, "total_MB"]) == 16.0
    # ...and stage 3 partitions every bucket across 4 ranks: 0.5/0.5/3
    assert float(zero.loc[3, "params_MB"]) == 0.5
    assert float(zero.loc[3, "grads_MB"]) == 0.5
    assert float(zero.loc[3, "optimizer_MB"]) == 3.0
    assert float(zero.loc[3, "total_MB"]) == 4.0


# -- the GNN lab workflow ------------------------------------------------------------


def test_gnn_lab_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._gnn_graphs.setValue(60)   # small but learnable
    page._gnn_epochs.setValue(150)
    page._gnn_layer.setCurrentText("gcn")
    page._start_gnn_lab()
    _settle(qtbot, qapp)

    status = page._report.text()
    assert "train accuracy" in status
    assert "label" in status
    assert "motif edges in mask" in status
    assert "fidelity" in status
    bound = page._table.model()._frame
    assert bound is not None
    # the top-12 scored edges table
    assert list(bound.columns) == ["edge", "score", "kept", "in_true_motif"]
    assert 1 <= len(bound) <= 12
    # the house motif has 6 edges and the mask keeps 30% of ~80
    assert "6" in status.split("motif edges in mask")[1][:6]
    # a GCN over 60 house graphs reaches well above chance (0.5)
    accuracy = float(status.split("train accuracy ")[1].split("%")[0])
    assert accuracy >= 65.0


# -- honest refusals -----------------------------------------------------------------


def test_busy_guard_blocks_concurrent_runs(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._gnn_graphs.setValue(200)
    page._gnn_epochs.setValue(300)
    page._start_gnn_lab()
    # while the GNN is training, every lab button must refuse
    from PySide6.QtCore import QThreadPool

    assert QThreadPool.globalInstance().activeThreadCount() > 0
    page._start_eff_prune()
    assert "already running" in page._report.text()
    page._start_eff_quant()
    page._start_eff_zero()
    assert "already running" in page._report.text()
    _settle(qtbot, qapp)
