"""Data table model/view tests: real frames, sorting, copy, empty state."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "name": [f"CMPD-{i:04d}" for i in range(12)],
            "pIC50": rng.normal(5.0, 1.0, 12).round(3),
            "logP": rng.normal(2.0, 0.5, 12).round(3),
        }
    )


def test_model_shape_and_headers(studio_app) -> None:
    from ui.widgets.data_table import PandasTableModel

    model = PandasTableModel(_frame())
    assert model.rowCount() == 12
    assert model.columnCount() == 3
    assert model.headerData(1, Qt.Orientation.Horizontal) == "pIC50"


def test_display_role_renders_values_and_nan(studio_app) -> None:
    from ui.widgets.data_table import PandasTableModel

    frame = _frame()
    frame.loc[3, "pIC50"] = float("nan")
    model = PandasTableModel(frame)
    index = model.index(3, 1)
    assert model.data(index, Qt.ItemDataRole.DisplayRole) == "—"
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "CMPD-0000"


def test_sort_reorders_rows(studio_app) -> None:
    from PySide6.QtCore import Qt
    from ui.widgets.data_table import PandasTableModel

    model = PandasTableModel(_frame())
    model.sort(1, Qt.SortOrder.DescendingOrder)
    values = [float(model.data(model.index(r, 1), Qt.ItemDataRole.DisplayRole)) for r in range(model.rowCount())]
    assert values == sorted(values, reverse=True)


def test_view_bind_and_status(studio_app) -> None:
    from ui.widgets.data_table import DataTableView

    view = DataTableView()
    view.bind_frame(_frame())
    assert "12 rows" in view.statusHint
    view.bind_frame(None, note="No dataset loaded")
    assert view.statusHint == "No dataset loaded"


def test_view_copy_selection(studio_app, qtbot) -> None:
    from ui.widgets.data_table import DataTableView

    view = DataTableView()
    view.bind_frame(_frame())
    text = view.copy_selection()  # no selection -> whole shown table
    assert text.splitlines()[0].split("\t") == ["name", "pIC50", "logP"]
    assert len(text.splitlines()) == 13


def test_truncation_guard(studio_app) -> None:
    from ui.widgets.data_table import PandasTableModel

    big = pd.DataFrame({"x": np.arange(6_000)})
    model = PandasTableModel(big)
    assert model.truncated and model.rowCount() == 5_000
