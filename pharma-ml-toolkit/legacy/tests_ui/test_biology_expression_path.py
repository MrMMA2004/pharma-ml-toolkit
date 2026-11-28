"""Biology expression-path sharing (R29-d).

Both the MAD and biomarker flows read ONE user-visible path row
(bundled default, Browse, live preview); editing it drops the
shared normalize cache so runs always use the file currently in
the field.
"""

from __future__ import annotations

import pandas as pd
from ui.pages.biology import _EXPRESSION_PATH, BiologyWorkbenchPage


def test_default_is_the_bundled_matrix(studio_app) -> None:
    page = BiologyWorkbenchPage()
    assert page._expression_path() == _EXPRESSION_PATH
    # the field shows the full default path
    assert page._expr_edit.text() == str(_EXPRESSION_PATH)


def test_preview_describes_the_bundled_matrix(studio_app) -> None:
    page = BiologyWorkbenchPage()
    text = page._expr_preview.text()
    assert "300 genes" in text
    assert text.startswith("expression matrix —")


def test_blank_field_falls_back_to_default(studio_app) -> None:
    page = BiologyWorkbenchPage()
    page._expr_edit.setText("   ")
    assert page._expression_path() == _EXPRESSION_PATH


def test_custom_path_is_used(studio_app, tmp_path) -> None:
    frame = pd.DataFrame(
        {"G1": [1.0, 2.0], "G2": [0.5, 0.25]},
        index=pd.Index(["lineA", "lineB"], name="cell_line"),
    )
    custom = tmp_path / "my_matrix.csv"
    frame.to_csv(custom)
    page = BiologyWorkbenchPage()
    page._expr_edit.setText(str(custom))
    page._expr_edit.editingFinished.emit()  # simulate commit
    assert page._expression_path() == custom
    assert "2 genes" in page._expr_preview.text()
    assert "lineA" in page._expr_preview.text()


def test_missing_path_reports_not_found(studio_app, tmp_path) -> None:
    page = BiologyWorkbenchPage()
    page._expr_edit.setText(str(tmp_path / "nope.csv"))
    page._expr_edit.editingFinished.emit()
    assert "not found" in page._expr_preview.text()


def test_malformed_csv_reports_unreadable(studio_app, tmp_path) -> None:
    # binary junk: pandas cannot parse this as CSV at all
    bad = tmp_path / "broken.csv"
    bad.write_bytes(bytes(range(256)) * 8)
    page = BiologyWorkbenchPage()
    page._expr_edit.setText(str(bad))
    page._expr_edit.editingFinished.emit()
    assert "unreadable" in page._expr_preview.text()


def test_path_change_drops_the_normalize_cache(studio_app, tmp_path) -> None:
    page = BiologyWorkbenchPage()
    page._expr_norm = object()  # pretend a run cached a matrix
    page._expr_edit.setText(str(tmp_path / "other.csv"))
    page._expr_edit.editingFinished.emit()
    assert page._expr_norm is None


def test_both_tabs_share_the_same_row(studio_app) -> None:
    """One QLineEdit instance serves both tabs (no second field)."""
    page = BiologyWorkbenchPage()
    # the add-row helper is idempotent by identity, not just text
    from PySide6.QtWidgets import QLineEdit

    edits = page.findChildren(QLineEdit)
    texts = [e.text() for e in edits]
    assert texts.count(str(_EXPRESSION_PATH)) == 1
