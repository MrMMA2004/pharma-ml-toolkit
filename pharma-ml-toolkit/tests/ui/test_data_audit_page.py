"""Data Explorer quality controls call the real audit and discard stale results."""

from ui.pages.data_explorer import DataExplorerPage


def test_data_audit_and_reload(qtbot, studio_app, tmp_path):
    page = DataExplorerPage()
    qtbot.addWidget(page)
    path = tmp_path / "data.csv"
    path.write_text("id,split,label,x\na,train,yes,1\na,test,no,2\nb,test,yes,\n")
    page.load_path(path)
    page._audit_key.setCurrentText("id")
    page._audit_split.setCurrentText("split")
    page._audit_target.setCurrentText("label")
    page._audit_button.click()
    qtbot.waitUntil(lambda: page._audit_button.isEnabled(), timeout=5000)
    text = page._audit_output.toPlainText()
    assert "Shared exact keys: 1" in text and "Duplicate keys (extra occurrences): 1" in text
    assert "cannot rule out" in text
    page.load_path(tmp_path / "missing.csv")
    assert page._audit_output.toPlainText() == ""
    page._audit_button.click()
    assert "Load a dataset first" in page._audit_output.toPlainText()


def test_audit_replaces_old_result_on_error(qtbot, studio_app, tmp_path):
    page = DataExplorerPage()
    qtbot.addWidget(page)
    path = tmp_path / "data.csv"
    path.write_text("id,split\na,train\n,test\n")
    page.load_path(path)
    page._audit_button.click()
    qtbot.waitUntil(lambda: page._audit_button.isEnabled(), timeout=5000)
    assert "Rows: 2" in page._audit_output.toPlainText()
    page._audit_key.setCurrentText("id")
    page._audit_split.setCurrentText("split")
    page._audit_button.click()
    qtbot.waitUntil(lambda: page._audit_button.isEnabled(), timeout=5000)
    text = page._audit_output.toPlainText()
    assert text.startswith("Audit failed:") and "Rows: 2" not in text


def test_reload_ignores_inflight_result(qtbot, studio_app, tmp_path, monkeypatch):
    import threading

    from pharmaml.ml import research_audit

    started, release = threading.Event(), threading.Event()
    original = research_audit.audit_dataset

    def delayed(*args, **kwargs):
        started.set()
        if not release.wait(timeout=5):
            raise RuntimeError("timed out")
        return original(*args, **kwargs)

    monkeypatch.setattr(research_audit, "audit_dataset", delayed)
    page = DataExplorerPage()
    qtbot.addWidget(page)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    first.write_text("x\n1\n2\n")
    second.write_text("y\n3\n")
    page.load_path(first)
    page._audit_button.click()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        page.load_path(second)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=5000)
    assert page._audit_output.toPlainText() == ""
