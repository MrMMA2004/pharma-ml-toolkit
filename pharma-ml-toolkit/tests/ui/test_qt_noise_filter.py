"""Qt console-noise filter tests (round 231 — the user-reported fix).

The verbatim console lines a Windows user reported against v1.2.142
(pip PySide6 6.8.x under Miniconda) must be dropped; everything else
must survive.  The live round-trip drives ``QtCore.qWarning`` through
the real ``qInstallMessageHandler`` path with ``capsys`` capturing the
handler's ``sys.stderr`` writes, so the assertions exercise the exact
plumbing Studio uses at startup.
"""

from __future__ import annotations

import inspect


def _import_module():
    from ui import qt_noise_filter

    return qt_noise_filter


# -- the verbatim report (user console, v1.2.142, Windows/miniconda) -------------

FONTS_DIR_LINE = (
    "QFontDatabase: Cannot find font directory "
    "C:/Users/np/miniconda3/envs/py312/Lib/site-packages/PySide6/lib/fonts."
)
FONTS_NOTE_LINE = (
    "Note that Qt no longer ships fonts. Deploy some "
    "(from https://dejavu-fonts.github.io/ for example) or switch to fontconfig."
)
STYLEHINTS_BODY = (
    "QObject::connect(QStyleHints, QStyleHints): "
    "unique connections require a pointer to member function of a QObject subclass"
)


def test_both_reported_windows_families_are_benign() -> None:
    module = _import_module()
    for line in (FONTS_DIR_LINE, FONTS_NOTE_LINE, STYLEHINTS_BODY):
        assert module.is_benign_qt_noise(line), line


def test_real_messages_survive_the_markers() -> None:
    module = _import_module()
    reals = [
        "",
        "QFont: Cannot create font family from a null descriptor",
        "QFontDatabase: loading the font file failed (permission denied)",
        "QObject::connect(QStyleHints, QScreen): signal not found",  # different pair
        "QOpenGLShaderProgram: shader compilation failed",
        "Cannot find font directory",  # benign by design (wheel notice)
    ]
    for line in reals[:-1]:
        assert not module.is_benign_qt_noise(line), line
    assert module.is_benign_qt_noise(reals[-1])


def test_live_roundtrip_filters_benign_and_passes_real(capsys) -> None:
    """qWarning -> handler -> stderr: benign dropped, real kept."""
    module = _import_module()
    from PySide6.QtCore import qInstallMessageHandler, qWarning

    baseline = qInstallMessageHandler(None)  # reset to the default handler
    assert module.install_qt_message_filter() is True
    try:
        qWarning(STYLEHINTS_BODY)
        qWarning(FONTS_DIR_LINE)
        qWarning(FONTS_NOTE_LINE)
        dropped = capsys.readouterr()
        assert "QStyleHints" not in dropped.err
        assert "font directory" not in dropped.err
        assert "no longer ships fonts" not in dropped.err

        qWarning("a real warning must survive the filter")
        kept = capsys.readouterr()
        assert "a real warning must survive the filter" in kept.err
    finally:
        qInstallMessageHandler(baseline)


def test_verbose_escape_hatch(monkeypatch) -> None:
    """PHARMAML_STUDIO_QT_VERBOSE=1 disables installation entirely."""
    module = _import_module()
    monkeypatch.setenv("PHARMAML_STUDIO_QT_VERBOSE", "1")
    assert module.install_qt_message_filter() is False
    monkeypatch.setenv("PHARMAML_STUDIO_QT_VERBOSE", "0")
    assert module.install_qt_message_filter() is True


def test_run_installs_the_filter_before_the_application_object() -> None:
    """Both warning families fire during QApplication CONSTRUCTION, so
    the filter must be installed before it -- the ordering is the fix."""
    from ui import run

    source = inspect.getsource(run)
    filter_at = source.index("install_qt_message_filter()")
    app_at = source.index("QApplication(")
    assert filter_at < app_at, "filter must run before the QApplication is built"
    assert "ensure_headless()" in source
