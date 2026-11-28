"""PK/PD Workbench page tests (increment 3): real engines, real series.

Everything runs offscreen against the same backend entry points the
playground cards call (``TwoCompartment`` / ``DosingEvent``,
``nca_summary`` + ``fit_terminal_phase``, ``ev``/``addl_until``/
``realize_addl``/``ev_days``); the assertions check the REAL numbers
those engines produce on the card's own default inputs.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _drain(pool_wait: float = 5.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots."""
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _page():
    from ui.pages.pkpd import PKPDWorkbenchPage

    return PKPDWorkbenchPage()


# -- construction -----------------------------------------------------------------


def test_page_constructs_with_three_tabs(studio_app) -> None:
    page = _page()
    assert page._tabs.count() == 3
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert labels == ["Regimen simulator", "Non-compartmental analysis", "Dosing events"]
    assert page._route_combo.currentText() == "oral"
    assert page._regimen_entries["dose_mg"].text() == "100"
    assert page._regimen_entries["tau_h"].text() == "24"
    assert page._nca_times.text().startswith("0, 0.5, 1, 2, 4")
    assert page._nca_concs.text().startswith("12.0, 10.2")
    assert page._ev_days.text() == "m, w, f"


def test_parse_helpers_surface_honest_errors(studio_app) -> None:
    from ui.pages.pkpd import _parse_float, _parse_int, _parse_series

    assert _parse_float(" 2.5 ", "cl") == 2.5
    try:
        _parse_float("abc", "ka")
    except ValueError as exc:
        assert "ka" in str(exc) and "abc" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        _parse_float("-1", "x", minimum=0.0)
    except ValueError as exc:
        assert "x must be >= 0" in str(exc)
    else:
        raise AssertionError("expected ValueError (minimum floors at parse level)")
    try:
        _parse_int("404", "n_doses", minimum=1, maximum=100)
    except ValueError as exc:
        assert "n_doses" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert _parse_series("0, 0.5, 1 2", "times") == [0.0, 0.5, 1.0, 2.0]


# -- regimen ----------------------------------------------------------------------


def test_regimen_simulation_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_regimen()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert text.startswith("Regimen — 3 oral dose(s)")
    assert "Cmax" in text and "AUC(0–72 h)" in text
    assert "matrix-exponential" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 3
    assert list(frame["time_h"]) == [0.0, 24.0, 48.0]
    assert list(frame["cmt"]) == [0, 0, 0]  # oral -> gut compartment
    assert (frame["amount_mg"] == 100.0).all()

    items = page._plot.widget().getPlotItem().listDataItems()
    assert items and items[0].getData()[0] is not None
    assert len(items[0].getData()[0]) == 145  # the 145-point time grid

    # dose markers: three dotted vertical lines at 0/24/48
    from pyqtgraph import InfiniteLine

    lines = [it for it in page._plot.widget().getPlotItem().items if isinstance(it, InfiniteLine)]
    assert sorted(float(line.value()) for line in lines) == [0.0, 24.0, 48.0]

    # the real engine numbers: with ka=1, CL=5, V2=50, Q=10, V3=100 the
    # central concentration peaks early and decays two-compartment.
    _assert_regimen_numbers(page)


def _assert_regimen_numbers(page) -> None:
    from pharmaml.pkpd.compartments import TwoCompartment
    from pharmaml.pkpd.ode import DosingEvent

    model = TwoCompartment(ka=1.0, cl=5.0, v2=50.0, q=10.0, v3=100.0)
    times = np.linspace(0.0, 72.0, 145)
    events = [
        DosingEvent(time=t, amount=100.0, compartment=0) for t in (0.0, 24.0, 48.0)
    ]
    amounts = model.amounts(times, 100.0, route="oral", events=events)
    expected = amounts[:, 1] / model.v2
    drawn = page._plot.widget().getPlotItem().listDataItems()[0].getData()[1]
    assert np.allclose(drawn, expected, rtol=1e-9)


def test_regimen_iv_route_switches_compartment(studio_app, qtbot) -> None:
    page = _page()
    page._route_combo.setCurrentText("iv")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_regimen()
    done.wait()
    _settle(qtbot, studio_app)

    frame = page.model_frame()
    assert frame is not None and len(frame) == 3
    assert list(frame["cmt"]) == [1, 1, 1]  # iv -> central compartment
    assert page.report_text().startswith("Regimen — 3 iv dose(s)")


def test_regimen_rejects_nonpositive_input(studio_app) -> None:
    page = _page()
    page._regimen_entries["dose_mg"].setText("0")
    page._start_regimen()
    assert "dose_mg must be > 0" in page.report_text()
    assert not page._runner.busy  # no job was started

    page._regimen_entries["dose_mg"].setText("100")
    page._regimen_entries["ka"].setText("nan-ish")
    page._start_regimen()
    assert "not a number" in page.report_text()


def test_regimen_n_doses_bounded(studio_app) -> None:
    page = _page()
    page._regimen_entries["n_doses"].setText("101")
    page._start_regimen()
    assert "between 1 and 100" in page.report_text()


# -- NCA ---------------------------------------------------------------------------


def test_nca_end_to_end_with_real_defaults(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_nca()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert text.startswith("NCA — 10 points")
    assert "trap" in text and "lin-up/log-down" in text
    assert "t½(λz)" in text and "window rows" in text
    assert "Vd" not in text  # no IV dose given

    frame = page.model_frame()
    assert frame is not None and len(frame) == 11
    metrics = dict(zip(frame["metric"], frame["value"]))
    # the card's own default series: one-compartment IV bolus, t1/2 ~ 4 h
    assert 50.0 < float(metrics["AUC (trapezoid)"]) < 65.0  # real value 56.485
    assert 3.5 < float(metrics["t½ terminal (trapezoid)"]) < 5.0  # real 4.233
    assert 50.0 < float(metrics["AUC (linear-up/log-down)"]) < 65.0  # real 55.398
    assert 0.99 < float(metrics["terminal R² (adj)"]) <= 1.0  # real 0.99997
    assert float(metrics["λz"]) > 0.15  # real 0.163 /h
    assert next(iter(frame["unit"])) == "mg·h/L"

    # observed profile: shaded line + sample markers
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 2
    line = items[0]
    assert line.curve.opts.get("fillLevel") == 0.0  # AUC shading
    assert line.curve.opts.get("brush") is not None  # the translucent fill brush
    x_data, _y = line.getData()
    assert len(x_data) == 10 and float(x_data[0]) == 0.0


def test_nca_with_iv_dose_adds_vd_and_cl(studio_app, qtbot) -> None:
    page = _page()
    page._nca_dose.setText("500")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_nca()
    done.wait()
    _settle(qtbot, studio_app)

    frame = page.model_frame()
    assert frame is not None and len(frame) == 13
    metrics = dict(zip(frame["metric"], frame["value"]))
    auc = float(metrics["AUC (trapezoid)"])
    assert abs(float(metrics["Vd (dose/AUC)"]) - 500.0 / auc) < 1e-9
    assert abs(float(metrics["CL (dose/AUC)"]) - 500.0 / auc) < 1e-9
    assert "Vd" in page.report_text() and "CL" in page.report_text()


def test_nca_mismatched_lengths_refused_before_job(studio_app) -> None:
    page = _page()
    page._nca_concs.setText("1, 2, 3")
    page._start_nca()
    assert "same number of points (10 vs 3)" in page.report_text()
    assert not page._runner.busy


def test_nca_invalid_series_surfaces_backend_error(studio_app, qtbot) -> None:
    page = _page()
    # unsorted times: validate_concentration_series must refuse (the card's
    # own validation, surfaced through the worker's failed channel)
    page._nca_times.setText("24, 0.5, 1, 2, 4, 6, 8, 12, 16, 0")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_nca()
    done.wait()
    _settle(qtbot, studio_app)
    assert "Job failed" in page.report_text()


# -- dosing events ------------------------------------------------------------------


def test_events_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_events()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    # addl_until(0, 12, 72) == 5 under the mrgsolve convention (doses at
    # 0, 12, ..., 60; the until boundary itself is not an addl dose)
    assert text.startswith("Events — 6 doses · total 3000 mg · +5 additional")
    assert "weekday schedule at 0h, 48h, 96h" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 9  # 6 addl + 3 weekday
    addl = frame[frame["source"] == "addl regimen"]
    weekly = frame[frame["source"] == "weekday schedule"]
    assert len(addl) == 6 and len(weekly) == 3
    assert list(addl["time_h"]) == [0.0, 12.0, 24.0, 36.0, 48.0, 60.0]
    assert list(weekly["time_h"]) == [0.0, 48.0, 96.0]  # m / w / f at ii=168
    assert (frame["evid"] == 1).all()
    assert addl["from_addl"].tolist() == [False] + [True] * 5


def test_events_without_weekday_codes(studio_app, qtbot) -> None:
    page = _page()
    page._ev_days.setText("")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_events()
    done.wait()
    _settle(qtbot, studio_app)

    frame = page.model_frame()
    assert frame is not None and len(frame) == 6
    assert "weekday schedule" not in page.report_text()


def test_events_rejects_bad_interval(studio_app) -> None:
    page = _page()
    page._ev_ii.setText("0")
    page._start_events()
    assert "interval must be > 0" in page.report_text()
    assert not page._runner.busy


# -- duplicate submissions / theming / integration -----------------------------------


def test_duplicate_submission_refused(studio_app, qtbot) -> None:
    page = _page()
    assert page._runner.submit("blocker", lambda progress: time.sleep(0.8)) is not None
    page._start_events()  # refused while the blocker runs
    assert "Busy: '" in page.report_text()
    # let the blocker finish AND flush its queued signals while the page
    # is still alive (destroying the runner first breaks _finish)
    qtbot.waitSignal(page._runner.job_ended, timeout=30_000).wait()
    _settle(qtbot, studio_app)


def test_palette_sync_reaches_plot(studio_app) -> None:
    from ui.theme_manager import ThemeManager

    theme = ThemeManager(studio_app)
    page = _page()
    theme.apply("qt-material", "light_teal.xml")
    page.apply_palette(theme.chart_palette())
    bg = page._plot.widget().backgroundBrush().color().name()
    theme.apply("qt-material", "dark_teal.xml")
    page.apply_palette(theme.chart_palette())
    bg2 = page._plot.widget().backgroundBrush().color().name()
    assert bg.lower() != bg2.lower()  # dark theme reached the plot


def test_main_window_integrates_pkpd_page(studio_app) -> None:
    from ui.main_window import MainWindow

    window = MainWindow()
    assert window.pkpd_page is not None
    window.go_to("pkpd")
    assert window.pages.currentIndex() == 3
    # theme switches reach the page's plot
    window.theme.apply("qt-material", "dark_teal.xml")
    bg = window.pkpd_page._plot.widget().backgroundBrush().color().name()
    assert bg.lower() == "#1d1f21"
    window.close()


def test_plot_panel_vlines_and_fill_survive_clear(studio_app) -> None:
    from pyqtgraph import InfiniteLine
    from ui.widgets.plot_widget import PlotPanel

    panel = PlotPanel()
    panel.line(np.arange(4, dtype=float), np.ones(4), fill=True)
    panel.vlines([1.0, 2.0])
    plot_item = panel.widget().getPlotItem()
    assert len([it for it in plot_item.items if isinstance(it, InfiniteLine)]) == 2
    assert plot_item.listDataItems()[0].curve.opts.get("fillLevel") == 0.0
    assert plot_item.listDataItems()[0].curve.opts.get("brush") is not None
    panel.clear()
    assert len([it for it in plot_item.items if isinstance(it, InfiniteLine)]) == 0


def test_pandas_frame_of_events_is_wellformed(studio_app, qtbot) -> None:
    """The event table frame rides the standard model/view path."""
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_events()
    done.wait()
    _settle(qtbot, studio_app)
    frame = page.model_frame()
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == [
        "time_h", "amount_mg", "cmt", "evid", "addl", "from_addl", "source",
    ]
