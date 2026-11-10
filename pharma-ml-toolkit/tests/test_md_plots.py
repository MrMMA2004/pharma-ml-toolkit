"""MD figures use actual series values and refuse unsafe correlations."""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.md import LigandResult, XvgSeries, build_study
from pharmaml.viz import close_figure, plot_md_correlation, plot_md_series, save_figure


def _series(values, time=None, label=""):
    values = np.asarray(values, dtype=float)
    return XvgSeries(np.arange(len(values)) if time is None else np.asarray(time), values, label=label)


def test_single_series_uses_actual_values_and_explicit_labels():
    source = _series([1, 2, 4], time=[0, 2, 4], label="ligand A")
    fig = plot_md_series(source, x_label="Time (ns)", y_label="RMSD (nm)")
    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 2, 4])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 2, 4])
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (ns)", "RMSD (nm)")
        assert ax.get_legend().get_texts()[0].get_text() == "ligand A"
    finally:
        close_figure(fig)


def test_study_comparison_and_exports(tmp_path):
    for name, values in {"A": "0 1\n1 2\n2 3\n", "B": "0 2\n1 4\n2 6\n"}.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "ligand_rmsd.xvg").write_text(values)
    study = build_study(tmp_path)
    fig = plot_md_series(study.metric_series("ligand_rmsd"))
    try:
        assert [line.get_label() for line in fig.axes[0].lines] == ["A", "B"]
        for ext in ("png", "svg", "pdf"):
            target = tmp_path / f"md.{ext}"
            save_figure(fig, target)
            assert target.stat().st_size > 0
    finally:
        close_figure(fig)


def test_correlation_ligand_values_and_undefined_constant(tmp_path):
    ligand = LigandResult("A", {"up": _series([1, 2, 3]),
                                 "down": _series([3, 2, 1]),
                                 "constant": _series([2, 2, 2])})
    fig = plot_md_correlation(ligand)
    try:
        ax = fig.axes[0]
        matrix = ax.images[0].get_array()
        assert matrix[0, 1] == pytest.approx(-1)
        assert np.ma.is_masked(matrix[2, 2])
        assert "N/A" in [text.get_text() for text in ax.texts]
        assert [tick.get_text() for tick in ax.get_xticklabels()] == list(ligand.series)
        for ext in ("png", "svg", "pdf"):
            target = tmp_path / f"corr.{ext}"
            save_figure(fig, target)
            assert target.stat().st_size > 0
    finally:
        close_figure(fig)


@pytest.mark.parametrize("bad", [{}, {"a": _series([1, 2])}, {"a": _series([1]), "b": _series([2])},
                                 {"a": _series([1, 2]), "b": _series([2, 3, 4])},
                                 {"a": _series([1, 2]), "b": _series([2, 3], time=[1, 2])},
                                 {"a": _series([1, np.nan]), "b": _series([2, 3])}])
def test_correlation_rejects_invalid_or_misaligned_data(bad):
    with pytest.raises(DataValidationError):
        plot_md_correlation(bad)


@pytest.mark.parametrize("bad", [{}, {"a": object()}, _series([1, float("inf")]), None])
def test_series_rejects_invalid_data(bad):
    with pytest.raises(DataValidationError):
        plot_md_series(bad)
