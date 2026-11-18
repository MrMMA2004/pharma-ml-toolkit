"""MD figures use actual series values and refuse unsafe correlations.

The round-250 union of the two MD-plot implementations: the released
stats-dict twins (``tests/test_viz_md_plots.py`` carries the
hand-computed numbers and the byte-reproducibility pins) plus this
file's Studio-page contract — figures stay open with ``close=False``,
mappings draw the cross-ligand comparison, LigandResult is accepted
directly, ±inf is refused, NaN correlations are disclosed, and
misaligned abscissas are refused outright ("axes must match") because a
figure may not correlate observations never taken at the same moments.
"""

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
    stats = plot_md_series(source, x_label="Time (ns)", y_label="RMSD (nm)",
                           close=False)
    fig = stats["figure"]
    try:
        ax = fig.axes[0]
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 2, 4])
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 2, 4])
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (ns)", "RMSD (nm)")
        assert ax.get_legend().get_texts()[0].get_text() == "ligand A"
        assert stats["closed"] is False
    finally:
        close_figure(fig)


def test_study_comparison_and_exports(tmp_path):
    for name, values in {"A": "0 1\n1 2\n2 3\n", "B": "0 2\n1 4\n2 6\n"}.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / "ligand_rmsd.xvg").write_text(values)
    study = build_study(tmp_path)
    stats = plot_md_series(study.metric_series("ligand_rmsd"), close=False)
    fig = stats["figure"]
    try:
        assert stats["n_series"] == 2 and stats["labels"] == ["A", "B"]
        assert [line.get_label() for line in fig.axes[0].lines] == ["A", "B"]
        assert stats["series"]["A"]["mean"] == pytest.approx(2.0)
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
    stats = plot_md_correlation(ligand, close=False)
    fig = stats["figure"]
    try:
        ax = fig.axes[0]
        matrix = ax.images[0].get_array()
        assert matrix[0, 1] == pytest.approx(-1)
        assert np.ma.is_masked(matrix[2, 2])
        assert "N/A" in [text.get_text() for text in ax.texts]
        assert stats["nan_cells"] == 2 * 3 - 1
        assert [tick.get_text() for tick in ax.get_xticklabels()] == list(ligand.series)
        for ext in ("png", "svg", "pdf"):
            target = tmp_path / f"corr.{ext}"
            save_figure(fig, target)
            assert target.stat().st_size > 0
    finally:
        close_figure(fig)


@pytest.mark.parametrize("bad", [{}, {"a": _series([1, 2])}, {"a": _series([1]), "b": _series([2])},
                                 {"a": _series([1, 2], time=[0, 1]),
                                  "b": _series([2, 3], time=[1, 2])},
                                 {"a": _series([1, float("inf")]),
                                  "b": _series([2, 3])}])
def test_correlation_rejects_invalid_misaligned_or_infinite_data(bad):
    with pytest.raises(DataValidationError):
        plot_md_correlation(bad)


def test_correlation_truncates_aligned_prefix_and_discloses_nan():
    # Aligned grids with different lengths truncate to the common prefix
    # (the pipeline's rule); a NaN pair discloses, never lies.
    stats = plot_md_correlation({
        "a": _series([1.0, 2.0, 3.0]),
        "b": _series([1.0, np.nan, 3.0]),
    })
    assert stats["n_points_aligned"] == 3
    # The (a,b) pair is NaN on both matrix sides AND b's self-cell: a
    # NaN-bearing series correlates with nothing, not even itself.
    assert stats["nan_cells"] == 3
    assert np.isnan(stats["r"][0, 1])


@pytest.mark.parametrize("bad", [{}, {"a": object()}, _series([1, float("inf")]), None])
def test_series_rejects_invalid_data(bad):
    with pytest.raises(DataValidationError):
        plot_md_series(bad)


def test_series_tolerates_nan_but_refuses_inf():
    stats = plot_md_series(_series([1.0, np.nan, 3.0]))
    assert stats["mean"] == pytest.approx(2.0)  # NaN-aware, disclosed
    with pytest.raises(DataValidationError, match="inf"):
        plot_md_series(_series([1.0, float("inf")]))


def test_smooth_window_is_single_series_only():
    mapping = {"A": _series([1.0, 2.0, 3.0], label="A")}
    with pytest.raises(DataValidationError, match="single"):
        plot_md_series(mapping, smooth_window=3)
    stats = plot_md_series(mapping["A"], smooth_window=3)
    assert len(stats["smoothed"]) == 3
