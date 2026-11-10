"""Tests for :mod:`pharmaml.viz.md_plots` — the MD viz twins.

The round-249 follow-on: the uploaded pipeline's two headline panels (the
per-metric line chart and the cross-metric correlation heatmap) as
first-class pharmaml figures.  The honest-test rule applies to figures
too: the returned stats dicts are checked against hand-computed values
(an arithmetic ramp's exact mean/std, a designed ±1 correlation, the NaN
disclosure a constant series owes), the styling resolution is checked
against the ten-metric classifier, and the byte-reproducible save
contract is checked by hashing two saves of the same figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from pharmaml.exceptions import DataValidationError
from pharmaml.md.xvg import XvgSeries, moving_average
from pharmaml.viz._compat import PALETTE
from pharmaml.viz.md_plots import plot_md_correlation, plot_md_series


def _ramp(label: str = "ligand_rmsd", n: int = 20) -> XvgSeries:
    """0.1 + 0.0005·t over 0..1900 ps — mean 0.575, pop-std 0.28831…"""
    t = np.arange(0.0, float(100 * n), 100.0)
    return XvgSeries(time=t, values=0.1 + 0.0005 * t, label=label)


def _flat(label: str = "protein_backbone_Rg", n: int = 20) -> XvgSeries:
    """A constant series — correlates with nothing, honestly."""
    t = np.arange(0.0, float(100 * n), 100.0)
    return XvgSeries(time=t, values=np.full(n, 1.4), label=label)


class TestPlotMDSeries:
    def test_stats_are_hand_computed(self) -> None:
        stats = plot_md_series(_ramp())
        assert stats["n"] == 20
        assert stats["mean"] == pytest.approx(0.575)
        assert stats["std"] == pytest.approx(0.05 * np.sqrt(399 / 12))
        assert stats["cv"] == pytest.approx(stats["std"] / stats["mean"])
        assert (stats["t_min"], stats["t_max"]) == (0.0, 1900.0)
        assert (stats["v_min"], stats["v_max"]) == (0.1, 1.05)

    def test_known_metric_styling_from_classifier(self) -> None:
        stats = plot_md_series(_ramp())
        assert stats["title"] == "Ligand RMSD"
        assert stats["x_label"] == "Time (ns)"
        assert stats["y_label"] == "RMSD (Å)"
        assert stats["color"] == "#7F1D1D"

    def test_kind_inference_for_unknown_names(self) -> None:
        # "replica_energy_2" is not one of the ten standard exports but
        # the classifier infers the energy kind from the substring.
        t = np.arange(5.0)
        series = XvgSeries(time=t, values=t * 2.0, label="replica_energy_2")
        stats = plot_md_series(series)
        assert stats["title"] == "Replica Energy 2"
        assert stats["x_label"] == "Frame"  # the inferred kind's axis
        assert stats["y_label"] == "Value"  # inferred kinds carry no unit
        assert stats["color"] == "#525252"  # energy → neutral-600

    def test_unknown_label_gets_honest_generics(self) -> None:
        stats = plot_md_series(_ramp(label="weird_metric"))
        assert stats["title"] == "weird_metric"
        assert (stats["x_label"], stats["y_label"]) == ("Time", "Value")
        assert stats["color"] == PALETTE[0]

    def test_title_and_color_overrides_win(self) -> None:
        stats = plot_md_series(_ramp(), title="Custom", color="#123456")
        assert stats["title"] == "Custom"
        assert stats["color"] == "#123456"

    def test_smoothed_overlay_matches_the_pipeline(self) -> None:
        ramp = _ramp()
        stats = plot_md_series(ramp, smooth_window=5)
        assert stats["smooth_window"] == 5
        assert np.allclose(stats["smoothed"], moving_average(ramp.values, 5))
        assert len(stats["smoothed"]) == 20  # centered: same length

    def test_smoothed_save_is_byte_reproducible(self, tmp_path: Path) -> None:
        ramp = _ramp()
        first = tmp_path / "a.png"
        second = tmp_path / "b.png"
        plot_md_series(ramp, smooth_window=3, path=first)
        plot_md_series(ramp, smooth_window=3, path=second)
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_size > 0

    def test_svg_and_pdf_paths(self, tmp_path: Path) -> None:
        for name in ("fig.svg", "fig.pdf"):
            target = tmp_path / name
            plot_md_series(_ramp(), path=target)
            assert target.stat().st_size > 0

    def test_nan_series_is_nan_aware(self) -> None:
        t = np.arange(6.0)
        values = np.array([1.0, np.nan, 3.0, 5.0, 7.0, 9.0])
        stats = plot_md_series(XvgSeries(time=t, values=values))
        assert stats["n"] == 6
        assert stats["mean"] == pytest.approx(5.0)
        assert stats["v_min"] == pytest.approx(1.0)
        assert stats["v_max"] == pytest.approx(9.0)

    @pytest.mark.parametrize("window", [1, 0, -3, True, "5"])
    def test_smooth_window_validation(self, window: object) -> None:
        with pytest.raises(DataValidationError, match="smooth_window"):
            plot_md_series(_ramp(), smooth_window=window)  # type: ignore[arg-type]

    def test_smooth_window_longer_than_series(self) -> None:
        with pytest.raises(DataValidationError, match="exceeds the series"):
            plot_md_series(_ramp(n=4), smooth_window=5)

    @pytest.mark.parametrize("bad", [None, [1.0, 2.0], "ligand_rmsd", 7])
    def test_non_series_input_raises(self, bad: object) -> None:
        with pytest.raises(DataValidationError, match="XvgSeries"):
            plot_md_series(bad)  # type: ignore[arg-type]

    def test_unsupported_format_named(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="Unsupported figure"):
            plot_md_series(_ramp(), path=tmp_path / "fig.bmp")


class TestPlotMDCorrelation:
    def test_designed_perfect_correlations(self) -> None:
        t = np.arange(10.0)
        linear = XvgSeries(time=t, values=t * 1.0, label="a")
        mirrored = XvgSeries(time=t, values=10.0 - t, label="b")
        corr = plot_md_correlation({"a": linear, "b": mirrored})
        assert corr["labels"] == ["a", "b"]
        assert corr["r"][0, 0] == pytest.approx(1.0)
        assert corr["r"][0, 1] == pytest.approx(-1.0)
        assert corr["nan_cells"] == 0

    def test_min_length_alignment_is_disclosed(self) -> None:
        t_long = np.arange(12.0)
        t_short = np.arange(7.0)
        corr = plot_md_correlation({
            "a": XvgSeries(time=t_long, values=t_long),
            "b": XvgSeries(time=t_short, values=t_short * -1.0),
        })
        assert corr["n_series"] == 2
        assert corr["n_points_aligned"] == 7
        assert corr["r"][0, 1] == pytest.approx(-1.0)

    def test_constant_series_nan_disclosure(self) -> None:
        t = np.arange(8.0)
        corr = plot_md_correlation({
            "a": XvgSeries(time=t, values=t),
            "flat": _flat(n=8),
            "c": XvgSeries(time=t, values=t[::-1].copy()),
        })
        k = 3
        assert corr["r"].shape == (k, k)
        # a constant series correlates with nothing: its whole row AND
        # column are NaN (the shared diagonal cell counted once).
        assert corr["nan_cells"] == 2 * k - 1
        flat_idx = corr["labels"].index("flat")
        assert np.all(np.isnan(corr["r"][flat_idx, :]))
        assert np.all(np.isnan(corr["r"][:, flat_idx]))

    def test_display_labels_untangle_underscores(self) -> None:
        t = np.arange(5.0)
        corr = plot_md_correlation({
            "ligand_rmsd": XvgSeries(time=t, values=t),
            "hbnum_intermolecular": XvgSeries(time=t, values=t + 1.0),
        })
        assert corr["display_labels"] == ["ligand rmsd", "hbnum intermolecular"]
        assert corr["labels"] == ["ligand_rmsd", "hbnum_intermolecular"]

    def test_heatmap_save_is_byte_reproducible(self, tmp_path: Path) -> None:
        t = np.arange(9.0)
        mapping = {
            "a": XvgSeries(time=t, values=t),
            "b": XvgSeries(time=t, values=2.0 * t + np.sin(t)),
            "c": XvgSeries(time=t, values=-t),
        }
        first, second = tmp_path / "c1.png", tmp_path / "c2.png"
        plot_md_correlation(mapping, path=first)
        plot_md_correlation(mapping, path=second)
        assert first.read_bytes() == second.read_bytes()

    def test_alternative_diverging_colormap(self, tmp_path: Path) -> None:
        t = np.arange(6.0)
        target = tmp_path / "puor.png"
        corr = plot_md_correlation(
            {"a": XvgSeries(time=t, values=t),
             "b": XvgSeries(time=t, values=-t)},
            path=target,
            cmap="PuOr",
        )
        assert target.stat().st_size > 0
        assert corr["r"][0, 1] == pytest.approx(-1.0)

    def test_needs_two_non_empty_series(self) -> None:
        # An XvgSeries cannot be constructed empty (its __post_init__
        # refuses), so the only reachable under-supply is a single series.
        with pytest.raises(
            DataValidationError, match="at least two non-empty series — got 1"
        ):
            plot_md_correlation({"a": _ramp()})

    def test_common_length_under_two(self) -> None:
        one = XvgSeries(time=np.array([0.0]), values=np.array([1.0]), label="a")
        two = XvgSeries(time=np.array([0.0]), values=np.array([2.0]), label="b")
        with pytest.raises(DataValidationError, match="common length >= 2"):
            plot_md_correlation({"a": one, "b": two})

    def test_input_contract(self) -> None:
        ramp = _ramp()
        with pytest.raises(DataValidationError, match="non-empty"):
            plot_md_correlation({})
        with pytest.raises(DataValidationError, match="non-empty"):
            plot_md_correlation([ramp, ramp])  # type: ignore[arg-type]
        with pytest.raises(DataValidationError, match="must be an XvgSeries"):
            plot_md_correlation({"a": ramp, "bad": "nope"})  # type: ignore[dict-item]
