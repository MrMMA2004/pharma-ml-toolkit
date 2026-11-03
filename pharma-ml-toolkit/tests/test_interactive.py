"""Tests for :mod:`pharmaml.viz.interactive`'s round-186 express layer
(plotly/plotly.py absorbed): the ``hovertext`` trace field, the named
templates, ``express_figure``'s four kinds with colour splits and hover
columns, the SVG rendering's native ``<title>`` tooltips and the
self-contained HTML export.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.viz.interactive import (
    EXPRESS_KINDS,
    TEMPLATES,
    Figure,
    _figure_to_svg,
    apply_template,
    express_bar,
    express_figure,
    express_histogram,
    express_line,
    express_scatter,
    save_interactive_html,
)


@pytest.fixture()
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dose": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "response": [0.2, 0.5, 0.9, 1.4, 2.1, 3.0],
            "arm": ["a", "a", "a", "b", "b", "b"],
            "pid": ["r1", "r2", "r3", "r4", "r5", "r6"],
        }
    )


class TestHovertext:
    def test_scatter_hovertext_rides_the_trace(self) -> None:
        figure = Figure()
        figure.add_scatter(
            [0.0, 1.0], [1.0, 0.0], name="t", mode="markers",
            hovertext=["lo", "hi"],
        )
        assert figure.traces[0]["hovertext"] == ["lo", "hi"]

    def test_bar_hovertext_rides_the_trace(self) -> None:
        figure = Figure()
        figure.add_bar(["a", "b"], [1.0, 2.0], name="t", hovertext=["x", "y"])
        assert figure.traces[0]["hovertext"] == ["x", "y"]

    def test_length_mismatch_refused(self) -> None:
        figure = Figure()
        with pytest.raises(DataValidationError):
            figure.add_scatter([0.0, 1.0], [1.0, 0.0], name="t", hovertext=["only-one"])
        with pytest.raises(DataValidationError):
            figure.add_bar(["a"], [1.0], name="t", hovertext=["x", "y"])

    def test_json_round_trip_keeps_hover(self) -> None:
        figure = Figure()
        figure.add_scatter(
            [0.0, 1.0], [1.0, 0.0], name="t", mode="markers", hovertext=["a", "b"]
        )
        rebuilt = Figure.from_json(figure.to_json())
        assert rebuilt.traces[0]["hovertext"] == ["a", "b"]


class TestTemplates:
    def test_registry_names(self) -> None:
        assert set(TEMPLATES) == {"plotly", "plotly_white", "ggplot2", "seaborn"}
        for spec in TEMPLATES.values():
            assert len(spec["colors"]) >= 5
            assert spec["paper"].startswith("#")
            assert spec["ink"].startswith("#")

    def test_apply_template_sets_layout(self) -> None:
        figure = Figure()
        figure.add_scatter([0.0], [1.0], name="t")
        assert apply_template(figure, "ggplot2") is figure
        assert figure.layout["template"] == "ggplot2"
        with pytest.raises(DataValidationError):
            apply_template(figure, "dark26")

    def test_svg_uses_the_template_palette_and_paper(self) -> None:
        plain = Figure()
        plain.add_scatter([0.0, 1.0], [0.0, 1.0], name="t", mode="markers")
        svg_plain = _figure_to_svg(plain)
        assert 'fill="#ffffff"' in svg_plain  # default paper

        themed = Figure()
        themed.add_scatter([0.0, 1.0], [0.0, 1.0], name="t", mode="markers")
        apply_template(themed, "ggplot2")
        svg_themed = _figure_to_svg(themed)
        assert "#f7f7f7" in svg_themed  # ggplot2 paper
        assert TEMPLATES["ggplot2"]["colors"][0] in svg_themed

    def test_svg_renders_native_title_tooltips(self) -> None:
        figure = Figure()
        figure.add_scatter(
            [0.0, 1.0], [0.0, 1.0], name="t", mode="markers",
            hovertext=["dose=1", "dose=2"],
        )
        svg = _figure_to_svg(figure)
        assert "<title>dose=1</title>" in svg
        assert "<title>dose=2</title>" in svg
        # bars get tooltips too
        bar = Figure()
        bar.add_bar(["a", "b"], [1.0, 2.0], name="t", hovertext=["x", "y"])
        assert "<title>x</title>" in _figure_to_svg(bar)


class TestExpressFigure:
    def test_scatter_one_trace_without_color(self, frame: pd.DataFrame) -> None:
        figure = express_scatter(frame, "dose", "response")
        assert len(figure.traces) == 1
        assert figure.traces[0]["type"] == "scatter"
        assert figure.traces[0]["mode"] == "markers"
        assert figure.traces[0]["x"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        assert figure.layout["x_title"] == "dose"
        assert figure.layout["y_title"] == "response"

    def test_color_splits_into_named_traces(self, frame: pd.DataFrame) -> None:
        figure = express_scatter(frame, "dose", "response", color="arm")
        assert len(figure.traces) == 2
        assert figure.traces[0]["name"] == "arm=a"
        assert figure.traces[0]["x"] == [1.0, 2.0, 3.0]
        assert figure.traces[1]["name"] == "arm=b"
        assert figure.traces[1]["y"] == [1.4, 2.1, 3.0]

    def test_hover_columns_ride_per_point(self, frame: pd.DataFrame) -> None:
        figure = express_scatter(frame, "dose", "response", hover=["pid", "arm"])
        assert figure.traces[0]["hovertext"][0] == "pid=r1 · arm=a"
        assert figure.traces[0]["hovertext"][3] == "pid=r4 · arm=b"

    def test_line_and_bar_kinds(self, frame: pd.DataFrame) -> None:
        line = express_line(frame, "dose", "response")
        assert line.traces[0]["mode"] == "lines"
        bar = express_bar(frame, "dose", "response")
        assert bar.traces[0]["type"] == "bar"

    def test_histogram_bins_server_side(self, frame: pd.DataFrame) -> None:
        figure = express_histogram(frame, "response")
        trace = figure.traces[0]
        assert trace["type"] == "bar"
        assert sum(trace["y"]) == 6  # every row lands in exactly one bin
        assert figure.layout["y_title"] == "count"
        assert figure.layout["x_title"] == "response"

    def test_histogram_degenerate_single_value(self) -> None:
        flat = pd.DataFrame({"v": [2.0, 2.0, 2.0]})
        figure = express_histogram(flat, "v")
        assert sum(figure.traces[0]["y"]) == 3

    def test_labels_rename_axes(self, frame: pd.DataFrame) -> None:
        figure = express_scatter(
            frame, "dose", "response", labels={"x": "log dose", "y": "Δresponse"}
        )
        assert figure.layout["x_title"] == "log dose"
        assert figure.layout["y_title"] == "Δresponse"

    def test_template_rides_the_layout(self, frame: pd.DataFrame) -> None:
        figure = express_scatter(frame, "dose", "response", template="seaborn")
        assert figure.layout["template"] == "seaborn"

    def test_guards(self, frame: pd.DataFrame) -> None:
        with pytest.raises(DataValidationError):
            express_figure(frame, "violin", "dose", "response")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "nope", "response")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "dose", "nope")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "dose", None)
        with pytest.raises(DataValidationError):
            express_figure(frame, "histogram", "dose", "response")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "dose", "response", template="dark26")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "dose", "response", color="nope")
        with pytest.raises(DataValidationError):
            express_figure(frame, "scatter", "dose", "response", hover=["nope"])
        with pytest.raises(DataValidationError):
            express_figure(pd.DataFrame(columns=["a"]), "scatter", "a", "a")

    def test_kinds_constant(self) -> None:
        assert EXPRESS_KINDS == ("scatter", "line", "bar", "histogram")


class TestHtmlExportWithHover:
    def test_export_carries_titles_and_json(
        self, frame: pd.DataFrame, tmp_path: object
    ) -> None:
        figure = express_scatter(frame, "dose", "response", hover=["pid"])
        target = tmp_path / "express.html"  # type: ignore[operator]
        written = save_interactive_html(figure, str(target))
        assert written.is_file()
        content = written.read_text(encoding="utf-8")
        assert "<title>pid=r1</title>" in content
        assert "application/json" in content
        assert "cdn" not in content.lower()
        assert "<script src" not in content  # no JavaScript at all

    def test_bytes_are_deterministic(
        self, frame: pd.DataFrame, tmp_path: object
    ) -> None:
        figure = express_scatter(frame, "dose", "response", hover=["pid"])
        first = save_interactive_html(figure, str(tmp_path / "a.html"))  # type: ignore[operator]
        second = save_interactive_html(figure, str(tmp_path / "b.html"))  # type: ignore[operator]
        assert first.read_bytes() == second.read_bytes()
