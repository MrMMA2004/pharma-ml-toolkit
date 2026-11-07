"""Headless checks for the playground's embedded-figure sizing policy."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(scope="module")
def playground() -> Any:
    """Compile only the two sizing methods; importing the GUI requires Tk."""
    path = Path(__file__).resolve().parents[1] / "playground.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    app = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlaygroundApp")
    names = {"_resize_host_figure", "_figure_host"}
    methods = [node for node in app.body if isinstance(node, ast.FunctionDef) and node.name in names]
    extracted = ast.ClassDef(
        name="PlaygroundApp",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[]))
    namespace: dict[str, Any] = {"tk": SimpleNamespace(TclError=RuntimeError)}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - extracted test subject
    return SimpleNamespace(PlaygroundApp=namespace["PlaygroundApp"])


class _Canvas:
    def __init__(self) -> None:
        self.widget = object()
        self.draws = 0

    def get_tk_widget(self) -> object:
        return self.widget

    def draw_idle(self) -> None:
        self.draws += 1


class _Host:
    def __init__(self, width: int, height: int, figure: Any, canvas: _Canvas) -> None:
        self.width = width
        self.height = height
        self._pg_figure = figure
        self._pg_canvas = canvas
        self._pg_design_size = (9.0, 6.0)

    def winfo_width(self) -> int:
        return self.width

    def winfo_height(self) -> int:
        return self.height

    def winfo_children(self) -> list[Any]:
        return []


def _resize(playground: Any, width: int, height: int) -> tuple[Any, _Canvas]:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9.0, 6.0), dpi=100)
    canvas = _Canvas()
    host = _Host(width, height, figure, canvas)
    playground.PlaygroundApp._resize_host_figure(object(), host)
    return figure, canvas


def test_tkagg_figure_shrinks_without_distortion(playground: Any) -> None:
    figure, canvas = _resize(playground, 462, 312)
    assert tuple(figure.get_size_inches()) == pytest.approx((4.5, 3.0))
    assert canvas.draws == 1


def test_tkagg_figure_upscale_is_capped(playground: Any) -> None:
    figure, _canvas = _resize(playground, 5000, 5000)
    assert tuple(figure.get_size_inches()) == pytest.approx((12.15, 8.1))


def test_default_figure_host_is_taller(playground: Any) -> None:
    signature = inspect.signature(playground.PlaygroundApp._figure_host)
    assert signature.parameters["height"].default == 390
