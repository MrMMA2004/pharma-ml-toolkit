"""Tests for the Part-3 §17 image-export engine (viz/export.py).

Every scientific-integrity rule the directive demands is pinned here:

* TIFF as the scientific path -- 16-bit preserved (``I;16``), lossless
  compressions, multi-page stacks, dpi tags;
* JPEG as the presentation path -- quality knob, 16-bit REFUSAL
  without the explicit downcast, alpha composited onto a background;
* validation as part of the export (§17.4) -- reopen + magic-byte
  sniff + dimension check, extension/content mismatch caught;
* the PNG/SVG paths kept functional (§17.1's "retain existing").

The numbers are the engine's own on fixed arrays (no golden files --
the assertions describe BEHAVIOUR: modes, formats, page counts,
refusals), which is what the acceptance criteria actually ask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.viz.export import (
    EXPORT_EXTENSIONS,
    EXPORT_FILTERS,
    TIFF_COMPRESSIONS,
    ExportFormat,
    ExportManager,
    ImageExportOptions,
    export_figure,
    export_image,
    export_matrix,
    sniff_format,
    validate_export,
)

# -- format plumbing ------------------------------------------------------------


def test_export_filters_cover_the_four_formats() -> None:
    """§17.1: dialogs carry TIFF .tif/.tiff and JPEG .jpg/.jpeg filters."""
    assert "*.tif *.tiff" in EXPORT_FILTERS[ExportFormat.TIFF]
    assert "*.jpg *.jpeg" in EXPORT_FILTERS[ExportFormat.JPEG]
    assert "*.png" in EXPORT_FILTERS[ExportFormat.PNG]
    assert "*.svg" in EXPORT_FILTERS[ExportFormat.SVG]


def test_extension_aliases_resolve_to_one_family() -> None:
    from pharmaml.viz.export import _format_for_extension

    for suffix in (".tif", ".TIFF"):
        assert _format_for_extension(suffix) is ExportFormat.TIFF
    for suffix in (".jpg", ".jpeg"):
        assert _format_for_extension(suffix) is ExportFormat.JPEG


def test_unknown_extension_refuses() -> None:
    from pharmaml.viz.export import _format_for_extension

    with pytest.raises(DataValidationError, match="unsupported image extension"):
        _format_for_extension(".bmp")


def test_option_validation_ranges() -> None:
    with pytest.raises(DataValidationError, match="jpeg_quality"):
        ImageExportOptions(jpeg_quality=96)
    with pytest.raises(DataValidationError, match="compression"):
        ImageExportOptions(compression="jpeg2000")
    assert TIFF_COMPRESSIONS == ("none", "tiff_lzw", "tiff_deflate")


# -- the core exports -----------------------------------------------------------


def test_uint8_png_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "tile.png"
    result = export_image(np.arange(16, dtype=np.uint8).reshape(4, 4), target)
    assert result.format == "PNG"
    assert result.mode == "L"
    assert (result.width, result.height) == (4, 4)
    assert result.validated
    assert sniff_format(target) == "PNG"


def test_uint16_tiff_preserves_16_bits(tmp_path: Path) -> None:
    """§17.2: high-bit-depth scientific data never silently becomes 8-bit."""
    target = tmp_path / "matrix.tiff"
    values = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    result = export_image(values, target, compression="tiff_lzw")
    assert result.format == "TIFF"
    assert result.mode == "I;16"
    assert result.compression == "tiff_lzw"

    from PIL import Image

    with Image.open(target) as reopened:
        assert reopened.mode == "I;16"
        np.testing.assert_array_equal(np.asarray(reopened), values)


@pytest.mark.parametrize("compression", TIFF_COMPRESSIONS)
def test_tiff_compression_knob(tmp_path: Path, compression: str) -> None:
    """§17.2: uncompressed and both lossless codecs are reachable."""
    result = export_image(
        np.arange(256, dtype=np.uint8).reshape(16, 16),
        tmp_path / f"m_{compression}.tiff",
        compression=compression,
    )
    assert result.validated
    assert result.compression == compression


def test_jpeg_quality_setting(tmp_path: Path) -> None:
    """§17.3: configurable quality -- higher quality, no smaller file."""
    values = np.repeat(np.arange(256, dtype=np.uint8), 8).reshape(64, 32)
    low = export_image(values, tmp_path / "q30.jpg", jpeg_quality=30)
    high = export_image(values, tmp_path / "q92.jpg", jpeg_quality=92)
    assert low.validated and high.validated
    assert high.size_bytes >= low.size_bytes


def test_uint16_jpeg_refuses_then_explicit_downcast(tmp_path: Path) -> None:
    """§17.2/§17.3: lossy 8-bit JPEG never silently eats 16-bit data."""
    values = np.arange(64, dtype=np.uint16).reshape(8, 8)
    with pytest.raises(DataValidationError, match="16-bit"):
        export_image(values, tmp_path / "refused.jpg")
    result = export_image(values, tmp_path / "explicit.jpg", downcast_16bit=True)
    assert any("downcast" in w for w in result.warnings)
    assert result.mode == "L"


def test_float_matrix_refuses_silent_normalisation(tmp_path: Path) -> None:
    values = np.linspace(0.0, 1000.0, 64).reshape(8, 8)
    with pytest.raises(DataValidationError, match="normalize_float"):
        export_image(values, tmp_path / "refused.png")
    result = export_image(values, tmp_path / "scaled.png", normalize_float=True)
    assert result.validated
    assert any("normalis" in w for w in result.warnings)


def test_rgba_jpeg_composites_onto_background(tmp_path: Path) -> None:
    """§17.3: no silent alpha drop -- an explicit background is used."""
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # red with half alpha
    rgba[..., 3] = 128
    result = export_image(rgba, tmp_path / "flat.jpg")
    assert result.mode == "RGB"
    assert any("alpha" in w for w in result.warnings)
    # background (255,255,255) + half-alpha red -> a light red, NOT the
    # pure red a dropped alpha would print.
    from PIL import Image

    with Image.open(tmp_path / "flat.jpg") as reopened:
        pixel = reopened.getpixel((0, 0))
    assert pixel[0] > pixel[1] >= 0
    assert 200 < pixel[0] < 256


def test_multipage_tiff_stack(tmp_path: Path) -> None:
    """§17.2: image stacks export as multi-page TIFF with page checks."""
    pages = [np.full((4, 4), value, dtype=np.uint16) for value in (100, 200, 300)]
    result = export_matrix(pages[0], tmp_path / "stack.tiff", pages=pages[1:])
    assert result.mode == "I;16"
    report = validate_export(
        tmp_path / "stack.tiff", expected_format="TIFF", expected_pages=3
    )
    assert report["ok"] and report["pages"] == 3


def test_multipage_refuses_non_tiff(tmp_path: Path) -> None:
    pages = [np.full((4, 4), 1, dtype=np.uint8)] * 2
    with pytest.raises(DataValidationError, match="TIFF"):
        export_matrix(pages[0], tmp_path / "stack.png", pages=pages[1:])


def test_reencode_existing_file_keeps_dimensions(tmp_path: Path) -> None:
    source = tmp_path / "src.png"
    export_image(np.arange(16, dtype=np.uint8).reshape(4, 4), source)
    result = export_image(source, tmp_path / "converted.tiff")
    assert result.format == "TIFF"
    assert (result.width, result.height) == (4, 4)
    assert result.validated


def test_same_file_source_and_destination_refuses(tmp_path: Path) -> None:
    source = tmp_path / "src.png"
    export_image(np.zeros((3, 3), dtype=np.uint8), source)
    with pytest.raises(DataValidationError, match="same file"):
        export_image(source, source)


def test_svg_raster_source_refuses_with_guidance(tmp_path: Path) -> None:
    """SVG stays vector-only: rasters are pointed at PNG/TIFF honestly."""
    with pytest.raises(DataValidationError, match="vector"):
        export_image(np.zeros((3, 3), dtype=np.uint8), tmp_path / "raster.svg")


def test_bad_shape_refuses(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="shape"):
        export_image(np.zeros((3, 3, 5), dtype=np.uint8), tmp_path / "bad.png")


# -- figure exports (matplotlib) -------------------------------------------------


def test_figure_tiff_and_svg(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3, 2))
    figure.subplots().plot([0, 1, 2], [0, 1, 4])
    raster = export_figure(
        figure, tmp_path / "fig.tiff", dpi=120, compression="tiff_deflate"
    )
    assert raster.format == "TIFF"
    assert abs(raster.width - 360) <= 2  # 3 in * 120 dpi
    assert raster.validated

    vector = export_figure(figure, tmp_path / "fig.svg", dpi=120)
    assert vector.format == "SVG"
    assert vector.mode == "vector"
    assert vector.validated
    content = (tmp_path / "fig.svg").read_text(encoding="utf-8")
    assert "<svg" in content


# -- validation (§17.4) -----------------------------------------------------------


def test_validation_catches_extension_content_mismatch(tmp_path: Path) -> None:
    """PNG bytes under a .tiff name: the sniff says so, honestly."""
    source = tmp_path / "truth.png"
    export_image(np.zeros((3, 3), dtype=np.uint8), source)
    liar = tmp_path / "liar.tiff"
    liar.write_bytes(source.read_bytes())  # PNG payload, TIFF name
    report = validate_export(liar, expected_format="TIFF")
    assert not report["ok"]
    assert "sniffed PNG" in report["reason"]


def test_validation_catches_wrong_dimensions(tmp_path: Path) -> None:
    target = tmp_path / "tile.png"
    export_image(np.zeros((4, 4), dtype=np.uint8), target)
    report = validate_export(target, expected_format="PNG", expected_size=(9, 9))
    assert not report["ok"]
    assert "expected size" in report["reason"]


def test_validation_reports_missing_file(tmp_path: Path) -> None:
    report = validate_export(tmp_path / "absent.png")
    assert not report["ok"]
    assert report["reason"] == "file does not exist"


def test_validation_never_raises_on_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.tiff"
    corrupt.write_bytes(b"II*\x00" + b"\x00" * 32)  # TIFF magic, dead body
    report = validate_export(corrupt)
    assert isinstance(report, dict) and "ok" in report


def test_sniff_unknown_for_garbage(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.bin"
    garbage.write_bytes(b"\x00" * 64)
    assert sniff_format(garbage) == "unknown"


# -- the export manager (§17.4) ----------------------------------------------------


def test_manager_batch_with_manifest(tmp_path: Path) -> None:
    manager = ExportManager(tmp_path / "bundle", "{name}.tiff")
    manager.export(np.ones((3, 3), dtype=np.uint8), "eye")
    manager.export(np.zeros((3, 3), dtype=np.uint8), "dark")
    manifest = tmp_path / "bundle" / "manifest.json"
    rows = json.loads(manifest.read_text(encoding="utf-8"))["exports"]
    assert sorted(row["name"] for row in rows) == ["dark", "eye"]
    for row in rows:
        assert row["validated"] is True
        assert row["format"] == "TIFF"


def test_manager_format_override_rewrites_extension(tmp_path: Path) -> None:
    manager = ExportManager(tmp_path / "mgr", "{name}.png")
    result = manager.export(
        np.zeros((3, 3), dtype=np.uint8), "shot", fmt=ExportFormat.JPEG
    )
    assert result.format == "JPEG"
    assert result.path.name == "shot.jpg"


# -- the dependency guard -----------------------------------------------------------


def test_require_pillow_error_shape(monkeypatch) -> None:
    """The Pillow-free environment gets the actionable install hint."""
    import builtins

    real_import = builtins.__import__

    def _no_pillow(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no PIL here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pillow)
    from pharmaml.viz import export as export_module

    with pytest.raises(DependencyError, match="pharmaml\\[reports\\]"):
        export_module.require_pillow()


def test_export_extensions_table_complete() -> None:
    for fmt in ExportFormat:
        assert EXPORT_EXTENSIONS[fmt], fmt
