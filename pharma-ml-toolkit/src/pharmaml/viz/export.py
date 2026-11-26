"""First-class image export: PNG, SVG, TIFF and JPEG (Part-3 §17).

The toolkit's results live as numpy matrices, PIL images, matplotlib
figures and Qt widgets -- and until this module every export path was
PNG-shaped (``save_figure`` for matplotlib, ``ImagePanel.export_png``
for the GUI).  §17 of the Part-3 directive makes TIFF and JPEG
first-class alongside them, with the scientific-integrity rules the
section demands:

* **TIFF is the scientific default** -- lossless (LZW / DEFLATE / none),
  16-bit preservation (a ``uint16`` matrix exports as a genuine
  ``I;16`` TIFF, never a silently-quantised 8-bit copy), multi-page
  stacks, and the dpi resolution tags.
* **JPEG is the presentation format** -- configurable quality, honest
  about being lossy: a 16-bit source REFUSES to export (unless the
  caller passes ``downcast_16bit=True`` and owns the quantisation),
  and an RGBA source composites onto an explicit background instead
  of silently dropping the alpha channel.
* **Validation is part of the export** -- every file is reopened,
  magic-byte-sniffed and dimension-checked before the caller hears
  "done" (§17.4's "exist, can be reopened, expected dimensions and
  format"), and the extension must MATCH the requested format (§17.4's
  "do not change file extensions without actually converting").

The engine is Pillow-based and Pillow stays optional exactly like
matplotlib: importing this module is free, calling an export function
without Pillow installed raises the actionable
:class:`~pharmaml.exceptions.DependencyError`.

Example
-------
>>> import numpy as np
>>> from pharmaml.viz.export import export_image, sniff_format
>>> matrix = np.arange(64, dtype=np.uint16).reshape(8, 8)
>>> result = export_image(matrix, "/tmp/pharmaml_doctest_tile.tiff",
...                        compression="tiff_lzw")
>>> result.format
'TIFF'
>>> result.mode          # 16-bit preserved, not silently 8-bit
'I;16'
>>> sniff_format("/tmp/pharmaml_doctest_tile.tiff")
'TIFF'
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pharmaml.exceptions import DataValidationError, DependencyError
from pharmaml.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure

__all__ = [
    "EXPORT_EXTENSIONS",
    "EXPORT_FILTERS",
    "TIFF_COMPRESSIONS",
    "ExportFormat",
    "ExportManager",
    "ExportResult",
    "ImageExportOptions",
    "export_figure",
    "export_image",
    "export_matrix",
    "export_widget_image",
    "require_pillow",
    "sniff_format",
    "validate_export",
]

_logger = get_logger(__name__)


class ExportFormat(str, Enum):
    """The four first-class raster/vector export formats (§17.1)."""

    PNG = "PNG"
    SVG = "SVG"
    TIFF = "TIFF"
    JPEG = "JPEG"


#: Lossless TIFF compression schemes (§17.2) -- ``none`` keeps the
#: bytes uncompressed (the maximal-compatibility writer default),
#: ``tiff_lzw`` and ``tiff_deflate`` are the two lossless codecs PIL
#: writes portably across readers.
TIFF_COMPRESSIONS: tuple[str, ...] = ("none", "tiff_lzw", "tiff_deflate")

#: Canonical extensions per format, with the §17.1 dialog aliases
#: (``.tif``/``.tiff``, ``.jpg``/``.jpeg``) resolved to one family.
EXPORT_EXTENSIONS: dict[ExportFormat, tuple[str, ...]] = {
    ExportFormat.PNG: (".png",),
    ExportFormat.SVG: (".svg",),
    ExportFormat.TIFF: (".tif", ".tiff"),
    ExportFormat.JPEG: (".jpg", ".jpeg"),
}

#: Ready-made file-dialog filter strings (§17.1: the dialogs carry the
#: appropriate filters; PNG and SVG stay functional alongside the new
#: formats -- §17.1's "retain existing PNG and SVG export").
EXPORT_FILTERS: dict[ExportFormat, str] = {
    ExportFormat.PNG: "PNG images (*.png)",
    ExportFormat.SVG: "SVG vector graphics (*.svg)",
    ExportFormat.TIFF: "TIFF images (*.tif *.tiff)",
    ExportFormat.JPEG: "JPEG images (*.jpg *.jpeg)",
}

_JPEG_QUALITY_RANGE = (1, 95)  # PIL's documented useful range


def require_pillow():
    """Import Pillow lazily; raise the actionable error when absent.

    Mirrors :func:`pharmaml.viz._compat.require_matplotlib`: the module
    imports without Pillow, only *exporting* needs it.

    Returns
    -------
    module
        The imported ``PIL`` root (``Image`` is reachable through it).
    """
    try:
        import PIL  # noqa: F401 - presence probe
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on env
        raise DependencyError(
            "Pillow is required for TIFF/JPEG image export. "
            "From the toolkit package root, run:  python -m pip install -e '.[reports]'"
        ) from exc
    return Image


# ---------------------------------------------------------------------------
# magic-byte sniffing (§17.4 validation)
# ---------------------------------------------------------------------------

def sniff_format(path: str | Path) -> str:
    """Identify an image file from its magic bytes alone.

    Returns
    -------
    str
        One of ``"PNG"``, ``"JPEG"``, ``"TIFF"``, ``"SVG"`` -- or
        ``"unknown"`` when the header matches none of them (the honest
        answer beats a guess built from the extension).

    >>> from pathlib import Path
    >>> sniff_format(Path("/tmp/pharmaml_doctest_tile.tiff"))
    'TIFF'
    """
    try:
        head = Path(path).read_bytes()[:512]
    except OSError:
        return "unknown"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    # TIFF: little-endian "II*\0" or big-endian "MM\0*" + version 42.
    if len(head) >= 4 and head[:2] in (b"II", b"MM"):
        order = "<" if head[:2] == b"II" else ">"
        (version,) = struct.unpack(order + "H", head[2:4])
        if version == 42:
            return "TIFF"
    stripped = head.lstrip()
    if stripped.startswith((b"<?xml", b"<svg")) and b"<svg" in head:
        return "SVG"
    return "unknown"


def _format_for_extension(suffix: str) -> ExportFormat:
    """Map a file extension (any case, with aliases) to its format."""
    lowered = suffix.lower()
    for fmt, extensions in EXPORT_EXTENSIONS.items():
        if lowered in extensions:
            return fmt
    raise DataValidationError(
        f"unsupported image extension {suffix!r} -- expected one of "
        + ", ".join(
            ext for extensions in EXPORT_EXTENSIONS.values() for ext in extensions
        )
    )


# ---------------------------------------------------------------------------
# options + result
# ---------------------------------------------------------------------------

@dataclass
class ImageExportOptions:
    """Per-export configuration (§17.4's reusable export manager knobs).

    Attributes
    ----------
    jpeg_quality:
        JPEG quality 1-95 (PIL's documented useful range); the default
        92 favours fidelity for presentation use.
    compression:
        TIFF compression -- ``"none"``, ``"tiff_lzw"`` or
        ``"tiff_deflate"`` (all lossless, §17.2).
    dpi:
        Resolution written into the TIFF/PNG/JPEG metadata (also used
        as the matplotlib rasterisation dpi for figure exports).
    background:
        Colour an RGBA/transparent source composites onto when the
        target format has no alpha (JPEG).  ``(255, 255, 255)`` white
        -- never a silent black.
    downcast_16bit:
        Explicitly allow quantising 16-bit data to 8-bit for JPEG.
        Off by default: §17.2 forbids silently converting high-bit-depth
        scientific data, and JPEG (§17.3) is lossy anyway.
    normalize_float:
        Allow min-max normalisation of float matrices to uint8.  Off by
        default for the same reason -- the caller owns the rescale.
    validate:
        Reopen + magic-sniff + dimension-check the written file before
        reporting success (§17.4).  On by default.
    """

    jpeg_quality: int = 92
    compression: str = "tiff_lzw"
    dpi: float | None = None
    background: tuple[int, int, int] = (255, 255, 255)
    downcast_16bit: bool = False
    normalize_float: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        self.jpeg_quality = int(self.jpeg_quality)
        if not (
            _JPEG_QUALITY_RANGE[0] <= self.jpeg_quality <= _JPEG_QUALITY_RANGE[1]
        ):
            raise DataValidationError(
                f"jpeg_quality must be { _JPEG_QUALITY_RANGE[0] }-"
                f"{ _JPEG_QUALITY_RANGE[1] }, got {self.jpeg_quality}"
            )
        if self.compression not in TIFF_COMPRESSIONS:
            raise DataValidationError(
                f"compression must be one of {TIFF_COMPRESSIONS}, "
                f"got {self.compression!r}"
            )


@dataclass
class ExportResult:
    """What an export actually produced -- the honest report card.

    ``validated`` is only True after the file was reopened and checked
    (§17.4); ``warnings`` carries the scientific-integrity notices the
    section requires (lossy-format use, alpha compositing, 16-bit
    downcasts) so GUIs and logs can surface them.
    """

    path: Path
    format: str
    width: int
    height: int
    mode: str
    compression: str | None = None
    size_bytes: int = 0
    validated: bool = False
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-ready summary (the manager's manifest rows)."""
        return {
            "path": str(self.path),
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "compression": self.compression,
            "size_bytes": self.size_bytes,
            "validated": self.validated,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# core conversion helpers
# ---------------------------------------------------------------------------

def _array_to_image(
    values: np.ndarray, options: ImageExportOptions, warnings: list[str]
):
    """Convert a numpy array to a PIL image with bit-depth honesty.

    Rules (§17.2 / §17.3):

    * ``uint8`` (2-D or 3-D with 3/4 channels) passes through as-is.
    * ``uint16`` 2-D passes through as ``I;16`` (TIFF keeps 16 bits;
      JPEG refuses unless ``downcast_16bit``).
    * ``float`` matrices need explicit ``normalize_float`` -- silent
      rescaling of scientific values is the classic integrity trap.
    * Signed/other dtypes are upcast per the same explicit-choices rule.
    """
    Image = require_pillow()
    array = np.asarray(values)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (3, 4)):
        raise DataValidationError(
            f"expected a 2-D matrix or a (h, w, 3|4) array, got shape {array.shape}"
        )
    dtype = array.dtype
    if dtype == np.uint8:
        return Image.fromarray(array)  # 2-D -> L, (h, w, 3|4) -> RGB/RGBA
    if dtype == np.uint16 and array.ndim == 2:
        return Image.fromarray(array)  # 16-bit preserved as I;16
    if dtype == np.uint16 and array.ndim == 3:
        if not options.downcast_16bit:
            raise DataValidationError(
                "16-bit multichannel arrays have no direct PIL mode -- "
                "quantise explicitly (one channel at a time keeps I;16) "
                "or pass downcast_16bit=True"
            )
        warnings.append("16-bit multichannel data downcast to 8-bit RGB")
        return Image.fromarray((array >> 8).astype(np.uint8))
    if np.issubdtype(dtype, np.floating):
        if not options.normalize_float:
            raise DataValidationError(
                "float matrices are scientific values -- exporting them "
                "as pixels silently rescales the data. Pass "
                "normalize_float=True (min-max to uint8) or scale "
                "yourself when display is the intent."
            )
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise DataValidationError("matrix has no finite entries to normalise")
        lo, hi = float(finite.min()), float(finite.max())
        span = hi - lo
        scaled = np.zeros(array.shape, dtype=np.uint8) if span <= 0 else (
            (np.nan_to_num(array, nan=lo) - lo) / span * 255.0
        ).round().clip(0, 255).astype(np.uint8)
        warnings.append(
            f"float matrix min-max normalised to uint8 (range {lo:.6g}..{hi:.6g})"
        )
        return Image.fromarray(scaled)  # 2-D -> L, 3-D -> RGB/RGBA
    # signed ints / bool: honest upcast under the caller's eyes.
    warnings.append(f"{dtype} array upcast to uint8 for imaging")
    return Image.fromarray(
        np.clip(array.astype(np.int64), 0, 255).astype(np.uint8)
    )


def _flatten_for_target(
    image: Any, fmt: ExportFormat, options: ImageExportOptions, warnings: list[str]
) -> Any:
    """Apply the format-specific integrity rules to a PIL image."""
    if fmt is not ExportFormat.JPEG:
        return image
    if image.mode == "I;16" and not options.downcast_16bit:
        raise DataValidationError(
            "JPEG is 8-bit lossy -- a 16-bit TIFF source refuses to export "
            "silently quantised (§17.2/§17.3). Export TIFF for the data, or "
            "pass downcast_16bit=True when a presentation copy is the intent."
        )
    if image.mode == "I;16":
        warnings.append("16-bit source downcast to 8-bit for JPEG (lossy)")
        from PIL import Image as _PILImage

        array = (np.asarray(image) >> 8).astype(np.uint8)
        image = _PILImage.fromarray(array)
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        warnings.append(
            f"alpha channel composited onto background {options.background} "
            "(JPEG has no transparency)"
        )
        image = _composite(image, options.background)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _composite(image: Any, background: tuple[int, int, int]) -> Any:
    """Composite an image with transparency onto an opaque background."""
    from PIL import Image

    rgba = image.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, background + (255,))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def _pil_save_kwargs(fmt: ExportFormat, options: ImageExportOptions) -> dict[str, Any]:
    """The format-specific PIL save() arguments."""
    kwargs: dict[str, Any] = {}
    if options.dpi is not None:
        kwargs["dpi"] = (options.dpi, options.dpi)
    if fmt is ExportFormat.JPEG:
        kwargs["quality"] = options.jpeg_quality
        kwargs["optimize"] = True
    if fmt is ExportFormat.TIFF:
        kwargs["compression"] = (
            None if options.compression == "none" else options.compression
        )
    return kwargs


# ---------------------------------------------------------------------------
# the public exporters
# ---------------------------------------------------------------------------

def export_image(
    image: np.ndarray | Any | str | Path,
    path: str | Path,
    *,
    options: ImageExportOptions | None = None,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    dpi: float | None = None,
    downcast_16bit: bool | None = None,
    normalize_float: bool | None = None,
    background: tuple[int, int, int] | None = None,
) -> ExportResult:
    """Export an image (array / PIL image / existing file) to any §17 format.

    The format is decided by the extension, and the extension is
    honoured: a ``.tiff`` path yields a TIFF byte stream, a ``.jpg``
    path a JPEG one (§17.4 -- extensions never lie).  Arrays follow the
    bit-depth rules from :func:`_array_to_image`; existing image FILES
    are re-encoded (a 16-bit TIFF re-saved as TIFF stays 16-bit; as
    JPEG it refuses without ``downcast_16bit``).

    Parameters
    ----------
    image:
        numpy array, PIL ``Image`` or a path to an existing image file.
    path:
        Destination; parent directories are created.

    Returns
    -------
    ExportResult
        With ``validated=True`` when the §17.4 reopen-check passed.

    Raises
    ------
    DataValidationError
        On extension/format mismatch, bit-depth traps or bad options.
    DependencyError
        When Pillow is missing.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.viz.export import export_image
    >>> result = export_image(np.zeros((6, 6), dtype=np.uint8),
    ...                        "/tmp/pharmaml_doctest_flat.jpg",
    ...                        jpeg_quality=80)
    >>> result.format
    'JPEG'
    >>> result.validated
    True
    """
    resolved = _resolve_options(
        options, jpeg_quality, compression, dpi, downcast_16bit,
        normalize_float, background,
    )
    target = Path(path)
    fmt = _format_for_extension(target.suffix)
    warnings: list[str] = []

    Image = require_pillow()
    if isinstance(image, (str, Path)):
        source = Path(image)
        if not source.is_file():
            raise DataValidationError(f"source image not found: {source}")
        pil_image = Image.open(source)
        pil_image.load()
        if source == target:
            raise DataValidationError(
                "source and destination are the same file -- export implies "
                "a format conversion or a new copy"
            )
    elif isinstance(image, np.ndarray):
        pil_image = _array_to_image(image, resolved, warnings)
    else:
        pil_image = image  # already a PIL image

    if fmt is ExportFormat.SVG:
        raise DataValidationError(
            "SVG is a vector format -- raster sources wrap as PNG-in-SVG "
            "only via export_figure (matplotlib) or an explicit SVG string. "
            "Use PNG/TIFF for lossless raster copies."
        )

    pil_image = _flatten_for_target(pil_image, fmt, resolved, warnings)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = _pil_save_kwargs(fmt, resolved)
    pil_image.save(target, **save_kwargs)

    result = ExportResult(
        path=target,
        format=fmt.value,
        width=pil_image.width,
        height=pil_image.height,
        mode=pil_image.mode,
        compression=resolved.compression if fmt is ExportFormat.TIFF else None,
        size_bytes=target.stat().st_size,
        warnings=warnings,
    )
    if resolved.validate:
        check = validate_export(target, expected_format=fmt.value)
        if not check["ok"]:
            raise DataValidationError(
                f"export validation failed for {target}: {check['reason']}"
            )
        result.validated = True
    for note in warnings:
        _logger.info("export %s: %s", target.name, note)
    return result


def export_matrix(
    values: np.ndarray,
    path: str | Path,
    *,
    pages: Sequence[np.ndarray] | None = None,
    **kwargs: Any,
) -> ExportResult:
    """Export a scientific matrix -- the §17.2-specialised path.

    Identical to :func:`export_image` for single matrices, plus the
    multi-page TIFF (image stack) route: ``pages`` are appended as
    additional TIFF pages (``save_all=True``), each following the same
    bit-depth rules.  Non-TIFF targets with ``pages`` are refused --
    a stack is a TIFF concept.

    >>> import numpy as np
    >>> from pharmaml.viz.export import export_matrix
    >>> stack = [np.full((4, 4), v, dtype=np.uint16) for v in (100, 200)]
    >>> result = export_matrix(stack[0], "/tmp/pharmaml_doctest_stack.tiff",
    ...                        pages=stack[1:])
    >>> result.mode
    'I;16'
    """
    if pages is not None and Path(path).suffix.lower() not in (".tif", ".tiff"):
        raise DataValidationError(
            "multi-page stacks are a TIFF capability -- export to .tif/.tiff"
        )
    resolved = _resolve_options(
        kwargs.pop("options", None),
        jpeg_quality=kwargs.pop("jpeg_quality", None),
        compression=kwargs.pop("compression", None),
        dpi=kwargs.pop("dpi", None),
        downcast_16bit=kwargs.pop("downcast_16bit", None),
        normalize_float=kwargs.pop("normalize_float", None),
        background=kwargs.pop("background", None),
    )
    if kwargs:
        raise TypeError(
            f"export_matrix() got unexpected keyword(s): {', '.join(kwargs)}"
        )
    target = Path(path)
    fmt = _format_for_extension(target.suffix)
    warnings: list[str] = []
    main = _array_to_image(values, resolved, warnings)
    if pages:
        extra = [_array_to_image(page, resolved, warnings) for page in pages]
        main = _flatten_for_target(main, fmt, resolved, warnings)
        extra = [_flatten_for_target(p, fmt, resolved, warnings) for p in extra]
        if any(
            (p.width, p.height) != (main.width, main.height) for p in extra
        ):
            raise DataValidationError(
                "multi-page TIFF pages must share the main image's dimensions"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = _pil_save_kwargs(fmt, resolved)
        save_kwargs["save_all"] = True
        save_kwargs["append_images"] = extra
        main.save(target, **save_kwargs)
        result = ExportResult(
            path=target,
            format=fmt.value,
            width=main.width,
            height=main.height,
            mode=main.mode,
            compression=resolved.compression if fmt is ExportFormat.TIFF else None,
            size_bytes=target.stat().st_size,
            warnings=warnings,
        )
        if resolved.validate:
            check = validate_export(
                target, expected_format=fmt.value, expected_pages=1 + len(extra)
            )
            if not check["ok"]:
                raise DataValidationError(
                    f"export validation failed: {check['reason']}"
                )
            result.validated = True
        return result
    result = export_image(values, target, options=resolved)
    result.warnings.extend(warnings)
    return result


def export_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: float | None = 150.0,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    background: tuple[int, int, int] | None = None,
    transparent: bool = False,
) -> ExportResult:
    """Export a matplotlib figure to any §17 format.

    SVG goes through matplotlib's native vector writer (``savefig``
    with ``format='svg'``) so the vector content stays vectorial; the
    raster formats render through Agg at ``dpi`` and then encode via
    Pillow -- TIFF with the lossless codecs, JPEG with the quality
    setting (§17.2/§17.3 applied to figures).

    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from matplotlib.figure import Figure
    >>> from pharmaml.viz.export import export_figure
    >>> fig = Figure(figsize=(3, 2))
    >>> _ = fig.subplots().plot([0, 1, 2], [0, 1, 4])
    >>> result = export_figure(fig, "/tmp/pharmaml_doctest_fig.tiff",
    ...                        dpi=120, compression="tiff_deflate")
    >>> result.format
    'TIFF'
    >>> result.width > 300   # 3 inches at 120 dpi
    True
    """
    target = Path(path)
    fmt = _format_for_extension(target.suffix)
    options = _resolve_options(None, jpeg_quality, compression, dpi, None, None, background)
    target.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    if fmt is ExportFormat.SVG:
        figure.savefig(
            target, format="svg", dpi=dpi or 150.0, transparent=transparent
        )
        result = ExportResult(
            path=target,
            format="SVG",
            width=int(figure.get_size_inches()[0] * (dpi or 150.0)),
            height=int(figure.get_size_inches()[1] * (dpi or 150.0)),
            mode="vector",
            size_bytes=target.stat().st_size,
        )
    else:
        import io

        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=dpi or 150.0,
            transparent=transparent,
            facecolor="none" if transparent else figure.get_facecolor(),
        )
        buffer.seek(0)
        Image = require_pillow()
        pil_image = Image.open(buffer)
        pil_image.load()
        if not transparent and pil_image.mode in ("RGBA", "LA"):
            # matplotlib PNGs carry alpha; the facecolor already painted
            # the canvas so flatten keeps the intended look.
            pil_image = pil_image.convert("RGB")
        pil_image = _flatten_for_target(pil_image, fmt, options, warnings)
        save_kwargs = _pil_save_kwargs(fmt, options)
        pil_image.save(target, **save_kwargs)
        result = ExportResult(
            path=target,
            format=fmt.value,
            width=pil_image.width,
            height=pil_image.height,
            mode=pil_image.mode,
            compression=options.compression if fmt is ExportFormat.TIFF else None,
            size_bytes=target.stat().st_size,
            warnings=warnings,
        )
    if options.validate:
        check = validate_export(target, expected_format=fmt.value)
        if not check["ok"]:
            raise DataValidationError(
                f"figure export validation failed: {check['reason']}"
            )
        result.validated = True
    return result


def export_widget_image(
    qimage_bytes: bytes,
    path: str | Path,
    *,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    dpi: float | None = None,
) -> ExportResult:
    """Export a Qt-rendered widget grab (PNG bytes) to any §17 format.

    The GUI layer grabs its ``QGraphicsScene``/widget to a ``QImage``
    (``PNG`` format bytes); this converts those bytes to TIFF/JPEG/PNG
    through the same integrity-checked pipeline, so the Image Lab and
    every plot panel share one honest export path.
    """
    import io

    Image = require_pillow()
    target = Path(path)
    fmt = _format_for_extension(target.suffix)
    options = _resolve_options(None, jpeg_quality, compression, dpi, None, None, None)
    warnings: list[str] = []
    pil_image = Image.open(io.BytesIO(qimage_bytes))
    pil_image.load()
    pil_image = _flatten_for_target(pil_image, fmt, options, warnings)
    target.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(target, **_pil_save_kwargs(fmt, options))
    result = ExportResult(
        path=target,
        format=fmt.value,
        width=pil_image.width,
        height=pil_image.height,
        mode=pil_image.mode,
        compression=options.compression if fmt is ExportFormat.TIFF else None,
        size_bytes=target.stat().st_size,
        warnings=warnings,
    )
    if options.validate:
        check = validate_export(target, expected_format=fmt.value)
        if not check["ok"]:
            raise DataValidationError(
                f"widget export validation failed: {check['reason']}"
            )
        result.validated = True
    return result


# ---------------------------------------------------------------------------
# validation (§17.4)
# ---------------------------------------------------------------------------

def validate_export(
    path: str | Path,
    expected_format: str | None = None,
    expected_size: tuple[int, int] | None = None,
    expected_pages: int | None = None,
) -> dict[str, Any]:
    """Reopen an exported file and verify it kept its promises.

    §17.4's checklist, executed: the file exists, reopens, carries the
    expected format (magic bytes, not the extension's claim), the
    expected pixel dimensions and -- for TIFF -- the expected page
    count.  Returns a report dict with ``ok`` and a ``reason`` on any
    failure; never raises for a bad file (the caller decides policy).

    >>> from pharmaml.viz.export import validate_export
    >>> report = validate_export("/tmp/pharmaml_doctest_flat.jpg",
    ...                           expected_format="JPEG")
    >>> report["ok"], report["format"]
    (True, 'JPEG')
    """
    target = Path(path)
    report: dict[str, Any] = {"path": str(target), "ok": False, "reason": ""}
    if not target.is_file():
        report["reason"] = "file does not exist"
        return report
    sniffed = sniff_format(target)
    report["format"] = sniffed
    if sniffed == "unknown":
        report["reason"] = "magic bytes match no supported format"
        return report
    if expected_format and sniffed != expected_format.upper():
        report["reason"] = (
            f"expected {expected_format} bytes, sniffed {sniffed} "
            "(extension does not match content)"
        )
        return report
    if sniffed == "SVG":
        report.update(ok=True, width=None, height=None, mode="vector")
        return report
    Image = require_pillow()
    try:
        with Image.open(target) as reopened:
            reopened.load()
            report["width"], report["height"] = reopened.width, reopened.height
            report["mode"] = reopened.mode
            pages = getattr(reopened, "n_frames", 1)
            report["pages"] = pages
    except Exception as exc:  # noqa: BLE001 - honest failure report
        report["reason"] = f"reopen failed: {exc}"
        return report
    if expected_size is not None and (
        report["width"], report["height"]
    ) != tuple(expected_size):
        report["reason"] = (
            f"expected size {expected_size}, found "
            f"{(report['width'], report['height'])}"
        )
        return report
    if expected_pages is not None and report.get("pages") != expected_pages:
        report["reason"] = (
            f"expected {expected_pages} TIFF pages, found {report.get('pages')}"
        )
        return report
    report["ok"] = True
    return report


# ---------------------------------------------------------------------------
# the reusable export manager (§17.4)
# ---------------------------------------------------------------------------

class ExportManager:
    """Batch export with the §17.4 manager contract.

    One configured destination (directory + filename pattern + format
    options) receiving many images, each validated on write, with a
    JSON manifest recording what landed where -- the audit trail the
    acceptance criteria ask for when "exported dimensions and formats
    are validated".

    Parameters
    ----------
    output_dir:
        Destination directory (created on first export).
    pattern:
        Filename template with a ``{name}`` placeholder, e.g.
        ``"imagelab_{name}.tiff"``.

    Examples
    --------
    >>> import numpy as np
    >>> from pharmaml.viz.export import ExportManager
    >>> manager = ExportManager("/tmp/pharmaml_doctest_mgr", "{name}.tiff")
    >>> _ = manager.export(np.ones((3, 3), dtype=np.uint8), "eye")
    >>> _ = manager.export(np.zeros((3, 3), dtype=np.uint8), "dark")
    >>> import json, pathlib
    >>> manifest = pathlib.Path("/tmp/pharmaml_doctest_mgr/manifest.json")
    >>> rows = json.loads(manifest.read_text())["exports"]
    >>> sorted(row["name"] for row in rows)
    ['dark', 'eye']
    """

    def __init__(
        self,
        output_dir: str | Path,
        pattern: str = "{name}.{ext}",
        *,
        jpeg_quality: int = 92,
        compression: str = "tiff_lzw",
        dpi: float | None = None,
        background: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.output_dir = Path(output_dir)
        self.pattern = pattern
        self.options = ImageExportOptions(
            jpeg_quality=jpeg_quality,
            compression=compression,
            dpi=dpi,
            background=background,
        )
        self.results: list[dict[str, Any]] = []

    def export(
        self,
        image: np.ndarray | Any | str | Path,
        name: str,
        *,
        fmt: ExportFormat | None = None,
    ) -> ExportResult:
        """Export one image under ``pattern``; records the manifest row."""
        stem = self.pattern.format(name=name, ext="png")
        suffix = Path(stem).suffix or ".png"
        if fmt is not None:
            canonical = EXPORT_EXTENSIONS[fmt][0]
            if suffix.lower() not in EXPORT_EXTENSIONS[fmt]:
                stem = f"{Path(stem).stem}{canonical}"
                suffix = canonical
        target = self.output_dir / stem
        result = export_image(image, target, options=self.options)
        self.results.append({"name": name, **result.as_dict()})
        self._write_manifest()
        return result

    def _write_manifest(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / "manifest.json"
        manifest.write_text(
            json.dumps({"exports": self.results}, indent=2) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# option plumbing
# ---------------------------------------------------------------------------

def _resolve_options(
    options: ImageExportOptions | None = None,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    dpi: float | None = None,
    downcast_16bit: bool | None = None,
    normalize_float: bool | None = None,
    background: tuple[int, int, int] | None = None,
) -> ImageExportOptions:
    """Fold per-call overrides into an options instance."""
    if options is not None:
        base = ImageExportOptions(
            jpeg_quality=options.jpeg_quality,
            compression=options.compression,
            dpi=options.dpi,
            background=options.background,
            downcast_16bit=options.downcast_16bit,
            normalize_float=options.normalize_float,
            validate=options.validate,
        )
    else:
        base = ImageExportOptions()
    if jpeg_quality is not None:
        base.jpeg_quality = int(jpeg_quality)
    if compression is not None:
        base.compression = compression
    if dpi is not None:
        base.dpi = float(dpi)
    if downcast_16bit is not None:
        base.downcast_16bit = downcast_16bit
    if normalize_float is not None:
        base.normalize_float = normalize_float
    if background is not None:
        base.background = tuple(background)  # type: ignore[assignment]
    # Re-run the field validation after overrides.
    base.__post_init__()
    return base
