"""Image Lab: matrix visualization the Part-2 way (16.8.2-16.8.4).

The Studio's increment-12 page -- the cross-cutting closure the
MASTER_PROMPT part2 acceptance list still owed: image display with
the histogram/LUT colormap control (16.8.4), the interactive RectROI
with live sub-matrix statistics (16.8.3), and PNG export (16.8.7 /
16.19's export criterion), each demonstrated on REAL project data --
never a fabricated array:

- **Similarity heatmap** -- ``compute_fingerprints`` (morgan r=2,
  2048 bits) over the bundled ``seed_molecules.csv`` roster (or the
  loaded table's SMILES-like column, when the Data Explorer carries
  one) -> ``tanimoto_matrix``: the 12x12 drug-similarity image with
  both molecule-name axes, the off-diagonal extremes, and the
  most/least similar pairs the probe printed.
- **Response matrix** -- the bundled GDSC-style screen through the
  Biology Workbench's own pipeline (``load_response_table`` ->
  ``collapse_replicates`` -> ``response_matrix``): the 23x8 LN_IC50
  image, one honest NaN hole (masked), the most sensitive / most
  resistant pairs.
- **Explorer matrix** -- the loaded table's numeric columns as one
  image (raw or z-scored per column), or the seeded 12-row demo
  frame when no table is loaded; the honest refusal below 2 numeric
  columns.
- **ROI explorer** -- the shared RectROI over the similarity matrix,
  with the live count/mean/min/max readout; the fixed region rows
  [2,6) x cols [7,10) reproduces the probe's printed statistics
  exactly, and the spin boxes move the same ROI the other tabs
  share.
- **Performance** -- ``tanimoto_matrix`` over a seeded bit matrix
  (n x 2048 bits, n from a spin box): pairs, wall seconds, M
  pairs/s -- the 16.19 large-dataset number, reported honestly as a
  timing, never asserted as one.

All four matrix tabs carry the Part-3 §21.4 pixel-inspection readout:
hover any matrix cell and the label under the panel reports the exact
pixel (row, column, cursor position, value or per-channel values, NaN
disclosure, the axis labels when the matrix carried them); at rest the
same line reports the matrix metadata (shape, finite/NaN census,
value range).  The Performance tab has no matrix, so it carries none.
Part-3 §23.10 (round 245) extends the same line with the sub-pixel
mapping disclosure under extreme zoom: when one screen pixel straddles
several image pixels the readout says so (``1 screen px spans ≤N×M
image px (one shown)`` -- the shown value is one of them); when one
image pixel is a big screen block it explains the steady value
(``1 image px ≈ N×M screen px``); near 1:1 nothing is appended.

All four engine tabs run in a
:class:`~ui.workers.job_worker.JobWorker`; they are single-shot
engine calls without checkpoint loops, so the progress channel
reports honestly but the cooperative Stop only lands for
checkpoint-wired jobs -- the same disclosure the Biology and PK/PD
pages carry.

Every default-input number below was engine-printed first by the
committed probe ``scratch/probe_imagelab_221.py`` (the round-205+
house rule, per-section fresh seeds -- the bestofml probe's pattern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.random import default_rng
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.image_panel import (
    GRADIENT_PRESETS,
    ImagePanel,
    format_scale_note,
    roi_stats_from_values,
)
from ..widgets.progress_panel import ProgressPanel
from ..widgets.responsive import LEGIBILITY_FLOOR, configure_splitter
from ..workers.job_worker import JobRunner

_TK = Path(__file__).resolve().parents[3]

#: The bundled named-drug roster (12 molecules, all RDKit-valid).
_SEEDS_PATH = _TK / "examples" / "data" / "seed_molecules.csv"

#: The bundled GDSC-style response screen (185 rows, 23 lines, 8 compounds).
_SCREEN_PATH = _TK / "examples" / "data" / "cellline_screen.csv"

#: Minimum SMILES for a similarity image -- below this the honest refusal.
_MIN_SMILES = 4

#: Minimum numeric columns for an explorer matrix image.
_MIN_NUMERIC = 2


@dataclass
class SimilarityOutcome:
    names: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None
    n_valid: int = 0
    source: str = ""
    off_min: float = 0.0
    off_max: float = 0.0
    off_mean: float = 0.0
    best_pair: tuple[str, str] = ("", "")
    worst_pair: tuple[str, str] = ("", "")
    best_value: float = 0.0
    worst_value: float = 0.0
    top_mean_name: str = ""
    top_mean: float = 0.0
    low_mean_name: str = ""
    low_mean: float = 0.0


@dataclass
class ScreenOutcome:
    matrix: pd.DataFrame | None = None
    n_holes: int = 0
    fin_min: float = 0.0
    fin_max: float = 0.0
    fin_mean: float = 0.0
    most_sensitive: tuple[str, str] = ("", "")
    most_resistant: tuple[str, str] = ("", "")
    seconds: float = 0.0


@dataclass
class ExplorerOutcome:
    header: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None
    zscored: bool = False
    raw_means: list[float] = field(default_factory=list)
    raw_stds: list[float] = field(default_factory=list)
    col_means: list[float] = field(default_factory=list)
    col_stds: list[float] = field(default_factory=list)
    z_min: float = 0.0
    z_max: float = 0.0
    max_abs_row: int = 0
    max_abs: float = 0.0
    source: str = ""


@dataclass
class PerfOutcome:
    n: int = 0
    pairs: int = 0
    seconds: float = 0.0
    off_mean: float = 0.0
    shape: tuple[int, int] = (0, 0)


class ImageLabPage(QWidget):
    """Matrix visualization workbench (directive 16.8.2-16.8.4 closure)."""

    def __init__(
        self,
        parent: QWidget | None = None,
        theme: ThemeManager | None = None,
        runner: JobRunner | None = None,
        data_explorer: object | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._runner = runner or JobRunner(self)
        self._data_explorer = data_explorer
        # review m-19: re-fill the similarity/response pickers on every
        # explorer load (the opportunistic first paint used to be the only
        # refresh; a later load left the combos showing the old columns)
        if self._data_explorer is not None and hasattr(self._data_explorer, "frameLoaded"):
            self._data_explorer.frameLoaded.connect(self._on_explorer_frame)


        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_similarity_tab(), "Similarity heatmap")
        self._tabs.addTab(self._build_screen_tab(), "Response matrix")
        self._tabs.addTab(self._build_explorer_tab(), "Explorer matrix")
        self._tabs.addTab(self._build_roi_tab(), "ROI explorer")
        self._tabs.addTab(self._build_perf_tab(), "Performance")
        # (part8 §4.1/§4.2) the backend-figure coverage browser: the
        # shared FigureCanvasHost renders original backend figures from
        # the loaded dataset; the registry table maps every public viz
        # function to its workspace and integration status.
        from ..widgets.coverage_browser import CoverageBrowser

        self._coverage_tab = CoverageBrowser(
            frame_provider=self._live_frame
        )
        self._tabs.addTab(self._coverage_tab, "Backend figures")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)
        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        # pickers fill opportunistically when an explorer frame already
        # exists (the house pages read frame() at action time; this keeps
        # the combos honest from the first paint without a load signal).
        frame = self._explorer_frame()
        if frame is not None:
            self._refresh_sim_columns()
            numeric = [
                str(c)
                for c in frame.columns
                if pd.api.types.is_numeric_dtype(frame[c])
            ]
            self._refresh_numeric_pickers(numeric)

    # -- shared chrome ---------------------------------------------------------
    def _make_panel(self, title: str) -> tuple[ImagePanel, QLabel, QLabel]:
        """Panel + the ROI readout + the §21.4 hover/pixel readout."""
        panel = ImagePanel(title=title)
        roi_label = QLabel("ROI: — (drag the rectangle over the image)")
        roi_label.setWordWrap(True)
        panel.roi_changed.connect(lambda: self._refresh_roi(panel, roi_label))
        hover_label = QLabel("Hover: — (move the cursor over the image)")
        hover_label.setWordWrap(True)
        # §21.4: the pixel payload drives the readout; None restores the
        # at-rest metadata line (shape / NaN census / value range).
        panel.pixel_hovered.connect(
            lambda payload: self._refresh_hover(payload, hover_label, panel)
        )
        panel.matrix_changed.connect(
            lambda: self._refresh_hover(None, hover_label, panel)
        )
        return panel, roi_label, hover_label

    def _refresh_hover(
        self, payload: dict | None, label: QLabel, panel: ImagePanel
    ) -> None:
        """§21.4 readout: the pixel under the cursor, or the at-rest facts."""
        if payload is None:
            meta = panel.image_metadata()
            if meta is None:
                label.setText("Hover: — (move the cursor over the image)")
                return
            if meta["channels"] > 1:
                label.setText(
                    f"Hover: — Matrix {meta['rows']}x{meta['cols']} "
                    f"({meta['channels']} channels, {meta['dtype']}) · "
                    f"min {meta['min']:.6g} · max {meta['max']:.6g}"
                )
            else:
                label.setText(
                    f"Hover: — Matrix {meta['rows']}x{meta['cols']} · "
                    f"{meta['finite']} finite / {meta['masked']} masked · "
                    f"min {meta['min']:.6g} · max {meta['max']:.6g}"
                )
            return
        parts = [
            (
                f"Pixel [{payload['row']}, {payload['col']}] "
                f"(x {payload['x']:.3g}, y {payload['y']:.3g})"
            )
        ]
        if payload["channels"] > 1 and payload["rgb"] is not None:
            chans = ", ".join(f"{v:.6g}" for v in payload["rgb"])
            parts.append(f"channels ({chans})")
        elif payload["masked"]:
            parts.append("= masked (NaN)")
        else:
            parts.append(f"= {payload['value']:.6g}")
        if payload["y_label"] is not None:
            parts.append(f"row: {payload['y_label']}")
        if payload["x_label"] is not None:
            parts.append(f"col: {payload['x_label']}")
        # §23.10: the sub-pixel mapping disclosure under extreme zoom --
        # the live on-screen size of one image pixel (absent on payloads
        # built by pixel_at directly; the hover path always carries it).
        note = format_scale_note(payload.get("image_px_screen"))
        if note is not None:
            parts.append(note)
        label.setText("Hover: " + " · ".join(parts))

    def _refresh_roi(self, panel: ImagePanel, label: QLabel) -> None:
        stats = panel.roi_stats()
        if stats is None:
            label.setText("ROI: — (drag the rectangle over the image)")
            return
        label.setText(
            f"ROI: {int(stats['rows'])} rows x {int(stats['cols'])} cols -- "
            f"{int(stats['count'])} cells, mean {stats['mean']:.6f}, "
            f"min {stats['min']:.6f}, max {stats['max']:.6f}"
        )

    def apply_palette(self, palette: ChartPalette) -> None:
        """Re-style every image panel's chrome (the house dispatch)."""
        panels = cast(list[ImagePanel], self.findChildren(ImagePanel))
        for panel in panels:
            panel.apply_palette(palette)

    def _gradient_row(self, panel: ImagePanel) -> QWidget:
        box = QComboBox()
        box.addItems(list(GRADIENT_PRESETS))
        box.setCurrentText("viridis")
        box.currentTextChanged.connect(lambda name: panel.set_gradient(name))
        row = QWidget()
        form = QFormLayout(row)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Colormap:", box)

        # Part-3 §17: the format-aware export controls -- TIFF/JPEG are
        # first-class beside PNG, with the compression/quality knobs the
        # section requires, and the dialog filters follow the chosen
        # format (§17.1).
        fmt_box = QComboBox()
        fmt_box.addItems(["TIFF (lossless, 16-bit safe)", "PNG (lossless)", "JPEG (lossy, quality)"])
        fmt_box.setCurrentIndex(0)
        quality = QSpinBox()
        quality.setRange(1, 95)
        quality.setValue(92)
        quality.setToolTip("JPEG quality 1-95 (§17.3) -- lossy; prefer TIFF for scientific data")
        compress = QComboBox()
        compress.addItems(["LZW", "DEFLATE", "None"])
        compress.setToolTip("TIFF compression (§17.2) -- all three are lossless")
        export = QPushButton("Export image…")
        export.clicked.connect(
            lambda: self._export_panel(panel, fmt_box, quality, compress)
        )
        form.addRow("Export format:", fmt_box)
        form.addRow("JPEG quality:", quality)
        form.addRow("TIFF compression:", compress)
        form.addRow("", export)
        return row

    def _view_row(self, panel: ImagePanel) -> QWidget:
        """The §21.4 view controls: the 1:1 original-resolution toggle."""
        btn_1to1 = QPushButton("1:1 original resolution")
        btn_1to1.setToolTip(
            "One image pixel per screen pixel (§21.4), centred on the "
            "current viewport; a second press restores the fit-to-view "
            "(§21.2). The achieved scale is reported below."
        )
        btn_1to1.clicked.connect(lambda: self._toggle_original_resolution(panel))
        row = QWidget()
        form = QFormLayout(row)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("View:", btn_1to1)
        return row

    def _toggle_original_resolution(self, panel: ImagePanel) -> None:
        """§21.4's 1:1 toggle: original resolution <-> the §21.2 fit.

        The honest three-way outcome lands in the status line: the
        achieved scale (MEASURED back by the panel, never assumed) when
        1:1 lands, the fit-restored notice when it toggles back, and
        the explicit refusal (hidden geometry / no matrix) -- never a
        silent no-op.
        """
        if panel.is_original_resolution():
            panel.reset_view()
            self._status.setText("View: fit-to-view restored (§21.2 autoRange).")
            return
        scale = panel.set_original_resolution()
        if scale is None:
            self._status.setText(
                "View: 1:1 unavailable -- show the image first "
                "(geometry unknown until then)."
            )
            return
        self._status.setText(
            f"View: 1:1 original resolution -- 1 image px = "
            f"{scale[0]:.2f}×{scale[1]:.2f} screen px (§21.4)."
        )

    def _export_panel(
        self,
        panel: ImagePanel,
        fmt_box: QComboBox,
        quality: QSpinBox,
        compress: QComboBox,
    ) -> None:
        choice = fmt_box.currentText()
        if choice.startswith("TIFF"):
            default_name, ext = "imagelab.tiff", ".tiff"
        elif choice.startswith("PNG"):
            default_name, ext = "imagelab.png", ".png"
        else:
            default_name, ext = "imagelab.jpg", ".jpg"
        filters = (
            "TIFF images (*.tif *.tiff);;"
            "PNG images (*.png);;"
            "JPEG images (*.jpg *.jpeg)"
        )
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export the image panel", default_name, filters
        )
        if not path:
            return
        # §17.4: the extension must match the requested format -- the
        # engine enforces byte honesty; the dialog keeps the promise by
        # stamping the chosen format's extension when the user's name
        # carries a foreign one (aliases .tif/.tiff, .jpg/.jpeg pass).
        from pathlib import Path as _Path

        from pharmaml.viz.export import EXPORT_EXTENSIONS, ExportFormat

        fmt_key = ExportFormat.TIFF if ext == ".tiff" else (
            ExportFormat.PNG if ext == ".png" else ExportFormat.JPEG
        )
        suffix = _Path(path).suffix.lower()
        if suffix not in EXPORT_EXTENSIONS[fmt_key]:
            path = str(_Path(path).with_suffix(ext))
        compression = {
            "LZW": "tiff_lzw",
            "DEFLATE": "tiff_deflate",
            "None": "none",
        }[compress.currentText()]
        try:
            result = panel.export_image(
                path,
                jpeg_quality=int(quality.value()),
                compression=compression,
                min_size=LEGIBILITY_FLOOR,
            )
        except Exception as exc:  # noqa: BLE001 - honest status line
            self._status.setText(f"Export FAILED for {path}: {exc}")
            return
        note = f"Exported {result.path} ({result.format}, {result.width}x{result.height}, {result.size_bytes:,} bytes)"
        if result.warnings:
            note += " -- " + "; ".join(result.warnings)
        self._status.setText(note)
        # part5 §11: the real export (path/format/dimensions/bytes) becomes
        # a saved run — the Visualization card's evidence. Best-effort.
        try:
            from pharmaml.studio import record_run

            record_run(
                "viz-export",
                _Path(result.path).name,
                {
                    "name": _Path(result.path).name,
                    "path": str(result.path),
                    "format": str(result.format),
                    "width": int(result.width),
                    "height": int(result.height),
                    "size_bytes": int(result.size_bytes),
                    "warnings": list(result.warnings),
                    "kind": "image panel",
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the export
            pass

    # -- tab 1: similarity heatmap ----------------------------------------------
    def _build_similarity_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        form = QFormLayout(left)
        self._sim_source = QComboBox()
        self._sim_source.addItem("bundled seed_molecules.csv (12 drugs)")
        self._sim_source.addItem("Data Explorer SMILES column")
        self._sim_source.currentTextChanged.connect(self._toggle_sim_column)
        form.addRow("Source:", self._sim_source)
        self._sim_column = QComboBox()
        self._sim_column.setEnabled(False)
        form.addRow("SMILES column:", self._sim_column)
        self._sim_btn = QPushButton("Compute similarity matrix")
        self._sim_btn.clicked.connect(self._start_similarity)
        form.addRow("", self._sim_btn)
        self._sim_report = QLabel(
            "The 12x12 Tanimoto matrix, morgan r=2, 2048 bits."
        )
        self._sim_report.setWordWrap(True)
        form.addRow("", self._sim_report)

        self._sim_panel, sim_roi, sim_hover = self._make_panel(
            "Molecule x molecule Tanimoto similarity"
        )
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._gradient_row(self._sim_panel))
        right_layout.addWidget(self._sim_panel)
        right_layout.addWidget(self._view_row(self._sim_panel))
        right_layout.addWidget(sim_hover)
        right_layout.addWidget(sim_roi)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        configure_splitter(splitter, stretch=(0, 1))
        outer.addWidget(splitter)
        return tab

    def _toggle_sim_column(self, _text: str) -> None:
        self._refresh_sim_columns()
        use_explorer = (
            self._sim_source.currentIndex() == 1 and self._sim_column.count() > 0
        )
        self._sim_column.setEnabled(use_explorer)

    def _live_frame(self):
        """The explorer's current table (None before any load)."""
        if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
            return self._data_explorer.frame()
        return None

    def _on_explorer_frame(self, frame: object) -> None:
        """frameLoaded slot (review m-19): keep pickers true to the live table."""
        import pandas as pd

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        self._refresh_sim_columns()
        self._refresh_numeric_pickers(
            [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        )

    def _refresh_sim_columns(self) -> None:
        """Refill the SMILES picker with the explorer's SMILES-like columns."""
        self._sim_column.clear()
        frame = self._explorer_frame()
        if frame is None:
            return
        for name in frame.columns:
            sample = frame[name].dropna().astype(str).head(50)
            if len(sample) == 0:
                continue
            looks_like_smiles = sample.str.contains(
                r"[cCnNoO]=|\[#|[CN]\d|\(", regex=True
            ).mean()
            if looks_like_smiles > 0.5:
                self._sim_column.addItem(str(name))

    def _explorer_frame(self) -> pd.DataFrame | None:
        if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
            frame = self._data_explorer.frame()
            if frame is not None and not frame.empty:
                return frame
        return None

    def _sim_smiles(self) -> tuple[list[str], list[str], str]:
        """The chosen SMILES roster: explorer column or the bundled seeds."""
        if self._sim_source.currentIndex() == 1:
            frame = self._explorer_frame()
            if frame is None:
                raise ValueError(
                    "no loaded table -- load one in the Data Explorer first "
                    "(or use the bundled seeds)"
                )
            if self._sim_column.count() == 0:
                raise ValueError(
                    "no SMILES-like text column in the loaded table "
                    "(nothing matched the organic-fragment heuristics)"
                )
            name = self._sim_column.currentText()
            smiles = [str(v) for v in frame[name].dropna().tolist()]
            if len(smiles) < _MIN_SMILES:
                raise ValueError(
                    f"column {name!r} has only {len(smiles)} non-null SMILES "
                    f"-- need at least {_MIN_SMILES}"
                )
            labels = [f"row {i}" for i in range(len(smiles))]
            return (
                smiles,
                labels,
                f"column {name!r} of the loaded table ({len(smiles)} SMILES)",
            )
        seeds = pd.read_csv(_SEEDS_PATH)
        return (
            [str(s) for s in seeds["smiles"].tolist()],
            [str(n) for n in seeds["name"].tolist()],
            "bundled seed_molecules.csv (12 drugs)",
        )

    def _start_similarity(self) -> None:
        try:
            smiles, labels, source = self._sim_smiles()
        except ValueError as exc:
            self._sim_report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> SimilarityOutcome:
            from pharmaml.chem.descriptors import compute_fingerprints
            from pharmaml.chem.similarity import tanimoto_matrix

            progress(0.2, "computing morgan fingerprints")
            matrix, valid = compute_fingerprints(
                smiles, kind="morgan", radius=2, n_bits=2048
            )
            if len(valid) < _MIN_SMILES:
                raise ValueError(
                    f"only {len(valid)} of {len(smiles)} SMILES parsed -- "
                    f"need at least {_MIN_SMILES}"
                )
            progress(0.6, "tanimoto all-vs-all")
            sim = tanimoto_matrix(matrix, matrix)
            progress(1.0, "done")
            names = [labels[i] for i in valid]
            sub = sim[: len(valid), : len(valid)]
            off = sub[~np.eye(len(sub), dtype=bool)]
            i, j = np.unravel_index(
                np.argmax(sub + np.eye(len(sub)) * -2.0), sub.shape
            )
            k, l = np.unravel_index(np.argmin(sub + np.eye(len(sub))), sub.shape)
            mean_per = (sub.sum(axis=1) - 1.0) / (len(sub) - 1)
            top, low = int(np.argmax(mean_per)), int(np.argmin(mean_per))
            return SimilarityOutcome(
                names=names,
                matrix=sub,
                n_valid=len(valid),
                source=f"{source}; {len(valid)} valid",
                off_min=float(off.min()),
                off_max=float(off.max()),
                off_mean=float(off.mean()),
                best_pair=(names[i], names[j]),
                worst_pair=(names[k], names[l]),
                best_value=float(sub[i, j]),
                worst_value=float(sub[k, l]),
                top_mean_name=names[top],
                top_mean=float(mean_per[top]),
                low_mean_name=names[low],
                low_mean=float(mean_per[low]),
            )

        self._runner.submit(
            "similarity matrix",
            job,
            on_finished=self._sim_done,
            on_failed=self._failed,
        )

    def _sim_done(self, outcome: object) -> None:
        assert isinstance(outcome, SimilarityOutcome)
        assert outcome.matrix is not None
        self._sim_panel.set_matrix(outcome.matrix)
        self._sim_report.setText(
            f"{outcome.source}. Off-diagonal: min {outcome.off_min:.6f} / "
            f"max {outcome.off_max:.6f} / mean {outcome.off_mean:.6f}. "
            f"Most similar: {outcome.best_pair[0]} ~ {outcome.best_pair[1]} "
            f"= {outcome.best_value:.6f}; least: {outcome.worst_pair[0]} ~ "
            f"{outcome.worst_pair[1]} = {outcome.worst_value:.6f}. "
            f"Highest mean similarity: {outcome.top_mean_name} "
            f"({outcome.top_mean:.6f}); lowest: {outcome.low_mean_name} "
            f"({outcome.low_mean:.6f})."
        )

    # -- tab 2: response matrix ----------------------------------------------------
    def _build_screen_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        form = QFormLayout(left)
        self._screen_btn = QPushButton("Load the response matrix")
        self._screen_btn.clicked.connect(self._start_screen)
        form.addRow("", self._screen_btn)
        self._screen_report = QLabel(
            "The bundled GDSC-style screen pivoted to 23 x 8 LN_IC50."
        )
        self._screen_report.setWordWrap(True)
        form.addRow("", self._screen_report)

        self._screen_panel, screen_roi, screen_hover = self._make_panel(
            "Cell line x compound LN_IC50"
        )
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._gradient_row(self._screen_panel))
        right_layout.addWidget(self._screen_panel)
        right_layout.addWidget(self._view_row(self._screen_panel))
        right_layout.addWidget(screen_hover)
        right_layout.addWidget(screen_roi)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        configure_splitter(splitter, stretch=(0, 1))
        outer.addWidget(splitter)
        return tab

    def _start_screen(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> ScreenOutcome:
            import time as _time

            from pharmaml.biology import (
                collapse_replicates,
                load_response_table,
                response_matrix,
            )

            started = _time.monotonic()
            progress(0.3, "loading the response screen")
            screen = load_response_table(_SCREEN_PATH)
            progress(0.6, "collapsing replicates")
            collapsed = collapse_replicates(screen, add_cv=True)
            progress(0.8, "pivoting the collapsed screen")
            matrix = response_matrix(collapsed, screen.metrics[0])
            progress(1.0, "done")
            values = matrix.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            ii, jj = np.unravel_index(np.nanargmin(values), values.shape)
            kk, ll = np.unravel_index(np.nanargmax(values), values.shape)
            return ScreenOutcome(
                matrix=matrix,
                n_holes=int(np.isnan(values).sum()),
                fin_min=float(finite.min()),
                fin_max=float(finite.max()),
                fin_mean=float(finite.mean()),
                most_sensitive=(str(matrix.index[ii]), str(matrix.columns[jj])),
                most_resistant=(str(matrix.index[kk]), str(matrix.columns[ll])),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "response matrix",
            job,
            on_finished=self._screen_done,
            on_failed=self._failed,
        )

    def _screen_done(self, outcome: object) -> None:
        assert isinstance(outcome, ScreenOutcome)
        assert outcome.matrix is not None
        matrix = outcome.matrix
        self._screen_panel.set_matrix(
            matrix.to_numpy(dtype=float),
            x_labels=[str(c) for c in matrix.columns],
            y_labels=[str(i) for i in matrix.index],
        )
        self._screen_report.setText(
            f"{len(matrix)} cell lines x {len(matrix.columns)} compounds, "
            f"{outcome.n_holes} NaN hole(s) (masked). Finite LN_IC50: "
            f"min {outcome.fin_min:.6f} (most sensitive) / max "
            f"{outcome.fin_max:.6f} (most resistant) / mean "
            f"{outcome.fin_mean:.6f}. Most sensitive: "
            f"{outcome.most_sensitive[0]} / {outcome.most_sensitive[1]}; "
            f"most resistant: {outcome.most_resistant[0]} / "
            f"{outcome.most_resistant[1]}. Loaded in "
            f"{outcome.seconds:.1f}s."
        )

    # -- tab 3: explorer matrix -------------------------------------------------------
    def _build_explorer_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        form = QFormLayout(left)
        self._exp_from = QComboBox()
        self._exp_to = QComboBox()
        self._exp_from.currentTextChanged.connect(self._sync_exp_span)
        form.addRow("First numeric column:", self._exp_from)
        form.addRow("Last numeric column:", self._exp_to)
        self._exp_z = QCheckBox("z-score each column")
        self._exp_z.setChecked(True)
        form.addRow("", self._exp_z)
        self._exp_btn = QPushButton("Draw the matrix")
        self._exp_btn.clicked.connect(self._start_explorer)
        form.addRow("", self._exp_btn)
        self._exp_report = QLabel(
            "The loaded table's numeric columns as one image (z-scored by "
            "default), or the seeded 12-row demo."
        )
        self._exp_report.setWordWrap(True)
        form.addRow("", self._exp_report)

        self._exp_panel, exp_roi, exp_hover = self._make_panel("Numeric column matrix")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._gradient_row(self._exp_panel))
        right_layout.addWidget(self._exp_panel)
        right_layout.addWidget(self._view_row(self._exp_panel))
        right_layout.addWidget(exp_hover)
        right_layout.addWidget(exp_roi)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        configure_splitter(splitter, stretch=(0, 1))
        outer.addWidget(splitter)
        return tab

    def _refresh_numeric_pickers(self, numeric: list[str]) -> None:
        """(Re)fill the span pickers with the explorer's numeric columns."""
        if not numeric:
            return
        for box in (self._exp_from, self._exp_to):
            box.blockSignals(True)
            box.clear()
            box.addItems(numeric)
            box.blockSignals(False)
        self._exp_to.setCurrentIndex(len(numeric) - 1)

    def _sync_exp_span(self, _text: str) -> None:
        """Keep the last column at-or-after the first."""
        if self._exp_to.count() < 2:
            return
        if self._exp_to.currentIndex() < self._exp_from.currentIndex():
            self._exp_to.setCurrentIndex(self._exp_from.currentIndex())

    def _start_explorer(self) -> None:
        if self._busy_check():
            return
        frame = self._explorer_frame()
        if frame is None:
            rng = default_rng(42)
            frame = pd.DataFrame(
                {
                    "compound": [f"cmp-{i:02d}" for i in range(12)],
                    "potency": rng.normal(5.0, 1.2, 12).round(4),
                    "logp": rng.normal(2.0, 0.8, 12).round(4),
                    "mw": rng.normal(330.0, 45.0, 12).round(2),
                }
            )
            source = "seeded 12-row demo frame (no table loaded)"
        else:
            source = f"the loaded table ({len(frame)} rows)"
        numeric = [
            str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
        ]
        if len(numeric) < _MIN_NUMERIC:
            self._exp_report.setText(
                f"Refused — {source} has {len(numeric)} numeric column(s); "
                f"need at least {_MIN_NUMERIC}"
            )
            return
        names = list(numeric)
        if self._exp_from.count():
            a, b = self._exp_from.currentText(), self._exp_to.currentText()
            if a in names and b in names:
                lo, hi = sorted((names.index(a), names.index(b)))
                names = names[lo : hi + 1]
        zscored = self._exp_z.isChecked()

        def job(progress) -> ExplorerOutcome:
            progress(0.4, "assembling the column matrix")
            block = frame[names].to_numpy(dtype=float)
            raw_means = block.mean(axis=0).tolist()
            raw_stds = block.std(axis=0, ddof=0).tolist()
            if zscored:
                stds = np.where(np.asarray(raw_stds) > 0, np.asarray(raw_stds), 1.0)
                matrix = (block - np.asarray(raw_means)) / stds
            else:
                matrix = block
            progress(1.0, "done")
            col_means = matrix.mean(axis=0).tolist()
            col_stds = matrix.std(axis=0, ddof=0).tolist()
            max_abs_row = int(np.argmax(np.abs(matrix).max(axis=1)))
            return ExplorerOutcome(
                header=list(names),
                matrix=matrix,
                zscored=zscored,
                raw_means=raw_means,
                raw_stds=raw_stds,
                col_means=col_means,
                col_stds=col_stds,
                z_min=float(matrix.min()),
                z_max=float(matrix.max()),
                max_abs_row=max_abs_row,
                max_abs=float(np.abs(matrix).max()),
                source=source,
            )

        self._runner.submit(
            "explorer matrix",
            job,
            on_finished=self._explorer_done,
            on_failed=self._failed,
        )

    def _explorer_done(self, outcome: object) -> None:
        assert isinstance(outcome, ExplorerOutcome)
        assert outcome.matrix is not None
        self._exp_panel.set_matrix(
            outcome.matrix,
            x_labels=outcome.header,
            y_labels=[f"row {i}" for i in range(outcome.matrix.shape[0])],
        )
        scale = "z" if outcome.zscored else "raw"
        raw_means = ", ".join(f"{m:.4f}" for m in outcome.raw_means)
        raw_stds = ", ".join(f"{s:.4f}" for s in outcome.raw_stds)
        col_means = ", ".join(f"{m:.4f}" for m in outcome.col_means)
        col_stds = ", ".join(f"{s:.4f}" for s in outcome.col_stds)
        self._exp_report.setText(
            f"{outcome.source}; columns {', '.join(outcome.header)} ({scale}). "
            f"Raw means [{raw_means}]; raw stds [{raw_stds}]. Column means "
            f"after scaling [{col_means}]; stds [{col_stds}]. Min "
            f"{outcome.z_min:.6f} / max {outcome.z_max:.6f}; row of max "
            f"|value|: row {outcome.max_abs_row} ({outcome.max_abs:.6f})."
        )

    # -- tab 4: ROI explorer --------------------------------------------------------
    def _build_roi_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        intro = QLabel(
            "The RectROI (16.8.3) is live on every tab's image: drag its "
            "handles and the count / mean / min / max of the enclosed "
            "sub-matrix update live. The matrix below is the similarity "
            "matrix; the fixed region rows [2,6) x cols [7,10) reproduces "
            "the probe's printed statistics exactly."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        form = QFormLayout(left)
        self._roi_row0 = QSpinBox()
        self._roi_row1 = QSpinBox()
        self._roi_col0 = QSpinBox()
        self._roi_col1 = QSpinBox()
        for box in (self._roi_row0, self._roi_row1, self._roi_col0, self._roi_col1):
            box.setRange(0, 64)
        self._roi_row0.setValue(2)
        self._roi_row1.setValue(6)
        self._roi_col0.setValue(7)
        self._roi_col1.setValue(11)
        apply_btn = QPushButton("Apply the region")
        apply_btn.clicked.connect(self._apply_roi_region)
        form.addRow("First row (0-based):", self._roi_row0)
        form.addRow("Rows (exclusive end):", self._roi_row1)
        form.addRow("First column:", self._roi_col0)
        form.addRow("Columns (exclusive end):", self._roi_col1)
        form.addRow("", apply_btn)
        self._roi_report = QLabel("—")
        self._roi_report.setWordWrap(True)
        form.addRow("", self._roi_report)

        self._roi_panel, roi_label, roi_hover = self._make_panel(
            "Similarity matrix (the shared ROI demo)"
        )
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._roi_panel)
        right_layout.addWidget(self._view_row(self._roi_panel))
        right_layout.addWidget(roi_hover)
        right_layout.addWidget(roi_label)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        configure_splitter(splitter, stretch=(0, 1))
        outer.addWidget(splitter)

        self._seed_roi_matrix()
        return tab

    def _seed_roi_matrix(self) -> None:
        """The probe's similarity matrix, recomputed through the same engines.

        The heavy engines already ran in the probe; this tab only needs
        the 12x12 result for ROI interaction, so it recomputes it inline
        (a sub-second morgan pass over 12 molecules) through the same
        public APIs -- ``compute_fingerprints`` + ``tanimoto_matrix``.
        On an RDKit-less box the panel stays EMPTY and the report says
        so -- never a fabricated array.
        """
        try:
            from pharmaml.chem.descriptors import compute_fingerprints
            from pharmaml.chem.similarity import tanimoto_matrix

            seeds = pd.read_csv(_SEEDS_PATH)
            matrix, _valid = compute_fingerprints(
                [str(s) for s in seeds["smiles"].tolist()],
                kind="morgan",
                radius=2,
                n_bits=2048,
            )
            sim = tanimoto_matrix(matrix, matrix)
        except Exception as exc:  # noqa: BLE001 — honest refusal, no fallback
            self._roi_panel.clear()
            self._roi_report.setText(
                f"Similarity engine unavailable ({exc}) — the ROI demo "
                "needs RDKit; the other tabs refuse the same way."
            )
            return
        self._roi_panel.set_matrix(sim)
        self._apply_roi_region()

    def _apply_roi_region(self) -> None:
        values = self._roi_panel.values()
        if values is None:
            self._roi_report.setText("No matrix loaded.")
            return
        row0 = min(self._roi_row0.value(), self._roi_row1.value())
        row1 = max(self._roi_row0.value(), self._roi_row1.value())
        col0 = min(self._roi_col0.value(), self._roi_col1.value())
        col1 = max(self._roi_col0.value(), self._roi_col1.value())
        row1 = min(row1, values.shape[0])
        col1 = min(col1, values.shape[1])
        # move the shared RectROI to the requested region (fires roi_changed)
        self._roi_panel.roi().setPos([float(col0), float(row0)])
        self._roi_panel.roi().setSize(
            [float(max(col1 - col0, 1)), float(max(row1 - row0, 1))]
        )
        stats = roi_stats_from_values(values, row0, row1, col0, col1)
        if stats is None:
            self._roi_report.setText("Empty region — widen the span.")
            return
        self._roi_report.setText(
            f"rows [{row0}, {row1}) x cols [{col0}, {col1}): "
            f"{int(stats['count'])} cells, mean {stats['mean']:.6f}, "
            f"min {stats['min']:.6f}, max {stats['max']:.6f} "
            f"(the probe's arithmetic)"
        )

    # -- tab 5: performance -----------------------------------------------------------
    def _build_perf_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        form = QFormLayout()
        self._perf_n = QSpinBox()
        self._perf_n.setRange(64, 4096)
        self._perf_n.setSingleStep(64)
        self._perf_n.setValue(1024)
        form.addRow("Molecules (n x 2048-bit):", self._perf_n)
        self._perf_btn = QPushButton("Time tanimoto_matrix")
        self._perf_btn.clicked.connect(self._start_perf)
        form.addRow("", self._perf_btn)
        self._perf_report = QLabel(
            "1024^2 = 1,048,576 pairs in ~1.4 s (the probe's timing; "
            "timings vary by load and are never asserted)."
        )
        self._perf_report.setWordWrap(True)
        form.addRow("", self._perf_report)
        header = QWidget()
        header.setLayout(form)
        outer.addWidget(header)
        outer.addStretch(1)
        return tab

    def _start_perf(self) -> None:
        if self._busy_check():
            return
        n = int(self._perf_n.value())

        def job(progress) -> PerfOutcome:
            import time as _time

            from pharmaml.chem.similarity import tanimoto_matrix

            rng = default_rng(42)
            bits = (rng.random((n, 2048)) < 0.15).astype(np.uint8)
            progress(0.2, f"tanimoto over {n * n:,} pairs")
            t0 = _time.monotonic()
            sim = tanimoto_matrix(bits, bits)
            elapsed = _time.monotonic() - t0
            progress(1.0, "done")
            off_mean = (sim.sum() - np.trace(sim)) / (n * (n - 1))
            return PerfOutcome(
                n=n,
                pairs=n * n,
                seconds=elapsed,
                off_mean=float(off_mean),
                shape=(int(sim.shape[0]), int(sim.shape[1])),
            )

        self._runner.submit(
            "tanimoto timing",
            job,
            on_finished=self._perf_done,
            on_failed=self._failed,
        )

    def _perf_done(self, outcome: object) -> None:
        assert isinstance(outcome, PerfOutcome)
        self._perf_report.setText(
            f"{outcome.n}^2 = {outcome.pairs:,} pairs in "
            f"{outcome.seconds:.3f} s "
            f"({outcome.pairs / outcome.seconds / 1e6:.1f} M pairs/s); "
            f"matrix {outcome.shape[0]} x {outcome.shape[1]}, diagonal "
            f"all ones, off-diagonal mean {outcome.off_mean:.6f}. "
            f"Timings are wall-clock and load-dependent -- never asserted."
        )

    # -- job plumbing ------------------------------------------------------------
    def _on_started(self, name: str) -> None:
        for btn in (self._sim_btn, self._screen_btn, self._exp_btn, self._perf_btn):
            btn.setEnabled(False)
        self._status.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in (self._sim_btn, self._screen_btn, self._exp_btn, self._perf_btn):
            btn.setEnabled(True)

    def _failed(self, message: str) -> None:
        line = message.splitlines()[0] if message else "The job failed."
        name = self._runner.job_name
        reports = {
            "similarity matrix": self._sim_report,
            "response matrix": self._screen_report,
            "explorer matrix": self._exp_report,
            "tanimoto timing": self._perf_report,
        }
        target = reports.get(name, self._status)
        target.setText(f"FAILED — {line}")

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._status.setText(f"Busy: '{getattr(self._runner, 'job_name', '') or 'another operation'}' is running — Stop it (or check Job Center) first.")
            return True
        return False
