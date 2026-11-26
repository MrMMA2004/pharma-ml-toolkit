"""Studio access to MD study metric plots and figure exports."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pharmaml.md import MDStudy, build_study
from pharmaml.viz import close_figure, plot_md_correlation, plot_md_series, save_figure

from ..widgets.responsive import configure_splitter
from ..workers.job_worker import JobRunner


class MolecularDynamicsPage(QWidget):
    """Load an MD study and visualize one metric or two aligned metrics."""

    def __init__(self, parent: QWidget | None = None, runner: JobRunner | None = None) -> None:
        super().__init__(parent)
        self._runner = runner or JobRunner(self)
        self._study: MDStudy | None = None
        self._figure: Any = None
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        title = QLabel("Molecular dynamics: metric plots")
        title.setObjectName("studioPageTitle")
        layout.addWidget(title)
        note = QLabel(
            "Open a ligand folder or a study containing ligand folders. "
            "Confirm the units and axis labels from your input files. "
            "Correlations require matching axes and are descriptive, not causal."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self._open_btn = QPushButton("Open study folder…")
        self._open_btn.clicked.connect(self._open_dialog)
        self._export_btn = QPushButton("Export figure…")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_dialog)
        controls.addWidget(self._open_btn)
        controls.addWidget(self._export_btn)
        # review P-10 fix: a bundled example study button — the page was the
        # emptiest first-open (99% background) with no guided entry.
        self._example_btn = QPushButton("Try the example study")
        self._example_btn.setToolTip(
            "Load a small generated two-ligand study (RMSD / RMSF exports) "
            "to try the plots without any data of your own"
        )
        self._example_btn.clicked.connect(self._load_example)
        controls.addWidget(self._example_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        options = QWidget()
        form = QFormLayout(options)
        self._ligand = QComboBox()
        self._metric = QComboBox()
        self._second_metric = QComboBox()
        self._x_label = QLineEdit("Time / index")
        self._y_label = QLineEdit("Value")
        # §17 F (Phase 5): the metric choices carry their glossary
        # explanations, so a hover states the unit and the caveat
        # (alignment for RMSD, average-structure for RMSF…).
        from pharmaml.studio import apply_help as _help

        _help(self._metric, "rmsd")
        _help(self._second_metric, "rmsf")
        _help(self._ligand, "plif")
        form.addRow("Ligand:", self._ligand)
        form.addRow("Metric:", self._metric)
        form.addRow("Second metric:", self._second_metric)
        form.addRow("X-axis label:", self._x_label)
        form.addRow("Y-axis label:", self._y_label)
        self._series_btn = QPushButton("Plot metric")
        self._correlation_btn = QPushButton("Plot correlation")
        self._series_btn.clicked.connect(self._plot_series)
        self._correlation_btn.clicked.connect(self._plot_correlation)
        form.addRow(self._series_btn)
        form.addRow(self._correlation_btn)
        self._ligand.currentIndexChanged.connect(self._update_metrics)
        self._metric.currentIndexChanged.connect(self._invalidate_figure)
        self._second_metric.currentIndexChanged.connect(self._invalidate_figure)
        self._x_label.textChanged.connect(self._invalidate_figure)
        self._y_label.textChanged.connect(self._invalidate_figure)
        splitter.addWidget(options)
        self._preview = QLabel("Open a study to begin")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(300, 220)
        # review C-1/P-7/P-10 fix: an honest empty state in the big preview
        # panel — glyph + headline + guidance instead of a bare white box.
        from PySide6.QtWidgets import QStackedWidget

        from ..widgets.empty_state import EmptyState

        self._empty = EmptyState(
            headline="No study loaded yet",
            guidance=(
                "A study is a folder of ligand sub-folders with .xvg exports "
                "(RMSD, RMSF, hydrogen bonds…). Open yours, or press "
                "'Try the example study' above to see the plots immediately."
            ),
            glyph="∿",
        )
        self._stack = QStackedWidget()
        self._stack.addWidget(self._preview)
        self._stack.addWidget(self._empty)
        self._stack.setCurrentWidget(self._empty)
        splitter.addWidget(self._stack)
        configure_splitter(splitter, min_sizes=(220, 300), stretch=(0, 1))
        splitter.setSizes([280, 700])
        layout.addWidget(splitter, stretch=1)
        self._status = QLabel("No study loaded")
        self._status.setObjectName("studioStatus")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._set_plot_buttons(False)

    def _set_plot_buttons(self, enabled: bool) -> None:
        self._series_btn.setEnabled(enabled)
        self._correlation_btn.setEnabled(enabled and self._second_metric.count() > 1)

    def _clear_figure(self) -> None:
        if self._figure is not None:
            close_figure(self._figure)
        self._figure = None
        self._pixmap = None
        self._preview.clear()
        self._export_btn.setEnabled(False)

    def _invalidate_figure(self, *_args: object) -> None:
        if self._figure is not None:
            self._clear_figure()
            self._stack.setCurrentWidget(self._empty)
            self._status.setText("Inputs changed; plot again before exporting.")

    def _load_example(self) -> None:
        """Review P-10 fix: generate + load a small bundled-style study."""
        if self._runner.busy:
            self._status.setText("Another job is running; try again when it finishes.")
            return
        from tempfile import mkdtemp

        root = Path(mkdtemp(prefix="pharmaml-md-example-")) / "example_study"
        for ligand, drift in (("ligand_A", 0.02), ("ligand_B", 0.05)):
            folder = root / ligand
            folder.mkdir(parents=True)
            steps = list(range(0, 100, 2))
            # plain GROMACS-style two-column .xvg (see study.py docstring)
            (folder / "ligand_rmsd.xvg").write_text(
                "# t rmsd\n" + "\n".join(
                    f"{t} {0.10 + drift * t + 0.01 * ((t % 10) - 5):.4f}"
                    for t in steps
                ) + "\n"
            )
            (folder / "ligand_rmsf.xvg").write_text(
                "# res rmsf\n" + "\n".join(
                    f"{i} {0.08 + 0.02 * ((i * 7) % 13) / 13.0:.4f}"
                    for i in range(1, 21)
                ) + "\n"
            )
        self.load_path(root)

    def _open_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open MD study folder")
        if folder:
            self.load_path(Path(folder))

    def load_path(self, folder: str | Path) -> None:
        """Load a study asynchronously; failed loads clear any previous results."""
        if self._runner.busy:
            self._status.setText("Another job is running; try again when it finishes.")
            return
        path = Path(folder)
        self._study = None
        self._ligand.clear()
        self._metric.clear()
        self._second_metric.clear()
        self._clear_figure()
        self._set_plot_buttons(False)
        self._open_btn.setEnabled(False)
        self._status.setText(f"Loading {path}…")

        def job(_progress):
            return build_study(path)

        self._runner.submit("Load MD study", job, on_finished=self._loaded, on_failed=self._load_failed)

    def _loaded(self, study: MDStudy) -> None:
        self._study = study
        self._ligand.blockSignals(True)
        self._ligand.clear()
        self._ligand.addItems(study.ligand_names)
        self._ligand.blockSignals(False)
        self._update_metrics()
        self._open_btn.setEnabled(True)
        # review C-1 fix: real content replaces the empty state
        self._stack.setCurrentWidget(self._preview)
        self._status.setText(
            f"Loaded {len(study)} ligand(s). Loader may convert units or smooth H-bonds; "
            "verify data and axis labels."
        )

    def study(self) -> MDStudy | None:
        """The actually-loaded study (the §11 evidence context reads it)."""
        return self._study

    def _load_failed(self, message: str) -> None:
        self._open_btn.setEnabled(True)
        self._status.setText(f"Could not load study: {message.splitlines()[0]}")

    def _update_metrics(self, *_args: object) -> None:
        self._clear_figure()
        self._metric.clear()
        self._second_metric.clear()
        if self._study is None or not self._ligand.currentText():
            self._set_plot_buttons(False)
            return
        keys = self._study.ligand(self._ligand.currentText()).metric_keys
        self._metric.addItems(keys)
        self._second_metric.addItems(keys)
        if len(keys) > 1:
            self._second_metric.setCurrentIndex(1)
        self._set_plot_buttons(bool(keys))

    def _show_figure(self, fig: Any) -> None:
        # Create the new preview before discarding the previous figure.
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format="png", dpi=100)
            pixmap = QPixmap()
            if not pixmap.loadFromData(buffer.getvalue()):
                raise ValueError("Could not render figure preview")
        except Exception:
            close_figure(fig)
            raise
        self._clear_figure()
        self._figure = fig
        self._pixmap = pixmap
        self._fit_preview()
        self._export_btn.setEnabled(True)

    def _fit_preview(self) -> None:
        if self._pixmap is not None:
            self._preview.setPixmap(self._pixmap.scaled(
                self._preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_preview()

    def _plot_series(self) -> None:
        if self._study is None:
            return
        key = self._metric.currentText()
        try:
            item = self._study.ligand(self._ligand.currentText()).series[key]
            result = plot_md_series(item, title=f"{self._ligand.currentText()} — {key}",
                                    x_label=self._x_label.text(), y_label=self._y_label.text(),
                                    close=False)
            self._show_figure(result["figure"])
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self._clear_figure()
            self._status.setText(f"Could not plot metric: {exc}")
            return
        self._status.setText("Metric plotted. Verify axis labels and units before sharing.")

    def _plot_correlation(self) -> None:
        if self._study is None:
            return
        first, second = self._metric.currentText(), self._second_metric.currentText()
        try:
            if first == second:
                raise ValueError("Select two different metrics")
            ligand = self._study.ligand(self._ligand.currentText())
            result = plot_md_correlation({first: ligand.series[first], second: ligand.series[second]},
                                         title=f"{ligand.name} — metric correlations",
                                         close=False)
            self._show_figure(result["figure"])
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self._clear_figure()
            self._status.setText(f"Could not plot correlation: {exc}")
            return
        self._status.setText("Pearson correlation of aligned observations; not evidence of causation.")

    def _export_dialog(self) -> None:
        if self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export MD figure", "md-figure.png",
                                              "Figures (*.png *.svg *.pdf)")
        if path:
            self.export_path(path)

    def export_path(self, path: str | Path) -> None:
        """Save the current figure as PNG, SVG or PDF based on its extension."""
        if self._figure is None:
            self._status.setText("Plot a figure before exporting.")
            return
        try:
            saved = save_figure(self._figure, path)
        except Exception as exc:  # noqa: BLE001 - shown in the UI
            self._status.setText(f"Could not export figure: {exc}")
            return
        # §12 (part5 round 4): every exported image is reopened and
        # validated for format + dimensions + nonzero size before
        # success is reported -- the same §17.4 discipline the raster
        # engine applies, extended to the vector/PDF figure paths the
        # MD page owns.  A failed validation is an honest failure line,
        # never a "Saved" over a broken file.
        check = self._validate_export(saved)
        if not check["ok"]:
            self._status.setText(f"Export FAILED for {saved}: {check['reason']}")
            return
        size = Path(saved).stat().st_size
        note = f"Saved figure to {saved} ({check['format']}"
        if check.get("width"):
            note += f", {check['width']}x{check['height']}"
        note += f", {size:,} bytes, validated)"
        self._status.setText(note)
        # part5 §11: the real export becomes a saved run (best-effort).
        try:
            from pharmaml.studio import record_run

            record_run(
                "viz-export",
                Path(saved).name,
                {
                    "name": Path(saved).name,
                    "path": str(saved),
                    "format": check["format"],
                    "width": check.get("width"),
                    "height": check.get("height"),
                    "size_bytes": int(size),
                    "kind": "md figure",
                },
            )
        except Exception:  # noqa: BLE001, S110 -- the record must never fail the export
            pass

    @staticmethod
    def _validate_export(path: Path) -> dict:
        """§12 reopen-check for a saved MD figure (never raises)."""
        from pharmaml.viz.export import sniff_format

        target = Path(path)
        report: dict = {"ok": False, "reason": "", "format": None,
                        "width": None, "height": None}
        try:
            if not target.is_file():
                report["reason"] = "file does not exist after save"
                return report
            size = target.stat().st_size
            if size == 0:
                report["reason"] = "file is empty"
                return report
            sniffed = sniff_format(target)
            if sniffed == "unknown" and target.suffix.lower() == ".pdf":
                # PDF is a matplotlib-native figure format (no raster
                # magic bytes): accept a real %PDF header, else refuse.
                head = target.read_bytes()[:1024]
                if head.startswith(b"%PDF"):
                    report.update(ok=True, format="PDF")
                    return report
                report["reason"] = "not a readable PDF (missing %PDF header)"
                return report
            if sniffed == "unknown":
                report["reason"] = "magic bytes match no supported format"
                return report
            report["format"] = sniffed
            if sniffed == "SVG":
                report.update(ok=True, width=None, height=None)
                return report
            try:
                from PIL import Image

                with Image.open(target) as reopened:
                    reopened.load()
                    report["width"], report["height"] = reopened.width, reopened.height
            except Exception as exc:  # noqa: BLE001 - reopen failure
                report["reason"] = f"reopen failed: {exc}"
                return report
            report["ok"] = True
        except Exception as exc:  # noqa: BLE001 - the check itself must not raise
            report["reason"] = f"validation error: {exc}"
        return report

    def closeEvent(self, event) -> None:
        self._clear_figure()
        super().closeEvent(event)
