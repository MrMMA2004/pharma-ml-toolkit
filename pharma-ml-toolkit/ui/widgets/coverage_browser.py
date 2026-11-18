"""The backend-figure coverage browser (part8 §4.1/§4.2 GUI surface).

A new Image Lab tab ("Backend figures"): the §4.2 registry drives a
filterable table of every public visualization function, and the
shared :class:`~ui.widgets.figure_canvas_host.FigureCanvasHost`
renders the ORIGINAL backend figure for the functions whose inputs
can be satisfied from the loaded dataset (numeric columns, target
labels, fitted artifacts where present).

The tab is the honest §4.2 surface: each row shows the integration
status (wired / native-equivalent / backend-only), and only rows
with a real GUI invocation are launchable — a backend-only function
is visible in the table but marked as such (never a dead button).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pharmaml.studio.plot_coverage import (
    BACKEND_ONLY,
    NATIVE_EQUIVALENT,
    WIRED,
    plot_coverage_table,
)

from ..widgets.figure_canvas_host import FigureCanvasHost

_STATUS_LABEL = {
    WIRED: "wired",
    NATIVE_EQUIVALENT: "Qt-native twin",
    BACKEND_ONLY: "backend only",
}


def _numeric_columns(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return []
    return [
        str(c) for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c])
    ]


def _label_columns(frame: pd.DataFrame | None) -> list[str]:
    if frame is None or frame.empty:
        return []
    return [
        str(c) for c in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[c])
    ]


class CoverageBrowser(QWidget):
    """§4.2: the registry table + the shared renderer, one tab."""

    def __init__(
        self,
        parent: QWidget | None = None,
        frame_provider: Callable[[], pd.DataFrame | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._frame_provider = frame_provider
        self._entries = plot_coverage_table()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        head = QHBoxLayout()
        self._filter = QComboBox()
        self._filter.addItem("All workspaces", "all")
        self._filter.addItem("Wired only", WIRED)
        self._filter.currentIndexChanged.connect(self._repopulate)
        head.addWidget(QLabel("Show:"))
        head.addWidget(self._filter)
        self._status_line = QLabel("")
        self._status_line.setObjectName("studioCardDetail")
        head.addStretch(1)
        head.addWidget(self._status_line)
        outer.addLayout(head)

        # §4.2 table: function | workspace | title | status
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Function", "Workspace", "What it plots", "GUI status"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        self._table.itemDoubleClicked.connect(self._on_double)
        outer.addWidget(self._table, stretch=2)

        # the §4.1 shared renderer
        self._host = FigureCanvasHost()
        outer.addWidget(self._host, stretch=3)

        self._repopulate()

    # -- table --------------------------------------------------------------------
    def _repopulate(self) -> None:
        scope = self._filter.currentData()
        rows = [e for e in self._entries if scope in (None, "all", e.status)]
        self._table.setRowCount(len(rows))
        for i, entry in enumerate(rows):
            for j, text in enumerate(entry.as_row()[:2] + (entry.title,)):
                item = QTableWidgetItem(text)
                if j == 3:
                    item.setText(_STATUS_LABEL[entry.status])
                self._table.setItem(i, j, item)
            # status column is index 3
            self._table.setItem(i, 3, QTableWidgetItem(_STATUS_LABEL[entry.status]))
        self._status_line.setText(
            f"{len(rows)} of {len(self._entries)} functions — "
            f"double-click a wired row to render it from the loaded dataset"
        )

    def _current_entry(self):
        row = self._table.currentRow()
        if row < 0:
            return None
        fn = self._table.item(row, 0).text()
        return next((e for e in self._entries if e.function == fn), None)

    def _on_double(self, item: QTableWidgetItem) -> None:
        self.launch_current()

    # -- §4.1 rendering -------------------------------------------------------------
    def launch_current(self) -> bool:
        """Render the selected function against the loaded dataset."""
        entry = self._current_entry()
        if entry is None:
            return False
        if entry.status != WIRED:
            self._host.clear()
            self._status_line.setText(
                f"{entry.function} — {entry.status}: no GUI invocation yet "
                "(see the coverage table)"
            )
            return False
        builder = self._builder_for(entry)
        if builder is None:
            self._host.clear()
            self._status_line.setText(
                f"{entry.function}: needs inputs the loaded dataset "
                "cannot supply (load a numeric table first)"
            )
            return False
        # statplots path: the builder stashes the rendered PNG bytes
        self._png_snapshot: bytes | None = None
        self._host.run_callable(builder)
        if getattr(self, "_png_snapshot", None):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                fh.write(self._png_snapshot)
                stable = fh.name
            self._host.set_rendered_image(stable)
            self._status_line.setText(f"{entry.function}: figure (rendered artifact)")
            return True
        state = self._host.state()
        self._status_line.setText(
            f"{entry.function}: {state}"
            + ("" if state != "error" else " — see the panel below")
        )
        return state == "figure"

    def _builder_for(self, entry) -> Callable[[], Any] | None:
        """Build the backend call from the live dataset.

        The invocation map is intentionally small and honest: each
        entry is the MINIMAL correct call for its function family,
        built from the frame's real columns.  Functions needing
        artifacts a plain frame cannot supply (SHAP arrays, docking
        poses, MD trajectories) are not fabricated here.
        """
        frame = self._frame_provider() if self._frame_provider else None
        if frame is None or frame.empty:
            return None
        numeric = _numeric_columns(frame)
        labels = _label_columns(frame)
        if not numeric:
            return None
        x = numeric[0]
        y = numeric[1] if len(numeric) > 1 else numeric[0]

        import importlib

        module = importlib.import_module(f"pharmaml.viz.{entry.module}")
        fn = getattr(module, entry.function.split(".")[-1])
        name = entry.function

        # -- statplots family (path-based API: render to a temp PNG, then
        #    display the backend's own artifact via the host) ---------------------
        def _png_call(call):
            import tempfile
            from pathlib import Path

            def build():
                # render to a temp PNG, then keep the artifact bytes
                # (the tempdir vanishes at scope exit, by design)
                with tempfile.TemporaryDirectory() as tmp:
                    png = Path(tmp) / "render.png"
                    call(str(png))
                    self._png_snapshot = png.read_bytes()

            return build

        if name == "statplots.plot_histogram":
            return _png_call(lambda png: fn(frame[x].dropna(), png))
        if name == "statplots.plot_box":
            return _png_call(lambda png: fn(frame[y].dropna(), png))
        if name == "statplots.plot_ecdf":
            return _png_call(lambda png: fn(frame[x].dropna(), png))
        if name == "statplots.plot_regression":
            return _png_call(lambda png: fn(frame[x], frame[y], png))

        # -- ml_plots frame-level family -------------------------------------------
        if name == "ml_plots.plot_class_distribution":
            target = labels[0] if labels else numeric[-1]
            counts = frame[target].value_counts()

            def _build():
                return fn(dict(counts))

            return _build
        if name == "ml_plots.plot_correlation_heatmap":
            sub = frame[numeric]
            return lambda: fn(sub.corr())
        if name == "ml_plots.plot_residuals":
            # only honest residuals: a real prediction column pair
            for cand in ("y_pred", "pred", "prediction"):
                if cand in frame.columns and y in frame.columns:
                    return lambda: fn(frame[y], frame[cand])
            return None
        if name in ("ml_plots.plot_predicted_vs_actual",):
            for actual in ("y_true", "actual", y):
                for pred in ("y_pred", "pred", "prediction"):
                    if actual in frame.columns and pred in frame.columns:
                        return lambda: fn(frame[actual], frame[pred])
            return None
        if name == "ml_plots.plot_learning_curve":
            # signature: (model, X, y, cv, scoring...) — a REAL fit on
            # the loaded frame, never fabricated arrays
            from sklearn.linear_model import Ridge

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 20:
                return None

            def _build():
                model = Ridge()
                return fn(model, Xf, yf, cv=min(5, len(Xf) // 4))

            return _build
        if name == "ml_plots.plot_cv_scores":
            # real 5-fold CV scores from the frame
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 20:
                return None

            def _build():
                scores = cross_val_score(Ridge(), Xf, yf, cv=5)
                return fn(scores, metric="R2")

            return _build
        if name == "ml_plots.plot_model_comparison":
            # signature: (leaderboard: DataFrame, metric=None) — the
            # leaderboard must carry a metric column; build a real one
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 20:
                return None

            def _build():
                rows = []
                for label, model in (
                    ("Ridge", Ridge()),
                    ("Ridge (alpha=0.1)", Ridge(alpha=0.1)),
                    ("Ridge (alpha=10)", Ridge(alpha=10.0)),
                ):
                    scores = cross_val_score(model, Xf, yf, cv=5)
                    rows.append(
                        {"model": label, "R2_mean": scores.mean(),
                         "R2_std": scores.std()}
                    )
                return fn(pd.DataFrame(rows), metric="R2")

            return _build
        if name == "ml_plots.plot_feature_importance":
            # signature: (model_or_mapping, feature_names, top_n) — a
            # real fitted forest's importances
            from sklearn.ensemble import RandomForestRegressor

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 20:
                return None

            def _build():
                forest = RandomForestRegressor(
                    n_estimators=50, random_state=0
                ).fit(Xf, yf)
                return fn(forest, feature_names=list(Xf.columns))

            return _build
        if name == "ml_plots.plot_permutation_importance":
            # the record needs feature/importance_mean/importance_std —
            # build the honest frame the backend's own extractor takes
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 20 or len(numeric) < 2:
                return None

            def _build():
                forest = RandomForestRegressor(
                    n_estimators=30, random_state=0
                ).fit(Xf, yf)
                imp = permutation_importance(
                    forest, Xf, yf, n_repeats=3, random_state=0
                )
                record = pd.DataFrame(
                    {
                        "feature": list(Xf.columns),
                        "importance_mean": imp.importances_mean,
                        "importance_std": imp.importances_std,
                    }
                )
                return fn(record, top_n=min(12, len(Xf.columns)))

            return _build
        if name == "ml_plots.plot_applicability_domain":
            # signature: (result: ApplicabilityDomainResult) — the REAL
            # backend analysis on the loaded frame
            from sklearn.linear_model import Ridge

            from pharmaml.ml.applicability import applicability_domain

            Xf = frame[numeric].dropna()
            yf = frame[y].loc[Xf.index]
            if len(Xf) < 30:
                return None

            def _build():
                model = Ridge().fit(Xf, yf)
                split = max(1, int(len(Xf) * 0.75))
                result = applicability_domain(
                    model, Xf.iloc[:split], yf.iloc[:split],
                    Xf.iloc[split:], n_models=15,
                )
                return fn(result)

            return _build
        if name == "ml_plots.plot_confusion_matrix":
            # a REAL confusion matrix from a fitted classifier when a
            # label column exists; otherwise honest refusal
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import confusion_matrix
            from sklearn.model_selection import train_test_split

            target = labels[0] if labels else None
            if target is None or len(frame) < 30 or len(numeric) < 1:
                return None

            def _build():
                Xf = frame[numeric].dropna()
                yf = frame[target].loc[Xf.index]
                if yf.nunique() < 2:
                    raise ValueError("target column needs >= 2 classes")
                Xtr, Xte, ytr, yte = train_test_split(
                    Xf, yf, test_size=0.3, random_state=0, stratify=yf
                )
                clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
                matrix = confusion_matrix(yte, clf.predict(Xte))
                return fn(matrix, labels=list(map(str, sorted(set(yte)))))

            return _build
        if name in ("ml_plots.plot_roc_curves", "ml_plots.plot_pr_curves",
                    "ml_plots.plot_calibration_curve"):
            return None  # needs fitted-classifier artifacts, not a frame
        if name == "ml_plots.plot_training_history":
            # history: the real DL page owns this artifact; a plain frame
            # cannot supply an honest one
            return None

        # -- pharma_plots ----------------------------------------------------------
        if name == "pharma_plots.plot_concentration_profile":
            tcol = next((c for c in ("time", "t", "Time") if c in frame.columns), None)
            if tcol is None:
                return None
            return lambda: fn(frame[tcol].to_numpy(), frame[y].to_numpy())
        if name == "pharma_plots.plot_dose_accumulation":
            return None  # needs dosing regimen artifacts

        # -- generative_plots -------------------------------------------------------
        if name == "generative_plots.plot_sa_distribution":
            # signature: (result, ...) — the generative run owns this
            # artifact; a plain frame cannot supply an honest one
            return None
        if name == "generative_plots.plot_pareto_front":
            # signature: (result: study-like, x: str, y: str) — needs the
            # generative study object, not a bare frame
            return None

        # -- md / docking / chem: artifact-driven, no frame path ---------------------
        return None
