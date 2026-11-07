"""Response AI Workbench: drug-response learning as a page (audit row 12).

The Studio's increment-11 page (GUI_AUDIT row 12 — the module
``pharmaml.ml.drugresponse``, absorbed from ``CSOgroup/CaDRReS``,
``Precily``'s multi-omics fusion discipline and ``DeepResponse``'s
variational autoencoder, given its FIRST GUI surface here: the
playground's Response AI tab predates the module and rides other
engines).  Eight tabs, each wired to the module's public APIs --
nothing on this page is fabricated:

- **Hill curve** -- ``fit_hill_curve`` + ``dose_response_curve`` +
  ``auc_log_dose`` (the pharmacology the screens actually measure):
  the seeded four-parameter log-logistic demo fit by the toolkit's
  own Levenberg-Marquardt engine, the fitted parameters against the
  generating truth, r², and the log-dose AUC both ways (model-based
  vs closed-form).
- **Response metrics** -- ``response_metrics`` (the evaluation layer
  every response paper reports: Pearson / Spearman / RMSE / MAE) on
  the seeded noisy pair, or two columns of the Data Explorer's loaded
  table.
- **Rank correlations** -- ``rank_correlations`` (Pearson, Spearman
  and Kendall tau-b in one call) on the seeded curved pair, or two
  explorer columns.
- **Response model** -- ``DrugResponseModel`` (the CaDRReS bilinear
  factorisation, training included): the seeded 12-cell-line x 4-drug
  screen with a 15% NaN mask, full-batch gradient descent over the
  observed pairs only, the honest in-sample pearson, and the
  predicted-response head for two cell lines.
- **Cross-validation** -- ``cross_validate_response`` over the same
  seeded screen, folds split *cell lines* so no cell line straddles a
  fold -- the mean pearson reported honestly even when it is ~0.
- **Informative regression** -- ``InformativeRegression`` (the
  cadrres-iw idea): per-feature prior variances shrink
  non-informative weights harder; the prior-shrunk coefficients
  against the generating truth.
- **Response VAE** -- ``ResponseVAE`` (the DeepResponse
  architecture: encoder → latent mu/logvar → a reparameterised sample
  → decoder with a response head, trained on the ELBO with the
  hand-derived backward pass): the bound improving.
- **Multi-omics fusion** -- ``MultiOmicsFusion`` (the Precily fusion
  discipline): z-score every block with statistics fitted at train
  time, then standardise block-wise so one wide block cannot dominate
  the concatenated vector by width alone -- the per-block audit with
  total variance equalised.

The metrics / rank tabs read the Data Explorer's loaded frame when one
exists (the same guard the playground's cards apply: at least 8 finite
aligned values) and otherwise fall back to the tab's own seeded demo
-- mirrored exactly.  All eight run in a
:class:`~ui.workers.job_worker.JobWorker`; they are single-shot engine
calls without checkpoint loops, so the progress channel reports
honestly but the cooperative Stop only lands for checkpoint-wired
jobs -- the same disclosure the Foundations and Best-of ML pages
carry.

Every default-input number below was engine-printed first by the
committed probe ``scratch/probe_response_218.py`` (the round-205+
house rule; round 219 re-seeded it per section so each tab is a
standalone seeded job, the bestofml probe's pattern).
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..theme_manager import ChartPalette, ThemeManager
from ..widgets.data_table import DataTableView, PandasTableModel
from ..widgets.plot_widget import PlotPanel
from ..widgets.progress_panel import ProgressPanel
from ..workers.job_worker import JobRunner


def _parse_float(text: str, name: str, *, minimum: float | None = None) -> float:
    """Parse one float with an honest, named error."""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not a number ({text!r})") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum} (got {value:g})") from None
    return value


def _parse_int(text: str, name: str, *, minimum: int, maximum: int) -> int:
    """Parse one int with an honest, named range error."""
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name}: not an integer ({text!r})") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum} (got {value})") from None
    return value


def _seeded_screen(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The seeded 12x4 screen (shared verbatim by the model and CV tabs).

    12 cell lines x 8 genomic features, 4 drugs with graded potency,
    responses generated through a sigmoid of a per-cell latent factor,
    then a 15% NaN mask -- the CaDRReS data contract in miniature:
    cell line x gene features and a response table with NaN where a
    pair was never screened.
    """
    n_cells, n_features = 12, 8
    X = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_cells, n_features)),
        index=[f"CELL_{i:02d}" for i in range(n_cells)],
        columns=[f"feat_{j}" for j in range(n_features)],
    )
    true_factor = rng.normal(0.0, 1.0, n_cells)
    drugs = ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
    potency = np.array([0.9, 0.7, 0.5, 0.3])
    table = pd.DataFrame(
        {
            d: 1.0 / (1.0 + np.exp(-(potency[k] * true_factor + rng.normal(0.0, 0.05, n_cells))))
            for k, d in enumerate(drugs)
        },
        index=X.index,
    )
    mask = rng.random(table.shape) < 0.15
    return X, table.mask(mask)


@dataclass
class HillOutcome:
    """What a Hill-fit job hands back (the fit + the demo data)."""

    fit: Any = None
    doses: Any = None
    response: Any = None
    clean_auc: float = 0.0
    noise: float = 0.01
    seconds: float = 0.0


@dataclass
class MetricsOutcome:
    """What a metrics job hands back (the report + the pair)."""

    metrics: dict[str, float] = field(default_factory=dict)
    y_true: Any = None
    y_pred: Any = None
    source: str = ""
    seconds: float = 0.0


@dataclass
class RankOutcome:
    """What a rank job hands back (the correlations + the pair)."""

    rc: Any = None
    x: Any = None
    y: Any = None
    source: str = ""
    seconds: float = 0.0


@dataclass
class ScreenOutcome:
    """What a response-model job hands back (history + predictions)."""

    history: Any = None
    pred: pd.DataFrame = field(default_factory=pd.DataFrame)
    observed: Any = None
    predicted: Any = None
    shape: tuple[int, int] = (0, 0)
    epochs: int = 0
    seconds: float = 0.0


@dataclass
class CVOutcome:
    """What a cross-validation job hands back (the fold ledger)."""

    report: Any = None
    seconds: float = 0.0


@dataclass
class InfoOutcome:
    """What an informative-regression job hands back."""

    losses: tuple[float, ...] = ()
    coef: Any = None
    intercept: float = 0.0
    seconds: float = 0.0


@dataclass
class VAEOutcome:
    """What a VAE job hands back (the training ledger)."""

    history: Any = None
    n_features: int = 0
    seconds: float = 0.0


@dataclass
class FusionOutcome:
    """What a fusion job hands back (the latent + the block audit)."""

    latent: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_stds: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


class ResponseAIWorkbenchPage(QWidget):
    """Fit Hill curves; score responses; train and cross-validate the
    CaDRReS factorisation; shrink with priors; train the VAE; fuse omics."""

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

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_hill_tab(), "Hill curve")
        self._tabs.addTab(self._build_metrics_tab(), "Response metrics")
        self._tabs.addTab(self._build_rank_tab(), "Rank correlations")
        self._tabs.addTab(self._build_model_tab(), "Response model")
        self._tabs.addTab(self._build_cv_tab(), "Cross-validation")
        self._tabs.addTab(self._build_info_tab(), "Informative regression")
        self._tabs.addTab(self._build_vae_tab(), "Response VAE")
        self._tabs.addTab(self._build_fusion_tab(), "Multi-omics fusion")
        lay.addWidget(self._tabs)

        self._progress = ProgressPanel()
        self._progress.attach(self._runner)
        self._runner.job_started.connect(self._on_started)
        self._runner.job_ended.connect(self._on_ended)
        lay.addWidget(self._progress)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._table = DataTableView()
        splitter.addWidget(self._table)
        self._plot = PlotPanel(title="Response AI")
        splitter.addWidget(self._plot)
        splitter.setSizes([560, 540])
        lay.addWidget(splitter, stretch=1)

        self._report = QLabel("No Response AI run yet.")
        self._report.setObjectName("studioStatus")
        self._report.setWordWrap(True)
        lay.addWidget(self._report)

    # -- tab construction ----------------------------------------------------
    def _build_hill_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._hill_noise = QLineEdit("0.01")
        form.addRow("noise:", self._hill_noise)
        self._hill_btn = QPushButton("Fit the Hill curve")
        self._hill_btn.setMinimumHeight(34)
        self._hill_btn.clicked.connect(self._start_hill)
        form.addRow("", self._hill_btn)
        hint = QLabel(
            "E(d) = bottom + (top - bottom)·d^h / (IC50^h + d^h) — the "
            "four-parameter log-logistic the screens actually measure.  "
            "Levenberg-Marquardt (the toolkit's own engine) fits the "
            "seeded 21-point demo; the table shows every fitted parameter "
            "against the generating truth, and the log-dose AUC both ways "
            "(model-based vs closed-form)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_metrics_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._met_x = QComboBox()
        self._met_x.addItem("synthetic noisy pair (seed 42)")
        form.addRow("observed column:", self._met_x)
        self._met_y = QComboBox()
        self._met_y.addItem("synthetic noisy pair (seed 42)")
        form.addRow("predicted column:", self._met_y)
        self._met_btn = QPushButton("Evaluate the pair")
        self._met_btn.setMinimumHeight(34)
        self._met_btn.clicked.connect(self._start_metrics)
        form.addRow("", self._met_btn)
        hint = QLabel(
            "The evaluation layer every response paper reports — Pearson, "
            "Spearman, RMSE and MAE of a predicted column against the "
            "observed one.  The pickers offer the Data Explorer's numeric "
            "columns when a table is loaded (at least 8 finite aligned "
            "values); the fallback is the seeded noisy pair."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_rank_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._rank_x = QComboBox()
        self._rank_x.addItem("synthetic curved pair (seed 42)")
        form.addRow("column X:", self._rank_x)
        self._rank_y = QComboBox()
        self._rank_y.addItem("synthetic curved pair (seed 42)")
        form.addRow("column Y:", self._rank_y)
        self._rank_btn = QPushButton("Correlate")
        self._rank_btn.setMinimumHeight(34)
        self._rank_btn.clicked.connect(self._start_rank)
        form.addRow("", self._rank_btn)
        hint = QLabel(
            "Pearson, Spearman and Kendall tau-b for two paired columns — "
            "linear, monotone and pairwise-agreement structure in one "
            "call.  The pickers offer the Data Explorer's numeric columns; "
            "the fallback is the seeded curved pair."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_model_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._drm_epochs = QLineEdit("120")
        form.addRow("epochs:", self._drm_epochs)
        self._drm_factors = QLineEdit("2")
        form.addRow("latent factors:", self._drm_factors)
        self._drm_btn = QPushButton("Train on the seeded screen")
        self._drm_btn.setMinimumHeight(34)
        self._drm_btn.clicked.connect(self._start_model)
        form.addRow("", self._drm_btn)
        hint = QLabel(
            "The CaDRReS bilinear factorisation in plain NumPy, training "
            "included: genomic features map to K latent gamma factors, "
            "each drug owns a weight vector over the layer, the predicted "
            "score is the sigmoid of the bilinear form — squared error "
            "over the OBSERVED pairs only (NaN carries no gradient), "
            "full-batch gradient descent, deterministic (lr 0.1, l2 "
            "0.001, weight seed 42).  The seeded 12x4 screen carries a "
            "15% NaN mask; the in-sample pearson is reported honestly."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_cv_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._cv_folds = QLineEdit("4")
        form.addRow("folds:", self._cv_folds)
        self._cv_btn = QPushButton("Cross-validate")
        self._cv_btn.setMinimumHeight(34)
        self._cv_btn.clicked.connect(self._start_cv)
        form.addRow("", self._cv_btn)
        hint = QLabel(
            "Folds split CELL LINES, never (cell line, drug) pairs — the "
            "leak the grouped design exists to prevent: a model that "
            "memorised a cell line's profile must not meet it in "
            "validation.  Each fold trains a fresh model on the other "
            "cell lines; the mean pearson is reported honestly even when "
            "it is ~0 (the toy screen barely carries signal)."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_info_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._info_btn = QPushButton("Fit with informative priors")
        self._info_btn.setMinimumHeight(34)
        self._info_btn.clicked.connect(self._start_info)
        form.addRow("", self._info_btn)
        hint = QLabel(
            "The cadrres-iw idea distilled: the penalty on feature j's "
            "weight is l2·β_j²/v_j where v_j is that feature's PRIOR "
            "VARIANCE — informative features (large v) keep their "
            "weights, non-informative ones shrink toward silence.  The "
            "seeded demo: generating truth [0.5, 1.5, -0.8] with priors "
            "[100, 4, 4] (lr 0.05, 300 epochs, l2 1.0); the table shows "
            "every shrunk coefficient against its truth."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_vae_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._vae_btn = QPushButton("Train the VAE")
        self._vae_btn.setMinimumHeight(34)
        self._vae_btn.clicked.connect(self._start_vae)
        form.addRow("", self._vae_btn)
        hint = QLabel(
            "The DeepResponse architecture: encoder → latent mu/logvar → "
            "a reparameterised sample → decoder with an optional response "
            "head, trained on the ELBO (reconstruction + KL + response "
            "MSE) with the backward pass hand-derived and verified "
            "against finite differences.  Seeded 60x10 features with a "
            "response vector (latent 4, hidden 16, lr 0.01, 80 epochs); "
            "the bound must improve — and does."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    def _build_fusion_tab(self) -> QWidget:
        tab = QWidget()
        form = QFormLayout(tab)
        self._fusion_btn = QPushButton("Fuse the blocks")
        self._fusion_btn.setMinimumHeight(34)
        self._fusion_btn.clicked.connect(self._start_fusion)
        form.addRow("", self._fusion_btn)
        hint = QLabel(
            "The fusion discipline every multi-omics network leans on: "
            "z-score every block column-wise with statistics fitted at "
            "train time, then divide the block by sqrt(n_features) so "
            "every block carries the same TOTAL variance — a wide "
            "expression block cannot drown a narrow mutation block by "
            "width alone.  The seeded demo: 30 samples x (12 expression "
            "+ 8 mutation) columns; the audit table shows the "
            "equalisation."
        )
        hint.setWordWrap(True)
        form.addRow("", hint)
        return tab

    # -- job plumbing -----------------------------------------------------------
    def _buttons(self) -> list[QPushButton]:
        return [
            self._hill_btn,
            self._met_btn,
            self._rank_btn,
            self._drm_btn,
            self._cv_btn,
            self._info_btn,
            self._vae_btn,
            self._fusion_btn,
        ]

    def _on_started(self, name: str) -> None:
        for btn in self._buttons():
            btn.setEnabled(False)
        self._report.setText(f"Running {name}…")

    def _on_ended(self) -> None:
        for btn in self._buttons():
            btn.setEnabled(True)

    def _busy_check(self) -> bool:
        if self._runner.busy:
            self._report.setText("A job is already running — Stop it first.")
            return True
        return False

    def _failed(self, message: str) -> None:
        self._report.setText(message.splitlines()[0] if message else "The job failed.")

    def _explorer_frame(self) -> pd.DataFrame | None:
        if self._data_explorer is not None and hasattr(self._data_explorer, "frame"):
            frame = self._data_explorer.frame()
            if frame is not None and not frame.empty:
                return frame
        return None

    def _refresh_column_pickers(self, numeric: list[str]) -> None:
        """(Re)fill the metrics / rank pickers with the explorer's numeric columns."""
        if not numeric:
            return
        for box, fallback in (
            (self._met_x, "synthetic noisy pair (seed 42)"),
            (self._met_y, "synthetic noisy pair (seed 42)"),
            (self._rank_x, "synthetic curved pair (seed 42)"),
            (self._rank_y, "synthetic curved pair (seed 42)"),
        ):
            box.clear()
            box.addItem(fallback)
            box.addItems(numeric)

    def _picked_columns(self, x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Two aligned finite columns off the explorer, or None to refuse."""
        frame = self._explorer_frame()
        if frame is None:
            return None
        names = [str(c) for c in frame.columns]
        if x_name not in names or y_name not in names:
            return None
        x = pd.to_numeric(frame[x_name], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(frame[y_name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 8:
            raise ValueError(
                f"columns {x_name!r} / {y_name!r} have only {int(mask.sum())} finite "
                "aligned rows — need at least 8"
            )
        return x[mask], y[mask], f"columns {x_name!r} / {y_name!r} of the loaded table ({int(mask.sum())} aligned rows)"

    # -- Hill curve ------------------------------------------------------------
    def _start_hill(self) -> None:
        try:
            noise = _parse_float(self._hill_noise.text(), "noise", minimum=0.0)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if noise > 1.0:
            self._report.setText("Input — noise must be <= 1")
            return
        if self._busy_check():
            return

        def job(progress) -> HillOutcome:
            from pharmaml.ml.drugresponse import auc_log_dose, dose_response_curve, fit_hill_curve

            started = _time.monotonic()
            progress(0.3, "the seeded 21-point 4PL demo")
            rng = np.random.default_rng(42)
            hill_true = {"hill_coef": 1.2, "ic50": 0.8, "top": 1.0, "bottom": 0.05}
            doses = np.logspace(-3, 3, 21)
            clean = np.asarray(
                dose_response_curve(
                    doses, hill_true["hill_coef"], hill_true["ic50"],
                    top=hill_true["top"], bottom=hill_true["bottom"],
                ),
                dtype=float,
            )
            response = clean + rng.normal(0.0, noise, doses.size)
            progress(0.7, "Levenberg-Marquardt fit")
            fit = fit_hill_curve(doses, response)
            clean_auc = auc_log_dose(doses, clean)
            progress(1.0, "done")
            return HillOutcome(
                fit=fit,
                doses=doses,
                response=response,
                clean_auc=clean_auc,
                noise=noise,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "Hill curve fit",
            job,
            on_finished=self._hill_done,
            on_failed=self._failed,
        )

    def _hill_done(self, outcome: object) -> None:
        assert isinstance(outcome, HillOutcome)
        fit = outcome.fit
        truth = {"hill_coef": 1.2, "ic50": 0.8, "top": 1.0, "bottom": 0.05}
        rows = [
            ("top", float(fit.parameters["top"]), truth["top"]),
            ("bottom", float(fit.parameters["bottom"]), truth["bottom"]),
            ("ic50", float(fit.parameters["ic50"]), truth["ic50"]),
            ("hill_coef", float(fit.parameters["hill_coef"]), truth["hill_coef"]),
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["parameter", "fitted", "true"]),
            note="the 4PL, fitted vs generating",
        )
        self._plot.clear()
        grid = np.logspace(-3.5, 3.5, 240)
        self._plot.line(np.log10(grid), np.asarray(fit.predict(grid), dtype=float), index=0)
        self._plot.scatter(
            np.log10(outcome.doses), outcome.response, label="measured", index=1
        )
        self._plot.set_labels(x="log10 dose", y="response", title="the fitted Hill curve")
        self._report.setText(
            f"seeded 4PL demo (seed 42) · noise {outcome.noise:g} · "
            f"r² {fit.r_squared:.6f} · "
            + ("converged" if fit.converged else "NOT converged")
            + f" · {fit.n_points} points · fitted AUC {fit.fitted_auc:.6f} vs "
            f"closed-form {outcome.clean_auc:.6f} (log-dose, {outcome.seconds:.1f}s)"
        )

    # -- response metrics ------------------------------------------------------------
    def _start_metrics(self) -> None:
        x_name = self._met_x.currentText()
        y_name = self._met_y.currentText()
        synthetic = x_name.startswith("synthetic") or y_name.startswith("synthetic")
        try:
            pair = None if synthetic else self._picked_columns(x_name, y_name)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> MetricsOutcome:
            from pharmaml.ml.drugresponse import response_metrics

            started = _time.monotonic()
            progress(0.4, "the pair to score")
            if pair is not None:
                y_true, y_pred, source = pair
            else:
                rng = np.random.default_rng(42)
                y_true = rng.uniform(0.0, 1.0, 60)
                y_pred = np.clip(y_true + rng.normal(0.0, 0.08, 60), 0.0, 1.0)
                source = "seeded noisy pair (seed 42)"
            progress(0.8, "Pearson / Spearman / RMSE / MAE")
            metrics = response_metrics(y_true, y_pred)
            progress(1.0, "done")
            return MetricsOutcome(
                metrics=dict(metrics),
                y_true=np.asarray(y_true, dtype=float),
                y_pred=np.asarray(y_pred, dtype=float),
                source=source,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "response metrics",
            job,
            on_finished=self._metrics_done,
            on_failed=self._failed,
        )

    def _metrics_done(self, outcome: object) -> None:
        assert isinstance(outcome, MetricsOutcome)
        m = outcome.metrics
        rows = [
            ("n", f"{int(m['n'])}"),
            ("pearson", f"{float(m['pearson']):.6f}"),
            ("spearman", f"{float(m['spearman']):.6f}"),
            ("rmse", f"{float(m['rmse']):.6f}"),
            ("mae", f"{float(m['mae']):.6f}"),
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["metric", "value"]),
            note="observed vs predicted",
        )
        self._plot.clear()
        self._plot.scatter(outcome.y_true, outcome.y_pred, label="pairs", index=0)
        lo = float(min(outcome.y_true.min(), outcome.y_pred.min()))
        hi = float(max(outcome.y_true.max(), outcome.y_pred.max()))
        self._plot.line(
            np.asarray([lo, hi], dtype=float), np.asarray([lo, hi], dtype=float),
            label="identity", index=1, style="dash",
        )
        self._plot.set_labels(x="observed", y="predicted", title="observed vs predicted")
        self._report.setText(
            f"{outcome.source} · n={int(m['n'])} · pearson {float(m['pearson']):.4f} · "
            f"spearman {float(m['spearman']):.4f} · rmse {float(m['rmse']):.4f} · "
            f"mae {float(m['mae']):.4f} ({outcome.seconds:.1f}s)"
        )

    # -- rank correlations ------------------------------------------------------------
    def _start_rank(self) -> None:
        x_name = self._rank_x.currentText()
        y_name = self._rank_y.currentText()
        synthetic = x_name.startswith("synthetic") or y_name.startswith("synthetic")
        try:
            pair = None if synthetic else self._picked_columns(x_name, y_name)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> RankOutcome:
            from pharmaml.ml.drugresponse import rank_correlations

            started = _time.monotonic()
            progress(0.4, "the pair to correlate")
            if pair is not None:
                x, y, source = pair
            else:
                rng = np.random.default_rng(42)
                x = rng.uniform(0.0, 10.0, 48)
                y = 0.7 * x + rng.normal(0.0, 1.5, 48)
                source = "seeded curved pair (seed 42)"
            progress(0.8, "Pearson / Spearman / Kendall")
            rc = rank_correlations(x, y)
            progress(1.0, "done")
            return RankOutcome(
                rc=rc,
                x=np.asarray(x, dtype=float),
                y=np.asarray(y, dtype=float),
                source=source,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "rank correlations",
            job,
            on_finished=self._rank_done,
            on_failed=self._failed,
        )

    def _rank_done(self, outcome: object) -> None:
        assert isinstance(outcome, RankOutcome)
        rc = outcome.rc
        rows = [
            ("n", f"{int(rc.n)}"),
            ("pearson", f"{float(rc.pearson):.6f}"),
            ("spearman", f"{float(rc.spearman):.6f}"),
            ("kendall", f"{float(rc.kendall):.6f}"),
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["statistic", "value"]),
            note="one call, three agreement scales",
        )
        self._plot.clear()
        self._plot.scatter(outcome.x, outcome.y, label="pairs", index=0)
        self._plot.set_labels(x="X", y="Y", title="the correlated pair")
        self._report.setText(
            f"{outcome.source} · n={int(rc.n)} · pearson {float(rc.pearson):.4f} · "
            f"spearman {float(rc.spearman):.4f} · kendall {float(rc.kendall):.4f} — "
            f"linear, monotone and pairwise agreement in one call ({outcome.seconds:.1f}s)"
        )

    # -- response model ------------------------------------------------------------
    def _start_model(self) -> None:
        try:
            epochs = _parse_int(self._drm_epochs.text(), "epochs", minimum=20, maximum=400)
            factors = _parse_int(self._drm_factors.text(), "latent factors", minimum=1, maximum=4)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> ScreenOutcome:
            from pharmaml.ml.drugresponse import DrugResponseModel

            started = _time.monotonic()
            progress(0.3, "the seeded 12x4 screen (15% NaN mask)")
            rng = np.random.default_rng(42)
            X, table = _seeded_screen(rng)
            progress(0.6, "full-batch gradient descent over the observed pairs")
            model = DrugResponseModel(n_factors=factors, lr=0.1, epochs=epochs, l2=0.001, seed=42)
            history = model.fit(X, table)
            progress(0.9, "the predicted-response head")
            pred = model.predict(["CELL_00", "CELL_07"])
            predicted_all = model.predict(list(table.index)).to_numpy(dtype=float)
            observed_all = table.to_numpy(dtype=float)
            mask = np.isfinite(observed_all)
            progress(1.0, "done")
            return ScreenOutcome(
                history=history,
                pred=pred,
                observed=observed_all[mask],
                predicted=predicted_all[mask],
                shape=(int(table.shape[0]), int(table.shape[1])),
                epochs=epochs,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "drug-response model",
            job,
            on_finished=self._model_done,
            on_failed=self._failed,
        )

    def _model_done(self, outcome: object) -> None:
        assert isinstance(outcome, ScreenOutcome)
        history = outcome.history
        self._table.bind_frame(outcome.pred, note="predicted response scores")
        self._plot.clear()
        self._plot.scatter(outcome.observed, outcome.predicted, label="observed pairs", index=0)
        lo = float(min(outcome.observed.min(), outcome.predicted.min()))
        hi = float(max(outcome.observed.max(), outcome.predicted.max()))
        self._plot.line(
            np.asarray([lo, hi], dtype=float), np.asarray([lo, hi], dtype=float),
            label="identity", index=1, style="dash",
        )
        self._plot.set_labels(
            x="observed", y="predicted",
            title=f"the {history.n_observed} observed pairs (in-sample)",
        )
        self._report.setText(
            f"seeded {outcome.shape[0]}x{outcome.shape[1]} screen (seed 42, 15% NaN mask) · "
            f"{history.n_observed} observed pairs · final loss {history.final_loss:.6f} "
            f"after {history.n_epochs} epochs · in-sample pearson {history.pearson_r:.6f} — "
            f"honestly modest, the toy screen is tiny ({outcome.seconds:.1f}s)"
        )

    # -- cross-validation ------------------------------------------------------------
    def _start_cv(self) -> None:
        try:
            folds = _parse_int(self._cv_folds.text(), "folds", minimum=2, maximum=8)
        except ValueError as exc:
            self._report.setText(f"Input — {exc}")
            return
        if self._busy_check():
            return

        def job(progress) -> CVOutcome:
            from pharmaml.ml.drugresponse import DrugResponseModel, cross_validate_response

            started = _time.monotonic()
            progress(0.3, "the same seeded screen")
            rng = np.random.default_rng(42)
            X, table = _seeded_screen(rng)
            progress(0.6, f"{folds} cell-line-grouped folds")
            report = cross_validate_response(
                lambda: DrugResponseModel(n_factors=2, lr=0.1, epochs=120, l2=0.001, seed=42),
                X, table, n_folds=folds, seed=42,
            )
            progress(1.0, "done")
            return CVOutcome(report=report, seconds=_time.monotonic() - started)

        self._runner.submit(
            "cross-validation",
            job,
            on_finished=self._cv_done,
            on_failed=self._failed,
        )

    def _cv_done(self, outcome: object) -> None:
        assert isinstance(outcome, CVOutcome)
        report = outcome.report
        rows = [
            (
                int(fold.fold),
                int(fold.n_test_cells),
                int(fold.n_observed),
                round(float(fold.metrics["pearson"]), 6),
                round(float(fold.metrics["rmse"]), 6),
            )
            for fold in report.folds
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["fold", "test cells", "observed", "pearson", "rmse"]),
            note="cell-line-grouped",
        )
        self._plot.clear()
        self._plot.bars(
            np.asarray([float(fold.metrics["pearson"]) for fold in report.folds], dtype=float),
            labels=[f"fold {fold.fold}" for fold in report.folds],
            index=0,
            width=0.5,
        )
        self._plot.set_labels(
            x="fold", y="pearson r", title=f"per-fold pearson over {report.n_folds} folds"
        )
        mean = report.summary["pearson"]
        self._report.setText(
            f"{report.n_folds} folds grouped by cell line (no cell straddles a fold) · "
            f"mean pearson {mean:.6f} · the honest read: the toy screen barely "
            f"carries signal ({outcome.seconds:.1f}s)"
        )

    # -- informative regression ------------------------------------------------------------
    def _start_info(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> InfoOutcome:
            from pharmaml.ml.drugresponse import InformativeRegression

            started = _time.monotonic()
            progress(0.4, "the seeded 40-row demo")
            rng = np.random.default_rng(42)
            Xr = np.column_stack(
                [np.ones(40), rng.normal(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)]
            )
            beta = np.array([0.5, 1.5, -0.8])
            yr = Xr @ beta + rng.normal(0.0, 0.1, 40)
            prior = np.array([100.0, 4.0, 4.0])  # informative: betas near +-1.5/0.8
            progress(0.8, "prior-shrunk gradient descent")
            info = InformativeRegression(lr=0.05, epochs=300, l2=1.0)
            losses = info.fit(Xr, yr, prior_variance=prior)
            progress(1.0, "done")
            return InfoOutcome(
                losses=tuple(float(v) for v in losses),
                coef=np.asarray(info.coef, dtype=float),
                intercept=float(info.intercept),
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "informative regression",
            job,
            on_finished=self._info_done,
            on_failed=self._failed,
        )

    def _info_done(self, outcome: object) -> None:
        assert isinstance(outcome, InfoOutcome)
        truth = [0.5, 1.5, -0.8]
        rows = [
            (term, round(float(outcome.coef[j]), 4), truth[j])
            for j, term in enumerate(("bias column (x0)", "x1", "x2"))
        ]
        rows.append(("model intercept", round(outcome.intercept, 4), 0.0))
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["term", "fitted", "generating"]),
            note="prior variance [100, 4, 4]",
        )
        self._plot.clear()
        self._plot.line(
            np.arange(1, len(outcome.losses) + 1, dtype=float),
            np.asarray(outcome.losses, dtype=float),
            index=0,
        )
        self._plot.set_labels(x="epoch", y="loss", title="the informative loss")
        coefs = ", ".join(f"{v:.3f}" for v in outcome.coef)
        self._report.setText(
            f"prior variance [100, 4, 4] per feature · coefs [{coefs}] vs generating "
            f"[0.5, 1.5, -0.8] · loss {outcome.losses[0]:.4f} → {outcome.losses[-1]:.4f} "
            f"over {len(outcome.losses)} epochs · intercept {outcome.intercept:.4f} "
            f"({outcome.seconds:.1f}s)"
        )

    # -- response VAE ------------------------------------------------------------
    def _start_vae(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> VAEOutcome:
            from pharmaml.ml.drugresponse import ResponseVAE

            started = _time.monotonic()
            progress(0.3, "the seeded 60x10 features + response")
            rng = np.random.default_rng(42)
            Xv = rng.normal(0.0, 1.0, (60, 10))
            yv = (Xv @ rng.normal(0.0, 0.5, 10)) + rng.normal(0.0, 0.1, 60)
            progress(0.7, "the ELBO (reconstruction + KL + response MSE)")
            vae = ResponseVAE(latent_dim=4, hidden=16, lr=0.01, epochs=80, seed=42)
            history = vae.fit(Xv, yv, seed=42)
            progress(1.0, "done")
            return VAEOutcome(history=history, n_features=10, seconds=_time.monotonic() - started)

        self._runner.submit(
            "response VAE",
            job,
            on_finished=self._vae_done,
            on_failed=self._failed,
        )

    def _vae_done(self, outcome: object) -> None:
        assert isinstance(outcome, VAEOutcome)
        history = outcome.history
        rows = [
            ("elbo first", f"{float(history.elbo[0]):.6f}"),
            ("elbo last", f"{float(history.elbo[-1]):.6f}"),
            ("reconstruction (last)", f"{float(history.reconstruction[-1]):.6f}"),
            ("kl divergence (last)", f"{float(history.kl_divergence[-1]):.6f}"),
            ("response loss (last)", f"{float(history.response_loss[-1]):.6f}"),
            ("epochs", f"{int(history.n_epochs)}"),
        ]
        self._table.bind_frame(
            pd.DataFrame(rows, columns=["ledger", "value"]),
            note="per-sample, per-epoch",
        )
        self._plot.clear()
        self._plot.line(
            np.arange(1, len(history.elbo) + 1, dtype=float),
            np.asarray(history.elbo, dtype=float),
            label="ELBO",
            index=0,
        )
        self._plot.set_labels(
            x="epoch", y="ELBO (negative loss, up = learning)",
            title="the evidence lower bound",
        )
        self._report.setText(
            f"60x{outcome.n_features} seeded features with a response head · ELBO "
            f"{float(history.elbo[0]):.6f} → {float(history.elbo[-1]):.6f} (the bound "
            f"improving) · reconstruction {float(history.reconstruction[-1]):.6f} · KL "
            f"{float(history.kl_divergence[-1]):.6f} · response MSE "
            f"{float(history.response_loss[-1]):.6f} · {int(history.n_epochs)} epochs "
            f"({outcome.seconds:.1f}s)"
        )

    # -- multi-omics fusion ------------------------------------------------------------
    def _start_fusion(self) -> None:
        if self._busy_check():
            return

        def job(progress) -> FusionOutcome:
            from pharmaml.ml.drugresponse import MultiOmicsFusion

            started = _time.monotonic()
            progress(0.4, "the seeded expression + mutation blocks")
            rng = np.random.default_rng(42)
            blocks = {
                "expr": pd.DataFrame(
                    rng.normal(0.0, 1.0, (30, 12)), index=[f"C{i:02d}" for i in range(30)]
                ),
                "mut": pd.DataFrame(
                    rng.integers(0, 2, (30, 8)).astype(float),
                    index=[f"C{i:02d}" for i in range(30)],
                ),
            }
            progress(0.8, "fit-time z-scores + block scaling")
            fusion = MultiOmicsFusion(equal_block_variance=True)
            latent = fusion.fit_transform(blocks)
            audit = fusion.block_report()
            raw_stds = {
                name: float(fusion.block_stds_[name].mean()) for name in fusion.block_names_
            }
            progress(1.0, "done")
            return FusionOutcome(
                latent=latent,
                audit=audit,
                raw_stds=raw_stds,
                seconds=_time.monotonic() - started,
            )

        self._runner.submit(
            "multi-omics fusion",
            job,
            on_finished=self._fusion_done,
            on_failed=self._failed,
        )

    def _fusion_done(self, outcome: object) -> None:
        assert isinstance(outcome, FusionOutcome)
        audit = outcome.audit.copy()
        audit["mean column std (raw)"] = [
            round(outcome.raw_stds.get(str(block), float("nan")), 6) for block in audit["block"]
        ]
        self._table.bind_frame(audit, note="the per-block audit")
        self._plot.clear()
        self._plot.bars(
            np.asarray([float(v) for v in audit["total_variance"]], dtype=float),
            labels=[str(b) for b in audit["block"]],
            index=0,
            width=0.5,
        )
        self._plot.set_labels(
            x="block", y="total column variance", title="the equalisation (both 1.0)"
        )
        names = ", ".join(f"{b} ({int(n)} cols)" for b, n in zip(audit["block"], audit["n_features"]))
        self._report.setText(
            f"{len(audit)} blocks fused → latent {outcome.latent.shape[0]}x"
            f"{outcome.latent.shape[1]} ({names}) · per-block total variance equalised "
            f"to 1.0 — the wide expression block cannot drown the narrow mutation "
            f"block by width alone ({outcome.seconds:.1f}s)"
        )

    # -- shared plumbing ------------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        self._plot.apply_palette(palette)

    def report_text(self) -> str:
        return self._report.text()

    def model_frame(self) -> pd.DataFrame | None:
        """The frame currently bound to the result table (or None)."""
        model = self._table.model()
        return model.frame if isinstance(model, PandasTableModel) else None
