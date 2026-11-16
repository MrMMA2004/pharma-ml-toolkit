"""The focused headless suite for ``pharmaml.ml.drugresponse``.

FEATURES.md's Response AI row carried a long-standing B note — the
module's only pins were the GUI page tests
(``tests/ui/test_responseai_page.py``, which exercise the module
through Qt workers). This suite is the headless twin: every public
surface of the module pinned directly, no Qt import, no page in the
loop. The seeded recipes are the page's own (seed 42, the 21-point 4PL
demo, the 12x4 screen with the 15% NaN mask) so the numbers cross-check
against the GUI path; the contract pins (bounds, NaN semantics,
determinism, fold grouping) hold for any caller — the web twin's bridge
included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.ml.drugresponse import (
    DrugResponseModel,
    InformativeRegression,
    MultiOmicsFusion,
    ResponseVAE,
    auc_log_dose,
    cross_validate_response,
    dose_response_curve,
    fit_hill_curve,
    response_metrics,
)
from pharmaml.ml.stats import rank_correlations


def _seeded_screen(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The page's ``_seeded_screen`` verbatim (12x4, 15% NaN mask)."""
    rng = np.random.default_rng(seed)
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


def _seeded_4pl(seed: int = 42, noise: float = 0.01):
    rng = np.random.default_rng(seed)
    doses = np.logspace(-3, 3, 21)
    clean = np.asarray(
        dose_response_curve(doses, 1.2, 0.8, top=1.0, bottom=0.05), dtype=float
    )
    return doses, clean + rng.normal(0.0, noise, doses.size), clean


# -- the Hill layer --------------------------------------------------------


class TestHillCurve:
    def test_demo_fit_recovers_the_generating_truth(self) -> None:
        doses, response, _ = _seeded_4pl()
        fit = fit_hill_curve(doses, response)
        assert fit.converged
        assert abs(fit.parameters["hill_coef"] - 1.2) < 0.1
        assert abs(fit.parameters["ic50"] - 0.8) < 0.1
        assert abs(fit.parameters["top"] - 1.0) < 0.1
        assert abs(fit.parameters["bottom"] - 0.05) < 0.05
        assert fit.r_squared > 0.99
        assert fit.n_points == 21

    def test_fit_is_deterministic_per_seed(self) -> None:
        first = fit_hill_curve(*_seeded_4pl()[:2])
        second = fit_hill_curve(*_seeded_4pl()[:2])
        assert first.parameters == second.parameters
        assert first.r_squared == second.r_squared

    def test_auc_agrees_between_model_and_closed_form(self) -> None:
        doses, response, clean = _seeded_4pl()
        fit = fit_hill_curve(doses, response)
        closed = auc_log_dose(doses, clean)
        assert abs(fit.fitted_auc - closed) < 0.05

    def test_curve_convention_top_is_low_dose_plateau(self) -> None:
        curve = np.asarray(
            dose_response_curve([0.01, 100.0], 1.0, 1.0, top=0.9, bottom=0.1),
            dtype=float,
        )
        assert curve[0] == pytest.approx(0.9, abs=0.02)
        assert curve[1] == pytest.approx(0.1, abs=0.02)

    def test_predict_evaluates_the_fitted_curve(self) -> None:
        doses, response, _ = _seeded_4pl()
        fit = fit_hill_curve(doses, response)
        mid = float(fit.predict(fit.parameters["ic50"]))
        lo = float(fit.predict(1e-3))
        hi = float(fit.predict(1e3))
        assert lo == pytest.approx(fit.parameters["top"], abs=0.05)
        assert hi == pytest.approx(fit.parameters["bottom"], abs=0.05)
        assert min(lo, hi) - 0.05 < mid < max(lo, hi) + 0.05

    @pytest.mark.parametrize(
        "doses,response,fragment",
        [
            ([1.0, 2.0], [0.5], "equal length"),
            ([-1.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.1] * 6, "positive"),
            ([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], [0.1] * 6, "finite"),
        ],
    )
    def test_fit_rejects_bad_input_honestly(self, doses, response, fragment) -> None:
        with pytest.raises(DataValidationError, match=fragment):
            fit_hill_curve(doses, response)


# -- the evaluation layer ---------------------------------------------------


class TestResponseMetrics:
    def test_seeded_pair_scores_high(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0.0, 1.0, 60)
        y_pred = np.clip(y_true + rng.normal(0.0, 0.08, 60), 0.0, 1.0)
        m = response_metrics(y_true, y_pred)
        assert m["n"] == 60
        assert m["pearson"] > 0.9
        assert m["spearman"] > 0.9
        assert 0.0 <= m["rmse"] < 0.2
        assert 0.0 <= m["mae"] < 0.2

    def test_nan_pairs_drop_before_everything(self) -> None:
        y_true = np.array([0.1, np.nan, 0.3, 0.4, np.nan, 0.6])
        y_pred = np.array([0.2, 0.25, 0.35, np.nan, 0.5, 0.65])
        m = response_metrics(y_true, y_pred)
        assert m["n"] == 3
        assert np.isfinite(m["pearson"])

    def test_perfect_prediction_scores_exactly(self) -> None:
        y = np.linspace(0.0, 1.0, 50)
        m = response_metrics(y, y)
        assert m["pearson"] == pytest.approx(1.0)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert m["mae"] == pytest.approx(0.0, abs=1e-12)

    def test_rank_trio_on_the_same_pair(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.uniform(0.0, 10.0, 48)
        y = 0.7 * x + rng.normal(0.0, 1.5, 48)
        rc = rank_correlations(x, y)
        assert rc.n == 48
        assert rc.pearson > 0.8
        assert rc.spearman > 0.8
        assert 0.0 < rc.kendall <= 1.0

    def test_rank_refuses_nan_input(self) -> None:
        with pytest.raises(DataValidationError, match="finite"):
            rank_correlations([0.1, np.nan, 0.3], [0.2, 0.3, 0.4])


# -- the response model -----------------------------------------------------


class TestDrugResponseModel:
    def test_seeded_screen_has_the_documented_shape_and_mask(self) -> None:
        X, table = _seeded_screen()
        assert X.shape == (12, 8)
        assert table.shape == (12, 4)
        assert list(table.columns) == ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
        observed = int(np.isfinite(table.to_numpy(dtype=float)).sum())
        assert 30 <= observed <= 48  # 15% mask on 48 pairs

    def test_training_is_deterministic_per_seed(self) -> None:
        X, table = _seeded_screen()
        first = DrugResponseModel(n_factors=2, epochs=60, seed=7).fit(X, table)
        second = DrugResponseModel(n_factors=2, epochs=60, seed=7).fit(X, table)
        assert first.train_losses == second.train_losses
        assert first.pearson_r == second.pearson_r

    def test_fit_reports_observed_pairs_and_finite_pearson(self) -> None:
        X, table = _seeded_screen()
        history = DrugResponseModel(n_factors=2, epochs=120).fit(X, table)
        observed = int(np.isfinite(table.to_numpy(dtype=float)).sum())
        assert history.n_observed == observed
        assert history.n_epochs == 120
        assert len(history.train_losses) == 120
        assert -1.0 <= history.pearson_r <= 1.0
        assert np.isfinite(history.final_loss)

    def test_loss_improves_over_training(self) -> None:
        X, table = _seeded_screen()
        history = DrugResponseModel(n_factors=2, epochs=120).fit(X, table)
        assert history.final_loss < history.train_losses[0]

    def test_predict_answers_the_screen_cells(self) -> None:
        X, table = _seeded_screen()
        model = DrugResponseModel(n_factors=2, epochs=60)
        model.fit(X, table)
        head = model.predict(["CELL_00", "CELL_07"])
        assert list(head.index) == ["CELL_00", "CELL_07"]
        assert list(head.columns) == list(table.columns)
        values = head.to_numpy(dtype=float)
        assert values.shape == (2, 4)
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()

    def test_predict_refuses_unknown_cells(self) -> None:
        X, table = _seeded_screen()
        model = DrugResponseModel(n_factors=2, epochs=20)
        model.fit(X, table)
        with pytest.raises(DataValidationError, match="unknown cell line"):
            model.predict(["CELL_99"])


class TestCrossValidateResponse:
    def test_folds_split_cell_lines_never_pairs(self) -> None:
        X, table = _seeded_screen()
        report = cross_validate_response(
            lambda: DrugResponseModel(n_factors=2, epochs=60),
            X,
            table,
            n_folds=4,
            seed=42,
        )
        assert report.n_folds == 4
        assert len(report.folds) == 4
        per_fold = [fold.n_test_cells for fold in report.folds]
        assert per_fold == [3, 3, 3, 3]
        # every cell line appears as a test cell exactly once
        assert sum(per_fold) == 12
        observed_total = sum(fold.n_observed for fold in report.folds)
        screen_observed = int(np.isfinite(table.to_numpy(dtype=float)).sum())
        assert observed_total == screen_observed

    def test_summary_is_the_across_fold_mean(self) -> None:
        X, table = _seeded_screen()
        report = cross_validate_response(
            lambda: DrugResponseModel(n_factors=2, epochs=60),
            X,
            table,
            n_folds=4,
            seed=42,
        )
        mean_of_folds = sum(fold.metrics["pearson"] for fold in report.folds) / 4
        assert report.summary["pearson"] == pytest.approx(mean_of_folds)

    def test_deterministic_per_seed(self) -> None:
        X, table = _seeded_screen()

        def factory() -> DrugResponseModel:
            return DrugResponseModel(n_factors=2, epochs=40)

        first = cross_validate_response(factory, X, table, n_folds=3, seed=11)
        second = cross_validate_response(factory, X, table, n_folds=3, seed=11)
        assert [f.metrics["pearson"] for f in first.folds] == [
            f.metrics["pearson"] for f in second.folds
        ]


# -- the deeper diagnostics (the page tabs the web card leaves desktop-side) --


class TestDeeperDiagnostics:
    def test_informative_regression_shrinks_uninformative_weights(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0.0, 1.0, (60, 3))
        y = 3.0 * X[:, 0] + 0.05 * X[:, 1] + 0.02 * X[:, 2] + rng.normal(0.0, 0.1, 60)
        # informative feature 0 (large prior variance) keeps its weight;
        # features 1-2 (prior variance 1) are shrunk 100x harder
        informative = InformativeRegression(lr=0.05, epochs=300)
        informative.fit(X, y, prior_variance=np.array([100.0, 1.0, 1.0]))
        plain = InformativeRegression(lr=0.05, epochs=300)
        plain.fit(X, y, prior_variance=np.array([100.0, 100.0, 100.0]))
        assert abs(informative.coef[0]) > 2.0
        assert abs(informative.coef[1]) < 0.5
        # the informative prior shrinks the noise features harder than
        # the equal (plain-ridge) prior does
        assert abs(informative.coef[1]) <= abs(plain.coef[1]) + 1e-9
        assert abs(informative.coef[2]) <= abs(plain.coef[2]) + 1e-9

    def test_response_vae_improves_its_elbo_bound(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(0.0, 1.0, (80, 6))
        vae = ResponseVAE(latent_dim=2, epochs=60, seed=42)
        history = vae.fit(X)
        assert len(history.elbo) == 60
        assert history.elbo[-1] > history.elbo[0]

    def test_multi_omics_fusion_equalises_block_variance(self) -> None:
        rng = np.random.default_rng(42)
        blocks = {
            "expression": pd.DataFrame(rng.normal(0.0, 1.0, (40, 20))),
            "mutation": pd.DataFrame(rng.normal(0.0, 3.0, (40, 4))),
        }
        fusion = MultiOmicsFusion()
        fused = fusion.fit_transform(blocks)
        assert fused.shape == (40, 24)
        # z-scored per block then standardised block-wise: each block's
        # per-column variance SUM is equalised, so one wide block cannot
        # dominate the concatenated vector by width alone
        var_expr = float(fused.iloc[:, :20].var(axis=0).sum())
        var_mut = float(fused.iloc[:, 20:].var(axis=0).sum())
        assert var_expr == pytest.approx(var_mut, rel=1e-9)
