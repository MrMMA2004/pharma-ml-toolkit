"""Tests for :mod:`pharmaml.chem.interpret` + its two viz twins, and
for :mod:`pharmaml.ml.interpret` (the round-187 InterpretML
absorption).

Round 100x-152 (GAP 2 of the ``Ml/D2.py`` audit): the upload's
interpretability block — Morgan-bit decoding (``decode_morgan_bit``),
the signed feature-target correlations (``analyze_correlations``'s
computation half) and the two figures that draw them
(``plot_signed_correlations`` / ``plot_correlation_heatmap``).

The suite pins:

* the decode record's contract on a REAL verified bit — caffeine's
  carbonyl bit 314 (radius 2, 2048 bits, connectivity invariants;
  hits ``(7, 1)`` and ``(10, 1)``, both decoding to the ``c=O``
  environment) — plus the FCFP flavour (bit 418) and aspirin's
  radius-0 hits (bit 650);
* the two pinned D2.py deviations: every hit decodes with ITS OWN
  radius (the original passed the fingerprint radius and discarded
  the hit's), and radius-0 hits are kept as single-atom environments
  (the original silently dropped them);
* bit-identity with the legacy ``GetMorganFingerprintAsBitVect``
  bitInfo route (both flavours) — the modern-generator port must
  reproduce the upload's own numbers;
* the correlation contract: signed, ``|r|``-descending, sign
  preserved, undefined (constant) columns dropped, non-numeric
  columns skipped with a warning, typed guards for every misuse;
* the figures: figure type, axis/tick labels, top-N truncation, the
  signed-bar colour convention (teal positive / red negative, the
  ``plot_permutation_importance`` twin), the annotation threshold and
  the guards.

The decode tests need RDKit (skipped as a class without it); the
correlation and plot tests run everywhere.

Round 187 appends the ml/interpret suite — the glassbox EBM and
partial dependence re-implemented offline from InterpretML 0.7.8 —
pinning: the quantile-binning contract (strictly increasing deduped
cuts, the ``max_bins - 2`` budget, lower-bound-inclusive value bins,
NaN -> the missing bin 0); the classifier on a separable synthetic
(accuracy + balanced predictions, exact ``eval_terms`` additivity,
seed determinism, early stopping via ``n_steps_``); the regressor on
``2*x0 - x1 + noise`` (r^2, the residual-mean polish); the FAST pair
screen picking the TRUE interacting pair of ``x0*x1``; the
postprocess (centered weighted means, zero-weight bins zeroed); the
term surfaces (importances, ``term_functions`` records); and the
partial-dependence/ICE record (percentile grid pins, the ICE matrix,
multiclass refusal, ``predict_fn`` overrides, the Doane density).
Every fit uses tiny boosting settings; the ONE default-settings model
runs with ``outer_bags=3, max_rounds=60``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from itertools import pairwise
from typing import ClassVar

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

from matplotlib.figure import Figure

from pharmaml.chem.interpret import (
    AtomEnvironment,
    MorganBitInfo,
    decode_morgan_bit,
    signed_feature_correlations,
)
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.interpret import (
    EbmBin,
    EBMClassifier,
    EBMRegressor,
    InterpretError,
    discretize,
    ebm_bins,
    partial_dependence,
)
from pharmaml.viz._compat import PALETTE
from pharmaml.viz.ml_plots import (
    plot_correlation_heatmap,
    plot_signed_correlations,
)

#: The doctest molecule — caffeine.
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)O2)C"
#: Caffeine's carbonyl bit (radius 2, 2048 bits, connectivity
#: invariants), verified live on RDKit 2026.09.1: the bit fires on the
#: two ``c=O`` environments, hits ``(7, 1)`` and ``(10, 1)``.
CAFFEINE_CARBONYL_BIT = 314
#: Caffeine's FCFP (feature-invariant) bit 418: one radius-2 hit at
#: atom 4, the purine-ring feature environment ``cnc(oc)c(c)n``.
CAFFEINE_FEATURE_BIT = 418
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
#: Aspirin's radius-0 bit (both hydroxyl-ish oxygen centers) — the
#: hit class the D2.py original silently dropped.
ASPIRIN_RADIUS0_BIT = 650


@contextmanager
def _pharmaml_logs_visible():
    """Re-enable propagation on the ``pharmaml`` logger for caplog.

    Same dance as ``test_ml_pipeline``: pytest's caplog handler lives
    on the true root logger, so records only reach it while
    propagation is on (restored on exit).
    """
    pharma_root = logging.getLogger("pharmaml")
    previous = pharma_root.propagate
    pharma_root.propagate = True
    try:
        yield
    finally:
        pharma_root.propagate = previous


@pytest.fixture()
def close_all():
    """Close every figure after each test (keeps the suite light)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ------------------------------------------------------------ decoding


class TestDecodeMorganBit:
    """decode_morgan_bit — the D2.py bit-to-chemistry decoder."""

    @pytest.fixture(autouse=True)
    def _rdkit(self):
        pytest.importorskip("rdkit")

    def test_real_bit_decodes_the_record_fields(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT)
        assert isinstance(info, MorganBitInfo)
        assert (info.bit, info.radius, info.n_bits, info.use_features) == (
            CAFFEINE_CARBONYL_BIT, 2, 2048, False,
        )
        assert info.smiles == CAFFEINE
        assert info.n_hits == len(info.hits) == 2
        assert all(isinstance(hit, AtomEnvironment) for hit in info.hits)

    def test_hit_fields_and_carbonyl_environment(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT)
        centers = [(hit.atom_index, hit.radius) for hit in info.hits]
        assert centers == [(7, 1), (10, 1)]
        # the first hit's radius-1 carbonyl environment: atoms 6+7
        assert sorted(info.hits[0].atom_indices) == [6, 7]
        assert info.hits[0].center_atom == "O"
        assert {hit.environment_smiles for hit in info.hits} == {"c=O"}
        assert {hit.environment_smarts for hit in info.hits} == {"[#6]=[#8]"}

    def test_hit_radius_is_the_hits_own_not_the_fingerprints(self) -> None:
        """The pinned D2.py bug fix: the original passed the FINGERPRINT
        radius to FindAtomEnvironmentOfRadiusN and discarded the hit's
        own radius — bit 314 fires at radius 1, so the original drew the
        (larger) radius-2 environment instead of the carbonyl pair."""
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT)
        assert info.radius == 2                       # the fingerprint radius
        assert {hit.radius for hit in info.hits} == {1}  # the HITS' radius
        # a radius-1 carbonyl environment is exactly two atoms
        assert all(len(hit.atom_indices) == 2 for hit in info.hits)

    def test_use_features_path_decodes_the_fcfp_flavour(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_FEATURE_BIT, use_features=True)
        assert info.use_features is True
        assert [(hit.atom_index, hit.radius) for hit in info.hits] == [(4, 2)]
        assert info.hits[0].environment_smiles == "cnc(oc)c(c)n"
        assert info.hits[0].center_atom == "C"
        assert len(info.hits[0].atom_indices) == 8

    def test_radius_zero_hits_kept_as_single_atom_environments(self) -> None:
        """The second pinned deviation: the D2.py original's
        ``submol.GetNumAtoms() > 0`` guard silently dropped radius-0
        hits (no bonds -> empty sub-molecule)."""
        info = decode_morgan_bit(ASPIRIN, ASPIRIN_RADIUS0_BIT)
        assert [(hit.atom_index, hit.radius) for hit in info.hits] == [(2, 0), (11, 0)]
        for hit in info.hits:
            assert hit.atom_indices == (hit.atom_index,)
            assert hit.environment_smiles == "O"
            assert hit.center_atom == "O"

    def test_radius_zero_fingerprint_decodes_single_atoms(self) -> None:
        """A radius-0 FINGERPRINT: every hit is a single-atom environment
        (ethanol's CH2 bit 80, verified live)."""
        info = decode_morgan_bit("CCO", 80, radius=0, with_drawing=False)
        assert info.radius == 0
        assert [(hit.atom_index, hit.radius) for hit in info.hits] == [(1, 0)]
        assert info.hits[0].atom_indices == (1,)
        assert info.hits[0].environment_smiles == "C"

    def test_drawing_is_the_rdkit_bit_legend_svg(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT)
        assert isinstance(info.drawing, str)
        assert info.drawing.startswith("<?xml")
        assert "<svg" in info.drawing

    def test_with_drawing_false_skips_the_svg(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT, with_drawing=False)
        assert info.drawing is None
        assert info.hits  # the chemistry still decodes

    def test_as_dict_round_trips_the_record(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT, with_drawing=False)
        payload = info.as_dict()
        assert sorted(payload) == [
            "bit", "drawing", "hits", "n_bits", "radius", "smiles", "use_features",
        ]
        assert payload["bit"] == CAFFEINE_CARBONYL_BIT
        assert payload["drawing"] is None
        first = payload["hits"][0]
        assert sorted(first) == [
            "atom_index", "atom_indices", "center_atom", "environment_smarts",
            "environment_smiles", "radius",
        ]
        assert first["atom_indices"] == [6, 7]
        assert first["environment_smiles"] == "c=O"

    def test_str_is_the_pinned_one_liner(self) -> None:
        info = decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT, with_drawing=False)
        assert str(info) == (
            "Morgan bit 314 (radius 2, 2048 bits, connectivity invariants): "
            "2 hit(s) on CN1C=NC2=C1C(=O)N(C(=O)O2)C — atom 7 @r1, atom 10 @r1"
        )
        flavour = decode_morgan_bit(
            CAFFEINE, CAFFEINE_FEATURE_BIT, use_features=True, with_drawing=False)
        assert "feature invariants" in str(flavour)

    def test_bit_not_set_raises_typed_naming_everything(self) -> None:
        with pytest.raises(DataValidationError, match="is not set by") as caught:
            decode_morgan_bit(CAFFEINE, 1)
        message = str(caught.value)
        assert "bit 1" in message.replace("Morgan bit", "bit")
        assert CAFFEINE in message
        assert "2048" in message and "radius 2" in message

    def test_unparseable_smiles_raises_typed(self) -> None:
        with pytest.raises(DataValidationError, match="does not parse"):
            decode_morgan_bit("not_a_smiles", CAFFEINE_CARBONYL_BIT)

    def test_bit_out_of_range_raises_typed(self) -> None:
        with pytest.raises(DataValidationError, match="out of range"):
            decode_morgan_bit("CCO", 2048)

    @pytest.mark.parametrize(("argument", "message"), [
        ("radius", "radius must be an integer >= 0"),
        ("n_bits", "n_bits must be an integer >= 1"),
        ("bit", "bit must be an integer"),
    ])
    def test_non_integer_arguments_raise_typed(
        self, argument: str, message: str
    ) -> None:
        kwargs = {"radius": 2, "n_bits": 2048, "bit": CAFFEINE_CARBONYL_BIT}
        kwargs[argument] = "2"  # type: ignore[assignment]
        with pytest.raises(DataValidationError, match=message):
            decode_morgan_bit(CAFFEINE, **kwargs)  # type: ignore[arg-type]

    def test_negative_radius_raises_typed(self) -> None:
        with pytest.raises(DataValidationError, match="radius must be >= 0"):
            decode_morgan_bit(CAFFEINE, CAFFEINE_CARBONYL_BIT, radius=-1)

    def test_agrees_with_the_legacy_bitinfo_route(self) -> None:
        """The modern-generator port must reproduce D2.py's own legacy
        call bit-for-bit (connectivity flavour)."""
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles(ASPIRIN)
        legacy_info: dict = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, bitInfo=legacy_info)
        for bit, hits in legacy_info.items():
            info = decode_morgan_bit(ASPIRIN, int(bit))
            assert [(hit.atom_index, hit.radius) for hit in info.hits] == [
                (int(atom), int(radius)) for atom, radius in hits
            ]

    def test_use_features_agrees_with_the_legacy_fcfp_route(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles(CAFFEINE)
        legacy_info: dict = {}
        rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=2048, bitInfo=legacy_info, useFeatures=True)
        assert CAFFEINE_FEATURE_BIT in legacy_info
        info = decode_morgan_bit(
            CAFFEINE, CAFFEINE_FEATURE_BIT, use_features=True)
        assert [(hit.atom_index, hit.radius) for hit in info.hits] == [
            (int(atom), int(radius))
            for atom, radius in legacy_info[CAFFEINE_FEATURE_BIT]
        ]


# -------------------------------------------------------- correlations


class TestSignedFeatureCorrelations:
    """signed_feature_correlations — the computation half of the bundle."""

    def test_pearson_signed_and_sorted_by_absolute_value(self) -> None:
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        frame = pd.DataFrame({
            "perfect": y,
            "inverse": [-value for value in y],
            "weak": [1.0, -1.0, 1.0, -1.0, 1.0],
        })
        corr = signed_feature_correlations(frame, y)
        assert isinstance(corr, pd.Series)
        # |r| 1.0, 1.0, 0.0 — ties keep column order (stable sort); the
        # alternating column is exactly uncorrelated with a ramp
        assert corr.index.tolist() == ["perfect", "inverse", "weak"]
        assert float(corr["perfect"]) == pytest.approx(1.0)
        assert float(corr["inverse"]) == pytest.approx(-1.0)
        assert float(corr["weak"]) == pytest.approx(0.0, abs=1e-12)

    def test_spearman_sees_monotone_curvature_as_perfect(self) -> None:
        y = [1.0, 2.0, 3.0, 4.0]
        cube = pd.DataFrame({"cube": [value ** 3 for value in y]})
        spearman = signed_feature_correlations(cube, y, method="spearman")
        pearson = signed_feature_correlations(cube, y)
        assert float(spearman.iloc[0]) == pytest.approx(1.0)
        assert float(pearson.iloc[0]) == pytest.approx(0.9514, abs=1e-4)
        assert float(pearson.iloc[0]) < 1.0

    def test_series_name_is_the_method(self) -> None:
        y = [1.0, 2.0, 3.0]
        frame = pd.DataFrame({"a": y})
        assert signed_feature_correlations(frame, y).name == "pearson"
        assert signed_feature_correlations(
            frame, y, method="spearman").name == "spearman"

    def test_non_numeric_columns_skipped_with_warning(self, caplog) -> None:
        y = [1.0, 2.0, 3.0, 4.0]
        frame = pd.DataFrame({"a": y, "smiles": ["CCO"] * 4})
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.chem.interpret"
        ):
            corr = signed_feature_correlations(frame, y)
        assert corr.index.tolist() == ["a"]
        assert "Skipping 1 non-numeric column(s)" in caplog.text
        assert "smiles" in caplog.text

    def test_complex_columns_skipped_not_half_correlated(self, caplog) -> None:
        """(100x-153) pandas counts complex dtype as "numeric", and the
        dtype=float cast would silently discard the imaginary part (an
        unread ComplexWarning) -- a half-honest correlation.  Complex
        columns must ride the non-numeric skip path instead."""
        y = [1.0, 2.0, 3.0, 4.0]
        frame = pd.DataFrame(
            {
                "a": y,
                "cplx": np.array([1 + 2j, 3 + 1j, 2 + 2j, 1 + 1j]),
            }
        )
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.chem.interpret"
        ):
            corr = signed_feature_correlations(frame, y)
        assert corr.index.tolist() == ["a"]
        assert "cplx" not in corr.index
        assert "Skipping 1 non-numeric column(s)" in caplog.text
        assert "cplx" in caplog.text

    def test_constant_column_dropped_not_zeroed(self, caplog) -> None:
        """The D2.py original appended a fake 0.0 on failure; a constant
        column's undefined correlation must be DROPPED, never zeroed."""
        y = [1.0, 2.0, 3.0, 4.0]
        frame = pd.DataFrame({"a": y, "flat": [7.0] * 4})
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.chem.interpret"
        ):
            corr = signed_feature_correlations(frame, y)
        assert "flat" not in corr.index
        assert float(corr["a"]) == pytest.approx(1.0)
        assert "Dropping 1 column(s)" in caplog.text
        assert "flat" in caplog.text

    def test_constant_extreme_scale_column_dropped_never_fake_zero(
        self, caplog
    ) -> None:
        """(100x-154) the audit's RED: the naive np.corrcoef path
        reported a FAKE r = 0.0 for a constant 1e300-scale column (the
        mean's ULP-scale rounding error makes the "zero" deviations
        ~1e284; their squares overflow to inf; 0/inf = 0 -- while
        scipy honestly returns NaN).  The offset-then-scale recipe's
        first-element re-centring is exact, so the column collapses to
        exact zeros at ANY scale and drops honestly."""
        y = [float(i) for i in range(38)]
        frame = pd.DataFrame(
            {
                "big": [1e300] * 38,
                "tiny": [1e-300] * 38,
                "plain": [5.0] * 38,
                "ok": y,
            }
        )
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.chem.interpret"
        ):
            corr = signed_feature_correlations(frame, y)
        assert "big" not in corr.index  # the fake-zero regression
        assert "tiny" not in corr.index
        assert "plain" not in corr.index
        assert float(corr["ok"]) == pytest.approx(1.0)
        assert "Dropping 3 column(s)" in caplog.text
        assert "extreme scales" in caplog.text  # the honest wording

    def test_extreme_scale_varying_columns_kept_matching_scipy(self) -> None:
        """(100x-154) the audit's A1/A2: 1e300-scale and 1e-300-scale
        VARYING columns were silently DROPPED by the naive path's
        internal overflow/underflow (NaN masquerading as "undefined")
        where scipy reports a defined r.  The recipe keeps them, and
        the values match scipy."""
        from scipy.stats import pearsonr

        rng = np.random.default_rng(154)
        y = np.linspace(-1.0, 1.0, 64)
        huge = 1e300 * (1.0 + 0.5 * rng.random(64))
        tiny = 1e-300 * (1.0 + rng.random(64))
        frame = pd.DataFrame({"huge": huge, "tiny": tiny})
        corr = signed_feature_correlations(frame, y)
        assert set(corr.index) == {"huge", "tiny"}  # neither dropped
        for column in ("huge", "tiny"):
            expected = float(pearsonr(frame[column].to_numpy(), y).statistic)
            assert float(corr[column]) == pytest.approx(expected, abs=1e-12)

    def test_huge_offset_stays_close_to_analytic_truth(self) -> None:
        """(100x-154) a 1e15-offset column whose jitter is an exact
        linear function of the target's jitter (analytic r = 1.0): no
        float64 path recovers the truth exactly (the input itself is
        quantized at construction, ULP(1e15) = 0.125 vs jitter ~1),
        but the recipe must stay within 2e-3 of the truth -- the
        naive path's mean-rounding error is what pushed BOTH the old
        desktop value and scipy itself further out."""
        rng = np.random.default_rng(154)
        small = rng.normal(0.0, 1.0, 64)
        frame = pd.DataFrame({"off": 1e15 + small})
        target = 1e15 + 2.0 * small + 3.0
        corr = signed_feature_correlations(frame, target)
        assert float(corr["off"]) == pytest.approx(1.0, abs=2e-3)

    def test_nan_cells_are_handled_pairwise(self) -> None:
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        frame = pd.DataFrame({"gap": [10.0, 20.0, 30.0, float("nan"), 50.0]})
        corr = signed_feature_correlations(frame, y)
        # the four pairwise-complete rows are still a perfect line
        assert float(corr["gap"]) == pytest.approx(1.0)

    def test_length_mismatch_raises_typed(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(DataValidationError, match="4 row\\(s\\) but target"):
            signed_feature_correlations(frame, [1.0, 2.0])

    def test_non_finite_target_raises_typed(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(DataValidationError, match="non-finite"):
            signed_feature_correlations(frame, [1.0, float("nan"), 3.0])

    def test_fewer_than_two_rows_raises_typed(self) -> None:
        frame = pd.DataFrame({"a": [1.0]})
        with pytest.raises(DataValidationError, match="at least 2 rows"):
            signed_feature_correlations(frame, [1.0])

    def test_no_numeric_columns_raises_typed(self) -> None:
        frame = pd.DataFrame({"smiles": ["CCO", "CCC"]})
        with pytest.raises(DataValidationError, match="no numeric columns"):
            signed_feature_correlations(frame, [1.0, 2.0])

    def test_every_column_undefined_raises_typed(self) -> None:
        frame = pd.DataFrame({"flat": [7.0] * 4, "also_flat": [1.0] * 4})
        with pytest.raises(DataValidationError, match="no correlation could"):
            signed_feature_correlations(frame, [1.0, 2.0, 3.0, 4.0])

    def test_unknown_method_raises_typed(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(DataValidationError, match="method must be one of"):
            signed_feature_correlations(frame, [1.0, 2.0, 3.0], method="kendall")

    def test_non_frame_input_raises_typed(self) -> None:
        with pytest.raises(DataValidationError, match="pandas.DataFrame"):
            signed_feature_correlations([[1.0, 2.0]], [1.0, 2.0])  # type: ignore[arg-type]


# ------------------------------------------------------------ figures


class TestPlotSignedCorrelations:
    """plot_signed_correlations — the top-N signed bar chart."""

    CORR = pd.Series({"logp": 0.9, "mw": -0.8, "tpsa": 0.3})

    def test_returns_figure_with_metric_label(self, close_all) -> None:
        fig = plot_signed_correlations(self.CORR)
        assert isinstance(fig, Figure)
        chart = fig.axes[0]
        assert chart.get_xlabel() == "Pearson r"
        assert chart.get_title() == "Top 3 features by |Pearson r|"

    def test_yticklabels_put_the_strongest_at_the_top(self, close_all) -> None:
        fig = plot_signed_correlations(self.CORR)
        labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        assert labels == ["tpsa", "mw", "logp"]  # bottom -> top

    def test_metric_label_spearman(self, close_all) -> None:
        fig = plot_signed_correlations(
            self.CORR, metric_label="Spearman ρ", title="Mono")
        assert fig.axes[0].get_xlabel() == "Spearman ρ"
        assert fig.axes[0].get_title() == "Mono"

    def test_top_truncation_takes_the_head(self, close_all) -> None:
        many = pd.Series({f"f{i:02d}": 0.5 - 0.03 * i for i in range(15)})
        assert len(plot_signed_correlations(many, top=10).axes[0].patches) == 10
        assert len(plot_signed_correlations(many, top=None).axes[0].patches) == 15
        assert len(plot_signed_correlations(many, top=99).axes[0].patches) == 15

    def test_signed_colors_mirror_permutation_convention(self, close_all) -> None:
        from matplotlib.colors import to_rgb

        fig = plot_signed_correlations(self.CORR)
        patches = fig.axes[0].patches  # bottom -> top: tpsa, mw, logp
        assert patches[0].get_facecolor()[:3] == to_rgb(PALETTE[0])  # +0.3 teal
        assert patches[1].get_facecolor()[:3] == to_rgb(PALETTE[3])  # -0.8 red
        assert patches[2].get_facecolor()[:3] == to_rgb(PALETTE[0])  # +0.9 teal

    def test_every_bar_carries_its_signed_value(self, close_all) -> None:
        fig = plot_signed_correlations(self.CORR)
        texts = [text.get_text() for text in fig.axes[0].texts]
        assert "+0.9" in "".join(texts)
        assert any(text.strip().startswith("-0.8") for text in texts)

    def test_ax_parameter_draws_onto_the_callers_axes(self, close_all) -> None:
        import matplotlib.pyplot as plt

        own_fig, own_ax = plt.subplots()
        returned = plot_signed_correlations(self.CORR, ax=own_ax)
        assert returned is own_fig
        assert len(own_fig.axes) == 1
        assert len(own_ax.patches) == 3

    def test_empty_series_raises_typed(self, close_all) -> None:
        with pytest.raises(DataValidationError, match="empty Series"):
            plot_signed_correlations(pd.Series(dtype=float))

    def test_all_nan_series_raises_typed(self, close_all) -> None:
        nan_series = pd.Series({"a": float("nan"), "b": float("nan")})
        with pytest.raises(DataValidationError, match="every correlation is NaN"):
            plot_signed_correlations(nan_series)

    def test_nan_entries_dropped_with_warning(self, close_all, caplog) -> None:
        corr = pd.Series({"a": 1.0, "b": float("nan"), "c": -0.5})
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.viz.ml_plots"
        ):
            fig = plot_signed_correlations(corr)
        assert len(fig.axes[0].patches) == 2
        assert "Dropping 1 NaN" in caplog.text

    def test_non_series_input_raises_typed(self, close_all) -> None:
        with pytest.raises(DataValidationError, match="pandas.Series"):
            plot_signed_correlations({"logp": 0.9})  # type: ignore[arg-type]

    def test_bad_top_raises_typed(self, close_all) -> None:
        with pytest.raises(DataValidationError, match="top_n must be a positive"):
            plot_signed_correlations(self.CORR, top=0)

    def test_end_to_end_from_signed_feature_correlations(self, close_all) -> None:
        """The D2.py story: correlations -> signed bars, strongest on top."""
        rng = np.random.default_rng(42)
        y = rng.uniform(2.0, 8.0, 60)
        frame = pd.DataFrame({
            "desc_a": 0.8 * y + rng.normal(0.0, 0.2, 60),
            "desc_b": -0.6 * y + rng.normal(0.0, 0.5, 60),
            "desc_c": rng.normal(0.0, 1.0, 60),
            "desc_d": rng.normal(0.0, 1.0, 60),
        })
        corr = signed_feature_correlations(frame, y)
        fig = plot_signed_correlations(corr, title="Top features by |r|")
        labels = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        assert labels[-1] == corr.index[0]  # strongest at the top
        assert len(fig.axes[0].patches) == 4


class TestPlotCorrelationHeatmap:
    """plot_correlation_heatmap — the seaborn-free matrix figure."""

    @pytest.fixture()
    def frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            rng.uniform(size=(60, 4)), columns=["a", "b", "c", "d"])

    def test_returns_figure_with_tick_parity(self, close_all, frame) -> None:
        fig = plot_correlation_heatmap(frame)
        assert isinstance(fig, Figure)
        chart = fig.axes[0]
        xticks = [tick.get_text() for tick in chart.get_xticklabels()]
        yticks = [tick.get_text() for tick in chart.get_yticklabels()]
        assert xticks == yticks == ["a", "b", "c", "d"]

    def test_columns_argument_fixes_the_order(self, close_all, frame) -> None:
        fig = plot_correlation_heatmap(frame, columns=["d", "a"])
        yticks = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        assert yticks == ["d", "a"]

    def test_small_matrix_is_annotated(self, close_all, frame) -> None:
        fig = plot_correlation_heatmap(frame)
        assert len(fig.axes[0].texts) == 16  # 4x4 cells

    def test_large_matrix_is_not_annotated(self, close_all) -> None:
        rng = np.random.default_rng(7)
        wide = pd.DataFrame(
            rng.uniform(size=(30, 14)), columns=[f"f{i:02d}" for i in range(14)])
        fig = plot_correlation_heatmap(wide)
        assert len(fig.axes[0].texts) == 0
        assert "unannotated" in fig.axes[0].get_title()
        assert "14 features" in fig.axes[0].get_title()

    def test_colour_scale_is_pinned_to_the_correlation_range(
        self, close_all, frame
    ) -> None:
        fig = plot_correlation_heatmap(frame)
        chart = fig.axes[0]
        assert chart.images[0].get_clim() == (-1.0, 1.0)
        # heatmap axes + the colour-bar axes
        assert len(fig.axes) == 2

    def test_spearman_method_runs_and_is_labelled(self, close_all) -> None:
        rng = np.random.default_rng(42)
        y = np.arange(40.0)
        frame = pd.DataFrame({
            "linear": y,
            "curved": y ** 3,
            "noise": rng.normal(0.0, 1.0, 40),
        })
        fig = plot_correlation_heatmap(frame, method="spearman")
        assert "(spearman)" in fig.axes[0].get_title()
        assert fig.axes[1].get_ylabel() == "spearman r"

    def test_constant_column_renders_nan_cells_unannotated(
        self, close_all
    ) -> None:
        frame = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 5.0, 4.0, 5.0],
            "flat": [7.0] * 5,
        })
        fig = plot_correlation_heatmap(frame)
        # 3x3 = 9 cells; the flat row/column contributes 5 NaN cells
        # (a-flat, flat-a, b-flat, flat-b, flat-flat) left unannotated
        assert len(fig.axes[0].texts) == 4

    def test_auto_mode_skips_non_numeric_with_warning(
        self, close_all, caplog
    ) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({
            "a": rng.uniform(size=30),
            "b": rng.uniform(size=30),
            "smiles": ["CCO"] * 30,
        })
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.viz.ml_plots"
        ):
            fig = plot_correlation_heatmap(frame)
        yticks = [tick.get_text() for tick in fig.axes[0].get_yticklabels()]
        assert yticks == ["a", "b"]
        assert "Skipping 1 non-numeric column(s)" in caplog.text

    def test_missing_column_raises_typed(self, close_all, frame) -> None:
        with pytest.raises(DataValidationError, match="Missing required column"):
            plot_correlation_heatmap(frame, columns=["a", "missing"])

    def test_explicit_non_numeric_column_raises_typed(self, close_all) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "smiles": ["CCO", "CCC"]})
        with pytest.raises(DataValidationError, match="are not numeric"):
            plot_correlation_heatmap(frame, columns=["a", "smiles"])

    def test_too_few_columns_raises_typed(self, close_all, frame) -> None:
        with pytest.raises(DataValidationError, match="at least 2 numeric columns"):
            plot_correlation_heatmap(frame, columns=["a"])

    def test_all_constant_input_raises_typed(self, close_all) -> None:
        frame = pd.DataFrame({"flat": [7.0] * 4, "also": [1.0] * 4})
        with pytest.raises(DataValidationError, match="every correlation is"):
            plot_correlation_heatmap(frame)

    def test_unknown_method_raises_typed(self, close_all, frame) -> None:
        with pytest.raises(DataValidationError, match="method must be"):
            plot_correlation_heatmap(frame, method="kendall")

    def test_non_frame_input_raises_typed(self, close_all) -> None:
        with pytest.raises(DataValidationError, match="pandas.DataFrame"):
            plot_correlation_heatmap([[1.0, 2.0], [3.0, 4.0]])  # type: ignore[arg-type]

    def test_custom_title(self, close_all, frame) -> None:
        fig = plot_correlation_heatmap(frame, title="Top 10 + target")
        assert fig.axes[0].get_title() == "Top 10 + target"

    def test_the_d2_bundle_end_to_end(self, close_all) -> None:
        """The upload's analyze_correlations story: correlations ->
        top-N signed bars -> top-features + target heatmap."""
        rng = np.random.default_rng(42)
        y = rng.uniform(2.0, 8.0, 80)
        frame = pd.DataFrame({
            "desc_a": 0.8 * y + rng.normal(0.0, 0.2, 80),
            "desc_b": -0.6 * y + rng.normal(0.0, 0.5, 80),
            "desc_c": rng.normal(0.0, 1.0, 80),
            "desc_d": rng.normal(0.0, 1.0, 80),
        })
        corr = signed_feature_correlations(frame, y)
        bar_fig = plot_signed_correlations(corr)
        assert [tick.get_text() for tick in bar_fig.axes[0].get_yticklabels()][-1] \
            == corr.index[0]

        top_columns = list(corr.index[:3]) + ["pEC50"]
        heat_frame = frame[list(corr.index[:3])].copy()
        heat_frame["pEC50"] = y
        heat_fig = plot_correlation_heatmap(heat_frame, columns=top_columns)
        yticks = [tick.get_text() for tick in heat_fig.axes[0].get_yticklabels()]
        assert yticks == top_columns
        assert len(heat_fig.axes[0].texts) == 16  # 4x4 annotated cells


# ------------------------------------------------------------ ml/interpret
#
# Round 187: the InterpretML 0.7.8 glassbox absorption.  Every fit uses
# tiny boosting settings (the ONE default-settings model is
# ``test_default_settings_small_model``).

#: The task-pinned tiny settings: separable synthetic, 300 rows.
FAST_SETTINGS: dict[str, object] = {
    "outer_bags": 2,
    "max_rounds": 40,
    "smoothing_rounds": 5,
    "interactions": 2,
    "early_stopping_rounds": 10,
    "random_state": 42,
}


def _separable_data(rows: int = 300, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """y = (x0 > 0.5) on uniform noise columns (both classes present)."""
    rng = np.random.default_rng(seed)
    X = rng.random((rows, 3))
    y = (X[:, 0] > 0.5).astype(int)
    return X, y


@pytest.fixture(scope="module")
def separable_model() -> tuple[EBMClassifier, np.ndarray, np.ndarray]:
    """One shared tiny fit of the separable synthetic."""
    X, y = _separable_data()
    model = EBMClassifier(**FAST_SETTINGS).fit(X, y)  # type: ignore[arg-type]
    return model, X, y


class TestEbmBins:
    """ebm_bins/discretize -- the EBM quantile-binning contract."""

    def test_cuts_strictly_increasing_within_budget(self) -> None:
        rng = np.random.default_rng(3)
        column = rng.random(500)
        for max_bins in (16, 64, 1024):
            record = ebm_bins(column, max_bins=max_bins)
            assert isinstance(record, EbmBin)
            cuts = record.cuts
            assert all(a < b for a, b in pairwise(cuts))  # strictly increasing
            assert len(cuts) <= max_bins - 2  # the budget pin
            assert len(cuts) >= 1  # continuous data earns cuts
            assert float(column.min()) <= min(cuts) and max(cuts) <= float(column.max())

    def test_repeated_quantiles_dedup(self) -> None:
        # Two clusters + a lone value: the quantiles collapse onto few
        # distinct values, and a constant column dedups to no cuts at all.
        column = np.array([0.0] * 50 + [10.0] * 49 + [11.0])
        record = ebm_bins(column, max_bins=32)
        assert len(record.cuts) <= 3
        assert all(a < b for a, b in pairwise(record.cuts))
        assert ebm_bins(np.full(40, 7.5), max_bins=64).cuts == ()

    def test_lower_bound_inclusive_pinned(self) -> None:
        record = ebm_bins(np.array([0.0, 1.0, 2.0, 3.0]), max_bins=6)
        assert record.cuts == (1.0, 2.0)
        # bin(x) = searchsorted(cuts, x, "right") + 1: below c1 -> 1,
        # c1 <= x < c2 -> 2, x >= c2 -> 3.
        values = np.array([-5.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0])
        assert discretize(values, record).tolist() == [1, 1, 1, 2, 2, 3, 3, 3]

    def test_nan_maps_to_missing_bin_zero(self) -> None:
        record = ebm_bins(np.array([0.0, 1.0, 2.0, 3.0]), max_bins=6)
        values = np.array([np.nan, 0.5, np.nan, 3.0])
        assert discretize(values, record).tolist() == [0, 1, 0, 3]
        # +/-inf is lumped with the missing bin (the documented choice).
        assert discretize(np.array([np.inf, -np.inf]), record).tolist() == [0, 0]

    def test_all_nan_column_and_bad_arguments_refused(self) -> None:
        with pytest.raises(InterpretError):
            ebm_bins(np.full(10, np.nan))
        with pytest.raises(InterpretError):
            ebm_bins(np.arange(10.0), max_bins=3)
        with pytest.raises(InterpretError):
            ebm_bins(np.arange(10.0), min_samples_bin=0)

    def test_min_samples_bin_merges_thin_bins(self) -> None:
        column = np.array([0.0] * 4 + [5.0] * 4 + [7.0])
        # cuts before merging are (0.0, 5.0); the empty x < 0 bin merges
        # away, then the 4-row [0, 5) bin survives min_samples_bin=4.
        assert ebm_bins(column, max_bins=8, min_samples_bin=4).cuts == (5.0,)
        # min_samples_bin=5 also absorbs the 4-row bin: one value bin left.
        assert ebm_bins(column, max_bins=8, min_samples_bin=5).cuts == ()


class TestEBMClassifier:
    """EBMClassifier -- the glassbox fit on a separable synthetic."""

    def test_separable_accuracy_and_balanced_predictions(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, X, y = separable_model
        predictions = model.predict(X)
        accuracy = float((predictions == y).mean())
        assert accuracy >= 0.95
        positives = int(predictions.sum())
        assert 0 < positives < len(y)  # balanced: both classes predicted
        assert set(np.unique(predictions)) == {0, 1}

    def test_predict_proba_rows_sum_to_one(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, X, _ = separable_model
        probs = model.predict_proba(X)
        assert probs.shape == (X.shape[0], 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        # sigma to machine precision (the module's numerically stable
        # branch differs from the naive formula only in the last ulp).
        expected = 1.0 / (1.0 + np.exp(-model.decision_function(X)))
        assert np.allclose(probs[:, 1], expected, rtol=1e-12, atol=1e-15)
        assert model.classes_.tolist() == [0, 1]

    def test_eval_terms_exact_additivity(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, X, _ = separable_model
        terms = model.eval_terms(X)
        assert terms.shape == (X.shape[0], len(model.terms_))
        assert np.array_equal(
            terms.sum(axis=1) + model.intercept_, model.decision_function(X)
        )

    def test_same_random_state_is_bit_deterministic(self) -> None:
        X, y = _separable_data(rows=150)
        settings = {
            "outer_bags": 2, "max_rounds": 15, "smoothing_rounds": 2,
            "interactions": 1, "early_stopping_rounds": 5, "random_state": 0,
        }
        first = EBMClassifier(**settings).fit(X, y).decision_function(X)  # type: ignore[arg-type]
        second = EBMClassifier(**settings).fit(X, y).decision_function(X)  # type: ignore[arg-type]
        other = EBMClassifier(**{**settings, "random_state": 43}).fit(X, y).decision_function(X)  # type: ignore[arg-type]
        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)

    def test_early_stopping_stops_before_the_step_cap(self) -> None:
        # One strong feature with 8% label noise: the validation logloss
        # bottoms out and rises, so the circular-buffer check fires well
        # before the max_rounds * n_terms cap.
        rng = np.random.default_rng(3)
        X = rng.random((300, 2))
        y = (X[:, 0] > 0.5).astype(int)
        flip = rng.random(300) < 0.08  # one mask, flipped once
        y[flip] = 1 - y[flip]
        model = EBMClassifier(
            outer_bags=2, learning_rate=0.4, max_rounds=50, smoothing_rounds=2,
            interactions=0, early_stopping_rounds=5, random_state=3,
        ).fit(X, y)
        cap = 2 * 50 * 2  # outer_bags * max_rounds * n_features
        assert 0 < model.n_steps_ < cap

    def test_invalid_targets_refused(self) -> None:
        X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.2, 0.8]])
        for bad_y in (
            np.array([0, 1, 2, 0]),      # multiclass -> typed binary refusal
            np.array([1, 2, 1, 2]),      # non-{0, 1} labels
            np.array([1, 1, 1, 1]),      # single class
            np.array([0.0, 1.0, np.nan, 0.0]),  # NaN target
            np.array([0, 1]),            # length mismatch
        ):
            with pytest.raises(InterpretError):
                EBMClassifier(outer_bags=2, max_rounds=5).fit(X, bad_y)

    def test_parameter_and_state_guards(self) -> None:
        X, y = _separable_data(rows=40)
        for kwargs in (
            {"max_leaves": 3},
            {"inner_bags": 2},
            {"interactions": "abc"},
            {"validation_size": 0.0},
            {"early_stopping_rounds": 0},
            {"exclude": [0, 1, 2]},
        ):
            with pytest.raises(InterpretError):
                EBMClassifier(**kwargs).fit(X, y)  # type: ignore[arg-type]
        unfitted = EBMClassifier()
        with pytest.raises(InterpretError):
            unfitted.predict(X)
        fitted = EBMClassifier(
            outer_bags=2, max_rounds=5, smoothing_rounds=1, interactions=0,
            early_stopping_rounds=2, random_state=0,
        ).fit(X, y)
        with pytest.raises(InterpretError):
            fitted.predict(X[:, :2])

    def test_nan_rows_train_and_predict(self) -> None:
        rng = np.random.default_rng(4)
        X = rng.random((200, 3))
        y = (X[:, 0] > 0.5).astype(int)
        mask = rng.random((200, 3)) < 0.1
        mask[:, 0] = False  # keep the signal column complete
        X[mask] = np.nan
        model = EBMClassifier(
            outer_bags=2, max_rounds=30, smoothing_rounds=3, interactions=0,
            early_stopping_rounds=5, random_state=2,
        ).fit(X, y)
        assert np.isfinite(model.decision_function(X)).all()
        assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)
        assert float((model.predict(X) == y).mean()) >= 0.9

    def test_default_settings_small_model(self) -> None:
        # The ONE near-default fit (only outer_bags/max_rounds shrunk):
        # everything else is the upstream 0.7.8 default table.
        X, y = _separable_data(rows=120, seed=8)
        model = EBMClassifier(outer_bags=3, max_rounds=60).fit(X, y)
        assert float((model.predict(X) == y).mean()) >= 0.9
        assert (model.term_importances() >= 0).all()
        assert model.feature_importances_.shape == (3,)
        assert model.n_features_in_ == 3

    def test_exclude_and_cyclic_progress(self) -> None:
        X, y = _separable_data(rows=150)
        excluded = EBMClassifier(
            outer_bags=2, max_rounds=15, smoothing_rounds=2, interactions=0,
            early_stopping_rounds=5, exclude=[1], random_state=0,
        ).fit(X, y)
        assert [term.features for term in excluded.terms_] == [(0,), (2,)]
        assert excluded.feature_importances_[1] == 0.0
        cyclic = EBMClassifier(
            outer_bags=2, max_rounds=15, smoothing_rounds=2, interactions=0,
            early_stopping_rounds=5, cyclic_progress=True, random_state=0,
        ).fit(X, y)
        assert float((cyclic.predict(X) == y).mean()) >= 0.9


class TestEBMRegressor:
    """EBMRegressor -- the RMSE glassbox on a linear surface."""

    @pytest.fixture(scope="class")
    def regression_model(self) -> tuple[EBMRegressor, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(5)
        X = rng.random((300, 3))
        y = 2.0 * X[:, 0] - X[:, 1] + 0.1 * rng.standard_normal(300)
        model = EBMRegressor(
            outer_bags=2, max_rounds=80, smoothing_rounds=5, interactions=0,
            early_stopping_rounds=10, random_state=1,
        ).fit(X, y)
        return model, X, y

    def test_linear_surface_r2_and_rmse(
        self,
        regression_model: tuple[EBMRegressor, np.ndarray, np.ndarray],
    ) -> None:
        model, X, y = regression_model
        predictions = model.predict(X)
        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        assert 1.0 - ss_res / ss_tot >= 0.9
        assert float(np.sqrt(np.mean((y - predictions) ** 2))) < 0.15

    def test_residual_mean_polish(
        self,
        regression_model: tuple[EBMRegressor, np.ndarray, np.ndarray],
    ) -> None:
        model, X, y = regression_model
        assert abs(float(np.mean(y - model.predict(X)))) < 1e-9

    def test_additivity_and_decision_function_match(
        self,
        regression_model: tuple[EBMRegressor, np.ndarray, np.ndarray],
    ) -> None:
        model, X, _ = regression_model
        terms = model.eval_terms(X)
        assert np.array_equal(
            terms.sum(axis=1) + model.intercept_, model.decision_function(X)
        )
        assert np.array_equal(model.predict(X), model.decision_function(X))

    def test_objective_guard(self) -> None:
        with pytest.raises(InterpretError):
            EBMRegressor(objective="mae")


class TestFastPairSelection:
    """The FAST screen -- the true interacting pair wins."""

    @pytest.fixture(scope="class")
    def pair_model(self) -> tuple[EBMClassifier, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(9)
        X = rng.random((300, 3))  # x0, x1 carry the signal; x2 is noise
        y = (X[:, 0] * X[:, 1] > 0.25).astype(int)
        model = EBMClassifier(
            outer_bags=2, max_rounds=30, smoothing_rounds=5, interactions=1,
            early_stopping_rounds=10, random_state=42,
        ).fit(X, y)
        return model, X, y

    def test_true_interacting_pair_selected(
        self, pair_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, X, y = pair_model
        pairs = [term.features for term in model.terms_ if len(term.features) == 2]
        assert pairs == [(0, 1)]
        assert float((model.predict(X) == y).mean()) >= 0.8

    def test_pair_term_record_structure(
        self, pair_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, _, _ = pair_model
        pair = model.terms_[-1]
        assert pair.features == (0, 1)
        assert pair.tensor.ndim == 2
        assert pair.tensor.shape == pair.bin_weights.shape == pair.std.shape
        record = model.term_functions()[-1]
        assert record.kind == "pair"
        assert record.scores.ndim == 2
        assert [len(c) for c in record.bin_centers] == list(record.scores.shape)
        assert record.scores_std.shape == record.scores.shape


class TestEbmPostprocess:
    """Centering, zero-weight bins -- the EBM postprocess contract."""

    def test_term_weighted_means_are_zero(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, _, _ = separable_model
        for term in model.terms_:
            mean = float(np.average(term.tensor, weights=term.bin_weights))
            assert abs(mean) < 1e-9

    def test_zero_weight_bins_are_zeroed(self) -> None:
        rng = np.random.default_rng(8)
        # A two-cluster column: the bins below the first cluster and the
        # missing bin (no NaNs anywhere) carry zero weight.
        X = np.column_stack([
            np.concatenate([np.zeros(100), np.full(100, 100.0)]),
            rng.random(200),
        ])
        y = (X[:, 0] > 50.0).astype(int)
        model = EBMClassifier(
            outer_bags=2, max_rounds=15, smoothing_rounds=2, interactions=1,
            early_stopping_rounds=5, random_state=0,
        ).fit(X, y)
        for term in model.terms_:
            empty = term.bin_weights == 0
            assert empty.any()
            assert float(np.abs(term.tensor[empty]).max()) == 0.0


class TestTermSurfaces:
    """term_importances / feature_importances_ / term_functions."""

    def test_importances_shapes_and_ordering(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, _, _ = separable_model
        importances = model.term_importances()
        assert importances.shape == (len(model.terms_),)
        assert (importances >= 0).all()
        # The signal feature dominates the noise features.
        mains = importances[:3]
        assert mains[0] > mains[1] and mains[0] > mains[2]
        feature_wise = model.feature_importances_
        assert feature_wise.shape == (3,)
        # Pairs are excluded from feature_importances_ by design.
        assert abs(float(feature_wise.sum()) - float(mains.sum())) < 1e-12

    def test_term_functions_records(
        self, separable_model: tuple[EBMClassifier, np.ndarray, np.ndarray]
    ) -> None:
        model, _, _ = separable_model
        records = model.term_functions()
        assert len(records) == len(model.terms_)
        for record in records:
            assert record.kind in ("main", "pair")
            assert len(record.bin_centers) == len(record.features)
            assert record.scores.shape == record.scores_std.shape
            assert record.scores.shape == record.weights.shape
            assert (record.scores_std >= 0).all()
            if record.kind == "main":
                assert record.scores.ndim == 1
                assert len(record.bin_centers[0]) == len(record.scores)
            else:
                assert record.scores.ndim == 2


class _Line:
    """A tiny linear model for the partial-dependence pins."""

    def __init__(self, coef: np.ndarray) -> None:
        self.coef = coef

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef


class TestPartialDependence:
    """partial_dependence -- the interpret.blackbox sweep, offline."""

    def test_percentile_grid_pinned_and_unique_short_grid(self) -> None:
        column = np.arange(100.0)
        X = np.column_stack([column, column * 2.0])
        record = partial_dependence(_Line(np.array([1.0, 0.0])), X, feature=0, num_points=5)
        assert record.grid.tolist() == pytest.approx([0.0, 24.75, 49.5, 74.25, 99.0])
        # Fewer unique values than num_points -> the sorted uniques.
        short = np.array([[3.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0]])
        record_short = partial_dependence(
            _Line(np.array([1.0, 0.0])), short, feature=0, num_points=10
        )
        assert record_short.grid.tolist() == [1.0, 2.0, 3.0]

    def test_monotone_model_gives_monotone_mean(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.random((80, 2))
        record = partial_dependence(
            _Line(np.array([2.0, -1.0])), X, feature=0, num_points=8
        )
        assert np.all(np.diff(record.mean) >= 0)
        assert record.mean.shape == record.grid.shape

    def test_ice_matrix_mean_and_bounds(self) -> None:
        rng = np.random.default_rng(6)
        X = rng.random((30, 3))
        record = partial_dependence(
            _Line(np.array([1.0, 2.0, 0.5])), X, feature=1, num_points=6
        )
        assert record.ice.shape == (30, record.grid.size)
        assert np.array_equal(record.mean, record.ice.mean(axis=0))
        assert np.array_equal(record.std, record.ice.std(axis=0))
        assert record.bounds.shape == (2, record.grid.size)
        assert np.array_equal(record.bounds[0], record.mean - record.std)
        assert np.array_equal(record.bounds[1], record.mean + record.std)

    def test_multiclass_predict_proba_refused(self) -> None:
        class _Multi:
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                return np.tile([0.2, 0.3, 0.5], (X.shape[0], 1))

        X = np.zeros((4, 2))
        with pytest.raises(InterpretError, match="binary-only"):
            partial_dependence(_Multi(), X, feature=0)

    def test_predict_fn_override_respected(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        record = partial_dependence(
            _Line(np.array([2.0, -1.0])),
            X,
            feature=0,
            num_points=3,
            predict_fn=lambda A: A @ np.array([3.0, -1.0]),
        )
        # grid [0, 1, 2]: 3*g - mean(col 1 of X) = 3*g - 1 -- and NOT the
        # model's own 2*g - 1, which is what a ignored override would give.
        assert record.mean.tolist() == pytest.approx([-1.0, 2.0, 5.0])
        assert record.mean.tolist() != pytest.approx([-1.0, 1.0, 3.0])

    def test_density_shapes(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.random((120, 2))
        record = partial_dependence(_Line(np.array([1.0, 0.0])), X, feature=0)
        assert int(record.density_counts.sum()) == 120
        assert record.density_edges.size == record.density_counts.size + 1

    def test_feature_resolution_and_guards(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        class _Named:
            feature_names_: ClassVar[tuple[str, ...]] = ("a", "b")

            def predict(self, X: np.ndarray) -> np.ndarray:
                return X[:, 1]

        record = partial_dependence(_Named(), X, feature="b", num_points=3)
        assert record.feature == 1
        assert record.feature_name == "b"
        with pytest.raises(InterpretError):
            partial_dependence(_Named(), X, feature="zz")
        with pytest.raises(InterpretError):
            partial_dependence(_Line(np.array([1.0, 0.0])), X, feature=5)
        with pytest.raises(InterpretError):
            partial_dependence(_Line(np.array([1.0])), X, feature=0, num_points=1)
        with pytest.raises(InterpretError):
            partial_dependence(_Line(np.array([1.0, 0.0])), X[:, 0], feature=0)

    def test_on_a_fitted_ebm_proba_column(self) -> None:
        # The default resolution path: predict_proba(X)[:, 1].
        X, y = _separable_data(rows=150, seed=13)
        model = EBMClassifier(
            outer_bags=2, max_rounds=15, smoothing_rounds=2, interactions=0,
            early_stopping_rounds=5, random_state=0,
        ).fit(X, y)
        record = partial_dependence(model, X, feature=0, num_points=8)
        assert record.feature == 0
        assert float(X[:, 0].min()) <= float(record.grid.min())
        assert float(record.grid.max()) <= float(X[:, 0].max())
        assert np.isfinite(record.ice).all()
        assert record.ice.shape == (150, record.grid.size)
