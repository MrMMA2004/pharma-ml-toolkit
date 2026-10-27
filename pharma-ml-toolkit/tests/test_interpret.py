"""Tests for :mod:`pharmaml.chem.interpret` and its two viz twins.

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
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

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
