"""Tests for :mod:`pharmaml.generative.scoring_transforms` (round 100x-164).

The REINVENT4 transform DSL port: the eight transforms with their
exact formulas, the registry/factory name normalisation, the
NaN-aware weighted aggregators, and the three filter/penalty
objectives (CustomAlerts, MatchingSubstructure, GroupCount) plus the
transformable raw components.

The suite pins:

* every transform's formula against hand-computed values (the stable
  sigmoid's k rescaling, the double sigmoid's centre maximum, the
  rectified decay, the step edges, the value mapping's NaN path);
* the hard-step degenerate case ``high == low`` (REINVENT4's own
  branch) and the reverse sigmoid's mirror;
* the registry: every class is registered under its REINVENT4 class
  name (``doublesigmoid`` etc.), dashes and underscores normalise, and
  unknown names raise the actionable error;
* the aggregators: weight normalisation, NaN-dropout (numerator AND
  denominator), the geometric clamp, and the typed refusals
  (mismatched lengths, negative weights, empty input);
* the raw-component objectives: MW through a step window, QED through
  a sigmoid, invalid SMILES -> 0.0 with ``n_invalid`` accounting, the
  unknown-component refusal;
* GroupCount with a halogen ceiling; CustomAlerts' blocklist veto;
  MatchingSubstructure's ``0.5 * (1 + match)``;
* every constructor's typed refusals (non-finite parameters, low >
  high, k <= 0, a bare string where a list is needed).

RDKit is required; the file skips as a module when it is absent.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.generative.scoring_transforms import (
    RAW_COMPONENTS,
    TRANSFORM_REGISTRY,
    CustomAlertsObjective,
    DoubleSigmoidTransform,
    ExponentialDecayTransform,
    GroupCountObjective,
    LeftStepTransform,
    MatchingSubstructureObjective,
    RawComponentObjective,
    ReverseSigmoidTransform,
    RightStepTransform,
    ScoreTransform,
    SigmoidTransform,
    StepTransform,
    ValueMappingTransform,
    make_transform,
    weighted_arithmetic_mean,
    weighted_geometric_mean,
)

_HAS_RDKIT = importlib.util.find_spec("rdkit") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_RDKIT, reason="RDKit not installed (the zip-only tree)"
)


# ---------------------------------------------------------------------------
# The transforms
# ---------------------------------------------------------------------------


class TestSigmoid:
    def test_window_centre_is_half(self) -> None:
        sig = SigmoidTransform(low=200.0, high=400.0, k=0.5)
        assert float(sig([300.0])[0]) == pytest.approx(0.5)

    def test_monotone_rise_across_the_window(self) -> None:
        sig = SigmoidTransform(low=0.0, high=10.0, k=1.0)
        values = sig([0.0, 2.5, 5.0, 7.5, 10.0])
        assert values[0] < values[1] < values[2] < values[3] < values[4]

    def test_reinvent_k_rescaling(self) -> None:
        # k' = 10 k / (high - low): with k=0.5 over a 200-wide window
        # the edges sit ~exp(ln(10) * 0.025 * 100) away from the centre
        sig = SigmoidTransform(low=200.0, high=400.0, k=0.5)
        edge = float(sig([200.0])[0])
        assert edge == pytest.approx(1.0 / (1.0 + 10.0 ** 2.5), rel=1e-6)

    def test_degenerate_high_equals_low_is_the_hard_step(self) -> None:
        sig = SigmoidTransform(low=5.0, high=5.0, k=1.0)
        assert sig([4.0, 6.0]).tolist() == [0.0, 1.0]

    def test_reverse_sigmoid_mirrors_the_sigmoid(self) -> None:
        forward = SigmoidTransform(low=0.0, high=10.0, k=1.0)([2.5, 5.0, 7.5])
        reverse = ReverseSigmoidTransform(low=0.0, high=10.0, k=1.0)(
            [2.5, 5.0, 7.5]
        )
        np.testing.assert_allclose(forward + reverse, np.ones(3), atol=1e-12)

    def test_typed_refusals(self) -> None:
        with pytest.raises(DataValidationError, match="low"):
            SigmoidTransform(low=math.nan, high=1.0)  # type: ignore[arg-type]
        with pytest.raises(DataValidationError, match="k"):
            SigmoidTransform(low=0.0, high=1.0, k=math.inf)  # type: ignore[arg-type]


class TestDoubleSigmoid:
    def test_centre_is_the_maximum(self) -> None:
        dbl = DoubleSigmoidTransform(low=250.0, high=450.0, k=100.0)
        assert float(dbl([350.0])[0]) == pytest.approx(1.0)

    def test_both_tails_decay(self) -> None:
        dbl = DoubleSigmoidTransform(low=250.0, high=450.0, k=100.0)
        out = dbl([100.0, 350.0, 700.0])
        assert out[1] > out[0] and out[1] > out[2]
        assert float(out[0]) < 0.01 and float(out[2]) < 0.01

    def test_zero_common_coefficient_gives_hard_steps(self) -> None:
        dbl = DoubleSigmoidTransform(low=0.0, high=10.0, k=0.0)
        out = dbl([-1.0, 5.0, 11.0])
        assert out.tolist() == [0.0, 1.0, 0.0]

    def test_low_above_high_refused(self) -> None:
        with pytest.raises(DataValidationError, match="exceeds"):
            DoubleSigmoidTransform(low=10.0, high=0.0)


class TestStepsAndDecay:
    def test_right_step_edge_is_inclusive(self) -> None:
        assert RightStepTransform(high=0.7)([0.7, 0.6999]).tolist() == [1.0, 0.0]

    def test_left_step_edge_is_inclusive(self) -> None:
        assert LeftStepTransform(low=0.3)([0.3, 0.31]).tolist() == [1.0, 0.0]

    def test_window_step_bounds_inclusive(self) -> None:
        step = StepTransform(low=0.2, high=0.8)
        assert step([0.2, 0.5, 0.8, 0.9]).tolist() == [1.0, 1.0, 1.0, 0.0]

    def test_exponential_decay_formula(self) -> None:
        decay = ExponentialDecayTransform(k=0.5)
        assert float(decay([2.0])[0]) == pytest.approx(math.exp(-1.0))
        assert float(decay([-3.0])[0]) == 1.0  # rectified below zero

    def test_exponential_decay_k_must_be_positive(self) -> None:
        with pytest.raises(DataValidationError, match="> 0"):
            ExponentialDecayTransform(k=0.0)

    def test_value_mapping(self) -> None:
        mapping = ValueMappingTransform({"MMP": 1.0, "No MMP": 0.0})
        assert mapping(["MMP", "No MMP"]).tolist() == [1.0, 0.0]
        assert math.isnan(float(mapping(["surprise"])[0]))


# ---------------------------------------------------------------------------
# The registry / factory
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_every_transform_class_is_registered(self) -> None:
        # REINVENT4's registry keys: the lowercase class name (the
        # toolkit's Transform suffix stripped)
        expected = {
            "sigmoid",
            "reversesigmoid",
            "doublesigmoid",
            "rightstep",
            "leftstep",
            "step",
            "exponentialdecay",
            "valuemapping",
        }
        assert set(TRANSFORM_REGISTRY) == expected

    def test_name_normalisation(self) -> None:
        for spelling in (
            "DoubleSigmoid",
            "double_sigmoid",
            "double-sigmoid",
            "DOUBLE-SIGMOID",
        ):
            assert isinstance(make_transform(spelling, low=0.0, high=1.0, k=1.0),
                              DoubleSigmoidTransform)

    def test_unknown_name_names_the_registry(self) -> None:
        with pytest.raises(DataValidationError, match="registry holds"):
            make_transform("no-such-transform")

    def test_bad_parameters_refused(self) -> None:
        with pytest.raises(DataValidationError, match="rejected"):
            make_transform("sigmoid", low=0.0, high=1.0, nonsense=3.0)

    def test_every_transform_is_a_score_transform(self) -> None:
        for transform_type in TRANSFORM_REGISTRY.values():
            assert issubclass(transform_type, ScoreTransform)


# ---------------------------------------------------------------------------
# The aggregators
# ---------------------------------------------------------------------------


class TestAggregators:
    def test_arithmetic_weight_normalisation(self) -> None:
        out = weighted_arithmetic_mean([([0.2, 0.8], 1.0), ([0.6, 0.4], 3.0)])
        np.testing.assert_allclose(out, [0.5, 0.5], atol=1e-12)

    def test_nan_drops_out_of_numerator_and_denominator(self) -> None:
        out = weighted_arithmetic_mean([([0.2], 1.0), ([float("nan")], 2.0)])
        assert float(out[0]) == pytest.approx(0.2)

    def test_geometric_mean(self) -> None:
        out = weighted_geometric_mean([([0.5], 1.0), ([0.8], 1.0)])
        assert float(out[0]) == pytest.approx(math.sqrt(0.4))

    def test_geometric_clamps_zeros(self) -> None:
        out = weighted_geometric_mean([([0.0], 1.0), ([0.5], 1.0)])
        assert float(out[0]) == pytest.approx(math.sqrt(1e-8 * 0.5))

    def test_length_mismatch_refused(self) -> None:
        with pytest.raises(DataValidationError, match="disagree"):
            weighted_arithmetic_mean([([0.1, 0.2], 1.0), ([0.3], 1.0)])

    def test_negative_weight_refused(self) -> None:
        with pytest.raises(DataValidationError, match="non-negative"):
            weighted_arithmetic_mean([([0.1], 1.0), ([0.3], -2.0)])

    def test_empty_input_refused(self) -> None:
        with pytest.raises(DataValidationError, match="at least one"):
            weighted_arithmetic_mean([])


# ---------------------------------------------------------------------------
# The raw components
# ---------------------------------------------------------------------------


class TestRawComponents:
    def test_registry_holds_the_reinvent_physchem_set(self) -> None:
        expected = {
            "molecular_weight",
            "logp",
            "tpsa",
            "qed",
            "h_bond_acceptors",
            "h_bond_donors",
            "rotatable_bonds",
            "num_rings",
            "aromatic_rings",
            "aliphatic_rings",
            "heavy_atoms",
            "heteroatoms",
            "csp3_fraction",
            "largest_ring",
            "graph_length",
            "num_atom_stereo_centers",
        }
        assert expected <= set(RAW_COMPONENTS)

    def test_molecular_weight_through_a_step_window(self) -> None:
        objective = RawComponentObjective(
            "molecular_weight", StepTransform(low=150, high=450)
        )
        assert objective.score("CCO") == 0.0            # ethanol: 46 Da
        assert objective.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0  # aspirin

    def test_raw_value_is_untransformed(self) -> None:
        objective = RawComponentObjective(
            "molecular_weight", StepTransform(low=0, high=1)
        )
        assert objective.raw_value("CCO") == pytest.approx(46.07, rel=1e-3)

    def test_invalid_smiles_scores_zero_and_counts(self) -> None:
        objective = RawComponentObjective(
            "logp", SigmoidTransform(low=0.0, high=5.0, k=1.0)
        )
        assert objective.score("not-a-smiles") == 0.0
        assert objective.n_invalid == 1
        assert math.isnan(objective.raw_value("not-a-smiles"))

    def test_unknown_component_refused(self) -> None:
        with pytest.raises(DataValidationError, match="unknown raw component"):
            RawComponentObjective("nope", StepTransform(low=0, high=1))

    def test_missing_transform_refused(self) -> None:
        with pytest.raises(DataValidationError, match="needs a ScoreTransform"):
            RawComponentObjective("logp")  # type: ignore[arg-type]


class TestGroupCount:
    def test_halogen_ceiling(self) -> None:
        fluorines = GroupCountObjective("[F]", LeftStepTransform(low=1))
        assert fluorines.score("CCF") == 1.0            # one F: at the ceiling
        assert fluorines.score("C(F)(F)F") == 0.0       # three F: over it

    def test_zero_matches_flow_through_the_transform(self) -> None:
        fluorines = GroupCountObjective("[F]", LeftStepTransform(low=1))
        assert fluorines.score("CCO") == 1.0             # zero F: still under

    def test_bad_smarts_refused(self) -> None:
        with pytest.raises(DataValidationError, match="parseable SMARTS"):
            GroupCountObjective("??[", StepTransform(low=0, high=1))


# ---------------------------------------------------------------------------
# The filter / penalty objectives
# ---------------------------------------------------------------------------


class TestCustomAlerts:
    def test_clean_molecule_scores_one(self) -> None:
        alerts = CustomAlertsObjective(["[N+](=O)[O-]", "C(=O)Cl"])
        assert alerts.score("CCO") == 1.0

    def test_any_alert_hit_zeroes_the_score(self) -> None:
        alerts = CustomAlertsObjective(["[N+](=O)[O-]", "C(=O)Cl"])
        assert alerts.score("Cc1ccc([N+](=O)[O-])cc1") == 0.0

    def test_bare_string_refused(self) -> None:
        with pytest.raises(DataValidationError, match="not a single"):
            CustomAlertsObjective("[N+](=O)[O-]")  # type: ignore[arg-type]

    def test_empty_list_refused(self) -> None:
        with pytest.raises(DataValidationError, match="filters nothing"):
            CustomAlertsObjective([])

    def test_invalid_smiles_zeroes_and_counts(self) -> None:
        alerts = CustomAlertsObjective(["[F]"])
        assert alerts.score("not-a-smiles") == 0.0
        assert alerts.n_invalid == 1


class TestMatchingSubstructure:
    def test_match_scores_one_and_miss_scores_half(self) -> None:
        penalty = MatchingSubstructureObjective(["c1ccccc1"])
        assert penalty.score("CC(=O)Oc1ccccc1C(=O)O") == 1.0
        assert penalty.score("CCO") == 0.5

    def test_any_of_the_patterns_counts(self) -> None:
        penalty = MatchingSubstructureObjective(["c1ccccc1", "C(=O)N"])
        assert penalty.score("CC(=O)Nc1ccccc1") == 1.0

    def test_invalid_smiles_scores_zero(self) -> None:
        penalty = MatchingSubstructureObjective(["c1ccccc1"])
        assert penalty.score("nonsense") == 0.0


# ---------------------------------------------------------------------------
# The MultiObjective integration (the geometric-mean product semantics)
# ---------------------------------------------------------------------------


class TestMultiObjectiveIntegration:
    def test_alerts_veto_inside_the_geometric_mean(self) -> None:
        from pharmaml.generative import MultiObjective, QedObjective

        objective = MultiObjective(
            [QedObjective(), CustomAlertsObjective(["[N+](=O)[O-]"])]
        )
        clean = objective.score("CCOc1ccccc1")
        nitro = objective.score("Cc1ccc([N+](=O)[O-])cc1")
        assert clean > 0.0
        assert nitro == 0.0  # the alert vetoes the whole product

    def test_penalty_enters_the_geometric_product(self) -> None:
        from pharmaml.generative import MultiObjective, QedObjective

        # REINVENT4 multiplies penalties into the final score; the
        # toolkit's geometric mean computes exactly that product -- the
        # penalty member is 0.5 for a non-match, so the equal-weight
        # total is sqrt(qed * 0.5)
        with_penalty = MultiObjective(
            [QedObjective(), MatchingSubstructureObjective(["c1ccccc1"])]
        )
        qed_value = QedObjective().score("CCO")
        assert with_penalty.score("CCO") == pytest.approx(
            math.sqrt(qed_value * 0.5), rel=1e-9
        )
        # and a matching molecule (penalty 1.0) carries no discount
        # beyond the geometric mean itself
        qed_match = QedObjective().score("CCOc1ccccc1")
        assert with_penalty.score("CCOc1ccccc1") == pytest.approx(
            math.sqrt(qed_match * 1.0), rel=1e-9
        )
