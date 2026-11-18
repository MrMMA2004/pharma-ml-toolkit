"""Round-176 tests: PK-Sim physiology, the mrgsolve/rxode2 event grammar,
OSPSuite-R sensitivity machinery."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pharmaml.exceptions import DataValidationError
from pharmaml.pkpd._physiology_data import (
    GFR_CONSTANTS,
    HUMAN_AGE_CURVES,
    ONTOGENY_TABLE,
    PLASMA_COMPOSITION,
    SPECIES_PHYSIOLOGY,
    TISSUE_COMPOSITION,
    TRANSPORTER_TYPES,
)
from pharmaml.pkpd.events import (
    EventRow,
    addl_until,
    ev,
    ev_days,
    ev_seq,
    realize_addl,
    simulate_regimen,
)
from pharmaml.pkpd.physiology import (
    ADULT_GFRSPEC,
    blood_plasma_ratio,
    bsa_dubois,
    bsa_mosteller,
    clint_to_clspec,
    create_individual,
    fu_mic,
    fu_ontogeny_adjusted,
    gfr_maturation_fraction,
    gfr_reference,
    kp_berezhkovskiy,
    kp_pk_sim_standard,
    kp_poulin_theil,
    kp_rodgers_rowland,
    kp_schmitt,
    ontogeny_factor,
    ontogeny_geo_sd,
    pma_years,
)
from pharmaml.pkpd.sensitivity import (
    PK_PARAMETER_REGISTRY,
    SPECIES,
    StandardPKParameter,
    create_individual_characteristics,
    match_paths,
    pk_parameter_frame,
    sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# PK-Sim physiology: the data layer
# ---------------------------------------------------------------------------


class TestPhysiologyData:
    def test_ontogeny_table_shape(self) -> None:
        # 508 points over 20 molecules, four group kinds.
        n_points = sum(
            len(series) for groups in ONTOGENY_TABLE.values() for series in groups.values()
        )
        assert len(ONTOGENY_TABLE) == 20
        assert n_points == 508
        assert set(ONTOGENY_TABLE["CYP3A4"]) == {"ONTOGENY_DUODENUM", "ONTOGENY_LIVER_NO_GI"}

    def test_species_liver_values(self) -> None:
        # The ICRP-scale static liver row for Rat, straight from the DB.
        rat = SPECIES_PHYSIOLOGY["Rat"]
        assert rat["volume_l"] == pytest.approx(0.0103)
        assert rat["qspec_l_min_kg"] == pytest.approx(0.194174757281553)
        assert rat["mppgl_mg_g"] == pytest.approx(0.054)
        assert len(SPECIES_PHYSIOLOGY) == 10

    def test_tissue_composition_liver(self) -> None:
        liver = TISSUE_COMPOSITION["Liver"]
        assert liver["f_iw"] == pytest.approx(0.481)
        assert liver["f_ew"] == pytest.approx(0.161)
        assert liver["albumin_ratio"] == pytest.approx(0.086)
        assert liver["acidic_phospholipids_rr"] == pytest.approx(4.56)

    def test_plasma_composition(self) -> None:
        # The four pH values PK-Sim's ionic terms run on.
        assert PLASMA_COMPOSITION["ph_plasma"] == pytest.approx(7.4)
        assert PLASMA_COMPOSITION["ph_interstitial"] == pytest.approx(7.4)
        assert PLASMA_COMPOSITION["ph_intracellular"] == pytest.approx(7.0)
        assert PLASMA_COMPOSITION["ph_blood_cells"] == pytest.approx(7.22)

    def test_gfr_constants(self) -> None:
        assert GFR_CONSTANTS["GFRmat"] == pytest.approx(0.11704)
        assert GFR_CONSTANTS["Volume (standard kidney)"] == pytest.approx(0.44)
        assert ADULT_GFRSPEC == pytest.approx(0.266, abs=1e-4)

    def test_age_curves_reference_male(self) -> None:
        bw = dict(HUMAN_AGE_CURVES["mean_bw"]["MALE"])
        assert bw[30.0] == pytest.approx(73.0)
        height = dict(HUMAN_AGE_CURVES["mean_height_dm"]["MALE"])
        assert height[30.0] == pytest.approx(17.6)  # decimetres — 176 cm

    def test_transporter_ontology(self) -> None:
        assert TRANSPORTER_TYPES["ABCB1"] == "Efflux"
        assert TRANSPORTER_TYPES["SLC22A1"] == "Influx"


# ---------------------------------------------------------------------------
# PK-Sim physiology: the formulas
# ---------------------------------------------------------------------------


class TestPmaAndIndividual:
    def test_pma_convention(self) -> None:
        assert pma_years(0.0, 40.0) == pytest.approx(40.0 * 7 / 365.25)
        assert pma_years(1.0) == pytest.approx(1.0 + 40.0 * 7 / 365.25)

    def test_create_individual_reference_male(self) -> None:
        ind = create_individual(age_years=30.0, gender="MALE")
        assert ind["Organism|Weight"] == pytest.approx(73.0)
        assert ind["Organism|Height"] == pytest.approx(17.6)
        assert ind["Organism|Hematocrit"] == pytest.approx(0.47)
        assert ind["Liver|Volume"] == pytest.approx(2.3578466712)
        # Direct (arterial) liver flow: Qspec * weight, L/min -> L/h.
        assert ind["Liver|Blood flow"] == pytest.approx(0.1794 * 2.3578466712 * 60.0)

    def test_create_individual_female(self) -> None:
        ind = create_individual(age_years=30.0, gender="FEMALE")
        assert ind["Organism|Weight"] == pytest.approx(60.0)

    def test_create_individual_interp_between_ages(self) -> None:
        # 20 y sits between the 15 y and 30 y rows -> linear interpolation.
        ind = create_individual(age_years=20.0, gender="MALE")
        bw = dict(HUMAN_AGE_CURVES["mean_bw"]["MALE"])
        expect = bw[15.0] + (bw[30.0] - bw[15.0]) * (20.0 - 15.0) / 15.0
        assert ind["Organism|Weight"] == pytest.approx(expect)

    def test_create_individual_rejects_bad_gender(self) -> None:
        with pytest.raises(DataValidationError, match="gender"):
            create_individual(age_years=30.0, gender="robot")


class TestGfrAndBsa:
    def test_gfr_maturation_neonate_and_adult(self) -> None:
        birth = gfr_maturation_fraction(40.0 * 7 / 365.25)
        assert birth == pytest.approx(0.3863, abs=1e-4)
        assert gfr_maturation_fraction(2.0) == pytest.approx(0.999998, abs=1e-5)
        assert gfr_maturation_fraction(30.0) == pytest.approx(1.0, abs=1e-12)

    def test_gfr_reference_adult_is_pksim_test_value(self) -> None:
        # PK-Sim's IndividualSpecs integration test: 0.266 L/min/kg.
        assert gfr_reference(30.0) == pytest.approx(0.266, abs=1e-3)

    def test_gfr_reference_neonatal_scaling(self) -> None:
        baby = gfr_reference(0.0)  # term neonate, PMA ~40 weeks
        assert baby == pytest.approx(0.266 * 0.3863, abs=1e-3)

    def test_bsa_mosteller_reference(self) -> None:
        assert bsa_mosteller(73.0, 176.0) == pytest.approx(1.889150, abs=1e-5)

    def test_bsa_dubois_close_to_mosteller(self) -> None:
        assert bsa_dubois(73.0, 176.0) == pytest.approx(bsa_mosteller(73.0, 176.0), rel=0.01)

    def test_bsa_rejects_nonpositive(self) -> None:
        with pytest.raises(DataValidationError):
            bsa_mosteller(0.0, 176.0)
        with pytest.raises(DataValidationError):
            bsa_dubois(73.0, -1.0)


class TestOntogeny:
    def test_cyp3a4_interpolation(self) -> None:
        # 0.8 y sits between the 0.77 -> 0.12 and 0.92 -> 0.2 rows.
        assert ontogeny_factor("CYP3A4", 0.8) == pytest.approx(0.136, abs=1e-4)

    def test_cyp3a4_adult_is_one(self) -> None:
        assert ontogeny_factor("CYP3A4", 30.0) == pytest.approx(1.0)

    def test_cyp3a7_switch_off(self) -> None:
        assert ontogeny_factor("CYP3A7", 0.72) == pytest.approx(1.0)
        assert ontogeny_factor("CYP3A7", 2.0) == pytest.approx(0.025, abs=1e-6)

    def test_liver_group_fallback(self) -> None:
        # CYP2D6 carries LIVER_ALL only; the fallback still resolves.
        assert ontogeny_factor("CYP2D6", 30.0) == pytest.approx(1.0)
        # CYP3A4's liver default is the GI-free series.
        assert ontogeny_factor("CYP3A4", 30.0, group="liver") == pytest.approx(1.0)

    def test_plasma_group_agp(self) -> None:
        assert ontogeny_factor("AGP", 30.767, group="plasma") == pytest.approx(1.0, abs=1e-3)

    def test_geo_sd_interpolates(self) -> None:
        sd = ontogeny_geo_sd("CYP3A4", 30.0)
        assert sd == pytest.approx(1.18)

    def test_unknown_molecule_refused(self) -> None:
        with pytest.raises(DataValidationError, match="unknown ontogeny molecule"):
            ontogeny_factor("CYP999", 1.0)

    def test_unknown_group_refused(self) -> None:
        with pytest.raises(DataValidationError, match="no ontogeny group"):
            ontogeny_factor("CYP3A4", 1.0, group="ONTOGENY_PLASMA")

    def test_fu_ontogeny_adjusted_neonate_more_unbound(self) -> None:
        # AGP at ~25% of adult levels in a neonate frees a bound drug.
        fu_ref = 0.05
        baby = fu_ontogeny_adjusted(fu_ref, "AGP", pma_years=0.8)
        adult = fu_ontogeny_adjusted(fu_ref, "AGP", pma_years=30.767)
        assert adult == pytest.approx(fu_ref, abs=1e-6)
        assert baby > adult

    def test_fu_ontogeny_caps_at_one(self) -> None:
        assert fu_ontogeny_adjusted(1.0, "AGP", pma_years=0.8) == 1.0


class TestPartitionCoefficients:
    def test_kprot_driven_pk_sim_standard(self) -> None:
        # Willmann's method on liver composition (hand-checked value).
        assert kp_pk_sim_standard(1.0, 0.5, "Liver") == pytest.approx(0.753757, abs=1e-5)

    def test_tissue_alias(self) -> None:
        assert kp_pk_sim_standard(1.0, 0.5, "liver") == kp_pk_sim_standard(1.0, 0.5, "Liver")
        assert kp_pk_sim_standard(1.0, 0.5, "adipose") == kp_pk_sim_standard(1.0, 0.5, "Fat")

    def test_unknown_tissue_refused(self) -> None:
        with pytest.raises(DataValidationError, match="unknown tissue"):
            kp_pk_sim_standard(1.0, 0.5, "Spleenish")

    def test_poulin_theil_unbound(self) -> None:
        assert kp_poulin_theil(2.0, 1.0, "Liver") == pytest.approx(3.668889, abs=1e-5)

    def test_poulin_theil_binding_lowers_kp(self) -> None:
        assert kp_poulin_theil(2.0, 0.05, "Liver") == pytest.approx(0.483193, abs=1e-5)

    def test_berezhkovskiy_below_poulin_when_bound(self) -> None:
        # The fu asymmetry inside the sums pushes K below PT's.
        assert kp_berezhkovskiy(2.0, 0.05, "Liver") == pytest.approx(0.322571, abs=1e-5)
        assert kp_berezhkovskiy(2.0, 1.0, "Liver") == kp_poulin_theil(2.0, 1.0, "Liver")

    def test_rodgers_rowland_neutral_and_base(self) -> None:
        assert kp_rodgers_rowland(2.0, 1.0, "Liver") == pytest.approx(2.72266, abs=1e-4)
        # A base (pKa 9) accumulates intracellularly through (1+X)/(1+Y)…
        base = kp_rodgers_rowland(2.0, 1.0, "Liver", pka=9.0, charge="base")
        assert base == pytest.approx(1.402381, abs=1e-4)

    def test_rodgers_rowland_acid_below_neutral(self) -> None:
        neutral = kp_rodgers_rowland(2.0, 1.0, "Liver")
        acid = kp_rodgers_rowland(2.0, 1.0, "Liver", pka=4.0, charge="acid")
        assert acid < neutral

    def test_rodgers_rowland_validation(self) -> None:
        with pytest.raises(DataValidationError, match="charged species need a pka"):
            kp_rodgers_rowland(2.0, 1.0, "Liver", charge="base")
        with pytest.raises(DataValidationError, match="charge"):
            kp_rodgers_rowland(2.0, 1.0, "Liver", charge="zwitterion")

    def test_schmitt_neutral_form(self) -> None:
        kprot = (0.81 + 0.11 * 10.0) / 24.92 * 5.0
        expected = (0.747 + 100.0 * 0.0135 + 10.0 * 0.0238 + kprot * 0.184) * 1.0
        assert kp_schmitt(2.0, 1.0, 1.0, "Liver") == pytest.approx(expected, abs=1e-9)

    def test_blood_plasma_ratio_bounds(self) -> None:
        # A purely hydrophilic unbound compound: K_rbc is the red-cell
        # water+lipid+protein core, B2P = 1 + HCT*(K_rbc - 1).
        kprot = (0.81 + 0.11) / 24.92 * 5.0
        k_rbc = 0.625 + 0.005 * 1.0 + 0.325 * kprot
        assert blood_plasma_ratio(0.0, 1.0, hct=0.47) == pytest.approx(
            1.0 + 0.47 * (k_rbc - 1.0), abs=1e-9
        )
        assert blood_plasma_ratio(3.0, 1.0, hct=0.47) > 1.0
        assert blood_plasma_ratio(0.0, 1.0, hct=0.0) == pytest.approx(1.0)


class TestIvive:
    def test_fu_mic_hydrophilic_and_lipophilic(self) -> None:
        assert fu_mic(0.0, 0.1) == pytest.approx(1.0)
        assert fu_mic(3.0, 0.1) == pytest.approx(0.00991, abs=1e-5)

    def test_fu_mic_bounds(self) -> None:
        with pytest.raises(DataValidationError):
            fu_mic(0.0, 1.0)

    def test_clint_to_clspec_pksim_acceptance(self) -> None:
        # InVitroSpecificClearanceSpecs: residual 0.8 over 20 min with the
        # 400x liver-to-protein scaling, fu_mic 0.28476, f_cell 0.667.
        spec = clint_to_clspec(
            clint_per_protein=math.log(1.0 / 0.8) / 20.0,
            liver_to_protein=400.0,
            fu_mic_value=0.28476,
            f_cell=0.667,
        )
        assert spec == pytest.approx(23.497, abs=1e-3)

    def test_clint_to_clspec_per_enzyme_normalisation(self) -> None:
        # 108 pmol CYP/mg -> CLspec/[Enzyme] = 1/108 L/umol/min.
        assert 1.0 / 108.0 == pytest.approx(0.009259, abs=1e-6)

    def test_clint_to_clspec_validation(self) -> None:
        with pytest.raises(DataValidationError, match="fu_mic_value"):
            clint_to_clspec(
                clint_per_protein=0.01, liver_to_protein=400.0, fu_mic_value=1.5, f_cell=0.667
            )


# ---------------------------------------------------------------------------
# mrgsolve / rxode2 event grammar
# ---------------------------------------------------------------------------


class TestEvConstructor:
    def test_tinf_converts_to_rate(self) -> None:
        row = ev(100.0, tinf=2.0)
        assert row.rate == pytest.approx(50.0)

    def test_total_converts_to_addl(self) -> None:
        row = ev(100.0, total=3, ii=12.0)
        assert row.addl == 2

    def test_until_converts_to_addl_mrgsolve(self) -> None:
        row = ev(100.0, ii=24.0, until=120.0)
        assert row.addl == 4

    def test_mutually_exclusive_refusals(self) -> None:
        with pytest.raises(DataValidationError, match="mutually exclusive"):
            ev(100.0, rate=10.0, tinf=2.0)
        with pytest.raises(DataValidationError, match="mutually exclusive"):
            ev(100.0, total=3, until=120.0, ii=24.0)

    def test_evid_zero_refused(self) -> None:
        with pytest.raises(DataValidationError, match="evid cannot be 0"):
            ev(100.0, evid=0)

    def test_modeled_rate_refused(self) -> None:
        with pytest.raises(DataValidationError, match="modeled rate"):
            EventRow(time=0.0, amt=100.0, rate=-1.0)

    def test_addl_requires_ii(self) -> None:
        with pytest.raises(DataValidationError, match="requires ii"):
            EventRow(time=0.0, amt=100.0, addl=2)

    def test_cmt_is_one_based(self) -> None:
        with pytest.raises(DataValidationError, match="1-based"):
            EventRow(time=0.0, amt=1.0, cmt=0)


class TestAddlUntil:
    def test_mrgsolve_rule(self) -> None:
        # Doses strictly before `until`: 0, 24, 48, 72, 96 -> addl 4.
        assert addl_until(0.0, 24.0, 120.0, convention="mrgsolve") == 4

    def test_rxode2_rule(self) -> None:
        # rxode2 adds the exactly-divisible boundary dose -> addl 5.
        assert addl_until(0.0, 24.0, 120.0, convention="rxode2") == 5

    def test_rxode2_nondivisible(self) -> None:
        # 0..110 at ii 24: span 110-24=86, ceil(86/24)=4, not divisible.
        assert addl_until(0.0, 24.0, 110.0, convention="rxode2") == 4

    def test_unknown_convention(self) -> None:
        with pytest.raises(DataValidationError, match="convention"):
            addl_until(0.0, 24.0, 120.0, convention="nonmem")


class TestRealizeAndSeq:
    def test_realize_addl_expansion(self) -> None:
        rows = realize_addl([ev(100.0, ii=24.0, addl=3)])
        assert [r.time for r in rows] == [0.0, 24.0, 48.0, 72.0]
        assert all(r.ii == 0.0 and r.addl == 0 for r in rows)
        assert rows[1].from_addl is True

    def test_realize_addl_truncation(self) -> None:
        rows = realize_addl([ev(100.0, ii=24.0, addl=5)], max_time=50.0)
        assert [r.time for r in rows] == [0.0, 24.0, 48.0]

    def test_realize_evid4_demotes_on_copies(self) -> None:
        rows = realize_addl([ev(100.0, ii=1.0, addl=1, evid=4)])
        assert rows[0].evid == 4
        assert rows[1].evid == 1

    def test_ev_seq_wait_and_ii_spacers(self) -> None:
        first = ev(100.0, ii=24.0, addl=2)
        wait_rows = ev_seq(first, ev(50.0), wait=10.0)
        assert [r.time for r in wait_rows] == [0.0, 24.0, 48.0, 82.0]
        ii_rows = ev_seq(first, ev(50.0), ii=10.0)
        assert [r.time for r in ii_rows] == [0.0, 24.0, 48.0, 58.0]

    def test_ev_days_offsets(self) -> None:
        rows = ev_days(100.0, ["m", "w", "f"])
        assert [r.time for r in rows] == [0.0, 48.0, 96.0]
        with pytest.raises(DataValidationError, match="weekday"):
            ev_days(100.0, ["x"])


class TestSimulateRegimen:
    def test_infusion_peak_at_infusion_end(self) -> None:
        # mrgsolve test-infusion.R: amt=3000, rate=30 -> tmax == 100.
        grid = np.arange(0.0, 121.0, 1.0)
        sol = simulate_regimen(lambda t, y: -0.03 * y, [0.0], grid, [ev(3000.0, rate=30.0)])
        assert float(grid[int(np.argmax(sol.states[:, 0]))]) == 100.0

    def test_ss_continuous_infusion_rate_over_cl(self) -> None:
        # mrgsolve: CL=1.3, rate=25 -> CP == 25/1.3.
        sol = simulate_regimen(
            lambda t, y: -0.065 * y,
            [0.0],
            [0.0, 6.0, 12.0],
            [ev(0.0, rate=25.0, ss=1, ii=12.0)],
        )
        cp = sol.concentration(0, volume=20.0)
        assert cp[0] == pytest.approx(25.0 / 1.3, abs=1e-4)
        assert cp[1] == pytest.approx(25.0 / 1.3, abs=1e-4)

    def test_ss_bolus_peak_closed_form(self) -> None:
        # rxode2 test-steady-state.R: c0 / (1 - exp(-ke*ii)).
        ke, dose, vol, ii = 0.05, 3.0, 20.0, 12.0
        sol = simulate_regimen(lambda t, y: -ke * y, [0.0], [0.0], [ev(dose, ss=1, ii=ii)])
        assert sol.concentration(0, volume=vol)[0] == pytest.approx(
            (dose / vol) / (1.0 - math.exp(-ke * ii)), abs=1e-6
        )

    def test_ss_two_superposes_linearly(self) -> None:
        # rxode2: ss=1 at t0 + ss=2 at t12 == the two steady states summed
        # (Y_after = Y_before + Y_ss_new).  The superposed regimen starts
        # at t=12; the observation sits past the jump.
        ke, ii = 0.05, 24.0
        rhs = lambda t, y: -ke * y
        base = simulate_regimen(rhs, [0.0], [13.0], [ev(10.0, ss=1, ii=ii)])
        added = simulate_regimen(
            rhs, [0.0], [13.0],
            [ev(10.0, ss=1, ii=ii), ev(20.0, time=12.0, ss=2, ii=ii)],
        )
        alone = simulate_regimen(
            rhs, [0.0], [13.0], [ev(20.0, time=12.0, ss=1, ii=ii)]
        )
        assert added.states[0, 0] == pytest.approx(
            base.states[0, 0] + alone.states[0, 0], abs=1e-6
        )

    def test_bioavailability_shrinks_infusion_duration(self) -> None:
        # mrgsolve test-D-R-F.R: F=0.5 halves the duration (tmax 10 -> 5).
        grid = np.arange(0.0, 25.0, 0.5)
        rhs = lambda t, y: -0.03 * y
        f1 = simulate_regimen(rhs, [0.0], grid, [ev(100.0, rate=10.0)])
        f05 = simulate_regimen(rhs, [0.0], grid, [ev(100.0, rate=10.0, f=0.5)])
        assert int(grid[int(np.argmax(f1.states[:, 0]))]) == 10
        assert int(grid[int(np.argmax(f05.states[:, 0]))]) == 5

    def test_lag_time_shifts_application(self) -> None:
        # mrgsolve test-alag.R: the dose lands at exactly t=2.8 — under
        # recsort 3 the observation at that time sees it (pre-dose under
        # the default recsort 1, which the tie-break test pins).
        grid = [0.0, 2.79, 2.8, 3.0]
        sol = simulate_regimen(
            lambda t, y: 0.0 * y, [0.0], grid, [ev(5.0, lag=2.8)], recsort=3
        )
        assert sol.states[1, 0] == pytest.approx(0.0)
        assert sol.states[2, 0] == pytest.approx(5.0)

    def test_evid3_reset_zeroes_state(self) -> None:
        rhs = lambda t, y: -0.1 * y
        sol = simulate_regimen(
            rhs, [0.0], [0.0, 4.0, 6.0, 8.0], [ev(10.0), ev(0.0, time=6.0, evid=3)]
        )
        assert sol.states[1, 0] == pytest.approx(10.0 * math.exp(-0.4), abs=1e-6)
        assert sol.states[3, 0] == pytest.approx(0.0, abs=1e-12)

    def test_evid4_reset_then_dose(self) -> None:
        rhs = lambda t, y: -0.1 * y
        sol = simulate_regimen(
            rhs, [0.0], [0.0, 4.0, 8.0], [ev(10.0), ev(7.0, time=6.0, evid=4)]
        )
        assert sol.states[1, 0] == pytest.approx(10.0 * math.exp(-0.4), abs=1e-6)
        assert sol.states[2, 0] == pytest.approx(7.0 * math.exp(-0.2), abs=1e-6)

    def test_evid8_replaces_amount(self) -> None:
        rhs = lambda t, y: -0.1 * y
        sol = simulate_regimen(
            rhs, [0.0], [0.0, 4.0, 8.0], [ev(10.0), ev(3.0, time=4.0, evid=8)]
        )
        # At t=4 (recsort 1: observation first) the old amount is recorded;
        # afterwards the replaced 3.0 decays.
        assert sol.states[2, 0] == pytest.approx(3.0 * math.exp(-0.4), abs=1e-6)

    def test_evid2_negative_cmt_zeroes_compartment(self) -> None:
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return np.array([-0.1 * y[0], -0.2 * y[1]])

        sol = simulate_regimen(
            rhs, [0.0, 0.0], [0.0, 4.0, 5.0, 10.0],
            [ev(10.0, cmt=1), ev(5.0, cmt=2, time=0.0), ev(0.0, time=5.0, evid=2, cmt=-2)],
        )
        # Before the off-record the second compartment decays normally.
        assert sol.states[1, 1] == pytest.approx(5.0 * math.exp(-0.8), abs=1e-6)
        # The observation AT the off time sees the post-off state (the
        # jump records sort first, the rxode2 convention).
        assert sol.states[2, 1] == pytest.approx(0.0, abs=1e-12)
        assert sol.states[3, 1] == pytest.approx(0.0, abs=1e-12)
        # Compartment 1 untouched.
        assert sol.states[3, 0] == pytest.approx(10.0 * math.exp(-1.0), abs=1e-6)

    def test_recsort_controls_tie_break(self) -> None:
        # A dose and an observation at t=1: recsort 1 observes pre-dose,
        # recsort 3 post-dose (mrgsolve's put_ev_first).
        rhs = lambda t, y: 0.0 * y
        grid = [1.0, 1.0001]
        pre = simulate_regimen(rhs, [0.0], grid, [ev(1.0, time=1.0)], recsort=1)
        post = simulate_regimen(rhs, [0.0], grid, [ev(1.0, time=1.0)], recsort=3)
        assert pre.states[0, 0] == pytest.approx(0.0)
        assert post.states[0, 0] == pytest.approx(1.0)
        # The record table carries the same ordering.
        assert list(pre.frame["evid"]) == [0, 1, 0]
        assert list(post.frame["evid"]) == [1, 0, 0]

    def test_recsort_addl_variant(self) -> None:
        rhs = lambda t, y: 0.0 * y
        # An addl dose landing exactly on the second observation time:
        # recsort 1 orders addl doses BEFORE observations, recsort 2 after.
        rows = [ev(1.0, time=1.0, ii=0.0001, addl=1)]
        before = simulate_regimen(rhs, [0.0], [1.0, 1.0001], rows, recsort=1)
        after = simulate_regimen(rhs, [0.0], [1.0, 1.0001], rows, recsort=2)
        assert before.states[0, 0] == pytest.approx(0.0)  # explicit dose after obs
        assert before.states[1, 0] == pytest.approx(2.0)  # addl dose before obs
        assert after.states[1, 0] == pytest.approx(1.0)   # addl dose after obs

    def test_multiple_doses_superpose_linearly(self) -> None:
        ke = 0.1
        rhs = lambda t, y: -ke * y
        # Under the default recsort 1 the observation at t=5 is PRE-dose
        # (mrgsolve: data-set doses follow padded observations); under
        # recsort 3 it is post-dose.
        pre = simulate_regimen(rhs, [0.0], [0.0, 5.0, 10.0], [ev(10.0), ev(10.0, time=5.0)])
        assert pre.states[1, 0] == pytest.approx(10.0 * math.exp(-0.5), abs=1e-6)
        post = simulate_regimen(
            rhs, [0.0], [0.0, 5.0, 10.0], [ev(10.0), ev(10.0, time=5.0)], recsort=3
        )
        assert post.states[1, 0] == pytest.approx(
            10.0 * math.exp(-0.5) + 10.0, abs=1e-5
        )
        assert post.states[2, 0] == pytest.approx(
            10.0 * math.exp(-1.0) + 10.0 * math.exp(-0.5), abs=1e-5
        )

    def test_bad_recsort_refused(self) -> None:
        with pytest.raises(DataValidationError, match="recsort"):
            simulate_regimen(lambda t, y: -y, [0.0], [1.0], [ev(1.0)], recsort=5)

    def test_non_increasing_times_refused(self) -> None:
        with pytest.raises(DataValidationError, match="strictly increasing"):
            simulate_regimen(lambda t, y: -y, [0.0], [1.0, 1.0], [ev(1.0)])


# ---------------------------------------------------------------------------
# OSPSuite-R semantics
# ---------------------------------------------------------------------------


class TestPkParameterRegistry:
    def test_registry_is_the_xml(self) -> None:
        assert len(PK_PARAMETER_REGISTRY) == 31
        assert PK_PARAMETER_REGISTRY["CL"].mode == "Single"
        assert PK_PARAMETER_REGISTRY["C_max"].mode == "Always"
        assert PK_PARAMETER_REGISTRY["AUC_inf_tDLast"].mode == "Multi"
        assert PK_PARAMETER_REGISTRY["Vd"].dimension == "Volume per body weight"
        assert PK_PARAMETER_REGISTRY["FractionAucLastToInf"].dimension == "Fraction"

    def test_standard_codes(self) -> None:
        assert int(StandardPKParameter.C_max) == 1
        assert int(StandardPKParameter.AUC_inf) == 12
        assert int(StandardPKParameter.CL) == 16
        assert int(StandardPKParameter.Vss) == 20
        assert int(StandardPKParameter.Tthreshold) == 22

    def test_frame(self) -> None:
        frame = pk_parameter_frame()
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ["displayName", "mode", "dimension"]
        assert frame.loc["Thalf", "dimension"] == "Time"


class TestSensitivityAnalysis:
    def test_elasticity_minus_one_and_zero(self) -> None:
        def auc(params: dict[str, float]) -> float:
            return 100.0 / params["CL"]

        res = sensitivity_analysis(auc, {"CL": 5.0, "V": 20.0}, "AUC_inf")
        assert res.value_for("CL") == pytest.approx(-1.0, abs=1e-9)
        assert res.value_for("V") == pytest.approx(0.0, abs=1e-9)

    def test_elasticity_one_half_power(self) -> None:
        def metric(params: dict[str, float]) -> float:
            return 3.0 * params["V"] ** 0.5

        res = sensitivity_analysis(metric, {"V": 4.0}, "C_max")
        assert res.value_for("V") == pytest.approx(0.5, abs=1e-9)

    def test_variation_points_are_the_osp_defaults(self) -> None:
        seen: list[float] = []

        def spy(params: dict[str, float]) -> float:
            seen.append(params["CL"])
            return 100.0 / params["CL"]

        sensitivity_analysis(spy, {"CL": 5.0}, "AUC_inf")
        varied = sorted({round(v, 6) for v in seen})
        # 90 %, 95 %, 100 %, 105 %, 110 % of the reference.
        assert varied == [4.5, 4.75, 5.0, 5.25, 5.5]

    def test_threshold_filter_keeps_dominant_parameters(self) -> None:
        def metric(params: dict[str, float]) -> float:
            return 100.0 / params["CL"] + 0.0 * params["V"] + 0.0 * params["Q"]

        res = sensitivity_analysis(metric, {"CL": 5.0, "V": 20.0, "Q": 3.0}, "AUC_inf")
        kept = res.thresholded()
        assert list(kept["Parameter"]) == ["CL"]

    def test_frame_schema(self) -> None:
        res = sensitivity_analysis(lambda p: 1.0 / p["CL"], {"CL": 5.0}, "AUC_inf")
        assert list(res.frame.columns) == ["QuantityPath", "Parameter", "PKParameter", "Value"]
        assert res.frame["PKParameter"].iloc[0] == "AUC_inf"

    def test_value_for_unknown_parameter(self) -> None:
        res = sensitivity_analysis(lambda p: 1.0 / p["CL"], {"CL": 5.0}, "AUC_inf")
        with pytest.raises(DataValidationError, match="no sensitivity"):
            res.value_for("KA")

    def test_nonpositive_output_refused(self) -> None:
        with pytest.raises(DataValidationError, match="positive outputs"):
            sensitivity_analysis(lambda p: 0.0, {"CL": 5.0}, "AUC_inf")

    def test_nonpositive_reference_refused(self) -> None:
        with pytest.raises(DataValidationError, match="> 0"):
            sensitivity_analysis(lambda p: 1.0, {"CL": -5.0}, "AUC_inf")


class TestIndividualCharacteristics:
    def test_human_requires_population(self) -> None:
        with pytest.raises(DataValidationError, match="requires a population"):
            create_individual_characteristics("Human")

    def test_unknown_population(self) -> None:
        with pytest.raises(DataValidationError, match="unknown human population"):
            create_individual_characteristics("Human", population="Martian")

    def test_unknown_species(self) -> None:
        with pytest.raises(DataValidationError, match="unknown species"):
            create_individual_characteristics("Dolphin")

    def test_animal_species_needs_no_population(self) -> None:
        ic = create_individual_characteristics("Rat", weight=0.25, age=8.0)
        assert ic.population is None
        assert ic.weight is not None and ic.weight.unit == "kg"

    def test_defaults_and_units(self) -> None:
        ic = create_individual_characteristics(
            "Human", population="European_ICRP_2002", age=43.0, weight=75.0, height=175.0
        )
        assert ic.age is not None and (ic.age.value, ic.age.unit) == (43.0, "year(s)")
        assert ic.gestational_age.value == 40.0
        assert ic.gestational_age.unit == "week(s)"
        assert ic.height is not None and ic.height.unit == "cm"

    def test_gender_validation(self) -> None:
        with pytest.raises(DataValidationError, match="gender"):
            create_individual_characteristics(
                "Human", population="European_ICRP_2002", gender="f"
            )

    def test_species_list(self) -> None:
        assert set(SPECIES) == {
            "Beagle", "Dog", "Human", "Minipig", "Monkey", "Mouse", "Rabbit", "Rat"
        }


class TestMatchPaths:
    def test_single_level_wildcard(self) -> None:
        paths = ["Organism|Weight", "Organism|Liver|Volume", "Organism|Liver|Plasma|Volume"]
        assert match_paths("Organism|*|Volume", paths) == ["Organism|Liver|Volume"]

    def test_recursive_wildcard(self) -> None:
        paths = ["Organism|Weight", "Organism|Liver|Volume", "Organism|Liver|Plasma|Volume"]
        assert match_paths("Organism|**|Volume", paths) == [
            "Organism|Liver|Volume",
            "Organism|Liver|Plasma|Volume",
        ]

    def test_exact_match(self) -> None:
        assert match_paths("Organism|Weight", ["Organism|Weight", "Organism|Height"]) == [
            "Organism|Weight"
        ]

    def test_partial_wildcard_in_segment(self) -> None:
        assert match_paths("Organism|Liv*", ["Organism|Liver", "Organism|Lung"]) == [
            "Organism|Liver"
        ]

    def test_empty_pattern_refused(self) -> None:
        with pytest.raises(DataValidationError):
            match_paths("", ["Organism|Weight"])
