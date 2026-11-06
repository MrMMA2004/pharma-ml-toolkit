"""PK/PD engines: ODEs, compartments, population NLME, PBPK, DDI.

The ``pkpd`` package collects the pharmacometrics engine layer the
toolkit absorbed from the PK/PD simulation and qualification
ecosystem (RxODE / nlmixr2 / pkpdutils / mrgsolve / PK-Sim-adjacent
Open-Systems-Pharmacology repositories):

* :mod:`pharmaml.pkpd.ode` — dosing-event ODE engine (adaptive
  Dormand–Prince RK5(4) + fixed-step RK4);
* :mod:`pharmaml.pkpd.compartments` — exact linear compartment models
  via the matrix exponential (the ``linCmt`` route), superposition,
  the Bateman function;
* :mod:`pharmaml.pkpd.nca_ext` — the pkpdutils NCA extensions (hybrid
  linear-log AUC, best-adjusted-R² terminal windows, steady-state
  and urine and sparse-design NCA);
* :mod:`pharmaml.pkpd.population` — FOCEi-lite nonlinear
  mixed-effects estimation with nlmixr2 reporting conventions;
* :mod:`pharmaml.pkpd.diagnostics` — VPC / prediction-corrected VPC,
  case bootstrap, trial operating characteristics;
* :mod:`pharmaml.pkpd.pbpk` — the minimal five-compartment PBPK model
  with IVIVE, well-stirred hepatic clearance and DDI math;
* :mod:`pharmaml.pkpd.ddi` — DDI qualification (GMFE, Guest limits,
  FDA classification) and the real 34-record digoxin P-gp network;
* :mod:`pharmaml.pkpd.compounds` — the digoxin and itraconazole-chain
  PBPK parameter sets from the OSP model snapshots;
* :mod:`pharmaml.pkpd.osplib` — the OSP PBPK model library registry
  (40 released models);
* :mod:`pharmaml.pkpd.physiology` — PK-Sim's static physiology layer
  (the 508-point ontogeny tables, ICRP age curves, ten species, the
  five partition-coefficient methods, IVIVE conversions);
* :mod:`pharmaml.pkpd.events` — the mrgsolve / rxode2 event grammar
  (``ev()``/``addl``/``until``/``ss``/evid codes/recsort over the ODE
  engine);
* :mod:`pharmaml.pkpd.sensitivity` — the OSPSuite-R PK-parameter
  registry, normalised sensitivities, individual characteristics and
  pipe-path wildcards.

Examples
--------
>>> from pharmaml.pkpd import DosingEvent, TwoCompartment, solve_ode
>>> from pharmaml.pkpd import fit_nlme            # population engine
>>> from pharmaml.pkpd import P_GP_DDI_RECORDS    # the real DDI network
>>> len(P_GP_DDI_RECORDS)
34
>>> from pharmaml.pkpd import OSP_PBPK_LIBRARY
>>> len(OSP_PBPK_LIBRARY)
40
>>> from pharmaml.pkpd import ONTOGENY_TABLE       # PK-Sim's 508 points
>>> len(ONTOGENY_TABLE)
20
"""

from __future__ import annotations

from pharmaml.pkpd.compartments import TwoCompartment, bateman, superposition
from pharmaml.pkpd.compounds import (
    ATP1A2_PROFILE,
    CYP3A4_PROFILE,
    DIGOXIN_PARAMS,
    ITRACONZOLE_COMPOUNDS,
    ITRACONZOLE_WEIBULL,
    P_GP_PROFILE,
    enzyme_turnover_rate,
)
from pharmaml.pkpd.ddi import (
    P_GP_DDI_RECORDS,
    DDIRecord,
    chen_prusoff,
    competitive_inhibition_factor,
    ddi_classification,
    ddi_network_summary,
    gmfe,
    guest_et_al_limits,
    qualification_metrics,
    within_guest,
    within_twofold,
)
from pharmaml.pkpd.diagnostics import (
    bootstrap_summary,
    case_bootstrap_subjects,
    operating_characteristics,
    prediction_corrected_vpc,
    vpc_statistics,
)
from pharmaml.pkpd.events import (
    EventRow,
    RegimenSolution,
    addl_until,
    ev,
    ev_days,
    ev_seq,
    realize_addl,
    simulate_regimen,
)
from pharmaml.pkpd.nca_ext import (
    TerminalFit,
    auc_linear_log,
    c0_back_extrapolated,
    fit_terminal_phase,
    observed_lag_time,
    sparse_auc_bailer,
    steady_state_metrics,
    urine_metrics,
)
from pharmaml.pkpd.ode import DosingEvent, ODESolution, solve_ode
from pharmaml.pkpd.osplib import (
    DOCUMENTED_GMFE,
    OSP_PBPK_LIBRARY,
    OSP_SUITE_VERSION,
    QUALIFICATION_FRAMEWORK_VERSION,
    PBPKModelEntry,
    find_model,
    library_frame,
    library_summary,
)
from pharmaml.pkpd.pbpk import (
    ICRP_ADULT,
    MinimalPBPK,
    allometric_scale,
    first_in_human_dose,
    hepatic_ddi_auc_ratio,
    ivive_clint_invivo,
    mabel_dose,
    noael_hed,
    weibull_dissolution,
    well_stirred_clearance,
)
from pharmaml.pkpd.physiology import (
    ADULT_GFRSPEC,
    MOLECULE_DEFAULTS,
    ONTOGENY_TABLE,
    SPECIES_PHYSIOLOGY,
    TISSUE_COMPOSITION,
    TRANSPORTER_TYPES,
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
from pharmaml.pkpd.population import (
    NLMEFit,
    SubjectObservations,
    bsv_cv_percent,
    fit_nlme,
    simulate_population,
)
from pharmaml.pkpd.pregnancy import (
    PREGNANCY_MODELS,
    MaternalFetalParameters,
    MaternalFetalSolution,
    fertilization_age_weeks,
    pk_sim_age_years,
    pregnancy_model_frame,
    simulate_maternal_fetal,
)
from pharmaml.pkpd.sensitivity import (
    PK_PARAMETER_REGISTRY,
    SPECIES,
    HumanPopulation,
    IndividualCharacteristics,
    StandardPKParameter,
    create_individual_characteristics,
    match_paths,
    pk_parameter_frame,
    sensitivity_analysis,
)

__all__ = [
    "ADULT_GFRSPEC",
    "ATP1A2_PROFILE",
    "CYP3A4_PROFILE",
    "DIGOXIN_PARAMS",
    "DOCUMENTED_GMFE",
    "ICRP_ADULT",
    "ITRACONZOLE_COMPOUNDS",
    "ITRACONZOLE_WEIBULL",
    "MOLECULE_DEFAULTS",
    "ONTOGENY_TABLE",
    "OSP_PBPK_LIBRARY",
    "OSP_SUITE_VERSION",
    "PK_PARAMETER_REGISTRY",
    "PREGNANCY_MODELS",
    "P_GP_DDI_RECORDS",
    "P_GP_PROFILE",
    "QUALIFICATION_FRAMEWORK_VERSION",
    "SPECIES",
    "SPECIES_PHYSIOLOGY",
    "TISSUE_COMPOSITION",
    "TRANSPORTER_TYPES",
    "DDIRecord",
    "DosingEvent",
    "EventRow",
    "HumanPopulation",
    "IndividualCharacteristics",
    "MaternalFetalParameters",
    "MaternalFetalSolution",
    "MinimalPBPK",
    "NLMEFit",
    "ODESolution",
    "PBPKModelEntry",
    "RegimenSolution",
    "StandardPKParameter",
    "SubjectObservations",
    "TerminalFit",
    "TwoCompartment",
    "addl_until",
    "allometric_scale",
    "auc_linear_log",
    "bateman",
    "blood_plasma_ratio",
    "bootstrap_summary",
    "bsa_dubois",
    "bsa_mosteller",
    "bsv_cv_percent",
    "c0_back_extrapolated",
    "case_bootstrap_subjects",
    "chen_prusoff",
    "clint_to_clspec",
    "competitive_inhibition_factor",
    "create_individual",
    "create_individual_characteristics",
    "ddi_classification",
    "ddi_network_summary",
    "enzyme_turnover_rate",
    "ev",
    "ev_days",
    "ev_seq",
    "fertilization_age_weeks",
    "find_model",
    "first_in_human_dose",
    "fit_nlme",
    "fit_terminal_phase",
    "fu_mic",
    "fu_ontogeny_adjusted",
    "gfr_maturation_fraction",
    "gfr_reference",
    "gmfe",
    "guest_et_al_limits",
    "hepatic_ddi_auc_ratio",
    "ivive_clint_invivo",
    "kp_berezhkovskiy",
    "kp_pk_sim_standard",
    "kp_poulin_theil",
    "kp_rodgers_rowland",
    "kp_schmitt",
    "library_frame",
    "library_summary",
    "mabel_dose",
    "match_paths",
    "noael_hed",
    "observed_lag_time",
    "ontogeny_factor",
    "ontogeny_geo_sd",
    "operating_characteristics",
    "pk_parameter_frame",
    "pk_sim_age_years",
    "pma_years",
    "prediction_corrected_vpc",
    "pregnancy_model_frame",
    "qualification_metrics",
    "realize_addl",
    "sensitivity_analysis",
    "simulate_maternal_fetal",
    "simulate_population",
    "simulate_regimen",
    "solve_ode",
    "sparse_auc_bailer",
    "steady_state_metrics",
    "superposition",
    "urine_metrics",
    "vpc_statistics",
    "weibull_dissolution",
    "well_stirred_clearance",
    "within_guest",
    "within_twofold",
]
