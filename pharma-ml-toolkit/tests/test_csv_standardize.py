"""Tests for the CSV -> canonical SMILES standardization pipeline.

The round-192 absorption: five reference chains (RDKit MolStandardize,
MolVS, the ChEMBL structure pipeline, datamol, the REINVENT4 default)
behind one front door, the dimorphite_dl protonation enumerator, the
OpenBabel-slot InChI round-trip, and the CSV batch driver (the
auto-detected SMILES column, the no-silent-drop invalid rows, the
duplicate register, the report exports).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

rdkit = pytest.importorskip("rdkit")

from pharmaml.chem.csv_standardize import (
    CHEMBL_SALTS,
    CHEMBL_SOLVENTS,
    PROFILES,
    PROTONATION_SMARTS,
    StandardizeOptions,
    detect_smiles_column,
    inchi_roundtrip_check,
    protonate_smiles,
    standardize_csv,
    standardize_smiles,
)
from pharmaml.exceptions import DataValidationError

logging.disable(logging.CRITICAL)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_CANONICAL = "CC(=O)Oc1ccccc1C(=O)O"
ASPIRIN_SODIUM = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"


# ---------------------------------------------------------------------------
# the profile registry + the embedded tables
# ---------------------------------------------------------------------------


class TestRegistryAndTables:
    def test_six_profiles_registered(self) -> None:
        assert set(PROFILES) == {
            "rdkit", "molvs", "chembl", "datamol", "reinvent", "plain",
        }

    def test_every_profile_builds_its_reference_story(self) -> None:
        for key, profile in PROFILES.items():
            assert key == profile.key
            assert profile.label
            assert profile.reference
            assert callable(profile.builder)

    def test_chembl_tables_complete(self) -> None:
        assert len(CHEMBL_SALTS) == 163          # their salts.smi, verbatim
        assert len(CHEMBL_SOLVENTS) == 9         # their solvents.smi
        assert ("Sodium", "[Na]") in CHEMBL_SALTS
        assert ("WATER", "[OH2]") in CHEMBL_SOLVENTS

    def test_every_chembl_entry_parses(self) -> None:
        from rdkit import Chem

        for _name, smi in CHEMBL_SALTS + CHEMBL_SOLVENTS:
            assert Chem.MolFromSmarts(smi) is not None, smi

    def test_protonation_table_complete_and_parses(self) -> None:
        from rdkit import Chem

        assert len(PROTONATION_SMARTS) == 41     # site_substructures.smarts
        names = [name for name, _smarts, _states in PROTONATION_SMARTS]
        assert names[0] == "*Azide"
        assert names[-1] == "*Aromatic_nitrogen_protonated"
        import math

        for _name, smarts, states in PROTONATION_SMARTS:
            assert Chem.MolFromSmarts(smarts) is not None, smarts
            for site, mean, std in states:
                assert site >= 0
                assert not math.isnan(mean)  # no NaN smuggled in
                assert not math.isnan(std)
                assert std >= 0


# ---------------------------------------------------------------------------
# standardize_smiles -- the front door, profile by profile
# ---------------------------------------------------------------------------


class TestStandardizeSmiles:
    def test_rdkit_profile_collapses_the_sodium_salt(self) -> None:
        outcome = standardize_smiles("CC(=O)[O-].[Na+]")
        assert outcome.canonical == "CC(=O)O"
        assert outcome.status == "standardized"
        assert "fragments-removed" in outcome.flags
        assert "uncharged" in outcome.flags

    def test_molvs_profile_matches_the_original_order(self) -> None:
        outcome = standardize_smiles("CC(=O)[O-].[Na+]", profile="molvs")
        assert outcome.canonical == "CC(=O)O"
        # MolVS's own order: sanitize FIRST, then RemoveHs, metals, ...
        assert outcome.steps[:2] == ["SanitizeMol", "RemoveHs"]

    def test_chembl_profile_strips_the_drug_salt(self) -> None:
        outcome = standardize_smiles(ASPIRIN_SODIUM, profile="chembl")
        assert outcome.canonical == ASPIRIN_CANONICAL
        assert "FragmentParent(solvents+salts)" in outcome.steps

    def test_chembl_profile_normalizes_the_charged_nitro(self) -> None:
        # already charge-separated: the chain keeps it (the parser
        # normalises pentavalent N during sanitisation -- the honest
        # no-op, no flag)
        outcome = standardize_smiles("c1ccc(cc1)[N+](=O)[O-]", profile="chembl")
        assert outcome.canonical == "O=[N+]([O-])c1ccccc1"
        assert "normalized" not in outcome.flags

    def test_sulfoxide_normalization_is_flagged(self) -> None:
        # the neutral sulfoxide drawing is what the normalizer exists
        # for (the parser does not pre-fix it)
        for profile in ("rdkit", "chembl", "datamol"):
            outcome = standardize_smiles("CS(C)=O", profile=profile)
            assert outcome.canonical == "C[S+](C)[O-]", profile
            assert "normalized" in outcome.flags, profile

    def test_chembl_pure_salt_pair_keeps_everything(self) -> None:
        # both fragments are ChEMBL counter-ions -> their "keep
        # everything" fallback (the honest port of the rule)
        outcome = standardize_smiles("CC(=O)[O-].[Na+]", profile="chembl")
        assert outcome.canonical == "CC(=O)[O-].[Na+]"
        assert outcome.status == "unchanged"

    def test_chembl_excludes_metal_complexes(self) -> None:
        outcome = standardize_smiles("c1ccccc1c1ccccc1[Fe]", profile="chembl")
        assert "chembl-excluded" in outcome.flags

    def test_datamol_profile_defaults(self) -> None:
        outcome = standardize_smiles("c1ccccc1O", profile="datamol")
        assert outcome.canonical == "Oc1ccccc1"

    def test_datamol_uncharge_is_opt_in_like_the_library(self) -> None:
        # datamol's own library default: uncharge=False -> the
        # ammonium stands (the toolkit default ticks it on)
        library = standardize_smiles("[NH3+]CCC", profile="datamol", uncharge=False)
        assert library.canonical == "CCC[NH3+]"
        neutral = standardize_smiles("[NH3+]CCC", profile="datamol")
        assert neutral.canonical == "CCCN"
        assert "uncharged" in neutral.flags

    def test_reinvent_profile_rejects_outside_the_window(self) -> None:
        outcome = standardize_smiles("C", profile="reinvent")  # 1 heavy atom
        assert outcome.status == "rejected"
        assert outcome.canonical is None

    def test_reinvent_profile_collapses_the_salt(self) -> None:
        assert standardize_smiles("CC(=O)[O-].[Na+]", profile="reinvent").canonical == "CC(=O)O"

    def test_plain_profile_is_the_baseline(self) -> None:
        outcome = standardize_smiles("CC(=O)[O-].[Na+]", profile="plain")
        assert outcome.canonical == "CC(=O)[O-].[Na+]"  # nothing but canonical form

    def test_tautomer_leg_collapses_enol_to_keto(self) -> None:
        keto = standardize_smiles("CC(=O)CC(C)=O", tautomer=True)
        enol = standardize_smiles("CC(O)=CC(C)=O", tautomer=True)
        assert keto.canonical == enol.canonical == "CC(=O)CC(C)=O"
        assert "tautomer-canonicalised" in enol.flags

    def test_stereo_parent_when_stereo_off(self) -> None:
        outcome = standardize_smiles("C[C@H](O)C(=O)O", keep_stereo=False)
        assert outcome.canonical == "CC(O)C(=O)O"
        outcome_kept = standardize_smiles("C[C@H](O)C(=O)O")
        assert outcome_kept.canonical == "C[C@H](O)C(=O)O"

    def test_invalid_smiles_is_kept_with_its_error(self) -> None:
        outcome = standardize_smiles("not-a-smiles")
        assert outcome.status == "invalid"
        assert outcome.canonical is None
        assert "unparseable" in (outcome.error or "")

    def test_blank_smiles_is_invalid_not_a_crash(self) -> None:
        for blank in ("", "   ", None):
            outcome = standardize_smiles(blank)  # type: ignore[arg-type]
            assert outcome.status == "invalid"
            assert "blank" in (outcome.error or "")

    def test_unknown_profile_raises_the_typed_error(self) -> None:
        with pytest.raises(DataValidationError, match="Unknown standardization profile"):
            standardize_smiles("CCO", profile="no-such-profile")

    def test_protonation_states_ride_the_outcome(self) -> None:
        outcome = standardize_smiles("CC(=O)O", protonate=True, min_ph=11.0, max_ph=12.0)
        assert outcome.protonation_states == ["CC(=O)[O-]"]

    def test_protonation_ambiguity_flagged(self) -> None:
        outcome = standardize_smiles("NCC(=O)O", protonate=True)
        assert len(outcome.protonation_states) >= 2
        assert "protonation-ambiguous" in outcome.flags

    def test_options_dataclass_roundtrip(self) -> None:
        options = StandardizeOptions(profile="chembl", tautomer=True)
        restored = StandardizeOptions(**options.as_dict())
        assert restored == options


# ---------------------------------------------------------------------------
# the dimorphite_dl protonation enumerator
# ---------------------------------------------------------------------------


class TestProtonateSmiles:
    def test_acetic_acid_deprotonates_at_physiological_ph(self) -> None:
        assert protonate_smiles("CC(=O)O", min_ph=6.4, max_ph=8.4) == ["CC(=O)[O-]"]

    def test_acetic_acid_stays_protonated_below_its_pka(self) -> None:
        assert protonate_smiles("CC(=O)O", min_ph=1.0, max_ph=2.0) == ["CC(=O)O"]

    def test_glycine_amine_is_ambiguous_at_physiological_ph(self) -> None:
        states = protonate_smiles("NCC(=O)O", min_ph=6.4, max_ph=8.4)
        assert "NCC(=O)[O-]" in states
        assert "[NH3+]CC(=O)[O-]" in states

    def test_imidazole_enumerates_both_ring_nitrogens(self) -> None:
        states = protonate_smiles("c1cnc[nH]1", min_ph=6.4, max_ph=8.4)
        assert 2 <= len(states) <= 4
        assert all("c1cn" in s or "c1nc" in s or s.startswith("c1") for s in states)

    def test_no_site_molecule_returns_its_own_canonical_form(self) -> None:
        assert protonate_smiles("CCO") == ["CCO"]

    def test_unparseable_returns_empty(self) -> None:
        assert protonate_smiles("not-a-smiles") == []

    def test_neutralisation_fixpoint_repairs_bad_azide(self) -> None:
        # R-N-N#N (the bad azide) is fixed to R-N=[N+]=N-H by the
        # neutralisation reactions before the rules read the molecule
        states = protonate_smiles("CN=N#N", min_ph=6.4, max_ph=8.4)
        assert states and all("[N+]" in s for s in states)

    def test_ph_window_respected_by_the_strong_acid(self) -> None:
        # the protonated nitro group: pKa -1000 -> deprotonated at
        # every window (the rule reads the -O-H form)
        for window in ((6.4, 8.4), (-2.0, -1.0)):
            states = protonate_smiles("C[N+](=O)O", min_ph=window[0], max_ph=window[1])
            assert states == ["C[N+](=O)[O-]"], window


# ---------------------------------------------------------------------------
# the OpenBabel slot -- the InChI round-trip
# ---------------------------------------------------------------------------


class TestInchiRoundTrip:
    def test_simple_molecules_agree_with_themselves(self) -> None:
        for smiles, canonical in (("CCO", "CCO"), ("c1ccccc1O", "Oc1ccccc1")):
            agreed, other = inchi_roundtrip_check(smiles)
            assert agreed
            assert other == canonical

    def test_unparseable_fails_cleanly(self) -> None:
        agreed, other = inchi_roundtrip_check("not-a-smiles")
        assert not agreed
        assert other == ""

    def test_salt_keeps_its_canonical_form(self) -> None:
        agreed, other = inchi_roundtrip_check("CC(=O)[O-].[Na+]")
        assert other == "CC(=O)[O-].[Na+]"
        assert agreed


# ---------------------------------------------------------------------------
# the SMILES-column auto-detection
# ---------------------------------------------------------------------------


class TestDetectSmilesColumn:
    def test_name_match_wins(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"name": ["x"], "smiles": ["CCO"]})
        assert detect_smiles_column(frame) == "smiles"

    def test_structure_and_smi_headers_match(self) -> None:
        import pandas as pd

        assert detect_smiles_column(pd.DataFrame({"structure": ["CCO"]})) == "structure"
        assert detect_smiles_column(pd.DataFrame({"canonical_SMILES": ["CCO"]})) == "canonical_SMILES"
        assert detect_smiles_column(pd.DataFrame({"mol": ["CCO"]})) == "mol"

    def test_parse_rate_fallback_when_no_name_matches(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"compound": ["CCO", "CCN", "CCC"], "junk": ["a", "b", "c"]})
        assert detect_smiles_column(frame) == "compound"

    def test_no_candidates_raise_the_named_error(self) -> None:
        import pandas as pd

        with pytest.raises(DataValidationError, match="No SMILES column detected"):
            detect_smiles_column(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    def test_low_parse_rate_column_is_rejected(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({"notes": ["hello", "world", "text"]})
        with pytest.raises(DataValidationError, match="No SMILES column detected"):
            detect_smiles_column(frame)


# ---------------------------------------------------------------------------
# the CSV batch driver
# ---------------------------------------------------------------------------


class TestStandardizeCsv:
    def test_demo_corpus_report(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {
                "name": [
                    "aspirin", "sodium salt", "bad",
                ],
                "SMILES": [ASPIRIN, "CC(=O)[O-].[Na+]", "not-a-smiles"],
            }
        )
        report = standardize_csv(frame, profile="rdkit", id_column="name")
        assert report.smiles_column == "SMILES"
        assert [r.identifier for r in report.records] == [
            "aspirin", "sodium salt", "bad",
        ]
        assert report.records[0].canonical == ASPIRIN_CANONICAL
        assert report.records[1].canonical == "CC(=O)O"
        assert report.records[2].status == "invalid"
        assert report.records[2].canonical is None

    def test_summary_counts(self) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame({"SMILES": [ASPIRIN, "CCO", "CCO", "bad", "CC(=O)[O-].[Na+]"]})
        )
        summary = report.summary()
        assert summary["rows"] == 5
        assert summary["standardized"] == 2
        assert summary["unchanged"] == 2   # both CCO rows are already canonical
        assert summary["invalid"] == 1
        assert summary["duplicate-rows"] == 1

    def test_invalid_rows_carry_their_errors_no_silent_drops(self) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame({"SMILES": ["CCO", "not-a-smiles", "CC(=O)[O-]"]})
        )
        invalid = [r for r in report.records if r.status == "invalid"]
        assert len(invalid) == 1
        assert "unparseable" in (invalid[0].error or "")
        assert len(report.records) == 3  # every input row answered

    def test_duplicate_register_flags_the_second_row(self) -> None:
        import pandas as pd

        report = standardize_csv(pd.DataFrame({"SMILES": ["CCO", "CCO", "CCO"]}))
        assert report.records[0].flags == []
        assert "duplicate-of-row-1" in report.records[1].flags
        assert "duplicate-of-row-1" in report.records[2].flags

    def test_to_frame_schema(self) -> None:
        import pandas as pd

        report = standardize_csv(pd.DataFrame({"SMILES": ["CCO"]}))
        frame = report.to_frame()
        assert list(frame.columns) == [
            "identifier", "original", "canonical", "status",
            "steps", "flags", "protonation_states", "error",
        ]
        assert frame.shape == (1, 8)
        assert frame.iloc[0]["canonical"] == "CCO"

    def test_write_csv_roundtrip(self, tmp_path: Path) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame({"SMILES": [ASPIRIN, "CC(=O)[O-].[Na+]"]})
        )
        target = tmp_path / "canonical.csv"
        path = report.write_csv(target)
        assert path == target and target.exists()
        reloaded = pd.read_csv(target)
        assert list(reloaded["canonical"]) == [ASPIRIN_CANONICAL, "CC(=O)O"]

    def test_explicit_smiles_column(self) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame(
                {"structure": ["CC(=O)[O-].[Na+]"], "note": ["the sodium salt"]}
            ),
            smiles_column="structure",
        )
        assert report.smiles_column == "structure"
        assert report.records[0].canonical == "CC(=O)O"

    def test_missing_smiles_column_raises(self) -> None:
        import pandas as pd

        with pytest.raises(DataValidationError, match="not found in the CSV"):
            standardize_csv(pd.DataFrame({"SMILES": ["CCO"]}), smiles_column="absent")

    def test_missing_id_column_raises(self) -> None:
        import pandas as pd

        with pytest.raises(DataValidationError, match="Identifier column"):
            standardize_csv(pd.DataFrame({"SMILES": ["CCO"]}), id_column="absent")

    def test_unknown_profile_raises(self) -> None:
        import pandas as pd

        with pytest.raises(DataValidationError, match="Unknown standardization profile"):
            standardize_csv(pd.DataFrame({"SMILES": ["CCO"]}), profile="nope")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(DataValidationError, match="CSV file not found"):
            standardize_csv("/no/such/file.csv")

    def test_path_input_with_tab_separator(self, tmp_path: Path) -> None:
        target = tmp_path / "corpus.tsv"
        target.write_text("SMILES\tname\nCC(=O)[O-].[Na+]\tacetate\n")
        report = standardize_csv(target)
        assert report.smiles_column == "SMILES"
        assert report.records[0].canonical == "CC(=O)O"

    def test_path_input_with_semicolons(self, tmp_path: Path) -> None:
        target = tmp_path / "corpus.csv"
        target.write_text("SMILES;name\nCCO;ethanol\n")
        report = standardize_csv(target)
        assert report.records[0].canonical == "CCO"

    def test_profile_rides_the_report_provenance(self) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame({"SMILES": [ASPIRIN_SODIUM]}), profile="chembl"
        )
        assert report.profile == "chembl"
        assert report.options["profile"] == "chembl"
        assert report.records[0].canonical == ASPIRIN_CANONICAL

    def test_protonation_states_in_the_report(self) -> None:
        import pandas as pd

        report = standardize_csv(
            pd.DataFrame({"SMILES": ["CC(=O)O", "NCC(=O)O"]}),
            protonate=True,
            min_ph=6.4,
            max_ph=8.4,
        )
        assert report.records[0].protonation_states == ["CC(=O)[O-]"]
        assert len(report.records[1].protonation_states) >= 2
        assert report.summary()["protonation-ambiguous"] >= 1
        states_text = report.to_frame().iloc[1]["protonation_states"]
        assert "NCC(=O)[O-]" in states_text and "[NH3+]" in states_text

    def test_all_six_profiles_run_the_demo_corpus(self) -> None:
        import pandas as pd

        frame = pd.DataFrame(
            {"SMILES": [ASPIRIN, "CC(=O)[O-].[Na+]", "CN(C)C.Cl", "c1ccccc1O"]}
        )
        for profile in PROFILES:
            report = standardize_csv(frame, profile=profile)
            assert report.profile == profile
            assert report.summary()["invalid"] == 0
            assert all(r.canonical for r in report.records)
