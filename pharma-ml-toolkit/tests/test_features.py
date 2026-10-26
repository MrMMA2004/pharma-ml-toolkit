"""Tests for :mod:`pharmaml.chem.features` — the multi-family feature factory.

Round 100x-47-A fail-first suite, written against the binding API contract
in ``worklog.md`` (Task ID: 100x-47) BEFORE the module existed: the first
run against the pristine tree failed at import time
(``ModuleNotFoundError: No module named 'pharmaml.chem.features'`` — the
witnessed red output is recorded in the agent's worklog entry).

The suite pins:

* the 7-family vocabulary, prefixes and bit-size constants,
* per-family column counts, canonical ordering and dropped-row accounting
  (including the > 10,000-column full run over the shipped
  ``examples/data/ligand_library.csv``),
* value-level agreement with the RDKit reference calls (atom-pair sparse
  count folding, Avalon count, ECFP/Morgan bits, Murcko scaffold),
* ranking: variance constants, random-forest informative-first regression,
  task inference (string target / 2 unique values), target guards and the
  optional-lightgbm fallback,
* the two-layer selection contract (per-family median prefilter, stable
  tie-break on frame column order, k edges) and the JSON-ready metadata.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit")

from pharmaml.chem.features import (
    ATOMPAIR_BITS,
    AVALON_BITS,
    ECFP3_BITS,
    ECFP3_RADIUS,
    ECFP_BITS,
    FAMILY_PREFIXES,
    FEATURE_FAMILIES,
    MORGAN_BITS,
    MORGAN_RADIUS,
    RDKIT_FP_BITS,
    RISM_DESCRIPTOR_COUNT,
    FeatureReport,
    FeatureSelection,
    compute_feature_frame,
    family_of_column,
    rank_features,
    select_top_features,
    selection_metadata,
)

try:  # LightGBM is an OPTIONAL ranker — the fallback path must not need it.
    import lightgbm  # noqa: F401

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - exercised only without lightgbm
    HAS_LIGHTGBM = False

requires_lightgbm = pytest.mark.skipif(
    not HAS_LIGHTGBM, reason="lightgbm is not installed (optional ranker)"
)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_CSV = PACKAGE_ROOT / "examples" / "data" / "ligand_library.csv"
FEATURES_SOURCE = PACKAGE_ROOT / "src" / "pharmaml" / "chem" / "features.py"

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"

#: The 10-row shipped library: 8 valid drug-like molecules followed by two
#: deliberately broken rows (``not_a_smiles`` and an empty cell -> NaN in
#: pandas, normalised to "" here so the recorded drop matches).
VALID_LIBRARY_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "CCOC(=O)c1ccc(N)cc1",
    "O=C1Oc2ccccc2C=C1",
    "COc1cc(C=O)ccc1O",
]


def library_smiles() -> list[str]:
    """SMILES column of the shipped ligand library, NaN -> ''."""
    frame = pd.read_csv(LIBRARY_CSV, dtype={"SMILES": str})
    return ["" if pd.isna(value) else str(value) for value in frame["SMILES"]]


@pytest.fixture(scope="module")
def full_library() -> tuple[pd.DataFrame, FeatureReport]:
    """All 8 families over the shipped 10-row library (8 valid molecules)."""
    return compute_feature_frame(library_smiles())


def _regression_case(n: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic frame with a KNOWN informative column and a noise-free target.

    ``y = 3 * f_informative`` exactly, so any sane importance ranker must put
    ``f_informative`` first and the constant column last (importance 0).
    """
    rng = np.random.default_rng(42)
    informative = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "f_informative": informative,
            "f_noise": rng.normal(size=n),
            "f_constant": np.full(n, 7.0),
        }
    )
    return frame, 3.0 * informative


def _spy_estimators(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Record which sklearn estimator class rank_features instantiates."""
    from sklearn import ensemble

    used: dict[str, int] = {}
    real_classifier = ensemble.RandomForestClassifier
    real_regressor = ensemble.RandomForestRegressor

    def spy_classifier(*args: Any, **kwargs: Any) -> Any:
        used["classifier"] = used.get("classifier", 0) + 1
        return real_classifier(*args, **kwargs)

    def spy_regressor(*args: Any, **kwargs: Any) -> Any:
        used["regressor"] = used.get("regressor", 0) + 1
        return real_regressor(*args, **kwargs)

    monkeypatch.setattr(ensemble, "RandomForestClassifier", spy_classifier)
    monkeypatch.setattr(ensemble, "RandomForestRegressor", spy_regressor)
    return used


# ---------------------------------------------------------------------------
# Module contract: vocabulary, prefixes, constants, import-path purity
# ---------------------------------------------------------------------------
class TestModuleContract:
    """The published constants must match the binding API contract exactly."""

    def test_feature_families_canonical_order(self) -> None:
        assert FEATURE_FAMILIES == (
            "descriptors",
            "atompair_count",
            "ecfp",
            "ecfp3",
            "avalon_count",
            "rdkit_fingerprint",
            "morgan_full",
            "morgan_scaffold",
            "maccs",
            "fcfp",
            "fcfp_count",
            "topological",
            "topological_count",
            "rdkit_count",
            "pattern",
            "layered",
            "estate",
            "erg",
            "shape_3d",
            "rism",
        )

    def test_family_prefixes_mapping(self) -> None:
        assert FAMILY_PREFIXES == {
            "descriptors": "desc_",
            "atompair_count": "apc_bit_",
            "ecfp": "ecfp_bit_",
            "ecfp3": "ecfp3_bit_",
            "avalon_count": "avalon_count_",
            "rdkit_fingerprint": "rdkit_bit_",
            "morgan_full": "morgan_full_bit_",
            "morgan_scaffold": "morgan_scaffold_bit_",
            "maccs": "maccs_bit_",
            "fcfp": "fcfp_bit_",
            "fcfp_count": "fcfp_count_",
            "topological": "topo_bit_",
            "topological_count": "topo_count_",
            "rdkit_count": "rdkit_count_",
            "pattern": "pattern_bit_",
            "layered": "layered_bit_",
            "estate": "estate_bit_",
            "erg": "erg_",
            "shape_3d": "d3d_",
            "rism": "rism_",
        }
        assert list(FAMILY_PREFIXES) == list(FEATURE_FAMILIES)

    def test_bit_size_constants(self) -> None:
        assert (
            ATOMPAIR_BITS,
            ECFP_BITS,
            ECFP3_BITS,
            ECFP3_RADIUS,
            AVALON_BITS,
            RDKIT_FP_BITS,
            MORGAN_BITS,
            MORGAN_RADIUS,
        ) == (2048, 2048, 2048, 3, 512, 2048, 2048, 2)

    def test_module_import_path_is_lightweight(self) -> None:
        """Module level may import only rdkit/numpy/pandas (plus the package).

        lightgbm, datamol, mordred AND sklearn must stay lazy: the variance
        path must work without them, and lightgbm must be strictly optional
        (contract: "imported lazily inside rank_features").
        """
        tree = ast.parse(FEATURES_SOURCE.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in tree.body:  # top-level statements only
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        banned = {"lightgbm", "datamol", "mordred", "sklearn"}
        for name in imported:
            assert name.split(".")[0] not in banned, (
                f"features.py imports {name!r} at module level; only "
                "rdkit/numpy/pandas/pharmaml belong on the import path"
            )

    def test_chem_package_exports_the_feature_factory(self) -> None:
        from pharmaml import chem

        for name in (
            "ATOMPAIR_BITS",
            "AVALON_BITS",
            "ECFP_BITS",
            "FEATURE_FAMILIES",
            "FAMILY_PREFIXES",
            "MORGAN_BITS",
            "MORGAN_RADIUS",
            "RDKIT_FP_BITS",
            "RISM_DESCRIPTOR_COUNT",
            "FeatureReport",
            "FeatureSelection",
            "compute_feature_frame",
            "compute_rism_descriptors",
            "family_of_column",
            "rank_features",
            "select_top_features",
            "selection_metadata",
        ):
            assert name in chem.__all__, f"{name} missing from pharmaml.chem.__all__"
        assert chem.compute_feature_frame is compute_feature_frame
        assert chem.FeatureReport is FeatureReport


# ---------------------------------------------------------------------------
# compute_feature_frame
# ---------------------------------------------------------------------------
class TestComputeFeatureFrame:
    """Family computation, column layout and dropped-row accounting."""

    def test_descriptors_family_only(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors

        names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        frame, report = compute_feature_frame([ETHANOL], families=("descriptors",))
        assert frame.shape == (1, len(names))
        assert list(frame.columns) == [f"desc_{name}" for name in names]
        assert report.families == {"descriptors": len(names)}
        assert report.total_columns == len(names)
        assert report.n_valid == 1
        assert report.smiles == [ETHANOL]
        assert report.dropped == []
        reference = dict(
            zip(names, rdMolDescriptors.Properties(names).ComputeProperties(
                Chem.MolFromSmiles(ETHANOL)
            ))
        )
        assert frame["desc_amw"].iloc[0] == pytest.approx(reference["amw"])
        assert frame["desc_lipinskiHBA"].iloc[0] == pytest.approx(reference["lipinskiHBA"])
        assert frame.to_numpy().dtype == np.float64

    def test_fingerprint_family_sizes_and_prefixes(self) -> None:
        frame, report = compute_feature_frame(
            [ETHANOL], families=("ecfp", "avalon_count")
        )
        assert frame.shape == (1, ECFP_BITS + AVALON_BITS)
        assert report.families == {"ecfp": ECFP_BITS, "avalon_count": AVALON_BITS}
        assert report.total_columns == ECFP_BITS + AVALON_BITS
        assert frame.columns[0] == "ecfp_bit_0"
        assert frame.columns[ECFP_BITS - 1] == "ecfp_bit_2047"
        assert frame.columns[ECFP_BITS] == "avalon_count_0"
        assert frame.columns[-1] == "avalon_count_511"
        assert frame.iloc[0, :ECFP_BITS].sum() > 0  # ethanol has radius-2 bits
        assert frame.iloc[0, ECFP_BITS:].sum() > 0

    def test_families_output_in_canonical_order(self) -> None:
        frame, report = compute_feature_frame(
            [ETHANOL], families=("avalon_count", "ecfp", "descriptors")
        )
        assert list(report.families) == ["descriptors", "ecfp", "avalon_count"]
        n_desc = report.families["descriptors"]
        assert frame.columns[0].startswith("desc_")
        assert frame.columns[n_desc] == "ecfp_bit_0"
        assert frame.columns[-1] == "avalon_count_511"

    def test_duplicate_families_are_deduplicated(self) -> None:
        frame, report = compute_feature_frame(
            [ETHANOL], families=("ecfp", "ecfp", "descriptors", "ecfp")
        )
        assert list(report.families) == ["descriptors", "ecfp"]  # canonical, deduped
        assert report.families["ecfp"] == ECFP_BITS
        assert frame.shape[1] == report.total_columns

    def test_dropped_rows_recorded_with_original_index(self) -> None:
        smiles = [ETHANOL, "not_a_smiles", "CCC", ""]
        frame, report = compute_feature_frame(smiles, families=("ecfp",))
        assert report.n_valid == 2
        assert report.smiles == [ETHANOL, "CCC"]
        assert report.dropped == [(1, "not_a_smiles"), (3, "")]
        assert frame.shape == (2, ECFP_BITS)
        assert frame.index.tolist() == [0, 2]  # original row positions

    def test_all_invalid_rows_return_empty_frame_with_full_report(self) -> None:
        frame, report = compute_feature_frame(
            ["not_a_smiles", ""], families=("ecfp",)
        )
        assert frame.shape == (0, ECFP_BITS)
        assert report.n_valid == 0
        assert report.smiles == []
        assert report.dropped == [(0, "not_a_smiles"), (1, "")]
        assert report.families == {"ecfp": ECFP_BITS}

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown feature family") as excinfo:
            compute_feature_frame([ETHANOL], families=("ecfp", "typo"))
        message = str(excinfo.value)
        for family in FEATURE_FAMILIES:
            assert family in message  # the error lists every valid family

    def test_empty_families_raises(self) -> None:
        with pytest.raises(ValueError, match="families is empty"):
            compute_feature_frame([ETHANOL], families=())

    def test_empty_smiles_raises(self) -> None:
        with pytest.raises(ValueError, match="smiles is empty"):
            compute_feature_frame([], families=("ecfp",))

    def test_bare_string_smiles_raises(self) -> None:
        with pytest.raises(ValueError, match="single string"):
            compute_feature_frame("CCO", families=("ecfp",))


class TestFamilyValues:
    """Value-level agreement with the underlying RDKit reference calls."""

    def test_atompair_fold_matches_sparse_counts(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        frame, _ = compute_feature_frame([ASPIRIN], families=("atompair_count",))
        assert frame.shape == (1, ATOMPAIR_BITS)
        row = frame.iloc[0].to_numpy()
        elements = (
            rdFingerprintGenerator.GetAtomPairGenerator()
            .GetSparseCountFingerprint(Chem.MolFromSmiles(ASPIRIN))
            .GetNonzeroElements()
        )
        expected = np.zeros(ATOMPAIR_BITS)
        for idx, count in elements.items():
            expected[idx % ATOMPAIR_BITS] += count
        assert np.array_equal(row, expected)
        assert row.sum() == sum(elements.values())  # count semantics survive folding
        assert (row > 1).any()  # folded counts, not binary bits

    def test_avalon_count_matches_reference(self) -> None:
        from rdkit import Chem
        from rdkit.Avalon import pyAvalonTools

        frame, _ = compute_feature_frame([ASPIRIN], families=("avalon_count",))
        assert frame.shape == (1, AVALON_BITS)
        row = frame.iloc[0].to_numpy()
        reference = pyAvalonTools.GetAvalonCountFP(
            Chem.MolFromSmiles(ASPIRIN), nBits=AVALON_BITS
        ).GetNonzeroElements()
        assert row.sum() == sum(reference.values())
        for idx, count in reference.items():
            assert row[idx] == count
        assert frame.columns[0] == "avalon_count_0"
        assert frame.columns[-1] == "avalon_count_511"

    def test_ecfp_bits_match_reference(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        frame, _ = compute_feature_frame([ETHANOL], families=("ecfp",))
        row = frame.iloc[0].to_numpy()
        on_bits = set(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(ETHANOL), radius=MORGAN_RADIUS, nBits=ECFP_BITS
            ).GetOnBits()
        )
        assert row.sum() == len(on_bits)
        assert set(np.nonzero(row)[0]) == on_bits

    def test_morgan_full_bits_match_reference(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        frame, _ = compute_feature_frame([ETHANOL], families=("morgan_full",))
        row = frame.iloc[0].to_numpy()
        on_bits = set(
            AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(ETHANOL), radius=MORGAN_RADIUS, nBits=MORGAN_BITS
            ).GetOnBits()
        )
        assert set(np.nonzero(row)[0]) == on_bits

    def test_rdkit_fingerprint_bits_match_reference(self) -> None:
        from rdkit import Chem

        frame, _ = compute_feature_frame([ASPIRIN], families=("rdkit_fingerprint",))
        row = frame.iloc[0].to_numpy()
        on_bits = set(
            Chem.RDKFingerprint(
                Chem.MolFromSmiles(ASPIRIN), fpSize=RDKIT_FP_BITS
            ).GetOnBits()
        )
        assert row.sum() == len(on_bits)
        assert set(np.nonzero(row)[0]) == on_bits

    def test_morgan_scaffold_acyclic_is_zero_row_not_dropped(self) -> None:
        frame, report = compute_feature_frame(
            ["CCCCCCCC"], families=("morgan_scaffold",)
        )
        assert report.n_valid == 1
        assert report.dropped == []
        assert frame.shape == (1, MORGAN_BITS)
        assert frame.iloc[0].sum() == 0.0  # no scaffold -> zero row, kept

    def test_morgan_scaffold_aspirin_is_benzene(self) -> None:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        frame, report = compute_feature_frame(
            [ASPIRIN], families=("morgan_scaffold",)
        )
        assert report.n_valid == 1
        scaffold = MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(ASPIRIN))
        assert Chem.MolToSmiles(scaffold) == "c1ccccc1"
        on_bits = set(
            AllChem.GetMorganFingerprintAsBitVect(
                scaffold, radius=MORGAN_RADIUS, nBits=MORGAN_BITS
            ).GetOnBits()
        )
        row = frame.iloc[0].to_numpy()
        assert set(np.nonzero(row)[0]) == on_bits
        assert row.sum() > 0


# ---------------------------------------------------------------------------
# The full 9-family run over the shipped ligand library
# ---------------------------------------------------------------------------
class TestEcfp3Family:
    """The radius-3 family (100x-145, ``Ml/D2.py``'s R3 half).

    Pins the properties that make ecfp3 a genuine new family rather
    than a renamed ecfp: the wider environments radius 3 sees, the
    2048-bit width, and the round-trip through ``family_of_column``.
    """

    def test_radius3_sees_more_environments_than_radius2(self) -> None:
        """Aspirin's radius-3 neighbourhoods outnumber its radius-2 ones."""
        frame, report = compute_feature_frame(
            ["CC(=O)Oc1ccccc1C(=O)O"], families=("ecfp", "ecfp3")
        )
        assert report.families == {"ecfp": 2048, "ecfp3": 2048}
        r2_bits = int(frame.iloc[0, :2048].to_numpy().sum())
        r3_bits = int(frame.iloc[0, 2048:].to_numpy().sum())
        assert r3_bits > r2_bits > 0
        # And they are genuinely DIFFERENT vectors, not a reordering.
        assert (frame.iloc[0, :2048].to_numpy() != frame.iloc[0, 2048:].to_numpy()).any()

    def test_column_prefixes_and_round_trip(self) -> None:
        frame, _report = compute_feature_frame(["CCO"], families=("ecfp3",))
        assert frame.shape == (1, 2048)
        assert frame.columns[0] == "ecfp3_bit_0"
        assert frame.columns[-1] == "ecfp3_bit_2047"
        assert family_of_column("ecfp3_bit_123") == "ecfp3"
        # No prefix collision: ecfp_bit_ vs ecfp3_bit_ resolve exactly.
        assert family_of_column("ecfp_bit_123") == "ecfp"

    def test_default_run_includes_ecfp3(self) -> None:
        """A bare compute_feature_frame() call computes the new family too."""
        frame, report = compute_feature_frame(["CCO"])
        assert "ecfp3" in report.families
        assert report.families["ecfp3"] == ECFP3_BITS
        assert "ecfp3_bit_0" in frame.columns

    def test_canonical_order_puts_ecfp3_after_ecfp(self) -> None:
        assert FEATURE_FAMILIES.index("ecfp3") == FEATURE_FAMILIES.index("ecfp") + 1
        assert ECFP3_RADIUS == 3


class TestFullFamilyRun:
    """The 10,000+-column contract over examples/data/ligand_library.csv."""

    def test_family_counts_and_total_columns(self, full_library: tuple[
        pd.DataFrame, FeatureReport
    ]) -> None:
        from rdkit.Chem import rdMolDescriptors

        frame, report = full_library
        n_props = len(rdMolDescriptors.Properties.GetAvailableProperties())
        assert report.families == {
            "descriptors": n_props,
            "atompair_count": 2048,
            "ecfp": 2048,
            "ecfp3": 2048,
            "avalon_count": 512,
            "rdkit_fingerprint": 2048,
            "morgan_full": 2048,
            "morgan_scaffold": 2048,
            "maccs": 167,
            "fcfp": 2048,
            "fcfp_count": 2048,
            "topological": 2048,
            "topological_count": 2048,
            "rdkit_count": 2048,
            "pattern": 2048,
            "layered": 2048,
            "estate": 79,
            "erg": 315,
            "shape_3d": 9,
            "rism": RISM_DESCRIPTOR_COUNT,
        }
        assert list(report.families) == list(FEATURE_FAMILIES)  # canonical order
        # 12,800 (the original seven blocks incl. avalon) + the ten
        # fak_ml families (7 x 2048 + 167 + 79 + 315) + shape_3d's 9.
        assert report.total_columns == (
            12_800 + 7 * 2048 + 167 + 79 + 315 + 9 + RISM_DESCRIPTOR_COUNT + n_props
        )
        assert report.total_columns > 10_000  # the headline contract
        assert report.total_columns == sum(report.families.values())
        assert frame.shape == (8, report.total_columns)
        assert len(frame.columns) == report.total_columns

    def test_column_blocks_follow_canonical_family_order(
        self, full_library: tuple[pd.DataFrame, FeatureReport]
    ) -> None:
        frame, report = full_library
        start = 0
        for family in FEATURE_FAMILIES:
            width = report.families[family]
            block = frame.columns[start : start + width]
            prefix = FAMILY_PREFIXES[family]
            assert all(column.startswith(prefix) for column in block)
            start += width
        assert start == frame.shape[1]
        assert frame.columns[0].startswith("desc_")
        assert frame.columns[-1] == "rism_q_neg_frac"  # the rism family closes the frame

    def test_dropped_and_valid_accounting(
        self, full_library: tuple[pd.DataFrame, FeatureReport]
    ) -> None:
        frame, report = full_library
        assert report.dropped == [(8, "not_a_smiles"), (9, "")]
        assert report.n_valid == 8
        assert report.smiles == VALID_LIBRARY_SMILES
        assert frame.index.tolist() == list(range(8))  # the two bad rows are last

    def test_values_are_floats_and_informative(
        self, full_library: tuple[pd.DataFrame, FeatureReport]
    ) -> None:
        frame, _report = full_library
        values = frame.to_numpy()
        assert values.dtype == np.float64
        assert np.isfinite(values).all()
        assert (values != 0).any(axis=1).all()  # every molecule has real features
        assert (values != 0).any(axis=0).sum() > 100  # many informative columns


# ---------------------------------------------------------------------------
# family_of_column
# ---------------------------------------------------------------------------
class TestFamilyOfColumn:
    """The inverse prefix map and its strictness."""

    @pytest.mark.parametrize("family", FEATURE_FAMILIES)
    def test_prefix_roundtrip(self, family: str) -> None:
        assert family_of_column(FAMILY_PREFIXES[family] + "17") == family

    def test_real_column_names(self) -> None:
        assert family_of_column("desc_exactmw") == "descriptors"
        assert family_of_column("apc_bit_2047") == "atompair_count"
        assert family_of_column("ecfp_bit_0") == "ecfp"
        assert family_of_column("avalon_count_511") == "avalon_count"
        assert family_of_column("rdkit_bit_0") == "rdkit_fingerprint"
        assert family_of_column("morgan_full_bit_42") == "morgan_full"
        assert family_of_column("morgan_scaffold_bit_7") == "morgan_scaffold"

    @pytest.mark.parametrize(
        "column", ["SMILES", "", "desc", "nonsense_bit_1", "DESC_MolWt"]
    )
    def test_unknown_prefix_raises(self, column: str) -> None:
        with pytest.raises(ValueError, match="unknown feature-column prefix"):
            family_of_column(column)


# ---------------------------------------------------------------------------
# rank_features
# ---------------------------------------------------------------------------
class TestRankFeaturesVariance:
    """method='variance' / 'auto' without a target."""

    def test_constant_zero_informative_positive(self) -> None:
        frame = pd.DataFrame(
            {
                "f_informative": [0, 1, 2, 3, 4],
                "f_noise": [1, -1, 1, -1, 1],
                "f_constant": [5, 5, 5, 5, 5],
            }
        )
        scores = rank_features(frame)
        assert scores.index.tolist() == ["f_informative", "f_noise", "f_constant"]
        assert float(scores["f_informative"]) == pytest.approx(2.5)
        assert float(scores["f_noise"]) == pytest.approx(1.2)
        assert float(scores["f_constant"]) == 0.0
        assert scores.is_monotonic_decreasing
        assert set(scores.index) == set(frame.columns)

    def test_auto_without_target_matches_variance(self) -> None:
        frame, _ = _regression_case()
        pd.testing.assert_series_equal(
            rank_features(frame), rank_features(frame, method="variance")
        )

    def test_frame_nan_values_are_filled_for_ranking(self) -> None:
        frame = pd.DataFrame(
            {
                "f_partial": [1.0, np.nan, 3.0, 4.0],
                "f_all_nan": [np.nan] * 4,
                "f_ok": [1.0, 2.0, 3.0, 4.0],
            }
        )
        scores = rank_features(frame)
        assert float(scores["f_all_nan"]) == 0.0
        assert float(scores["f_partial"]) == pytest.approx(10.0 / 3.0)
        assert float(scores["f_ok"]) == pytest.approx(5.0 / 3.0)
        assert scores.index.tolist() == ["f_partial", "f_ok", "f_all_nan"]


class TestRankFeaturesModels:
    """Model-based ranking on a synthetic frame with a known answer."""

    def test_random_forest_regression_informative_first(self) -> None:
        frame, y = _regression_case()
        scores = rank_features(frame, y, method="random_forest")
        assert scores.index[0] == "f_informative"
        assert float(scores["f_constant"]) == 0.0
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.is_monotonic_decreasing
        assert set(scores.index) == set(frame.columns)

    def test_classification_from_string_target(self) -> None:
        # A regression estimator cannot fit string labels, so a clean run
        # proves the classification task was inferred from the target.
        frame, y = _regression_case()
        labels = pd.Series(np.where(y > 0, "high", "low"))
        scores = rank_features(frame, labels, method="random_forest")
        assert scores.index[0] == "f_informative"
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.is_monotonic_decreasing

    def test_two_unique_numeric_target_uses_classifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        used = _spy_estimators(monkeypatch)
        frame, y = _regression_case()
        labels = pd.Series((y > 0).astype(int))
        rank_features(frame, labels, method="random_forest")
        assert used.get("classifier", 0) == 1
        assert used.get("regressor", 0) == 0

    def test_many_unique_numeric_target_uses_regressor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        used = _spy_estimators(monkeypatch)
        frame, y = _regression_case()
        rank_features(frame, pd.Series(y), method="random_forest")
        assert used.get("regressor", 0) == 1
        assert used.get("classifier", 0) == 0

    def test_explicit_task_overrides_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        used = _spy_estimators(monkeypatch)
        frame, y = _regression_case()
        labels = pd.Series((y > 0).astype(int))
        rank_features(frame, labels, method="random_forest", task="regression")
        assert used.get("regressor", 0) == 1
        assert used.get("classifier", 0) == 0

    def test_target_length_mismatch_raises_before_fitting(self) -> None:
        frame, y = _regression_case()
        with pytest.raises(ValueError, match="does not match frame length"):
            rank_features(frame, y[:-1], method="random_forest")

    def test_target_nan_raises_before_fitting(self) -> None:
        frame, y = _regression_case()
        bad = list(map(float, y))
        bad[3] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            rank_features(frame, bad, method="random_forest")

    @pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
    def test_target_inf_raises_before_fitting(self, bad_value: float) -> None:
        frame, y = _regression_case()
        bad = list(map(float, y))
        bad[7] = bad_value
        with pytest.raises(ValueError, match="non-finite"):
            rank_features(frame, bad, method="random_forest")

    def test_target_required_for_model_methods(self) -> None:
        frame, _ = _regression_case()
        with pytest.raises(ValueError, match="requires a target"):
            rank_features(frame, method="random_forest")

    def test_unknown_method_raises(self) -> None:
        frame, _ = _regression_case()
        with pytest.raises(ValueError, match="unknown ranking method"):
            rank_features(frame, method="gradient_boosting")

    def test_unknown_task_raises(self) -> None:
        frame, y = _regression_case()
        with pytest.raises(ValueError, match="unknown task"):
            rank_features(frame, y, method="random_forest", task="clustering")


class TestRankFeaturesLightGBM:
    """LightGBM is strictly optional: use it when present, fall back when not."""

    @requires_lightgbm
    def test_lightgbm_method_regression(self) -> None:
        frame, y = _regression_case()
        scores = rank_features(frame, y, method="lightgbm")
        assert set(scores.index) == set(frame.columns)
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.is_monotonic_decreasing
        assert float(scores["f_constant"]) == 0.0

    @requires_lightgbm
    def test_auto_with_target_uses_a_model_ranker(self) -> None:
        frame, y = _regression_case()
        scores = rank_features(frame, y)  # auto -> lightgbm here, rf elsewhere
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.is_monotonic_decreasing
        assert set(scores.index) == set(frame.columns)

    def test_lightgbm_missing_falls_back_to_random_forest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "lightgbm", None)
        frame, y = _regression_case()
        scores = rank_features(frame, y, method="lightgbm")
        assert np.isfinite(scores.to_numpy()).all()
        assert scores.is_monotonic_decreasing
        assert scores.index[0] == "f_informative"  # the random-forest fallback
        assert float(scores["f_constant"]) == 0.0

    def test_lightgbm_auto_missing_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "lightgbm", None)
        frame, y = _regression_case()
        scores = rank_features(frame, y)  # auto must not explode without lightgbm
        assert scores.index[0] == "f_informative"
        assert scores.is_monotonic_decreasing


# ---------------------------------------------------------------------------
# select_top_features
# ---------------------------------------------------------------------------
SELECTION_COLUMNS = [
    "desc_a",
    "desc_b",
    "desc_c",
    "ecfp_bit_0",
    "ecfp_bit_1",
    "morgan_full_bit_0",
]
SELECTION_SCORES = [0.10, 0.50, 0.30, 0.40, 0.20, 0.35]


@pytest.fixture()
def selection_frame() -> pd.DataFrame:
    return pd.DataFrame(np.zeros((3, len(SELECTION_COLUMNS))), columns=SELECTION_COLUMNS)


def selection_scores() -> pd.Series:
    return pd.Series(SELECTION_SCORES, index=SELECTION_COLUMNS)


class TestSelectTopFeatures:
    """Layer 1 (per-family median) + layer 2 (top-K, stable tie-break)."""

    def test_two_layer_selection_exact(self, selection_frame: pd.DataFrame) -> None:
        # Layer 1 medians: descriptors {0.10, 0.50, 0.30} -> keep 0.50, 0.30;
        # ecfp {0.40, 0.20} -> keep 0.40; morgan_full {0.35} -> keep 0.35.
        # Layer 2 top-3 of [0.50, 0.40, 0.35, 0.30] -> desc_b, ecfp_bit_0, morgan.
        selection = select_top_features(
            selection_frame, selection_scores(), k=3, method="demo"
        )
        assert isinstance(selection, FeatureSelection)
        assert selection.selected == ["desc_b", "ecfp_bit_0", "morgan_full_bit_0"]
        assert selection.family_counts == {"descriptors": 1, "ecfp": 1, "morgan_full": 1}
        assert selection.method == "demo"
        assert list(selection.scores.index) == SELECTION_COLUMNS
        assert selection.scores.tolist() == SELECTION_SCORES

    def test_k_ge_candidates_returns_all_in_score_order(
        self, selection_frame: pd.DataFrame
    ) -> None:
        selection = select_top_features(selection_frame, selection_scores(), k=10)
        assert selection.selected == [
            "desc_b",
            "ecfp_bit_0",
            "morgan_full_bit_0",
            "desc_c",
        ]
        assert selection.family_counts == {"descriptors": 2, "ecfp": 1, "morgan_full": 1}

    def test_median_prefilter_disabled_considers_all_columns(
        self, selection_frame: pd.DataFrame
    ) -> None:
        scores = selection_scores()
        selection = select_top_features(selection_frame, scores, k=5, median_prefilter=False)
        # desc_a and ecfp_bit_1 stay candidates without the median filter.
        assert selection.selected == [
            "desc_b",
            "ecfp_bit_0",
            "morgan_full_bit_0",
            "desc_c",
            "ecfp_bit_1",
        ]
        assert selection.family_counts == {"descriptors": 2, "ecfp": 2, "morgan_full": 1}
        # ... while the prefilter leaves only 4 candidates for the same k=5.
        filtered = select_top_features(selection_frame, scores, k=5)
        assert filtered.selected == [
            "desc_b",
            "ecfp_bit_0",
            "morgan_full_bit_0",
            "desc_c",
        ]

    def test_tie_break_prefers_frame_column_order(self) -> None:
        columns = ["desc_x", "ecfp_bit_0", "desc_y"]
        frame = pd.DataFrame(np.zeros((2, 3)), columns=columns)
        scores = pd.Series([0.5, 0.5, 0.5], index=columns)
        selection = select_top_features(frame, scores, k=2)
        assert selection.selected == ["desc_x", "ecfp_bit_0"]

    def test_single_column_family_survives_median_prefilter(self) -> None:
        columns = ["desc_a", "ecfp_bit_3"]
        frame = pd.DataFrame(np.zeros((2, 2)), columns=columns)
        scores = pd.Series([0.25, 0.75], index=columns)
        selection = select_top_features(frame, scores, k=2)
        assert selection.selected == ["ecfp_bit_3", "desc_a"]
        assert selection.family_counts == {"descriptors": 1, "ecfp": 1}

    def test_family_counts_use_canonical_family_order(
        self, selection_frame: pd.DataFrame
    ) -> None:
        selection = select_top_features(selection_frame, selection_scores(), k=10)
        assert list(selection.family_counts) == ["descriptors", "ecfp", "morgan_full"]

    @pytest.mark.parametrize("bad_k", [0, -3])
    def test_k_non_positive_raises(
        self, selection_frame: pd.DataFrame, bad_k: int
    ) -> None:
        with pytest.raises(ValueError, match="k must be > 0"):
            select_top_features(selection_frame, selection_scores(), k=bad_k)

    @pytest.mark.parametrize("bad_k", [2.5, "5", True])
    def test_k_non_integer_raises(
        self, selection_frame: pd.DataFrame, bad_k: Any
    ) -> None:
        with pytest.raises(ValueError, match="k must be an integer"):
            select_top_features(selection_frame, selection_scores(), k=bad_k)

    def test_scores_index_mismatch_raises(
        self, selection_frame: pd.DataFrame
    ) -> None:
        reversed_ = pd.Series(SELECTION_SCORES, index=list(reversed(SELECTION_COLUMNS)))
        with pytest.raises(ValueError, match="scores.index must equal frame.columns"):
            select_top_features(selection_frame, reversed_, k=3)
        short = pd.Series([1.0, 2.0], index=["desc_a", "desc_b"])
        with pytest.raises(ValueError, match="scores.index must equal frame.columns"):
            select_top_features(selection_frame, short, k=3)

    def test_unknown_prefix_column_raises(self) -> None:
        frame = pd.DataFrame(np.zeros((2, 2)), columns=["desc_a", "SMILES"])
        scores = pd.Series([1.0, 2.0], index=["desc_a", "SMILES"])
        with pytest.raises(ValueError, match="SMILES"):
            select_top_features(frame, scores, k=1)


# ---------------------------------------------------------------------------
# selection_metadata
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metadata_case() -> tuple[FeatureSelection, FeatureReport]:
    """A small end-to-end case: compute -> rank by hand -> select -> metadata."""
    smiles = [ETHANOL, "CCC", "not_a_smiles"]
    frame, report = compute_feature_frame(smiles, families=("descriptors", "ecfp"))
    n_desc = report.families["descriptors"]
    n_ecfp = report.families["ecfp"]
    scores = pd.Series(
        [0.9] * n_desc + list(np.linspace(0.2, 0.8, n_ecfp)), index=frame.columns
    )
    selection = select_top_features(frame, scores, k=5, method="random_forest")
    return selection, report


class TestSelectionMetadata:
    """The JSON-ready report dictionary."""

    def test_key_set_is_exact(
        self, metadata_case: tuple[FeatureSelection, FeatureReport]
    ) -> None:
        selection, report = metadata_case
        meta = selection_metadata(selection, report, k=5)
        assert set(meta) == {
            "selected_features",
            "descriptor_names",
            "fingerprint_sizes",
            "family_counts",
            "families_computed",
            "k_requested",
            "method",
            "n_molecules",
            "dropped",
            "total_features",
        }

    def test_values_match_selection_and_report(
        self, metadata_case: tuple[FeatureSelection, FeatureReport]
    ) -> None:
        selection, report = metadata_case
        meta = selection_metadata(selection, report, k=5)
        assert meta["selected_features"] == selection.selected
        assert all(column.startswith("desc_") for column in meta["selected_features"])
        assert meta["descriptor_names"] == [
            column[len("desc_"):] for column in selection.selected
        ]
        assert all(isinstance(name, str) for name in meta["descriptor_names"])
        assert meta["fingerprint_sizes"] == {"ecfp": ECFP_BITS}
        assert meta["family_counts"] == selection.family_counts
        assert meta["families_computed"] == report.families
        assert meta["k_requested"] == 5
        assert meta["method"] == "random_forest"
        assert meta["n_molecules"] == 2
        assert meta["dropped"] == [[2, "not_a_smiles"]]
        assert all(
            isinstance(row, int) and isinstance(smi, str)
            for row, smi in meta["dropped"]
        )
        assert meta["total_features"] == report.total_columns

    def test_json_serializable_without_default(
        self, metadata_case: tuple[FeatureSelection, FeatureReport]
    ) -> None:
        selection, report = metadata_case
        meta = selection_metadata(selection, report, k=5)
        text = json.dumps(meta)  # plain types only, no default= needed
        restored = json.loads(text)
        assert restored["selected_features"] == meta["selected_features"]
        assert restored["dropped"] == [[2, "not_a_smiles"]]
        assert restored["k_requested"] == 5

    def test_families_argument_filters_fingerprint_sizes(
        self, metadata_case: tuple[FeatureSelection, FeatureReport]
    ) -> None:
        selection, report = metadata_case
        meta = selection_metadata(selection, report, k=5, families=("ecfp", "avalon_count"))
        assert list(meta["fingerprint_sizes"]) == ["ecfp", "avalon_count"]
        assert meta["fingerprint_sizes"] == {"ecfp": ECFP_BITS, "avalon_count": AVALON_BITS}
        assert meta["families_computed"] == report.families  # untouched by the argument

    def test_full_run_reports_all_fingerprint_sizes(
        self, full_library: tuple[pd.DataFrame, FeatureReport]
    ) -> None:
        from rdkit.Chem import rdMolDescriptors

        frame, report = full_library
        scores = pd.Series(np.linspace(1.0, 0.0, frame.shape[1]), index=frame.columns)
        selection = select_top_features(frame, scores, k=3)
        meta = selection_metadata(selection, report, k=3)
        assert meta["fingerprint_sizes"] == {
            "ecfp": 2048,
            "ecfp3": 2048,
            "rdkit_fingerprint": 2048,
            "morgan_full": 2048,
            "morgan_scaffold": 2048,
            "avalon_count": 512,
            "atompair_count": 2048,
            "maccs": 167,
            "fcfp": 2048,
            "fcfp_count": 2048,
            "topological": 2048,
            "topological_count": 2048,
            "rdkit_count": 2048,
            "pattern": 2048,
            "layered": 2048,
            "estate": 79,
            "erg": 315,
        }
        assert meta["n_molecules"] == 8
        assert meta["dropped"] == [[8, "not_a_smiles"], [9, ""]]
        assert meta["total_features"] == (
            12_800 + 7 * 2048 + 167 + 79 + 315 + 9 + RISM_DESCRIPTOR_COUNT + len(
                rdMolDescriptors.Properties.GetAvailableProperties()
            )
        )
        assert meta["k_requested"] == 3
        assert meta["selected_features"] == selection.selected


# ---------------------------------------------------------------------------
# The ten fak_ml.ipynb fingerprint families + shape_3d (100x-148)
# ---------------------------------------------------------------------------
class TestFakmlFamilies:
    """Cell 40's 16 datamol fingerprint types, ported as first-class families.

    The ten new families must reproduce ``dm.to_fp`` byte-exactly (the
    notebook's own benchmark vocabulary) with two honest exceptions,
    pinned below: datamol's empty-sparse bug on ethanol torsion counts,
    and shape_3d, which has no datamol counterpart.
    """

    FAKML_TYPES: tuple[tuple[str, str, str], ...] = (
        # (datamol type, OUR family name, column prefix)
        ("maccs", "maccs", "maccs_bit_"),
        ("fcfp", "fcfp", "fcfp_bit_"),
        ("fcfp-count", "fcfp_count", "fcfp_count_"),
        ("topological", "topological", "topo_bit_"),
        ("topological-count", "topological_count", "topo_count_"),
        ("rdkit-count", "rdkit_count", "rdkit_count_"),
        ("pattern", "pattern", "pattern_bit_"),
        ("layered", "layered", "layered_bit_"),
        ("estate", "estate", "estate_bit_"),
        ("erg", "erg", "erg_"),
    )

    def test_prefix_uniqueness_all_pairs(self) -> None:
        """No family prefix may be a prefix of another (exact inverse)."""
        prefixes = list(FAMILY_PREFIXES.values())
        for i, first in enumerate(prefixes):
            for second in prefixes[i + 1 :]:
                assert not second.startswith(first), (
                    f"prefix {first!r} is a prefix of {second!r}"
                )
                assert not first.startswith(second), (
                    f"prefix {second!r} is a prefix of {first!r}"
                )

    def test_real_column_names_new_families(self) -> None:
        assert family_of_column("maccs_bit_166") == "maccs"
        assert family_of_column("fcfp_bit_0") == "fcfp"
        assert family_of_column("fcfp_count_2047") == "fcfp_count"
        assert family_of_column("topo_bit_9") == "topological"
        assert family_of_column("topo_count_9") == "topological_count"
        assert family_of_column("rdkit_count_5") == "rdkit_count"
        assert family_of_column("pattern_bit_1") == "pattern"
        assert family_of_column("layered_bit_1") == "layered"
        assert family_of_column("estate_bit_78") == "estate"
        assert family_of_column("erg_314") == "erg"
        assert family_of_column("d3d_pmi1") == "shape_3d"

    def test_datamol_parity(self) -> None:
        """Every new family matches dm.to_fp byte-exactly (the benchmark vocab)."""
        datamol = pytest.importorskip("datamol")
        from rdkit import Chem

        families = tuple(family for _dm, family, _prefix in self.FAKML_TYPES)
        frame, report = compute_feature_frame(
            [ASPIRIN, ETHANOL, "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
            families=families,
        )
        assert report.n_valid == 3
        mols = [Chem.MolFromSmiles(s) for s in (ASPIRIN, ETHANOL, "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]
        for dm_type, _family, prefix in self.FAKML_TYPES:
            columns = [c for c in frame.columns if c.startswith(prefix)]
            assert len(columns) > 0, f"family columns missing for {dm_type}"
            for row, mol in enumerate(mols):
                if dm_type == "topological-count" and row == 1:
                    # datamol's own fp_to_array bug: ethanol has no
                    # 4-atom torsion, the empty sparse count vect
                    # crashes its unpack -- OUR zero vector is the
                    # honest answer (pinned in the next test).
                    continue
                reference = np.asarray(datamol.to_fp(mol, fp_type=dm_type), dtype=float)
                ours = frame.iloc[row][columns].to_numpy(dtype=float)
                assert np.allclose(reference, ours, equal_nan=True), (
                    f"{dm_type} diverged from datamol on row {row}"
                )

    def test_ethanol_torsion_count_is_zero_vector(self) -> None:
        """Ethanol has no 4-atom torsion: an all-zero count row, not a drop."""
        frame, report = compute_feature_frame([ETHANOL], families=("topological_count",))
        assert report.n_valid == 1
        columns = [c for c in frame.columns if c.startswith("topo_count_")]
        assert len(columns) == 2048
        assert not frame.iloc[0][columns].to_numpy().any()

    def test_bit_sizes(self) -> None:
        from pharmaml.chem.features import (
            ERG_BITS,
            ESTATE_BITS,
            MACCS_BITS,
            TOPOLOGICAL_BITS,
        )

        assert (MACCS_BITS, TOPOLOGICAL_BITS, ESTATE_BITS, ERG_BITS) == (167, 2048, 79, 315)

    def test_shape_3d_family(self) -> None:
        frame, report = compute_feature_frame([ASPIRIN, ETHANOL], families=("shape_3d",))
        assert report.families == {"shape_3d": 9}
        assert list(frame.columns) == [
            "d3d_asphericity",
            "d3d_eccentricity",
            "d3d_pmi1",
            "d3d_pmi2",
            "d3d_pmi3",
            "d3d_radius_of_gyration",
            "d3d_spherocity_index",
            "d3d_npr1",
            "d3d_npr2",
        ]
        values = frame.to_numpy()
        assert np.isfinite(values).all()
        # PMI ordering + NPR bounds hold for both molecules.
        assert (values[:, 2] <= values[:, 3] + 1e-9).all()
        assert (values[:, 3] <= values[:, 4] + 1e-9).all()
        assert (values[:, 7] <= values[:, 8] + 1e-9).all()
        assert (values[:, 7] > 0).all() and (values[:, 8] <= 1.0 + 1e-9).all()

    def test_shape_3d_is_deterministic(self) -> None:
        first, _ = compute_feature_frame([ASPIRIN], families=("shape_3d",))
        second, _ = compute_feature_frame([ASPIRIN], families=("shape_3d",))
        assert first.equals(second)

    def test_shape_3d_nan_row_for_failed_pipeline(self, monkeypatch) -> None:
        """A failed 3D pipeline keeps the ROW as NaN (xxx.ipynb's behaviour).

        Probed honestly: no tested SMILES fails BOTH force fields on
        this build (even ``[He]`` embeds with degenerate zeros), so the
        features-level NaN contract is pinned through the same broken
        ``_shape_vector`` the descriptors suite uses -- the row must
        stay, the values must go.
        """
        from pharmaml.chem import descriptors as descriptors_module

        def broken(mol_h, n_conformers):  # pragma: no cover - stub
            raise ValueError("embedding exploded")

        monkeypatch.setattr(descriptors_module, "_shape_vector", broken)
        frame, report = compute_feature_frame([ASPIRIN], families=("shape_3d",))
        assert report.n_valid == 1  # the row is kept
        values = frame.to_numpy()
        assert values.shape == (1, 9)
        assert np.isnan(values).all()

    def test_shape_3d_single_atom_degenerate_zeros(self) -> None:
        """A single-atom molecule embeds to a POINT: finite zeros, kept.

        Probed: ``[Cl-]`` and friends all embed on this build; their
        moment-of-inertia descriptors are legitimately zero for a point
        particle -- the honest degenerate geometry, not a failure.
        """
        frame, report = compute_feature_frame(["[Cl-]"], families=("shape_3d",))
        assert report.n_valid == 1
        values = frame.to_numpy()
        assert np.isfinite(values).all()
        assert (values[:, 2:5] == 0).all()  # PMI1-3 of a point particle

    def test_new_families_rank_and_select(self) -> None:
        """The new families flow through the two-layer selection unchanged."""
        frame, _report = compute_feature_frame(
            [ASPIRIN, ETHANOL, "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
            families=("maccs", "estate", "erg", "shape_3d"),
        )
        scores = pd.Series(
            np.linspace(1.0, 0.0, frame.shape[1]), index=frame.columns
        )
        selection = select_top_features(frame, scores, k=5)
        assert len(selection.selected) == 5
        assert selection.family_counts  # at least one family survived layer 1
