"""(part8 Phase D) the structured feature-generation configuration.

Pins the commission contract:
- exact individual descriptor selection (a subset of the catalog),
- independent fingerprint radius / bit-size configuration,
- binary vs count representation for the Morgan-style families,
- the config flowing GUI → validation → backend → frame → metadata,
- reproducibility (as_dict/from_dict roundtrip, identical frames).
"""

from __future__ import annotations

import numpy as np
import pytest

from pharmaml.chem.features import (
    FeatureConfig,
    compute_feature_frame,
    descriptor_catalog,
)

SMILES = ["CCO", "c1ccccc1O", "CCN"]


class TestValidation:
    def test_default_config_matches_legacy_width(self) -> None:
        frame, _report = compute_feature_frame(SMILES, families=("ecfp",))
        cfg_frame, _r2 = compute_feature_frame(
            SMILES, config=FeatureConfig(families=("ecfp",))
        )
        assert frame.shape == cfg_frame.shape
        assert list(frame.columns) == list(cfg_frame.columns)
        np.testing.assert_array_equal(frame.to_numpy(), cfg_frame.to_numpy())

    @pytest.mark.parametrize("kwargs", [
        {"families": ("ecfp",), "morgan_radius": -1},
        {"families": ("ecfp",), "morgan_bits": 0},
        {"families": ("ecfp",), "morgan_bits": 65_537},
        {"families": ("nope",)},
        {"families": ("ecfp",), "descriptors": ()},
        {"families": ("ecfp",), "descriptors": ("NotARealOne",)},
    ])
    def test_invalid_configs_raise(self, kwargs) -> None:
        with pytest.raises(ValueError):
            FeatureConfig(**kwargs)


class TestDescriptorSubset:
    def test_exact_subset_selection(self) -> None:
        catalog = descriptor_catalog()
        assert len(catalog) >= 40  # the real RDKit property list
        chosen = tuple(catalog[:5])
        cfg = FeatureConfig(families=("descriptors",), descriptors=chosen)
        frame, report = compute_feature_frame(SMILES, config=cfg)
        assert list(frame.columns) == [f"desc_{name}" for name in chosen]
        assert report.families["descriptors"] == 5
        # the values match the FULL run's same columns (no re-scaling)
        full, _r = compute_feature_frame(SMILES, families=("descriptors",))
        for name in chosen:
            np.testing.assert_array_equal(
                frame[f"desc_{name}"].to_numpy(), full[f"desc_{name}"].to_numpy()
            )

    def test_subset_order_is_catalog_order_not_request_order(self) -> None:
        catalog = descriptor_catalog()
        requested = (catalog[3], catalog[0], catalog[1])
        cfg = FeatureConfig(families=("descriptors",), descriptors=requested)
        frame, _ = compute_feature_frame(SMILES[:1], config=cfg)
        assert list(frame.columns) == [
            f"desc_{catalog[0]}", f"desc_{catalog[1]}", f"desc_{catalog[3]}",
        ]


class TestFingerprintOverrides:
    def test_independent_radius_and_bits(self) -> None:
        cfg = FeatureConfig(families=("ecfp",), morgan_radius=3, morgan_bits=512)
        frame, report = compute_feature_frame(SMILES, config=cfg)
        assert frame.shape == (3, 512)
        assert report.families["ecfp"] == 512
        assert all(c.startswith("ecfp_bit_") for c in frame.columns)

    def test_radius_changes_the_chemistry(self) -> None:
        """Radius 3 sees wider atom environments than radius 2 -- the
        vectors must differ on a molecule with radius-3 structure."""
        wide = ["CCOc1ccccc1C(=O)Oc1ccccc1C(=O)O"]  # aspirin
        r2, _ = compute_feature_frame(
            wide, config=FeatureConfig(families=("ecfp",), morgan_radius=2)
        )
        r3, _ = compute_feature_frame(
            wide, config=FeatureConfig(families=("ecfp",), morgan_radius=3)
        )
        assert not np.array_equal(r2.to_numpy(), r3.to_numpy())

    def test_count_mode(self) -> None:
        cfg = FeatureConfig(families=("ecfp",), as_counts=True)
        frame, _ = compute_feature_frame(["CCOc1ccccc1"], config=cfg)
        values = frame.to_numpy()
        assert values.max() > 1.0  # counts, not bits
        assert frame.shape[1] == 2048  # same vocabulary width

    def test_non_morgan_families_ignore_the_overrides(self) -> None:
        """Morgan parameters must not leak into families where they have
        no scientific meaning (the commission's explicit rule)."""
        cfg = FeatureConfig(
            families=("maccs",), morgan_radius=5, morgan_bits=64, as_counts=True
        )
        frame, _ = compute_feature_frame(SMILES, config=cfg)
        assert frame.shape == (3, 167)  # MACCS stays its fixed 167


class TestReproducibility:
    def test_dict_roundtrip(self) -> None:
        cfg = FeatureConfig(
            families=("ecfp", "maccs"), morgan_radius=3, morgan_bits=1024,
            as_counts=True, n_jobs=1,
        )
        rebuilt = FeatureConfig.from_dict(cfg.as_dict())
        assert rebuilt == cfg
        f1, _ = compute_feature_frame(SMILES, config=cfg)
        f2, _ = compute_feature_frame(SMILES, config=rebuilt)
        np.testing.assert_array_equal(f1.to_numpy(), f2.to_numpy())

    def test_metadata_carries_the_config(self) -> None:
        cfg = FeatureConfig(
            families=("ecfp",), morgan_radius=3, morgan_bits=512,
        )
        record = cfg.as_dict()
        assert record["morgan_radius"] == 3
        assert record["morgan_bits"] == 512
        assert record["families"] == ["ecfp"]
        assert record["descriptors"] is None

    def test_deterministic_across_runs(self) -> None:
        cfg = FeatureConfig(families=("ecfp", "maccs"))
        f1, _ = compute_feature_frame(SMILES, config=cfg)
        f2, _ = compute_feature_frame(SMILES, config=cfg)
        np.testing.assert_array_equal(f1.to_numpy(), f2.to_numpy())
