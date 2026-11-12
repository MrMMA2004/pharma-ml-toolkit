"""Explicit cross-engine geometry must never silently use legacy sizing."""
from __future__ import annotations

import math
import warnings

import pytest

from pharmaml.docking.preparation import GridBox, write_gpf
from pharmaml.exceptions import DataValidationError


def test_existing_vina_box_remains_unchanged():
    box = GridBox(1, 2, 3, npts=(20, 30, 40), spacing=0.375)
    assert box.size == pytest.approx((7.125, 10.875, 14.625))
    assert box.as_vina_kwargs() == box.as_vina_kwargs(convention="legacy")


def test_explicit_autogrid_box_uses_intervals():
    box = GridBox(1, 2, 3, npts=(20, 30, 40), spacing=0.375)
    assert box.autogrid_size == pytest.approx((7.5, 11.25, 15.0))
    result = box.as_vina_kwargs(convention="autogrid")
    assert (result["center_x"], result["center_y"], result["center_z"]) == (1, 2, 3)
    assert (result["size_x"], result["size_y"], result["size_z"]) == pytest.approx(box.autogrid_size)


@pytest.mark.parametrize("npts", [(21, 20, 20), (0, 20, 20), (-2, 20, 20), (True, 20, 20),
                                  (20.0, 20, 20), (20, 20), (20, 20, 20, 20), None])
def test_autogrid_conversion_rejects_unverified_grid_dimensions(npts):
    with pytest.raises(DataValidationError, match="even"):
        GridBox(0, 0, 0, npts=npts).as_vina_kwargs(convention="autogrid")


@pytest.mark.parametrize("spacing", [0, -1, math.nan, math.inf])
def test_autogrid_conversion_requires_finite_positive_spacing(spacing):
    with pytest.raises(DataValidationError, match="spacing"):
        GridBox(0, 0, 0, spacing=spacing).as_vina_kwargs(convention="autogrid")


def test_autogrid_conversion_requires_finite_centre():
    with pytest.raises(DataValidationError, match="center"):
        GridBox(math.nan, 0, 0).as_vina_kwargs(convention="autogrid")


def test_unknown_geometry_convention_rejected():
    with pytest.raises(DataValidationError, match="convention"):
        GridBox(0, 0, 0).as_vina_kwargs(convention="unknown")


def test_odd_gpf_warns_without_changing_requested_values(tmp_path):
    ligand = tmp_path / "l.pdbqt"
    ligand.write_text("ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     C\n")
    with pytest.warns(UserWarning, match="Odd AutoGrid npts"):
        result = write_gpf(ligand, box=GridBox(0, 0, 0, npts=(21, 20, 20)))
    assert result.read_text().startswith("npts 21 20 20")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_gpf(ligand, box=GridBox(0, 0, 0, npts=(20, 20, 20)))
    assert not any("Odd AutoGrid npts" in str(item.message) for item in caught)
