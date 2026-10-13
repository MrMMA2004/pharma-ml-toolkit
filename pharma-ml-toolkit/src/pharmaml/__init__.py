"""PharmaML — a production-ready toolkit for computational pharmacy.

The package consolidates three domains into one coherent, typed and
tested codebase:

* :mod:`pharmaml.chem`     -- cheminformatics helpers (unit/endpoint
  conversion, descriptors, fingerprints, Tanimoto similarity).
* :mod:`pharmaml.ml`       -- QSAR machine learning (preprocessing,
  model registry, training with cross-validation, evaluation,
  persistence).
* :mod:`pharmaml.pharma`   -- pharmaceutical calculations (units,
  pharmacokinetics, pharmacodynamics, dosage design).
* :mod:`pharmaml.docking`  -- molecular-docking workflows (ligand and
  grid preparation, AutoDock4/Vina batch execution, result parsing and
  scoring).
* :mod:`pharmaml.utils`    -- shared I/O and validation helpers.

Typical usage::

    from pharmaml.ml import compare_models, train_model
    from pharmaml.pharma.pk import auc_trapezoid, clearance

Only ``numpy``/``pandas``/``scikit-learn`` (and a handful of small
utilities) are required at import time.  Heavy optional dependencies
(RDKit, XGBoost/LightGBM/CatBoost, Meeko) are imported lazily and raise
informative :class:`~pharmaml.exceptions.DependencyError` messages when
missing.
"""

from pharmaml.exceptions import (
    DataValidationError,
    DependencyError,
    DockingError,
    DockingParseError,
    PharmaMLError,
    UnitConversionError,
)

__version__ = "1.0.0"

__all__ = [
    "PharmaMLError",
    "DataValidationError",
    "DependencyError",
    "DockingError",
    "DockingParseError",
    "UnitConversionError",
    "__version__",
]
