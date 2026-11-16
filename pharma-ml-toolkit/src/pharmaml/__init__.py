"""PharmaML — a production-ready toolkit for computational pharmacy.

The package consolidates three domains into one coherent, typed and
tested codebase:

* :mod:`pharmaml.data`     -- bring your own data: ``load_dataset``
  reads user CSV/TSV/Excel files, auto-detects the target column,
  infers classification (2+ classes) vs regression and audits data
  quality.
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
* :mod:`pharmaml.md`      -- molecular-dynamics analysis (GROMACS
  ``.xvg``/``.ndx`` study parsing, trajectory metrics, hydrogen bonds,
  PCA/tICA essential dynamics, ANM/GNM elastic networks, protein-ligand
  interaction fingerprints).
* :mod:`pharmaml.viz`      -- visual output for every domain: figures
  (confusion matrices, ROC, parity, PK profiles, similarity heat-maps,
  docking rankings ...), multi-format table export and self-contained
  HTML reports.
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
    MDAnalysisError,
    PharmaMLError,
    UnitConversionError,
)

__version__ = "1.2.212"

__all__ = [
    "DataValidationError",
    "DependencyError",
    "DockingError",
    "DockingParseError",
    "MDAnalysisError",
    "PharmaMLError",
    "UnitConversionError",
    "__version__",
]
