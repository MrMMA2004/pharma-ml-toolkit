"""Molecular-dynamics analysis — trajectories, metrics, interactions.

One package for the MD half of computational pharmacy, seeded by the
user's own GROMACS post-processing script (``upload/MD_analysis.py``)
and shaped by the reference stack the field runs on —
``MDAnalysis/mdanalysis`` and ``mdtraj/mdtraj`` (the trajectory core
and selections), ``gromacs/gromacs`` / ``Amber-MD/cpptraj`` /
``Amber-MD/pytraj`` (the metric vocabulary), ``getcontacts/getcontacts``
(contact and H-bond timelines), ``Valdes-Tresanco-MS/gmx_MMPBSA``
(binding free-energy legs), ``prody/ProDy`` (elastic network models),
``deeptime-ml/deeptime`` (PCA/tICA essential dynamics),
``chemosim-lab/ProLIF`` (protein–ligand interaction fingerprints), and
the viewing stacks (``nglviewer/nglview``, ``schrodinger/pymol-open-source``,
``Eigenstate/vmd-python``) the web tab stands in for.

Submodules:

* :mod:`pharmaml.md.xvg` — the uploaded pipeline's ``.xvg``/``.ndx``
  parsers, unit conversions and metric metadata;
* :mod:`pharmaml.md.trajectory` — :class:`Topology` /
  :class:`Trajectory` with the MDAnalysis selection mini-language;
* :mod:`pharmaml.md.align` — Kabsch superposition and invariant RMSD;
* :mod:`pharmaml.md.metrics` — RMSD/RMSF/Rg timelines, contact maps,
  native contacts (Q), g(r), dihedrals;
* :mod:`pharmaml.md.hbond` — geometric H-bond detection and salt
  bridges;
* :mod:`pharmaml.md.dimred` — trajectory PCA and tICA;
* :mod:`pharmaml.md.enm` — ANM/GNM elastic network models;
* :mod:`pharmaml.md.plif` — protein–ligand interaction fingerprints;
* :mod:`pharmaml.md.study` — the uploaded ligand-study pipeline
  (folders → parsed series → stats → correlations → CSV).

Examples
--------
>>> from pharmaml.md import build_study, rmsd_trajectory, synthetic_trajectory
>>> traj = synthetic_trajectory(n_frames=8, seed=0)
>>> rmsd = rmsd_trajectory(traj)
>>> (len(rmsd), rmsd[0] < 1e-12)
(8, True)
"""

from pharmaml.md.align import kabsch_rotation, rmsd, superpose
from pharmaml.md.dimred import ProjectionResult, tica, trajectory_pca
from pharmaml.md.enm import (
    anm_fluctuations,
    anm_hessian,
    anm_modes,
    cross_correlation,
    gnm_fluctuations,
    kirchhoff_matrix,
)
from pharmaml.md.hbond import find_hbonds, hbond_timeline, salt_bridges
from pharmaml.md.metrics import (
    center_of_mass,
    contact_map,
    dihedral,
    native_contact_fraction,
    pairwise_distances,
    radial_distribution_function,
    radius_of_gyration,
    radius_of_gyration_trajectory,
    rmsd_trajectory,
    rmsf,
)
from pharmaml.md.plif import InteractionFingerprint, interaction_fingerprint
from pharmaml.md.study import (
    LigandResult,
    MDStudy,
    build_study,
    metric_correlations,
    read_ligand_folder,
)
from pharmaml.md.trajectory import (
    Topology,
    Trajectory,
    resolve_atom_indices,
    synthetic_hbond_trajectory,
    synthetic_trajectory,
)
from pharmaml.md.xvg import (
    KJ_TO_KCAL,
    METRIC_FILES,
    MetricInfo,
    XvgSeries,
    classify_metric,
    metric_stats,
    moving_average,
    parse_ndx_hbnum,
    parse_xvg,
    ps_to_ns,
    to_kcal,
)

__all__ = [
    "KJ_TO_KCAL",
    "METRIC_FILES",
    "InteractionFingerprint",
    "LigandResult",
    "MDStudy",
    "MetricInfo",
    "ProjectionResult",
    "Topology",
    "Trajectory",
    "XvgSeries",
    "anm_fluctuations",
    "anm_hessian",
    "anm_modes",
    "build_study",
    "center_of_mass",
    "classify_metric",
    "contact_map",
    "cross_correlation",
    "dihedral",
    "find_hbonds",
    "gnm_fluctuations",
    "hbond_timeline",
    "interaction_fingerprint",
    "kabsch_rotation",
    "kirchhoff_matrix",
    "metric_correlations",
    "metric_stats",
    "moving_average",
    "native_contact_fraction",
    "pairwise_distances",
    "parse_ndx_hbnum",
    "parse_xvg",
    "ps_to_ns",
    "radial_distribution_function",
    "radius_of_gyration",
    "radius_of_gyration_trajectory",
    "read_ligand_folder",
    "resolve_atom_indices",
    "rmsd",
    "rmsd_trajectory",
    "rmsf",
    "salt_bridges",
    "superpose",
    "synthetic_hbond_trajectory",
    "synthetic_trajectory",
    "tica",
    "to_kcal",
    "trajectory_pca",
]