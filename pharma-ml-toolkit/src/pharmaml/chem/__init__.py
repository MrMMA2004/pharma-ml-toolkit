"""Cheminformatics helpers: conversion, descriptors, similarity, tables, standardization."""

from pharmaml.chem.converters import (
    SdfWriteReport,
    add_pic50_column,
    ic50_to_pic50,
    ki_to_pki,
    sdf_to_table,
    smi_to_csv,
    split_table_into_chunks,
    table_to_sdf,
    table_to_smi,
)
from pharmaml.chem.descriptors import (
    compute_descriptors,
    compute_fingerprints,
    compute_maccs,
)
from pharmaml.chem.similarity import (
    tanimoto_from_bitarrays,
    tanimoto_from_sets,
    tanimoto_matrix,
)
from pharmaml.chem.standardize import (
    UNSUPPORTED_METALS,
    PdbLigandReport,
    canonicalize_tautomer,
    smiles_to_pdb,
)
from pharmaml.chem.tables import (
    TableComparison,
    annotate_with_smiles,
    compare_tables,
    merge_table_folder,
    merge_tables,
)

__all__ = [
    "UNSUPPORTED_METALS",
    "PdbLigandReport",
    "SdfWriteReport",
    "TableComparison",
    "add_pic50_column",
    "annotate_with_smiles",
    "canonicalize_tautomer",
    "compare_tables",
    "compute_descriptors",
    "compute_fingerprints",
    "compute_maccs",
    "ic50_to_pic50",
    "ki_to_pki",
    "merge_table_folder",
    "merge_tables",
    "sdf_to_table",
    "smi_to_csv",
    "smiles_to_pdb",
    "split_table_into_chunks",
    "table_to_sdf",
    "table_to_smi",
    "tanimoto_from_bitarrays",
    "tanimoto_from_sets",
    "tanimoto_matrix",
]
