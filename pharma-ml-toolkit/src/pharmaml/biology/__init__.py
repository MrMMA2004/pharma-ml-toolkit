"""Cell-line biology: drug-response screens, omics and biomarkers.

The third domain of the toolkit (besides chemistry and docking):
analysing *cell line* data the way GDSC/CCLE pharmacogenomics projects
do -- from zero to a ranked biomarker table.

* :mod:`pharmaml.biology.response` -- GDSC/CCLE-style drug-response
  screens: load and audit long-format (cell line, compound, metric)
  tables, collapse replicates, convert ``LN_IC50`` <-> pIC50, pivot
  sensitivity matrices, summarise compounds and call
  sensitive/resistant cell lines.
* :mod:`pharmaml.biology.omics` -- gene-expression feature tables
  (rows = cell lines, columns = genes, matching the ML ``X``
  convention): log2/z-score normalisation, per-gene imputation,
  MAD-based feature selection and per-gene correlation with one
  drug's response.
* :mod:`pharmaml.biology.biomarkers` -- the biomarker-discovery call
  :func:`sensitivity_associations` (every gene vs. one drug, with
  Benjamini-Hochberg FDR control), plus :func:`top_biomarkers` and
  :func:`direction_of_association`.

Typical flow::

    from pharmaml.biology import (
        collapse_replicates, load_response_table, response_matrix,
        normalize_expression, impute_expression, sensitivity_associations,
        top_biomarkers,
    )

    screen = collapse_replicates(load_response_table("screen.csv"))
    pic50 = response_matrix(screen, "LN_IC50").apply(ln_ic50_to_pic50)
    expression = impute_expression(normalize_expression(expr_frame))
    hits = top_biomarkers(sensitivity_associations(expression, pic50["drug"]))

Direction conventions: ``LN_IC50``-style metrics read *lower = more
sensitive*, pIC50-style metrics read *higher = more sensitive*; every
direction-sensitive function documents its assumption and exposes a
switch.
"""

from pharmaml.biology.biomarkers import (
    direction_of_association,
    sensitivity_associations,
    top_biomarkers,
)
from pharmaml.biology.omics import (
    correlate_with_response,
    impute_expression,
    most_variable_genes,
    normalize_expression,
)
from pharmaml.biology.response import (
    ResponseScreen,
    audit_response_table,
    call_sensitive_resistant,
    collapse_replicates,
    ln_ic50_to_pic50,
    load_response_table,
    pic50_to_ln_ic50,
    response_matrix,
    summarize_compounds,
)

__all__ = [
    # response.py
    "ResponseScreen",
    "load_response_table",
    "audit_response_table",
    "collapse_replicates",
    "ln_ic50_to_pic50",
    "pic50_to_ln_ic50",
    "response_matrix",
    "summarize_compounds",
    "call_sensitive_resistant",
    # omics.py
    "normalize_expression",
    "impute_expression",
    "most_variable_genes",
    "correlate_with_response",
    # biomarkers.py
    "sensitivity_associations",
    "top_biomarkers",
    "direction_of_association",
]
