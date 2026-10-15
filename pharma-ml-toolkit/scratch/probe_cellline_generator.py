"""Smoke: planted EGFR biomarker recovery from the synthetic generator."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from make_synthetic_cellline_data import (  # noqa: E402
    make_response_screen,
    make_expression_matrix,
)
from pharmaml.biology import (  # noqa: E402
    load_response_table,
    collapse_replicates,
    sensitivity_associations,
    top_biomarkers,
)
from pharmaml.biology.omics import normalize_expression  # noqa: E402

screen = make_response_screen()
expr = make_expression_matrix()
s = load_response_table(screen, tissue_column="tissue")
clean = collapse_replicates(s)
expr_norm = normalize_expression(expr)
erlo = clean.frame[clean.frame["compound"] == "erlotinib"].set_index("cell_line")["LN_IC50"]
assoc = sensitivity_associations(expr_norm, erlo)
top = top_biomarkers(assoc)
print("top-3 genes for erlotinib:", list(top["gene"][:3]))
egfr = assoc[assoc["gene"] == "EGFR"].iloc[0]
print(
    "EGFR: rho=%+.3f p=%.2e fdr=%.2e sig=%s"
    % (egfr["rho"], egfr["p_value"], egfr["fdr"], egfr["significant"])
)
