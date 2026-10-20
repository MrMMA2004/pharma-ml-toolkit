"""DATABASE AUDIT (100x-103): every example-data file checked for gaps and errors.

Independently verifies (no reliance on the builder's own checks):
  GAPS    - missing/empty cells, ragged rows, duplicate columns, coverage
            (cellline screen vs expression, campaign folders vs ligands.csv)
  ERRORS  - duplicate rows/ids, invalid SMILES (vs the planted-invalid design),
            label censuses vs notebook pins, numeric ranges (pIC50 assay
            window, LN_IC50, descriptors), non-finite values, PK time
            monotonicity + AUC pins, PDB parseability, seed_molecules byte
            pin, trailing-newline/whitespace/CRLF conventions.

Designed data quirks (documented audit cases, NOT errors):
  - screening_library.csv: 2 planted invalid SMILES (not_a_smiles, C1CC)
  - ligand_library.csv: 1 planted invalid (not_a_smiles) + 1 empty SMILES cell
  - cellline_screen.csv: 1 planted duplicate replicate + 1 planted NaN
  - docking_campaign: 12 complete runs + 1 MISSING_LOG + 1 INCOMPLETE
  - walkthrough_compounds.csv: 12-vocab SMILES resampled across 180 rows

Prints one line per check: PASS/FAIL(+detail). Exit code 1 on any FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"

FAILURES: list[str] = []
CHECKS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)
    if not ok:
        FAILURES.append(line)


def audit_generic_csv(path: Path, expect_nan_cols: dict[str, int] | None = None) -> pd.DataFrame:
    """Structural audit: ragged rows, dup columns, empty cells (designed ones exempt)."""
    expect_nan_cols = expect_nan_cols or {}
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    header = lines[0].split(",")
    ncol = len(header)
    name = path.name

    dups = {c for c in header if header.count(c) > 1}
    check(f"{name}: unique column names", not dups, f"dupes={dups}")

    ragged = [
        (i + 1, len(ln.split(",")))
        for i, ln in enumerate(lines[1:])
        if ln.strip() and len(ln.split(",")) != ncol
    ]
    check(f"{name}: no ragged rows", not ragged, f"ragged={ragged[:5]}")

    blank = [i + 1 for i, ln in enumerate(lines[1:]) if not ln.strip() and i + 1 < len(lines) - 1]
    check(f"{name}: no interior blank lines", not blank, f"blank_lines={blank}")

    has_nl = raw.endswith("\n")
    if name == "seed_molecules.csv":
        check(f"{name}: NO trailing newline (pinned byte size)", not has_nl)
    else:
        check(f"{name}: single trailing newline", has_nl and not raw.endswith("\n\n"))

    df = pd.read_csv(path)
    nan_cells = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    check(
        f"{name}: missing cells == designed {expect_nan_cols}",
        nan_cells == expect_nan_cols,
        f"got={nan_cells}",
    )
    return df


def audit_smiles(df: pd.DataFrame, col: str, name: str, expect_invalid: set[str], expect_nan: int = 0) -> None:
    invalid = [str(s) for s in df[col].dropna() if Chem.MolFromSmiles(str(s)) is None]
    got = set(invalid)
    nan_count = int(df[col].isna().sum())
    check(
        f"{name}: SMILES validity ({len(df)} rows, {len(got)} invalid, {nan_count} empty, expected {len(expect_invalid)}/{expect_nan})",
        got == expect_invalid and nan_count == expect_nan,
        f"unexpected={sorted(got - expect_invalid)} missing_planted={sorted(expect_invalid - got)} nan={nan_count}",
    )


def audit_numeric(df: pd.DataFrame, cols: list[str], name: str, lo: float, hi: float) -> None:
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()  # designed NaN handled by the structural audit
        nonfinite = int((~np.isfinite(vals)).sum())
        if nonfinite:
            check(f"{name}: {c} finite", False, f"nonfinite={nonfinite}")
            continue
        out_of_range = int(((vals < lo) | (vals > hi)).sum())
        check(f"{name}: {c} in [{lo:g}, {hi:g}]", out_of_range == 0, f"out={out_of_range}")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    print("=" * 78)
    print("DATABASE AUDIT -- examples/data + tests/data")
    print("=" * 78)

    # ---------------- QSAR classification ----------------
    df = audit_generic_csv(DATA / "qsar_classification.csv")
    check("qsar_classification: 420 rows", len(df) == 420, f"rows={len(df)}")
    census = df["activity"].value_counts().to_dict()
    check(
        "qsar_classification: label census {active:118, weak:155, inactive:147}",
        census == {"active": 118, "weak": 155, "inactive": 147},
        f"census={census}",
    )
    check("qsar_classification: compound_id unique", df["compound_id"].is_unique)
    check("qsar_classification: no full-row duplicates", not df.duplicated().any())
    audit_numeric(df, [c for c in df.columns if c.startswith("descr_")], "qsar_classification", -8, 8)
    audit_smiles(df, "SMILES", "qsar_classification", set())

    # ---------------- QSAR regression (assay-plausible window) ----------------
    df = audit_generic_csv(DATA / "qsar_regression.csv")
    check("qsar_regression: 260 rows", len(df) == 260, f"rows={len(df)}")
    audit_numeric(df, [c for c in df.columns if c.startswith("descr_")], "qsar_regression", -9, 9)
    audit_numeric(df, ["pIC50"], "qsar_regression", 3, 11)
    check("qsar_regression: no full-row duplicates", not df.duplicated().any())

    # ---------------- QSAR multiclass ----------------
    df = audit_generic_csv(DATA / "qsar_multiclass.csv")
    check("qsar_multiclass: 360 rows", len(df) == 360, f"rows={len(df)}")
    census = df["Target"].value_counts().to_dict()
    check(
        "qsar_multiclass: label census {inactive:125, weak:102, moderate:79, strong:54}",
        census == {"inactive": 125, "weak": 102, "moderate": 79, "strong": 54},
        f"census={census}",
    )
    audit_numeric(df, [c for c in df.columns if c.startswith("descr_")], "qsar_multiclass", -8, 8)

    # ---------------- walkthrough family ----------------
    df = audit_generic_csv(DATA / "walkthrough_compounds.csv")
    check("walkthrough_compounds: 180 rows", len(df) == 180, f"rows={len(df)}")
    census = df["activity"].value_counts().to_dict()
    check(
        "walkthrough_compounds: label census {active:39, inactive:88, weak:53}",
        census == {"active": 39, "inactive": 88, "weak": 53},
        f"census={census}",
    )
    audit_smiles(df, "SMILES", "walkthrough_compounds", set())
    seed_vocab = set(pd.read_csv(DATA / "seed_molecules.csv")["smiles"])
    check(
        "walkthrough_compounds: SMILES == the 12-molecule seed vocabulary",
        set(df["SMILES"]) == seed_vocab and df["SMILES"].nunique() == 12,
        f"unique={df['SMILES'].nunique()}",
    )
    audit_numeric(df, [c for c in df.columns if c.startswith("descr_")], "walkthrough_compounds", -8, 8)

    df = audit_generic_csv(DATA / "walkthrough_classification.csv")
    check("walkthrough_classification: 300 rows", len(df) == 300, f"rows={len(df)}")
    check(
        "walkthrough_classification: Target is 0/1 binary",
        set(df["Target"].unique()) <= {0, 1},
        f"values={sorted(df['Target'].unique())}",
    )
    audit_numeric(df, [c for c in df.columns if c.startswith("descr_")], "walkthrough_classification", -8, 8)

    df = audit_generic_csv(DATA / "walkthrough_multiclass.csv")
    check("walkthrough_multiclass: 300 rows", len(df) == 300, f"rows={len(df)}")
    check(
        "walkthrough_multiclass: Target in 0..2 (3 classes)",
        set(df["Target"].unique()) <= {0, 1, 2},
        f"values={sorted(df['Target'].unique())}",
    )

    # ---------------- screening library (2 planted invalid) ----------------
    df = audit_generic_csv(DATA / "screening_library.csv")
    check("screening_library: 79 rows", len(df) == 79, f"rows={len(df)}")
    check("screening_library: name unique", df["name"].is_unique)
    audit_smiles(df, "SMILES", "screening_library", {"not_a_smiles", "C1CC"})

    # ---------------- ligand library (2 planted invalid rows) ----------------
    df = audit_generic_csv(DATA / "ligand_library.csv", expect_nan_cols={"SMILES": 1})
    check("ligand_library: 10 rows", len(df) == 10, f"rows={len(df)}")
    check("ligand_library: name unique", df["name"].is_unique)
    audit_smiles(df, "SMILES", "ligand_library", {"not_a_smiles"}, expect_nan=1)

    # ---------------- library with descriptors ----------------
    df = audit_generic_csv(DATA / "library_with_descriptors.csv")
    check("library_with_descriptors: 77 rows (the 79 - 2 invalid)", len(df) == 77, f"rows={len(df)}")
    audit_smiles(df, "SMILES", "library_with_descriptors", set())
    census = df["activity"].value_counts().to_dict()
    check(
        "library_with_descriptors: activity labels consistent set",
        set(census) == {"active", "inactive"},
        f"census={census}",
    )
    check(
        "library_with_descriptors: MolWt in [60, 900] (real small molecules incl.)",
        df["MolWt"].between(60, 900).all(),
        f"min={df['MolWt'].min()} max={df['MolWt'].max()}",
    )
    check(
        "library_with_descriptors: MolLogP in [-2, 8]",
        df["MolLogP"].between(-2, 8).all(),
        f"min={df['MolLogP'].min()} max={df['MolLogP'].max()}",
    )
    check(
        "library_with_descriptors: TPSA in [0, 250]",
        df["TPSA"].between(0, 250).all(),
        f"min={df['TPSA'].min()} max={df['TPSA'].max()}",
    )
    int_cols = ["NumHDonors", "NumHAcceptors", "NumRotatableBonds", "RingCount"]
    for c in int_cols:
        check(
            f"library_with_descriptors: {c} non-negative integer",
            (df[c] >= 0).all() and float(df[c].mod(1).abs().max()) == 0.0,
            f"min={df[c].min()}",
        )
    screen = pd.read_csv(DATA / "screening_library.csv")
    valid_screen = set(screen.loc[~screen["SMILES"].isin({"not_a_smiles", "C1CC"}), "SMILES"])
    check(
        "library_with_descriptors: SMILES == screening_library's 77 valid",
        set(df["SMILES"]) == valid_screen,
        f"only_in_desc={sorted(set(df['SMILES']) - valid_screen)[:3]} only_in_screen={sorted(valid_screen - set(df['SMILES']))[:3]}",
    )
    from rdkit.Chem import Descriptors as RD

    bad: list[str] = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES"])
        assert mol is not None
        if abs(RD.MolWt(mol) - row["MolWt"]) > 0.02:
            bad.append(f"{row['SMILES']}: MolWt {row['MolWt']} vs RDKit {RD.MolWt(mol):.2f}")
        if abs(RD.MolLogP(mol) - row["MolLogP"]) > 0.02:
            bad.append(f"{row['SMILES']}: MolLogP {row['MolLogP']} vs RDKit {RD.MolLogP(mol):.2f}")
        if abs(RD.TPSA(mol) - row["TPSA"]) > 0.1:
            bad.append(f"{row['SMILES']}: TPSA {row['TPSA']} vs RDKit {RD.TPSA(mol):.2f}")
    check("library_with_descriptors: descriptors match RDKit recomputation", not bad, f"bad={bad[:5]}")

    # ---------------- cellline screen + expression (planted duplicate + NaN) ----------------
    screen = audit_generic_csv(DATA / "cellline_screen.csv", expect_nan_cols={"LN_IC50": 1})
    check("cellline_screen: 185 rows", len(screen) == 185, f"rows={len(screen)}")
    check(
        "cellline_screen: 23 cell lines / 11 tissues",
        screen["cell_line"].nunique() == 23 and screen["tissue"].nunique() == 11,
        f"lines={screen['cell_line'].nunique()} tissues={screen['tissue'].nunique()}",
    )
    per_line = screen.groupby("cell_line")["tissue"].nunique()
    check("cellline_screen: tissue unique per cell line", (per_line == 1).all())
    audit_numeric(screen, ["LN_IC50"], "cellline_screen", -7, 7)  # ln(IC50 µM): 1 nM .. 1 mM
    dupe = screen[screen.duplicated(subset=["cell_line", "compound"], keep=False)]
    check(
        "cellline_screen: exactly 1 planted duplicate pair (assay repeat)",
        len(dupe) == 2 and dupe["cell_line"].nunique() == 1,
        f"dupe_rows={len(dupe)}",
    )

    expr = audit_generic_csv(DATA / "cellline_expression.csv")
    check("cellline_expression: 23 rows", len(expr) == 23, f"rows={len(expr)}")
    check(
        "cellline_expression: 300 feature columns (298 GENE + EGFR + KRT8)",
        len(expr.columns) == 301,
        f"cols={len(expr.columns)}",
    )
    check("cellline_expression: cell_line unique", expr["cell_line"].is_unique)
    audit_numeric(expr, [c for c in expr.columns if c != "cell_line"], "cellline_expression", 0, 14)
    missing_expr = sorted(set(screen["cell_line"]) - set(expr["cell_line"]))
    check(
        "cellline cross-coverage: every screened line has expression",
        not missing_expr,
        f"missing={missing_expr}",
    )
    orphan_expr = sorted(set(expr["cell_line"]) - set(screen["cell_line"]))
    check(
        "cellline cross-coverage: no orphan expression rows",
        not orphan_expr,
        f"orphans={orphan_expr}",
    )

    # ---------------- PK profile ----------------
    pk = audit_generic_csv(DATA / "pk_profile.csv")
    check("pk_profile: 49 rows (t=0..24 by 0.5)", len(pk) == 49, f"rows={len(pk)}")
    t = pk["time_h"].to_numpy()
    check("pk_profile: time strictly increasing", bool((np.diff(t) > 0).all()))
    check(
        "pk_profile: t[0]=0, t[-1]=24, step 0.5",
        t[0] == 0.0 and t[-1] == 24.0 and abs(np.diff(t).max() - 0.5) < 1e-12,
    )
    c = pk["conc_ng_ml"].to_numpy()
    check("pk_profile: concentrations non-negative", bool((c >= 0).all()), f"min={c.min()}")
    from pharmaml.pharma.pk import auc_trapezoid

    auc_lin = float(auc_trapezoid(t, c))
    pos = c > 0
    auc_log = float(auc_trapezoid(t[pos], c[pos], method="log"))
    check("pk_profile: AUC linear pin 17.9607", abs(auc_lin - 17.9607) < 5e-4, f"got={auc_lin:.4f}")
    check("pk_profile: AUC log pin 17.2536 (log-trapezoid, positive points)", abs(auc_log - 17.2536) < 5e-4, f"got={auc_log:.4f}")

    # ---------------- seed molecules ----------------
    raw = (DATA / "seed_molecules.csv").read_bytes()
    check("seed_molecules: 443 B pin", len(raw) == 443, f"bytes={len(raw)}")
    seed = audit_generic_csv(DATA / "seed_molecules.csv")
    check("seed_molecules: 12 rows", len(seed) == 12, f"rows={len(seed)}")
    check("seed_molecules: name unique", seed["name"].is_unique)
    audit_smiles(seed, "smiles", "seed_molecules", set())

    # ---------------- docking campaign ----------------
    camp = DATA / "docking_campaign"
    lig = audit_generic_csv(camp / "ligands.csv")
    check("campaign ligands.csv: 14 rows", len(lig) == 14, f"rows={len(lig)}")
    audit_smiles(lig, "SMILES", "campaign ligands.csv", set())
    folders = sorted(p for p in camp.iterdir() if p.is_dir())
    check("campaign: 14 ligand folders", len(folders) == 14, f"folders={len(folders)}")
    folder_names = {p.name.split("_", 1)[1] for p in folders}
    check(
        "campaign: folder names match ligands.csv names",
        folder_names == set(lig["name"]),
        f"only_folders={sorted(folder_names - set(lig['name']))} only_lig={sorted(set(lig['name']) - folder_names)}",
    )
    statuses = {}
    for p in folders:
        dlg, glg, readme = p / "r.dlg", p / "r.glg", p / "README.md"
        if dlg.is_file() and dlg.stat().st_size == 682 and glg.is_file():
            statuses[p.name] = "OK"
        elif readme.is_file() and not dlg.is_file():
            statuses[p.name] = "MISSING_LOG"
        elif dlg.is_file() and dlg.stat().st_size == 65:
            statuses[p.name] = "INCOMPLETE"
        else:
            statuses[p.name] = "UNEXPECTED"
    tally: dict[str, int] = {}
    for v in statuses.values():
        tally[v] = tally.get(v, 0) + 1
    check(
        "campaign: 12 OK + 1 MISSING_LOG + 1 INCOMPLETE (planted audit cases)",
        tally == {"OK": 12, "MISSING_LOG": 1, "INCOMPLETE": 1},
        f"tally={tally}",
    )
    from pharmaml.docking.parsing import parse_dlg_summary

    summary = parse_dlg_summary(camp / "01_aspirin" / "r.dlg")
    expect = {
        "n_clusters": 5,
        "top_rank": 1,
        "top_lowest_energy": -4.91,
        "top_mean_energy": -4.74,
        "top_num_in_cluster": 15,
    }
    if isinstance(summary, dict):
        got = {k: summary[k] for k in expect}
    else:
        got = {
            "n_clusters": summary.n_clusters,
            "top_rank": summary.top_rank,
            "top_lowest_energy": summary.top_lowest_energy,
            "top_mean_energy": summary.top_mean_energy,
            "top_num_in_cluster": summary.top_num_in_cluster,
        }
    check("campaign: aspirin dlg summary pin", got == expect, f"got={got}")

    # ---------------- PDB files ----------------
    for path, pin_bytes in [
        (DATA / "lysozyme_1aki.pdb", 116_397),
        (ROOT / "tests" / "data" / "1crn.pdb", 49_491),
    ]:
        ok_bytes = path.stat().st_size == pin_bytes
        check(f"{path.name}: byte pin {pin_bytes}", ok_bytes, f"bytes={path.stat().st_size}")
        mol = Chem.MolFromPDBFile(str(path), sanitize=False, removeHs=False)
        check(
            f"{path.name}: RDKit-parseable PDB",
            mol is not None and mol.GetNumAtoms() > 0,
            f"atoms={0 if mol is None else mol.GetNumAtoms()}",
        )

    # ---------------- conventions: whitespace, CRLF, encoding ----------------
    for path in sorted(DATA.rglob("*.csv")):
        df2 = pd.read_csv(path, dtype=str, keep_default_na=False)
        ws = [(col, val) for col in df2.columns for val in df2[col] if val != val.strip()]
        check(f"{path.name}: no leading/trailing whitespace in cells", not ws, f"ws={ws[:4]}")
        raw2 = path.read_bytes()
        check(f"{path.name}: LF-only, utf-8 clean", b"\r" not in raw2 and raw2.decode("utf-8") is not None)

    print("=" * 78)
    print(f"RESULT: {CHECKS - len(FAILURES)}/{CHECKS} checks passed, {len(FAILURES)} failed")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("DATABASE CLEAN: no gaps, no errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
