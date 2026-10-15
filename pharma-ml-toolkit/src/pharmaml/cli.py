"""Command-line interface — the scaffolding entry point of PharmaML.

Replaces the legacy workflow of "open a per-algorithm .py file, edit
the hardcoded CSV filename at the bottom, run it".  One command covers
training, comparison, splitting, conversion, PK analysis and docking
scans::

    pharmaml ml train    --data qsar.csv --target Target --model rf --plots
    pharmaml ml train    --data qsar.csv --task auto        # infer classification/regression
    pharmaml ml train    --data qsar.csv --keep-categorical  # one-hot string features
    pharmaml ml compare  --data qsar.csv --target Target --task classification --plot
    pharmaml data inspect --data qsar.csv                  # audit your own file
    pharmaml data split  --data qsar.csv --test-size 0.2 --val-size 0.1
    pharmaml data convert --data ligands.csv --to sdf      # .smi/.csv/.sdf round-trip
    pharmaml chem pic50  --data activity.xlsx --column "IC50 (nM)"
    pharmaml pharma pk   --data profile.csv --time time_h --conc conc_ng_ml
    pharmaml docking prep-ligands --ligands ligands.csv --out ligand_folders/
    pharmaml docking prep-protein --protein protein.pdb --out r.pdbqt
    pharmaml docking scan --directory docking_run/
    pharmaml docking dpf --ligand ligand_folders/1/l.pdbqt --ga-runs 50
    pharmaml docking collect --dir docking_run/ --pattern '^\\d+\\.pdbqt$'
    pharmaml docking torsion --dir ligand_folders/
    pharmaml docking gpf-scan --dir docking_run/
    pharmaml docking vina-scan --dir vina_results/
    pharmaml biology response-audit --data screen.csv --metric LN_IC50
    pharmaml biology top-biomarkers --expression expr.csv --response resp.csv --response-column pIC50
    pharmaml models list

``--model`` accepts registry names and their aliases (``rf``,
``logistic``, ``xgb`` ... — see ``pharmaml.ml.MODEL_ALIASES``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pharmaml import __version__
from pharmaml.logger import configure_logging, get_logger

_LOG = get_logger("cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharmaml",
        description="PharmaML toolkit — QSAR ML, pharma calculations, docking.",
    )
    parser.add_argument("--version", action="version", version=f"pharmaml {__version__}")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    parser.add_argument(
        "--log-file", type=Path, default=None, help="also write logs to this file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- models list ------------------------------------------------
    models_parser = subparsers.add_parser("models", help="model registry utilities")
    models_parser.add_argument("action", choices=["list"], help="list available models")

    # ---- data --------------------------------------------------------
    data_parser = subparsers.add_parser(
        "data", help="bring-your-own-data utilities (inspect user files)"
    )
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)

    inspect = data_sub.add_parser(
        "inspect", help="audit a user CSV/Excel file before training"
    )
    inspect.add_argument("--data", required=True, help="CSV/TSV/Excel file to inspect")
    inspect.add_argument(
        "--target", default=None, help="target column (auto-detected when omitted)"
    )
    inspect.add_argument(
        "--head", type=int, default=0, metavar="N",
        help="also print the first N rows of the table",
    )

    split = data_sub.add_parser(
        "split",
        help="split a table into stratified train/val/test files "
        "(upgrades 'Split CSV file.py')",
    )
    split.add_argument("--data", required=True, help="CSV/TSV/Excel file to split")
    split.add_argument(
        "--target", default=None,
        help="column to stratify on (auto-detected when omitted)",
    )
    split.add_argument("--test-size", type=float, default=0.2, metavar="FRAC")
    split.add_argument(
        "--val-size", type=float, default=None, metavar="FRAC",
        help="also carve a validation set of this fraction",
    )
    split.add_argument(
        "--no-stratify", action="store_true",
        help="plain random split (no class balancing)",
    )
    split.add_argument("--seed", type=int, default=42)
    split.add_argument(
        "--out-dir", default=None,
        help="output directory (default: alongside --data)",
    )
    split.add_argument(
        "--prefix", default=None,
        help="output filename prefix (default: input file stem)",
    )

    convert = data_sub.add_parser(
        "convert",
        help="convert molecular file formats (.smi / .csv / .tsv / .xlsx / .sdf)",
    )
    convert.add_argument(
        "--data", required=True,
        help="input file: .smi, a CSV/TSV/Excel table, or .sdf",
    )
    convert.add_argument(
        "--to", required=True, choices=("csv", "sdf", "smi"),
        help="target format",
    )
    convert.add_argument("--smiles-col", default="SMILES", help="SMILES column name")
    convert.add_argument(
        "--name-col", default=None,
        help="molecule name column (default: none -> ligand_N titles)",
    )
    convert.add_argument(
        "--no-3d", action="store_true",
        help="skip 3D embedding + UFF optimisation (table -> SDF only)",
    )
    convert.add_argument(
        "--out", default=None,
        help="output path (default: input stem + target extension)",
    )

    # ---- ml ----------------------------------------------------------
    ml_parser = subparsers.add_parser("ml", help="machine-learning workflows")
    ml_sub = ml_parser.add_subparsers(dest="ml_command", required=True)

    train = ml_sub.add_parser("train", help="train one model end-to-end")
    train.add_argument("--data", required=True, help="CSV/Excel with features+target")
    train.add_argument("--target", default=None, help="target column (auto-detected)")
    train.add_argument(
        "--model", default="random_forest",
        help="registry model name or alias (rf, logistic, xgb, ...)",
    )
    train.add_argument(
        "--task", choices=["classification", "regression", "auto"], default="auto",
        help="task type; 'auto' infers it from the target column (default)",
    )
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv", type=int, default=5)
    train.add_argument("--no-tune", action="store_true", help="skip grid search")
    train.add_argument(
        "--keep-categorical", action="store_true",
        help="keep string feature columns and one-hot encode them "
        "(default: numeric columns only)",
    )
    train.add_argument(
        "--plots", action="store_true",
        help="emit the full visual bundle (report.html + figures + "
        "tables) into <out>/report/ — requires pharmaml[reports]",
    )
    train.add_argument("--out", default="artifacts", help="artifact output directory")

    compare = ml_sub.add_parser("compare", help="cross-validated model leaderboard")
    compare.add_argument("--data", required=True)
    compare.add_argument("--target", default=None)
    compare.add_argument(
        "--task", choices=["classification", "regression", "auto"], default="auto"
    )
    compare.add_argument("--cv", type=int, default=5)
    compare.add_argument("--models", nargs="*", default=None, help="subset of models")
    compare.add_argument(
        "--keep-categorical", action="store_true",
        help="keep string feature columns and one-hot encode them",
    )
    compare.add_argument(
        "--out", default=None,
        help="write the leaderboard table to this path (csv/tsv/md/json/html)",
    )
    compare.add_argument(
        "--plot", action="store_true",
        help="save the leaderboard bar chart + report bundle into <out-dir>/ "
        "— requires pharmaml[reports]",
    )
    compare.add_argument(
        "--out-dir", default="reports", help="directory for --plot artifacts"
    )

    # ---- chem --------------------------------------------------------
    chem_parser = subparsers.add_parser("chem", help="cheminformatics utilities")
    chem_sub = chem_parser.add_subparsers(dest="chem_command", required=True)

    pic50 = chem_sub.add_parser("pic50", help="add a pIC50 column to a table")
    pic50.add_argument("--data", required=True)
    pic50.add_argument("--column", default="IC50 (nM)")
    pic50.add_argument("--unit", default="nM", choices=["M", "mM", "uM", "nM", "pM"])
    pic50.add_argument("--out", default=None, help="output path (default: <stem>_pic50.csv)")

    descriptors = chem_sub.add_parser(
        "descriptors", help="compute RDKit descriptors for a SMILES table"
    )
    descriptors.add_argument("--data", required=True)
    descriptors.add_argument("--smiles-col", default="SMILES")
    descriptors.add_argument(
        "--set", dest="descriptor_set",
        choices=["rdkit_all", "morgan", "mordred"], default="rdkit_all",
    )
    descriptors.add_argument("--out", default=None)

    chem_compare = chem_sub.add_parser(
        "compare", help="reconcile a ligand column across two tables"
    )
    chem_compare.add_argument("--data-a", required=True, help="first table (csv/xlsx)")
    chem_compare.add_argument("--data-b", required=True, help="second table (csv/xlsx)")
    chem_compare.add_argument("--column", required=True, help="ligand-name column")
    chem_compare.add_argument(
        "--keep-case", action="store_true",
        help="do not fold names to lowercase before comparing",
    )
    chem_compare.add_argument(
        "--keep-extension", action="store_true",
        help="do not strip file extensions (.pdbqt, ...) before comparing",
    )
    chem_compare.add_argument(
        "--out", default=None, help="optional CSV path for the differences report"
    )
    chem_compare.add_argument(
        "--figure", default=None,
        help="optional figure path (png/svg/pdf) for the overlap breakdown",
    )

    chem_merge = chem_sub.add_parser(
        "merge", help="merge every table of one folder into a single file"
    )
    chem_merge.add_argument("--dir", required=True, help="folder to scan (non-recursive)")
    chem_merge.add_argument("--extension", default=".csv")
    chem_merge.add_argument(
        "--first-columns", default=None,
        help="comma-separated column names to move to the front",
    )
    chem_merge.add_argument("--out", default=None, help="output path (default: <dir>/merged_output.csv)")

    # ---- pharma ------------------------------------------------------
    pharma_parser = subparsers.add_parser("pharma", help="pharmaceutical calculations")
    pharma_sub = pharma_parser.add_subparsers(dest="pharma_command", required=True)

    pk = pharma_sub.add_parser("pk", help="non-compartmental PK summary")
    pk.add_argument("--data", required=True, help="CSV with time & concentration columns")
    pk.add_argument("--time", default="time", help="time column name")
    pk.add_argument("--conc", default="conc", help="concentration column name")
    pk.add_argument("--out", default=None, help="optional JSON output path")

    dose = pharma_sub.add_parser("dose", help="design a dosing regimen")
    dose.add_argument("--cl", type=float, required=True, help="clearance (L/h)")
    dose.add_argument("--vd", type=float, required=True, help="volume (L)")
    dose.add_argument("--target-css", type=float, required=True, help="target avg Css")
    dose.add_argument("--tau", type=float, default=12.0, help="interval (h)")
    dose.add_argument("--f", type=float, default=1.0, help="bioavailability fraction")

    approvals = pharma_sub.add_parser(
        "approvals", help="collect approved drug names from DGIdb-style exports"
    )
    approvals.add_argument(
        "--data", required=True, action="append",
        help="csv/tsv export (repeat the flag for several files)",
    )
    approvals.add_argument("--drug-column", default="drug")
    approvals.add_argument("--approval-column", default="regulatory approval")
    approvals.add_argument("--status", default="approved")
    approvals.add_argument("--out", default=None, help="optional single-column CSV output")

    # ---- docking -----------------------------------------------------
    docking_parser = subparsers.add_parser("docking", help="docking utilities")
    docking_sub = docking_parser.add_subparsers(dest="docking_command", required=True)

    scan = docking_sub.add_parser("scan", help="summarise a docking run directory")
    scan.add_argument("--directory", required=True)
    scan.add_argument("--log-name", default="r.dlg")
    scan.add_argument("--out", default=None, help="output CSV path")

    check = docking_sub.add_parser("check", help="audit docking completion")
    check.add_argument("--directory", required=True)
    check.add_argument("--log-name", default="r.dlg")
    check.add_argument("--out", default=None)

    prep_ligands = docking_sub.add_parser(
        "prep-ligands",
        help="import a ligand CSV and prepare PDBQT folders",
    )
    prep_ligands.add_argument("--ligands", required=True, help="ligand table (CSV/TSV/Excel)")
    prep_ligands.add_argument("--out", required=True, help="campaign output directory")
    prep_ligands.add_argument("--smiles-col", default=None, help="SMILES column (auto-detected)")
    prep_ligands.add_argument("--name-col", default=None, help="ligand name column (auto-detected)")
    prep_ligands.add_argument(
        "--naming",
        default="index_name",
        choices=("index", "index_name", "name"),
        help="folder naming scheme (index reproduces the legacy scripts)",
    )

    prep_protein = docking_sub.add_parser(
        "prep-protein",
        help="import a protein PDB and prepare a receptor PDBQT",
    )
    prep_protein.add_argument("--protein", required=True, help="protein file (.pdb or .pdbqt)")
    prep_protein.add_argument("--out", default=None, help="receptor PDBQT path (default <stem>.pdbqt)")
    prep_protein.add_argument("--keep-hetatm", action="store_true", help="keep non-water HETATM records")
    prep_protein.add_argument("--chain", default=None, help="keep only this chain (e.g. A)")
    prep_protein.add_argument(
        "--altloc",
        default="first",
        help="alternate-location conformer: 'first' or a letter like A",
    )

    dpf = docking_sub.add_parser(
        "dpf",
        help="write an AutoDock4 docking parameter file (r.dpf) for a ligand",
    )
    dpf.add_argument("--ligand", required=True, help="ligand PDBQT to dock")
    dpf.add_argument(
        "--receptor", default="r.pdbqt",
        help="receptor PDBQT filename (parity with the GPF generator)",
    )
    dpf.add_argument("--ga-runs", type=int, default=100, help="number of GA runs (ga_run)")
    dpf.add_argument("--out", default=None, help="DPF path (default <ligand_dir>/r.dpf)")

    collect = docking_sub.add_parser(
        "collect",
        help="collect matching files (r.dlg dialogs, numbered PDBQTs) into one folder",
    )
    collect.add_argument("--dir", required=True, help="campaign directory to scan recursively")
    collect.add_argument(
        "--filename",
        default=None,
        help="exact filename to collect (default r.dlg when no pattern is given)",
    )
    collect.add_argument(
        "--pattern",
        default=None,
        help="regex the filename must match, e.g. '^\\d+\\.pdbqt$'",
    )
    collect.add_argument("--out", default=None, help="output folder (default <dir>/collected)")
    collect.add_argument(
        "--dry-run", action="store_true",
        help="report the copies without performing them",
    )

    torsion = docking_sub.add_parser(
        "torsion",
        help="audit TORSDOF / active REMARK / BRANCH consistency of ligand PDBQTs",
    )
    torsion.add_argument("--dir", required=True, help="directory tree with .pdbqt ligands")

    gpf_scan = docking_sub.add_parser(
        "gpf-scan",
        help="report the ligand_types line of every sub-directory's r.gpf",
    )
    gpf_scan.add_argument("--dir", required=True, help="campaign directory")

    vina_scan = docking_sub.add_parser(
        "vina-scan",
        help="rank Vina .log files of a results tree by best affinity",
    )
    vina_scan.add_argument("--dir", required=True, help="directory tree with Vina logs")

    campaign = docking_sub.add_parser(
        "campaign",
        help="campaign-tree file management (distribute / cleanup / delete-named / list / reindex)",
    )
    campaign_sub = campaign.add_subparsers(dest="campaign_command", required=True)

    campaign_list = campaign_sub.add_parser(
        "list", help="list file stems of one folder (optionally to a CSV)"
    )
    campaign_list.add_argument("--dir", required=True, help="folder whose files are listed")
    campaign_list.add_argument(
        "--ext", default=".csv", help="file extension to filter by (default .csv)"
    )
    campaign_list.add_argument(
        "--out", default=None, help="optional single-column CSV (name header)"
    )

    campaign_distribute = campaign_sub.add_parser(
        "distribute",
        help="copy files into every immediate sub-directory (receptor/binaries push)",
    )
    campaign_distribute.add_argument(
        "--dir", required=True, help="target directory whose sub-directories receive copies"
    )
    campaign_distribute.add_argument(
        "--file",
        action="append",
        required=True,
        dest="files",
        help="source file to distribute (repeatable)",
    )
    campaign_distribute.add_argument(
        "--overwrite", action="store_true", help="replace existing destination files"
    )

    campaign_cleanup = campaign_sub.add_parser(
        "cleanup",
        help="delete every file not on the keep-list (allowlist + regex)",
    )
    campaign_cleanup.add_argument("--dir", required=True, help="tree to clean")
    campaign_cleanup.add_argument(
        "--keep",
        action="append",
        default=[],
        help="filename to preserve (repeatable)",
    )
    campaign_cleanup.add_argument(
        "--keep-pattern",
        default=None,
        help="regex filenames must match to survive, e.g. '^\\d+\\.pdbqt$'",
    )
    campaign_cleanup.add_argument(
        "--remove-empty",
        action="store_true",
        help="also remove directories left (or already) empty",
    )
    campaign_cleanup.add_argument(
        "--dry-run", action="store_true", help="preview without deleting anything"
    )

    campaign_delete = campaign_sub.add_parser(
        "delete-named",
        help="delete every file with one name anywhere under a tree",
    )
    campaign_delete.add_argument("--dir", required=True, help="tree to sweep")
    campaign_delete.add_argument("--name", required=True, help="filename to remove")
    campaign_delete.add_argument(
        "--dry-run", action="store_true", help="count without deleting"
    )

    campaign_reindex = campaign_sub.add_parser(
        "reindex",
        help="reindex a results table onto the full ligand-number range "
        "(missing ligands appear as blank rows instead of silently absent)",
    )
    campaign_reindex.add_argument(
        "--data", required=True, help="results table to reindex (CSV/XLSX)"
    )
    campaign_reindex.add_argument(
        "--column",
        required=True,
        help="column whose values embed the ligand number (e.g. '5_result.dlg')",
    )
    campaign_reindex.add_argument(
        "--start", type=int, default=0, help="first ligand number (default 0)"
    )
    campaign_reindex.add_argument(
        "--stop", type=int, default=300, help="last ligand number (default 300)"
    )
    campaign_reindex.add_argument(
        "--pattern",
        default=r"(\d+)",
        help="regex whose first capture group extracts the number (default r'(\\d+)')",
    )
    campaign_reindex.add_argument(
        "--out",
        default=None,
        help="optional output CSV/XLSX (index written as a 'number' column)",
    )

    # ---- biology -----------------------------------------------------
    biology_parser = subparsers.add_parser(
        "biology", help="cell line data utilities (drug-response screens, omics)"
    )
    biology_sub = biology_parser.add_subparsers(
        dest="biology_command", required=True
    )

    bio_audit = biology_sub.add_parser(
        "response-audit",
        help="audit a GDSC/CCLE-style drug-response table",
    )
    bio_audit.add_argument(
        "--data", required=True,
        help="long-format response table (CSV/TSV/Excel), one row per "
        "(cell line, compound) pair",
    )
    bio_audit.add_argument(
        "--cell-column", default="cell_line", help="cell line column name",
    )
    bio_audit.add_argument(
        "--compound-column", default="compound", help="compound column name",
    )
    bio_audit.add_argument(
        "--tissue-column", default=None,
        help="optional tissue annotation column (kept out of the metrics)",
    )
    bio_audit.add_argument(
        "--metric", default=None,
        help="metric for the per-compound summary (default: first metric)",
    )

    bio_top = biology_sub.add_parser(
        "top-biomarkers",
        help="correlate gene expression with one drug's response",
    )
    bio_top.add_argument(
        "--expression", required=True,
        help="expression table: rows = cell lines, columns = genes "
        "(first column may hold the cell-line identifiers)",
    )
    bio_top.add_argument(
        "--response", required=True,
        help="long-format response table (cell line, compound, metric)",
    )
    bio_top.add_argument(
        "--response-column", required=True,
        help="metric column in the response table (e.g. pIC50 or LN_IC50)",
    )
    bio_top.add_argument(
        "--compound", default=None,
        help="compound to analyse (required when the table holds several)",
    )
    bio_top.add_argument(
        "--cell-column", default="cell_line",
        help="cell line column name in both tables",
    )
    bio_top.add_argument(
        "--compound-column", default="compound",
        help="compound column name in the response table",
    )
    bio_top.add_argument(
        "--method", choices=("spearman", "pearson"), default="spearman",
        help="correlation method (default spearman)",
    )
    bio_top.add_argument(
        "--lower-is-sensitive", action="store_true",
        help="low response values mean sensitive (LN_IC50 style, default)",
    )
    bio_top.add_argument(
        "--higher-is-sensitive", action="store_true",
        help="high response values mean sensitive (pIC50 style)",
    )
    bio_top.add_argument(
        "--top", type=int, default=15, metavar="N",
        help="number of biomarkers to report (default 15)",
    )
    bio_top.add_argument(
        "--out", default=None,
        help="write the hits table to this path (csv/tsv/xlsx)",
    )

    return parser


def _cmd_models(args: argparse.Namespace) -> int:
    from pharmaml.ml.models import list_models

    entries = list_models(only_available=False)
    print(f"{'model':<22} {'task':<16} {'available':<10} notes")
    print("-" * 78)
    for entry in entries:
        flag = "yes" if entry["available"] else f"needs {entry['requires']}"
        print(f"{entry['name']:<22} {entry['task']:<16} {flag:<10} {entry['notes']}")
    return 0


def _load_cli_dataset(args: argparse.Namespace):
    """Load the ``--data`` file via :mod:`pharmaml.data` with task inference.

    Both ``ml train`` and ``ml compare`` accept ``--task auto`` (the
    default), which lets :func:`pharmaml.data.load_dataset` decide
    classification (2+ classes) vs regression from the target column,
    and ``--keep-categorical`` to keep string feature columns for
    one-hot encoding.
    """
    from pharmaml.data import load_dataset

    task = None if getattr(args, "task", "auto") == "auto" else args.task
    return load_dataset(
        args.data,
        target=getattr(args, "target", None),
        task=task,
        keep_categorical=getattr(args, "keep_categorical", False),
    )


def _cmd_data_inspect(args: argparse.Namespace) -> int:
    from pharmaml.data import load_dataset

    dataset = load_dataset(args.data, target=args.target)
    print(dataset.summary())
    if args.head > 0:
        print(f"\nfirst {args.head} row(s):")
        print(dataset.frame.head(args.head).to_string(index=False))
    return 0


def _cmd_data_split(args: argparse.Namespace) -> int:
    from pharmaml.data import split_table

    result = split_table(
        path=args.data,
        target=args.target,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify=not args.no_stratify,
        random_state=args.seed,
    )
    out_dir = args.out_dir or str(Path(args.data).parent)
    prefix = args.prefix or Path(args.data).stem
    result.write(out_dir, prefix)
    print(result)
    return 0


def _smi_bridge_frame(path: Path):
    """Read a ``.smi`` file into a SMILES/ID table (CLI-internal bridge).

    Lets ``--data x.smi --to sdf|smi`` reuse the table converters
    without a temporary CSV on disk (the legacy two-step workflow).
    """
    import pandas as pd

    from pharmaml.exceptions import DataValidationError

    rows: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append((parts[0], " ".join(parts[1:])))
    frame = pd.DataFrame(rows, columns=["SMILES", "ID"])
    if frame.empty:
        raise DataValidationError(f"no SMILES lines found in {path}")
    return frame


def _cmd_data_convert(args: argparse.Namespace) -> int:
    from pharmaml.chem.converters import (
        sdf_to_table,
        smi_to_csv,
        table_to_sdf,
        table_to_smi,
    )
    from pharmaml.exceptions import DataValidationError
    from pharmaml.utils.io import read_table, write_table

    src = Path(args.data)
    target = args.to
    suffix = src.suffix.lower()
    out = Path(args.out) if args.out else src.with_suffix(f".{target}")

    if suffix == ".smi":
        if target == "csv":
            written = smi_to_csv(src, out)
            print(f"converted {src.name} -> {written}")
            return 0
        if target == "smi":
            raise DataValidationError(
                f"input {src.name} is already a .smi file -- nothing to convert"
            )
        frame = _smi_bridge_frame(src)
        report = table_to_sdf(
            frame,
            out,
            smiles_col=args.smiles_col,
            name_col=args.name_col,
            embed3d=not args.no_3d,
        )
    elif suffix == ".sdf":
        if target == "sdf":
            raise DataValidationError(
                f"input {src.name} is already an SDF file -- use --to csv or --to smi"
            )
        if target == "csv":
            frame = sdf_to_table(src, out=out, smiles_col=args.smiles_col)
            print(f"converted {src.name} -> {out} ({len(frame)} molecules)")
            return 0
        frame = sdf_to_table(src, smiles_col=args.smiles_col)
        name_col = args.name_col or ("name" if "name" in frame.columns else None)
        written = table_to_smi(
            frame, out, smiles_col=args.smiles_col, name_col=name_col
        )
        print(f"converted {src.name} -> {written} ({len(frame)} molecules)")
        return 0
    else:  # CSV/TSV/Excel table input
        if target == "smi":
            written = table_to_smi(
                src, out, smiles_col=args.smiles_col, name_col=args.name_col
            )
            print(f"converted {src.name} -> {written}")
            return 0
        if target == "csv":
            frame = read_table(src)
            written = write_table(frame, out)
            print(f"converted {src.name} -> {written} ({len(frame)} rows)")
            return 0
        report = table_to_sdf(
            src,
            out,
            smiles_col=args.smiles_col,
            name_col=args.name_col,
            embed3d=not args.no_3d,
        )

    # --to sdf: full audit + output path; invalid SMILES -> exit 1
    print(report)
    print(f"output -> {report.output_sdf}")
    if report.failures:
        print(
            f"{report.failures} invalid SMILES skipped: "
            f"{report.failed_smiles}",
            file=sys.stderr,
        )
        return 1
    return 0


def _train_figure_bundle(result, report, dataset, X_test, y_test):
    """Collect the standard figure/table set for ``ml train --plots``."""
    import pandas as pd

    from pharmaml.viz import metrics_to_dataframe, ml_plots

    figures: dict[str, Any] = {}
    tables: dict[str, Any] = {}

    if dataset.task == "classification":
        figures["confusion_matrix"] = ml_plots.plot_confusion_matrix(report)
        if hasattr(result.model, "predict_proba"):
            figures["roc_curves"] = ml_plots.plot_roc_curves(
                result.model, X_test, y_test
            )
        if dataset.class_counts:
            figures["class_distribution"] = ml_plots.plot_class_distribution(
                dataset.class_counts
            )
            tables["class_counts"] = pd.DataFrame(
                {"class": list(dataset.class_counts),
                 "count": list(dataset.class_counts.values())}
            )
    else:
        y_pred = result.model.predict(X_test)
        figures["predicted_vs_actual"] = ml_plots.plot_predicted_vs_actual(
            y_test, y_pred, metrics=report.as_dict()
        )
        figures["residuals"] = ml_plots.plot_residuals(y_test, y_pred)

    if result.cv_scores is not None:
        figures["cv_scores"] = ml_plots.plot_cv_scores(result)
    try:
        figures["feature_importance"] = ml_plots.plot_feature_importance(result.model)
    except Exception:  # noqa: BLE001 - estimator exposes no importances
        pass  # e.g. kernel SVM / KNN — importance plot is simply skipped

    tables.setdefault(
        "held_out_metrics", metrics_to_dataframe(report)
    )
    return figures, tables


def _cmd_ml_train(args: argparse.Namespace) -> int:
    from pharmaml.config import PipelineConfig
    from pharmaml.ml.evaluation import evaluate_model
    from pharmaml.ml.persistence import save_model
    from pharmaml.ml.preprocessing import make_train_test_split
    from pharmaml.ml.training import train_model

    dataset = _load_cli_dataset(args)
    X_train, X_test, y_train, y_test = make_train_test_split(
        dataset.X, dataset.y, test_size=args.test_size
    )

    result = train_model(
        args.model, dataset.task, X_train, y_train, tune=not args.no_tune, cv=args.cv
    )
    report = evaluate_model(result.model, X_test, y_test, dataset.task)

    config = PipelineConfig()
    model_path = config.output_path(f"{result.name}_{dataset.task}.joblib")
    save_model(
        result.model,
        model_path,
        name=result.name,
        task=dataset.task,
        features=dataset.feature_names,
        best_params=result.best_params,
        metrics=report.as_dict(),
    )
    print(f"dataset:  {dataset.n_samples} rows x {dataset.n_features} features")
    if dataset.class_counts:
        print(f"classes:  {dataset.class_counts}")
    print(f"task:     {dataset.task}")
    print(f"model:    {result.name}")
    print(f"best:     {result.best_params or '(defaults)'}")
    print(f"held-out: {report}")
    print(f"saved:    {model_path}")

    if args.plots:
        from pharmaml.viz import save_visual_bundle

        figures, tables = _train_figure_bundle(result, report, dataset, X_test, y_test)
        bundle_dir = Path(args.out) / "report"
        written = save_visual_bundle(
            figures,
            tables,
            metrics=report,
            out_dir=bundle_dir,
            title=f"pharmaml ml train — {result.name} ({dataset.task})",
            subtitle=str(args.data),
        )
        print(f"plots:    {bundle_dir / 'report.html'} (+{len(written) - 1} files)")
    return 0


def _cmd_ml_compare(args: argparse.Namespace) -> int:
    from pharmaml.ml.models import list_models
    from pharmaml.ml.training import compare_models

    dataset = _load_cli_dataset(args)
    leaderboard = compare_models(
        dataset.X, dataset.y, dataset.task, models=args.models, cv=args.cv
    )
    print(leaderboard.to_string(index=False))

    if args.out:
        from pharmaml.utils.io import write_table

        path = write_table(leaderboard, args.out)
        print(f"\nleaderboard -> {path}")

    if args.plot:
        from pharmaml.viz import ml_plots, save_visual_bundle

        figure = ml_plots.plot_model_comparison(leaderboard)
        written = save_visual_bundle(
            figures={"model_comparison": figure},
            tables={"leaderboard": leaderboard},
            out_dir=args.out_dir,
            title="pharmaml ml compare",
            subtitle=str(args.data),
        )
        print(f"plot:     {Path(args.out_dir) / 'report.html'} (+{len(written) - 1} files)")

    if not args.models:
        unavailable = [e["name"] for e in list_models(only_available=False) if not e["available"]]
        if unavailable:
            print(f"\n(skipped, optional deps missing: {', '.join(unavailable)})")
    return 0


def _cmd_chem_pic50(args: argparse.Namespace) -> int:
    from pharmaml.chem.converters import add_pic50_column
    from pharmaml.utils.io import read_table, write_table

    frame = read_table(args.data)
    converted = add_pic50_column(frame, source_col=args.column, unit=args.unit)
    out = Path(args.out) if args.out else Path(args.data).with_name(
        Path(args.data).stem + "_pic50.csv"
    )
    write_table(converted, out)
    print(f"pIC50 column added -> {out} ({len(converted)} rows)")
    return 0


def _cmd_chem_descriptors(args: argparse.Namespace) -> int:
    from pharmaml.chem.descriptors import compute_descriptors
    from pharmaml.utils.io import read_table, write_table

    frame = read_table(args.data)
    if args.smiles_col not in frame.columns:
        print(f"error: column {args.smiles_col!r} not found", file=sys.stderr)
        return 2
    descriptors = compute_descriptors(frame[args.smiles_col], descriptor_set=args.descriptor_set)
    # ``descriptors`` is indexed by *position* in the SMILES column, so the
    # join must be positional: aligning on the caller frame's index labels
    # would silently attach the wrong row's descriptors to any frame that
    # lacks a default RangeIndex.
    merged = frame.reset_index(drop=True).join(
        descriptors.drop(columns=["SMILES"]), how="left"
    )
    merged.index = frame.index
    out = Path(args.out) if args.out else Path(args.data).with_name(
        Path(args.data).stem + "_descriptors.csv"
    )
    write_table(merged, out)
    print(f"descriptors -> {out} ({descriptors.shape[1] - 1} columns)")
    return 0


def _cmd_chem_compare(args: argparse.Namespace) -> int:
    import pandas as pd

    from pharmaml.chem.tables import compare_tables

    comparison = compare_tables(
        args.data_a,
        args.data_b,
        column=args.column,
        strip_extension=not args.keep_extension,
        lowercase=not args.keep_case,
    )
    counts = comparison.summary()
    print(f"column {comparison.column!r}: " + ", ".join(
        f"{key}={counts[key]}" for key in ("common", "only_in_a", "only_in_b")
    ))
    for label, values in (
        ("common", comparison.common),
        ("only in A", comparison.only_in_a),
        ("only in B", comparison.only_in_b),
    ):
        if values:
            print(f"\n{label} ({len(values)}):")
            print("  " + "\n  ".join(values))
    if args.out:
        frame = pd.DataFrame(
            {
                f"only in {Path(args.data_a).name}": pd.Series(comparison.only_in_a),
                f"only in {Path(args.data_b).name}": pd.Series(comparison.only_in_b),
                "common": pd.Series(comparison.common),
            }
        )
        frame.to_csv(args.out, index=False)
        print(f"\ndifferences report -> {args.out}")
    if args.figure:
        from pharmaml.viz import close_figure, plot_table_comparison, save_figure

        fig = plot_table_comparison(
            comparison,
            labels=(Path(args.data_a).name, Path(args.data_b).name),
            title=f"Ligand-set reconciliation — column {comparison.column!r}",
        )
        figure_path = save_figure(fig, args.figure)
        close_figure(fig)
        print(f"overlap figure -> {figure_path}")
    return 0


def _cmd_chem_merge(args: argparse.Namespace) -> int:
    from pharmaml.chem.tables import merge_table_folder

    first_columns = (
        [c.strip() for c in args.first_columns.split(",") if c.strip()]
        if args.first_columns
        else None
    )
    merged = merge_table_folder(
        args.dir, extension=args.extension, out_path=args.out,
        first_columns=first_columns,
    )
    out = Path(args.out) if args.out else Path(args.dir) / "merged_output.csv"
    print(f"merged {len(merged)} rows x {merged.shape[1]} columns -> {out}")
    return 0


def _cmd_pharma_approvals(args: argparse.Namespace) -> int:
    from pharmaml.pharma.approvals import filter_approved_drugs

    names = filter_approved_drugs(
        list(args.data),
        drug_column=args.drug_column,
        approval_column=args.approval_column,
        approval_status=args.status,
        out_path=args.out,
    )
    print(f"{len(names)} unique drug name(s) with status {args.status!r}")
    for name in names:
        print(f"  {name}")
    if args.out:
        print(f"\ndrug list -> {args.out}")
    return 0


def _cmd_pharma_pk(args: argparse.Namespace) -> int:
    import json

    from pharmaml.pharma.pk import nca_summary
    from pharmaml.utils.io import read_table
    from pharmaml.utils.validation import require_columns

    frame = read_table(args.data)
    require_columns(frame, [args.time, args.conc])
    summary = nca_summary(frame[args.time].to_numpy(), frame[args.conc].to_numpy())
    payload = {k: (None if v != v else v) for k, v in summary.as_dict().items()}
    print(json.dumps(payload, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved -> {args.out}")
    return 0


def _cmd_pharma_dose(args: argparse.Namespace) -> int:
    from pharmaml.pharma.dosage import loading_dose, maintenance_dose
    from pharmaml.pharma.pk import half_life, volume_of_distribution

    md = maintenance_dose(args.cl, args.target_css, args.tau, f=args.f)
    ld = loading_dose(args.vd, args.target_css, f=args.f)
    ke = args.cl / args.vd
    print(f"maintenance dose : {md:.2f} per {args.tau:g} h")
    print(f"loading dose     : {ld:.2f}")
    print(f"ke               : {ke:.4f} 1/h   (t1/2 = {half_life(ke):.2f} h)")
    print(f"Vd               : {volume_of_distribution(args.cl, ke):.2f} L")
    return 0


def _cmd_docking_scan(args: argparse.Namespace) -> int:
    from pharmaml.docking.parsing import scan_docking_directory
    from pharmaml.utils.io import write_table

    summary = scan_docking_directory(args.directory, log_name=args.log_name)
    print(summary.head(20).to_string(index=False))
    if args.out:
        write_table(summary, args.out)
        print(f"saved -> {args.out}")
    return 0


def _cmd_docking_check(args: argparse.Namespace) -> int:
    from pharmaml.docking.parsing import check_docking_completion
    from pharmaml.utils.io import write_table

    audit = check_docking_completion(args.directory, log_name=args.log_name)
    print(audit.to_string(index=False))
    problems = int((audit["status"] != "OK").sum())
    if args.out:
        write_table(audit, args.out)
    return 1 if problems else 0


def _cmd_docking_prep_ligands(args: argparse.Namespace) -> int:
    from pharmaml.docking import prepare_ligand_library

    report = prepare_ligand_library(
        args.ligands,
        args.out,
        smiles_col=args.smiles_col,
        name_col=args.name_col,
        folder_naming=args.naming,
    )
    print(report.library)
    print()
    print(report)
    if report.n_failed:
        # prepare_ligand_library already wrote failed_smiles.csv next to
        # the campaign (rar parity); just point the user at it.
        print(f"failed ligands -> {report.root / 'failed_smiles.csv'}")
        return 1
    return 0


def _cmd_docking_prep_protein(args: argparse.Namespace) -> int:
    from pharmaml.docking import prepare_protein

    altloc = args.altloc
    if altloc and altloc.lower() != "first":
        altloc = altloc.strip().upper()  # accept 'b' as 'B'
    report = prepare_protein(
        args.protein,
        args.out,
        keep_hetatm=args.keep_hetatm,
        chain=args.chain,
        default_altloc=altloc,
    )
    print(report)
    print(f"receptor -> {report.output_pdbqt}")
    return 0


def _cmd_docking_dpf(args: argparse.Namespace) -> int:
    from pharmaml.docking import write_dpf

    dpf_path = write_dpf(
        args.ligand,
        receptor_file=args.receptor,
        output_path=args.out,
        ga_run=args.ga_runs,
    )
    print(f"dpf -> {dpf_path}")
    return 0


def _cmd_docking_collect(args: argparse.Namespace) -> int:
    from pharmaml.docking import collect_files

    filename = args.filename
    if filename is None and args.pattern is None:
        filename = "r.dlg"  # rar default (extaacer.py / dlg_extracter.py)
    report = collect_files(
        args.dir,
        filename=filename,
        pattern=args.pattern,
        out_dir=args.out,
        dry_run=args.dry_run,
    )
    print(report.to_string(index=False))
    n_copied = int(report["copied"].sum()) if not report.empty else 0
    print(
        f"{len(report)} match(es), {n_copied} copied"
        + (" (dry run)" if args.dry_run else "")
    )
    return 0


def _cmd_docking_torsion(args: argparse.Namespace) -> int:
    from pharmaml.docking import scan_torsion_consistency

    report = scan_torsion_consistency(args.dir)
    print(report.to_string(index=False))
    return 0


def _cmd_docking_gpf_scan(args: argparse.Namespace) -> int:
    from pharmaml.docking import scan_gpf_ligand_types

    report = scan_gpf_ligand_types(args.dir)
    print(report.to_string(index=False))
    return 0


def _cmd_docking_vina_scan(args: argparse.Namespace) -> int:
    from pharmaml.docking import scan_vina_directory

    report = scan_vina_directory(args.dir)
    print(report.to_string(index=False))
    return 0


def _cmd_docking_campaign(args: argparse.Namespace) -> int:
    """Dispatch ``pharmaml docking campaign <verb>`` sub-commands."""
    from pharmaml.docking import (
        cleanup_tree,
        delete_named_files,
        distribute_files,
        list_file_names,
        reindex_by_number,
    )

    verb = args.campaign_command
    if verb == "list":
        names = list_file_names(args.dir, extension=args.ext, out_path=args.out)
        for name in names:
            print(name)
        print(f"{len(names)} file stem(s)")
        return 0
    if verb == "distribute":
        written = distribute_files(args.files, args.dir, overwrite=args.overwrite)
        for destination in written:
            print(destination)
        print(
            f"{len(written)} cop(y/ies) written"
            + (" (overwrite on)" if args.overwrite else "")
        )
        return 0
    if verb == "cleanup":
        report = cleanup_tree(
            args.dir,
            keep=args.keep,
            keep_pattern=args.keep_pattern,
            remove_empty=args.remove_empty,
            dry_run=args.dry_run,
        )
        print(report.to_string(index=False))
        n_deleted = int((report["action"] == "deleted").sum())
        n_dirs = int((report["action"] == "removed_dir").sum())
        print(
            f"{n_deleted} file(s) deleted, {n_dirs} director(y/ies) removed"
            + (" (dry run)" if args.dry_run else "")
        )
        return 0
    if verb == "delete-named":
        deleted = delete_named_files(args.dir, filename=args.name, dry_run=args.dry_run)
        print(
            f"{deleted} file(s) named {args.name!r} "
            + ("would be deleted (dry run)" if args.dry_run else "deleted")
        )
        return 0
    if verb == "reindex":
        from pharmaml.utils.io import read_table, write_table

        frame = read_table(args.data)
        result = reindex_by_number(
            frame,
            column=args.column,
            start=args.start,
            stop=args.stop,
            pattern=args.pattern,
        ).rename_axis("number")
        total = int(len(result))
        present = int(result[args.column].notna().sum())
        print(
            f"{present}/{total} ligand number(s) present in range "
            f"{args.start}..{args.stop}"
        )
        missing = [int(n) for n in result.index[result[args.column].isna()]]
        if missing and len(missing) <= 20:
            print("missing numbers: " + ", ".join(str(n) for n in missing))
        elif missing:
            print(f"{len(missing)} missing number(s) (not listed; more than 20)")
        if args.out:
            written = write_table(result, args.out, index=True)
            print(f"reindexed table -> {written}")
        else:
            print(result.to_string())
        return 0
    return 2  # pragma: no cover - argparse guards this


def _cmd_biology_response_audit(args: argparse.Namespace) -> int:
    """``pharmaml biology response-audit``: screen summary + audit + stats."""
    from pharmaml.biology import (
        audit_response_table,
        load_response_table,
        summarize_compounds,
    )

    screen = load_response_table(
        args.data,
        cell_column=args.cell_column,
        compound_column=args.compound_column,
        tissue_column=args.tissue_column,
    )
    print(screen.summary())
    print("\naudit:")
    print(audit_response_table(screen).to_string(index=False))
    metric = args.metric if args.metric is not None else screen.metrics[0]
    print(f"\nper-compound summary (metric {metric!r}):")
    print(summarize_compounds(screen, metric).to_string())
    return 0


def _cmd_biology_top_biomarkers(args: argparse.Namespace) -> int:
    """``pharmaml biology top-biomarkers``: expression vs. one drug."""
    import pandas as pd

    from pharmaml.biology import (
        direction_of_association,
        sensitivity_associations,
        top_biomarkers,
    )
    from pharmaml.biology.response import load_response_table
    from pharmaml.exceptions import DataValidationError
    from pharmaml.utils.io import read_table, write_table

    if args.lower_is_sensitive and args.higher_is_sensitive:
        raise DataValidationError(
            "pass at most one of --lower-is-sensitive / "
            "--higher-is-sensitive (they are opposites)"
        )
    lower_is_sensitive = not args.higher_is_sensitive

    screen = load_response_table(
        args.response,
        cell_column=args.cell_column,
        compound_column=args.compound_column,
        metrics=[args.response_column],
    )
    compounds = screen.compounds
    if args.compound is not None:
        if args.compound not in compounds:
            raise DataValidationError(
                f"compound {args.compound!r} not found in the response table "
                f"-- available compounds: {compounds}"
            )
        chosen = args.compound
    elif len(compounds) == 1:
        chosen = compounds[0]
    else:
        shown = ", ".join(compounds[:5])
        more = " ..." if len(compounds) > 5 else ""
        raise DataValidationError(
            f"response table holds {len(compounds)} compound(s) ({shown}{more}) "
            "-- pass --compound to pick one"
        )

    subset = screen.frame[screen.frame[screen.compound_col] == chosen]
    if subset[screen.cell_col].duplicated().any():
        n_dup = int(subset[screen.cell_col].duplicated().sum())
        _LOG.warning(
            "%d replicate row(s) for compound %r -- collapsed by median",
            n_dup,
            chosen,
        )
        # dropna=False keeps rows whose cell-line identifier is missing
        # as their own group (warn-don't-drop); the correlation
        # alignment then reports them as unmatched.
        subset = subset.groupby(
            screen.cell_col, as_index=False, dropna=False
        ).agg({args.response_column: "median"})
    response = subset.set_index(screen.cell_col)[args.response_column]
    if int(response.notna().sum()) < 3:
        raise DataValidationError(
            f"only {int(response.notna().sum())} non-NaN response value(s) for "
            f"compound {chosen!r} -- at least 3 are needed to correlate"
        )

    expression = read_table(args.expression)
    if args.cell_column in expression.columns:
        expression = expression.set_index(args.cell_column)
    elif not pd.api.types.is_numeric_dtype(expression[expression.columns[0]]):
        first = expression.columns[0]
        _LOG.debug(
            "Using expression column %r as the cell-line index", first
        )
        expression = expression.set_index(first)
    else:
        raise DataValidationError(
            "cannot identify a cell-line column in the expression table -- "
            f"name it {args.cell_column!r} or make the first column hold the "
            "cell-line identifiers"
        )

    associations = sensitivity_associations(
        expression, response, method=args.method
    )
    hits = top_biomarkers(associations, n=args.top).copy()
    hits["direction"] = [
        direction_of_association(rho, lower_is_sensitive=lower_is_sensitive)
        for rho in hits["rho"]
    ]
    print(f"compound : {chosen} ({len(response)} cell line(s))")
    print(f"response : column {args.response_column!r}, method {args.method}")
    print(
        f"genes    : {expression.shape[1]} correlated, "
        f"{int(associations['significant'].sum())} significant (FDR)"
    )
    print()
    print(hits.to_string(index=False))
    if args.out:
        written = write_table(hits, args.out)
        print(f"\nbiomarkers -> {written}")
    return 0


_HANDLERS = {
    ("models", None): _cmd_models,
    ("data", "inspect"): _cmd_data_inspect,
    ("data", "split"): _cmd_data_split,
    ("data", "convert"): _cmd_data_convert,
    ("ml", "train"): _cmd_ml_train,
    ("ml", "compare"): _cmd_ml_compare,
    ("chem", "pic50"): _cmd_chem_pic50,
    ("chem", "descriptors"): _cmd_chem_descriptors,
    ("chem", "compare"): _cmd_chem_compare,
    ("chem", "merge"): _cmd_chem_merge,
    ("pharma", "pk"): _cmd_pharma_pk,
    ("pharma", "dose"): _cmd_pharma_dose,
    ("pharma", "approvals"): _cmd_pharma_approvals,
    ("docking", "scan"): _cmd_docking_scan,
    ("docking", "check"): _cmd_docking_check,
    ("docking", "prep-ligands"): _cmd_docking_prep_ligands,
    ("docking", "prep-protein"): _cmd_docking_prep_protein,
    ("docking", "dpf"): _cmd_docking_dpf,
    ("docking", "collect"): _cmd_docking_collect,
    ("docking", "torsion"): _cmd_docking_torsion,
    ("docking", "gpf-scan"): _cmd_docking_gpf_scan,
    ("docking", "vina-scan"): _cmd_docking_vina_scan,
    ("docking", "campaign"): _cmd_docking_campaign,
    ("biology", "response-audit"): _cmd_biology_response_audit,
    ("biology", "top-biomarkers"): _cmd_biology_top_biomarkers,
}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (returns a process exit code)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    configure_logging(
        level=10 if args.verbose else 30,  # DEBUG / INFO
        log_file=args.log_file,
    )

    sub = getattr(args, f"{args.command}_command", None)
    handler = _HANDLERS.get((args.command, sub))
    if handler is None:  # pragma: no cover - argparse guards this
        parser.error("unknown command combination")
    try:
        return handler(args)
    except Exception as exc:
        _LOG.error("%s", exc)
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
