"""Command-line interface — the scaffolding entry point of PharmaML.

Replaces the legacy workflow of "open a per-algorithm .py file, edit
the hardcoded CSV filename at the bottom, run it".  One command covers
training, comparison, conversion, PK analysis and docking scans::

    pharmaml ml train    --data qsar.csv --target Target --model random_forest
    pharmaml ml train    --data qsar.csv --task auto        # infer classification/regression
    pharmaml ml compare  --data qsar.csv --target Target --task classification
    pharmaml data inspect --data qsar.csv                  # audit your own file
    pharmaml chem pic50  --data activity.xlsx --column "IC50 (nM)"
    pharmaml pharma pk   --data profile.csv --time time_h --conc conc_ng_ml
    pharmaml docking scan --directory docking_run/
    pharmaml models list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    # ---- ml ----------------------------------------------------------
    ml_parser = subparsers.add_parser("ml", help="machine-learning workflows")
    ml_sub = ml_parser.add_subparsers(dest="ml_command", required=True)

    train = ml_sub.add_parser("train", help="train one model end-to-end")
    train.add_argument("--data", required=True, help="CSV/Excel with features+target")
    train.add_argument("--target", default=None, help="target column (auto-detected)")
    train.add_argument("--model", default="random_forest", help="registry model name")
    train.add_argument(
        "--task", choices=["classification", "regression", "auto"], default="auto",
        help="task type; 'auto' infers it from the target column (default)",
    )
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--cv", type=int, default=5)
    train.add_argument("--no-tune", action="store_true", help="skip grid search")
    train.add_argument("--out", default="artifacts", help="artifact output directory")

    compare = ml_sub.add_parser("compare", help="cross-validated model leaderboard")
    compare.add_argument("--data", required=True)
    compare.add_argument("--target", default=None)
    compare.add_argument(
        "--task", choices=["classification", "regression", "auto"], default="auto"
    )
    compare.add_argument("--cv", type=int, default=5)
    compare.add_argument("--models", nargs="*", default=None, help="subset of models")

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
    classification (2+ classes) vs regression from the target column.
    """
    from pharmaml.data import load_dataset

    task = None if getattr(args, "task", "auto") == "auto" else args.task
    return load_dataset(args.data, target=getattr(args, "target", None), task=task)


def _cmd_data_inspect(args: argparse.Namespace) -> int:
    from pharmaml.data import load_dataset

    dataset = load_dataset(args.data, target=args.target)
    print(dataset.summary())
    if args.head > 0:
        print(f"\nfirst {args.head} row(s):")
        print(dataset.frame.head(args.head).to_string(index=False))
    return 0


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
    model_path = config.output_path(f"{args.model}_{dataset.task}.joblib")
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
    return 0


def _cmd_ml_compare(args: argparse.Namespace) -> int:
    from pharmaml.ml.models import list_models
    from pharmaml.ml.training import compare_models

    dataset = _load_cli_dataset(args)
    leaderboard = compare_models(
        dataset.X, dataset.y, dataset.task, models=args.models, cv=args.cv
    )
    print(leaderboard.to_string(index=False))

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
    merged = frame.join(descriptors.drop(columns=["SMILES"]), how="left")
    out = Path(args.out) if args.out else Path(args.data).with_name(
        Path(args.data).stem + "_descriptors.csv"
    )
    write_table(merged, out)
    print(f"descriptors -> {out} ({descriptors.shape[1] - 1} columns)")
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


_HANDLERS = {
    ("models", None): _cmd_models,
    ("data", "inspect"): _cmd_data_inspect,
    ("ml", "train"): _cmd_ml_train,
    ("ml", "compare"): _cmd_ml_compare,
    ("chem", "pic50"): _cmd_chem_pic50,
    ("chem", "descriptors"): _cmd_chem_descriptors,
    ("pharma", "pk"): _cmd_pharma_pk,
    ("pharma", "dose"): _cmd_pharma_dose,
    ("docking", "scan"): _cmd_docking_scan,
    ("docking", "check"): _cmd_docking_check,
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
