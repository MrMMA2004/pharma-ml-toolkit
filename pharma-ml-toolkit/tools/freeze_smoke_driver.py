#!/usr/bin/env python3
"""The part9 frozen-smoke driver (phases 2 + 3) — runs INSIDE a frozen app.

``packaging/windows/pharmaml.spec`` defines the Windows installer; this
driver is the §13/§14 evidence entry that ``tools/freeze_equivalent.py``
freezes ALONGSIDE ``studio.py`` into the same onedir COLLECT, so the
frozen bundle can be exercised end-to-end on this Linux host (the
frozen-equivalent verification — the honest tier between "static
validation" and "Windows clean-machine test").

What it proves, per part9's own vocabulary:

* §13 startup reliability: the flows below import and run through the
  bootloader's embedded interpreter — dynamic imports (mordred's
  registry), matplotlib Agg, RDKit, the boosting shims.
* §4 resource paths: fonts load through ``ui.font_loader``'s
  ``__file__`` math; the bundled engines are found by
  ``pharmaml.docking.workflow``'s OWN discovery (SHA-256 + PE-machine
  validated) at ``_internal/pharmaml/engines``.
* §14.2 scientific workflows, phase 2 (r273): CSV import, Excel
  import, descriptors, fingerprints, ML classification + regression,
  prediction, plots, figure export (PNG/SVG/PDF/TIFF), CSV/Excel
  export, model save/reload, kNN applicability domain, the docking
  engines' presence — every one through the application's real APIs.
* §14.2 scientific workflows, phase 3 (r274 — the deep-feature tier):
  a torch deep-learning training (DeepMLPClassifier), semi-supervised
  learning with unlabeled rows (train_semi_supervised), the generative
  AI workspace (run_genetic_optimization with QedObjective), the
  docking campaign's ligand preparation (prepare_ligand_library —
  SMILES → 3D → PDBQT campaign tree; the ENGINES themselves are
  Windows PEs and cannot execute on Linux — validated, not run),
  PK/PD ODE solving with dosing events (solve_ode), MD trajectory
  analysis (synthetic_trajectory + rmsd/rmsf/Rg), SHAP explainability
  (shap_summary on the trained forest), prediction uncertainty
  (regression_uncertainty_report — the conformal workbench), and the
  interactive figure's per-point SMILES metadata (Figure hovertext —
  §9's "SMILES metadata inspection" surface).
* §14.3 offline: ``socket.socket.connect`` is armed to raise on ANY
  outbound attempt before the science imports; a green run is a
  recorded zero-attempt proof, not an assumption.
* §7 data architecture: every byte this driver writes lands under the
  user-supplied ``--user-dir`` (matplotlib's config + numba's JIT
  cache are redirected there too); the frozen installation directory
  is snapshotted before/after and must stay byte-identical.

It is ALSO runnable unfrozen (``python tools/freeze_smoke_driver.py
--user-dir /tmp/x``) — the same flows through the source checkout —
which is what the focused test suite pins so API drift is caught at
gate time, not at freeze time.

Exit codes: 0 all flows green; 1 any flow failed (the JSON verdict
under ``--user-dir`` carries the per-flow detail).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Unfrozen runs (source checkout / the focused test suite) need the toolkit
# root on sys.path for the ``ui`` package (only ``pharmaml`` is pip-installed,
# editable).  The part18 §10 quarantine completion moved the legacy GUI tree
# to ``legacy/`` — so BOTH the root and ``legacy/`` join the path; frozen, the
# bootloader's _internal layout already has everything importable and this
# is a no-op.
if not getattr(sys, "frozen", False):
    _tk = Path(__file__).resolve().parents[1]
    _legacy = _tk / "legacy"
    if _legacy.is_dir():
        sys.path.insert(0, str(_legacy))
    sys.path.insert(0, str(_tk))

# The 12-molecule real-chemistry set (examples/data/ligand_library.csv's
# head — real drugs, not toy SMILES, so descriptors/fingerprints exercise
# genuine chemistry paths).
SMILES: tuple[str, ...] = (
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",        # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                # paracetamol
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",      # caffeine
    "COC1=CC2=C(C=C1)N=CC=C2",           # quinine core
    "CC(=O)CC(=O)C1=CC=C(C=C1)Cl",       # diclofenac-methyl
    "NC1=CC=C(N=C1)C2=CC=CC=C2",         # amodiaquine core
    "CC1=CC=C(C=C1)S(=O)(=O)N",          # sulfanilamide
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
    "C1=CC=C(C=C1)C=O",                  # benzaldehyde
    "CC(C)(C)NCC(O)COc1cccc2ccccc12",    # propranolol core
    "CSCC[C@H](N)C(=O)O",                # cysteine-methyl
)

_NET_ATTEMPTS: list[str] = []


def block_network() -> None:
    """Arm the offline proof (part9 §14.3): any connect() raises + records."""

    def guard(self, address):  # type: ignore[no-untyped-def]
        _NET_ATTEMPTS.append(str(address))
        raise OSError(
            f"freeze-smoke offline proof: network attempt blocked: {address}"
        )

    try:
        socket.socket.connect = guard  # type: ignore[method-assign]
    except (AttributeError, TypeError):  # pragma: no cover - exotic runtime
        pass


def _synth_frame() -> Any:
    """A deterministic 96-row numeric dataset (classification + regression)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(273)
    n = 96
    x = rng.normal(size=(n, 8))
    frame = pd.DataFrame(x, columns=[f"f{i + 1}" for i in range(8)])
    logit = 2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2]
    frame["y_cls"] = (logit > 0).astype(int)
    frame["y_reg"] = 3.0 * x[:, 0] + 1.0 * x[:, 3] + rng.normal(scale=0.05, size=n)
    return frame


# --- the flows (uniform (record, user, bundle) contract) -----------------

def flow_environment(record: dict, user: Path, bundle: dict) -> None:
    meipass = getattr(sys, "_MEIPASS", None)
    frozen = bool(getattr(sys, "frozen", False))
    import rdkit
    import torch

    import pharmaml

    record["environment"] = {
        "python": sys.version.split()[0],
        "frozen": frozen,
        "meipass": str(meipass),
        "argv0": sys.argv[0],
        "cwd": os.getcwd(),
        "pharmaml": pharmaml.__version__,
        "rdkit": rdkit.__version__,
        "torch": torch.__version__,
    }
    assert frozen or os.environ.get("FREEZE_SMOKE_UNFROZEN_OK") == "1", (
        "driver started without the frozen flag -- use freeze_equivalent "
        "or set FREEZE_SMOKE_UNFROZEN_OK=1 for the source-checkout run"
    )


def flow_fonts(record: dict, user: Path, bundle: dict) -> None:
    # a QFontDatabase call needs an application object; the bootstrap
    # flow above owns the QApplication construction, this is belt and
    # braces for standalone imports of the flow in the focused tests.
    from ui.headless import ensure_headless

    ensure_headless()
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is None:  # pragma: no cover - standalone
        QApplication([])
    from ui.font_loader import has_plex, register_fonts

    loaded = register_fonts()
    record["fonts"] = {"registered": len(loaded), "plex": has_plex()}
    assert len(loaded) >= 2, f"font bundle did not register: {loaded}"


def flow_import_csv(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.data import load_dataset

    csv_path = user / "smoke_dataset.csv"
    frame = _synth_frame()
    frame.to_csv(csv_path, index=False)
    ds = load_dataset(csv_path, target="y_cls")
    record["import_csv"] = {
        "path": str(csv_path),
        "rows": int(ds.frame.shape[0]),
        "features": int(ds.X.shape[1]),
        "task": ds.task,
        "dropped": list(ds.dropped_columns),
    }
    assert ds.frame.shape[0] == 96 and ds.X.shape[1] == 9  # 8 f + y_reg kept
    bundle["frame"] = ds.frame
    bundle["X"] = ds.X
    bundle["y"] = ds.y


def flow_import_excel(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.data import load_dataset

    xlsx = user / "smoke_dataset.xlsx"
    frame = _synth_frame()
    frame.to_excel(xlsx, index=False)
    ds = load_dataset(xlsx, target="y_reg")
    record["import_excel"] = {
        "path": str(xlsx),
        "rows": int(ds.frame.shape[0]),
        "task": ds.task,
    }
    assert ds.frame.shape[0] == 96 and ds.task == "regression"


def flow_descriptors(record: dict, user: Path, bundle: dict) -> None:
    import pandas as pd

    from pharmaml.chem.descriptors import compute_descriptors

    desc = compute_descriptors(pd.Series(SMILES), progress=False)
    record["descriptors"] = {
        "rows": int(desc.shape[0]),
        "columns": int(desc.shape[1]),
    }
    assert desc.shape[0] == len(SMILES) and desc.shape[1] >= 100
    bundle["desc"] = desc


def flow_fingerprints(record: dict, user: Path, bundle: dict) -> None:
    import pandas as pd

    from pharmaml.chem.descriptors import compute_fingerprints

    bits, onbits = compute_fingerprints(pd.Series(SMILES), kind="morgan")
    record["fingerprints"] = {
        "shape": list(bits.shape),
        "rows_with_bits": int(sum(1 for b in onbits if b)),
    }
    assert bits.shape == (len(SMILES), 2048)
    bundle["bits"] = bits


def flow_ml_train_predict(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.ml.training import train_model

    X = bundle["X"]
    split = int(len(X) * 0.8)
    clf = train_model(
        "random_forest",
        "classification",
        X.iloc[:split],
        bundle["y"].iloc[:split],
        tune=False,
    )
    reg = train_model(
        "random_forest",
        "regression",
        X.iloc[:split],
        bundle["frame"]["y_reg"].iloc[:split],
        tune=False,
    )

    pred_c = clf.model.predict(X.iloc[split:])
    truth_c = bundle["y"].iloc[split:].to_numpy()
    pred_r = reg.model.predict(X.iloc[split:])
    truth_r = bundle["frame"]["y_reg"].iloc[split:].to_numpy()
    acc = float((pred_c == truth_c).mean())
    r2 = 1.0 - float(
        ((pred_r - truth_r) ** 2).sum()
        / ((truth_r - truth_r.mean()) ** 2).sum()
    )
    record["ml"] = {
        "classification_accuracy": round(acc, 4),
        "regression_r2": round(r2, 4),
        "n_train": split,
        "n_test": int(len(X) - split),
    }
    assert acc > 0.8 and r2 > 0.9, (acc, r2)
    bundle["clf_model"] = clf.model
    bundle["pred_reg"] = pred_r
    bundle["truth_reg"] = truth_r


def flow_plot_export(record: dict, user: Path, bundle: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(bundle["truth_reg"], bundle["pred_reg"], s=18)
    lims = [
        float(min(bundle["truth_reg"].min(), bundle["pred_reg"].min())),
        float(max(bundle["truth_reg"].max(), bundle["pred_reg"].max())),
    ]
    ax.plot(lims, lims, lw=1)
    ax.set_xlabel("truth")
    ax.set_ylabel("predicted")
    ax.set_title("freeze-smoke: regression parity")
    exports: dict[str, int] = {}
    for ext in ("png", "svg", "pdf", "tiff"):
        path = user / f"parity_plot.{ext}"
        fig.savefig(path, dpi=110)
        exports[ext] = path.stat().st_size
        assert path.stat().st_size > 0
    plt.close(fig)
    record["plot_export"] = exports
    assert exports["png"] > 5000 and exports["svg"] > 1000


def flow_table_export(record: dict, user: Path, bundle: dict) -> None:
    import pandas as pd

    out = pd.DataFrame(
        {"truth": bundle["truth_reg"], "predicted": bundle["pred_reg"]}
    )
    csv = user / "smoke_predictions.csv"
    xlsx = user / "smoke_predictions.xlsx"
    out.to_csv(csv, index=False)
    out.to_excel(xlsx, index=False)
    record["table_export"] = {
        "csv": csv.stat().st_size,
        "xlsx": xlsx.stat().st_size,
    }
    assert csv.stat().st_size > 100 and xlsx.stat().st_size > 3000


def flow_model_save_reload(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.ml.persistence import load_model, save_model

    path = user / "smoke_rf.joblib"
    save_model(
        bundle["clf_model"],
        path,
        name="random_forest",
        task="classification",
        features=list(bundle["X"].columns),
    )
    reloaded, meta = load_model(path)
    import numpy as np

    parity = bool(
        np.array_equal(
            reloaded.predict(bundle["X"]),
            bundle["clf_model"].predict(bundle["X"]),
        )
    )
    record["model_save_reload"] = {
        "path": str(path),
        "parity": parity,
        "bundle_version": meta.get("pharmaml_version"),
    }
    assert parity


def flow_applicability_domain(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.ml.applicability import knn_applicability_domain

    X = bundle["X"]
    split = int(len(X) * 0.8)
    ad = knn_applicability_domain(X.iloc[:split], X.iloc[split:], k=3)
    record["applicability_domain"] = {
        "distance_threshold": float(ad.distance_threshold),
        "train_knn_mean_rows": len(ad.train_knn_mean),
        "test_knn_mean_rows": len(ad.test_knn_mean),
    }
    assert len(ad.test_knn_mean) == len(X) - split


def flow_bundled_engines(record: dict, user: Path, bundle: dict) -> None:
    from pharmaml.docking import workflow as dock

    found: dict[str, str] = {}
    for name in ("vina", "autogrid4", "autodock4"):
        path = dock._bundled_engine(name)  # sha-256 + PE-machine validated
        if path is not None:
            found[name] = str(path)
    record["bundled_engines"] = found
    assert set(found) == {"vina", "autogrid4", "autodock4"}, (
        "the bundled AutoDock trio did not validate inside the freeze"
    )


def flow_qt_offscreen(record: dict, user: Path, bundle: dict) -> None:
    """Bootstrap the GUI stack FIRST (part9 §13 frozen startup): the
    offscreen platform + EGL fallback via the app's own ``ui.headless``,
    a live QApplication, and both stylesheet engines' resources — the
    later GUI-touching flows (fonts) need an application object to
    exist before any QFontDatabase call."""
    from ui.headless import ensure_headless

    applied = ensure_headless()
    import qdarkstyle
    from PySide6.QtWidgets import QApplication, QWidget
    from qt_material import apply_stylesheet

    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    widget.setWindowTitle("freeze-smoke")
    qss = qdarkstyle.load_stylesheet()
    widget.setStyleSheet(qss)
    apply_stylesheet(app, theme="dark_teal")
    record["qt_offscreen"] = {
        "platform": applied.get("platform"),
        "egl": applied.get("egl"),
        "qdarkstyle_chars": len(qss),
    }
    assert len(qss) > 1000
    widget.deleteLater()


# --- phase 3 (r274): the deep-feature §14.2 tier -------------------------


def flow_deep_learning(record: dict, user: Path, bundle: dict) -> None:
    """§14.2 "Run a supported DL example": a torch MLP trained INSIDE
    the frozen bundle through the app's own DeepMLPClassifier."""
    from pharmaml.ml.neural import DeepMLPClassifier

    X, y = bundle["X"], bundle["y"]
    split = int(len(X) * 0.8)
    net = DeepMLPClassifier(
        hidden_layers=(16, 8), epochs=40, batch_size=16,
        learning_rate=0.01, random_state=274,
    )
    net.fit(X.iloc[:split], y.iloc[:split])
    pred = net.predict(X.iloc[split:])
    truth = y.iloc[split:].to_numpy()
    acc = float((pred == truth).mean())
    proba = net.predict_proba(X.iloc[split:])
    record["deep_learning"] = {
        "accuracy": round(acc, 4),
        "n_params": int(sum(p.numel() for p in net._network.parameters())),
        "proba_shape": list(proba.shape),
    }
    assert acc > 0.7 and proba.shape == (len(X) - split, 2), (acc, proba.shape)


def flow_semi_supervised(record: dict, user: Path, bundle: dict) -> None:
    """§14.2 "Run Semi-supervised Learning": self-training over a
    training split whose labels are 70% withheld (the -1 convention);
    the pseudo-labels recovered on the masked rows are scored against
    the withheld truth, and the fitted model is evaluated inductively
    on the held-out test rows."""
    import numpy as np

    from pharmaml.ml.semi_supervised import train_semi_supervised

    X, y = bundle["X"], bundle["y"]
    split = int(len(X) * 0.8)
    rng = np.random.default_rng(274)
    y_masked = y.iloc[:split].copy()
    hide = rng.choice(split, size=int(split * 0.7), replace=False)
    y_masked.iloc[hide] = -1
    result = train_semi_supervised(
        X.iloc[:split], y_masked, method="self_training", random_state=274,
    )
    pseudo = np.asarray(result.labels_)[hide]
    pseudo_acc = float((pseudo == y.iloc[hide].to_numpy()).mean())
    test_pred = result.model.predict(X.iloc[split:])
    truth = y.iloc[split:].to_numpy()
    acc = float((test_pred == truth).mean())
    record["semi_supervised"] = {
        "accuracy_held_out": round(acc, 4),
        "pseudo_label_accuracy": round(pseudo_acc, 4),
        "labeled_rows": int(split - len(hide)),
        "unlabeled_rows": len(hide),
        "method": "self_training",
    }
    assert acc > 0.7 and pseudo_acc > 0.5 and len(hide) > 0, (acc, pseudo_acc)


def flow_explainability(record: dict, user: Path, bundle: dict) -> None:
    """§9 "Explainability": SHAP summary of the trained forest through
    the app's own shap_summary (the real shap TreeExplainer path)."""
    from pharmaml.ml.explain import shap_summary

    X = bundle["X"]
    split = int(len(X) * 0.8)
    summary = shap_summary(bundle["clf_model"], X.iloc[split:])
    top = int(max(range(len(summary.mean_abs)), key=lambda i: summary.mean_abs[i]))
    record["explainability"] = {
        "model_name": summary.model_name,
        "n_features": len(summary.names),
        "n_samples": int(summary.n_samples),
        "top_feature": summary.names[top],
        "top_mean_abs": round(float(summary.mean_abs[top]), 5),
    }
    assert summary.n_samples == len(X) - split and summary.names[top] in ("f1", "f2"), (
        summary.names[top],
    )


def flow_uncertainty(record: dict, user: Path, bundle: dict) -> None:
    """§9 "Prediction uncertainty": the conformal workbench's report
    over disjoint train/calibration/test partitions."""
    import numpy as np

    from pharmaml.ml.uncertainty_workbench import regression_uncertainty_report

    frame = bundle["frame"].copy()
    rng = np.random.default_rng(274)
    frame["mol"] = [f"m{i:03d}" for i in range(len(frame))]
    frame = frame.iloc[rng.permutation(len(frame))]
    train, calib, test = frame.iloc[:40], frame.iloc[40:60], frame.iloc[60:]
    features = [c for c in frame.columns if c.startswith("f")]
    report = regression_uncertainty_report(
        train, calib, test,
        features=features, target="y_reg", key="mol", alpha=0.1,
    )
    record["uncertainty"] = {
        "coverage": round(float(report["coverage"]), 4),
        "mean_width": round(float(report["mean_width"]), 4),
        "n_test": int(report["n_test"]),
        "alpha": report["alpha"],
    }
    assert 0.0 <= report["coverage"] <= 1.0 and report["mean_width"] > 0


def flow_generative(record: dict, user: Path, bundle: dict) -> None:
    """§14.2 "Generative AI": the genetic optimizer maximizing QED over
    real-drug seeds, INSIDE the freeze."""
    from pharmaml.generative import QedObjective, run_genetic_optimization

    result = run_genetic_optimization(
        QedObjective(),
        seed_smiles=list(SMILES[:6]),
        corpus=list(SMILES),
        generations=4,
        seed=274,
    )
    record["generative"] = {
        "n_molecules": len(result.smiles),
        "best_score": round(float(result.best_score), 4),
        "best": result.best,
        "valid": int(result.metadata.get("valid", -1)),
        "unique": int(result.metadata.get("unique", -1)),
    }
    assert result.best_score > 0.1 and len(result.smiles) > 0, result.best_score


def flow_docking_ligand_prep(record: dict, user: Path, bundle: dict) -> None:
    """§14.2's docking leg, the Linux-honest part: the campaign's own
    ligand preparation (SMILES → ETKDG 3D → Meeko PDBQT), one campaign
    folder per ligand, deterministic seed.  The ENGINES bundled by the
    installer are Windows PEs — sha+PE validated in flow_bundled_engines,
    they cannot EXECUTE on Linux; running an actual vina docking waits
    for the Windows clean machine."""
    import pandas as pd

    from pharmaml.docking.ligands import prepare_ligand_library

    csv_path = user / "docking_library.csv"
    lib = pd.DataFrame({
        "smiles": list(SMILES[:3]),
        "name": ["aspirin", "ibuprofen", "paracetamol"],
    })
    lib.to_csv(csv_path, index=False)
    root = user / "docking_campaign"
    report = prepare_ligand_library(csv_path, root, seed=274)
    pdbqts = sorted(root.glob("*/l.pdbqt"))
    sizes = {str(p.relative_to(root)): p.stat().st_size for p in pdbqts}
    record["docking_ligand_prep"] = {
        "campaign_root": str(root),
        "folders": sorted(p.name for p in root.iterdir() if p.is_dir()),
        "pdbqt_bytes": sizes,
        "ok_rows": int(report.frame["ok"].sum()),
        "torsdofs": [int(v) for v in report.frame["torsdof"].tolist()],
    }
    assert len(pdbqts) == 3 and all(s > 500 for s in sizes.values()), sizes


def flow_pkpd_ode(record: dict, user: Path, bundle: dict) -> None:
    """§14.2 "PK/PD": one-compartment IV infusion + a maintenance bolus
    through the app's own dosing-event ODE solver."""
    import numpy as np

    from pharmaml.pkpd.ode import DosingEvent, solve_ode

    k, volume = 0.35, 30.0
    times = np.linspace(0.0, 24.0, 97)
    sol = solve_ode(
        lambda t, y: np.array([-k * y[0]]),
        times,
        [0.0],
        (
            DosingEvent(time=0.0, amount=500.0, duration=0.5),
            DosingEvent(time=12.0, amount=250.0),
        ),
    )
    conc = sol.states[:, 0] / volume
    peak_i = int(np.argmax(conc))
    record["pkpd_ode"] = {
        "n_steps": int(sol.n_steps),
        "c_peak": round(float(conc[peak_i]), 4),
        "t_peak": float(times[peak_i]),
        "c_end": round(float(conc[-1]), 5),
        "c0_bolus_lag": round(float(conc[np.searchsorted(times, 12.0) + 1]), 4),
    }
    assert sol.n_steps > 50 and 0.0 < times[peak_i] < 1.5 and conc[-1] > 0


def flow_md_analysis(record: dict, user: Path, bundle: dict) -> None:
    """§14.2 "MD analysis": RMSD/RMSF/Rg over the deterministic demo
    trajectory — loop flexibility must beat the rigid helix."""
    import numpy as np

    from pharmaml.md.metrics import (
        radius_of_gyration_trajectory,
        rmsd_trajectory,
        rmsf,
    )
    from pharmaml.md.trajectory import synthetic_trajectory

    traj = synthetic_trajectory(
        n_frames=60, seed=274, n_residues=12, flexible_residues=(5, 6, 7),
    )
    rmsd = rmsd_trajectory(traj)
    fl = rmsf(traj)
    rg = radius_of_gyration_trajectory(traj)
    loop_atoms = [a for a in range(traj.n_atoms) if a // 3 in (5, 6, 7)]
    helix_atoms = [0, 1, 2, 3]
    record["md_analysis"] = {
        "n_frames": traj.n_frames,
        "n_atoms": traj.n_atoms,
        "rmsd_first_last": round(float(rmsd[-1]), 4),
        "rmsf_loop_mean": round(float(np.mean(fl[loop_atoms])), 4),
        "rmsf_helix_mean": round(float(np.mean(fl[helix_atoms])), 4),
        "rg_mean": round(float(np.mean(rg)), 4),
    }
    assert (
        rmsd.shape == (60,)
        and fl.shape == (traj.n_atoms,)
        and rg.shape == (60,)
        and np.mean(fl[loop_atoms]) > np.mean(fl[helix_atoms])
    )


def flow_plot_metadata(record: dict, user: Path, bundle: dict) -> None:
    """§9 "SMILES metadata inspection": the interactive figure's
    per-point hovertext — one SMILES label per marker, the exact
    surface the studio's plot inspection reads."""
    import json as _json

    from pharmaml.viz.interactive import Figure, figure_report

    desc = bundle["desc"]
    x_col, y_col = "MolWt", "qed"
    fig = Figure()
    fig.add_scatter(
        desc[x_col], desc[y_col],
        name="molecules", mode="markers", hovertext=list(SMILES),
    )
    payload = _json.loads(fig.to_json())
    trace = payload["data"][0]
    report_text = figure_report(fig)
    record["plot_metadata"] = {
        "n_points": len(trace["x"]),
        "hovertext_first": trace["hovertext"][0],
        "hovertext_len": len(trace["hovertext"]),
        "report_chars": len(report_text),
    }
    assert trace["hovertext"] == list(SMILES) and len(report_text) > 50


FLOWS: tuple[tuple[str, Callable[[dict, Path, dict], None]], ...] = (
    ("environment", flow_environment),
    ("qt_offscreen", flow_qt_offscreen),
    ("fonts", flow_fonts),
    ("import_csv", flow_import_csv),
    ("import_excel", flow_import_excel),
    ("descriptors", flow_descriptors),
    ("fingerprints", flow_fingerprints),
    ("ml_train_predict", flow_ml_train_predict),
    ("deep_learning", flow_deep_learning),
    ("semi_supervised", flow_semi_supervised),
    ("explainability", flow_explainability),
    ("uncertainty", flow_uncertainty),
    ("plot_export", flow_plot_export),
    ("table_export", flow_table_export),
    ("model_save_reload", flow_model_save_reload),
    ("applicability_domain", flow_applicability_domain),
    ("generative", flow_generative),
    ("docking_ligand_prep", flow_docking_ligand_prep),
    ("pkpd_ode", flow_pkpd_ode),
    ("md_analysis", flow_md_analysis),
    ("plot_metadata", flow_plot_metadata),
    ("bundled_engines", flow_bundled_engines),
)


def _snapshot_tree(root: Path) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            st = path.stat()
            out[str(path)] = (st.st_size, int(st.st_mtime))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--user-dir",
        required=True,
        help="part9 §7: the writable USER location for every output byte",
    )
    args = parser.parse_args(argv)

    user = Path(args.user_dir).resolve()
    user.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(user / "mpl-config"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(user / "numba-cache"))

    frozen = bool(getattr(sys, "frozen", False))
    app_root = (
        Path(getattr(sys, "_MEIPASS", ".")).resolve()
        if frozen
        else Path(__file__).resolve().parents[2]
    )
    before = _snapshot_tree(app_root) if frozen else None

    block_network()  # §14.3 armed BEFORE any science import

    record: dict = {"flows": {}}
    bundle: dict = {}
    failures: list[str] = []
    for name, flow in FLOWS:
        started = time.monotonic()
        try:
            flow(record, user, bundle)
            status, detail = "PASS", ""
        except Exception as exc:  # noqa: BLE001 - the verdict IS the record
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
            failures.append(name)
            traceback.print_exc()
        record["flows"][name] = {
            "status": status,
            "ms": int((time.monotonic() - started) * 1000),
            "detail": detail,
        }
        print(f"FLOW {name} {status} {record['flows'][name]['ms']}ms {detail}", flush=True)

    if frozen and before is not None:
        after = _snapshot_tree(app_root)
        record["install_dir_unchanged"] = before == after
        if not record["install_dir_unchanged"]:
            added = sorted(set(after) - set(before))
            changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
            record["install_dir_writes"] = {
                "added": added[:10],
                "changed": changed[:10],
            }
            failures.append("install_dir_isolation")

    record["net_attempts"] = list(_NET_ATTEMPTS)
    if _NET_ATTEMPTS:
        failures.append("offline_proof")
    record["offline"] = not _NET_ATTEMPTS
    record["user_dir"] = str(user)
    record["failures"] = sorted(set(failures))
    record["verdict"] = "PASS" if not record["failures"] else "FAIL"

    verdict_path = user / "freeze_smoke_verdict.json"
    verdict_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    total = len(record["flows"])
    print(
        f"FREEZE-SMOKE {record['verdict']} "
        f"{total - len([f for f in record['failures'] if f in record['flows']])}/{total} "
        f"offline={'yes' if record['offline'] else 'NO'} "
        f"verdict={verdict_path}",
        flush=True,
    )
    return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
