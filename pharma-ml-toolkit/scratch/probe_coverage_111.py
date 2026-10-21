#!/usr/bin/env python3
"""Headless E2E probe for the 100x-111 surfaces: the Batch GPF/DPF
campaign card (write_config_batch + the receptor-types field) and the
Scan Vina results folder button (scan_vina_directory).

Driven exactly the way a mouse would drive it: pickers replaced by
direct attribute sets, Run handlers invoked, renders flushed through
the app's own job pump.  The visible outcome (grid text report, tree
rows, info label) is asserted -- the library suite pins the calls.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import playground as pg
from playground import PlaygroundApp

FAILURES: list[str] = []
DIALOGS: list[str] = []


def _dismiss_dialogs() -> None:
    for name in ("showerror", "showwarning", "showinfo"):
        setattr(pg.messagebox, name,
                lambda *a, **k: DIALOGS.append(str(a)) or None)
    for name in ("askyesno", "askokcancel"):
        setattr(pg.messagebox, name,
                lambda *a, **k: DIALOGS.append(str(a)) or True)


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


LIGAND = (
    "REMARK  Name = probe_ligand\n"
    "ROOT\n"
    "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     C\n"
    "ATOM      2  O   LIG A   1       2.000   0.000   0.000  1.00  0.00     OA\n"
    "ATOM      3  N   LIG A   1       0.000   2.000   0.000  1.00  0.00     NA\n"
    "ENDROOT\n"
    "TORSDOF 1\n"
)

VINA_LOG = (
    "Estimated Free Energy of Binding   = -7.051 (kcal/mol)\n"
    "REMARK VINA RESULT:    -7.051      0.000      0.000\n"
    "REMARK VINA RESULT:    -6.802      1.938      3.655\n"
)


def main() -> int:
    _dismiss_dialogs()
    tmp = Path(tempfile.mkdtemp(prefix="probe111_"))
    app = PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    # ---- fixtures -------------------------------------------------- #
    campaign = tmp / "campaign"
    for name in ("01_aspirin", "02_ibuprofen"):
        folder = campaign / name
        folder.mkdir(parents=True)
        (folder / "l.pdbqt").write_text(LIGAND, encoding="utf-8")
    (campaign / "03_empty").mkdir()

    vina_root = tmp / "vina_results"
    (vina_root / "lig01").mkdir(parents=True)
    (vina_root / "lig01" / "lig01.log").write_text(VINA_LOG, encoding="utf-8")
    (vina_root / "lig02.log").write_text(VINA_LOG.replace("-7.051", "-8.35"),
                                         encoding="utf-8")
    (vina_root / "crashed.log").write_text("Vina crashed before scoring\n")

    single = tmp / "single"
    single.mkdir()
    ligand_path = single / "l.pdbqt"
    ligand_path.write_text(LIGAND, encoding="utf-8")

    # ---- 1. Batch GPF/DPF for campaign ------------------------------ #
    print("== Docking: Batch GPF/DPF for the campaign tree ==")
    real_askdir = pg.filedialog.askdirectory
    pg.filedialog.askdirectory = lambda **_k: str(campaign)
    try:
        app._dock_write_config_batch()
        app._pump_probe()
    finally:
        pg.filedialog.askdirectory = real_askdir

    report_text = app._dock_grid_text.get("1.0", "end")
    check("batch report shows the configured count",
          "configured      2/3 folders" in report_text
          and "without a ligand" in report_text,
          report_text.splitlines()[:2])
    check("batch report lists every folder",
          all(name in report_text for name in
              ("01_aspirin", "02_ibuprofen", "03_empty"))
          and "no ligand PDBQT" in report_text)
    check("both r.gpf/r.dpf pairs exist on disk",
          all((campaign / n / f).is_file() for n in
              ("01_aspirin", "02_ibuprofen")
              for f in ("r.gpf", "r.dpf")))
    gpf_text = (campaign / "01_aspirin" / "r.gpf").read_text(encoding="utf-8")
    check("default receptor types in the written GPF",
          "A C HD N NA OA SA" in gpf_text)

    # ---- 2. receptor-types field -> single-ligand path --------------- #
    print("== Docking: receptor types field (gpferreceptorni.py) ==")
    app._dock_set_ligand_path(ligand_path)
    app._dock_receptor_types_entry.delete(0, "end")
    app._dock_receptor_types_entry.insert(0, "A C HD N Ni OA SA")
    app._dock_compute_grid()
    app._pump_probe()
    grid_text = app._dock_grid_text.get("1.0", "end")
    check("single-ligand grid report echoes the Ni types",
          "receptor types A C HD N Ni OA SA" in grid_text,
          grid_text.splitlines()[3] if len(grid_text.splitlines()) > 3 else "")
    single_gpf = (single / "r.gpf").read_text(encoding="utf-8")
    check("Ni types landed in the written single GPF",
          "A C HD N Ni OA SA" in single_gpf)
    app._dock_receptor_types_entry.delete(0, "end")  # back to default

    # ---- 3. Batch with custom receptor types ------------------------- #
    print("== Docking: batch with the Ni receptor types ==")
    for name in ("01_aspirin", "02_ibuprofen"):
        (campaign / name / "r.gpf").unlink()  # force a fresh write
    app._dock_receptor_types_entry.delete(0, "end")
    app._dock_receptor_types_entry.insert(0, "A C HD N Ni OA SA")
    real_askdir = pg.filedialog.askdirectory
    pg.filedialog.askdirectory = lambda **_k: str(campaign)
    try:
        app._dock_write_config_batch()
        app._pump_probe()
    finally:
        pg.filedialog.askdirectory = real_askdir
    batch_gpf = (campaign / "01_aspirin" / "r.gpf").read_text(encoding="utf-8")
    check("batch GPF carries the Ni receptor types",
          "A C HD N Ni OA SA" in batch_gpf)
    app._dock_receptor_types_entry.delete(0, "end")

    # ---- 4. Scan Vina results folder --------------------------------- #
    print("== Docking: Scan Vina results folder ==")
    real_askdir = pg.filedialog.askdirectory
    pg.filedialog.askdirectory = lambda **_k: str(vina_root)
    try:
        app._dock_scan_vina()
        app._pump_probe()
    finally:
        pg.filedialog.askdirectory = real_askdir

    rows = app._dock_results_tree.get_children()
    check("three logs in the results tree", len(rows) == 3, f"rows={len(rows)}")
    values = [app._dock_results_tree.item(row, "values") for row in rows]
    check("strongest binder first (lig02, -8.35)",
          values and values[0][0] == "lig02" and "-8.35" in str(values[0][2]),
          str(values[:1]))
    check("crashed log keeps its NaN row",
          any(v[0] == "crashed" for v in values))
    info = app._dock_results_info.cget("text")
    check("info label reports the best binder",
          "3 Vina logs" in info and "lig02" in info and "-8.35" in info,
          info)
    check("results frame is the scan table (saveable)",
          app._dock_results_source == "vina"
          and list(app._dock_results_frame.columns) ==
          ["ligand", "log", "best_affinity", "n_poses"])

    app.destroy()
    print()
    if DIALOGS:
        print(f"note: {len(DIALOGS)} dialog(s) fired: {DIALOGS[:3]}")
    if FAILURES:
        print(f"PROBE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL PROBES PASSED")
    return 0


def _pump(self: PlaygroundApp, rounds: int = 40) -> None:
    """Pump the event loop until the running job finishes."""
    import time

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        for _ in range(rounds):
            self.update_idletasks()
            self.update()
        if not getattr(self, "_busy", False):
            for _ in range(3):
                self.update_idletasks()
                self.update()
            return
        time.sleep(0.02)
    raise TimeoutError("probe pump timed out -- a job never finished")


PlaygroundApp._pump_probe = _pump  # type: ignore[attr-defined]

if __name__ == "__main__":
    raise SystemExit(main())
