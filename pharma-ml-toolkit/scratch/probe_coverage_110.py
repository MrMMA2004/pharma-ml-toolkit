#!/usr/bin/env python3
"""Headless E2E probe for the 100x-110 coverage cards (Utilities tab,
converters, similarity scan, docking audit/vbatch/smi-pdbqt, clean frame).

Every new GUI surface is driven exactly the way a mouse would drive it:
pickers are replaced by direct attribute sets, Run handlers are invoked,
and the render path is flushed through the app's own job pump.  The probe
asserts the *visible* outcome (tree rows, summary labels) instead of the
library calls -- the library suite already pins those.
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
    """Make every modal dialog print-and-return (True for questions)."""
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


def main() -> int:
    _dismiss_dialogs()
    tmp = Path(tempfile.mkdtemp(prefix="probe110_"))
    app = PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()

    # ---- fixtures -------------------------------------------------- #
    table_a = tmp / "ligands_a.csv"
    table_a.write_text(
        "Ligand,SMILES,score\n"
        "lig1,CCO,1.5\nlig2,CCOCC,2.5\nlig3,c1ccccc1,3.5\n")
    table_b = tmp / "ligands_b.csv"
    table_b.write_text(
        "Ligand,SMILES\n"
        "lig1,CCO\nlig4,CNC\n")
    big_csv = tmp / "big.csv"
    big_csv.write_text("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(25)))
    measure_csv = tmp / "measure.csv"
    measure_csv.write_text("compound,IC50 (nM)\nA,100\nB,<10\nC,xyz\n")
    approval = tmp / "approvals.tsv"
    approval.write_text(
        "drug\tregulatory approval\naspirin\tapproved\ncaffeine\twithdrawn\n"
        "ibuprofen\tapproved\n")
    sparse = tmp / "sparse.csv"
    sparse.write_text("ligand,energy\n5_result.dlg,-7.1\n6_result.dlg,-8.2\n"
                      "8_result.dlg,-6.0\n")

    print("== Utilities: merge & split ==")
    app._util_merge_files = [table_a, table_b]
    app._util_merge_mode.set("merge picked files")
    app._util_merge_mode_changed()
    app._util_merge_run()
    app._pump_probe()
    merged = app._util_frames.get("merge")
    check("merge picked files", merged is not None and len(merged) == 5,
          f"{len(merged) if merged is not None else 0} rows")
    firstcols = app._util_merge_firstcols
    firstcols.delete(0, "end")
    firstcols.insert(0, "score,SMILES")
    app._util_merge_run()
    app._pump_probe()
    merged = app._util_frames.get("merge")
    check("first-columns reorder",
          merged is not None and next(iter(merged.columns)) == "score",
          f"first col {next(iter(merged.columns)) if merged is not None else '-'}")

    parts_dir = tmp / "parts"
    app._util_merge_split_file = big_csv
    app._util_merge_mode.set("split into chunks")
    app._util_merge_mode_changed()
    app._util_merge_chunk.delete(0, "end")
    app._util_merge_chunk.insert(0, "10")
    # split asks for an output folder via dialog -- stub it
    real_askdir = pg.filedialog.askdirectory
    pg.filedialog.askdirectory = lambda **_k: str(parts_dir)
    try:
        app._util_merge_run()
        app._pump_probe()
    finally:
        pg.filedialog.askdirectory = real_askdir
    parts = sorted(parts_dir.glob("*.csv")) if parts_dir.exists() else []
    check("split into chunks", len(parts) == 3,
          f"{len(parts)} part files")

    print("== Utilities: compare & annotate ==")
    app._util_cmp_a = table_a
    app._util_cmp_b = table_b
    app._util_cmp_refresh_info()
    app._util_cmp_mode.set("compare a column (A vs B)")
    app._util_compare_run()
    app._pump_probe()
    cmp_frame = app._util_frames.get("compare")
    sections = cmp_frame["section"].tolist() if cmp_frame is not None else []
    check("compare tables",
          sections.count("common") == 1 and sections.count("only in A") == 2
          and sections.count("only in B") == 1, str(sections))
    app._util_cmp_mode.set("annotate names with SMILES")
    app._util_compare_run()
    app._pump_probe()
    ann = app._util_frames.get("compare")
    check("annotate with SMILES",
          ann is not None and "Ligand_SMILES" in ann.columns
          and ann.loc[ann["Ligand"] == "lig1", "Ligand_SMILES"].iloc[0] == "CCO",
          f"cols {list(ann.columns) if ann is not None else '-'}")

    print("== Utilities: approved drugs ==")
    app._util_appr_files = [approval]
    app._util_approved_run()
    app._pump_probe()
    appr = app._util_frames.get("approved")
    check("approved filter",
          appr is not None and appr["drug"].tolist() == ["aspirin", "ibuprofen"],
          str(appr["drug"].tolist() if appr is not None else "-"))

    print("== Utilities: campaign file tools ==")
    camp = tmp / "campaign"
    for sub in ("lig1", "lig2", "lig3"):
        (camp / sub).mkdir(parents=True)
    (camp / "lig1" / "r.dlg").write_text("fake dlg")
    (camp / "lig2" / "r.dlg").write_text(
        "fake dlg\nSuccessful Completion\n")
    (camp / "lig1" / "1.pdbqt").write_text(
        "REMARK x\nTORSDOF 2\nBRANCH\nBRANCH\n")
    (camp / "stray.txt").write_text("stray")

    app._util_tool_folder = camp
    app._util_tool_mode.set("collect files (name/pattern)")
    app._util_tool_mode_changed()
    app._util_tool_filename.delete(0, "end")
    app._util_tool_filename.insert(0, "r.dlg")
    app._util_tools_run()
    app._pump_probe()
    tools_frame = app._util_frames.get("tools")
    check("collect files (dry run, filename ∪ pattern)",
          tools_frame is not None and len(tools_frame) == 3,
          f"{len(tools_frame) if tools_frame is not None else 0} matches "
          f"(2×r.dlg + 1.pdbqt via the pattern union)")
    collected = camp / "collected"
    check("collect copies nothing on dry run", not collected.exists()
          or not list(collected.glob("*.dlg")),
          "dry-run safety")

    app._util_tool_mode.set("list file names")
    app._util_tools_run()
    app._pump_probe()
    listed = app._util_frames.get("tools")
    check("list file names",
          listed is not None and "name" in listed.columns,
          f"{len(listed) if listed is not None else 0} names")

    app._util_tool_files = [tmp / "bin_marker.txt"]
    (tmp / "bin_marker.txt").write_text("bin")
    app._util_tool_mode.set("distribute files into folders")
    app._util_tools_run()
    app._pump_probe()
    copies = list(camp.rglob("bin_marker.txt"))
    check("distribute files", len(copies) == 3,
          f"{len(copies)} copies")

    app._util_tool_mode.set("cleanup tree (keep list)")
    app._util_tool_dry_var.set(False)
    app._util_tools_run()
    app._pump_probe()
    app._util_tool_dry_var.set(True)
    survivors = [p.name for p in camp.rglob("*") if p.is_file()]
    check("cleanup keeps allowlist ∪ keep-pattern",
          all(name in {"r.pdbqt", "r.dlg", "r.gpf", "r.dpf",
                       "bin_marker.txt", "1.pdbqt"} for name in survivors),
          f"{survivors} (1.pdbqt survives via the keep pattern, "
          f"stray.txt + distributed markers deleted)")

    print("== Utilities: reindex by number ==")
    app._util_reindex_path = sparse
    app._util_reindex_column.delete(0, "end")
    app._util_reindex_column.insert(0, "ligand")
    app._util_reindex_start.delete(0, "end")
    app._util_reindex_start.insert(0, "5")
    app._util_reindex_stop.delete(0, "end")
    app._util_reindex_stop.insert(0, "8")
    app._util_reindex_run()
    app._pump_probe()
    reindexed = app._util_frames.get("reindex")
    check("reindex dense range",
          reindexed is not None and len(reindexed) == 4,
          f"{len(reindexed) if reindexed is not None else 0} rows")

    print("== Chemistry: converters ==")
    sdf_out = tmp / "ligands.sdf"
    app._conv_input = table_a
    app._conv_output = sdf_out
    app._conv_mode.set("CSV table → 3D SDF")
    app._conv_run()
    app._pump_probe()
    check("CSV → 3D SDF", sdf_out.exists() and sdf_out.stat().st_size > 0,
          f"{sdf_out.stat().st_size if sdf_out.exists() else 0} bytes")

    smi_out = tmp / "list.smi"
    smi_out.write_text("CCO ethanol\nCC(C)CO isopropanol\n")
    smi_csv_out = tmp / "list.csv"
    app._conv_input = smi_out
    app._conv_output = smi_csv_out
    app._conv_mode.set(".smi list → CSV")
    app._conv_run()
    app._pump_probe()
    check(".smi → CSV", smi_csv_out.exists()
          and "SMILES" in smi_csv_out.read_text(),
          smi_csv_out.read_text()[:60].replace("\n", "|"))

    app._conv_input = sdf_out
    app._conv_mode.set("SDF → table")
    app._conv_run()
    app._pump_probe()
    sdf_frame = app._conv_frame
    check("SDF → table",
          sdf_frame is not None and len(sdf_frame) == 3
          and "score" in sdf_frame.columns,
          f"{len(sdf_frame) if sdf_frame is not None else 0} rows")

    app._conv_input = measure_csv
    app._conv_mode.set("append pIC50 column")
    app._conv_source_col.delete(0, "end")
    app._conv_source_col.insert(0, "IC50 (nM)")
    app._conv_run()
    app._pump_probe()
    pic_frame = app._conv_frame
    valid = int(pic_frame["pIC50"].notna().sum()) if pic_frame is not None else 0
    check("pIC50 column (censored)",
          pic_frame is not None and valid == 2
          and abs(pic_frame.loc[0, "pIC50"] - 7.0) < 1e-9,
          f"{valid}/3 valid (the '<10' censored row + the junk row stay NaN)")

    print("== Chemistry: similarity scan ==")
    app._scan_mode.set("matrix: references × library")
    app._scan_run()
    app._pump_probe()
    matrix_frame = app._scan_frame
    check("similarity matrix",
          matrix_frame is not None
          and matrix_frame.shape == (2, 1 + 7)
          and abs(float(matrix_frame.iloc[0, 1]) - 1.0) < 1e-9,
          f"shape {matrix_frame.shape if matrix_frame is not None else '-'}"
          f" (aspirin × aspirin = 1.0)")

    app._scan_mode.set("scan: pairs ≥ threshold")
    app._scan_threshold.delete(0, "end")
    app._scan_threshold.insert(0, "0.4")
    app._scan_run()
    app._pump_probe()
    scan_frame = app._scan_frame
    identity_pairs = set()
    if scan_frame is not None:
        identity_pairs = {
            (row.reference, row.target) for row in scan_frame.itertuples()
            if row.similarity >= 0.999}
    check("threshold scan",
          scan_frame is not None and len(scan_frame) == 2
          and len(identity_pairs) == 2
          and {"reference", "target", "similarity"} <= set(scan_frame.columns),
          f"{len(scan_frame) if scan_frame is not None else 0} pairs ≥ 0.4 "
          f"(both identity pairs; aspirin×paracetamol sits below the "
          f"rdkit path-fingerprint threshold -- honest, not a bug)")

    app._scan_mode.set("top-k: nearest of first reference")
    app._scan_k.delete(0, "end")
    app._scan_k.insert(0, "3")
    app._scan_run()
    app._pump_probe()
    top_frame = app._scan_frame
    check("top-k matches",
          top_frame is not None and len(top_frame) == 3
          and top_frame["similarity"].iloc[0] >= top_frame["similarity"].iloc[-1],
          f"best {top_frame['similarity'].iloc[0]:.3f}")

    print("== Docking: audit card ==")
    gpf = tmp / "r.gpf"
    gpf.write_text(
        "ligand_types C HD OA\nreceptor r.pdbqt\nreceptor_types C HD OA A\n"
        "npts 40 40 40\nspacing 0.375\ngridcenter 1.0 2.0 3.0\n"
        "smooth 0.5\ndielectric -0.1465\nmap 1.C.map\n")
    app._dock_audit_gpf = gpf
    app._dock_audit_mode.set("GPF file → parameters")
    app._dock_audit_run()
    app._pump_probe()
    params_frame = app._dock_audit_frame
    check("GPF → parameters",
          params_frame is not None and len(params_frame) == 9
          and params_frame.loc[0, "value"] == "C HD OA",
          f"{len(params_frame) if params_frame is not None else 0} params")

    app._dock_audit_mode.set("folder: torsion consistency")
    app._dock_audit_folder = camp
    app._dock_audit_run()
    app._pump_probe()
    torsion_frame = app._dock_audit_frame
    check("torsion consistency scan",
          torsion_frame is not None and len(torsion_frame) >= 1
          and {"ligand", "torsdof", "branches", "status"}
          <= set(torsion_frame.columns),
          f"{len(torsion_frame) if torsion_frame is not None else 0} ligands")

    app._dock_audit_mode.set("folder: completion check")
    app._dock_audit_run()
    app._pump_probe()
    completion_frame = app._dock_audit_frame
    statuses = completion_frame["status"].tolist() if (
        completion_frame is not None) else []
    check("completion check",
          "OK" in statuses and "INCOMPLETE" in statuses
          and "MISSING_LOG" in statuses,
          f"{statuses} (lig2 carries the Successful Completion sentinel, "
          f"lig1 does not, lig3 has no log at all)")

    print("== Docking: single SMILES → PDBQT ==")
    smi_dir = tmp / "smi_out"
    app._dock_smi_out = smi_dir
    app._dock_smi_entry.delete(0, "end")
    app._dock_smi_entry.insert(0, "CC(=O)Oc1ccccc1C(=O)O")
    app._dock_smi_run()
    app._pump_probe()
    pdbqt = list(smi_dir.glob("*.pdbqt")) if smi_dir.exists() else []
    check("SMILES → PDBQT", len(pdbqt) == 1,
          pdbqt[0].name if pdbqt else "none")

    print("== Docking: Vina config batch (fake engine) ==")
    import os

    binroot = tmp / "bin"
    binroot.mkdir()
    (binroot / "vina").write_text(
        "#!/bin/bash\n"
        "out=''; lig=''; prev=''\n"
        "for arg in \"$@\"; do\n"
        "  [ \"$prev\" = --out ] && out=\"$arg\"\n"
        "  [ \"$prev\" = --ligand ] && lig=\"$arg\"\n"
        "  prev=\"$arg\"\n"
        "done\n"
        "{ echo 'REMARK VINA RESULT:    -8.2      0.000      0.000'\n"
        "  echo 'REMARK VINA RESULT:    -7.1      1.923      2.941'\n"
        "  grep '^ATOM\\|^HETATM' \"$lig\"; } > \"$out\"\n"
        "exit 0\n"
    )
    os.chmod(binroot / "vina", 0o755)
    os.environ["PATH"] = f"{binroot}{os.pathsep}{os.environ.get('PATH', '')}"

    vbatch_dir = tmp / "vbatch"
    (vbatch_dir / "ligands").mkdir(parents=True)
    for name in ("l1", "l2"):
        (vbatch_dir / "ligands" / f"{name}.pdbqt").write_text(
            "REMARK ok\nATOM      1  C   LIG A   1       0.000   0.000   "
            "0.000  1.00  0.00           C\n")
    receptor = vbatch_dir / "r.pdbqt"
    receptor.write_text("ATOM receptor\n")
    conf = vbatch_dir / "vina.conf"
    conf.write_text(
        "receptor = r.pdbqt\ncenter_x = 0\nsize_x = 20\n")
    vout = vbatch_dir / "results"

    app._dock_vbatch_conf = conf
    app._dock_vbatch_ligands = vbatch_dir / "ligands"
    app._dock_vbatch_out = vout
    app._dock_vbatch_refresh_info()
    app._dock_vbatch_run()
    app._pump_probe()
    outs = sorted(vout.glob("*_out.pdbqt")) if vout.exists() else []
    check("Vina config batch",
          len(outs) == 2 and app._dock_vbatch_frame is not None
          and len(app._dock_vbatch_frame) == 2,
          f"{len(outs)} out-pdbqt files, "
          f"{len(app._dock_vbatch_frame) if app._dock_vbatch_frame is not None else 0} rows")

    print("== Data & ML: clean numeric frame ==")
    mixed = tmp / "mixed.csv"
    mixed.write_text(
        "SMILES,name,MolWt,LogP,target\n"
        "CCO,ethanol,46.1,-0.3,1.0\nCC(C)CO,iso,60.1,0.3,2.0\n")
    app._ml_load_path(mixed)
    app._pump_probe()
    app._ml_keep_entry.delete(0, "end")
    app._ml_keep_entry.insert(0, "SMILES")
    app._ml_clean_frame()
    app._pump_probe()
    cleaned_active = list(app._ml_frame.columns)
    check("clean numeric frame",
          "SMILES" in cleaned_active and "name" not in cleaned_active
          and {"MolWt", "LogP", "target"} <= set(cleaned_active),
          str(cleaned_active))

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
