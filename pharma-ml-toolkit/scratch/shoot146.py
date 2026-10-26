"""SHOOT146 -- screenshot the round-146 surfaces (100x-146).

Opens the playground headlessly (Xvfb :99) and captures:
  * pg146-bio-expr-bar.png -- the Cell Lines tab while a genuinely
    large expression matrix (>= 8 MB, above the progressive gate)
    streams through read_indexed_table_with_progress: the DETERMINATE
    bar mid-read with the rows/ETA status label ("loading … — N/M ·
    ~x s left") -- the last marquee in the loader story, now determinate.
  * pg146-bio-phase.png -- the same tab's screen loader with the
    round's phase note: after the streamed read, the status bar says
    "auditing rows" (then "collapsing replicates") while the bar stays
    up -- the AUDIT/COLLAPSE phases finally speak.
  * pg146-ml-perm-save.png -- the Data & ML Results card after Train &
    evaluate + Permutation importance, with the round's Save CSV…
    button; the monkeypatched dialog writes the real table to disk and
    the facts print its header + top rows.

Run:  DISPLAY=:99 python scratch/shoot146.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PLAYGROUND_NO_SESSION", "1")

import playground  # noqa: E402

OUT = ROOT / "download"


def _grab(app: "playground.PlaygroundApp", filename: str) -> Path:
    from PIL import ImageGrab

    x, y = app.winfo_rootx(), app.winfo_rooty()
    w, h = app.winfo_width(), app.winfo_height()
    shot = OUT / filename
    ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(shot)
    print(f"wrote {shot}")
    return shot


def _select_tab(app: "playground.PlaygroundApp", name: str) -> None:
    for widget in app._walk_widgets(app):
        if widget.winfo_class() == "TNotebook":
            for tab_id in widget.tabs():
                if name in str(widget.tab(tab_id, "text")):
                    widget.select(tab_id)
                    return
    raise RuntimeError(f"tab not found: {name}")


def _drain(app: "playground.PlaygroundApp", timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def _spy_dialogs() -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(
            playground.messagebox,
            name,
            lambda *a, _seen=seen, _n=name, **k: _seen.append((_n, str(a))),
        )
    return seen


def main() -> int:
    dialogs = _spy_dialogs()
    app = playground.PlaygroundApp(persist_session=False)
    ok_bar = ok_phase = ok_perm = False
    facts: list[str] = []
    try:
        app.geometry("1280x900+0+0")
        app.update_idletasks()
        app.update()
        OUT.mkdir(exist_ok=True)

        # -- shot 1: the expression matrix's determinate bar ------------ #
        _select_tab(app, "Cell Lines")
        with tempfile.TemporaryDirectory(prefix="pg146_expr_") as tmp:
            expr_csv = Path(tmp) / "expr_matrix.csv"
            header = "cell_line," + ",".join(f"gene_{i:03d}" for i in range(12))
            rows = [header]
            for i in range(400_000):
                values = ",".join(f"{((i * 7 + j * 13) % 997) / 97:.2f}" for j in range(12))
                rows.append(f"CL{i % 20000:05d},{values}")
            expr_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
            size_mb = expr_csv.stat().st_size / (1024 * 1024)
            print(f"expression matrix: {size_mb:.1f} MB, 400k rows x 12 genes")

            app._bio_load_expr_file(expr_csv)
            determinate_seen = False
            label_mid = ""
            grabbed = False
            deadline = time.time() + 90.0
            while app._busy and time.time() < deadline:
                app.update()
                if str(app._progress.cget("mode")) == "determinate":
                    determinate_seen = True
                    label_mid = str(app._job_label.cget("text"))
                    if "/" in label_mid and not grabbed:
                        _grab(app, "pg146-bio-expr-bar.png")
                        grabbed = True
                time.sleep(0.002)
            assert not app._busy, "expression load job hung"
            info = str(app._bio_expr_info.cget("text"))
            facts.append(f"expr info: {info!r}")
            facts.append(f"expr mid-label: {label_mid!r}")
            ok_bar = (
                determinate_seen
                and grabbed
                and "loading" in label_mid
                and "/" in label_mid
                and "cell lines" in info
                and "× 12 genes" in info
            )

        # -- shot 2: the screen loader's phase note ---------------------- #
        with tempfile.TemporaryDirectory(prefix="pg146_screen_") as tmp:
            screen_csv = Path(tmp) / "screen.csv"
            rows = ["cell_line,compound,ln_ic50"]
            for i in range(450_000):
                rows.append(
                    f"line_{i % 500:03d},drug_{i % 40:03d},{(i % 900) / 100 + 0.25:.2f}"
                )
            screen_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
            print(f"screen: {screen_csv.stat().st_size / (1024 * 1024):.1f} MB")

            app._bio_load_file(screen_csv)
            phase_seen = ""
            grabbed = False
            deadline = time.time() + 120.0
            while app._busy and time.time() < deadline:
                app.update()
                label = str(app._job_label.cget("text"))
                if "auditing rows" in label and not grabbed:
                    phase_seen = label
                    _grab(app, "pg146-bio-phase.png")
                    grabbed = True
                time.sleep(0.002)
            assert not app._busy, "screen load job hung"
            info = str(app._bio_info.cget("text"))
            facts.append(f"screen info: {info!r}")
            facts.append(f"phase label: {phase_seen!r}")
            ok_phase = (
                grabbed
                and "auditing rows" in phase_seen
                and "rows" in info
                and "after collapsing replicates" in info
            )

        # -- shot 3: the permutation Save-CSV twin ----------------------- #
        _select_tab(app, "Data & ML")
        app._ml_load_example()
        app._model_combo.set("ridge")
        app._ml_refresh_hyperparam_panel()
        app._ml_train()
        assert _drain(app), "train job hung"
        app._ml_plot_permutation()
        assert _drain(app), "permutation job hung"
        saved = OUT / "pg146-permutation-table.csv"
        playground.filedialog.asksaveasfilename = (  # type: ignore[assignment]
            lambda t="", _p=saved, **k: str(_p)
        )
        app._ml_perm_save()
        app.update_idletasks()
        app.update()
        app._jump_to_card(5, "Results")
        app.update_idletasks()
        app.update()
        _grab(app, "pg146-ml-perm-save.png")
        note = str(app._ml_result_note.cget("text"))
        csv_head = saved.read_text(encoding="utf-8").splitlines()[:4]
        facts.append(f"result note: {note!r}")
        facts.append(f"csv head: {csv_head!r}")
        saved.unlink()
        ok_perm = (
            "permutation importance" in note
            and csv_head[0] == "feature,importance_mean,importance_std"
            and len(csv_head) > 2
        )
    finally:
        app.destroy()

    print("SHOOT146 facts:")
    for line in facts:
        print(f"  {line}")
    ok = ok_bar and ok_phase and ok_perm and not [d for d in dialogs if "error" in d[0]]
    print(f"SHOOT146_OK: {ok} (bar={ok_bar} phase={ok_phase} perm={ok_perm})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
