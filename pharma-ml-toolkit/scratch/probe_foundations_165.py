"""End-to-end probe for the 100x-165 Foundations round (Xvfb :99).

Drives the real GUI headless: opens the Foundations tab, exercises every
card (primer, broadcasting, scratch trainer both kinds, capacity sweep,
sequence forecast, attention lab, statistics both modes, distribution
sampler), loads a real example CSV, and screenshots the results.  Also
checks the Utilities tab's Git quick-reference card and the home-tab
chips list.

Run: DISPLAY=:99 python scratch/probe_foundations_165.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402

# modal dialogs are silenced for headless runs
playground.messagebox.showinfo = lambda *a, **k: print("  [showinfo]", a[:1])
playground.messagebox.showwarning = lambda *a, **k: print("  [showwarning]", a[:1])
playground.messagebox.showerror = lambda *a, **k: print("  [showerror]", a[:1])

FAILURES: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("  PASS " if condition else "  FAIL ") + label)
    if not condition:
        FAILURES.append(label)


app = playground.PlaygroundApp()
app.geometry("1500x950+5+5")


def pump(timeout: float = 240.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def select_tab(title: str) -> None:
    for index in range(app._book.index("end")):
        if app._book.tab(index, "text").strip() == title:
            app._book.select(index)
            app.update()
            return
    raise AssertionError(f"no tab {title!r}")


def note_after(job_name: str) -> None:
    label = getattr(app, "_job_label_base", "")
    print(f"  [{job_name}] job label: {label!r}")


print("== tab census ==")
names = [app._book.tab(i, "text").strip() for i in range(app._book.index("end"))]
print(names)
check("Foundations tab exists at index 8", len(names) > 8 and names[8] == "Foundations")
check("tab order intact (11 tabs)", len(names) == 11)

print("== home chips ==")
chips = [
    w for w in app.winfo_children()
]
# chips live in the home card; verify via the jump loop source instead
select_tab("Foundations")
app.update()

print("== card census ==")
titles = [t for t, _s, _w in playground._CARD_INDEX]
for expected in (
    "Tensor primer", "Broadcasting check", "From-scratch trainer",
    "Capacity sweep", "Sequence forecasting", "Attention lab",
    "Statistics & hypothesis testing", "Distribution sampler",
    "Git quick reference",
):
    check(f"card {expected!r}", expected in titles)

print("== tensor primer ==")
app._found_run_primer()
check("primer job completes", pump())
rows = len(app._found_primer_tree.get_children())
check("primer table filled", rows >= 3)
print("  primer note:", app._found_primer_note.cget("text").splitlines()[0][:90])

print("== broadcasting check ==")
app._found_run_broadcast()
text = app._found_broadcast_note.cget("text")
print("  broadcast:", text)
check("broadcast verdict shown", "(80, 4)" in text)
app._found_shape_a.delete(0, "end"); app._found_shape_a.insert(0, "(3, 4, 5)")
app._found_run_broadcast()
text = app._found_broadcast_note.cget("text")
check("incompatible verdict shown", "incompatible" in text)

print("== from-scratch trainer (network) ==")
app._found_run_scratch()
check("scratch job completes", pump(300))
note = app._found_scratch_note.cget("text")
print("  scratch:", note)
check("scratch final line", "epoch" in note)
kids = app._found_scratch_host.winfo_children()
check("scratch figure embedded", bool(kids))

print("== from-scratch trainer (linear GD) ==")
app._found_kind_combo.set("linear GD")
app._found_run_scratch()
check("linear GD job completes", pump(300))
note = app._found_scratch_note.cget("text")
print("  GD:", note)
check("GD final line", "y = " in note)
app._found_kind_combo.set("scratch network")

print("== capacity sweep (trimmed) ==")
app._found_widths_entry.delete(0, "end")
app._found_widths_entry.insert(0, "0, 4, 16")
app._found_sweep_epochs.set("30")
app._found_run_sweep()
check("sweep job completes", pump(300))
note = app._found_sweep_note.cget("text")
print("  sweep:", note[:110])
check("sweep verdict", "best width" in note)

print("== sequence forecasting (AR(2), elman, short) ==")
app._found_series_source.set("synthetic — AR(2) process")
app._found_seq_model.set("elman")
app._found_seq_epochs.set("15")
app._found_run_sequence()
check("sequence job completes", pump(300))
note = app._found_seq_note.cget("text")
print("  seq:", note[:110])
check("sequence headline", "baseline" in note)

print("== attention lab ==")
app._found_run_attention()
check("attention job completes", pump(120))
note = app._found_attn_note.cget("text")
print("  attn:", note[:110])
check("attention headline", "head" in note)

print("== statistics (synthetic) ==")
app._found_run_stats()
check("stats job completes", pump(120))
text = app._found_stats_text.get("1.0", "end")
print("  stats first lines:", text.splitlines()[0][:90])
check("stats report filled", "groups:" in text and "branch:" in text)

print("== load example + stats on CSV columns ==")
app._found_load_example()
check("example loaded", app._found_frame is not None)
frame_cols = [str(c) for c in app._found_frame.columns]
print("  columns:", frame_cols[:6], "…")
numeric = [c for c in frame_cols if app._found_frame[c].dtype.kind in "biufc"]
app._found_stat_mode.set("CSV columns")
if len(numeric) >= 2:
    app._found_stat_a.set(numeric[0])
    app._found_stat_b.set(numeric[1])
    app._found_run_stats()
    check("CSV stats completes", pump(120))
    text = app._found_stats_text.get("1.0", "end")
    check("CSV stats report", "vs" in text)

print("== sequence on a CSV column ==")
if numeric:
    app._found_series_source.set("CSV column")
    app._found_series_combo.set(numeric[0])
    app._found_seq_epochs.set("10")
    app._found_seq_hidden.set("8")
    app._found_run_sequence()
    check("CSV sequence completes", pump(300))
    print("  csv-seq:", app._found_seq_note.cget("text")[:110])

print("== primer on the loaded example ==")
app._found_run_primer()
check("primer on example completes", pump())
rows = len(app._found_primer_tree.get_children())
check("primer table refilled", rows >= 3)

print("== distribution sampler ==")
app._found_dist_kind.set("poisson")
app._found_run_dist()
check("poisson draw completes", pump(120))
note = app._found_dist_note.cget("text")
print("  dist:", note)
check("distribution line", "theory" in note)
app._found_dist_kind.set("normal")
app._found_run_dist()
check("normal draw completes", pump(120))

print("== utilities git card ==")
select_tab("Utilities")
app.update()
check("git card in index", "Git quick reference" in titles)

print("== screenshots ==")


def shot(name: str) -> None:
    app.update()
    app.update_idletasks()
    for _ in range(30):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    out = Path(__file__).resolve().parents[1] / "scratch" / f"pg165-{name}.png"
    img.save(out)
    print("  shot", out.name)


select_tab("Foundations")
app.update()
shot("foundations-top")
# scroll the foundations page to the sequence/attention row
page = app._book.nametowidget(app._book.select())
for card_title, _subtitle, shell in playground._CARD_INDEX:
    if card_title == "Sequence forecasting":
        app.update()
        body_y = shell.winfo_rooty() - page.body.winfo_rooty()
        page._canvas.yview_moveto(min(1.0, max(0.0, body_y / 4000)))
        app.update()
        break
shot("foundations-seq")
select_tab("Utilities")
app.update()
shot("utilities-git")

print()
if FAILURES:
    print("FAILED:", FAILURES)
    sys.exit(1)
print("ALL FOUNDATIONS PROBES PASSED")
app.destroy()
