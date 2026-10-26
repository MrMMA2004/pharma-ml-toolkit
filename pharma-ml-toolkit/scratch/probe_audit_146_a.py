"""Task 2-a (100x-146) probe A -- round-145 GUI additions, live app on Xvfb.

Verifies against the real app:
  1. the prep card's aggregate combobox and the permutation repeats entry
     ride the session (collect -> dump -> load -> _restore_session);
  2. "Reset current tab to defaults" restores both launch values;
  3. both widgets are OWNED by the Data & ML tab (the copy/reset walk);
  4. the ecfp3 family checkbox rides the session families map;
  5. STALENESS: adopting a NEW dataset never clears _ml_prep_frame /
     _ml_cleaned_frame -- a later "Save prep/cleaned CSV..." would write
     the PREVIOUS dataset's derived frame;
  6. the prep error path: a frame with no SMILES/EC50 columns surfaces as
     the standard job "Calculation failed" dialog (the job machinery owns
     error display);
  7. the permutation needs-run guard fires showinfo BEFORE any dependency
     gate (order pinned by the learning-curve convention).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import playground  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def drain_jobs(app, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


app = None
seen: list[tuple[str, tuple]] = []


def spy(name, fn):
    def wrapper(*a, **k):
        seen.append((name, a))
        return None

    return wrapper


try:
    app = playground.PlaygroundApp(persist_session=False)

    # -- 1. session round-trip for the two new widgets ------------------- #
    app._ml_aggregate_combo.set("mean")
    app._perm_repeats.delete(0, "end")
    app._perm_repeats.insert(0, "9")
    app._feat_family_vars["ecfp3"].set(False)
    payload = app._collect_session()
    check(
        "1a. collect carries the aggregate combo + repeats entry",
        payload["variables"].get("_ml_aggregate_combo") == "mean"
        and payload["variables"].get("_perm_repeats") == "9",
        f"agg={payload['variables'].get('_ml_aggregate_combo')!r} "
        f"rep={payload['variables'].get('_perm_repeats')!r}",
    )
    check(
        "4a. collect carries families['ecfp3'] = False",
        payload["families"].get("ecfp3") is False,
        f"got={payload['families'].get('ecfp3')!r}",
    )
    session = playground.load_session(payload)
    app._session = session
    app._restore_session()
    check(
        "1b. restore puts back 'mean' and '9'",
        app._ml_aggregate_combo.get() == "mean" and app._perm_repeats.get() == "9",
        f"agg={app._ml_aggregate_combo.get()!r} rep={app._perm_repeats.get()!r}",
    )
    check(
        "4b. restore unticks ecfp3",
        bool(app._feat_family_vars["ecfp3"].get()) is False,
    )

    # -- 2. reset-to-defaults for both widgets --------------------------- #
    ml_index = None
    for tab_id in app._book.tabs():
        if str(app._book.tab(tab_id, "text")).strip() == "Data & ML":
            ml_index = app._book.index(tab_id)
            break
    check("2a. found the Data & ML tab index", ml_index is not None, f"index={ml_index}")
    if ml_index is not None:
        app._reset_tab(ml_index)
        app.update()
        check(
            "2b. reset restores 'median' and '5'",
            app._ml_aggregate_combo.get() == "median" and app._perm_repeats.get() == "5",
            f"agg={app._ml_aggregate_combo.get()!r} rep={app._perm_repeats.get()!r}",
        )
    # ecfp3 lives on the CHEMISTRY tab -- the per-tab reset is scoped, so
    # an ML-tab reset must NOT touch it; the Chemistry reset re-ticks it
    check(
        "2c. ML-tab reset leaves ecfp3 untouched (per-tab scoping)",
        bool(app._feat_family_vars["ecfp3"].get()) is False,
    )
    chem_index = None
    for tab_id in app._book.tabs():
        if str(app._book.tab(tab_id, "text")).strip() == "Chemistry":
            chem_index = app._book.index(tab_id)
            break
    if chem_index is not None:
        app._reset_tab(chem_index)
        app.update()
        check(
            "2d. Chemistry-tab reset re-ticks ecfp3 (the families walk)",
            bool(app._feat_family_vars["ecfp3"].get()) is True,
        )

    # -- 3. ownership: both names owned by the Data & ML tab ------------- #
    owners = app._tab_input_names()
    owned = owners.get(ml_index, set())
    check(
        "3. _ml_aggregate_combo + _perm_repeats owned by the ML tab",
        "_ml_aggregate_combo" in owned and "_perm_repeats" in owned,
        f"agg_in={'_ml_aggregate_combo' in owned} rep_in={'_perm_repeats' in owned}",
    )

    # -- 5. staleness across a new load ---------------------------------- #
    import pandas as pd

    app._ml_frame = pd.DataFrame(
        {
            "Ligand SMILES": ["CCO", "CCO", "CCN"],
            "EC50 (nM)": ["<50", 200.0, 1000.0],
        }
    )
    app._ml_aggregate_combo.set("median")
    orig_askyesno = playground.messagebox.askyesno
    playground.messagebox.askyesno = lambda *a, **k: True
    try:
        app._ml_prep_endpoint()
        ok = drain_jobs(app)
        check("5a. prep job completed", ok)
    finally:
        playground.messagebox.askyesno = orig_askyesno
    first_prep = app._ml_prep_frame
    check("5b. prep frame landed", first_prep is not None and len(first_prep) == 2)
    # a new dataset is adopted through the shared adoption body
    other = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    app._ml_adopt_frame(other, Path("/tmp/probe146_other.csv"))
    check(
        "5c. STALE: _ml_prep_frame survives a new dataset (Save writes the OLD frame)",
        app._ml_prep_frame is first_prep,
    )
    check(
        "5d. same for _ml_cleaned_frame after a clean",
        True,  # see 5e: run the clean path too
    )
    app._target_combo.set("a")
    app._ml_clean_frame()
    ok = drain_jobs(app)
    cleaned = app._ml_cleaned_frame
    newer = pd.DataFrame({"c": [9.0]})
    app._ml_adopt_frame(newer, Path("/tmp/probe146_newer.csv"))
    check(
        "5e. STALE: _ml_cleaned_frame survives a new dataset too (pre-existing twin)",
        ok and app._ml_cleaned_frame is cleaned,
    )

    # -- 6. prep error path (no SMILES/EC50 columns) --------------------- #
    seen.clear()
    for name in ("showerror", "showwarning", "showinfo"):
        setattr(playground.messagebox, name, spy(name, getattr(playground.messagebox, name)))
    app._ml_frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    app._ml_prep_endpoint()
    ok = drain_jobs(app)
    check(
        "6. no-column prep surfaces the job 'Calculation failed' dialog",
        ok and any(n == "showerror" and "Calculation failed" in str(a) for n, a in seen),
        f"seen={[(n, str(a)[:60]) for n, a in seen][:3]}",
    )

    # -- 7. permutation needs-run guard order ---------------------------- #
    seen.clear()
    app._ml_last_run = None
    app._ml_plot_permutation()
    check(
        "7. permutation fires 'No trained model' (needs-run first)",
        any(n == "showinfo" and "No trained model" in str(a) for n, a in seen),
        f"seen={[(n, str(a)[:50]) for n, a in seen][:2]}",
    )

finally:
    if app is not None:
        app.destroy()

print()
print("PROBE_A_RESULT:", "ALL_PASS" if not fails else f"FAILURES={fails}")
