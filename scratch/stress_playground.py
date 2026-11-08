#!/usr/bin/env python
"""scratch/stress_playground.py — the round-205 heavy-work verification.

The owner's report was "Playground.py collapses during heavy work" --
the OOM story: a heavy batch outruns the RAM, the kernel kills the
biggest process (the Tk app), the window vanishes with every loaded
input.  This driver proves the fix end to end on the REAL app under
Xvfb, driving the REAL card flows through the REAL worker-thread path:

  Phase A   heavy factory job (237 molecules x 20 families) with a
            mid-job Stop click -- must land "cancelled", window alive
  Phase A2  the SAME heavy job re-run without stopping -- must complete
            (a stop never poisons the next job)
  Phase B   the memory watchdog: a fake RSS over the ceiling trips the
            same stop channel mid-job
  Phase C   the pre-flight guard: a fake MemAvailable refuses the batch
            with its own arithmetic, no job starts
  Phase D   the ML n_jobs memory cap maths
  Phase E   the recalculation + xgboost regression phases (heavy paths
            still complete after all the new machinery)

Exit codes: 0 = every phase verified; non-zero = a phase failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
sys.path.insert(0, str(TOOLKIT))


def _start_xvfb() -> str:
    for num in range(113, 129):
        display = f":{num}"
        proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1600x900x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            if Path(f"/tmp/.X11-unix/X{num}").exists():
                return display
            time.sleep(0.2)
        proc.terminate()
    raise RuntimeError("Xvfb did not come up")


os.environ["DISPLAY"] = _start_xvfb()
print(f"[env] DISPLAY={os.environ['DISPLAY']}", flush=True)

import playground as pg  # noqa: E402

GATES: list[str] = []


def _gate(title: str, message: str, *_a: object, **_k: object) -> None:
    GATES.append(f"{title}: {message}")
    print(f"[gate] {title}: {message}", flush=True)


for _name in ("showwarning", "showerror", "showinfo"):
    setattr(pg.messagebox, _name, _gate)


def rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def pump(app: object, phase: str, deadline_s: float) -> str:
    """Pump the Tk loop while the job runs; return the final job label."""
    start = time.monotonic()
    while True:
        try:
            app.update()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 — the window itself died
            print(f"[{phase}] PUMP-ERROR {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            return "PUMP-ERROR"
        if not app._busy:  # type: ignore[attr-defined]
            try:
                return str(app._job_label.cget("text"))  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return "LABEL-GONE"
        if time.monotonic() - start > deadline_s:
            print(f"[{phase}] HUNG after {deadline_s:.0f}s — worker stacks:", flush=True)
            frames = sys._current_frames()
            for thread in threading.enumerate():
                if thread.name == "playground-job":
                    frame = frames.get(thread.ident)
                    if frame is not None:
                        traceback.print_stack(frame)
            return "HUNG"
        time.sleep(0.02)


def load_corpus() -> list[str]:
    import csv as _csv

    with open(
        TOOLKIT / "examples" / "data" / "screening_library.csv",
        newline="",
        encoding="utf-8",
    ) as handle:
        corpus = [row["SMILES"] for row in _csv.DictReader(handle) if row.get("SMILES")]
    return corpus * 3


def arm_factory(app: object, corpus: list[str]) -> None:
    app._feat_pasted_text.delete("1.0", "end")
    app._feat_pasted_text.insert("1.0", "\n".join(corpus))
    for family, _label in pg._FEATURE_FAMILY_LABELS:
        app._feat_family_vars[family].set(True)
    app._feat_ranker_combo.set("Variance (no target needed)")
    app._feat_njobs_spin.set("4")


def main() -> int:
    failures: list[str] = []
    app = pg.PlaygroundApp(persist_session=False)
    for _ in range(4):
        app.update_idletasks()
        app.update()
    corpus = load_corpus()
    print(f"[corpus] {len(corpus)} molecules (screening_library x3)", flush=True)

    # -- Phase A: heavy job + mid-job Stop ------------------------------ #
    arm_factory(app, corpus)
    t0 = time.monotonic()
    app._feat_compute()
    assert app._busy, "A: the job started"
    stop_disabled = bool(app._stop_button.instate(["disabled"]))
    assert not stop_disabled, "A: Stop enabled while busy"
    time.sleep(2.0)
    for _ in range(6):
        app.update()
        time.sleep(0.1)
    app._request_cancel()
    for _ in range(6):
        app.update()
        time.sleep(0.1)
    label = pump(app, "A-stop", 300)
    print(f"[A-stop] label={label!r} elapsed={time.monotonic() - t0:.1f}s", flush=True)
    if "cancelled" not in label:
        failures.append(f"A-stop: label {label!r} lacks 'cancelled'")
    if app._busy:
        failures.append("A-stop: app still busy")
    assert bool(app._stop_button.instate(["disabled"])), "A: Stop disabled when idle"
    assert not app._cancel_requested, "A: stop flag reset for the next job"

    # -- Phase A2: the same heavy job, NOT stopped ---------------------- #
    arm_factory(app, corpus)
    t0 = time.monotonic()
    app._feat_compute()
    label = pump(app, "A2-complete", 1800)
    frame = getattr(app, "_feat_frame", None)
    print(
        f"[A2-complete] label={label!r} elapsed={time.monotonic() - t0:.1f}s "
        f"frame={None if frame is None else frame.shape}",
        flush=True,
    )
    if "done" not in label:
        failures.append(f"A2: label {label!r} lacks 'done'")
    if frame is None or frame.shape[0] < 200:
        failures.append(f"A2: frame missing/small: {None if frame is None else frame.shape}")

    # -- Phase B: the memory watchdog trip ------------------------------ #
    arm_factory(app, corpus)
    real_rss, real_meminfo = pg._rss_mb, pg._meminfo_mb
    gates_before_b = len(GATES)
    app._feat_compute()
    ceiling = app._job_mem_floor_mb + app._job_mem_budget_mb + 64.0
    pg._rss_mb = lambda: ceiling  # a fake RSS over the job's ceiling
    pg._meminfo_mb = lambda field: 4096.0 if field == "MemAvailable" else real_meminfo(field)
    # capture the flag DURING the ticks: _poll_jobs resets it the moment
    # the (possibly cache-fast) job completes, and the durable evidence
    # is the label + the stop dialog's reason text anyway.
    tripped = False
    reason = ""
    for _ in range(20):  # ~2-3 watchdog polls
        app.update()
        time.sleep(0.16)
        if app._cancel_requested:
            tripped = True
            reason = app._cancel_reason
            break
    pg._rss_mb, pg._meminfo_mb = real_rss, real_meminfo
    label = pump(app, "B-watchdog", 300)
    dialog_evidence = any("memory ceiling" in gate for gate in GATES[gates_before_b:])
    print(
        f"[B-watchdog] tripped={tripped} reason={reason!r} "
        f"dialog={dialog_evidence} label={label!r}",
        flush=True,
    )
    if not (tripped or dialog_evidence):
        failures.append("B: the watchdog never tripped")
    if reason != "memory ceiling" and not dialog_evidence:
        failures.append(f"B: reason {reason!r} is not the memory ceiling")
    if "cancelled" not in label:
        failures.append(f"B: label {label!r} lacks 'cancelled'")

    # -- Phase C: the pre-flight guard refuses -------------------------- #
    arm_factory(app, corpus)
    real_meminfo = pg._meminfo_mb
    pg._meminfo_mb = lambda field: 200.0 if field == "MemAvailable" else real_meminfo(field)
    gates_before = len(GATES)
    app._feat_compute()
    pg._meminfo_mb = real_meminfo
    for _ in range(6):
        app.update()
        time.sleep(0.05)
    refused = any("Batch too large" in gate for gate in GATES[gates_before:])
    print(f"[C-guard] refused={refused} busy={app._busy}", flush=True)
    if not refused:
        failures.append("C: the oversized batch was not refused")
    if app._busy:
        failures.append("C: a job started despite the guard")

    # -- Phase D: the ML n_jobs memory cap ------------------------------ #
    real_meminfo = pg._meminfo_mb
    pg._meminfo_mb = lambda field: 1600.0 if field == "MemAvailable" else real_meminfo(field)
    capped, note = app._effective_ml_n_jobs(8)
    uncapped, note2 = app._effective_ml_n_jobs(2)
    pg._meminfo_mb = real_meminfo
    print(f"[D-cap] capped={capped}{note!r} uncapped={uncapped}{note2!r}", flush=True)
    if capped != 2 or "memory cap" not in note:
        failures.append(f"D: cap maths wrong: {capped} {note!r}")
    if uncapped != 2 or note2 != "":
        failures.append(f"D: small requests must pass untouched: {uncapped} {note2!r}")

    # -- Phase E: recalc + xgboost regressions -------------------------- #
    t0 = time.monotonic()
    app._recl_load_example()
    app._recl_smiles_text.delete("1.0", "end")
    app._recl_smiles_text.insert("1.0", "\n".join(corpus))
    app._recl_njobs_spin.set("4")
    app._recl_run()
    label = pump(app, "E-recalc", 900)
    print(f"[E-recalc] label={label!r} elapsed={time.monotonic() - t0:.1f}s", flush=True)
    if "done" not in label:
        failures.append(f"E-recalc: label {label!r} lacks 'done'")

    t0 = time.monotonic()
    app._ml_load_example()
    app._model_combo.set("xgboost")
    app._ml_njobs_spin.set("2")
    app._ml_train()
    label = pump(app, "E-xgboost", 600)
    print(f"[E-xgboost] label={label!r} elapsed={time.monotonic() - t0:.1f}s", flush=True)
    if "done" not in label:
        failures.append(f"E-xgboost: label {label!r} lacks 'done'")

    app.update()
    app.destroy()
    if failures:
        print(f"STRESS FAIL — {len(failures)} failure(s):", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        return 1
    print("STRESS PASS — heavy work verified: stop, watchdog, guard, cap, regressions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
