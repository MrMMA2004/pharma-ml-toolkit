"""Audit 144 probe B v4 (read-only auditor): reporter instrumentation for the
ML loader (removes the Tk tick-phase sampling race) + a >8 MB screen through
_bio_load_file (collapse-aware). Throwaway.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS: list[str] = []
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    log(RESULTS[-1])


def spy_dialogs(playground) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        def spy(title, message, _n=name, **_k):
            seen.append((_n, f"{title}: {message}"))
            log(f"DIALOG {_n}: {title}: {message}")

        setattr(playground.messagebox, name, spy)
    return seen


def wait_job(app, timeout: float) -> float:
    waited = 0.0
    while app._busy and waited < timeout:
        app.update()
        time.sleep(0.002)
        waited += 0.002
    return waited


def main() -> int:
    log("importing playground...")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import playground

    dialogs = spy_dialogs(playground)
    log("playground imported")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        big = tmp_path / "wide_library.csv"
        rows = ["smiles,value"] + [
            f"CCO(C)COc1ccccc1CCNc1ccc(O)cc1,{i}.5" for i in range(400_000)
        ]
        big.write_text("\n".join(rows) + "\n", encoding="utf-8")
        log(f"big csv written {big.stat().st_size / (1024 * 1024):.1f} MB")

        screen = tmp_path / "screen.csv"
        srows = ["cell_line,compound,ln_ic50"] + [
            f"line_{i % 600:03d},drug_{i % 50:03d},{(i % 900) / 100 + 0.25:.2f}"
            for i in range(420_000)
        ]
        screen.write_text("\n".join(srows) + "\n", encoding="utf-8")
        log(f"screen written {screen.stat().st_size / (1024 * 1024):.1f} MB")
        check("screen above threshold", screen.stat().st_size >= 8 * 1024 * 1024,
              f"{screen.stat().st_size / (1024 * 1024):.1f} MB")

        log("building app...")
        app = playground.PlaygroundApp(persist_session=False)
        try:
            # -- ML: instrument the reporter (tick-phase independent) -- #
            reports: list[tuple[int, object, str]] = []
            original = app._report_progress

            def spy_report(done, total=None, note=""):
                reports.append((done, total, note))
                original(done, total, note)

            app._report_progress = spy_report  # type: ignore[method-assign]
            app._ml_load_file(big)
            waited = wait_job(app, 120.0)
            check("ML load finishes", not app._busy, f"waited {waited:.1f}s")
            check("ML reports fired", len(reports) >= 5, f"{len(reports)} reports")
            totals = {t for _d, t, _n in reports}
            dones = [d for d, _t, _n in reports]
            check("ML totals are the estimate (>= rows)", all(
                t is None or t >= 400_000 for t in totals), str(totals))
            check("ML done reaches 400000", dones and dones[-1] == 400_000, str(dones[-3:]))
            check("ML adopted", app._ml_frame is not None and len(app._ml_frame) == 400_000)
            app._report_progress = original  # type: ignore[method-assign]

            # -- BIO: >8MB screen through load_response_table(report) -- #
            reports.clear()
            app._report_progress = spy_report  # type: ignore[method-assign]
            app._bio_load_file(screen)
            waited = wait_job(app, 240.0)
            check("BIO load finishes", not app._busy, f"waited {waited:.1f}s")
            check("BIO reports fired through load_response_table",
                  len(reports) >= 5, f"{len(reports)} reports")
            dones = [d for d, _t, _n in reports]
            check("BIO done reaches 420000", dones and dones[-1] == 420_000,
                  str(dones[-3:]))
            check("BIO screen adopted (collapsed by default checkbox)",
                  app._bio_screen is not None and len(app._bio_screen.frame) == 30_000,
                  f"rows={len(app._bio_screen.frame) if app._bio_screen else None}")
            check("BIO audit text mentions replicates",
                  "replicate" in str(app._bio_audit_text.get("1.0", "end")).lower())
            app._report_progress = original  # type: ignore[method-assign]
        finally:
            app.destroy()
            log("app destroyed")

        if dialogs:
            log(f"dialogs seen: {dialogs}")
    print("PROBE144B4_RESULTS")
    for line in RESULTS:
        print(line)
    return 1 if any(line.startswith("FAIL") for line in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
