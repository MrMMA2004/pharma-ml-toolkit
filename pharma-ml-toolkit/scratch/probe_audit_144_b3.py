"""Audit 144 probe B v3 (read-only auditor): the four reporting loaders'
determinate bars end-to-end (ML, DL, feature factory, Cell Lines screen),
with files genuinely above the 8 MB threshold. Throwaway.
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


def wait_job(app, timeout: float) -> tuple[bool, float, bool, float, str]:
    determinate_seen = False
    fraction_seen = 0.0
    label_seen = ""
    waited = 0.0
    while app._busy and waited < timeout:
        app.update()
        time.sleep(0.002)
        waited += 0.002
        if str(app._progress.cget("mode")) == "determinate":
            determinate_seen = True
            maximum = int(app._progress.cget("maximum"))
            value = int(app._progress["value"])
            if maximum > 0:
                fraction_seen = max(fraction_seen, value / maximum)
            label_seen = str(app._job_label.cget("text"))
    return (not app._busy), waited, determinate_seen, fraction_seen, label_seen


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
        check("big csv above threshold", big.stat().st_size >= 8 * 1024 * 1024,
              f"{big.stat().st_size / (1024 * 1024):.1f} MB")

        screen = tmp_path / "screen.csv"
        srows = ["cell_line,compound,ln_ic50"] + [
            f"line_{i % 500:03d},drug_{i % 40:03d},{(i % 900) / 100 + 0.25:.2f}"
            for i in range(350_000)
        ]
        screen.write_text("\n".join(srows) + "\n", encoding="utf-8")
        log(f"screen written {screen.stat().st_size / (1024 * 1024):.1f} MB")
        check("screen above threshold", screen.stat().st_size >= 8 * 1024 * 1024,
              f"{screen.stat().st_size / (1024 * 1024):.1f} MB")

        log("building app...")
        app = playground.PlaygroundApp(persist_session=False)
        try:
            for label, starter, adopted in (
                ("ML", app._ml_load_file,
                 lambda: app._ml_frame is not None and len(app._ml_frame) == 400_000),
                ("DL", app._dl_load_file,
                 lambda: app._dl_frame is not None and len(app._dl_frame) == 400_000),
                ("FEAT", app._feat_load_file,
                 lambda: app._feat_csv_frame is not None
                 and len(app._feat_csv_frame) == 400_000),
                ("BIO", app._bio_load_file,
                 lambda: app._bio_screen is not None and len(app._bio_screen.frame) == 350_000),
            ):
                path = screen if label == "BIO" else big
                starter(path)
                done, waited, det, frac, seen_label = wait_job(app, 120.0)
                check(f"{label} load finishes", done, f"waited {waited:.1f}s")
                check(f"{label} determinate bar observed", det)
                check(f"{label} bar progressed past 10%", frac > 0.10, f"{frac:.2f}")
                check(f"{label} label carried rows/total", "/" in seen_label, seen_label[:90])
                check(f"{label} adopted", adopted())
                check(f"{label} bar back to indeterminate",
                      str(app._progress.cget("mode")) == "indeterminate")
        finally:
            app.destroy()
            log("app destroyed")

        if dialogs:
            log(f"dialogs seen: {dialogs}")
    print("PROBE144B3_RESULTS")
    for line in RESULTS:
        print(line)
    return 1 if any(line.startswith("FAIL") for line in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
