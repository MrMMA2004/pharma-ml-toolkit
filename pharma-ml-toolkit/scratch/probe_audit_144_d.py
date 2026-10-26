"""Task 144-a probe D -- the Cell Lines screen loader with a genuinely
>= 9 MB screen (the predecessor's b3 used a 7.7 MB file that fell under
the 8 MB gate, and expected the UNcollapsed row count while the collapse
checkbox defaults on).  Re-verifies: determinate bar, rows/total label,
adoption with the collapsed count, error dialogs, bar reset.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg", force=True)
import playground  # noqa: E402

RESULTS: list[str] = []
T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    log(RESULTS[-1])


def spy_dialogs() -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for name in ("showinfo", "showerror", "showwarning"):
        def spy(title, message, _n=name, **_k):
            seen.append((_n, f"{title}: {message}"))
            log(f"DIALOG {_n}: {title}: {message}")

        setattr(playground.messagebox, name, spy)
    return seen


def wait_job(app, timeout: float):
    determinate = False
    label_seen = ""
    fraction = 0.0
    waited = 0.0
    while app._busy and waited < timeout:
        app.update()
        time.sleep(0.002)
        waited += 0.002
        if str(app._progress.cget("mode")) == "determinate":
            determinate = True
            maximum = int(app._progress.cget("maximum"))
            value = int(app._progress["value"])
            if maximum > 0:
                fraction = max(fraction, value / maximum)
            label_seen = str(app._job_label.cget("text"))
    return (not app._busy), waited, determinate, fraction, label_seen


def main() -> int:
    dialogs = spy_dialogs()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        screen = tmp_path / "screen.csv"
        rows = ["cell_line,compound,ln_ic50"] + [
            f"line_{i % 500:03d},drug_{i % 40:03d},{(i % 900) / 100 + 0.25:.2f}"
            for i in range(450_000)
        ]
        screen.write_text("\n".join(rows) + "\n", encoding="utf-8")
        size_mb = screen.stat().st_size / (1024 * 1024)
        check("screen above the 8 MB gate", size_mb >= 8.0, f"{size_mb:.1f} MB")

        app = playground.PlaygroundApp(persist_session=False)
        try:
            check("collapse checkbox default is ON", bool(app._bio_collapse_var.get()))
            app._bio_load_file(screen)
            done, waited, det, frac, label = wait_job(app, 180.0)
            check("screen load finishes", done, f"waited {waited:.1f}s")
            check("determinate bar observed", det)
            check("bar progressed past 10%", frac > 0.10, f"{frac:.2f}")
            check("label carried rows/total", "/" in label, label[:90])
            check("screen adopted (collapsed 500x40=20000 rows)",
                  app._bio_screen is not None and len(app._bio_screen.frame) == 20_000,
                  f"rows={len(app._bio_screen.frame) if app._bio_screen is not None else None}")
            check("audit text mentions the duplicate pairs",
                  "dup" in app._bio_audit_text.get("1.0", "end").lower())
            check("bar back to indeterminate",
                  str(app._progress.cget("mode")) == "indeterminate")
            check("no error dialogs", not dialogs, str(dialogs[:1]))

            # and the un-collapsed twin: flip the checkbox off, reload
            app._bio_collapse_var.set(False)
            app._bio_load_file(screen)
            done, waited, det, frac, label = wait_job(app, 180.0)
            check("un-collapsed reload finishes", done, f"waited {waited:.1f}s")
            check("un-collapsed screen adopted at full 450k rows",
                  app._bio_screen is not None and len(app._bio_screen.frame) == 450_000,
                  f"rows={len(app._bio_screen.frame) if app._bio_screen is not None else None}")
            check("no error dialogs after both loads", not dialogs, str(dialogs[:1]))
        finally:
            app.destroy()
            log("app destroyed")
    print("PROBE_D_RESULTS")
    for line in RESULTS:
        print(line)
    return 1 if any(line.startswith("FAIL") for line in RESULTS) else 0


if __name__ == "__main__":
    sys.exit(main())
