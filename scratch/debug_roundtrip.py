#!/usr/bin/env python
"""Debug the cancellable-job roundtrip under Xvfb (round 205)."""
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLKIT = Path("/home/z/my-project/pharma-ml-toolkit")
sys.path.insert(0, str(TOOLKIT))

# Xvfb :120
for num in range(120, 130):
    d = f":{num}"
    proc = subprocess.Popen(["Xvfb", d, "-screen", "0", "1280x800x24"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if Path(f"/tmp/.X11-unix/X{num}").exists():
            break
        time.sleep(0.2)
    else:
        proc.terminate()
        continue
    os.environ["DISPLAY"] = d
    print(f"[env] DISPLAY={d}")
    break

import playground as pg  # noqa: E402
from pharmaml.exceptions import JobCancelled  # noqa: E402

for name in ("showwarning", "showerror", "showinfo"):
    setattr(pg.messagebox, name, lambda *a, **k: print(f"[dialog] {a}", flush=True))

app = pg.PlaygroundApp(persist_session=False)
for _ in range(4):
    app.update_idletasks()
    app.update()

def work(notify):
    print(f"[work] start, notify={notify}", flush=True)
    for index in range(10):
        time.sleep(0.02)
        keep = notify(index, 10, "tick")
        print(f"[work] tick {index} keep={keep}", flush=True)
        if keep is False:
            raise JobCancelled("stopped mid-tick")
    return "all done"

app._start_job("tick loop", work, None)
print(f"[main] right after start: busy={app._busy} cancellable={app._job_cancellable}", flush=True)
for i in range(3):
    app.update()
    time.sleep(0.02)
print(f"[main] after warm-up: busy={app._busy} cancellable={app._job_cancellable}", flush=True)
print(f"[main] label={app._job_label.cget('text')!r}", flush=True)
app._request_cancel()
deadline = time.time() + 30
while time.time() < deadline:
    app.update()
    if not app._busy:
        break
    time.sleep(0.02)
print(f"[main] final label={app._job_label.cget('text')!r} busy={app._busy}", flush=True)
app.destroy()
