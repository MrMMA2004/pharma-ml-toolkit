"""Screenshot the playground's new cards via ffmpeg x11grab (scrolled)."""
import subprocess, sys, time, os
sys.path.insert(0, ".")
import playground

app = playground.PlaygroundApp()
tab = int(sys.argv[1])
out = sys.argv[2]
fraction = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
app._book.select(tab)
app.update()
page = app._book.nametowidget(app._book.tabs()[tab])
page._canvas.yview_moveto(fraction)
for _ in range(60):
    app.update()
    time.sleep(0.02)
subprocess.run(["ffmpeg", "-y", "-f", "x11grab", "-video_size", "1600x1000",
                "-i", os.environ.get("DISPLAY", ":99"), "-frames:v", "1", out],
               capture_output=True)
print("saved", out, os.path.getsize(out) if os.path.exists(out) else 0)
app.destroy()
