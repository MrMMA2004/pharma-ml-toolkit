"""Round-167 screenshots: the Response AI tab, populated headlessly."""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402


def pump(app, timeout=180.0):
    deadline = time.monotonic() + timeout
    while app._busy and time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)


def shot(app, name: str) -> None:
    app.update()
    app.update_idletasks()
    for _ in range(30):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    out = Path(__file__).resolve().parents[1] / "scratch" / f"pg167-{name}.png"
    img.save(out)
    print("  shot", out.name)


app = playground.PlaygroundApp(persist_session=False)
for _ in range(4):
    app.update_idletasks()
    app.update()
titles = [app._book.tab(i, "text").strip() for i in range(app._book.index("end"))]
app._book.select(titles.index("Response AI"))
app.update()

app._rai_run_visible(); pump(app)
app._rai_run_fewshot(); pump(app)
app._rai_run_bayes(); pump(app)
app.geometry("1680x1150+0+0")
app.update_idletasks(); app.update(); time.sleep(0.4)
shot(app, "responseai-top")

page = app._book.nametowidget(app._book.select())
canvas = getattr(page, "_canvas", None)
if canvas is not None:
    canvas.yview_moveto(0.45)
    app.update(); time.sleep(0.3)
    app._rai_run_curvfit(); pump(app)
    app._rai_run_rankcorr(); pump(app)
    shot(app, "responseai-cards")

app.destroy()
print("SHOT RESPONSEAI 167 DONE")
