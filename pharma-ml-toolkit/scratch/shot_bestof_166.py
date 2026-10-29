"""Round-166 screenshots: the Best-of ML tab, populated headlessly (PIL)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import playground  # noqa: E402


def pump(app, timeout=120.0):
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
    out = Path(__file__).resolve().parents[1] / "scratch" / f"pg166-{name}.png"
    img.save(out)
    print("  shot", out.name)


app = playground.PlaygroundApp(persist_session=False)
for _ in range(4):
    app.update_idletasks()
    app.update()
titles = [app._book.tab(i, "text").strip() for i in range(app._book.index("end"))]
app._book.select(titles.index("Best-of ML"))
app.update()

app._bestof_run_search(); pump(app)
app._bestof_run_coverage(); pump(app)
app._bestof_run_dp(); pump(app)
app._bestof_run_recommender(); pump(app)
app._bestof_run_lsh(); pump(app)
app._bestof_run_adversarial(); pump(app)
app._bestof_run_timeseries(); pump(app)

app.geometry("1680x1150+0+0")
app.update_idletasks(); app.update(); time.sleep(0.4)
shot(app, "bestof-tab-top")

# scroll the page down to the demo cards row
page = app._book.nametowidget(app._book.select())
canvas = getattr(page, "_canvas", None)
if canvas is not None:
    canvas.yview_moveto(0.55)
    app.update(); time.sleep(0.3)
    shot(app, "bestof-tab-cards")

app._found_load_example()
app.update()
app._bestof_run_dp(); pump(app)
canvas = getattr(page, "_canvas", None)
if canvas is not None:
    canvas.yview_moveto(0.25)
    app.update(); time.sleep(0.3)
    shot(app, "bestof-tab-loaded")

app.destroy()
print("SHOT BESTOF 166 DONE")
