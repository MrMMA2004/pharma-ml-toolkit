"""Round-166 probe: exercise every Best-of ML card handler headlessly."""
import sys
import time

sys.path.insert(0, ".")
import playground  # noqa: E402


def pump_until_idle(app, timeout=120.0):
    deadline = time.monotonic() + timeout
    while app._busy and time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    assert not app._busy, "job never finished"


def run(app, handler_name, *args):
    getattr(app, handler_name)(*args)
    pump_until_idle(app)


app = playground.PlaygroundApp(persist_session=False)
for _ in range(4):
    app.update_idletasks()
    app.update()

# tab exists and is visitable
titles = [app._book.tab(i, "text").strip() for i in range(app._book.index("end"))]
assert any("Best-of ML" in t for t in titles), f"tab missing: {titles}"
app._book.select(titles.index("  Best-of ML  ") if "  Best-of ML  " in titles else next(i for i, t in enumerate(titles) if "Best-of ML" in t))
app.update()

# 1. search
run(app, "_bestof_run_search")
hits = app._bestof_tree.get_children()
assert len(hits) >= 1, "search produced no rows"
first = app._bestof_tree.item(hits[0], "values")
print("SEARCH:", first[0], "|", first[1], "| stars", first[2])

# 2. category browse
app._bestof_category.set("Vector Similarity Search (ANN)")
run(app, "_bestof_show_category")
print("CATEGORY rows:", len(app._bestof_tree.get_children()))
assert len(app._bestof_tree.get_children()) == 13

# 3. top 10
app._bestof_category.set("all categories")
run(app, "_bestof_show_top")
print("TOP10 rows:", len(app._bestof_tree.get_children()))

# 4. coverage audit
run(app, "_bestof_run_coverage")
print("COVERAGE rows:", len(app._bestof_coverage_tree.get_children()))
assert len(app._bestof_coverage_tree.get_children()) == 35

# 5. DP statistics (synthetic fallback — no dataset loaded)
run(app, "_bestof_run_dp")
print("DP note:", app._bestof_dp_note.cget("text").splitlines()[0][:90])
assert len(app._bestof_dp_tree.get_children()) >= 3

# 6. recommender
run(app, "_bestof_run_recommender")
print("CF note:", app._bestof_cf_note.cget("text").splitlines()[0][:110])

# 7. LSH sweep
run(app, "_bestof_run_lsh", )
print("LSH rows:", len(app._bestof_lsh_tree.get_children()))
print("LSH note:", app._bestof_lsh_note.cget("text")[:100])

# 8. adversarial audit (synthetic two-blob fallback)
run(app, "_bestof_run_adversarial")
print("ADV note:", app._bestof_adv_note.cget("text")[:110])
assert len(app._bestof_adv_tree.get_children()) == 4

# 9. time-series analysis (synthetic sine fallback)
run(app, "_bestof_run_timeseries")
print("TS note:", app._bestof_ts_note.cget("text")[:110])
assert len(app._bestof_ts_tree.get_children()) >= 6

# 10. now load the example via Foundations and re-run the data cards
app._found_load_example()
app.update()
run(app, "_bestof_run_dp")
print("DP after load:", app._bestof_dp_note.cget("text").splitlines()[0][:80])
run(app, "_bestof_run_timeseries")
print("TS after load:", app._bestof_ts_note.cget("text")[:80])
run(app, "_bestof_run_lsh")
print("LSH after load:", app._bestof_lsh_note.cget("text")[:80])
run(app, "_bestof_run_adversarial")
print("ADV after load:", app._bestof_adv_note.cget("text")[:80])

app.destroy()
print("PROBE BESTOF 166: ALL CARDS OK")
