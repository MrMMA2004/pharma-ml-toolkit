"""Round-167 probe: exercise every Response AI card headlessly."""
import sys
import time

sys.path.insert(0, ".")
import playground  # noqa: E402


def pump(app, timeout=180.0):
    deadline = time.monotonic() + timeout
    while app._busy and time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.01)
    assert not app._busy, "job never finished"


app = playground.PlaygroundApp(persist_session=False)
for _ in range(4):
    app.update_idletasks()
    app.update()
titles = [app._book.tab(i, "text").strip() for i in range(app._book.index("end"))]
assert "Response AI" in titles, f"tab missing: {titles}"
app._book.select(titles.index("Response AI"))
app.update()

# 1. visible network
app._rai_run_visible(); pump(app)
rows = app._rai_vis_tree.get_children()
assert len(rows) >= 4, f"visible rows: {len(rows)}"
print("VISIBLE:", app._rai_vis_note.cget("text")[:100])
print("  ranking:", [app._rai_vis_tree.item(r, "values")[0] for r in rows])

# 2. few-shot transfer
app._rai_run_fewshot(); pump(app)
print("FEWSHOT:", app._rai_fs_note.cget("text")[:90])
assert len(app._rai_fs_tree.get_children()) == 4

# 3. GP Bayesian search
app._rai_run_bayes(); pump(app)
print("BAYES:", app._rai_bayes_note.cget("text")[:100])
assert len(app._rai_bayes_tree.get_children()) == 12

# 4. dose-response fitting
app._rai_run_curvfit(); pump(app)
print("DOSE:", app._rai_dose_note.cget("text")[:100])
assert len(app._rai_dose_tree.get_children()) == 4

# 5. rank correlations (synthetic fallback)
app._rai_run_rankcorr(); pump(app)
print("RANK:", app._rai_rank_note.cget("text")[:100])

# 6. power spectrum (synthetic fallback)
app._rai_run_spectrum(); pump(app)
print("SPECTRUM:", app._rai_spec_note.cget("text")[:110])

# 7. hierarchical clustering (synthetic fallback)
app._rai_run_hclust(); pump(app)
print("HCLUST:", app._rai_hc_note.cget("text")[:100])

# now load the example via Foundations and re-run the data cards
app._found_load_example()
app.update()
app._rai_run_rankcorr(); pump(app)
print("RANK loaded:", app._rai_rank_note.cget("text")[:80])
app._rai_run_spectrum(); pump(app)
print("SPECTRUM loaded:", app._rai_spec_note.cget("text")[:80])
app._rai_run_hclust(); pump(app)
print("HCLUST loaded:", app._rai_hc_note.cget("text")[:80])

app.destroy()
print("PROBE RESPONSEAI 167: ALL CARDS OK")
