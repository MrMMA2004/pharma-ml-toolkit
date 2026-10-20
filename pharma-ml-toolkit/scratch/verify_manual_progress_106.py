"""100x-106 end-to-end verifier: manual hyperparameters + progress/ETA.

Drives the REAL app headlessly (Xvfb): loads the example dataset, pins
ridge's alpha by hand, runs Train & evaluate through the worker-thread
machinery, then a RandomSearch and a Compare-models job -- asserting
the trained estimator carries the pinned value, the status bar went
determinate with a live ETA label, and everything reset to idle after.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

import playground  # noqa: E402


def pump(app: playground.PlaygroundApp, seconds: float) -> None:
    """Pump the Tk event loop so after() pollers and renders run."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.update_idletasks()
        app.update()
        time.sleep(0.02)


def wait_for_idle(app: playground.PlaygroundApp, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while app._busy and time.monotonic() < deadline:
        pump(app, 0.1)
    assert not app._busy, "job did not finish within the timeout"
    pump(app, 0.3)  # let the render + final poll cycles land


app = playground.PlaygroundApp(persist_session=False)
pump(app, 0.5)

# -- 1. manual hyperparameters: load, pick ridge, pin alpha -------- #
app._ml_load_example()
assert app._model_combo.get() == "auto"
app._model_combo.set("ridge")
app._ml_refresh_hyperparam_panel()
entries = app._ml_param_entries
assert list(entries) == ["alpha"]
entries["alpha"].insert(0, "0.5")
app._ml_sync_params_var()
assert json.loads(app._ml_params_var.get()) == {"ridge": {"alpha": "0.5"}}

# -- 2. Train & evaluate with the pinned value ---------------------- #
app._ml_train()
wait_for_idle(app)
trained = app._ml_last_run[0]
fitted_alpha = trained.model.named_steps["reg"].alpha
print(f"trained ridge alpha = {fitted_alpha} (pinned 0.5)")
assert abs(fitted_alpha - 0.5) < 1e-12, "the pinned alpha reached the fit"
assert trained.best_params == {"alpha": 0.5}
label = app._job_label.cget("text")
assert label == "idle", f"status reset after train: {label!r}"
assert str(app._progress.cget("mode")) == "indeterminate", "bar reset"

# -- 3. RandomSearch with progress + ETA ---------------------------- #
app._tune_trials.set("5")
app._ml_tune()
seen_determinate = False
seen_eta = False
while app._busy:
    pump(app, 0.05)
    label = app._job_label.cget("text")
    if str(app._progress.cget("mode")) == "determinate":
        seen_determinate = True
    if " left" in label and "/" in label:
        seen_eta = True
wait_for_idle(app)
print("tune label snapshots: determinate", seen_determinate, "· ETA", seen_eta)
assert seen_determinate and seen_eta, "progress bar + ETA never appeared"
results = app._tune_last[0]
print(f"search finished: {len(results.trials)} trials, best "
      f"{results.best_score:.4f}")
assert len(results.trials) == 5
assert app._job_label.cget("text") == "idle"

# -- 4. Compare models with per-model progress ----------------------- #
app._ml_compare()
seen_notes = []
while app._busy:
    pump(app, 0.05)
    label = app._job_label.cget("text")
    if "/" in label:
        seen_notes.append(label)
wait_for_idle(app)
assert seen_notes, "compare progress never appeared"
print(f"compare progress samples: {seen_notes[:2]} ... {seen_notes[-1:]}")
assert app._job_label.cget("text") == "idle"

# -- 5. per-model JSON survives a model switch ----------------------- #
app._model_combo.set("elastic_net")
app._ml_refresh_hyperparam_panel()
assert json.loads(app._ml_params_var.get()) == {"ridge": {"alpha": "0.5"}}, (
    "switching models must NOT drop ridge's saved values"
)
app._model_combo.set("ridge")
app._ml_refresh_hyperparam_panel()
assert app._ml_param_entries["alpha"].get() == "0.5", (
    "switching back restores ridge's pinned value"
)

app.destroy()
print("VERIFY-106: manual hyperparameters + progress/ETA end-to-end OK")
