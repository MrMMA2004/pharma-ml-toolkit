"""Round-40 artifact shooter: the Unsupervised tab's own figures."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_shoot", ROOT / "playground.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.PlaygroundApp(persist_session=False)
try:
    app._ml_load_path(ROOT / "examples" / "data" / "qsar_classification.csv")
    app._unsup_select_all()
    target = app._target_combo.get()

    def run_and_wait():
        app._unsup_run()
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and app._busy:
            app.update_idletasks(); app.update(); time.sleep(0.02)
        app.update()

    def save(name):
        from pharmaml.viz import save_figure
        save_figure(app._last_figure, ROOT / "download" / name)
        print("saved", name, flush=True)

    # 1) PCA variance ladder
    app._unsup_method.set("PCA (variance ladder)")
    run_and_wait(); save("pg157-unsup-pca-variance.png")
    # 2) Isomap embedding coloured by the target class
    app._unsup_method.set("Isomap")
    app._unsup_colour.set(target)
    run_and_wait(); save("pg157-unsup-isomap-embedding.png")
    # 3) k-means diagnostics (auto k)
    app._unsup_method.set("K-Means (elbow + silhouette)")
    app._unsup_colour.set("(none)")
    run_and_wait(); save("pg157-unsup-kmeans-diagnostics.png")
    # 4) GMM AIC/BIC selection
    app._unsup_method.set("Gaussian mixture (AIC/BIC)")
    run_and_wait(); save("pg157-unsup-gmm-selection.png")
    # 5) KDE density with the hist underlay
    app._unsup_method.set("KDE (one column)")
    run_and_wait(); save("pg157-unsup-kde-density.png")
    # 6) hexbin density over the first two descriptors
    app._unsup_method.set("Hexbin density (two columns)")
    run_and_wait(); save("pg157-unsup-hexbin.png")
    print("ALL SHOTS DONE", flush=True)
finally:
    app.destroy()
