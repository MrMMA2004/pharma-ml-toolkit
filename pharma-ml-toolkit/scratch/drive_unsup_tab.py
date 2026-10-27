"""Live drive of the Unsupervised tab (round 40 pre-test)."""
import importlib.util, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main() -> int:
    spec = importlib.util.spec_from_file_location("playground_live", ROOT / "playground.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    app = module.PlaygroundApp(persist_session=False)
    try:
        # 1) load the classification example through the ML tab loader
        example = ROOT / "examples" / "data" / "qsar_classification.csv"
        assert example.is_file(), example
        app._ml_load_path(example)
        assert app._ml_frame is not None and len(app._ml_frame) > 20
        info = app._unsup_info.cget("text")
        print("dataset info:", info)
        assert "numeric columns" in info

        # 2) select ALL numeric columns
        app._unsup_select_all()
        columns = app._unsup_selected_columns()
        print("selected columns:", columns)
        assert len(columns) >= 3

        # 3) run each method and wait for the job
        def run_and_wait(label):
            app._unsup_run()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if not app._busy:
                    break
                app.update_idletasks(); app.update()
                time.sleep(0.02)
            assert not app._busy, f"{label} job never finished"
            summary = app._unsup_summary.cget("text")
            print(f"--- {label} done", flush=True)
            return summary

        # PCA
        app._unsup_method.set(app._UNSUP_METHODS[0])
        summary = run_and_wait("PCA")
        assert "PCA(" in summary and "explained variance" in summary
        assert app._unsup_result is not None
        # scatter twin (PC scores, coloured by target)
        target = app._target_combo.get()
        app._unsup_colour.set(target)
        app._unsup_scatter()
        # export frame
        frame = app._unsup_result["record"].as_frame()
        assert frame.shape[1] >= 2 and frame.shape[0] == len(app._ml_frame)

        # K-Means auto
        app._unsup_method.set("K-Means (elbow + silhouette)")
        app._unsup_k.delete(0, "end"); app._unsup_k.insert(0, "auto")
        summary = run_and_wait("KMeans")
        assert "k-means" in summary and "silhouette" in summary
        # scatter with cluster labels
        app._unsup_colour.set("(none)")
        app._unsup_scatter()

        # GMM
        app._unsup_method.set("Gaussian mixture (AIC/BIC)")
        summary = run_and_wait("GMM")
        assert "GaussianMixture(" in summary

        # KDE (one column)
        app._unsup_method.set("KDE (one column)")
        summary = run_and_wait("KDE")
        assert "KDE(" in summary and "bandwidth" in summary

        # Isomap
        app._unsup_method.set("Isomap")
        summary = run_and_wait("Isomap")
        assert "isomap embedding" in summary
        app._unsup_scatter()

        # t-SNE with honest perplexity
        app._unsup_method.set("t-SNE")
        app._unsup_perplexity.delete(0, "end"); app._unsup_perplexity.insert(0, "10")
        summary = run_and_wait("t-SNE")
        assert "tsne embedding" in summary

        # Hexbin (two columns)
        app._unsup_method.set("Hexbin density (two columns)")
        summary = run_and_wait("Hexbin")
        assert "hexbin density" in summary.lower()

        # 4) guards: no dataset -> warning recorded, no columns -> warning
        app._unsup_clear_selection()
        app._unsup_run()
        app.update()
        print("guard: run with no selection kept busy =", app._busy)

        # 5) redraw + export buttons after a fresh run
        app._unsup_select_all()
        app._unsup_method.set(app._UNSUP_METHODS[0])
        run_and_wait("PCA again")
        app._unsup_plot()
        print("redraw ok; host children:", len(app._unsup_host.winfo_children()))
        assert app._unsup_host.winfo_children()

        print("ALL LIVE CHECKS PASSED")
        return 0
    finally:
        app.destroy()

if __name__ == "__main__":
    sys.exit(main())
