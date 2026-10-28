"""Round-44 artifact shooter: the SELFIES cards + the MCTS optimiser (100x-162/163)."""
import importlib.util
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playground_shoot", ROOT / "playground.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.PlaygroundApp(persist_session=False)
try:
    def shot(name: str) -> None:
        app.update()
        app.update_idletasks()
        for _ in range(24):
            app.update()
            time.sleep(0.01)
        from PIL import ImageGrab  # type: ignore

        img = ImageGrab.grab(xdisplay=":99")
        img.save(f"download/{name}.png")
        print("shot", name, flush=True)

    def wait_job(timeout: float = 300.0) -> bool:
        deadline = time.monotonic() + timeout
        while app._busy and time.monotonic() < deadline:
            app.update()
            time.sleep(0.02)
        while time.monotonic() < deadline:
            app.update()
            time.sleep(0.02)
            if not app._busy:
                return True
        return not app._busy

    # -- 1. Chemistry tab, the two SELFIES cards in view ------------------ #
    app._book.select(app._book.tabs()[1])  # Chemistry tab
    app._jump_to_card(1, "SELFIES codec")
    app.update()
    shot("pg162-chem-selfies-cards")

    # -- 2. the codec card: encode aspirin ------------------------------- #
    app._selfies_convert()
    if not wait_job():
        raise SystemExit("codec convert job timed out")
    print("encoded:", app._selfies_result.cget("text")[:80], flush=True)
    print("tokens:", app._selfies_tokens.cget("text")[:100], flush=True)
    shot("pg162-chem-selfies-convert")

    # -- 3. the decode direction ----------------------------------------- #
    encoded = app._selfies_result.cget("text")
    app._selfies_mode.set("SELFIES → SMILES")
    app._selfies_entry.delete(0, "end")
    app._selfies_entry.insert(0, encoded)
    app._selfies_convert()
    if not wait_job():
        raise SystemExit("codec decode job timed out")
    print("decoded:", app._selfies_result.cget("text"), flush=True)
    # the decoder returns the kekulé aromatic form -- chemically identical,
    # and canonical-equal under RDKit (the round-trip audit's own criterion)
    from rdkit import Chem  # type: ignore

    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(app._selfies_result.cget("text")))
    assert canonical == "CC(=O)Oc1ccccc1C(=O)O", (
        f"round-trip mismatch: {app._selfies_result.cget('text')!r} -> {canonical!r}"
    )

    # -- 4. the round-trip audit over the card's default molecule list ---- #
    app._selfies_audit_run()
    if not wait_job():
        raise SystemExit("audit job timed out")
    print("audit:", app._selfies_audit_result.cget("text").splitlines()[0], flush=True)
    shot("pg162-chem-selfies-audit")

    # -- 5. the robustness experiment + its figure ----------------------- #
    app._selfies_robustness_run()
    if not wait_job(timeout=600.0):
        raise SystemExit("robustness job timed out")
    print("robustness:", app._selfies_audit_result.cget("text").splitlines()[0], flush=True)
    shot("pg162-chem-selfies-robustness")

    # -- 6. the random-SELFIES generator --------------------------------- #
    app._selfies_random_run()
    if not wait_job():
        raise SystemExit("random job timed out")
    print("random:", app._selfies_audit_result.cget("text").splitlines()[1][:90], flush=True)

    # -- 7. the Generative tab with the MCTS optimiser selected ---------- #
    app._book.select(app._book.tabs()[8])  # Generative tab
    app._method_combo.set("monte carlo tree search (MCTS)")
    app._on_method_change()
    app.update()
    shot("pg162-gen-mcts-params")

    # -- 8. a real MCTS run: QED, 6 cycles, light prior ------------------ #
    app._mcts_iterations.set("6")
    app._mcts_simulations.set("2")
    app._mcts_prior_epochs.set("4")
    app._run_evolution()
    if not wait_job(timeout=600.0):
        raise SystemExit("MCTS evolution job timed out")
    print("MCTS best:", app._ga_summary.cget("text")[:100], flush=True)
    print("MCTS stats:\n", app._ga_stats_text.get("1.0", "end").strip()[:400], flush=True)
    shot("pg162-gen-mcts-results")

    print("ALL SHOTS DONE", flush=True)
finally:
    app.destroy()
