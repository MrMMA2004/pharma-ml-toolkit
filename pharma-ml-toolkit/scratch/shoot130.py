"""Screenshots + artifacts for the 100x-130 round: the Ctrl+F finder's
"Recently jumped" recency row -- three real jumps seed the memory, then
the dialog screenshot shows the chip row under the query (the newest
chip first, live titles only), and a second shot shows the finder
filtered with a chip still reachable.  The desktop app window carries
the Surgery-adjacent Generative tab so the round's desktop context is
visible; the web twin's Surgery sub-tab screenshots come from the
agent-browser E2E (shoot130-web).

Run: DISPLAY=:99 /home/z/.venv/bin/python scratch/shoot130.py
"""
import sys
import time

sys.path.insert(0, ".")
import playground

# keep any modal quiet in headless runs
playground.messagebox.showinfo = lambda *a, **k: None

app = playground.PlaygroundApp()
app.geometry("1280x860+10+10")


def pump(timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if not app._busy:
            return True
        time.sleep(0.02)
    return False


def shot(name: str) -> None:
    app.update()
    app.update_idletasks()
    for _ in range(24):
        app.update()
        time.sleep(0.01)
    from PIL import ImageGrab  # type: ignore

    img = ImageGrab.grab(xdisplay=":99")
    img.save(f"/home/z/my-project/pharma-ml-toolkit/download/{name}.png")
    print("shot", name)


try:
    pump()
    # seed the recency memory with real jumps (newest first): two cards
    # on other tabs, then the Generative tab's own two surgery cards --
    # exactly the cards the web twin mirrored this round.
    app._jump_to_card(1, "Affinity conversion")
    app._jump_to_card(0, "PharmaML Playground")
    live_cards = app._live_cards()
    for title in ("Ring topology moves", "Breed two molecules"):
        for card_title, _subtitle, _shell, index in live_cards:
            if card_title == title:
                app._jump_to_card(index, title)
                break
    pump()
    assert app._finder_recent[:2] == ["Breed two molecules", "Ring topology moves"], (
        app._finder_recent
    )

    # the dialog with the chip row visible
    app._find_card_dialog()
    import tkinter as tkmod

    dialogs = [
        child
        for child in app.winfo_children()
        if isinstance(child, tkmod.Toplevel) and child.title() == "Find a card"
    ]
    win = dialogs[0]
    shot("pg130-finder-recency-chips")

    # a filtered finder (the chips stay reachable above the rows)
    entry = next(child for child in win.winfo_children() if child.winfo_class() == "TEntry")
    entry.delete(0, "end")
    entry.insert(0, "nca")
    entry.event_generate("<KeyRelease>", keysym="n")
    pump(timeout=10.0)
    shot("pg130-finder-recency-filtered")
    win.destroy()

    # the Generative tab itself (the desktop surgery cards the web twin
    # mirrored this round) with the last litter drawn
    live_cards = app._live_cards()
    for card_title, _subtitle, _shell, index in live_cards:
        if card_title == "Breed two molecules":
            app._jump_to_card(index, card_title)
            break
    app._breed_a.delete(0, "end")
    app._breed_a.insert(0, "CC(=O)Oc1ccccc1C(=O)O")
    app._breed_b.delete(0, "end")
    app._breed_b.insert(0, "CC(C)Cc1ccc(O)cc1")
    app._breed_count.set("4")
    app._breed_seed.delete(0, "end")
    app._breed_seed.insert(0, "42")
    app._gen_breed()
    assert pump(), "breed job hung"
    app._gen_draw_breed()
    assert pump(), "litter grid job hung"
    shot("pg130-desktop-breed-card")

    print("SHOOT130_OK")
finally:
    app.destroy()
    pump(timeout=30.0)
