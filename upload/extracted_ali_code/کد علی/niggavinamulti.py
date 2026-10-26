import os
import subprocess
import threading
from queue import Queue

# ---------------- CONFIG ----------------
CONF_PATH = "conf.txt"
LIGAND_ROOT = "ligands"
RESULTS_ROOT = "results"
MAX_THREADS = os.cpu_count() // 2 or 4  # reasonable default
# ----------------------------------------


def get_ligands(root):
    """Find ligands in ligands/[ligand_name]/[ligand_name].pdbqt"""
    ligands = {}
    for folder in os.listdir(root):
        subdir = os.path.join(root, folder)
        if os.path.isdir(subdir):
            lig = os.path.join(subdir, f"{folder}.pdbqt")
            if os.path.exists(lig):
                ligands[folder] = lig
    return ligands


def is_complete(result_dir, name):
    """Check if ligand docking is complete and valid."""
    out = os.path.join(result_dir, f"{name}_output.pdbqt")
    log = os.path.join(result_dir, f"{name}.log")
    return (
        os.path.exists(out) and os.path.exists(log)
        and os.path.getsize(out) > 500
        and os.path.getsize(log) > 200
    )


def cleanup(result_dir, name):
    """Remove half-done results."""
    for f in [
        os.path.join(result_dir, f"{name}_output.pdbqt"),
        os.path.join(result_dir, f"{name}.log")
    ]:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"🧹 Cleaned: {f}")
            except Exception as e:
                print(f"⚠️ Could not remove {f}: {e}")


def dock_ligand(name, path, lock):
    """Run vina on a single ligand (one thread)."""
    result_dir = os.path.join(RESULTS_ROOT, name)
    os.makedirs(result_dir, exist_ok=True)

    out = os.path.join(result_dir, f"{name}_output.pdbqt")
    log = os.path.join(result_dir, f"{name}.log")

    if is_complete(result_dir, name):
        with lock:
            print(f"⏭️  {name} already complete — skipping.")
        return

    cleanup(result_dir, name)


    with open(log, "w") as logfile:
        cmd = ["vina", "--config", CONF_PATH, "--ligand", path, "--out", out]
        subprocess.run(cmd, stdout=logfile, stderr=logfile)


    with lock:
        print(f"\n🚀 Starting docking: {name}")
        print("-----------------------------------------------------------")

    try:
        # Show vina progress live (each thread has its own process)
        subprocess.run(cmd, check=True)
        with lock:
            print(f"✅ {name}: Docking done.\n")
    except subprocess.CalledProcessError:
        with lock:
            print(f"❌ {name}: Docking failed.\n")

    with lock:
        print("-----------------------------------------------------------\n")


def worker(queue, lock):
    """Thread worker to process ligands from the queue."""
    while True:
        item = queue.get()
        if item is None:
            break
        name, path = item
        dock_ligand(name, path, lock)
        queue.task_done()


def main():
    if not os.path.exists(CONF_PATH):
        print(f"❌ {CONF_PATH} not found.")
        return

    ligands = get_ligands(LIGAND_ROOT)
    if not ligands:
        print("❌ No ligands found.")
        return

    os.makedirs(RESULTS_ROOT, exist_ok=True)

    print(f"Found {len(ligands)} ligands.")
    print(f"Running up to {MAX_THREADS} dockings in parallel.\n")

    queue = Queue()
    lock = threading.Lock()

    threads = []
    for _ in range(MAX_THREADS):
        t = threading.Thread(target=worker, args=(queue, lock), daemon=True)
        t.start()
        threads.append(t)

    # Add ligands to queue
    for name, path in ligands.items():
        queue.put((name, path))

    # Wait for completion
    queue.join()

    # Stop workers
    for _ in range(MAX_THREADS):
        queue.put(None)
    for t in threads:
        t.join()

    print("🏁 All docking tasks finished or up to date!")


if __name__ == "__main__":
    main()
