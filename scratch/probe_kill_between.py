"""Probe v5: exact pytest shape outside pytest.

1. Xvfb :98 up -> app1 (copy A) -> destroy + gc   [check X socket fds]
2. Xvfb :98 killed (the class-teardown moment)
3. Xvfb :97 up, DISPLAY=:97 -> app2 (copy B)      [the death step]
"""
import gc
import importlib.util
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/z/my-project/pharma-ml-toolkit")
PLAYGROUND_PATH = "/home/z/my-project/pharma-ml-toolkit/playground.py"


def note(msg):
    with open("/tmp/probe5_evidence.log", "a") as fh:
        fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def unix_socket_inodes() -> dict[str, str]:
    """inode -> unix path, from /proc/net/unix."""
    table: dict[str, str] = {}
    try:
        with open("/proc/net/unix") as fh:
            next(fh)  # header
            for line in fh:
                parts = line.split()
                if len(parts) >= 8:
                    table[parts[6]] = parts[7] if len(parts) > 7 else ""
    except OSError:
        pass
    return table


def x_fds() -> str:
    inodes = unix_socket_inodes()
    found = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            link = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            inode = link[8:-1]
            path = inodes.get(inode, "")
            if path.startswith("/tmp/.X11-unix/X"):
                found.append(f"fd{fd}->{path}")
    return ", ".join(found) if found else "none"


def load_copy(name: str):
    spec = importlib.util.spec_from_file_location(name, PLAYGROUND_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def quiet(mod) -> None:
    for nm in ("showerror", "showwarning", "showinfo"):
        setattr(mod.messagebox, nm, lambda *a, **k: None)


def start_xvfb(display: str):
    proc = subprocess.Popen(["Xvfb", display, "-screen", "0", "1280x800x24"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sock = f"/tmp/.X11-unix/X{display[1:]}"
    deadline = time.time() + 10
    while time.time() < deadline and not os.path.exists(sock):
        time.sleep(0.1)
    os.environ["DISPLAY"] = display
    return proc


copy_a = load_copy("playground_safety_under_test")
quiet(copy_a)
xvfb98 = start_xvfb(":98")

note("app1 build starting; X fds=" + x_fds())
app1 = copy_a.PlaygroundApp(persist_session=False)
note("app1 built; X fds=" + x_fds())
app1.destroy()
del app1
gc.collect()
note("app1 destroyed+gc; X fds=" + x_fds())

# -- the class-teardown moment: kill :98 while the connection may linger -- #
xvfb98.terminate()
try:
    xvfb98.wait(timeout=5)
except Exception:  # noqa: BLE001
    xvfb98.kill()
time.sleep(0.5)
note("Xvfb :98 killed; X fds=" + x_fds())

# -- test 2's setup: fresh :97 + app2 -- #
copy_b = load_copy("playground_wf_under_test")
quiet(copy_b)
xvfb97 = start_xvfb(":97")
note("Xvfb :97 up; X fds=" + x_fds())
note("app2 build starting (the death step)")
app2 = copy_b.PlaygroundApp(persist_session=False)
note("app2 built OK -- probe5 PASSED; X fds=" + x_fds())
app2.destroy()
del app2
gc.collect()
xvfb97.terminate()
note("probe5 end")
