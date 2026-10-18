import subprocess, sys, os, pathlib
# fresh HOME with NO prior session: self-test must not create one
home = pathlib.Path("/home/z/my-project/scratch/pg51/clean_home")
import shutil
shutil.rmtree(home, ignore_errors=True)
home.mkdir(parents=True)
env = dict(os.environ, DISPLAY=":91", HOME=str(home))
out = subprocess.run([sys.executable, "playground.py", "--self-test"],
                     cwd="/home/z/my-project/pharma-ml-toolkit", env=env,
                     capture_output=True, text=True)
print(out.stdout.strip().splitlines()[-1])
session = home / ".pharmaml" / "playground-session.json"
print("session file created:", session.exists())
assert not session.exists(), "self-test must not write a session file"
print("SELF-TEST CLEAN")
