"""r342 (part15 roadmap #4) — the backend performance lanes, sandbox venv.

Measured honestly in THIS environment (Linux sandbox, /home/z/.venv,
pharmaml 1.2.240, the byte-frozen r334 release tree): backend startup
(process spawn -> PHARMAML-READY handshake), /api/v1/health latency
(first + warm), the idle resource snapshot via the r324 read-only
instrument (GET /api/v1/system/resources, sampled twice ~10 s apart
for the idle-drift observation), graceful shutdown (SIGTERM -> exit).
Evidence lands in scratch/p15_r342_backend_perf.txt.

The Windows-box matrix rows (desktop startup, time-to-first-UI, VM
numbers) stay NOT-MEASURED by rule in PERFORMANCE_BASELINES.md.
"""
from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request

VENV_PY = "/home/z/.venv/bin/python"
DATA_DIR = "/tmp/r342-perf-data"
EVIDENCE = "/home/z/my-project/scratch/p15_r342_backend_perf.txt"
TOOLKIT_CWD = "/home/z/my-project/pharma-ml-toolkit"


def get(url: str, token: str) -> tuple[dict, float]:
    req = urllib.request.Request(url, headers={"X-Session-Token": token})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read().decode())
    return body, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    lines: list[str] = []
    lines.append("# r342 backend performance lanes (part15 roadmap #4)")
    lines.append(f"# context: {platform.system()} {platform.release()} "
                 f"py={sys.version.split()[0]} pharmaml-tree=v1.2.240 "
                 f"cpus={os.cpu_count()} cwd={TOOLKIT_CWD}")
    lines.append(f"# data-dir (fresh): {DATA_DIR}")

    # -- lane 1: backend startup (spawn -> PHARMAML-READY handshake) --
    env = {**os.environ, "PHARMAML_STUDIO_DATA": DATA_DIR}
    t_spawn = time.perf_counter()
    proc = subprocess.Popen(
        [VENV_PY, "-m", "pharmaml.api"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, env=env, cwd=TOOLKIT_CWD,
    )
    ready_line = proc.stdout.readline().strip()
    startup_s = time.perf_counter() - t_spawn
    assert ready_line.startswith("PHARMAML-READY"), ready_line
    hs = json.loads(ready_line[len("PHARMAML-READY "):])
    lines.append(f"backend_startup_spawn_to_ready_s: {startup_s:.3f}")
    lines.append(f"handshake: port={hs['port']} pid={hs['pid']} "
                 f"version={hs['version']}")

    base = f"http://127.0.0.1:{hs['port']}/api/v1"
    token = hs["token"]

    # -- lane 2: health latency (first + warm) --
    first, first_ms = get(f"{base}/health", token)
    warm = [get(f"{base}/health", token)[1] for _ in range(3)]
    lines.append(f"health_first_ms: {first_ms:.1f} "
                 f"(status={first['status']} core={first['scientific_core']})")
    lines.append(f"health_warm_ms: min={min(warm):.1f} "
                 f"median={sorted(warm)[1]:.1f} (3 probes)")

    # -- lane 3: idle resources (the r324 read-only instrument) --
    time.sleep(2.0)
    res_a, _ = get(f"{base}/system/resources", token)
    time.sleep(10.0)
    res_b, _ = get(f"{base}/system/resources", token)
    pa, pb = res_a["process"], res_b["process"]
    rss_a = pa["memory_rss_bytes"] / (1024 * 1024)
    rss_b = pb["memory_rss_bytes"] / (1024 * 1024)
    lines.append(f"resources_source: {res_a['source']} "
                 f"scope={res_a['scope']}")
    lines.append(f"backend_process_idle_rss_MiB: a={rss_a:.1f} "
                 f"(+2 s, threads={pa['num_threads']}) b={rss_b:.1f} "
                 f"(+12 s, threads={pb['num_threads']}) drift={rss_b - rss_a:+.2f} MiB")
    lines.append(f"jobs_idle: active={res_a['jobs']['active']} "
                 f"by_state={res_a['jobs']['by_state']}")
    lines.append(f"gpu: {json.dumps(res_a['gpu'])}")
    sysinfo = res_a["system"]
    lines.append("system: " + json.dumps(sysinfo))

    # -- lane 4: graceful shutdown (SIGTERM -> exit) --
    t_term = time.perf_counter()
    proc.send_signal(signal.SIGTERM)
    code = proc.wait(timeout=30)
    shutdown_s = time.perf_counter() - t_term
    lines.append(f"graceful_shutdown_sigterm_to_exit_s: {shutdown_s:.3f} "
                 f"(exit_code={code})")

    out = "\n".join(lines) + "\n"
    with open(EVIDENCE, "w") as fh:
        fh.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
