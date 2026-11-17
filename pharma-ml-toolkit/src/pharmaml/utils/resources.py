"""Memory-aware execution presets (part6 §17 E, Phase 5).

§17 E demands optional resource monitoring and "memory-aware execution
presets for low-resource computers", with the explicit honesty rule:

    "Do not claim an accurate GPU or memory estimate without actual
     system evidence."

This module therefore READS the system before saying anything:

* :func:`available_memory_bytes` — the cgroup limit when the process
  runs in one (v2 ``memory.max``, v1 ``memory.limit_in_bytes``),
  intersected with ``/proc/meminfo``'s ``MemAvailable``; ``None`` when
  neither file is readable.  psutil is deliberately NOT a dependency —
  the two files are enough, and adding a hard dependency for a probe
  contradicts §12's "do not turn every optional dependency into a
  mandatory dependency".
* :func:`snapshot` — the evidence bundle the GUI shows (CPU count,
  memory numbers or ``None``-honesty, and the source of each number).

The PRESETS are named resource policies over the two knobs the code
already honours — joblib ``n_jobs`` (via :func:`utils.compute`) and
the streaming reader's ``chunk_rows`` — plus a display name and an
honest "what this trades" note.  No preset claims a GPU estimate: GPUs
are reported only as "not probed" (torch is optional and this module
never imports it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pharmaml.utils.compute import cpu_count

# -- the honest system probe ---------------------------------------------------


def _read_int(path: Path) -> int | None:
    """First integer on the file's first line, or None."""
    try:
        text = path.read_text(encoding="ascii").splitlines()[0]
    except (OSError, IndexError):
        return None
    for token in text.split():
        try:
            return int(token)
        except ValueError:
            continue
    return None


def _cgroup_limit() -> int | None:
    """The container memory limit, v2 then v1, when one is enforced."""
    # cgroup v2: /sys/fs/cgroup/memory.max ("max" = no limit)
    v2 = _read_int(Path("/sys/fs/cgroup/memory.max"))
    if v2 is not None and v2 > 0:
        return v2
    # cgroup v1: the limit from the (possibly unified-mounted) hierarchy
    try:
        mount_lines = Path("/proc/self/cgroup").read_text(
            encoding="ascii").splitlines()
    except OSError:
        return None
    for line in mount_lines:
        # 0::/  (v2)  |  9:memory:/path  (v1)
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[1] == "memory":
            base = Path("/sys/fs/cgroup/memory") / parts[2].lstrip("/")
            limit = _read_int(base / "memory.limit_in_bytes")
            # v1 uses a huge sentinel when unlimited — not a real limit
            if limit is not None and 0 < limit < 1 << 60:
                return limit
    return None


def _mem_available() -> int | None:
    """MemAvailable from /proc/meminfo (kB line), in bytes."""
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024  # kB -> bytes
                except ValueError:
                    return None
    return None


def available_memory_bytes() -> int | None:
    """The best evidenced memory ceiling, or None when unknowable.

    Inside a container the cgroup limit governs; on a plain host
    MemAvailable does. Both readable → the smaller (the truth is the
    tighter bound); neither → None (§17 E: no claim without evidence).
    """
    limits = [v for v in (_cgroup_limit(), _mem_available()) if v]
    return min(limits) if limits else None


@dataclass(frozen=True)
class ResourceSnapshot:
    """What the system could actually evidence, with sources."""

    cpu_count: int
    memory_bytes: int | None
    memory_source: str  # "cgroup+meminfo" | "meminfo" | "cgroup" | "unknown"

    def memory_gib(self) -> str:
        if self.memory_bytes is None:
            return "unknown"
        return f"{self.memory_bytes / (1 << 30):.1f} GiB"


def snapshot() -> ResourceSnapshot:
    """The live probe (cheap: two or three small file reads)."""
    cgroup = _cgroup_limit()
    meminfo = _mem_available()
    if cgroup and meminfo:
        source = "cgroup+meminfo"
    elif cgroup:
        source = "cgroup"
    elif meminfo:
        source = "meminfo"
    else:
        source = "unknown"
    limits = [v for v in (cgroup, meminfo) if v]
    return ResourceSnapshot(
        cpu_count=cpu_count(),
        memory_bytes=min(limits) if limits else None,
        memory_source=source,
    )


# -- the execution presets -----------------------------------------------------


@dataclass(frozen=True)
class ExecutionPreset:
    """One named resource policy (§17 E) over the two real knobs."""

    key: str          # persisted identifier
    label: str        # GUI text
    n_jobs: int       # resolved worker count (>0; never the 0 nonsense)
    chunk_rows: int   # the streaming reader's chunk size
    trades: str       # the honest "what you give up" sentence


PRESETS: tuple[ExecutionPreset, ...] = (
    ExecutionPreset(
        key="balanced",
        label="Balanced (default)",
        n_jobs=-1,          # all cores — resolved via effective_n_jobs
        chunk_rows=50_000,  # the reader's default
        trades="Full CPU use and the default streaming chunk size.",
    ),
    ExecutionPreset(
        key="eco",
        label="Low-memory (eco)",
        n_jobs=1,
        chunk_rows=10_000,
        trades="One worker and 10k-row chunks: slower, but a fraction "
               "of the peak RAM — for machines under memory pressure.",
    ),
)


def find_preset(key: str) -> ExecutionPreset:
    """The preset for ``key`` (unknown keys fall back to balanced)."""
    for preset in PRESETS:
        if preset.key == key:
            return preset
    return PRESETS[0]


def preset_for_low_memory(threshold_bytes: int = 2 << 30) -> str:
    """The preset key the snapshot honestly recommends.

    "eco" only when memory was evidenced AND is under the threshold;
    otherwise "balanced" — an unknown memory size is never treated as
    a low-memory signal (§17 E honesty).
    """
    snap = snapshot()
    if snap.memory_bytes is not None and snap.memory_bytes < threshold_bytes:
        return "eco"
    return "balanced"


# -- persistence (QSettings-free; the GUI passes the value through) -------------

_SETTINGS_ENV = "PHARML_EXECUTION_PRESET"


def stored_preset() -> str:
    """The persisted preset key (env override, else balanced)."""
    key = os.environ.get(_SETTINGS_ENV, "")
    return key if any(p.key == key for p in PRESETS) else "balanced"


def store_preset(key: str) -> bool:
    """Persist the preset choice in-process (env var; the GUI's
    QSettings remain the durable store — this is the session echo)."""
    if not any(p.key == key for p in PRESETS):
        return False
    os.environ[_SETTINGS_ENV] = key
    return True
