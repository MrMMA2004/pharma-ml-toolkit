"""The pinned-stack requirements.txt can never drift from pyproject.toml.

The contract (see CHANGELOG 1.2.40): ``pyproject.toml`` keeps the loose
compatibility ranges, ``requirements.txt`` is the exact snapshot the QA
chain runs green under. These tests pin that relationship — every core
dependency declared by the package is pinned exactly once, every pin
satisfies its declared range, and every commented optional block covers
the extras it claims to mirror.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ = ROOT / "requirements.txt"
PYPROJECT = ROOT / "pyproject.toml"

_PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9_.+*-]*)$")
_RANGE = re.compile(r"^([A-Za-z0-9_.-]+)\s*(.*)$")


def _parse_requirements() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Core pins + commented optional blocks (block name -> pinned names)."""
    core: dict[str, str] = {}
    blocks: dict[str, list[str]] = {}
    current: list[str] | None = None  # names inside an optional comment run
    block_names: list[str] = []
    for raw in REQ.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            # Optional-block headers carry the "name==version" pins inline.
            pins = _PIN.findall(body)
            if pins:
                if current is None:
                    # First pin of a new optional run.
                    current = []
                    block_names.append(f"block-{len(blocks)}")
                    blocks[block_names[-1]] = current
                for name, version in pins:
                    core.setdefault(name, version) if False else None
                    current.append(f"{name}=={version}")
            continue
        match = _PIN.match(line)
        assert match, f"unpinned requirement line: {raw!r}"
        name, version = match.groups()
        assert name not in core, f"duplicate core pin for {name}"
        core[name] = version
    return core, blocks


def _pyproject_deps() -> dict[str, str]:
    """Dependencies declared under [project] (name -> range spec)."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"^dependencies\s*=\s*\[(.*?)^\]", text, re.DOTALL | re.MULTILINE
    )
    assert match, "pyproject.toml has no [project] dependencies list"
    deps: dict[str, str] = {}
    for item in re.findall(r'"([^"]+)"', match.group(1)):
        spec = _RANGE.match(item)
        assert spec, f"unparsable dependency: {item!r}"
        name, rng = spec.groups()
        deps[_normalize(name)] = rng.strip()
    return deps


def _pyproject_extras() -> dict[str, dict[str, str]]:
    """Optional-dependency extras (extra name -> {package: range})."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(
        r"^\[project\.optional-dependencies\]\s*(.*?)^\[", text, re.DOTALL | re.MULTILINE
    )
    assert match, "pyproject.toml has no optional-dependencies section"
    extras: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in match.group(1).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = re.match(r"^([A-Za-z0-9_-]+)\s*=\s*\[", line)
        if header:
            current = header.group(1)
            extras[current] = {}
            continue
        assert current is not None, f"orphan line in extras: {raw!r}"
        for item in re.findall(r'"([^"]+)"', line):
            spec = _RANGE.match(item)
            assert spec, f"unparsable extra dependency: {item!r}"
            extras[current][_normalize(spec.group(1))] = spec.group(2).strip()
    return extras


def _normalize(name: str) -> str:
    """PEP 503 name normalization (rdkit-pypi == rdkit_pypi etc.)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _satisfies(version: str, range_spec: str) -> bool:
    """True when `version` satisfies the loose range in `range_spec`.

    Handles the specs this package declares: ``>=X``, ``>=X,<Y`` and the
    empty string (no constraint).
    """
    if not range_spec:
        return True
    parts = [p.strip() for p in range_spec.split(",")]
    for part in parts:
        if not part:
            continue
        op, _, bound = part.partition(">=") if ">=" in part else part.partition("<")
        if ">=" in part:
            op, bound = ">=", part[2:].strip()
        elif "<" in part:
            op, bound = "<", part[part.index("<") :].lstrip("<").strip()
        else:
            continue
        v = _version_key(version)
        b = _version_key(bound)
        if op == ">=" and not v >= b:
            return False
        if op == "<" and not v < b:
            return False
    return True


def _version_key(version: str) -> tuple[int, ...]:
    digits = re.findall(r"\d+", version.split("+")[0])
    return tuple(int(d) for d in digits)


class TestCorePins:
    """The core block: complete, exact, and within the declared ranges."""

    def test_core_pins_every_declared_dependency(self) -> None:
        core, _ = _parse_requirements()
        declared = _pyproject_deps()
        missing = set(declared) - set(core)
        assert not missing, f"pyproject deps not pinned in requirements.txt: {sorted(missing)}"

    def test_core_declares_nothing_extra(self) -> None:
        core, _ = _parse_requirements()
        declared = set(_pyproject_deps())
        # torch lives in the optional deep block (commented), so the core
        # block may only carry declared core names.
        extra = set(core) - declared
        assert not extra, f"core pins not declared in pyproject.toml: {sorted(extra)}"

    def test_every_core_pin_satisfies_its_declared_range(self) -> None:
        core, _ = _parse_requirements()
        declared = _pyproject_deps()
        for name, version in core.items():
            spec = declared.get(_normalize(name), "")
            assert _satisfies(version, spec), (
                f"{name}=={version} violates the declared range {spec!r}"
            )

    def test_pins_are_exact(self) -> None:
        for raw in REQ.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                assert "==" in line, f"non-exact pin: {raw!r}"

    def test_python_note_present(self) -> None:
        assert "3.12" in REQ.read_text(encoding="utf-8")


class TestOptionalBlocks:
    """The commented optional blocks cover the declared extras."""

    def test_every_extra_package_appears_pinned_somewhere(self) -> None:
        _, blocks = _parse_requirements()
        pinned_optional: set[str] = set()
        for names in blocks.values():
            for entry in names:
                pinned_optional.add(_normalize(entry.split("==")[0]))
        extras = _pyproject_extras()
        for extra, deps in extras.items():
            for name in deps:
                assert _normalize(name) in pinned_optional, (
                    f"extra '{extra}' dependency {name!r} has no pinned block"
                )

    def test_blocks_only_name_declared_packages(self) -> None:
        _, blocks = _parse_requirements()
        declared_optional: set[str] = set()
        for deps in _pyproject_extras().values():
            declared_optional.update(_normalize(d) for d in deps)
        for names in blocks.values():
            for entry in names:
                name = _normalize(entry.split("==")[0])
                assert name in declared_optional, (
                    f"optional block pins undeclared package: {entry!r}"
                )

    def test_torch_cpu_index_documented(self) -> None:
        text = REQ.read_text(encoding="utf-8")
        assert "download.pytorch.org/whl/cpu" in text, (
            "the torch block must spell out the CPU-wheel index URL"
        )
