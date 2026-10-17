"""29 driver: multi-atom linker path transplants (notebook side).

Round 100x-29's flagship: ``scaffold_hop_crossover``'s transplant
geometry now excises whole short acyclic *paths* (up to four atoms --
an ethylene bridge, an oxyethylene segment) and zips them into a cut
ring bond of the other parent: ring homologation by a whole linker
chain, with the signature-matching invariant enforced at every seam and
flipped-only matches excising the reversed path.  This driver lands the
notebook documentation:

1. PATCH the generative notebook's ring-topology markdown (transplant
   bullet now covers paths) and INSERT a linker-path demo cell pair
   after the 28 ring demo (file-driven: both parents looked up from the
   bundled corpus by SMILES value, no inline fabrication).
2. PATCH the walkthrough's 8 bullet to mention path transplants.
3. EXECUTE both notebooks with nbclient (cwd=examples/, worker-cap env)
   -- the structural GA runs legitimately drift: the corpus carries
   nicotine (saturated pyrrolidine ring) and chain donors
   (cimetidine, lysine, metoprolol, propranolol), so the GA can now
   reach the path geometry.
4. TWINS -- regenerate the *_no_output.ipynb variants (idempotent).
5. VERIFY the banned fabrication tokens are absent and the pinned
   spot-check values appear in the executed outputs.

Pins (deterministic, independent of GA trajectories):

  generative 6b path demo (shared Random(0), call order fixed):
    nicotine x cimetidine -> 'CN1CCSCCC1c1cccnc1'   (5-ring + -CH2-S- -> 7-ring)
    nicotine x lysine     -> 'CN1CCCCCC1c1cccnc1'   (5-ring + -CH2CH2- -> 7-ring)
  generative 6b ring demos (28, pool-invariant -- both parents fully
  ringed, so no path candidates exist; byte-identical):
    morpholine x piperidine -> 'C1CCNCCOCCNCC1'
    benzodiazepinone x aspirin -> 'CC1=NCC(=O)N(C)c2ccccc2O1'
  generative 6b bridge demo (aromatic parents, unchanged):
    aspirin x naproxen -> 'COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1'
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(ROOT / "tests"))  # BANNED_FABRICATION_TOKENS

from test_notebook_variants import BANNED_FABRICATION_TOKENS  # noqa: E402

WORKER_ENV = {
    **os.environ,
    "LOKY_MAX_CPU_COUNT": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

GEN_NOTEBOOK = "complete_generative_workflow.ipynb"
WALK_NOTEBOOK = "pharmaml_walkthrough.ipynb"

PATH_MD_MARKER = "### Linker paths"

PATH_MD = """\
### Linker paths — multi-atom transplants

The transplant geometry also excises whole short chains: a contiguous
acyclic path of up to four atoms — an ethylene bridge, an oxyethylene
segment — is cut loose from its two hosts and zipped into a cut ring
bond of the other parent. Ring expansion by a whole linker, not just
one atom: the ring-size axis of scaffold morphing (piperidine →
azepane → azocane). Both demo parents are looked up from the bundled
corpus again — nothing invented inline.
"""

PATH_CODE = '''\
# multi-atom linker paths -- ring expansion by a whole chain, corpus parents
draw = random.Random(0)

# nicotine's saturated pyrrolidine ring x cimetidine's -CH2-S- path:
# the 5-ring becomes a 7-ring carrying the transplanted sulfur
nicotine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CN1CCCC1c1cccnc1")]
cimetidine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("CN=C(NC#N)NCCSCc1[nH]cnc1C")]
expanded = scaffold_hop_crossover(nicotine, cimetidine, draw)
print(f"ring parent   : {nicotine}   rings={ring_sizes(nicotine)}")
print(f"path donor    : {cimetidine}")
print(
    f"expanded      : {expanded}   rings={ring_sizes(expanded)}"
    "  <- the pyrrolidine gained cimetidine's -CH2-S-\\n"
)

# lysine's -CH2-CH2- path: the same 5-ring becomes a 7-ring azepane
lysine = DEFAULT_CORPUS[DEFAULT_CORPUS.index("NCCCCC(N)C(=O)O")]
homologated = scaffold_hop_crossover(nicotine, lysine, draw)
print(f"ring parent   : {nicotine}   rings={ring_sizes(nicotine)}")
print(f"path donor    : {lysine}  (the middle -CH2CH2-)")
print(
    f"homologated   : {homologated}   rings={ring_sizes(homologated)}"
    "  <- two methylenes longer"
)
'''

# --- markdown cell replacements (old snippet -> new snippet) ------------

GEN_RINGMD_OLD = """\
* **transplant** — a two-ended linker atom (acyclic, two single bonds,
  homologous flanks) is excised from one parent and zipped into a cut
  ring bond of the other: ring homologation — the corpus
  benzodiazepinone's 7-ring below becomes an 8-ring oxazepinone with
  aspirin's ester oxygen.
"""

GEN_RINGMD_NEW = """\
* **transplant** — a two-ended linker — a single atom (acyclic, two
  single bonds, homologous flanks) or a whole contiguous acyclic *path*
  of up to four atoms — is excised from one parent and zipped into a
  cut ring bond of the other: ring homologation by one atom or a whole
  chain — the corpus benzodiazepinone's 7-ring below becomes an 8-ring
  oxazepinone with aspirin's ester oxygen, and the linker-path demo
  after it expands nicotine's pyrrolidine by cimetidine's `-CH2-S-`.
"""

WALK_8_OLD = """\
  swapped cross-wise, two rings merged into one, or a linker atom
  transplanted into a cut ring bond — so t"""

WALK_8_NEW = """\
  swapped cross-wise, two rings merged into one, or a two-ended linker
  — a single atom or a short acyclic path — transplanted into a cut
  ring bond — so t"""

# (notebook, marker after which to insert, [(kind, source), ...])
INSERTIONS = {
    GEN_NOTEBOOK: (
        "the ester O is a two-ended linker",  # end of the 28 ring demo cell
        [("markdown", PATH_MD), ("code", PATH_CODE)],
    ),
}

REPLACEMENTS = {
    GEN_NOTEBOOK: [(GEN_RINGMD_OLD, GEN_RINGMD_NEW)],
    WALK_NOTEBOOK: [(WALK_8_OLD, WALK_8_NEW)],
}

PINS = {
    GEN_NOTEBOOK: [
        "CN1CCSCCC1c1cccnc1",   # path demo: cimetidine -CH2-S- transplant
        "CN1CCCCCC1c1cccnc1",   # path demo: lysine -CH2CH2- transplant
        "C1CCNCCOCCNCC1",       # merge demo child (12-ring, from 28)
        "CC1=NCC(=O)N(C)c2ccccc2O1",  # transplant demo child (from 28)
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
    ],
    WALK_NOTEBOOK: [
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
    ],
}


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def insert_cells() -> None:
    import nbformat

    for name, (marker, cells) in INSERTIONS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        if any(PATH_MD_MARKER in "".join(c.source) for c in nb.cells):
            print(f"insert: {name} already carries the linker-path cells")
            continue
        anchor = next(
            (i for i, c in enumerate(nb.cells) if marker in "".join(c.source)),
            None,
        )
        if anchor is None:
            _fail(f"{name}: anchor marker {marker!r} not found")
        for offset, (kind, source) in enumerate(cells, start=1):
            if kind == "markdown":
                nb.cells.insert(anchor + offset, nbformat.v4.new_markdown_cell(source))
            else:
                nb.cells.insert(anchor + offset, nbformat.v4.new_code_cell(source))
        nbformat.write(nb, path)
        print(f"inserted {len(cells)} cell(s) into {name} after cell {anchor}")


def replace_markdown() -> None:
    import nbformat

    for name, pairs in REPLACEMENTS.items():
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        changed = 0
        for old, new in pairs:
            hit = False
            for cell in nb.cells:
                src = "".join(cell.source)
                if old in src:
                    cell.source = src.replace(old, new)
                    hit = True
                    changed += 1
                    break
            if not hit and new.splitlines()[0] not in "".join(
                c.source for c in nb.cells if c.cell_type == "markdown"
            ):
                _fail(f"{name}: replacement anchor not found:\n{old[:80]}...")
        if changed:
            nbformat.write(nb, path)
        print(f"{name}: {changed} markdown replacement(s) applied")


def execute_notebooks() -> None:
    import nbformat
    from nbclient import NotebookClient

    for name in (GEN_NOTEBOOK, WALK_NOTEBOOK):
        path = EXAMPLES / name
        nb = nbformat.read(path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        started = time.perf_counter()
        client = NotebookClient(
            nb,
            timeout=2400,
            kernel_name="python3",
            resources={"metadata": {"path": str(EXAMPLES)}},
        )
        client.execute()
        nbformat.write(nb, path)
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        executed = sum(1 for c in code_cells if c.get("execution_count"))
        errors = [
            o
            for c in code_cells
            for o in c.get("outputs", [])
            if o.output_type == "error"
        ]
        if errors or executed != len(code_cells):
            _fail(
                f"{name}: {len(errors)} error output(s), "
                f"{executed}/{len(code_cells)} cells executed"
            )
        print(
            f"executed {name}: {len(code_cells)} code cells, "
            f"{time.perf_counter() - started:.1f}s"
        )


def regenerate_twin() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scratch" / "strip_notebook_outputs.py")],
        capture_output=True,
        text=True,
        env=WORKER_ENV,
    )
    if result.returncode != 0:
        _fail(f"strip_notebook_outputs.py failed: {result.stderr[-400:]}")
    print("twins regenerated (5 notebooks, idempotent)")


def verify() -> None:
    import nbformat

    for name, pins in PINS.items():
        nb = nbformat.read(EXAMPLES / name, as_version=4)
        chunks: list[str] = []
        for cell in nb.cells:
            chunks.append("".join(cell.source))
            for out in cell.get("outputs", []):
                if isinstance(out.get("text"), str):
                    chunks.append(out["text"])
                plain = out.get("data", {}).get("text/plain")
                if isinstance(plain, str):
                    chunks.append(plain)
        text = "\n".join(chunks)
        for token in BANNED_FABRICATION_TOKENS:
            if token in text:
                _fail(f"{name}: banned fabrication token {token!r} present")
        for pin in pins:
            if pin not in text:
                _fail(f"{name}: pin {pin!r} missing from executed output")
        print(f"verified {name}: guard clean, {len(pins)} pin(s) present")


def main() -> None:
    insert_cells()
    replace_markdown()
    execute_notebooks()
    regenerate_twin()
    verify()
    print("DONE: linker-path transplants documented, executed and pinned")


if __name__ == "__main__":
    main()
