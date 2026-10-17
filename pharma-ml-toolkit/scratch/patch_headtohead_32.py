"""32 driver: the head-to-head structural arm (notebook + script twin).

Round 100x-32's showcase item (carried since 100x-29): the generative
notebook's §9 "GA vs RL" head-to-head gains its **structural arm** --
the §6b ``ga_hop`` run (scaffold-preserving mutation + the
five-geometry scaffold-hop crossover) joins the comparison frame, and
a second overlap line asks whether framework-constrained recombination
visits the same molecules as unconstrained character splicing.

While here, two narrative-consistency defects in the same notebook:

* §6b's intro still counted "Three geometries" (stale since the 29-31
  operator growth) -- now five, enumerated;
* the recap's §9 bullet and CLI hint predate both the structural arm
  and the ``--scaffold-hop`` CLI flag.

The runnable script twin ``examples/run_generative_design.py`` grows
the same fourth arm (its §3 adds the structural GA run, its §5 frame
and leaderboard the ``ga (structural)`` row), keeping notebook and
script in lock-step as the recap promises.

Steps: PATCH the notebook (idempotent) -> PATCH the script twin ->
EXECUTE the notebook with nbclient (cwd=examples/, worker-cap env) ->
REGENERATE the *_no_output.ipynb twins -> RUN the script twin ->
VERIFY banned-fabrication tokens absent and the pins present.
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
SCRIPT_TWIN = "run_generative_design.py"

# --- notebook: §6b intro count fix (three -> five geometries) -----------

GEN_6B_OLD_1 = "Three geometries compete for the draw: **bridge cuts**"
GEN_6B_NEW_1 = "Five geometries compete for the draw: **bridge cuts**"

GEN_6B_OLD_2 = """\
parents, Nicolaou et al.'s BREED tradition), plus the two **ring-bond
cuts** below. The operator also works standalone —
"""
GEN_6B_NEW_2 = """\
parents, Nicolaou et al.'s BREED tradition), the **ring-bond cuts**
below — merges, linker transplants, fused-ring splices — plus the
**ring insert**, a whole ring spliced into a cut acyclic bond as the
linker. The operator also works standalone —
"""

# --- notebook: §9 markdown (whole-cell source replacement) --------------

HEADTOHEAD_MD_OLD_FIRST_LINE = "## 9 · GA vs RL — same objective, same seed, head-to-head"

HEADTOHEAD_MD = """\
## 9 · GA vs RL — same objective, same seed, head-to-head

Four arms, one objective, one seed: the de novo GA (character-splice
crossover), the seeded GA, the same GA in **structural mode** (§6b's
scaffold-preserving mutation plus the five-geometry scaffold-hop
crossover — every candidate is a framework recombination with real
ring chemistry), and the RL policy. The optimisers are complementary,
not competing: the GA explores a wide combinatorial space cheaply; the
structural GA searches the narrower space of framework
recombinations; RL trains a *policy* that can then generate on its
own. On this objective and budget the numbers land where they land —
the interesting lines are the last two: whether the searches even
visit the same molecules, and whether framework-constrained
recombination gives up anything against unconstrained character
splicing.

CLI twins (the seeded arm takes `--seeds your_table.csv`):

```bash
pharmaml generative run --method ga --recipe druglike --iterations 20 --seed 42
pharmaml generative run --method ga --recipe druglike --scaffold --scaffold-hop --iterations 15 --seed 42
pharmaml generative run --method rl --recipe druglike --iterations 15 --out molecules.csv
```
"""

# --- notebook: §9 code cell (whole-cell source replacement) --------------

HEADTOHEAD_CODE_OLD_FIRST_LINE = "methods = pd.DataFrame("

HEADTOHEAD_CODE = '''\
methods = pd.DataFrame(
    {
        "method": ["ga (de novo)", "ga (seeded)", "ga (structural)", "rl"],
        "best_score": [
            ga_denovo.best_score,
            ga_seeded.best_score,
            ga_hop.best_score,
            rl.best_score,
        ],
        "best_smiles": [ga_denovo.best, ga_seeded.best, ga_hop.best, rl.best],
        "unique": [len(ga_denovo), len(ga_seeded), len(ga_hop), len(rl)],
        "elapsed_s": [
            ga_denovo.metadata["elapsed_s"],
            ga_seeded.metadata["elapsed_s"],
            ga_hop.metadata["elapsed_s"],
            rl.metadata["elapsed_s"],
        ],
    }
)
print(methods.to_string(index=False))

overlap = set(ga_denovo.smiles[:10]) & set(rl.smiles[:10])
print(f"\\ntop-10 overlap GA (de novo) vs RL: "
      f"{overlap or 'none — two searches, different corners'}")
overlap_structural = set(ga_hop.smiles[:10]) & set(ga_denovo.smiles[:10])
print(f"top-10 overlap GA (structural) vs GA (de novo): "
      f"{overlap_structural or 'none — recombination, other molecules'}")
'''

# --- notebook: recap bullets ---------------------------------------------

RECAP_OLD_1 = "* **GA vs RL** — same objective, complementary searches (§9);"
RECAP_NEW_1 = """\
* **GA vs RL** — same objective, same seed, four arms head-to-head:
  de novo GA, seeded GA, structural-mode GA and the RL policy (§9);
"""

RECAP_OLD_2 = "(+ `--scaffold`, `--seeds`);"
RECAP_NEW_2 = "(+ `--scaffold`, `--scaffold-hop`, `--seeds`);"

NOTEBOOK_REPLACEMENTS = [
    (GEN_6B_OLD_1, GEN_6B_NEW_1),
    (GEN_6B_OLD_2, GEN_6B_NEW_2),
    (RECAP_OLD_1, RECAP_NEW_1),
    (RECAP_OLD_2, RECAP_NEW_2),
]

# --- script twin: run_generative_design.py -------------------------------

SCRIPT_REPLACEMENTS = [
    # docstring: the GA now runs three times
    (
        "3. run the SMILES genetic algorithm twice -- de novo (corpus seeds) and\n"
        "   seeded from the user's molecules;",
        "3. run the SMILES genetic algorithm three times -- de novo (corpus\n"
        "   seeds), seeded from the user's molecules, and in structural mode\n"
        "   (scaffold-hop crossover + scaffold-preserving mutation);",
    ),
    (
        "5. print both leaderboards, a short GA-vs-RL comparison, and save the",
        "5. print the leaderboards, a four-arm GA-vs-RL comparison, and save the",
    ),
    # section 3: the structural run joins de novo + seeded
    (
        '    print(f"seeded  : {ga_seeded.summary()}")\n',
        '    print(f"seeded  : {ga_seeded.summary()}")\n'
        "    ga_hop = GeneticOptimizer(\n"
        '        objective, population_size=48, generations=20, seed=42,\n'
        '        mutation="scaffold", crossover="scaffold_hop",\n'
        "    ).optimize(seed_smiles=seed_smiles)\n"
        '    print(f"structural (scaffold-hop crossover): {ga_hop.summary()}")\n',
    ),
    # section 5: the four-arm frame
    (
        '            "method": ["ga (de novo)", "ga (seeded)", "rl"],\n'
        '            "best_score": [ga_denovo.best_score, ga_seeded.best_score, rl.best_score],\n'
        '            "best_smiles": [ga_denovo.best, ga_seeded.best, rl.best],\n'
        '            "unique": [len(ga_denovo), len(ga_seeded), len(rl)],\n'
        '            "elapsed_s": [\n'
        '                ga_denovo.metadata["elapsed_s"],\n'
        '                ga_seeded.metadata["elapsed_s"],\n'
        '                rl.metadata["elapsed_s"],\n'
        "            ],",
        '            "method": ["ga (de novo)", "ga (seeded)", "ga (structural)", "rl"],\n'
        '            "best_score": [\n'
        "                ga_denovo.best_score,\n"
        "                ga_seeded.best_score,\n"
        "                ga_hop.best_score,\n"
        "                rl.best_score,\n"
        "            ],\n"
        '            "best_smiles": [ga_denovo.best, ga_seeded.best, ga_hop.best, rl.best],\n'
        '            "unique": [len(ga_denovo), len(ga_seeded), len(ga_hop), len(rl)],\n'
        '            "elapsed_s": [\n'
        '                ga_denovo.metadata["elapsed_s"],\n'
        '                ga_seeded.metadata["elapsed_s"],\n'
        '                ga_hop.metadata["elapsed_s"],\n'
        '                rl.metadata["elapsed_s"],\n'
        "            ],",
    ),
    # section 5: the structural overlap line
    (
        '    print(f"top-10 overlap between GA (de novo) and RL: {overlap}")\n',
        '    print(f"top-10 overlap between GA (de novo) and RL: {overlap}")\n'
        "    overlap_structural = (\n"
        "        set(ga_hop.smiles[:10]) & set(ga_denovo.smiles[:10])\n"
        '    ) or "none -- framework recombination reaches other molecules"\n'
        '    print(f"top-10 overlap between GA (structural) and GA (de novo): "\n'
        '          f"{overlap_structural}")\n',
    ),
    # leaderboard: the structural arm's rows join the bundle
    (
        '            ga_denovo.to_dataframe().assign(method="ga_de_novo"),\n'
        '            ga_seeded.to_dataframe().assign(method="ga_seeded"),\n',
        '            ga_denovo.to_dataframe().assign(method="ga_de_novo"),\n'
        '            ga_seeded.to_dataframe().assign(method="ga_seeded"),\n'
        '            ga_hop.to_dataframe().assign(method="ga_hop"),\n',
    ),
]

# --- pins -----------------------------------------------------------------

PINS = {
    GEN_NOTEBOOK: [
        "ga (structural)",  # the fourth methods row
        "top-10 overlap GA (structural) vs GA (de novo):",
        "Five geometries compete for the draw",  # 6b intro count fix
        "--scaffold --scaffold-hop",  # CLI twin line
        "four arms head-to-head",  # recap bullet
        # 31 regression pins (must survive the re-execution untouched)
        "NCc1ccc(C(=O)O)cc1",  # insert demo: ibuprofen benzene into glycine
        "CCOc1ccccc1C(C)=O",  # insert demo: aspirin benzene into ethyl acetate
        "COc1ccc2cc(C(C)C(=O)C(=O)c3ccccc3OC(C)=O)ccc2c1",  # bridge demo
    ],
}

SCRIPT_PINS = [
    "ga (structural)",
    "top-10 overlap between GA (structural) and GA (de novo):",
    "structural (scaffold-hop crossover):",
]


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def patch_notebook() -> None:
    import nbformat

    path = EXAMPLES / GEN_NOTEBOOK
    nb = nbformat.read(path, as_version=4)
    if any("ga (structural)" in "".join(c.source) for c in nb.cells):
        print(f"notebook: already carries the structural arm")
        return
    # whole-cell replacements for the §9 markdown + code cells
    for cell in nb.cells:
        src = "".join(cell.source)
        if cell.cell_type == "markdown" and src.startswith(
            HEADTOHEAD_MD_OLD_FIRST_LINE
        ):
            cell.source = HEADTOHEAD_MD
            print("notebook: §9 markdown replaced (four-arm narrative)")
        elif (
            cell.cell_type == "code"
            and src.startswith(HEADTOHEAD_CODE_OLD_FIRST_LINE)
            and "ga_seeded.best_score" in src
        ):
            cell.source = HEADTOHEAD_CODE
            print("notebook: §9 code cell replaced (structural row + overlap)")
    # substring replacements (§6b intro + recap bullets)
    for old, new in NOTEBOOK_REPLACEMENTS:
        hit = False
        for cell in nb.cells:
            src = "".join(cell.source)
            if old in src:
                cell.source = src.replace(old, new, 1)
                hit = True
                break
        if not hit:
            _fail(f"notebook: replacement anchor not found:\n{old[:70]}...")
    if "ga (structural)" not in "".join(
        c.source for c in nb.cells if c.cell_type == "code"
    ):
        _fail("notebook: §9 code cell was not replaced (anchor missed)")
    nbformat.write(nb, path)
    print("notebook: written")


def patch_script() -> None:
    path = EXAMPLES / SCRIPT_TWIN
    text = path.read_text(encoding="utf-8")
    if "ga (structural)" in text:
        print("script twin: already carries the structural arm")
        return
    for old, new in SCRIPT_REPLACEMENTS:
        if old not in text:
            _fail(f"script twin: replacement anchor not found:\n{old[:70]}...")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("script twin: 6 replacement(s) applied")


def execute_notebook() -> None:
    import nbformat
    from nbclient import NotebookClient

    path = EXAMPLES / GEN_NOTEBOOK
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
        o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"
    ]
    if errors or executed != len(code_cells):
        _fail(
            f"{GEN_NOTEBOOK}: {len(errors)} error output(s), "
            f"{executed}/{len(code_cells)} cells executed"
        )
    print(
        f"executed {GEN_NOTEBOOK}: {len(code_cells)} code cells, "
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


def run_script_twin() -> None:
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / SCRIPT_TWIN)],
        capture_output=True,
        text=True,
        env=WORKER_ENV,
        cwd=str(EXAMPLES),
    )
    elapsed = time.perf_counter() - started
    out = result.stdout + result.stderr
    evidence = Path("/tmp/task32a")
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "script_twin_output.txt").write_text(out, encoding="utf-8")
    if result.returncode != 0:
        _fail(f"script twin exited {result.returncode}: {out[-600:]}")
    for pin in SCRIPT_PINS:
        if pin not in out:
            _fail(f"script twin output missing pin {pin!r}")
    print(f"script twin: OK, {elapsed:.1f}s, {len(SCRIPT_PINS)} pin(s) present")


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
    patch_notebook()
    patch_script()
    execute_notebook()
    regenerate_twin()
    run_script_twin()
    verify()
    print("DONE: head-to-head structural arm landed, executed and pinned")


if __name__ == "__main__":
    main()
