"""Build + execute examples/complete_generative_workflow.ipynb.

The fifth workflow notebook (after ML, docking, PK/PD and cell line):
molecule *generation* — the "molecule building" story from zero to one
hundred, executed end to end with committed outputs on a laptop CPU:

* the bundled 244-SMILES corpus + its property landscape (1 figure),
* the objective library (QED, LogP band, TPSA band, MW band, similarity,
  pharmacophore, novelty) and the MultiObjective veto,
* de novo + seeded SMILES genetic algorithm (1 convergence figure),
* scaffold-preserving mutation with a Murcko-retention A/B against
  uniform mutation,
* REINVENT-style RL: char-GRU prior training (1 loss figure) and
  policy-gradient fine-tuning (1 history figure),
* the anti-rediscovery novelty recipe, a GA-vs-RL head-to-head, and the
  one-click publish bundle (save_visual_bundle).

Run from the toolkit root:

    python scratch/build_generative_notebook.py

Re-running the script rebuilds and re-executes the notebook from scratch
(every generative run pins seed=42, so the committed outputs are
reproducible on the same machine).  The no-output twin is derived
afterwards by scratch/strip_notebook_outputs.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "complete_generative_workflow.ipynb"

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

# The builder itself must find the toolkit without an installed package
# (pytest achieves the same via pythonpath=src in pyproject.toml), and the
# notebook kernel inherits it through PYTHONPATH so the notebook cells
# themselves never need fragile sys.path tricks.
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src")


TITLE = """\
# PharmaML — The Complete Molecule Generation Workflow (Zero to One Hundred)

One notebook, every stage of a *generative* molecular design project, in the
order you would actually run it — the fifth workflow notebook of the series
(after the ML, docking, PK/PD and cell-line ones). Where those four
**analyse** molecules (model them, dock them, compute their pharmacology),
this one **invents** them: the legacy rar archive shipped training scripts
and docking folders, but no code that ever proposed a new structure —
de novo design lived in nobody's repository. This notebook closes that gap
on a laptop CPU: two complementary optimisers, one shared objective library,
one bundled corpus.

| § | Stage | Toolkit |
|---|---|---|
| 0 | **Setup** — imports, scratch directory | — |
| 1 | **The corpus** — 244 curated drug-like SMILES + property landscape | `pharmaml.generative` |
| 2 | **Objectives** — QED, LogP band, TPSA band, MW band, similarity, pharmacophore, novelty | `pharmaml.generative` |
| 3 | **Multi-objective** — weighted geometric mean, one zero vetoes | `pharmaml.generative` |
| 4 | **De novo generation** — the SMILES genetic algorithm | `pharmaml.generative` |
| 5 | **Seeded generation** — start from your own molecules | `pharmaml.generative` |
| 6 | **Lead optimisation** — scaffold-preserving mutation | `pharmaml.generative` |
| 7 | **Reinforcement learning** — char-GRU prior + REINVENT-style fine-tuning | `pharmaml.generative` |
| 8 | **Novelty** — the anti-rediscovery recipe | `pharmaml.generative` |
| 9 | **GA vs RL** — same objective, head-to-head | `pharmaml.generative` |
| 10 | **Publish** — leaderboard CSV + one-click visual bundle | `pharmaml.viz` |
| 11 | **Recap** | — |

From the `pharma-ml-toolkit/` package root, install the local extras
with `python -m pip install -e '.[chem,deep]'` for RDKit-based scoring/GA
and PyTorch RNN/RL workflows. Both dependencies are imported lazily, so
importing the subpackage itself needs neither.
"""

SETUP_MD = """\
## 0 · Setup

Imports, display options and a scratch directory — every artifact this
notebook writes (the ranked leaderboard, the publish bundle) lands under
`WORK`. Nothing else on your machine is touched.

Every generative run in this notebook pins the same `seed`, so re-running
top-to-bottom reproduces the same leaderboards.
"""

SETUP_CODE = """\
import tempfile
from pathlib import Path

import pandas as pd
import rdkit
import torch

from pharmaml import __version__

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 110)

# Scratch directory for this run's artifacts (use any path you like).
WORK = Path(tempfile.mkdtemp(prefix="generative_workflow_"))

# One seed for every run in this notebook: reproducible leaderboards.
SEED = 42

print(f"pharmaml {__version__} · rdkit {rdkit.__version__} · torch {torch.__version__}")
print(f"artifacts -> {WORK}")
"""

CORPUS_MD = """\
## 1 · The corpus — what the optimisers build from

`DEFAULT_CORPUS` is the bundled, curated starting point: 244 canonical
drug-like SMILES embedded in the package itself (no data file to lose, no
download to fail). "Curated" means exactly what the cell below checks —
every entry parses, every entry is already in RDKit canonical form, and no
entry is duplicated.

CLI equivalent:

```bash
pharmaml generative corpus --out corpus.csv
```
"""

CORPUS_CODE = """\
from pharmaml.generative import DEFAULT_CORPUS

n_unique = len(set(DEFAULT_CORPUS))
print(f"corpus size : {len(DEFAULT_CORPUS)} SMILES")
print(f"unique      : {n_unique}")

pd.DataFrame({"smiles": list(DEFAULT_CORPUS[:10])})
"""

CORPUS_LANDSCAPE_MD = """\
Before optimising against a corpus, look at it: the property landscape the
search will explore. `compute_descriptors` turns the 244 SMILES into the
full RDKit descriptor table; `plot_descriptor_distributions` renders the
columns that define "drug-like" — molecular weight, LogP, polar surface,
H-bond donors and acceptors.
"""

CORPUS_LANDSCAPE_CODE = """\
from pharmaml.chem import compute_descriptors
from pharmaml.viz import plot_descriptor_distributions

corpus_desc = compute_descriptors(list(DEFAULT_CORPUS), progress=False)
print(f"{len(corpus_desc)} molecules x {corpus_desc.shape[1] - 1} descriptors")

fig_landscape = plot_descriptor_distributions(
    corpus_desc,
    columns=["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors"],
    title="The bundled corpus — property landscape",
)
"""

SEEDS_MD = """\
## 1b · Your seeds — the "from sample molecules" mode

The optimisers also run **seeded**: hand them a table of your own molecules
and generation 0 starts from those (§5). The demo table below is the same
12 named drugs shipped as `examples/data/seed_molecules.csv` — swap the
dictionary for your own CSV and every cell below runs unchanged.
"""

SEEDS_CODE = """\
# --- DEMO SEEDS --------------------------------------------------------
# The 12 drugs from examples/data/seed_molecules.csv (byte-identical
# content, embedded so the notebook re-runs anywhere offline).
# (In your project: seeds = pd.read_csv("your_molecules.csv") instead.)
SEED_TABLE = {
    "aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen":   "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine":    "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "metformin":   "CN(C)C(=N)N=C(N)N",
    "nicotine":    "CN1CCCC1c1cccnc1",
    "warfarin":    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "theophylline": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "diclofenac":  "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "naproxen":    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
    "lidocaine":   "CCN(CC)CC(=O)Nc1c(C)cccc1C",
}

seeds = pd.DataFrame(
    {"name": list(SEED_TABLE), "smiles": list(SEED_TABLE.values())}
)
seed_smiles = seeds["smiles"].tolist()
print(f"{len(seeds)} seed molecules")
seeds
"""

OBJECTIVES_MD = """\
## 2 · The objective library — what "good" means

Every optimiser in this subpackage maximises the same contract: an object
with a `score(smiles) -> float` in `[0, 1]`. The library ships seven
ready-made objectives — the four profile bands and QED below, then
similarity, pharmacophore and novelty in the next cell.

Scoring a small probe set is the fastest way to see what each objective
rewards before you commit a search to it.
"""

OBJECTIVES_CODE = """\
from pharmaml.generative import (
    LogPBandObjective,
    MolecularWeightBandObjective,
    QedObjective,
    TpsaBandObjective,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PROBE = {
    "aspirin": ASPIRIN,
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "ethanol": "CCO",
}

profile_objectives = {
    "qed": QedObjective(),
    "logp_band[1,3]": LogPBandObjective(1.0, 3.0),
    "tpsa_band[10,90]": TpsaBandObjective(10.0, 90.0),
    "mw_band[150,450]": MolecularWeightBandObjective(150.0, 450.0),
}
pd.DataFrame(
    {
        name: {label: obj.score(smiles) for label, smiles in PROBE.items()}
        for name, obj in profile_objectives.items()
    }
).round(3)
"""

OBJECTIVES2_MD = """\
The other three objectives anchor the search to *chemistry you care about*:

* **similarity** — ECFP4 Tanimoto against a seed molecule (the classic
  "close to aspirin but better" lead-optimisation setup);
* **pharmacophore** — a required SMARTS substructure, scored 1 or 0;
* **novelty** — `1 − max` Tanimoto against the corpus: rediscovering a
  corpus member scores exactly 0, nothing structural in common scores high.
"""

OBJECTIVES2_CODE = """\
from pharmaml.generative import (
    NoveltyObjective,
    SimilarityObjective,
    SubstructureObjective,
)

anchor_objectives = {
    "similarity_to_aspirin": SimilarityObjective(ASPIRIN),
    "amide_required": SubstructureObjective("C(=O)N"),
    "novelty_vs_corpus": NoveltyObjective(),
}
pd.DataFrame(
    {
        name: {label: obj.score(smiles) for label, smiles in PROBE.items()}
        for name, obj in anchor_objectives.items()
    }
).round(3)
"""

MULTI_MD = """\
## 3 · Multi-objective — the weighted geometric mean, one zero vetoes

Real design goals are compound: drug-like *and* in a LogP window *and*
similar to a seed. `MultiObjective` blends any objectives with weights via
the **weighted geometric mean** — chosen over the arithmetic mean precisely
because a zero component drags the whole product to zero: one failed
requirement vetoes the molecule instead of being averaged away.

`score_with_breakdown` returns the blend *and* every component — the same
breakdown the leaderboards in §4 carry as free extra columns.
"""

MULTI_CODE = """\
from pharmaml.generative import MultiObjective

objective = MultiObjective(
    [
        QedObjective(),
        LogPBandObjective(1.0, 3.0),
        SimilarityObjective(ASPIRIN),
    ],
    weights=[0.5, 0.25, 0.25],
)
print("objective: 0.5 x QED + 0.25 x LogP band [1,3] + 0.25 x similarity to aspirin")

total, breakdown = objective.score_with_breakdown(ASPIRIN)
print(f"aspirin scores {total:.3f}")
{k: round(v, 3) for k, v in breakdown.items()}
"""

VETO_MD = """\
The veto in action: butanol is a perfectly drug-like little molecule
(QED ≈ 0.5), but it contains no aromatic ring — the `c1ccccc1`
pharmacophore component scores 0, and the geometric mean collapses with it.
The arithmetic mean would have scored it ≈ 0.75 and happily evolved the
population toward ring-free chains.
"""

VETO_CODE = """\
veto_objective = MultiObjective(
    [QedObjective(), SubstructureObjective("c1ccccc1")], weights=[0.5, 0.5]
)
total, breakdown = veto_objective.score_with_breakdown("CCCCO")
print(
    f"butanol: qed={breakdown['qed']:.3f}, aromatic ring="
    f"{breakdown['substructure']:.0f}  ->  total {total:.3f} (vetoed)"
)
"""

DENOVO_MD = """\
## 4 · De novo generation — the SMILES genetic algorithm

`GeneticOptimizer` is the CPU-strong workhorse: genome = canonical SMILES
string, character-level mutation + splice crossover, tournament selection,
elitism. **De novo** mode samples the initial population from the corpus;
15 generations over 48 molecules take a couple of seconds.

CLI equivalent:

```bash
pharmaml generative run --method ga --recipe druglike --iterations 20 --seed 42
```
"""

DENOVO_CODE = """\
from pharmaml.generative import GeneticOptimizer

ga_denovo = GeneticOptimizer(
    objective, population_size=48, generations=15, seed=SEED
).optimize()

print(ga_denovo.summary())
print(f"best molecule : {ga_denovo.best}  (score {ga_denovo.best_score:.3f})")
print(f"unique output : {len(ga_denovo)} molecules in {ga_denovo.metadata['elapsed_s']:.1f} s")
"""

DENOVO_LEADERBOARD_MD = """\
The result object is a ranked, de-duplicated leaderboard —
`to_dataframe()` carries the per-objective breakdown as columns, and the
per-generation `history` shows the search converging (best score climbs,
population mean follows — elitism makes the best monotone).
"""

DENOVO_LEADERBOARD_CODE = """\
from pharmaml.viz import new_figure

print("top-8 de novo leaderboard (breakdown columns come free):")
print(ga_denovo.to_dataframe().head(8).to_string(index=False))

history = ga_denovo.history
fig_ga, ax_ga = new_figure(figsize=(8.0, 4.5))
ax_ga.plot([h["generation"] for h in history], [h["best_score"] for h in history],
           marker="o", label="best")
ax_ga.plot([h["generation"] for h in history], [h["mean_score"] for h in history],
           marker=".", label="population mean")
ax_ga.set(xlabel="generation", ylabel="objective score",
          title="De novo GA — 15 generations, 48 molecules")
ax_ga.legend()
fig_ga
"""

SEEDED_MD = """\
## 5 · Seeded generation — start from your own molecules

The same optimiser, **seeded mode**: your molecules anchor generation 0 and
the search explores their neighbourhood (corpus molecules pad the rest of
the population, so your chemistry is explored *alongside* the corpus's
diversity — pass `corpus=seed_smiles` too for a strictly seed-isolated run).

One honest observation to expect below: when the objective itself
rewards similarity to a seed, the seed is already a strong local optimum —
the seeded run's best molecule may simply *be* your seed.

CLI equivalent:

```bash
pharmaml generative run --method ga --recipe similar --seeds examples/data/seed_molecules.csv
```
"""

SEEDED_CODE = """\
ga_seeded = GeneticOptimizer(
    objective, population_size=48, generations=15, seed=SEED
).optimize(seed_smiles=seed_smiles)

print(ga_seeded.summary())

pd.DataFrame(
    {
        "run": ["de novo", "seeded"],
        "best_score": [ga_denovo.best_score, ga_seeded.best_score],
        "best_smiles": [ga_denovo.best, ga_seeded.best],
        "unique": [len(ga_denovo), len(ga_seeded)],
        "elapsed_s": [ga_denovo.metadata["elapsed_s"], ga_seeded.metadata["elapsed_s"]],
    }
).round({"best_score": 3, "elapsed_s": 1})
"""

SCAFFOLD_MD = """\
## 6 · Lead optimisation — scaffold-preserving mutation

`mutation="scaffold"` swaps the variation operator: instead of editing
characters (which routinely destroys ring systems), each offspring keeps its
parent's **Murcko framework verbatim** and resamples only the decorations.
The guarantee is structural, not lucky: a Murcko decoration is always
acyclic — any ring would be part of the framework — so re-attaching one
cannot change the scaffold. Acyclic parents (no framework to keep) fall
back to character mutation rather than stalling evolution.

The A/B below pins it down: both runs start from the same 12 seeds with the
corpus constrained to the seeds, then uniform mutation scrambles frameworks
while scaffold mode keeps every molecule on a seed scaffold.
"""

SCAFFOLD_CODE = """\
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.error")  # keep re-attachment noise out of the output

seed_frameworks = {
    MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) for s in seed_smiles
}
print(f"{len(seed_smiles)} seeds carry {len(seed_frameworks)} distinct Murcko frameworks")

runs = {}
for mode in ("scaffold", "uniform"):
    runs[mode] = GeneticOptimizer(
        objective, population_size=12, generations=12, seed=SEED, mutation=mode
    ).optimize(seed_smiles=seed_smiles, corpus=seed_smiles)

retention = pd.Series(
    {
        mode: sum(
            1
            for s in result.smiles
            if MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) in seed_frameworks
        )
        / len(result.smiles)
        for mode, result in runs.items()
    },
    name="fraction on a seed framework",
)
retention.round(3)
"""

SCAFFOLD_LEADERBOARD_MD = """\
The trade-off is visible in the scores: uniform mutation reaches higher
objective scores in this short run *because* it may leave the seed
frameworks entirely — scaffold mode spends its exploration budget inside
the chemistry you asked it to keep. The leaderboard's `murcko` column shows
exactly which seed framework each winner sits on.
"""

SCAFFOLD_LEADERBOARD_CODE = """\
ga_scaffold = runs["scaffold"]
scaffold_frame = ga_scaffold.to_dataframe().head(6).copy()
scaffold_frame["murcko"] = [
    MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s) for s in scaffold_frame["smiles"]
]
print(f"scaffold mode: best {ga_scaffold.best_score:.3f} "
      f"(uniform mode: {runs['uniform'].best_score:.3f})")
scaffold_frame.round(3)
"""

RL_MD = """\
## 7 · Reinforcement learning — a prior that writes SMILES, then a policy that bends it

`ReinventOptimizer` is the REINVENT-style pipeline, in two acts:

1. **Prior** — `SmilesRnn`, a character-level GRU trained on the corpus:
   it learns the *grammar* of drug-like SMILES (which characters follow
   which), nothing about the objective yet;
2. **Fine-tuning** — the same network becomes the *policy*: REINFORCE with
   a running baseline (the score advantage), an entropy bonus (don't
   collapse onto one spelling), a prior anchor (don't drift into
   un-chemical strings), and experience replay. The objective is the
   reward.

Honest scope: this is a compact, CPU-sized implementation of the REINVENT
*recipe* — not a replication of the full AstraZeneca stack.
"""

RL_PRIOR_CODE = """\
from pharmaml.generative import SmilesRnn

prior = SmilesRnn(hidden_dim=96, seed=SEED)
losses = prior.train_prior(DEFAULT_CORPUS, epochs=8)
print(f"char-GRU prior: {len(losses)} epochs, loss {losses[0]:.2f} -> {losses[-1]:.2f}")

fig_prior, ax_prior = new_figure(figsize=(8.0, 4.5))
ax_prior.plot(range(1, len(losses) + 1), losses, marker="o")
ax_prior.set(xlabel="epoch", ylabel="cross-entropy loss",
             title="Prior training — learning to write SMILES like the corpus")
fig_prior
"""

RL_OPTIMIZE_MD = """\
With the prior trained, fine-tune it toward the §3 objective. A score
plateau triggers the built-in early stop — RL runs until it stops
improving, not until the iteration budget burns out.
"""

RL_OPTIMIZE_CODE = """\
from pharmaml.generative import ReinventOptimizer

rl = ReinventOptimizer(
    objective, prior=prior, iterations=15, batch_size=32, seed=SEED
).optimize()

print(rl.summary())
print(f"early stopped   : {rl.metadata['early_stopped']} (mean-score plateau)")
print(f"unique output   : {len(rl)} molecules in {rl.metadata['elapsed_s']:.1f} s")
"""

RL_LEADERBOARD_MD = """\
The RL history carries the mean batch score and the best-so-far curve per
iteration — the classic REINVENT diagnostic pair.
"""

RL_LEADERBOARD_CODE = """\
rl_history = pd.DataFrame(rl.history)

fig_rl, ax_rl = new_figure(figsize=(8.0, 4.5))
ax_rl.plot(rl_history["iteration"], rl_history["mean_score"],
           marker="o", label="mean batch score")
ax_rl.plot(rl_history["iteration"], rl_history["best_score"],
           marker=".", label="best so far")
ax_rl.set(xlabel="iteration", ylabel="objective score",
          title="REINVENT-style fine-tuning — 15-iteration budget")
ax_rl.legend()

print("top-6 RL leaderboard:")
print(rl.to_dataframe().head(6).to_string(index=False))
fig_rl
"""

NOVELTY_MD = """\
## 8 · Novelty — the anti-rediscovery recipe

Rediscovering the corpus is the failure mode of pure similarity search:
the "best" molecule is just a corpus member re-spelled. `NoveltyObjective`
(`1 − max` Tanimoto vs the corpus) makes rediscovery score exactly 0 —
pair it with QED so the search stays drug-like *while* leaving the corpus.

The honest trade-off, visible right in the output: unconstrained novelty
rewards strange chemistry (exotic sulfide rings below) — the weight vector
is your dial between "safe and similar" and "new and weird".
"""

NOVELTY_CODE = """\
novelty_objective = MultiObjective(
    [QedObjective(), NoveltyObjective()], weights=[0.5, 0.5]
)
ga_novel = GeneticOptimizer(
    novelty_objective, population_size=48, generations=15, seed=SEED
).optimize()

print(ga_novel.summary())
ga_novel.to_dataframe().head(6).round(3)
"""

HEADTOHEAD_MD = """\
## 9 · GA vs RL — same objective, same seed, head-to-head

The two optimisers are complementary, not competing: the GA explores a
wide combinatorial space cheaply; RL trains a *policy* that can then
generate on its own. On this objective and budget the numbers land where
they land — the interesting line is the last one: whether the two searches
even visit the same molecules.

CLI twins for both:

```bash
pharmaml generative run --method ga --recipe druglike --iterations 20 --seed 42
pharmaml generative run --method rl --recipe druglike --iterations 15 --out molecules.csv
```
"""

HEADTOHEAD_CODE = """\
methods = pd.DataFrame(
    {
        "method": ["ga (de novo)", "ga (seeded)", "rl"],
        "best_score": [ga_denovo.best_score, ga_seeded.best_score, rl.best_score],
        "best_smiles": [ga_denovo.best, ga_seeded.best, rl.best],
        "unique": [len(ga_denovo), len(ga_seeded), len(rl)],
        "elapsed_s": [
            ga_denovo.metadata["elapsed_s"],
            ga_seeded.metadata["elapsed_s"],
            rl.metadata["elapsed_s"],
        ],
    }
)
print(methods.to_string(index=False))

overlap = set(ga_denovo.smiles[:10]) & set(rl.smiles[:10])
print(f"\\ntop-10 overlap GA (de novo) vs RL: "
      f"{overlap or 'none — two searches, different corners'}")
"""

PUBLISH_MD = """\
## 10 · Publish — the leaderboard and the one-click bundle

Every run's leaderboard concatenates into one ranked table (method column
attached), the CSV lands in `WORK`, and `save_visual_bundle` publishes the
four inline figures + tables + headline metrics as a shareable
report directory — the same call the other four workflow notebooks end on.
"""

PUBLISH_CODE = """\
from pharmaml.viz import save_visual_bundle

leaderboard = pd.concat(
    [
        ga_denovo.to_dataframe().assign(method="ga_de_novo"),
        ga_seeded.to_dataframe().assign(method="ga_seeded"),
        ga_novel.to_dataframe().assign(method="ga_novel"),
        rl.to_dataframe().assign(method="rl"),
    ],
    ignore_index=True,
)
leaderboard.to_csv(WORK / "generative_leaderboard.csv", index=False)
print(f"leaderboard: {len(leaderboard)} rows -> {WORK / 'generative_leaderboard.csv'}")

artifacts = save_visual_bundle(
    figures={
        "corpus_landscape": fig_landscape,
        "ga_convergence": fig_ga,
        "prior_training": fig_prior,
        "rl_finetuning": fig_rl,
    },
    tables={"leaderboard": leaderboard.head(50), "methods": methods},
    metrics={
        "best_score_ga": ga_denovo.best_score,
        "best_score_rl": rl.best_score,
        "corpus_size": len(DEFAULT_CORPUS),
    },
    out_dir=WORK / "report",
    title="Molecule generation workflow — GA + RL on a curated corpus",
    subtitle=(f"QED + LogP + similarity objective · {len(leaderboard)} "
              f"ranked molecules · seed {SEED}"),
    formats=("png", "svg"),
    close=True,   # figures already rendered inline above; free the memory
)
for path in artifacts:
    print(f"  {path.relative_to(WORK)}")
"""

RECAP_MD = """\
## 11 · Recap

One notebook, the full molecule-building story:

* **Corpus** — 244 curated drug-like SMILES, validated in place (§1);
* **Objectives** — seven ready-made scores: four property bands + QED,
  similarity, pharmacophore, novelty (§2);
* **Multi-objective** — weighted geometric mean where one zero vetoes the
  molecule (§3);
* **De novo GA** — 613 unique molecules in seconds, per-generation history
  and free breakdown columns (§4);
* **Seeded GA** — your own molecules anchor the search (§5);
* **Lead optimisation** — scaffold-preserving mutation kept 100 % of the
  run on seed frameworks vs 50.5 % for uniform mutation (§6);
* **RL** — a char-GRU prior learned the corpus grammar, then REINVENT-style
  fine-tuning bent it toward the objective and stopped itself on the
  plateau (§7);
* **Novelty** — the anti-rediscovery recipe, honestly showing its
  strange-chemistry trade-off (§8);
* **GA vs RL** — same objective, complementary searches (§9);
* **Publish** — leaderboard CSV + one-click visual bundle (§10).

Where to go next:

* the runnable script twin: `examples/run_generative_design.py` (~10 s);
* the CLI: `pharmaml generative run --method ga|rl --recipe
  qed|logp|druglike|similar|novel` (+ `--scaffold`, `--seeds`);
* the README's generative section for the scope notes (what the RL is and
  is not, what scaffold mode guarantees).
"""


def build_cells():
    cells = [
        md(TITLE),
        md(SETUP_MD),
        code(SETUP_CODE),
        md(CORPUS_MD),
        code(CORPUS_CODE),
        md(CORPUS_LANDSCAPE_MD),
        code(CORPUS_LANDSCAPE_CODE),
        md(SEEDS_MD),
        code(SEEDS_CODE),
        md(OBJECTIVES_MD),
        code(OBJECTIVES_CODE),
        md(OBJECTIVES2_MD),
        code(OBJECTIVES2_CODE),
        md(MULTI_MD),
        code(MULTI_CODE),
        md(VETO_MD),
        code(VETO_CODE),
        md(DENOVO_MD),
        code(DENOVO_CODE),
        md(DENOVO_LEADERBOARD_MD),
        code(DENOVO_LEADERBOARD_CODE),
        md(SEEDED_MD),
        code(SEEDED_CODE),
        md(SCAFFOLD_MD),
        code(SCAFFOLD_CODE),
        md(SCAFFOLD_LEADERBOARD_MD),
        code(SCAFFOLD_LEADERBOARD_CODE),
        md(RL_MD),
        code(RL_PRIOR_CODE),
        md(RL_OPTIMIZE_MD),
        code(RL_OPTIMIZE_CODE),
        md(RL_LEADERBOARD_MD),
        code(RL_LEADERBOARD_CODE),
        md(NOVELTY_MD),
        code(NOVELTY_CODE),
        md(HEADTOHEAD_MD),
        code(HEADTOHEAD_CODE),
        md(PUBLISH_MD),
        code(PUBLISH_CODE),
        md(RECAP_MD),
    ]
    return cells


def main() -> int:
    nb = nbformat.v4.new_notebook(cells=build_cells())

    # Carry the same kernel metadata as the other workflow notebooks.
    template = nbformat.read(
        ROOT / "examples" / "complete_pk_workflow.ipynb", as_version=4
    )
    nb.metadata = template.metadata

    # Execute the whole notebook with committed outputs (worker caps keep
    # the sandbox's tight memory envelope safe).
    os.environ.update(
        {
            "LOKY_MAX_CPU_COUNT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()

    nbformat.write(nb, NOTEBOOK)

    # Sanity report
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    errors = [
        out
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    figures = sum(
        1
        for c in code_cells
        for out in c.get("outputs", [])
        if out.get("data", {}).get("image/png")
    )
    executed = sum(1 for c in code_cells if c.get("execution_count"))
    print(f"notebook written: {NOTEBOOK}")
    print(f"code cells: {len(code_cells)}, executed: {executed}, "
          f"errors: {len(errors)}, inline figures: {figures}")
    for err in errors:
        print("  ERROR:", err.get("ename"), str(err.get("evalue"))[:120])
    return 1 if (errors or executed != len(code_cells)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
