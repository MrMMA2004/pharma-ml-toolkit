"""Molecule generation from 0 to 100: objectives, GA and RL optimisers.

End-to-end generative-design story on a laptop CPU:

1. load the bundled curated corpus (244 canonical drug-like SMILES)
   plus a user seed table (named molecules, incl. aspirin);
2. define a custom multi-objective: QED drug-likeness + a LogP window +
   Morgan similarity to the aspirin seed (one zero vetoes a molecule);
3. run the SMILES genetic algorithm three times -- de novo (corpus
   seeds), seeded from the user's molecules, and in structural mode
   (scaffold-hop crossover + scaffold-preserving mutation);
4. train a character-level GRU prior on the corpus, then fine-tune it
   with REINVENT-style reinforcement learning toward the same objective;
5. print the leaderboards, a four-arm GA-vs-RL comparison, and save the
   ranked table to ``examples/artifacts/generative_design_leaderboard.csv``;
6. run the GA once more with adaptive operator selection and watch it
   learn -- the per-generation operator mix (its trajectory, printed for
   the first/last generation and saved as a figure) is the GA's own
   learning curve, written to
   ``examples/artifacts/generative_design_operator_trajectory.png``.

Usage::

    python examples/run_generative_design.py

From the `pharma-ml-toolkit/` package root, install RDKit with
``python -m pip install -e '.[chem]'`` and, for the RL half, torch with
``python -m pip install -e '.[deep]'``; the trajectory figure needs matplotlib
(``python -m pip install -e '.[reports]'``). Runtime: ~17 s on one
CPU core (the structural GA arm is the slowest of the four).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pharmaml.generative import (
    DEFAULT_CORPUS,
    GeneticOptimizer,
    LogPBandObjective,
    MultiObjective,
    QedObjective,
    ReinventOptimizer,
    SimilarityObjective,
    SmilesRnn,
)
from pharmaml.viz import close_figure, plot_operator_trajectory, save_figure

HERE = Path(__file__).resolve().parent
SEEDS_CSV = HERE / "data" / "seed_molecules.csv"
OUT_CSV = HERE / "artifacts" / "generative_design_leaderboard.csv"
OUT_PNG = HERE / "artifacts" / "generative_design_operator_trajectory.png"

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def main() -> None:
    # ---- 1. Corpus + user seeds --------------------------------------
    seeds = pd.read_csv(SEEDS_CSV)
    seed_smiles = [smiles for smiles in seeds["smiles"].dropna().tolist()]
    print("=== 1. Inputs ===")
    print(f"bundled corpus : {len(DEFAULT_CORPUS)} canonical drug-like SMILES")
    print(f"user seeds     : {len(seed_smiles)} molecule(s) from {SEEDS_CSV.name}")
    print(f"anchor seed    : aspirin = {ASPIRIN}")

    # ---- 2. The objective --------------------------------------------
    objective = MultiObjective(
        [
            QedObjective(),
            LogPBandObjective(1.0, 3.0),
            SimilarityObjective(ASPIRIN),
        ],
        weights=[0.5, 0.25, 0.25],
    )
    print("\n=== 2. Objective (weighted geometric mean) ===")
    print("0.5 x QED  +  0.25 x LogP band [1, 3]  +  0.25 x similarity to aspirin")
    total, breakdown = objective.score_with_breakdown(ASPIRIN)
    print(f"aspirin scores {total:.3f}  ({', '.join(f'{k}={v:.2f}' for k, v in breakdown.items())})")

    # ---- 3. Genetic algorithm (de novo + seeded) ----------------------
    print("\n=== 3. Genetic algorithm (tournament SMILES GA) ===")
    ga_denovo = GeneticOptimizer(
        objective, population_size=48, generations=20, seed=42
    ).optimize()
    print(f"de novo : {ga_denovo.summary()}")
    ga_seeded = GeneticOptimizer(
        objective, population_size=48, generations=20, seed=42
    ).optimize(seed_smiles=seed_smiles)
    print(f"seeded  : {ga_seeded.summary()}")
    ga_hop = GeneticOptimizer(
        objective, population_size=48, generations=20, seed=42,
        mutation="scaffold", crossover="scaffold_hop",
    ).optimize(seed_smiles=seed_smiles)
    print(f"structural (scaffold-hop crossover): {ga_hop.summary()}")
    print("\ntop-5 de novo leaderboard:")
    print(ga_denovo.to_dataframe().head(5).to_string(index=False))

    # ---- 4. Reinforcement learning (REINVENT-style fine-tuning) -------
    print("\n=== 4. Reinforcement learning (char-GRU prior + policy gradient) ===")
    prior = SmilesRnn(hidden_dim=96, seed=42)
    losses = prior.train_prior(DEFAULT_CORPUS, epochs=8)
    print(f"prior trained on corpus: final loss {losses[-1]:.3f}")
    rl = ReinventOptimizer(objective, prior=prior, iterations=15,
                           batch_size=32, seed=42).optimize()
    print(rl.summary())
    print(f"early stop: {rl.metadata['early_stopped']}  "
          f"(patience plateau of the mean score)")
    print("\ntop-5 RL leaderboard:")
    print(rl.to_dataframe().head(5).to_string(index=False))

    # ---- 5. Comparison + save -----------------------------------------
    print("\n=== 5. GA vs RL (same objective, same seed) ===")
    frame = pd.DataFrame(
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
    print(frame.to_string(index=False))
    overlap = (
        set(ga_denovo.smiles[:10]) & set(rl.smiles[:10])
    ) or "none -- the two searches explore different corners"
    print(f"top-10 overlap between GA (de novo) and RL: {overlap}")
    overlap_structural = (
        set(ga_hop.smiles[:10]) & set(ga_denovo.smiles[:10])
    ) or "none -- framework recombination reaches other molecules"
    print(f"top-10 overlap between GA (structural) and GA (de novo): "
          f"{overlap_structural}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.concat(
        [
            ga_denovo.to_dataframe().assign(method="ga_de_novo"),
            ga_seeded.to_dataframe().assign(method="ga_seeded"),
            ga_hop.to_dataframe().assign(method="ga_hop"),
            rl.to_dataframe().assign(method="rl"),
        ],
        ignore_index=True,
    )
    leaderboard.to_csv(OUT_CSV, index=False)
    print(f"\nranked leaderboard ({len(leaderboard)} rows) -> {OUT_CSV}")

    # ---- 6. Adaptive operators (the GA that learns) ------------------
    print("\n=== 6. Adaptive operators (the GA that learns) ===")
    ga_adaptive = GeneticOptimizer(
        objective, population_size=16, generations=12, seed=42,
        mutation="topology", operator_selection="adaptive",
    ).optimize()
    print(ga_adaptive.summary())
    traj = ga_adaptive.metadata["operator_trajectory"]
    print(f"mutation mix, generation {traj[0]['generation']:>2}: "
          f"{traj[0]['mutation']}")
    print(f"mutation mix, generation {traj[-1]['generation']:>2}: "
          f"{traj[-1]['mutation']}")
    fig = plot_operator_trajectory(ga_adaptive)
    save_figure(fig, OUT_PNG)
    close_figure(fig)
    print(f"operator trajectory -> {OUT_PNG}")


if __name__ == "__main__":
    main()
