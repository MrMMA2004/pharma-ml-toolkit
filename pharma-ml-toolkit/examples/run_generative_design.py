"""Molecule generation from 0 to 100: objectives, GA and RL optimisers.

End-to-end generative-design story on a laptop CPU:

1. load the bundled curated corpus (244 canonical drug-like SMILES)
   plus a user seed table (named molecules, incl. aspirin);
2. define a custom multi-objective: QED drug-likeness + a LogP window +
   Morgan similarity to the aspirin seed (one zero vetoes a molecule);
3. run the SMILES genetic algorithm twice -- de novo (corpus seeds) and
   seeded from the user's molecules;
4. train a character-level GRU prior on the corpus, then fine-tune it
   with REINVENT-style reinforcement learning toward the same objective;
5. print both leaderboards, a short GA-vs-RL comparison, and save the
   ranked table to ``examples/artifacts/generative_design_leaderboard.csv``.

Usage::

    python examples/run_generative_design.py

Requires RDKit (``pip install "pharmaml[chem]"``) and, for the RL half,
torch (``pip install "pharmaml[deep]"``).  Runtime: ~10 s on one CPU core.
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

HERE = Path(__file__).resolve().parent
SEEDS_CSV = HERE / "data" / "seed_molecules.csv"
OUT_CSV = HERE / "artifacts" / "generative_design_leaderboard.csv"

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
    print(frame.to_string(index=False))
    overlap = (
        set(ga_denovo.smiles[:10]) & set(rl.smiles[:10])
    ) or "none -- the two searches explore different corners"
    print(f"top-10 overlap between GA (de novo) and RL: {overlap}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.concat(
        [
            ga_denovo.to_dataframe().assign(method="ga_de_novo"),
            ga_seeded.to_dataframe().assign(method="ga_seeded"),
            rl.to_dataframe().assign(method="rl"),
        ],
        ignore_index=True,
    )
    leaderboard.to_csv(OUT_CSV, index=False)
    print(f"\nranked leaderboard ({len(leaderboard)} rows) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
