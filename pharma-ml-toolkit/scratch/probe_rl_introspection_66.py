"""100x-66 probe: RL prior-drift introspection invariants + figure strings."""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "src")

import math

from pharmaml.generative import DEFAULT_CORPUS, QedObjective, ReinventOptimizer, SmilesRnn

prior = SmilesRnn(hidden_dim=96, seed=42)
prior.train_prior(DEFAULT_CORPUS, epochs=8)

# 1) fields present + iteration-0 exact zero
run = ReinventOptimizer(QedObjective(), prior=prior, iterations=4, batch_size=16, seed=7).optimize()
print("keys:", sorted(run.history[0]))
print("iter0 drift:", run.history[0]["drift"])
print("iter0 agent==prior:", run.history[0]["agent_logprob"] == run.history[0]["prior_logprob"])
for e in run.history:
    assert math.isfinite(e["drift"]) and e["drift"] >= 0
    assert math.isfinite(e["entropy"]) and e["entropy"] >= 0
    assert math.isfinite(e["agent_logprob"]) and math.isfinite(e["prior_logprob"])
print("drift series:", [round(e["drift"], 4) for e in run.history])
print("entropy series:", [round(e["entropy"], 4) for e in run.history])
print("agent_lp series:", [round(e["agent_logprob"], 2) for e in run.history])
print("prior_lp series:", [round(e["prior_logprob"], 2) for e in run.history])

# 2) anchor ordering probe: anchor 0 vs 0.3, several seeds
for seed in (7, 42, 123):
    d0 = ReinventOptimizer(QedObjective(), prior=prior, iterations=8, batch_size=16,
                           seed=seed, anchor_weight=0.0).optimize()
    d3 = ReinventOptimizer(QedObjective(), prior=prior, iterations=8, batch_size=16,
                           seed=seed, anchor_weight=0.3).optimize()
    print(f"seed {seed}: final drift anchor=0 -> {d0.history[-1]['drift']:.3f}, "
          f"anchor=0.3 -> {d3.history[-1]['drift']:.3f}, "
          f"brakes={d3.history[-1]['drift'] < d0.history[-1]['drift']}")

# 3) figure renders + exact strings
from pharmaml.viz.generative_plots import plot_rl_introspection

fig = plot_rl_introspection(run)
print("axes:", len(fig.axes))
print("title:", repr(fig.axes[0].get_title(loc="left")))
print("lp legend:", [t.get_text() for t in fig.axes[1].get_legend().get_texts()])
print("score legend:", [t.get_text() for t in fig.axes[0].get_legend().get_texts()])
print("drift ylabel:", repr(fig.axes[2].get_ylabel()))
print("entropy ylabel:", repr(fig.axes[3].get_ylabel()))
print("xlabel:", repr(fig.axes[3].get_xlabel()))

# no-metadata variant (no anchor annotation)
from pharmaml.generative import GenerationResult
bare = GenerationResult(smiles=["CCO"], scores=[0.5], history=[
    {"iteration": 0, "best_score": 0.5, "mean_score": 0.3, "drift": 0.0,
     "entropy": 2.0, "agent_logprob": -1.0, "prior_logprob": -1.0},
])
fig2 = plot_rl_introspection(bare)
print("bare ylabel:", repr(fig2.axes[2].get_ylabel()))

# 4) save to PNG to eyeball later
fig.savefig("scratch/probe_rl_introspection.png", dpi=110)
print("PROBE OK")
