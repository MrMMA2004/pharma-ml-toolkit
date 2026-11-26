# Synthesizability-aware generation

The GA and reinforcement-learning CLI recipes can optionally incorporate the
existing Ertl synthetic-accessibility (SA) objective in their optimization:

```bash
# Run from pharma-ml-toolkit/ after installing the optional chemistry extra.
pharmaml generative run --method ga --recipe qed --sa-weight 1 \
  --iterations 5 --population 16 --pareto --out candidates.csv
```

`--sa-weight` is a **positive relative weight** in the weighted geometric
mean. The original recipe's weights are unchanged: `qed` and `logp` have
weight 1; `druglike` and `novel` give each component weight 1; `similar`
uses similarity 0.7 and QED 0.3. The SA objective is added to these weights,
not used as a replacement. The same option works for `--method rl`, which
additionally requires PyTorch. Without `--sa-weight`, behavior is unchanged.

The output column named `sa_score` is the **optimization reward** in [0, 1],
not the raw Ertl score in [1, 10]. The default reward is 1 at raw SA ≤ 3,
0 at raw SA ≥ 7, and linear in between. Higher reward is better. A zero
reward vetoes the blended score, but this is **not a hard filter**: a
zero-scoring candidate can still appear in a result. `--pareto` considers
the reward alongside all recipe objectives; for a QED-plus-SA run,
`--pareto-plot` plots that pair. With more objectives the plot uses the
first two objective columns, not necessarily SA.

For the raw Ertl score, sorting, filtering and distribution, every run's
`GenerationResult` speaks part3 §24.3/24.4 natively (round 265's wiring —
no manual dance needed):

```python
from pharmaml.viz import plot_sa_distribution

frame = result.sa_frame(sort="easiest")     # leaderboard + raw sa_score (1-10)
picked = result.filter_by_sa(min_sa=2.0, max_sa=4.0)
print("accepted:", picked.n_kept, "rejected:", picked.n_rejected)
print(picked.rejected[["smiles", "reason"]].head())   # both sides preserved
print(result.sa_summary())                # direction + limitations always ride
payload = plot_sa_distribution(result)    # histogram, drug-like band shaded
```

On a run whose `MultiObjective` included the SA reward, `sa_frame()` renames
that breakdown column to `sa_score_reward` before appending the raw
`sa_score` — the reward and the raw score never silently clobber each other.
`filter_by_sa` bounds are inclusive; with neither bound set nothing is
filtered at all (the spec's "do not permanently discard generated
candidates unless the user explicitly enables such filtering"), and when a
filter is active the rejected rows are preserved with per-row reasons —
`kept + rejected == total`, always.

A CSV read back from disk (or any plain SMILES column) can still use the
chemistry API directly:

```python
import pandas as pd
from pharmaml.chem.sa_scoring import sa_scores, filter_by_sa

frame = pd.read_csv("candidates.csv")
batch = sa_scores(frame["smiles"])
filtered = filter_by_sa(batch.results, max_sa=4.0)
print("accepted:", len(filtered.kept), "rejected:", len(filtered.rejected))
# Keep rejected results and their errors for review rather than discarding them.
```

SA is a computational estimate based on molecular complexity and fragment
familiarity; it does not plan or verify a synthesis, starting materials, or
reaction conditions. A favorable score is not evidence that a compound can
actually be synthesized. From the inner `pharma-ml-toolkit/` package root,
install RDKit and its Contrib SA_Score database with
`python -m pip install -e '.[chem]'`; some alternative RDKit builds omit that data.
