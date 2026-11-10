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

For raw scores or explicit post-run filtering, use the separate chemistry API:

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
actually be synthesized. Install RDKit and its Contrib SA_Score database with
`pip install 'pharmaml[chem]'`; some alternative RDKit builds omit that data.
