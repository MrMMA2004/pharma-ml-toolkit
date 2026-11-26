# Multi-objective study: plot and shortlist

`MultiObjectiveStudy` minimizes **every** objective. For example, minimizing
`(validation_loss, fit_seconds)` makes the resulting trade-off explicit.
The sklearn adapter `tune_model_multi` reports `score` as the **negative**
mean cross-validation score so that it follows this convention.

```python
from pharmaml.ml import MultiObjectiveStudy
from pharmaml.ml.hpo import SearchSpace
from pharmaml.viz import close_figure, plot_study_pareto, save_figure

study = MultiObjectiveStudy(
    lambda params: (params["x"], 1.0 - params["x"]),
    SearchSpace.uniform("x", 0.0, 1.0),
    n_objectives=2,
    sampler="random",
    seed=42,
)
study.run(20)
names = ["loss", "cost"]
report = study.report(names)
shortlist = report.prune(3)
print(shortlist.to_dataframe())
# Each trial also retains the original hyperparameters in trial.params.
for trial in shortlist.trials:
    print(trial.number, trial.params)
fig = plot_study_pareto(study, "loss", "cost", objective_names=names)
save_figure(fig, "study-pareto.svg")  # also PNG or PDF
close_figure(fig)
```

The plot shows completed trials and the front calculated across **all**
objectives. With three or more objectives, a non-dominated trial can look
dominated in the displayed two-dimensional projection. No line is drawn
between points to avoid implying a two-dimensional front. From the inner
`pharma-ml-toolkit/` package root, install plotting dependencies with
`python -m pip install -e '.[reports]'`.

`prune(k)` ranks the original full front by its NSGA-II crowding distance,
keeping at most `k` trials, returned in original study order. The selection
is one-shot: crowding is not recomputed after each removal. Ties are resolved
by original order; if there are more boundary members than slots, not every
boundary can survive. The shortlist retains the original full-front crowding
scores, the full-front size, and total study-trial count. It is a separate
result, **not** a report of the shortlist's hypervolume; the original report
retains the full study's hypervolume and its fixed-reference curve. A diverse
shortlist is a starting point for investigation, not an automatically optimal
model choice.
