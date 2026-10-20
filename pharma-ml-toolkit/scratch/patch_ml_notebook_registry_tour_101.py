"""101 driver: the §11 registry-tour cells (notebook side of 100x-101).

Round 100x-101 audited "do the notebooks and playground.py cover the
entire content".  The playground was already complete — its model
combobox lists ``list_models(task, only_available=True)`` dynamically,
so all 24 families (and every future one) are selectable.  The
notebooks were not: only 10 of 24 families appeared in any executed
notebook (11 with the tuning notebook's hsvr).  Thirteen families —
adaboost, catboost, elastic_net_logistic, extra_trees,
gradient_boosting, lasso, lasso_logistic, lightgbm, pls, ridge,
ridge_classifier, svm, xgboost — had no notebook demonstration.

This driver lands the fix as a self-updating section instead of a
hand-written list:

1. INSERT two cells after the §10 leaderboard plot (anchor:
   ``fig_leaderboard``-bearing cell, before ``## 11 · Interpret``):
   markdown intro + code cell "the registry tour" that walks
   ``list_models(task, only_available=False)`` for BOTH tasks, fits
   every available family via ``train_model(..., tune=False)`` on the
   splits the notebook already built (§5 classification, §9
   regression), scores each on the same held-out rows via
   ``evaluate_model``, and renders one DataFrame.  Unavailable
   families stay visible with their missing requirement — nothing
   vanishes silently.
2. RENUMBER the downstream sections: 11 Interpret→12, 12 Deploy→13,
   13 Publish→14, 14 Recap→15 (markdown headers + the §0 ToC rows).
3. RECAP table gains the registry-tour row.
4. EXECUTION is deferred to scratch/reexecute_all_notebooks_64.py —
   the version bump to 1.2.44 re-executes every notebook anyway (they
   all print ``pharmaml.__version__``), and that single foreground
   pass picks the new cells up with it.

The tour doubles as a full-registry smoke test inside the executed
notebook: any family that fails to fit on plain numeric data fails the
notebook execution — 37 registry entries (20 regression + 17
classification across the 24 families), one table.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "examples" / "complete_ml_workflow.ipynb"

MD_INTRO = """## 11 · The full registry — every family, one tour

The leaderboard above compared four hand-picked families. The registry
holds **more — 24 model families**, each registered for one or both
tasks, with the optional engines (`xgboost`, `lightgbm`, `catboost` and
the `torch` deep nets) appearing automatically once their extra is
installed. The tour below fits **every** available family on the two
splits this notebook already built — the §5 classification split and
the §9 regression split — and scores each on the same held-out test
rows, so the numbers stay directly comparable:

- `available` — `yes`, or the missing requirement (an unavailable family
  is listed honestly, it does not vanish silently);
- `held-out` — accuracy (classification) / R² (regression) on the
  untouched test split;
- `fit_s` — the `train_model` wall-clock fit time.

`list_models()` is the self-updating guarantee: a family added to the
registry tomorrow joins this tour without any edit here. The same call
powers the desktop playground's model picker and the CLI's
`pharmaml ml models`, so the three surfaces cannot drift apart.
"""

CODE_TOUR = '''from pharmaml.ml.models import list_models
from pharmaml.ml import evaluate_model

# One tour, every family: list_models() walks the registry itself, so
# this cell never needs editing when a family joins (extra_trees at
# 1.2.41, gradient_boosting at 1.2.42, pls at 1.2.43 ...).
tour_rows = []
for task, X_tr, X_te, y_tr, y_te in (
    ("classification", X_train, X_test, y_train, y_test),
    ("regression", rX_tr, rX_te, ry_tr, ry_te),
):
    for entry in list_models(task, only_available=False):
        name = entry["name"]
        if not entry["available"]:
            tour_rows.append(
                {"family": name, "task": task,
                 "available": f"no ({entry['requires']} not installed)"}
            )
            continue
        tour_result = train_model(name, task, X_tr, y_tr, tune=False)
        tour_report = evaluate_model(tour_result.model, X_te, y_te, task)
        score = tour_report.accuracy if task == "classification" else tour_report.r2
        tour_rows.append(
            {"family": name, "task": task, "available": "yes",
             "held-out": f"{score:.3f}", "fit_s": round(tour_result.fit_seconds, 2)}
        )

registry_tour = pd.DataFrame(tour_rows)
n_ok = int((registry_tour["available"] == "yes").sum())
print(f"{len(registry_tour)} registry entries — {n_ok} available in this "
      f"environment, each fitted on the same splits and scored on the "
      f"same held-out rows")
registry_tour
'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- locate the §10 plot cell (the last code cell of §10) ----------
    idx_plot = None
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and "plot_model_comparison(" in "".join(cell["source"]):
            idx_plot = i
    assert idx_plot is not None, "anchor cell (plot_model_comparison) not found"

    # idempotence: skip if the tour is already present
    if any("registry_tour" in "".join(c["source"]) for c in cells):
        print("registry tour already present — nothing to do")
        return

    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": MD_INTRO,
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": CODE_TOUR,
    }
    cells.insert(idx_plot + 1, md_cell)
    cells.insert(idx_plot + 2, code_cell)
    print(f"inserted 2 cells after index {idx_plot}")

    # --- renumber the downstream section headers -----------------------
    renames = {
        "## 11 · Interpret": "## 12 · Interpret",
        "## 12 · Deploy": "## 13 · Deploy",
        "## 13 · Publish": "## 14 · Publish",
        "## 14 · Recap": "## 15 · Recap",
    }
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        for old, new in renames.items():
            if old in src:
                cell["source"] = src.replace(old, new)
                src = cell["source"]
                print(f"renamed header: {old} -> {new}")

    # --- §0 ToC rows ----------------------------------------------------
    toc = cells[0]
    src = "".join(toc["source"])
    src = src.replace(
        "| 10 | Leaderboard — every model, cross-validated | `pharmaml.ml` |\n"
        "| 11 | Interpret — feature importance | `pharmaml.viz` |\n"
        "| 12 | Deploy — save/load, score new compounds | `pharmaml.ml` |\n"
        "| 13 | Publish — one-click HTML report + png/svg/pdf + csv/md/json | `pharmaml.viz` |",
        "| 10 | Leaderboard — every model, cross-validated | `pharmaml.ml` |\n"
        "| 11 | Registry tour — all 24 families, one table | `pharmaml.ml` |\n"
        "| 12 | Interpret — feature importance | `pharmaml.viz` |\n"
        "| 13 | Deploy — save/load, score new compounds | `pharmaml.ml` |\n"
        "| 14 | Publish — one-click HTML report + png/svg/pdf + csv/md/json | `pharmaml.viz` |",
    )
    toc["source"] = src

    # --- recap table row ------------------------------------------------
    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        src = "".join(cell["source"])
        if "## 15 · Recap" in src and "| Compare | `compare_models(ds.X, ds.y, ds.task)` |" in src:
            cell["source"] = src.replace(
                "| Compare | `compare_models(ds.X, ds.y, ds.task)` |",
                "| Compare | `compare_models(ds.X, ds.y, ds.task)` |\n"
                "| Registry tour | `list_models(task)` — every family, fitted |",
            )
            print("recap row added")

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written: {NB}")


if __name__ == "__main__":
    main()
