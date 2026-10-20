"""102 driver: the §11 registry-tour cumulative-budget row (notebook
side of 100x-102's task-7).

Round 100x-101's open item (d): "the tour's fit_s column could gain a
cumulative-budget row (Hyperband economics tie-in)".  The tour fits
every available family once at defaults; this driver makes the cost
of that sweep a first-class read:

1. CODE cell (the tour): gains a ``cum fit_s`` running-total column
   (what the sweep has spent once it reaches each family, registry
   order) and one ``(total)`` summary row; the print gains the
   economics line — a k-fold CV sweep of the same families multiplies
   the budget by k, and successive halving (the tuning notebook's
   Hyperband ladder) is the machinery that cuts such multi-config
   budgets.
2. MARKDOWN cell (the §11 intro): the ``fit_s`` bullet gains the
   ``cum fit_s`` + ``(total)`` sentence.

Execution is deferred to scratch/reexecute_all_notebooks_64.py — the
version bump to 1.2.45 re-executes every notebook anyway, and that
single foreground pass picks the new cells up with it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "examples" / "complete_ml_workflow.ipynb"

CODE_OLD_TAIL = """registry_tour = pd.DataFrame(tour_rows)
n_ok = int((registry_tour["available"] == "yes").sum())
print(f"{len(registry_tour)} registry entries — {n_ok} available in this "
      f"environment, each fitted on the same splits and scored on the "
      f"same held-out rows")
registry_tour
"""

CODE_NEW_TAIL = """registry_tour = pd.DataFrame(tour_rows)
n_ok = int((registry_tour["available"] == "yes").sum())

# The cumulative-budget read: the running fit cost in tour order (what
# the sweep has spent once it reaches each family) plus one total row —
# the number a k-fold CV sweep multiplies by k, and successive halving
# (the tuning notebook's Hyperband ladder) exists to cut.
registry_tour["cum fit_s"] = (
    registry_tour["fit_s"].fillna(0.0).cumsum().round(2)
)
total_s = float(registry_tour["fit_s"].sum())
registry_tour.loc[len(registry_tour)] = {
    "family": "(total)", "task": "—",
    "available": f"{n_ok} fitted", "held-out": "—",
    "fit_s": round(total_s, 2), "cum fit_s": round(total_s, 2),
}
print(f"{len(registry_tour) - 1} registry entries — {n_ok} available in this "
      f"environment, each fitted on the same splits and scored on the "
      f"same held-out rows")
print(f"tour fit budget: {total_s:.1f} s total — a 5-fold CV sweep of the "
      f"same families would spend ~{5 * total_s:.0f} s; the tuning "
      f"notebook's Hyperband ladder is the machinery that cuts such "
      f"multi-config budgets by halving")
registry_tour
"""

MD_OLD_BULLET = """- `fit_s` — the `train_model` wall-clock fit time.
"""

MD_NEW_BULLET = """- `fit_s` — the `train_model` wall-clock fit time;
- `cum fit_s` — the running fit budget in tour order, closing with one
  `(total)` row — the number a k-fold CV sweep multiplies by k, and
  Hyperband's successive halving (the tuning notebook's ladder) exists
  to cut.
"""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    patched_code = patched_md = False
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and CODE_OLD_TAIL in src:
            cell["source"] = src.replace(CODE_OLD_TAIL, CODE_NEW_TAIL).splitlines(keepends=True)
            patched_code = True
        elif cell["cell_type"] == "markdown" and MD_OLD_BULLET in src:
            cell["source"] = src.replace(MD_OLD_BULLET, MD_NEW_BULLET).splitlines(keepends=True)
            patched_md = True
    if not patched_code:
        raise SystemExit("tour code cell tail not found — anchor drifted")
    if not patched_md:
        raise SystemExit("§11 markdown bullet not found — anchor drifted")
    NB.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"patched {NB}: cum fit_s column + (total) row + economics print")


if __name__ == "__main__":
    main()
