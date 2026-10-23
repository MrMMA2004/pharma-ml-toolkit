"""118 driver: the §14 Trust cells (notebook side of the AD upload).

The 2026-10 user upload Applicability_domain_analysis.py became toolkit
surface in 100x-118 (ml/applicability.py + the playground card + the
`ml ad` CLI).  This driver lands the notebook stage — the coverage
contract's third axis: every covered file needs a code-cell import/call
surface in an executed notebook, not just the GUI.

1. INSERT three cells after the §13 Deploy section's CLI note (anchor:
   the markdown cell whose text contains 'pharmaml ml check' and ends
   the section, before '## 14 · Publish'):
     - markdown "## 14 · Trust — applicability domain & prediction reliability"
     - code: the analysis on the §9 regression split + one wild
       extrapolation row → thresholds, verdict counts, reliability table
     - code: the combined two-panel figure (plot_applicability_domain)
2. RENUMBER the downstream sections: 14 Publish→15, 15 Recap→16.
3. The §0 header ToC gains the Trust row (and Publish moves to 15).
4. The Recap table gains the Trust one-liner.
5. EXECUTION is deferred to scratch/reexecute_all_notebooks_64.py — the
   version bump to 1.2.57 re-executes every notebook anyway (they all
   print pharmaml.__version__), and that single foreground pass picks
   the new cells up with it.

Run from the toolkit root:
    python scratch/patch_ml_notebook_ad_118.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "examples" / "complete_ml_workflow.ipynb"

MD_TRUST = """\
## 14 · Trust — applicability domain & prediction reliability

A prediction nobody can judge is a rumor. The uploaded
`Applicability_domain_analysis.py` workflow asks the two questions that
turn it into evidence: **is the query compound inside the training
descriptor space** (mean kNN distance vs a percentile threshold of the
training set's own distances), and **how much does the answer wobble**
when the training set is bootstrap-resampled (the ensemble's per-row
spread vs its own training-side threshold)? Both answers fold into one
`reliable` verdict per compound.

The same analysis ships from the terminal:

```text
pharmaml ml ad --data my_compounds.csv --model ridge --out Applicability_Domain_Results.csv
pharmaml ml ad --bundle ridge.joblib --train-data train.csv --test-data test.csv
```
"""

CODE_TRUST = """\
from pharmaml.ml import applicability_domain

# The query set: the held-out rows PLUS one wild extrapolation -- a
# descriptor profile 30 standard deviations off the training cloud.
far_row = ds_reg.X.max() + 30.0 * ds_reg.X.std()
X_query = pd.concat(
    [rX_te, pd.DataFrame([far_row], columns=ds_reg.X.columns)],
    ignore_index=True,
)

ad = applicability_domain(reg_result.model, rX_tr, ry_tr, X_query, k=5, n_models=15)
print(f"distance threshold {ad.distance_threshold:.3g} -- "
      f"{int(ad.inside_ad.sum())}/{len(ad.inside_ad)} query rows inside the domain")
print(f"{ad.n_models}-member bootstrap ensemble in {ad.fit_seconds:.1f} s "
      f"({ad.uncertainty_kind}); uncertainty threshold "
      f"{ad.uncertainty_threshold:.3g}")
print(f"reliable verdicts: {int(ad.reliable.sum())}/{len(ad.reliable)} "
      "compound(s) -- inside AD AND low uncertainty")

names = [f"held-out #{i}" for i in range(len(rX_te))] + ["extrapolation"]
reliability = ad.to_frame(names=names)
reliability.tail(6)
"""

CODE_TRUST_FIG = """\
from pharmaml.viz import plot_applicability_domain

# The upload's combined figure: (a) the training distance density with
# the threshold, (b) every query compound at its kNN distance against
# the ensemble prediction, green inside / red outside the domain.
fig_ad = plot_applicability_domain(
    ad, y_label="predicted pIC50",
    title="Applicability domain — random forest (regression)",
)
"""

RECAP_ROW = "| Trust the prediction | `applicability_domain(model, Xtr, ytr, Xquery)` |"


def _text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # -- 1. locate the anchor ------------------------------------------ #
    anchor = None
    for index, cell in enumerate(cells):
        text = _text(cell)
        if cell["cell_type"] == "markdown" and "pharmaml ml check" in text:
            anchor = index
    publish = None
    for index, cell in enumerate(cells):
        if _text(cell).startswith("## 14 · Publish"):
            publish = index
    recap = None
    for index, cell in enumerate(cells):
        if _text(cell).startswith("## 15 · Recap"):
            recap = index
    assert anchor is not None and publish is not None and recap is not None, (
        anchor, publish, recap)
    assert anchor + 1 == publish, "unexpected cells between Deploy and Publish"

    # -- 2. insert the Trust cells before Publish ---------------------- #
    def md(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source":
                source.splitlines(keepends=True)}

    def code(source: str) -> dict:
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": source.splitlines(keepends=True)}

    cells[publish:publish] = [md(MD_TRUST), code(CODE_TRUST), code(CODE_TRUST_FIG)]
    publish += 3
    recap += 3

    # -- 3. renumber the downstream sections --------------------------- #
    cells[publish]["source"] = (
        "## 15 · Publish — the one-click visual bundle\n\n"
        "`save_visual_bundle` renders every figure and table into a "
        "directory **and**\na self-contained `report.html` (figures "
        "embedded as base64 — no external\nassets, renders identical "
        "offline). Tables are exported as csv *and*\nmarkdown; figures "
        "as png *and* svg *and* pdf.\n"
    ).splitlines(keepends=True)
    recap_text = _text(cells[recap])
    cells[recap]["source"] = recap_text.replace(
        "## 15 · Recap", "## 16 · Recap"
    ).splitlines(keepends=True)
    # the recap table gains the Trust row right after Deploy
    recap_text = _text(cells[recap])
    deploy_row = next(
        line for line in recap_text.splitlines() if "| Deploy |" in line)
    patched = recap_text.replace(
        deploy_row, deploy_row + "\n" + RECAP_ROW)
    cells[recap]["source"] = patched.splitlines(keepends=True)

    # -- 4. the §0 header ToC ------------------------------------------ #
    header = _text(cells[0])
    old_deploy = "| 13 | Deploy — save/load, score new compounds | `pharmaml.ml` |"
    new_deploy = (
        old_deploy
        + "\n| 14 | **Trust** — applicability domain & prediction reliability |"
        " `pharmaml.ml` |"
        + "\n| 15 | Publish — one-click HTML report + png/svg/pdf + csv/md/json |"
        " `pharmaml.viz` |"
    )
    assert old_deploy in header, "header ToC Deploy row missing"
    header = header.replace(
        "| 14 | Publish — one-click HTML report + png/svg/pdf + csv/md/json |"
        " `pharmaml.viz` |", "")
    header = header.replace(old_deploy, new_deploy)
    cells[0]["source"] = header.splitlines(keepends=True)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                  encoding="utf-8")

    # -- 5. verify ------------------------------------------------------ #
    nb2 = json.loads(NB.read_text(encoding="utf-8"))
    texts = [_text(c) for c in nb2["cells"]]
    assert any(t.startswith("## 14 · Trust") for t in texts)
    assert any(t.startswith("## 15 · Publish") for t in texts)
    assert any(t.startswith("## 16 · Recap") for t in texts)
    assert any("applicability_domain(" in t for t in texts)
    assert any("plot_applicability_domain(" in t for t in texts)
    print("patched: Trust section inserted, Publish→15, Recap→16, "
          "ToC + recap rows updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
