#!/usr/bin/env python
"""scratch/patch_ml_notebook_205.py — add the frozen-feature-set roundtrip
section to both complete_ml_workflow twins (round 205).

The provenance guard flagged examples/data/selection_metadata.json as an
orphan: no executed notebook loads it.  The honest fix is a notebook
section that LOADS it and replays it through the round-204 API
(recalculate_selected_features) — the "apply the frozen feature set"
step the QSAR workflow docs describe.

Idempotent: re-running finds the marker and leaves the cells alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MARKER = "recalculate_selected_features"

MD_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 16 · The frozen feature set — replay a saved selection\n",
        "\n",
        "`select_top_features` freezes a feature set as a small JSON "
        "ledger; `recalculate_selected_features` replays it on any new "
        "batch — only the families the selected columns need are "
        "computed, and the result carries exactly the recorded columns "
        "in the recorded rank order.  `selection_metadata.json` is the "
        "bundled top-24 of the descriptors + ecfp families over eight "
        "drug-like molecules, written by the factory's own saver — the "
        "same file the playground's **Recalculate from metadata** card "
        "loads.",
    ],
}

CODE_CELL = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Train once, select once, then recompute the SAME features for every\n",
        "# new screening batch — no re-ranking, no vocabulary drift.\n",
        "import json\n",
        "\n",
        "from pharmaml.chem.features import recalculate_selected_features\n",
        "\n",
        "frozen = json.loads(\n",
        "    (DATA_DIR / \"selection_metadata.json\").read_text(encoding=\"utf-8\")\n",
        ")\n",
        "fresh_batch = [\n",
        "    \"c1ccc(cc1)O\",              # phenol\n",
        "    \"CCN(CC)C(=O)c1ccc(N)cc1\",  # procainamide\n",
        "    \"NCCc1ccc(O)c(O)c1\",        # dopamine\n",
        "    \"C1CCCCC1CO\",               # cyclohexylmethanol\n",
        "]\n",
        "recalc_frame, recalc_report = recalculate_selected_features(\n",
        "    fresh_batch, frozen\n",
        ")\n",
        "print(\n",
        "    f\"replayed {recalc_frame.shape[1]} selected features over \"\n",
        "    f\"{recalc_report.n_valid} fresh molecules; \"\n",
        "    f\"families recomputed: {recalc_report.families}\"\n",
        ")\n",
        "recalc_frame.iloc[:, :6]",
    ],
}

RECAP_OLD = "## 16 · Recap"
RECAP_NEW = "## 17 · Recap"
RECAP_ROW_OLD = "| Compare models fairly | `compare_models(frame, y, task)` |"
RECAP_ROW_NEW = (
    "| Compare models fairly | `compare_models(frame, y, task)` |\n"
    "| Replay a frozen feature set | `recalculate_selected_features(new, metadata)` |"
)


def patch(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"]]
    if any(MARKER in s for s in sources):
        print(f"{path.name}: marker present, skipped")
        return
    recap_at = next(i for i, s in enumerate(sources) if s.startswith(RECAP_OLD))
    nb["cells"].insert(recap_at, json.loads(json.dumps(CODE_CELL)))
    nb["cells"].insert(recap_at, json.loads(json.dumps(MD_CELL)))
    recap = nb["cells"][recap_at + 2]
    recap["source"] = [
        (RECAP_NEW + line[len(RECAP_OLD):]) if line.startswith(RECAP_OLD) else line
        for line in recap["source"]
    ]
    recap["source"] = [
        (RECAP_ROW_NEW if line.rstrip("\n") == RECAP_ROW_OLD else line)
        for line in recap["source"]
    ]
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{path.name}: patched at cell {recap_at} (recap renumbered to 17)")


for name in (
    "complete_ml_workflow.ipynb",
    "complete_ml_workflow_no_output.ipynb",
):
    patch(EXAMPLES / name)

sys.exit(0)
