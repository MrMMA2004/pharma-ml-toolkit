"""102 repair: normalise the char-split stream outputs the 1.2.45
re-execution wrote (the site-packages rewrite iterated a bare str —
see reexecute_all_notebooks_64.py's fixed note).

Walks every executed notebook, joins each stream output's text list,
re-chunks it by lines (the nbformat convention: a list of line
strings, each keeping its trailing newline), and applies the
site-packages prefix scrub the char-split pass silently missed (a
single character can never match a full path — the deprecation
warnings' build-machine prefixes rode through un-rewritten).  The
prefix is computed from sysconfig, never hardcoded.  Idempotent: proper
line chunks join and re-split to themselves.  The no-output twins are
regenerated afterwards by strip_notebook_outputs.py — this driver only
touches the executed eight.
"""

from __future__ import annotations

import json
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

_SITE_PACKAGES = sysconfig.get_paths()["purelib"]


def normalise(value: object) -> object:
    """One stream text -> scrubbed nbformat line-chunk convention."""
    if isinstance(value, str):
        joined = value
    elif isinstance(value, list):
        joined = "".join(t for t in value if isinstance(t, str))
    else:
        return value
    joined = joined.replace(_SITE_PACKAGES + "/", "site-packages/")
    return joined.splitlines(keepends=True)

NOTEBOOKS = [
    "complete_ml_workflow.ipynb",
    "complete_docking_workflow.ipynb",
    "complete_pk_workflow.ipynb",
    "complete_cellline_workflow.ipynb",
    "complete_generative_workflow.ipynb",
    "complete_deep_learning_workflow.ipynb",
    "complete_tuning_workflow.ipynb",
    "pharmaml_walkthrough.ipynb",
]


def main() -> int:
    for name in NOTEBOOKS:
        path = EXAMPLES / name
        before = path.stat().st_size
        nb = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for cell in nb["cells"]:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    old = output.get("text", "")
                    new = normalise(old)
                    if new != old:
                        output["text"] = new
                        changed += 1
        path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
        after = path.stat().st_size
        print(f"{name}: {changed} stream outputs normalised, "
              f"{before:,} -> {after:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
