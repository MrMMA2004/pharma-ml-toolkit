#!/usr/bin/env python3
"""r367 QA-section insert + the cascade renumber (the r366 precedent).

Inserts the r367 executed-this-round section as the new section 1 of
docs/QA/TEST_EXECUTION_REPORT.md and renumbers the existing
"## N." executed/prior sections +1 (sections 17/18 stay named, the
standard-skips/regression sections keep their stable titles).
"""
from __future__ import annotations

import re
from pathlib import Path

QA = Path("/home/z/my-project/pharma-ml-toolkit/docs/QA/TEST_EXECUTION_REPORT.md")

R367_SECTION = """## 1. Executed this round (r367 — the part18 campaign, trace 1a219777c8a0fde9: the DL ARCHITECTURE EXPANSION shipped with the full §9 chain + the ui/-pages quarantine COMPLETION + the v1.2.243 NEW-version ship)

| Lane | Command (class) | Result |
| --- | --- | --- |
| The sync + state | `git fetch` + the ff-only rule | `origin/main` = local = `10b4fc8c` (the r366 continuation close-out terminal) — nothing to absorb; **no `arena/*` refs exist**; the PAT credential file intact (93 chars, 600-mode, re-derived from `upload/MASTER_PROMPT.pdf`'s two-line splice and verified equal, never printed) |
| The venv drift guard | rdkit/torch vs `requirements.txt` | **no drift**: rdkit 2026.9.1 / torch 2.14.1+cpu at the pins (numpy/pandas/scikit-learn/fastapi re-checked); no repair fired |
| **The DL architecture expansion (part18 §5 + §9)** — the round's headline increment | the five families, backend → API → frontend | **five new families shipped with the full integration chain**: the tabular Transformer (`src/pharmaml/ml/transformer.py` — FT-Transformer-spirit feature-token attention, from scratch, no pretrained-weights claim; rides the `/api/v1/dl/train` catalog + the four hyperparams `d_model/heads/n_blocks/dim_feedforward` in the route's `_HYPERPARAM_SPEC`), the sequence CNN (`src/pharmaml/ml/cnn.py` — 1-D temporal conv stack, 3-D-only input contract refusing 2-D tabular matrices with the honest pointer, the `synthetic_motif_sequences` fixture; the dedicated `/api/v1/dl/cnn-train` + `/api/v1/dl/cnn-predict` lanes), the tabular autoencoder (`src/pharmaml/ml/autoencoder.py` — latent/reconstruction/anomaly lane; `/api/v1/dl/autoencoder-train` + `/api/v1/dl/autoencoder-encode`), the feature-space GAN (`src/pharmaml/generative/gan.py` — `/api/v1/generative/gan/train` + `/gan/sample` with the quality report), and the feature-space DDPM diffusion (`src/pharmaml/generative/diffusion.py` — `/api/v1/generative/diffusion/train` + `/diffusion/sample`) — the API surface **134 → 144 paths** (`contracts/openapi.json` regenerated, `frontend/src/types/api.d.ts` re-grown); the `FeaturePriorRecord` store kind (feature-space, `molecular: false`, the honest scope line); the one-path-per-class rule enforced in `routes/ml.py` (the sequence models refuse tabular predict, the autoencoder refuses supervised predict + SHAP, each with the pointer to its own lane); the frontend consumers: the DLPage's CNN + autoencoder lanes (the transformer rides the catalog select), the GenerativePage's feature-GAN/diffusion lanes, the two service clients; the hyperparameter-vocabulary mirror drift **caught by the `test_dl_workspace.py` gate and fixed at source** (the four transformer keys joined `DL_HYPERPARAM_KEYS`) |
| **The ui/-pages quarantine COMPLETION (part18 §10)** — the second increment | the legacy GUI tree's move + the reference re-aims | `ui/` (78 files) → **`legacy/ui/`**, `tests/ui/` (74 files, 972 tests) → **`legacy/tests_ui/`** (out of the default `testpaths` gate, run explicitly via `pytest legacy/tests_ui`); the Qt-free `pharmaml.studio` persistence modules STAYED in `src/` (they legitimately serve the FastAPI backend); the path-math fixes across the moved tree (the pages' `_TK` walk depth, `font_loader.py`'s bounded walk-up fallback, the launcher's own-dir path), the root `conftest.py` now puts `legacy/` on the path (the EGL stub still preloads for EVERY invocation), the packaging spec's pathex + `tools/freeze_equivalent.py` + `tools/freeze_smoke_driver.py` + `tools/release_readiness.py` adapted, the checkpoint ship gate's ruff/GUI-gate paths re-aimed, the capability registry's 5 module fields + 46 test refs re-aimed at `legacy.*`/`legacy/tests_ui/*` (the existence gate green), the chunk runner + the live docs trued; the **new `TestGuiTreeQuarantined` guards** (4 tests in `tests/test_legacy_retirement.py`) pin the move so a rollback cannot resurrect the Qt pages in-tree; the collection census honestly re-measured **9,238/513 → 8,274/441** (the 972 quarantined GUI tests left the default gate; sampled green from the new home: pages_smoke + phase4 22, doctor + nine_cards + phase5 40) |
| The canonical gates | the local verification slice | the collection **8,274 tests over 441 files, 0 errors, exit 0** (measurement-last, re-taken on the shipped tree); the DL families `tests/test_cnn.py + test_transformer.py + test_autoencoder.py + test_generative_gan_diffusion.py + tests/api/test_dl_architectures.py + test_dl_semi_generative.py` **144 passed**; the DL drift gate `tests/test_dl_workspace.py` **12 passed** (after the vocabulary fix); retirement + quarantine guards **17 passed** (13 + the 4 new); capability registry **27 passed**; checkpoint tooling + release audit + phase6 green; the packaging validator `--static` **27/27**; the API contracts **6 passed** + hygiene + frontend-foundation green; the frontend vitest **357/357 over 41 files** (+2 the CNN/autoencoder + feature-generative lanes, after the unused-const tsc fix) + `tsc --noEmit` exit 0; the stress-suite GUI lane green from `tests/` (the quarantined import resolved via the root conftest's `legacy/` path) |
| The notebooks + carriers | the re-execution + the four-version-carrier lockstep | the eight notebooks re-executed at 1.2.243 (all print the shipped `__version__`); **1.2.242 → 1.2.243** across `VERSION` / `pyproject.toml` / `src/pharmaml/__init__.py` / the root web carrier `toolkit-version.ts`, the CHANGELOG leads with 1.2.243; the SBOM root component re-generated at 1.2.243; `packaging/windows/version_info.txt` + the DEPENDENCY_INVENTORY trued; FEATURES.md's freeze-stamp line trued |
| The zip ship | `scratch/build_zip_367.py` (the canonical recipe + the r367 must-carry set) | **pharma-ml-toolkit-v1.2.243.zip: 1,346 entries / 1,190 files / 28,048,953 B, sha-256 `4dedf8cb…db2b1`**, testzip clean; the must-carry gate green (the five DL modules + their five suites + the two frontend lanes + the quarantined `legacy/ui/` + `legacy/tests_ui/` pair all present), the retired-pair gate green, the composition gate 0 violations, the leak scan **0 real leaks**, the `version_info.txt` in-archive quad verified at 1.2.243; shipped to the **six homes** (download / toolkit dist-archive / toolkit download / upload / root dist-archive / offsite mirror) each testzip-verified |
| The lint lane | `bun run lint` | **BLOCKED locally this firing** by the sandbox node_modules janitor (the documented saga: node_modules wiped between tool calls and mid-run four times this round; the restored tree intermittently missing the eslint formatter modules at import time) — `tsc --noEmit` exit 0 covers the type contract; the CI lint lane carries the gate on push (honest register, never a silent skip) |
| The penetration lane | the standing budget | **no dispatch this firing** (the standing 360-minute full-breadth deep scan rides the weekly Monday 04:30 UTC slot per the r364 close-out; the API-side suite 67 rides the release builds' api gates); the infra layers remain verified end-to-end from r364 |
| The census pins (measurement-last) | `src/lib/toolkit-stats.ts` + `scratch/census.py` | the four carriers re-trued on the shipped tree: collection **8,274** (verified pin = live re-measure), vitest **357** over 41 files, the walk **1,873 files** / the counted lines re-walked AFTER this round's own ledger edits + the two in-tree zip homes; the registry pins 33/24 unchanged |
| The counter | `scratch/100runs-counter.json` | runs_completed **158 → 159** (this firing), the note rides the ship commit |

"""
# end R367_SECTION

text = QA.read_text(encoding="utf-8")

# renumber existing "## N." sections (the executed/prior cascade) +1
def bump(m: re.Match) -> str:
    return f"## {int(m.group(1)) + 1}."

# only bump the numbered executed/prior sections (## 1. .. ## 16.), and
# NOT the stable-tail sections (## 4. Standard skips ... rides the same
# numbering cascade per the r366 precedent, which renumbered everything)
anchor = "## 1. Executed this round (r366"
idx = text.find(anchor)
assert idx > 0, "the r366 section anchor vanished"
head, body = text[:idx], text[idx:]
body = re.sub(r"^## (\d+)\.", bump, body, flags=re.M)
text = head + R367_SECTION + body
QA.write_text(text, encoding="utf-8")
print("r367 section inserted; cascade renumber done")
print("new heading list:")
for m in re.finditer(r"^## (\d+)\. (.{60})", text, re.M):
    print(f"  {m.group(1)}. {m.group(2)}")
