# Round 189+ state after the rollbacks — recreated after the FIFTH rollback (2026-11-05)

> **Provenance note:** the original version of this file was written after the 4th
> rollback but was itself swallowed by the 5th rollback (the filesystem snapshot
> reverted to commit `2eb0ede` — every commit above it vanished). This recreation
> was written in the 2026-11-05 session from the verified conversation summary,
> and every claim below has been re-verified against the live tree with evidence.

## 1. What the rollbacks destroyed (confirmed gone, do not trust older file lists)

Four engine modules from round 188/189 were **never committed** and were lost:

| Module | Was | Evidence of loss |
|---|---|---|
| `src/pharmaml/ml/hpo.py` | TPE/successive-halving/hyperband engine, 78 tests | `git log --all --diff-filter=A` has no record; file absent |
| `src/pharmaml/ml/pruning.py` | magnitude/structured pruning engine, 67 tests | same |
| `src/pharmaml/ml/quantization.py` | NF4/int8/LLM.int8() engine, 79 tests | same |
| `src/pharmaml/ml/harness.py` | ZeRO-style sharded training harness, 81 tests | same |

Also lost: the venv repairs (twice — the 5th rollback reverted `/home/z/.venv`
to numpy 2.1.3 with 8 packages missing), and the first version of this file.

## 2. Authoritative survivors (verified in tree at `2eb0ede`, 2026-11-05)

| File | Lines | Notes |
|---|---|---|
| `src/pharmaml/ml/explain.py` | 2285 | round-187 absorption, SHAP galleries |
| `src/pharmaml/ml/interpret.py` | 1795 | round-187 absorption |
| `src/pharmaml/ml/geometric.py` | 1424 | geometric deep learning |
| `src/pharmaml/ml/gnn.py` | 780 | message passing |
| `src/pharmaml/ml/gnnxai.py` | 480 | GNN explanations |
| `src/pharmaml/ml/autograd.py` | 1054 | from-scratch autograd |
| `src/pharmaml/ml/xai.py` | 1929 | captum/PAIR-saliency/bertviz lineages |
| `src/pharmaml/ml/tuner.py` | 2542 | keras-tuner lineage |
| `src/pharmaml/chem/featurize.py` | 174 | + 3 test files |
| `scratch/gh187_review.md` | 157 | the round-187 absorption checklist |

playground.py = **30,026 lines**, VERSION = **1.2.106**, tests = **53 files**
(suite was 4020 passing / 12 skipped at v1.2.106 across 147 collected files —
the 53 top-level `test_*.py` files expand via parametrization).

## 3. Rebuild specifications (from the round-188/189 design, preserved verbatim)

- **TPE (ml/hpo.py):** γ = min(⌈0.1·n⌉, 25) startup trials; parzen estimators
  over log-transformed search spaces; expected-improvement acquisition.
- **NF4 (ml/quantization.py):** 16-level NF4 lookup table with double-quant
  (block scales, block size 64); absmax per-block; dequant on demand.
- **LLM.int8() (ml/quantization.py):** outlier threshold |x| ≥ 6.0 split into a
  separate high-precision path; int8 matmul on the inlier stream.
- **ZeRO (ml/harness.py):** stages 0–3 (none / optimizer / gradients+optimizer /
  params+grads+optimizer), with stage-3 gather/scatter around forward.
- **EarlyStopping (ml/harness.py):** 5 branches (improve / patience tick / best
  restore / plateau-min-delta / stop) with best-state checkpointing.
- **pruning (ml/pruning.py):** magnitude + structured (channel) schedules,
  iterative retraining loop.

## 4. Environment contract (requirements.txt is the pinned truth)

numpy==1.26.4 · torch==2.14.1+cpu (index `https://download.pytorch.org/whl/cpu`)
· rdkit==2026.9.1 · meeko==0.8.0 · gemmi==0.7.5 · shap==0.54.1 · selfies==2.1.1
· matplotlib==3.10.9 · ruff==0.17.1 · mypy==2.4.0 · pytest==9.0.2
Install style: `pip install -c /tmp/constraint.txt` (numpy==1.26.4) to stop drift.
After the 5th rollback the venv repair is scripted in `scratch/venv_repair.sh`.

## 5. Recovery status in this session (2026-11-05)

- venv repair: relaunched from `scratch/venv_repair.sh` (log:
  `scratch/venv_repair.log`) — see log tail for completion.
- **The project history is now OFF-SITE:** pushed to
  `github.com/MrMMA2004/pharma-ml-toolkit` (private), verified by SHA match +
  clean test clone + zero token leakage. Future rollbacks can no longer destroy
  the git history — clone back from the remote if it ever happens again.

## 6. Next steps (P5–P7 of the MASTER_PROMPT audit)

1. Finish audit phases P2–P6 (see `PROJECT_AUDIT_PROGRESS.md`).
2. Rebuild the four lost engines per §3 specs + tests.
3. Fold torchlens / alibi / Quantus lineages into interpret/explain (planned
   in round 189, never landed).
4. GUI cards in playground.py for each rebuilt engine.
5. Full gate (doctests / ruff / mypy / suite), ship v1.2.107, push, verify.
