"""Probe 7: NoveltyObjective contract edges + CLI edge behaviours."""
from __future__ import annotations

import io
import contextlib

from pharmaml.exceptions import DataValidationError
from pharmaml.generative import DEFAULT_CORPUS, NoveltyObjective
from pharmaml.generative.scoring import NoveltyObjective as NoveltyDirect

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"

# --- corpus edge cases -------------------------------------------------------
# 1. corpus of 1
o1 = NoveltyObjective(corpus=[ASPIRIN])
print("corpus of 1:", o1.corpus, "paracetamol:", round(o1.score(PARACETAMOL), 4))

# 2. duplicates (same + different spellings)
o2 = NoveltyObjective(corpus=[ASPIRIN, ASPIRIN, "OCC(=O)Oc1ccccc1C(=O)O", f" {ASPIRIN}"])
print("duplicates deduped:", len(o2.corpus) == 1, o2.corpus)
print("dup-corpus paracetamol == single-corpus:", o2.score(PARACETAMOL) == o1.score(PARACETAMOL))

# 3. radius=0 / n_bits=32
o3 = NoveltyObjective(corpus=[ASPIRIN], radius=0, n_bits=32)
print("radius=0 n_bits=32 paracetamol:", round(o3.score(PARACETAMOL), 4))
o4 = NoveltyObjective(corpus=[ASPIRIN], radius=6, n_bits=4096)
print("radius=6 n_bits=4096 paracetamol:", round(o4.score(PARACETAMOL), 4))

# 4. non-str mixed in: silent or warned?
import logging

logger = io.StringIO()
handler = logging.StreamHandler(logger)
pharma_log = logging.getLogger("pharmaml.generative.scoring")
pharma_log.addHandler(handler)
pharma_log.setLevel(logging.DEBUG)
o5 = NoveltyObjective(corpus=[42, ASPIRIN, None, b"CCO"])
captured = logger.getvalue()
print("non-str dropped silently?", "42" not in captured and not captured.strip() or captured)
print("mixed corpus:", o5.corpus, "| warning emitted:", bool(captured.strip()))

# 5. generator corpus
o6 = NoveltyObjective(corpus=(s for s in [ASPIRIN, PARACETAMOL]))
print("generator corpus:", o6.corpus)

# 6. bool radius/n_bits exclusion (21-b precedent)
for bad in ({"radius": True}, {"n_bits": True}):
    try:
        NoveltyObjective(corpus=[ASPIRIN], **bad)  # type: ignore[arg-type]
        print(f"BOOL ACCEPTED (defect?): {bad}")
    except DataValidationError:
        print(f"bool rejected ok: {bad}")

# 7. degenerate chemistry: H2, salts, single atoms
for smi in ("[H][H]", "C", "[Na+].[Cl-]", "O"):
    try:
        val = NoveltyObjective(corpus=[ASPIRIN]).score(smi)
        print(f"candidate {smi!r}: novelty {val:.4f}")
    except Exception as exc:
        print(f"candidate {smi!r}: RAISED {type(exc).__name__}: {exc}")

# corpus containing degenerate entries
try:
    o7 = NoveltyObjective(corpus=["[H][H]", "C"])
    print("degenerate corpus ok:", o7.corpus, "aspirin novelty:", round(o7.score(ASPIRIN), 4))
except Exception as exc:
    print("degenerate corpus RAISED:", type(exc).__name__, exc)

# 8. NaN-ish: does score stay finite?
print("score finite for valid:", 0.0 <= NoveltyObjective(corpus=[ASPIRIN]).score("CCO") <= 1.0)

# 9. consistency: corpus member scores 0 exactly; invalid 0.0 + counted
o8 = NoveltyObjective(corpus=[ASPIRIN])
o8.score("not_a_smiles")
o8.score("")
print("invalid accounting:", o8.n_invalid, o8.n_scored)

# 10. docstring example math
od = NoveltyObjective(corpus=[ASPIRIN])
print("docstring example:", round(od.score(PARACETAMOL), 3), "== 0.778")

# 11. huge corpus performance sanity
import time

t0 = time.perf_counter()
oh = NoveltyObjective()
for _ in range(50):
    oh.score(PARACETAMOL)
print(f"50 scores vs 244-member corpus: {time.perf_counter() - t0:.2f}s")

# 12. corpus kwarg = DEFAULT_CORPUS explicitly
print("explicit DEFAULT_CORPUS == default:", NoveltyObjective(corpus=DEFAULT_CORPUS).corpus == NoveltyObjective().corpus)
