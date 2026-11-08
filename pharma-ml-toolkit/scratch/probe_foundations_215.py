"""Probe: print the real numbers the Foundations Workbench tests assert.

Run from the toolkit root: python scratch/probe_foundations_215.py

Every number the new page's tests pin is engine-printed here FIRST (the
round-205+ house rule): the group comparison on the card's own synthetic
demo (rng 42, N(1.2, 0.4, 60) vs N(1.0, 0.45, 50)), the distribution
sampler on the card's defaults, the Welch ANOVA + Tukey HSD +
Games-Howell on the inferential card's default pasted groups, and the
table profiler on the pasted-CSV demo table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pharmaml.ml.inferential import games_howell, tukey_hsd, welch_anova
from pharmaml.ml.stats import compare_groups, sample_distribution
from pharmaml.profiling import profile_dataset

print("== compare_groups (synthetic demo, rng 42) ==")
rng = np.random.default_rng(42)
a = rng.normal(1.2, 0.4, 60)
b = rng.normal(1.0, 0.45, 50)
report = compare_groups(a, b, alpha=0.05, level=0.95, seed=42)
print("branch:", report.branch)
print("test:", report.result.line())
print("headline:", report.headline())
print("A:", report.group_a.line())
print("B:", report.group_b.line())
print("A five-number:", report.group_a.five_number())
print("normality A:", report.normality_a.line())
print("normality B:", report.normality_b.line())
print("CI A:", report.interval_a.line())
print("CI B:", report.interval_b.line())
print("effect:", report.result.effect_line() if hasattr(report.result, "effect_line") else "n/a")
print("a mean/std:", round(float(a.mean()), 6), round(float(a.std()), 6))
print("b mean/std:", round(float(b.mean()), 6), round(float(b.std()), 6))

print()
print("== sample_distribution (normal n=500 loc=0 scale=1 seed=42) ==")
sample = sample_distribution("normal", n=500, loc=0.0, scale=1.0, seed=42)
print(sample.line())
print("min/max:", float(sample.values.min()), float(sample.values.max()))
print()
print("== sample_distribution (binomial n=500 p=0.3 trials=10 seed=42) ==")
binom = sample_distribution("binomial", n=500, p=0.3, trials=10, seed=42)
print(binom.line())
print("value range:", int(binom.values.min()), int(binom.values.max()))

print()
print("== inferential (default pasted groups) ==")
groups = {
    "control": [2.1, 2.4, 2.2, 2.5, 2.3],
    "low": [2.8, 3.1, 2.9, 3.0],
    "high": [3.6, 3.9, 3.4, 3.8, 3.7],
}
matrix = [groups[name] for name in ("control", "low", "high")]
anova = welch_anova(matrix)
row = anova.iloc[0]
print("welch F:", round(float(row["F"]), 6), "p-unc:", float(row["p-unc"]))
tukey = tukey_hsd(matrix)
print(tukey.to_string(index=False))
howell = games_howell(matrix)
print(howell.to_string(index=False))

print()
print("== profile_dataset (pasted demo table) ==")
frame = pd.read_csv(
    __import__("io").StringIO(
        "dose,route,response,note\n"
        "10,IV,1.0,a\n"
        "12,ORAL,1.1,b\n"
        "9,IV,0.9,a\n"
        "11,ORAL,,b\n"
        "10,IV,1.05,c\n"
        "12,ORAL,1.2,a\n"
    )
)
profile = profile_dataset(frame)
print("headline:", profile.headline())
print("as_frame:")
print(profile.as_frame().to_string(index=False))
print("alerts:", [(alert.column, alert.kind) for alert in profile.alerts])
