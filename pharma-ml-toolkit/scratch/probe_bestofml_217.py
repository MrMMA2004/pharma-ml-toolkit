"""Probe: print the real numbers the Best-of ML Workbench tests assert.

Run from the toolkit root: python scratch/probe_bestofml_217.py

Every number the new page's tests pin is engine-printed here FIRST (the
round-205+ house rule): the catalogue's search/top/coverage verdicts, the
DP statistics on the seeded synthetic column (epsilon 1.0, 8 bins), the
recommender's leave-one-out evaluation (item-based, k 8, top 5), the LSH
recall sweep over the three-blob vectors (probes 2, k 5), the adversarial
audit on the two-blob logistic model, the time-series analysis of the
seeded period-24 signal, and the apriori pipeline over the card's default
transactions.
"""

from __future__ import annotations

import numpy as np

print("== catalogue summary ==")
from pharmaml.ml.catalog import (
    catalog_summary,
    coverage_table,
    search_catalog,
    catalog_top,
)

summary = catalog_summary()
print("n_projects:", summary.n_projects)
print("n_categories:", summary.n_categories)
print("total_stars:", summary.total_stars)

print()
print("== search 'differential privacy' (all categories, limit 200) ==")
hits = search_catalog("differential privacy", limit=200)
print("n hits:", len(hits))
for project in hits[:3]:
    print(f"  {project.name} | {project.category_title} | {project.stars} | {project.license}")

print()
print("== catalog_top(n=10) ==")
top = catalog_top(n=10)
for project in top[:3]:
    print(f"  {project.name} | {project.stars}")

print()
print("== coverage_table() verdict counts ==")
table = coverage_table()
counts: dict[str, int] = {}
for entry in table:
    counts[entry.status] = counts.get(entry.status, 0) + 1
print("counts:", counts, "total entries:", len(table))
for entry in table[:3]:
    print(f"  {entry.title} | {entry.status} | {entry.note[:60]}")

print()
print("== dp_statistics (synthetic N(50, 5) n=400, eps 1.0, bins 8, seed 42) ==")
from pharmaml.ml.privacy import dp_sgd_noise_scale, dp_statistics

rng = np.random.default_rng(42)
values = rng.normal(50.0, 5.0, 400)
report = dp_statistics(values, epsilon=1.0, n_bins=8, seed=42)
print("true_count:", report.true_count, "dp_count:", round(float(report.dp_count), 4))
print("true_sum:", round(float(report.true_sum), 4), "dp_sum:", round(float(report.dp_sum), 4))
print("true_mean:", round(float(report.true_mean), 6), "dp_mean:", round(float(report.dp_mean), 6))
print("n_values:", report.n_values)
print("bounds:", tuple(round(float(b), 4) for b in report.bounds), "from_data:", report.bounds_from_data)
worst = max(report.histogram, key=lambda b: abs(b.dp_count - b.true_count))
print("worst bin:", round(float(worst.lower), 4), round(float(worst.upper), 4), "true", worst.true_count, "dp", round(float(worst.dp_count), 4))
print("histogram bins:", len(report.histogram))
plan = dp_sgd_noise_scale(epsilon=4.0, delta=1e-5, steps=1500)
print("rdp sigma:", round(float(plan.sigma), 4))

print()
print("== recommender (item-based, k 8, top 5, seed 42) ==")
from pharmaml.ml.recommend import evaluate_recommender, recommend_itembased, synthetic_ratings

matrix = synthetic_ratings(seed=42)
print("shape:", matrix.shape, "observed:", int((~np.isnan(matrix)).sum()))
evaluation = evaluate_recommender(matrix, kind="item", k=8, top_n=5, seed=42)
print("hits_cf:", evaluation.hits_cf, "hits_pop:", evaluation.hits_popularity, "n_eval:", evaluation.n_eval_users)
print("lift:", round(float(evaluation.lift), 4))
recs = recommend_itembased(matrix, user=0, k=8, top_n=3)
for rec in recs.recommendations:
    print(f"  item {rec.item} pred {round(float(rec.predicted_rating), 4)} weight {round(float(rec.neighbour_weight), 4)}")

print()
print("== LSH sweep (three-blob 360x8, probes 2, k 5, seed 42) ==")
from pharmaml.ml.annsearch import ann_benchmark

rng = np.random.default_rng(42)
vectors = np.vstack(
    [
        rng.normal(size=(120, 8)) + 2.0,
        rng.normal(size=(120, 8)) - 2.0,
        rng.normal(size=(120, 8)) * 0.5,
    ]
)
total_empty = 0
for planes in (2, 4, 6, 8, 10):
    bench = ann_benchmark(vectors, n_planes=planes, probes=2, k=5, seed=42)
    total_empty += bench.empty_queries
    print(
        f"  planes {planes}: recall@5 {round(float(bench.recall_at_k), 4)} "
        f"touched {round(float(bench.candidate_ratio), 4)} cheaper {round(float(bench.compression), 2)}x"
    )
print("total empty queries:", total_empty)

print()
print("== adversarial audit (two-blob, logistic, radii .05/.1/.25/.5, seed 42) ==")
from sklearn.linear_model import LogisticRegression

from pharmaml.ml.adversarial import adversarial_audit

rng = np.random.default_rng(42)
n_each = 80
x1 = np.concatenate([rng.normal(0.0, 0.8, n_each), rng.normal(2.5, 0.8, n_each)])
x2 = np.concatenate([rng.normal(0.0, 0.8, n_each), rng.normal(-1.5, 0.8, n_each)])
x3 = rng.normal(0.0, 1.0, 2 * n_each)
x = np.column_stack([x1, x2, x3])
y = np.concatenate([np.zeros(n_each), np.ones(n_each)])
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x, y.astype(int))
scales = x.std(axis=0)
scales = np.where(scales > 1e-12, scales, 1.0)
audit = adversarial_audit(
    model.predict,
    x,
    y,
    radii=(0.05, 0.1, 0.25, 0.5),
    task="classification",
    tolerance=1.0,
    feature_scales=scales,
    seed=42,
)
print("n_samples:", audit.n_samples, "n_features:", audit.n_features, "task:", audit.task)
print("clean_accuracy:", round(float(audit.clean_accuracy), 4))
for result in audit.radii_results:
    print(f"  radius {result.radius}: broken {result.n_broken}/{audit.n_samples} rate {round(float(result.broken_rate), 4)}")
print("median_margin:", round(float(audit.median_margin), 4) if np.isfinite(audit.median_margin) else "inf")

print()
print("== time series (sine 240 points, period 24) ==")
from pharmaml.ml.sequences import synthetic_series
from pharmaml.ml.timeseries import (
    autocorrelation,
    detect_changepoints,
    partial_autocorrelation,
    seasonal_decomposition,
    window_features,
)

series = synthetic_series("sine", 240, seed=42)
print("series size:", series.size)
acf = autocorrelation(series, max_lag=30)
print("acf peak_lag:", acf.peak_lag, "significant:", len(acf.significant_lags), "of", acf.values.size - 1)
pacf = partial_autocorrelation(series, max_lag=30)
print("pacf lag-1:", round(float(pacf.values[1]), 4))
decomposition = seasonal_decomposition(series, period=24)
print("seasonal_strength:", round(float(decomposition.seasonal_strength), 4))
print("trend_strength:", round(float(decomposition.trend_strength), 4))
changes = detect_changepoints(series)
print("changepoints:", len(changes.points), "critical:", round(float(changes.critical_value), 4))
if changes.points:
    first = changes.points[0]
    print(f"  first: index {first.index} ({first.direction}) stat {round(float(first.statistic), 4)}")
table = window_features(series, window=24)
print("window table:", len(table), "x", len(table.columns), "cols:", list(table.columns))
print("mean slope:", round(float(table["slope"].mean()), 6))

print()
print("== apriori (default transactions, support .30, confidence .40) ==")
from pharmaml.ml.apriori import apriori_pipeline

transactions = [
    ["aspirin", "metformin"],
    ["aspirin", "statin"],
    ["aspirin", "metformin", "statin"],
    ["metformin"],
    ["aspirin", "metformin"],
]
result = apriori_pipeline(transactions, min_support=0.30, min_confidence=0.40)
print("n_transactions:", result.n_transactions, "n_items:", result.n_items)
print("frequent itemsets:", len(result.frequent), "rules:", len(result.rules))
print(result.rules.to_string(index=False))
