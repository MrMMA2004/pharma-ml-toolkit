"""Probe: print the real numbers the Deep Learning Workbench tests will assert.

Run: from the toolkit venv, ``python scripts/probe_dl_213.py`` (repo
root; the TK constant below is parents-based, so any cwd works).

The page's two bundled datasets are the QSAR regression table (260 rows x
21 cols, target pIC50, 20 descriptor features) and the QSAR classification
table (420 rows x 11 cols, target activity).  The probe exercises every
engine the page wires -- DeepMLPRegressor / DeepMLPClassifier (torch),
NumpyMLPRegressor (the numpy autograd twin), evaluate_model,
make_train_test_split, RunTrace + smooth_series + histogram_event +
numpy_layer_stack (runboard), umap_embed + trustworthiness (the map) and
SampleSet/DataView + find_duplicates/find_outliers/uniqueness/
sort_by_similarity (curation) -- plus the torch-refusal path the weight
histogram surfaces honestly.
"""

from __future__ import annotations

import time
from pathlib import Path

TK = Path(__file__).resolve().parents[1] / "pharma-ml-toolkit"
import sys

sys.path.insert(0, str(TK / "src"))

import numpy as np
import pandas as pd


def numeric_features(frame: pd.DataFrame, target: str) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if column != target and pd.api.types.is_numeric_dtype(frame[column])
    ]


def train_reg(engine: str):
    from pharmaml.ml import evaluate_model
    from pharmaml.ml.splits import make_train_test_split

    frame = pd.read_csv(TK / "examples" / "data" / "qsar_regression.csv")
    target = "pIC50"
    numeric = numeric_features(frame, target)
    print(f"qsar_regression.csv: {frame.shape[0]} rows x {frame.shape[1]} cols, {len(numeric)} features")
    x_train, x_test, y_train, y_test = make_train_test_split(
        frame[numeric], frame[target], strategy="random", random_state=42
    )
    print(f"split: {len(x_train)} train / {len(x_test)} test (random, seed 42)")
    kwargs = dict(
        hidden_layers=(32, 16),
        activation="relu",
        epochs=40,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=8,
        val_split=0.15,
        random_state=42,
    )
    started = time.perf_counter()
    if engine == "torch":
        from pharmaml.ml.neural import DeepMLPRegressor

        network = DeepMLPRegressor(dropout=0.1, **kwargs)
    else:
        from pharmaml.ml.numpy_neural import NumpyMLPRegressor

        # dropout=0.1 exactly as the Studio page passes it (the shared
        # knob set -- the page keeps one honest budget for both engines).
        network = NumpyMLPRegressor(optimizer="adam", dropout=0.1, **kwargs)
    network.fit(x_train, y_train)
    seconds = time.perf_counter() - started
    report = evaluate_model(
        network, x_test, y_test, "regression",
        keep_predictions=True, X_train=x_train, y_train=y_train,
    )
    print(f"\n=== {type(network).__name__} (regression, {seconds:.1f}s) ===")
    print(f"n_iter_={network.n_iter_} epochs_run={len(network.history_)}")
    best = min(network.history_, key=lambda record: record.get("val_loss") or record["train_loss"])
    print(f"best_epoch={best['epoch']} best_val_loss={best.get('val_loss'):.6f} final_train={network.history_[-1]['train_loss']:.6f}")
    print(f"R2={report.r2:.6f} RMSE={report.rmse:.6f} MAE={report.mae:.6f}")
    print(f"pearson_r={report.pearson_r:.6f} train_r2={report.train_r2:.6f} r2_gap={report.r2_gap:.6f}")
    print(f"n_features_in_={network.n_features_in_}")
    return network, report, x_test, frame, target


def train_cls():
    from pharmaml.ml import evaluate_model
    from pharmaml.ml.neural import DeepMLPClassifier
    from pharmaml.ml.splits import make_train_test_split

    frame = pd.read_csv(TK / "examples" / "data" / "qsar_classification.csv")
    target = "activity"
    numeric = numeric_features(frame, target)
    print(f"\nqsar_classification.csv: {frame.shape[0]} rows x {frame.shape[1]} cols, {len(numeric)} features: {numeric}")
    x_train, x_test, y_train, y_test = make_train_test_split(
        frame[numeric], frame[target], strategy="random", random_state=42
    )
    network = DeepMLPClassifier(
        hidden_layers=(32, 16), activation="relu", epochs=40, batch_size=32,
        learning_rate=1e-3, weight_decay=0.0, patience=8, val_split=0.15,
        dropout=0.1, random_state=42,
    )
    network.fit(x_train, y_train)
    report = evaluate_model(network, x_test, y_test, "classification")
    print(f"\n=== DeepMLPClassifier (classification) ===")
    print(f"n_iter_={network.n_iter_} epochs_run={len(network.history_)}")
    print(f"labels={report.labels}")
    print(f"accuracy={report.accuracy:.6f} balanced={report.balanced_accuracy:.6f} f1_macro={report.f1_macro:.6f}")
    print(f"confusion=\n{report.confusion}")
    print("per_class:")
    for label, row in report.per_class.items():
        print(f"  {label}: precision={row['precision']:.4f} recall={row['recall']:.4f} f1={row['f1']:.4f} support={row['support']}")


def probe_runboard(network) -> None:
    from pharmaml.ml.runboard import RunTrace, histogram_event, smooth_series
    from pharmaml.ml.xai import numpy_layer_stack

    print("\n=== runboard: RunTrace + smooth_series ===")
    trace = RunTrace("last-run")
    for record in network.history_:
        epoch = int(record.get("epoch", 0))
        trace.add_scalar("loss/train", float(record.get("train_loss", 0.0)), step=epoch)
        if record.get("val_loss") is not None:
            trace.add_scalar("loss/val", float(record["val_loss"]), step=epoch)
    print(f"tags={trace.tags('scalar')}")
    train_events = trace.scalars("loss/train")
    val_events = trace.scalars("loss/val")
    print(f"epochs_recorded={len(train_events)} val_events={len(val_events)}")
    raw = [event.value for event in train_events]
    smoothed = smooth_series(raw, beta=0.6)
    print(f"raw_train[:3]={[round(v, 6) for v in raw[:3]]}")
    print(f"ema_train[:3]={[round(v, 6) for v in smoothed[:3]]} ema_last={smoothed[-1]:.6f}")
    print(f"summary={trace.summary()!r}")

    print("\n=== runboard: numpy_layer_stack + histogram_event ===")
    stack = numpy_layer_stack(network)
    print(f"layers={len(stack.layers)}")
    for index, (weight, bias) in enumerate(stack.layers, start=1):
        values = np.asarray(weight, dtype=float).reshape(-1)
        event = histogram_event(values, step=0)
        bias_note = ""
        if bias is not None:
            bias_note = f" bias_std={histogram_event(np.asarray(bias, dtype=float).reshape(-1), step=0).std:.6f}"
        print(
            f"layer{index}: {event.count} weights min={event.min:.6f} max={event.max:.6f} "
            f"mean={event.mean:.6f} std={event.std:.6f}{bias_note}"
        )
        if index == 1:
            print(f"layer1 buckets={len(event.bucket_counts)} edges={len(event.bucket_edges)}")


def probe_runboard_torch_refusal(network) -> None:
    from pharmaml.ml.xai import numpy_layer_stack

    print("\n=== runboard: the torch refusal (honest disclosure) ===")
    try:
        numpy_layer_stack(network)
        print("UNEXPECTED: torch estimator accepted")
    except Exception as exc:
        print(f"refused: {type(exc).__name__}: {str(exc).splitlines()[0]}")


def probe_umap(matrix: np.ndarray) -> None:
    from pharmaml.ml.umap import trustworthiness, umap_embed

    print(f"\n=== umap_embed on {matrix.shape[0]} x {matrix.shape[1]} ===")
    started = time.perf_counter()
    record = umap_embed(matrix, n_neighbors=15, min_dist=0.1, metric="euclidean", n_epochs=200, random_state=42)
    seconds = time.perf_counter() - started
    embedding = record.embedding_array
    score = trustworthiness(matrix, embedding, n_neighbors=min(10, matrix.shape[0] // 3 or 1))
    print(f"seconds={seconds:.1f}s n_edges={record.n_edges} mean_weight={record.mean_weight:.6f}")
    print(f"a={record.a:.6f} b={record.b:.6f} init={record.init}")
    print(f"sigma_stats={tuple(round(v, 6) for v in record.sigma_stats)}")
    print(f"embedding_shape={embedding.shape} trustworthiness={score:.6f}")
    print(f"record_str={str(record)!r}")


def probe_curation(frame: pd.DataFrame) -> None:
    from pharmaml.ml.curation import (
        SampleSet,
        find_duplicates,
        find_outliers,
        sort_by_similarity,
        uniqueness,
    )

    print("\n=== curation: the view algebra ===")
    view = SampleSet.from_frame(frame).view()
    print(f"identity: {view.summary()!r}")
    view = view.filter("pIC50", ">", 7.0)
    print(f"filter pIC50 > 7.0: {view.summary()!r} rows={len(view.to_frame())}")
    view = view.sort_by("pIC50", reverse=True).limit(5)
    top = view.to_frame()
    print(f"sort desc + limit 5: rows={len(top)} top_pIC50={top['pIC50'].iloc[0]:.6f}")
    print(f"stages={view.stages}")

    print("\n=== curation: the audits ===")
    duplicates = find_duplicates(SampleSet.from_frame(frame).view())
    print(f"duplicates: groups={len(duplicates.groups)} summary={duplicates.summary()!r}")
    outliers = find_outliers(SampleSet.from_frame(frame).view(), method="zscore", threshold=4.0)
    print(f"outliers(zscore>4): n_flagged_any={outliers.n_flagged_any} fields={len(outliers.flagged)} summary={outliers.summary()!r}")
    for field, ids in list(outliers.flagged.items())[:3]:
        print(f"  {field}: {len(ids)} flagged {ids[:5]}")
    scores = uniqueness(SampleSet.from_frame(frame).view(), k=5)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    print(f"uniqueness: scored={len(scores)} top3={[(sid, round(score, 6)) for sid, score in ranked[:3]]}")
    similarity = sort_by_similarity(SampleSet.from_frame(frame).view(), 0, k=5)
    print(f"similarity(row 0, k=5): {[(sid, round(score, 6)) for sid, score in similarity]}")


def main() -> int:
    torch_network, torch_report, x_test, frame, target = train_reg("torch")
    numpy_network, numpy_report, _, _, _ = train_reg("numpy")
    train_cls()
    probe_runboard(numpy_network)
    probe_runboard_torch_refusal(torch_network)
    probe_umap(np.asarray(x_test, dtype=float))
    probe_curation(frame)
    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
