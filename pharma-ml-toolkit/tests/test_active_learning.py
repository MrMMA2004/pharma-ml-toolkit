"""Active Learning tests (part8 §14.1).

The full workflow contract: every strategy ranks; the retention set
(SMILES, IDs, scores, uncertainty, rationale) is present; labels are
never fabricated (unexported IDs refused, NaN skipped); the retrain
step refits on the grown labeled set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pharmaml.ml.active_learning import (
    QUERY_STRATEGIES,
    ActiveLearningLoop,
)


@pytest.fixture()
def frames():
    rng = np.random.default_rng(0)
    labeled = pd.DataFrame(
        {
            "x1": rng.normal(size=40),
            "x2": rng.normal(size=40),
            "y": rng.integers(0, 2, 40),
            "smiles": [f"C{'C' * i}O" for i in range(40)],
        }
    )
    pool = pd.DataFrame(
        {
            "x1": rng.normal(size=60),
            "x2": rng.normal(size=60),
            "smiles": [f"C{'C' * i}N" for i in range(60)],
        }
    )
    return labeled, pool


def _loop(frames):
    labeled, pool = frames
    return ActiveLearningLoop(
        labeled, pool,
        feature_columns=["x1", "x2"],
        target_column="y",
        smiles_column="smiles",
        random_state=0,
    )


class TestStrategies:
    @pytest.mark.parametrize("strategy", QUERY_STRATEGIES)
    def test_every_strategy_ranks_and_retains(self, frames, strategy) -> None:
        loop = _loop(frames)
        batch = loop.query(n_query=5, strategy=strategy)
        assert len(batch) == 5
        # §14.1 retention set
        assert {"id", "smiles", "predicted", "uncertainty", "rationale"} <= set(
            batch.columns
        )
        assert (batch["uncertainty"] >= 0).all()
        assert batch["rationale"].str.len().gt(10).all()

    def test_query_is_deterministic_per_seed(self, frames) -> None:
        first = _loop(frames).query(n_query=8, strategy="entropy")
        second = _loop(frames).query(n_query=8, strategy="entropy")
        assert list(first["id"]) == list(second["id"])

    def test_unknown_strategy_refused(self, frames) -> None:
        loop = _loop(frames)
        with pytest.raises(ValueError, match="unknown strategy"):
            loop.query(n_query=3, strategy="vibes")

    def test_n_query_cannot_exceed_the_pool(self, frames) -> None:
        loop = _loop(frames)
        batch = loop.query(n_query=10_000, strategy="diversity")
        assert len(batch) == 60


class TestWorkflowContract:
    def test_the_full_loop_without_fabrication(self, frames) -> None:
        loop = _loop(frames)
        batch = loop.query(n_query=6, strategy="committee")

        # the lab measures SOME of them (half — the rest stay unlabeled)
        measured = batch.head(3)[["id"]].copy()
        measured["experimental_label"] = [0, 1, 1]
        added = loop.import_experimental_labels(measured)
        assert added == 3
        assert len(loop.labeled) == 43
        assert len(loop.pool) == 57

        # retrain and run round 2
        loop.retrain()
        second = loop.query(n_query=4, strategy="margin")
        assert len(second) == 4
        assert loop.history[-1].round_index == 1

    def test_unexported_ids_are_refused(self, frames) -> None:
        loop = _loop(frames)
        loop.query(n_query=3, strategy="uncertainty")
        with pytest.raises(ValueError, match="never exported"):
            loop.import_experimental_labels(
                pd.DataFrame(
                    {"id": ["row-59"], "experimental_label": [1]}
                )
            )

    def test_nan_labels_are_skipped_not_guessed(self, frames) -> None:
        loop = _loop(frames)
        batch = loop.query(n_query=4, strategy="uncertainty")
        labels = batch[["id"]].copy()
        labels["experimental_label"] = [0, np.nan, 1, np.nan]
        assert loop.import_experimental_labels(labels) == 2
        assert len(loop.labeled) == 42

    def test_constructor_validates_its_inputs(self, frames) -> None:
        labeled, pool = frames
        with pytest.raises(ValueError, match="missing features"):
            ActiveLearningLoop(
                labeled.drop(columns=["x2"]), pool,
                feature_columns=["x1", "x2"], target_column="y",
            )
        with pytest.raises(ValueError, match="target column"):
            ActiveLearningLoop(
                labeled, pool, feature_columns=["x1"], target_column="nope",
            )
        broken = labeled.copy()
        broken.loc[0, "y"] = np.nan
        with pytest.raises(ValueError, match="NaN labels"):
            ActiveLearningLoop(
                broken, pool, feature_columns=["x1", "x2"], target_column="y",
            )

    def test_custom_model_accepted(self, frames) -> None:
        from sklearn.linear_model import LogisticRegression

        labeled, pool = frames
        loop = ActiveLearningLoop(
            labeled, pool,
            feature_columns=["x1", "x2"], target_column="y",
            model=LogisticRegression(),
        )
        batch = loop.query(n_query=3, strategy="margin")
        assert len(batch) == 3

    def test_history_records_every_round(self, frames) -> None:
        loop = _loop(frames)
        loop.query(n_query=3, strategy="uncertainty")
        loop.query(n_query=2, strategy="entropy")
        assert [r.strategy for r in loop.history] == ["uncertainty", "entropy"]
        assert [r.n_query for r in loop.history] == [3, 2]
