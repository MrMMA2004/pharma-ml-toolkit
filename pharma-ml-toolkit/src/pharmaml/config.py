"""Typed runtime configuration shared across pipeline components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Immutable settings for a full train/evaluate/persist run.

    Attributes
    ----------
    random_state:
        Seed threaded through every stochastic component (splits,
        estimators, CV shuffling) to make runs reproducible.
    test_size:
        Held-out fraction used by the train/test split (0 < size < 1).
    cv_folds:
        Number of stratified K-folds used in cross-validation.
    n_jobs:
        Parallel workers for grid search; ``-1`` uses all cores.
    output_dir:
        Directory where artifacts (models, reports, tables) are written.
        Created lazily by :func:`output_path`.
    """

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    output_dir: Path = field(default_factory=lambda: Path("artifacts"))

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(
                f"test_size must be strictly between 0 and 1, got {self.test_size}"
            )
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")

    def output_path(self, filename: str) -> Path:
        """Return ``output_dir / filename``, creating the directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / filename
