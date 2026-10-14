"""QSAR machine learning: preprocessing, training, evaluation, persistence."""

from pharmaml.ml.preprocessing import (
    clean_numeric_frame,
    build_preprocessor,
    has_categorical_columns,
    split_features_target,
    make_train_test_split,
)
from pharmaml.ml.models import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    list_models,
    build_model,
    resolve_model_name,
)
from pharmaml.ml.training import train_model, compare_models
from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.persistence import save_model, load_model

__all__ = [
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "list_models",
    "build_model",
    "resolve_model_name",
    "clean_numeric_frame",
    "build_preprocessor",
    "has_categorical_columns",
    "split_features_target",
    "make_train_test_split",
    "train_model",
    "compare_models",
    "evaluate_model",
    "save_model",
    "load_model",
]
