"""QSAR machine learning: preprocessing, training, evaluation, persistence."""

from pharmaml.ml.evaluation import evaluate_model
from pharmaml.ml.models import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    build_model,
    list_models,
    resolve_model_name,
)
from pharmaml.ml.neural import (
    DeepMLPClassifier,
    DeepMLPRegressor,
    ResidualMLPClassifier,
    ResidualMLPRegressor,
)
from pharmaml.ml.persistence import load_model, save_model
from pharmaml.ml.preprocessing import (
    build_preprocessor,
    clean_numeric_frame,
    has_categorical_columns,
    make_train_test_split,
    split_features_target,
)
from pharmaml.ml.training import compare_models, train_model

__all__ = [
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "DeepMLPClassifier",
    "DeepMLPRegressor",
    "ResidualMLPClassifier",
    "ResidualMLPRegressor",
    "build_model",
    "build_preprocessor",
    "clean_numeric_frame",
    "compare_models",
    "evaluate_model",
    "has_categorical_columns",
    "list_models",
    "load_model",
    "make_train_test_split",
    "resolve_model_name",
    "save_model",
    "split_features_target",
    "train_model",
]
