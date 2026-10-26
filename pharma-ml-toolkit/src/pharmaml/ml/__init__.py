"""QSAR machine learning: preprocessing, training, evaluation, persistence,
hyperparameter tuning, applicability-domain analysis."""

from pharmaml.ml.applicability import (
    ApplicabilityDomainResult,
    BootstrapUncertainty,
    applicability_domain,
    bootstrap_uncertainty,
    knn_applicability_domain,
)
from pharmaml.ml.evaluation import (
    PermutationImportance,
    evaluate_model,
    permutation_feature_importance,
    predict_with_uncertainty,
)
from pharmaml.ml.hierarchical import HierarchicalSVR
from pharmaml.ml.models import (
    MODEL_ALIASES,
    MODEL_REGISTRY,
    build_model,
    build_stacking,
    build_voting,
    list_models,
    resolve_model_name,
)
from pharmaml.ml.neural import (
    DeepMLPClassifier,
    DeepMLPRegressor,
    ResidualMLPClassifier,
    ResidualMLPRegressor,
)
from pharmaml.ml.persistence import (
    ModelInputReport,
    load_model,
    save_model,
    validate_bundle_input,
)
from pharmaml.ml.preprocessing import (
    build_preprocessor,
    clean_numeric_frame,
    has_categorical_columns,
    make_train_test_split,
    split_features_target,
)
from pharmaml.ml.training import compare_models, train_model
from pharmaml.ml.tuner import (
    Hyperband,
    HyperParameters,
    RandomSearch,
    TrialResult,
    TunerResults,
    build_ann_classifier,
    build_ann_regressor,
    build_deep_classifier,
    build_deep_regressor,
    build_hsvr,
    make_cv_splitter,
    resource_bounds,
)

__all__ = [
    "MODEL_ALIASES",
    "MODEL_REGISTRY",
    "ApplicabilityDomainResult",
    "BootstrapUncertainty",
    "DeepMLPClassifier",
    "DeepMLPRegressor",
    "HierarchicalSVR",
    "HyperParameters",
    "Hyperband",
    "ModelInputReport",
    "PermutationImportance",
    "RandomSearch",
    "ResidualMLPClassifier",
    "ResidualMLPRegressor",
    "TrialResult",
    "TunerResults",
    "applicability_domain",
    "bootstrap_uncertainty",
    "build_ann_classifier",
    "build_ann_regressor",
    "build_deep_classifier",
    "build_deep_regressor",
    "build_hsvr",
    "build_model",
    "build_preprocessor",
    "build_stacking",
    "build_voting",
    "clean_numeric_frame",
    "compare_models",
    "evaluate_model",
    "has_categorical_columns",
    "knn_applicability_domain",
    "list_models",
    "load_model",
    "make_cv_splitter",
    "make_train_test_split",
    "permutation_feature_importance",
    "predict_with_uncertainty",
    "resolve_model_name",
    "resource_bounds",
    "save_model",
    "split_features_target",
    "train_model",
    "validate_bundle_input",
]
