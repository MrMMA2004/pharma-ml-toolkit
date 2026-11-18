"""Version-aware estimator classification without requiring every predictor to be sklearn."""
from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier, is_regressor
from sklearn.pipeline import Pipeline

from pharmaml.exceptions import DataValidationError


def estimator_kind(model: Any) -> str | None:
    """Return classifier/regressor or None for an untagged lightweight predictor.

    Sklearn estimators must expose a valid, unambiguous estimator type. The
    old attribute is used only for compatibility with legacy custom predictors
    and to fail closed on a classifier with incorrectly ordered mixins.
    """
    legacy = getattr(model, "_estimator_type", None)
    if isinstance(model, ClassifierMixin) or legacy == "classifier":
        return "classifier"
    if isinstance(model, Pipeline) and model.steps:
        final = model.steps[-1][1]
        if final is not None and final != "passthrough" and estimator_kind(final) == "classifier":
            return "classifier"
    tagged = isinstance(model, BaseEstimator) or callable(getattr(model, "__sklearn_tags__", None))
    if not tagged:
        if legacy not in (None, "regressor"):
            raise DataValidationError("model has an invalid estimator type")
        return legacy

    tag_method = getattr(model, "__sklearn_tags__", None)
    if callable(tag_method):
        try:
            tags = tag_method()
            if not hasattr(tags, "estimator_type") or not hasattr(tags, "requires_fit"):
                raise ValueError("tags must provide estimator_type and requires_fit")
        except (AttributeError, TypeError, ValueError) as exc:
            raise DataValidationError("model has invalid sklearn estimator tags") from exc
    try:
        classifier = is_classifier(model)
        regressor = is_regressor(model)
    except (AttributeError, TypeError, ValueError) as exc:
        raise DataValidationError("model has invalid sklearn estimator tags") from exc
    if classifier and regressor:
        raise DataValidationError("model has an ambiguous sklearn estimator type")
    if classifier:
        return "classifier"
    if regressor:
        return "regressor"
    raise DataValidationError("model must declare a valid sklearn classifier or regressor estimator type")
