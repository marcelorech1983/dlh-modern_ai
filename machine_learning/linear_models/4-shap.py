#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """Set up a SHAP explainer using your training data and
    calculate explanation values to show how each feature
    impacts your model's predictions."""
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values
