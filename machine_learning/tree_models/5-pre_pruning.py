#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """Search over pre-pruning hyperparameters using grid search
    cross-validation to find the best configuration
    for a decision tree classifier."""
    # Define and store all parameter grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 5)),
        'min_samples_leaf': list(range(2, 5)),
        'min_samples_split': list(range(2, 5))
    }
    search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid)
    search.fit(X, y)
    return search.best_params_
