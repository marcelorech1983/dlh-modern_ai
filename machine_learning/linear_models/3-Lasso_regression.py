#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import linear_model


def lasso_regression(random_state):
    """Set up and return an untrained Lasso regression model, which uses L1
    regularization to shrink less important features down to zero."""
    model = linear_model.Lasso(alpha=1.0, random_state=random_state)
    return model
