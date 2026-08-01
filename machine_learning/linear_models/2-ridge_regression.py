#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import linear_model


def ridge_regression(random_state):
    """Set up and return an untrained Ridge regression model using L2
    regularization to help keep the model stable and prevent overfitting."""
    model = linear_model.Ridge(alpha=1.0, random_state=random_state)
    return model
