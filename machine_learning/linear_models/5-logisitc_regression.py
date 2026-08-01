#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Set up and return an untrained logistic regression model
    ready to handle binary classification tasks."""
    model = linear_model.LogisticRegression(random_state = random_state)
    return model
