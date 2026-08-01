#!/usr/bin/env python3
"""A collection of simple helper functions for
building and using linear models."""
from sklearn import linear_model


def Linear_Regression():
    """Set up and return a standard, untrained linear
    regression model ready to fit your data."""
    model = linear_model.LinearRegression()
    return model
