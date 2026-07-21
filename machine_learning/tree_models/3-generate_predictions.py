#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import tree


def generate_predictions(clf, X):
    """Use a trained tree-based classifier to predict class
    labels for a given set of input features."""
    pred = clf.predict(X)
    return pred
