#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""


def train_tree(clf, X, y):
    """Fit a Scikit-learn tree-based classifier model using
    the provided input features and target labels."""
    clf.fit(X, y)
    return None
