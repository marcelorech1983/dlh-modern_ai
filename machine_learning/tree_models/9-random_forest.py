#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """Set up and return a standard Random Forest model with a
    chosen number of trees and a set random seed."""
    clf = ensemble.RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state)
    return clf
