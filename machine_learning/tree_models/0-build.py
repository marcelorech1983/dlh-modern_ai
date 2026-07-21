#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Initialize a Scikit-learn DecisionTreeClassifier using Gini impurity
    with configurable minimum leaf and split sample sizes."""
    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    return clf
