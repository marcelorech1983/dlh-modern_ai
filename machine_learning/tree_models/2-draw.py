#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """Print a readable text outline showing the decision rules,
    feature splits, and leaf nodes of a trained decision tree."""
    print(tree.export_text(clf,
                           feature_names=feature_names,
                           class_names=class_names))
    return None
