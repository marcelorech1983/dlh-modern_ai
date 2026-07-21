#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """Generate a formatted text report showing precision,
    recall, and F1-score for each class using
    Scikit-learn's classification evaluation tools."""
    class_report = metrics.classification_report(
        y_true=true_labels,
        y_pred=predicted_labels,
        target_names=class_names)
    return class_report
