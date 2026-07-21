#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""


def get_pruning_path(clf, X, y):
    """Find the best pruning settings for a decision tree by testing
    different values for split size, depth, and leaf sizes."""
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    return ccp_alphas, impurities
