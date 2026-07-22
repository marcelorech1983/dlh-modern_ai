#!/usr/bin/env python3
"""This module provides utilities for creating
and training decision tree models."""
import numpy as np


def feature_importance(rf):
    """Get feature importance scores from a trained random forest and sort
    their index positions from least to most important."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
