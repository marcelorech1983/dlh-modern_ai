#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import preprocessing


def Standardize(X):
    """Standardize tabular data using Scikit-learn's
    StandardScaler to give each feature a mean of 0
    and a standard deviation of 1."""
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
