#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import preprocessing


def Standardize(X):
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled
