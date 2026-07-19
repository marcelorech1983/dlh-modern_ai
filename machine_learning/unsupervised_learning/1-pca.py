#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """Perform Principal Component Analysis (PCA) on tabular
    data to reduce its dimensionality by preserving a specified
    number of components or a fraction of variance."""
    PCA = decomposition.PCA
    # Fitted PCA instance
    pca = PCA(n_components=n_components, random_state=random_state)
    # Data transformed into principal component space
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
