#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """Group tabular data into a specified number of clusters by
    initializing and fitting a K-Means clustering model using Scikit-learn."""
    KMeans = cluster.KMeans
    # Set up the KMeans model structure
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    # Train the model on the input dataset
    kmeans = model.fit(X)
    return kmeans
