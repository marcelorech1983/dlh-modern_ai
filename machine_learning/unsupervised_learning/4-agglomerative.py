#!/usr/bin/env python3
"""This module provides utilities for preprocessing and
implementing unsupervised learning algorithms."""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(
        X, n_clusters, random_state, n_components, use_pca_data=True):
    """Perform Agglomerative hierarchical clustering on tabular data,
    optionally reducing its dimensionality via PCA first,
    and calculate the resulting silhouette score."""
