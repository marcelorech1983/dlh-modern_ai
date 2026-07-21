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
    # reduce first if asked, otherwise cluster the raw data
    if use_pca_data:
        data, pca_model = Apply_PCA(X, n_components, random_state)
    else:
        data = X
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                            linkage='ward')
    model.fit(data)
    # silhouette needs 2+ clusters, otherwise leave it None
    if n_clusters > 1:
        score = metrics.silhouette_score(data, model.labels_)
    else:
        score = None
    return model, data, score
